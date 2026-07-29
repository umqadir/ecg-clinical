#!/usr/bin/env python3
"""Run the single sealed inference pass for one protected evaluation cohort.

Every input this pass consumes is bound to a digest inside the sealed evaluation
choices: the normalization constants, the protocol, the label manifest, the
waveform audit, and the cohort's four store files. They are re-hashed here,
before any model is loaded, because this is the one step that legitimately opens
the protected signal stores. Resumption is authenticated rather than assumed: a
cached per-seed prediction is reused only when the retained receipt vouches for
its digest and its contents re-validate against the store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

from ecg_clinical.data import InferenceRecordDataset, WaveformStore, load_normalization
from ecg_clinical.engine import predict_records
from ecg_clinical.integrity import sha256_file, verify_evaluation_seal, verify_preregistration_seal
from ecg_clinical.locking import exclusive_device_lock
from ecg_clinical.metrics import apply_temperature
from ecg_clinical.models import build_model

COHORT_STORE_NAMES = {
    "ptb_test": "ptb_xl",
    "chapman_shaoxing": "chapman_shaoxing",
    "ningbo": "ningbo",
}
TARGET_COUNT = 16
STORE_FILE_DIGESTS = (
    ("signals.npy", "signals_sha256"),
    ("targets.npy", "targets_sha256"),
    ("metadata.parquet", "metadata_sha256"),
    ("status.npy", "status_sha256"),
)
SCALAR_BOUND_INPUTS = ("normalization", "protocol", "label_manifest", "waveform_audit")


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def require_bound_path(root: Path, *, option: str, given: Path, declared: str) -> None:
    """Refuse a runtime path that differs from the one the seal binds."""

    expected = (root / declared).resolve()
    actual = resolve_under_root(root, given).resolve()
    if actual != expected:
        raise RuntimeError(
            f"{option} points at {actual}, but the sealed evaluation choices bind this input "
            f"to {expected}"
        )


def verify_bound_inputs(
    root: Path,
    choices: dict,
    cohort: str,
    *,
    store_root: Path,
    normalization: Path,
) -> dict:
    """Re-hash every sealed runtime input and return the cohort's bound entry.

    Called before any model is loaded, so a rebuilt store, a substituted
    normalization file, or an edited protocol aborts the pass rather than
    silently changing the one-time result.
    """

    bound = choices.get("bound_inputs")
    if not isinstance(bound, dict):
        raise RuntimeError(
            "the sealed evaluation choices carry no bound_inputs section; protected "
            "inference cannot verify its runtime inputs and is refused"
        )
    for key in SCALAR_BOUND_INPUTS:
        entry = bound.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError(f"bound input {key} is missing from the sealed evaluation choices")
        path = root / str(entry["path"])
        if not path.is_file():
            raise RuntimeError(f"bound input {key} is missing from the working tree: {path}")
        observed = sha256_file(path)
        if observed != entry.get("sha256"):
            raise RuntimeError(
                f"bound input {key} does not match the evaluation seal: {path} hashes to "
                f"{observed}, the sealed choices record {entry.get('sha256')}"
            )
    require_bound_path(
        root,
        option="--normalization",
        given=normalization,
        declared=str(bound["normalization"]["path"]),
    )

    cohorts = bound.get("cohorts")
    if not isinstance(cohorts, dict) or cohort not in cohorts:
        raise RuntimeError(f"the sealed evaluation choices bind no store for cohort {cohort}")
    cohort_entry = cohorts[cohort]
    declared_store = str(cohort_entry["store_root"])
    require_bound_path(
        root,
        option="--store-root",
        given=Path(store_root) / COHORT_STORE_NAMES[cohort],
        declared=declared_store,
    )
    store_path = root / declared_store
    for filename, digest_key in STORE_FILE_DIGESTS:
        path = store_path / filename
        if not path.is_file():
            raise RuntimeError(f"cohort {cohort} store file is missing: {path}")
        observed = sha256_file(path)
        if observed != cohort_entry.get(digest_key):
            raise RuntimeError(
                f"cohort {cohort} store file does not match the evaluation seal: {path} hashes "
                f"to {observed}, the sealed choices record {cohort_entry.get(digest_key)}"
            )
    return cohort_entry


def verify_cohort_indices(indices: np.ndarray, cohort_entry: dict, cohort: str) -> None:
    """Require the derived record set to be exactly the sealed one."""

    expected_records = int(cohort_entry["records"])
    if len(indices) != expected_records:
        raise RuntimeError(
            f"cohort {cohort} resolves to {len(indices)} records, the sealed choices record "
            f"{expected_records}"
        )
    observed = array_sha256(indices)
    if observed != cohort_entry.get("record_indices_sha256"):
        raise RuntimeError(
            f"cohort {cohort} record indices do not match the evaluation seal: derived indices "
            f"hash to {observed}, the sealed choices record "
            f"{cohort_entry.get('record_indices_sha256')}"
        )


def validate_prediction_arrays(
    probabilities: np.ndarray,
    targets: np.ndarray,
    record_indices: np.ndarray,
    *,
    expected_indices: np.ndarray,
    expected_targets: np.ndarray,
    where: Path,
) -> None:
    expected_shape = (len(expected_indices), TARGET_COUNT)
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            f"protected probabilities have shape {probabilities.shape}, expected "
            f"{expected_shape}: {where}"
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"protected probabilities contain non-finite values: {where}")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise RuntimeError(
            f"protected probabilities fall outside [0, 1] "
            f"({probabilities.min()!r} to {probabilities.max()!r}): {where}"
        )
    if not np.array_equal(record_indices, expected_indices):
        raise RuntimeError(
            f"protected predictions are not in the sealed cohort record order: {where}"
        )
    if not np.array_equal(targets, expected_targets):
        raise RuntimeError(
            f"protected prediction targets do not match the waveform store at the sealed "
            f"cohort records: {where}"
        )


def load_cached_prediction(
    path: Path,
    *,
    expected_digest: str | None,
    expected_indices: np.ndarray,
    expected_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accept a cached per-seed prediction only if the receipt vouches for it.

    A file with no receipt entry, a different digest, or contents that disagree
    with the sealed record set aborts the pass. It is never silently recomputed,
    because a suspect file in the protected output directory is itself evidence
    that something went wrong.
    """

    if expected_digest is None:
        raise RuntimeError(
            f"cached protected prediction {path} is not recorded in the retained receipt; "
            "refusing to reuse a file this pass did not produce"
        )
    observed = sha256_file(path)
    if observed != expected_digest:
        raise RuntimeError(
            f"cached protected prediction {path} hashes to {observed}, the retained receipt "
            f"records {expected_digest}"
        )
    saved = np.load(path)
    probabilities = np.asarray(saved["probabilities"], dtype=np.float32)
    targets = np.asarray(saved["targets"], dtype=np.uint8)
    record_indices = np.asarray(saved["record_indices"], dtype=np.int64)
    validate_prediction_arrays(
        probabilities,
        targets,
        record_indices,
        expected_indices=expected_indices,
        expected_targets=expected_targets,
        where=path,
    )
    return probabilities, targets, record_indices


def sealed_checkpoint_digests(choices: dict) -> dict[str, str]:
    digests: dict[str, str] = {}
    for architecture, architecture_choices in choices["architectures"].items():
        for run in architecture_choices["runs"]:
            digests[f"{architecture}-seed-{int(run['seed'])}"] = str(run["checkpoint_sha256"])
    return digests


def prepare_receipt(
    receipt_path: Path,
    *,
    cohort: str,
    records: int,
    device: torch.device,
    choices_commit: object,
    checkpoint_digests: dict[str, str],
) -> dict:
    """Start a receipt, or resume an in-progress one that agrees with this pass.

    An existing in-progress receipt is never overwritten. It carries the digests
    that authorize cached per-seed predictions, so silently starting over would
    discard exactly the evidence resumption depends on.
    """

    fresh: dict[str, object] = {
        "schema_version": 2,
        "cohort": cohort,
        "records": records,
        "device": str(device),
        "evaluation_seal_choices_commit": choices_commit,
        "sealed_checkpoint_sha256": dict(checkpoint_digests),
        "status": "in_progress",
        "completed_seed_passes": [],
    }
    if not receipt_path.exists():
        return fresh
    existing = json.loads(receipt_path.read_text())
    if existing.get("status") == "complete":
        raise RuntimeError(f"protected inference already completed for {cohort}")
    for field, expected in (
        ("cohort", cohort),
        ("evaluation_seal_choices_commit", choices_commit),
        ("records", records),
    ):
        if existing.get(field) != expected:
            raise RuntimeError(
                f"the retained in-progress receipt at {receipt_path} disagrees with this pass on "
                f"{field}: receipt records {existing.get(field)!r}, this pass expects "
                f"{expected!r}. Resolve the disagreement deliberately rather than restarting."
            )
    recorded_digests = existing.get("sealed_checkpoint_sha256")
    if recorded_digests != checkpoint_digests:
        raise RuntimeError(
            f"the retained in-progress receipt at {receipt_path} records different sealed "
            "checkpoint digests than the current evaluation seal; resolve the disagreement "
            "deliberately rather than restarting"
        )
    existing["status"] = "in_progress"
    existing.setdefault("completed_seed_passes", [])
    return existing


def recorded_prediction_digests(receipt: dict) -> dict[tuple[str, int], str]:
    digests: dict[tuple[str, int], str] = {}
    for entry in receipt.get("completed_seed_passes", []):
        digests[(str(entry["architecture"]), int(entry["seed"]))] = str(entry["prediction_sha256"])
    return digests


def record_seed_pass(receipt: dict, architecture: str, seed: int, digest: str) -> None:
    for entry in receipt["completed_seed_passes"]:
        if str(entry["architecture"]) == architecture and int(entry["seed"]) == seed:
            entry["prediction_sha256"] = digest
            return
    receipt["completed_seed_passes"].append(
        {"architecture": architecture, "seed": seed, "prediction_sha256": digest}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=sorted(COHORT_STORE_NAMES), required=True)
    parser.add_argument("--store-root", type=Path, default=Path("data/cache/waveforms"))
    parser.add_argument(
        "--normalization", type=Path, default=Path("data/derived/training_normalization.json")
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/predictions/protected"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--training-lock",
        type=Path,
        default=Path("/tmp/ml-train.lock"),
        help=(
            "Shared accelerator lock, at the path fixed by the registered protocol. Acquired "
            "around the inference passes so this run serializes with any concurrent training "
            "on the same device. Scheduling only; no result depends on it."
        ),
    )
    parser.add_argument(
        "--record-batch-size",
        type=int,
        default=32,
        help=(
            "records per inference batch. Window averaging is strictly per record, "
            "so this affects throughput only, never the record-level predictions."
        ),
    )
    return parser.parse_args()


def run_cohort_inference(
    *,
    root: Path,
    choices: dict,
    dataset: InferenceRecordDataset,
    indices: np.ndarray,
    expected_targets: np.ndarray,
    device: torch.device,
    cohort: str,
    record_batch_size: int,
    cohort_root: Path,
    receipt: dict,
    receipt_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Run every sealed checkpoint over the cohort and build the model ensembles.

    All GPU work happens inside this function; the caller holds the exclusive
    device lock around it. Per-seed predictions are cached to disk, so an
    interrupted pass resumes without recomputing completed seeds. The sealed
    checkpoint digest is verified on every path, cached or computed.
    """

    cached_digests = recorded_prediction_digests(receipt)
    ensemble_outputs: dict[str, np.ndarray] = {}
    reference_targets: np.ndarray | None = None
    reference_indices: np.ndarray | None = None
    for architecture, architecture_choices in choices["architectures"].items():
        seed_probabilities: list[np.ndarray] = []
        for run in architecture_choices["runs"]:
            seed = int(run["seed"])
            output_path = cohort_root / f"{architecture}_seed_{seed}.npz"
            checkpoint_path = root / str(run["checkpoint"])
            checkpoint_digest = sha256_file(checkpoint_path)
            if checkpoint_digest != run["checkpoint_sha256"]:
                raise RuntimeError(
                    f"checkpoint hash mismatch: {checkpoint_path} hashes to {checkpoint_digest}, "
                    f"the sealed choices record {run['checkpoint_sha256']}"
                )
            if output_path.exists():
                logging.info(
                    "reusing cached %s seed %d predictions for %s", architecture, seed, cohort
                )
                probabilities, targets, record_indices = load_cached_prediction(
                    output_path,
                    expected_digest=cached_digests.get((architecture, seed)),
                    expected_indices=indices,
                    expected_targets=expected_targets,
                )
            else:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model = build_model(architecture).to(device)
                model.load_state_dict(checkpoint["model_state"])
                logging.info(
                    "starting %s seed %d on %s (%d records)",
                    architecture,
                    seed,
                    cohort,
                    len(indices),
                )
                probabilities, targets, record_indices = predict_records(
                    model,
                    dataset,
                    device=device,
                    record_batch_size=record_batch_size,
                    progress=lambda current, total, model_name=architecture, run_seed=seed: (
                        logging.info("%s seed %d batch %d/%d", model_name, run_seed, current, total)
                        if current % 100 == 0 or current == total
                        else None
                    ),
                )
                validate_prediction_arrays(
                    probabilities,
                    targets,
                    record_indices,
                    expected_indices=indices,
                    expected_targets=expected_targets,
                    where=output_path,
                )
                temporary = output_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        probabilities=probabilities,
                        targets=targets,
                        record_indices=record_indices,
                    )
                os.replace(temporary, output_path)
                del model, checkpoint
            if reference_targets is None:
                reference_targets = targets
                reference_indices = record_indices
            elif not np.array_equal(reference_targets, targets) or not np.array_equal(
                reference_indices, record_indices
            ):
                raise RuntimeError("protected predictions do not share targets and record order")
            seed_probabilities.append(probabilities)
            record_seed_pass(receipt, architecture, seed, sha256_file(output_path))
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        ensemble = np.mean(seed_probabilities, axis=0)
        temperature = float(architecture_choices["temperature"])
        ensemble_outputs[f"{architecture}_probabilities"] = ensemble.astype(np.float32)
        ensemble_outputs[f"{architecture}_calibrated_probabilities"] = apply_temperature(
            ensemble, temperature
        ).astype(np.float32)
    if reference_targets is None or reference_indices is None:
        raise RuntimeError("protected inference produced no predictions")
    return ensemble_outputs, reference_targets, reference_indices


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_preregistration_seal(root)
    evaluation_seal, choices = verify_evaluation_seal(root)
    cohort_entry = verify_bound_inputs(
        root,
        choices,
        args.cohort,
        store_root=args.store_root,
        normalization=args.normalization,
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = choose_device(args.device)
    mean, standard_deviation = load_normalization(resolve_under_root(root, args.normalization))
    store = WaveformStore(root / str(cohort_entry["store_root"]))
    if args.cohort == "ptb_test":
        indices = np.asarray(store.indices_for_folds((10,)), dtype=np.int64)
    else:
        indices = np.arange(len(store.metadata), dtype=np.int64)
    verify_cohort_indices(indices, cohort_entry, args.cohort)
    expected_targets = np.asarray(store.targets[indices], dtype=np.uint8)
    dataset = InferenceRecordDataset(store, indices, mean, standard_deviation)

    cohort_root = resolve_under_root(root, args.output_root) / args.cohort
    cohort_root.mkdir(parents=True, exist_ok=True)
    receipt_path = cohort_root / "receipt.json"
    receipt = prepare_receipt(
        receipt_path,
        cohort=args.cohort,
        records=len(indices),
        device=device,
        choices_commit=evaluation_seal["choices_commit"],
        checkpoint_digests=sealed_checkpoint_digests(choices),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cohort_root / "inference.log"),
        ],
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    with exclusive_device_lock(args.training_lock):
        ensemble_outputs, reference_targets, reference_indices = run_cohort_inference(
            root=root,
            choices=choices,
            dataset=dataset,
            indices=indices,
            expected_targets=expected_targets,
            device=device,
            cohort=args.cohort,
            record_batch_size=args.record_batch_size,
            cohort_root=cohort_root,
            receipt=receipt,
            receipt_path=receipt_path,
        )

    ensemble_path = cohort_root / "ensembles.npz"
    np.savez_compressed(
        ensemble_path,
        targets=reference_targets,
        record_indices=reference_indices,
        **ensemble_outputs,
    )
    receipt.update(
        {
            "status": "complete",
            "targets_sha256": array_sha256(reference_targets),
            "record_indices_sha256": array_sha256(reference_indices),
            "ensemble_file": os.path.relpath(ensemble_path, root),
            "ensemble_sha256": sha256_file(ensemble_path),
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    logging.info("protected inference complete for %s", args.cohort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
