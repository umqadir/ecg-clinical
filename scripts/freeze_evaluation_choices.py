#!/usr/bin/env python3
"""Freeze validation-derived checkpoints, thresholds, and temperatures.

This is a validation-only step. It reads fold-9 predictions, the sealed
preregistration artifacts, and cohort metadata, and it never opens a waveform
signal store: the signal and target digests written into ``bound_inputs`` are
copied from the pre-seal waveform audit rather than recomputed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecg_clinical.integrity import verify_preregistration_seal
from ecg_clinical.metrics import (
    apply_temperature,
    binary_negative_log_likelihood,
    calibration_summary,
    discrimination_summary,
    fit_scalar_temperature,
    select_f1_thresholds,
)

ARCHITECTURES = ("xresnet1d101", "s4d")
SEEDS = (17, 29, 43)

COHORT_STORE_NAMES = {
    "ptb_test": "ptb_xl",
    "chapman_shaoxing": "chapman_shaoxing",
    "ningbo": "ningbo",
}
VALIDATION_FOLD = 9
TEST_FOLD = 10
REQUIRED_COMPLETED_EPOCHS = 50
TARGET_COUNT = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def relative_to_root(root: Path, path: Path) -> str:
    return os.path.relpath(resolve_under_root(root, path), root)


def fold_indices(metadata: pd.DataFrame, fold: int) -> np.ndarray:
    """Return the store row indices of one PTB-XL stratification fold.

    This matches ``WaveformStore.indices_for_folds`` exactly, and is cast to
    int64 so its digest is stable across platforms.
    """

    return np.flatnonzero((metadata["strat_fold"] == fold).to_numpy()).astype(np.int64)


def cohort_record_indices(cohort: str, metadata: pd.DataFrame) -> np.ndarray:
    """The record-index array protected inference will use for one cohort."""

    if cohort == "ptb_test":
        return fold_indices(metadata, TEST_FOLD)
    return np.arange(len(metadata), dtype=np.int64)


def build_bound_inputs(
    root: Path,
    *,
    store_root: Path,
    normalization: Path,
    protocol: Path,
    label_manifest: Path,
    waveform_audit: Path,
) -> dict[str, object]:
    """Bind every runtime input of protected inference to a digest.

    Signal and target digests are copied from the committed waveform audit. The
    signal stores are not opened here; they are protected data and are re-hashed
    only by protected inference, which legitimately reads them.
    """

    audit_path = resolve_under_root(root, waveform_audit)
    audit = json.loads(audit_path.read_text())
    audit_cohorts = audit.get("cohorts")
    if not isinstance(audit_cohorts, dict):
        raise RuntimeError(f"waveform audit has no cohorts section: {audit_path}")

    bound: dict[str, object] = {}
    for key, path in (
        ("normalization", normalization),
        ("protocol", protocol),
        ("label_manifest", label_manifest),
        ("waveform_audit", waveform_audit),
    ):
        absolute = resolve_under_root(root, path)
        if not absolute.is_file():
            raise RuntimeError(f"bound input is missing: {key} at {absolute}")
        bound[key] = {"path": relative_to_root(root, path), "sha256": sha256(absolute)}

    cohorts: dict[str, object] = {}
    for cohort, store_name in COHORT_STORE_NAMES.items():
        cohort_store = resolve_under_root(root, store_root) / store_name
        audit_entry = audit_cohorts.get(store_name)
        if not isinstance(audit_entry, dict):
            raise RuntimeError(f"waveform audit has no entry for store {store_name}")
        for digest_key in ("signals_sha256", "targets_sha256"):
            if not isinstance(audit_entry.get(digest_key), str):
                raise RuntimeError(f"waveform audit for {store_name} has no {digest_key}")
        metadata_path = cohort_store / "metadata.parquet"
        status_path = cohort_store / "status.npy"
        for required in (metadata_path, status_path):
            if not required.is_file():
                raise RuntimeError(f"cohort store file is missing: {required}")
        metadata = pd.read_parquet(metadata_path)
        audited_records = audit_entry.get("records")
        if audited_records is not None and int(audited_records) != len(metadata):
            raise RuntimeError(
                f"cohort {cohort} store metadata holds {len(metadata)} records but the "
                f"waveform audit records {audited_records}"
            )
        indices = cohort_record_indices(cohort, metadata)
        if indices.size == 0:
            raise RuntimeError(f"cohort {cohort} resolves to an empty record set")
        cohorts[cohort] = {
            "store_root": relative_to_root(root, Path(store_root) / store_name),
            "signals_sha256": audit_entry["signals_sha256"],
            "targets_sha256": audit_entry["targets_sha256"],
            "metadata_sha256": sha256(metadata_path),
            "status_sha256": sha256(status_path),
            "records": int(indices.size),
            "record_indices_sha256": array_sha256(indices),
        }
    bound["cohorts"] = cohorts
    return bound


def validate_run_manifest(manifest: dict, *, architecture: str, seed: int, where: Path) -> None:
    if manifest.get("model") != architecture:
        raise RuntimeError(
            f"run manifest model {manifest.get('model')!r} does not match its directory "
            f"architecture {architecture!r}: {where}"
        )
    if manifest.get("seed") != seed:
        raise RuntimeError(
            f"run manifest seed {manifest.get('seed')!r} does not match its directory "
            f"seed {seed}: {where}"
        )
    if manifest.get("status") != "complete":
        raise RuntimeError(f"run is not complete ({manifest.get('status')!r}): {where}")
    if manifest.get("completed_epochs") != REQUIRED_COMPLETED_EPOCHS:
        raise RuntimeError(
            f"run reports {manifest.get('completed_epochs')!r} completed epochs, the frozen "
            f"protocol requires {REQUIRED_COMPLETED_EPOCHS}: {where}"
        )
    if manifest.get("protected_evaluation_accessed") is not False:
        raise RuntimeError(
            "run manifest does not attest that protected evaluation data was untouched "
            f"(protected_evaluation_accessed={manifest.get('protected_evaluation_accessed')!r}): "
            f"{where}"
        )
    limits = manifest.get("development_limits", {})
    if any(value is not None for value in limits.values()):
        raise RuntimeError(f"development-limited run cannot be sealed: {where}")


def validate_checkpoint(
    checkpoint: dict, *, architecture: str, seed: int, manifest: dict, where: Path
) -> None:
    """Require the checkpoint to be the one the manifest names as selected.

    Training writes a zero-based epoch index into the checkpoint and reports the
    one-based epoch number in the manifest, so the selected epoch matches when
    ``checkpoint["epoch"] + 1 == manifest["best_epoch"]``.
    """

    if checkpoint.get("model") != architecture:
        raise RuntimeError(
            f"checkpoint model {checkpoint.get('model')!r} does not match {architecture!r}: {where}"
        )
    if checkpoint.get("seed") != seed:
        raise RuntimeError(
            f"checkpoint seed {checkpoint.get('seed')!r} does not match {seed}: {where}"
        )
    best_epoch = manifest.get("best_epoch")
    if not isinstance(best_epoch, int):
        raise RuntimeError(f"run manifest does not record an integer best_epoch: {where}")
    if int(checkpoint.get("epoch", -1)) + 1 != best_epoch:
        raise RuntimeError(
            f"checkpoint holds epoch {checkpoint.get('epoch')!r} (one-based "
            f"{int(checkpoint.get('epoch', -1)) + 1}) but the manifest selects epoch "
            f"{best_epoch}: {where}"
        )


def validate_validation_predictions(
    probabilities: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    *,
    expected_indices: np.ndarray,
    expected_targets: np.ndarray,
    where: Path,
) -> None:
    """Require the stored fold-9 predictions to be well formed and really fold 9."""

    expected_shape = (len(expected_indices), TARGET_COUNT)
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            f"validation probabilities have shape {probabilities.shape}, expected "
            f"{expected_shape}: {where}"
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"validation probabilities contain non-finite values: {where}")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise RuntimeError(
            f"validation probabilities fall outside [0, 1] "
            f"({probabilities.min()!r} to {probabilities.max()!r}): {where}"
        )
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError(
            f"validation record indices are not exactly the PTB-XL fold-{VALIDATION_FOLD} "
            f"records: {where}"
        )
    if not np.array_equal(targets, expected_targets):
        raise RuntimeError(
            f"validation targets do not match the waveform store at the fold-"
            f"{VALIDATION_FOLD} records: {where}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--label-manifest",
        type=Path,
        default=Path("data/derived/preregistration/harmonized_labels.json"),
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/preregistered_protocol.json")
    )
    parser.add_argument(
        "--normalization", type=Path, default=Path("data/derived/training_normalization.json")
    )
    parser.add_argument(
        "--waveform-audit", type=Path, default=Path("data/derived/waveform_audit.json")
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/cache/waveforms"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation_choices.json"))
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("artifacts/predictions/validation_ensembles.npz"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    registration = verify_preregistration_seal(root)
    label_manifest_path = resolve_under_root(root, args.label_manifest)
    label_manifest = json.loads(label_manifest_path.read_text())
    label_keys = [label["label_key"] for label in label_manifest["labels"]]
    headline_indices = np.asarray(
        [label_keys.index(key) for key in label_manifest["headline_label_keys"]], dtype=np.int64
    )

    bound_inputs = build_bound_inputs(
        root,
        store_root=args.store_root,
        normalization=args.normalization,
        protocol=args.protocol,
        label_manifest=args.label_manifest,
        waveform_audit=args.waveform_audit,
    )

    # Prove the validation set really is fold 9 rather than trusting the run
    # artifacts to describe themselves. Metadata and targets are not protected;
    # the PTB signal store is never opened here.
    ptb_store = resolve_under_root(root, args.store_root) / COHORT_STORE_NAMES["ptb_test"]
    ptb_metadata = pd.read_parquet(ptb_store / "metadata.parquet")
    expected_indices = fold_indices(ptb_metadata, VALIDATION_FOLD)
    store_targets = np.load(ptb_store / "targets.npy", mmap_mode="r")
    expected_targets = np.asarray(store_targets[expected_indices], dtype=np.uint8)
    if expected_indices.size == 0:
        raise RuntimeError(f"PTB-XL store holds no fold-{VALIDATION_FOLD} records: {ptb_store}")

    choices: dict[str, object] = {
        "schema_version": 1,
        "study_id": "ecg-clinical-shift-v1",
        "registration_commit": registration["registration_commit"],
        "label_keys": label_keys,
        "headline_label_keys": label_manifest["headline_label_keys"],
        "validation_cohort": "PTB-XL 1.0.3 fold 9",
        "architectures": {},
        "bound_inputs": bound_inputs,
        "protected_inference_completed": False,
    }
    reference_targets: np.ndarray | None = None
    reference_indices: np.ndarray | None = None
    ensemble_outputs: dict[str, np.ndarray] = {}
    for architecture in ARCHITECTURES:
        seed_probabilities: list[np.ndarray] = []
        run_records: list[dict[str, object]] = []
        for seed in SEEDS:
            run_directory = resolve_under_root(root, args.run_root) / architecture / f"seed_{seed}"
            manifest_path = run_directory / "run_manifest.json"
            checkpoint_path = run_directory / "best.pt"
            prediction_path = run_directory / "validation.npz"
            best_prediction_path = run_directory / "best_validation_predictions.npz"
            manifest = json.loads(manifest_path.read_text())
            validate_run_manifest(
                manifest, architecture=architecture, seed=seed, where=run_directory
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            validate_checkpoint(
                checkpoint,
                architecture=architecture,
                seed=seed,
                manifest=manifest,
                where=checkpoint_path,
            )
            del checkpoint
            # Training copies the selected epoch's predictions to validation.npz
            # only when a run completes. Requiring the two files to be identical
            # proves the sealed thresholds and temperature are fit on the same
            # epoch the sealed checkpoint holds, not on a later or stale epoch.
            if sha256(prediction_path) != sha256(best_prediction_path):
                raise RuntimeError(
                    f"validation predictions are not the selected epoch's predictions: "
                    f"{prediction_path} differs from {best_prediction_path}"
                )
            prediction = np.load(prediction_path)
            probabilities = np.asarray(prediction["probabilities"], dtype=np.float64)
            targets = np.asarray(prediction["targets"], dtype=np.uint8)
            indices = np.asarray(prediction["record_indices"], dtype=np.int64)
            validate_validation_predictions(
                probabilities,
                targets,
                indices,
                expected_indices=expected_indices,
                expected_targets=expected_targets,
                where=prediction_path,
            )
            if reference_targets is None:
                reference_targets = targets
                reference_indices = indices
            elif not np.array_equal(targets, reference_targets) or not np.array_equal(
                indices, reference_indices
            ):
                raise RuntimeError("validation predictions do not share records and targets")
            seed_probabilities.append(probabilities)
            run_records.append(
                {
                    "run_id": f"{architecture}-seed-{seed}",
                    "seed": seed,
                    "run_directory": relative_to_root(root, run_directory),
                    "checkpoint": relative_to_root(root, checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                    "selected_epoch": manifest["best_epoch"],
                    "validation_macro_auroc": manifest["best_validation_macro_auroc"],
                    "validation_nll": manifest["best_validation_nll"],
                }
            )
        if reference_targets is None:
            raise RuntimeError("no validation targets loaded")
        ensemble = np.mean(seed_probabilities, axis=0)
        thresholds = select_f1_thresholds(reference_targets, ensemble)
        temperature = fit_scalar_temperature(reference_targets, ensemble)
        calibrated = apply_temperature(ensemble, temperature)
        choices["architectures"][architecture] = {
            "runs": run_records,
            "ensemble_rule": "arithmetic mean of three seed probabilities",
            "thresholds_by_label": dict(zip(label_keys, thresholds.tolist(), strict=True)),
            "temperature": temperature,
            "validation": {
                "uncalibrated_discrimination": discrimination_summary(
                    reference_targets, ensemble, headline_indices
                ),
                "uncalibrated_calibration": calibration_summary(
                    reference_targets, ensemble, headline_indices
                ),
                "calibrated_calibration": calibration_summary(
                    reference_targets, calibrated, headline_indices
                ),
                "uncalibrated_nll_all_targets": binary_negative_log_likelihood(
                    reference_targets, ensemble
                ),
                "calibrated_nll_all_targets": binary_negative_log_likelihood(
                    reference_targets, calibrated
                ),
            },
        }
        ensemble_outputs[f"{architecture}_probabilities"] = ensemble.astype(np.float32)
        ensemble_outputs[f"{architecture}_calibrated_probabilities"] = calibrated.astype(np.float32)
    if reference_targets is None or reference_indices is None:
        raise RuntimeError("validation ensemble is empty")
    choices["validation_targets_sha256"] = array_sha256(reference_targets)
    choices["validation_record_indices_sha256"] = array_sha256(reference_indices)
    choices["validation_records"] = len(reference_targets)
    choices["seal_instruction"] = (
        "Commit this file before any PTB-XL fold-10 or external model inference, then create "
        "EVALUATION_SEAL.json referencing that commit."
    )
    output = resolve_under_root(root, args.output)
    validation_output = resolve_under_root(root, args.validation_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(choices, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        validation_output,
        targets=reference_targets,
        record_indices=reference_indices,
        **ensemble_outputs,
    )
    print(json.dumps(choices, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
