"""Unit tests for the guards protected inference applies before and during a pass.

The module under test is a script, so it is imported by path. Nothing here runs
inference or opens the project's real waveform stores.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import load_script, write_fake_store

inference = load_script("run_protected_inference")

STORE_ROOT = Path("data/cache/waveforms")
NORMALIZATION = Path("data/derived/training_normalization.json")
PROTOCOL = Path("configs/preregistered_protocol.json")
LABEL_MANIFEST = Path("data/derived/preregistration/harmonized_labels.json")
WAVEFORM_AUDIT = Path("data/derived/waveform_audit.json")
RECORDS = 6


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bound_root(tmp_path: Path) -> tuple[Path, dict, np.ndarray, np.ndarray]:
    """Create a tiny cohort store and the sealed choices that bind it."""

    for relative, payload in (
        (NORMALIZATION, {"per_lead_mean_mv": [0.0] * 12}),
        (PROTOCOL, {"protocol": "frozen"}),
        (LABEL_MANIFEST, {"labels": []}),
        (WAVEFORM_AUDIT, {"cohorts": {}}),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    store = tmp_path / STORE_ROOT / "chapman_shaoxing"
    contents = write_fake_store(store, records=RECORDS)
    np.save(store / "signals.npy", np.zeros((RECORDS, 12, 4), dtype=np.float16))
    indices = np.arange(RECORDS, dtype=np.int64)

    choices = {
        "bound_inputs": {
            "normalization": {
                "path": str(NORMALIZATION),
                "sha256": sha256_file(tmp_path / NORMALIZATION),
            },
            "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(tmp_path / PROTOCOL)},
            "label_manifest": {
                "path": str(LABEL_MANIFEST),
                "sha256": sha256_file(tmp_path / LABEL_MANIFEST),
            },
            "waveform_audit": {
                "path": str(WAVEFORM_AUDIT),
                "sha256": sha256_file(tmp_path / WAVEFORM_AUDIT),
            },
            "cohorts": {
                "chapman_shaoxing": {
                    "store_root": str(STORE_ROOT / "chapman_shaoxing"),
                    "signals_sha256": sha256_file(store / "signals.npy"),
                    "targets_sha256": sha256_file(store / "targets.npy"),
                    "metadata_sha256": sha256_file(store / "metadata.parquet"),
                    "status_sha256": sha256_file(store / "status.npy"),
                    "records": RECORDS,
                    "record_indices_sha256": inference.array_sha256(indices),
                }
            },
        }
    }
    return tmp_path, choices, indices, contents["targets"]


def verify(root: Path, choices: dict) -> dict:
    return inference.verify_bound_inputs(
        root,
        choices,
        "chapman_shaoxing",
        store_root=STORE_ROOT,
        normalization=NORMALIZATION,
    )


def test_bound_inputs_verify_when_every_digest_matches(tmp_path: Path) -> None:
    root, choices, _, _ = build_bound_root(tmp_path)

    entry = verify(root, choices)

    assert entry["records"] == RECORDS


def test_missing_bound_inputs_section_is_refused(tmp_path: Path) -> None:
    root, choices, _, _ = build_bound_root(tmp_path)
    choices.pop("bound_inputs")

    with pytest.raises(RuntimeError, match="no bound_inputs section"):
        verify(root, choices)


@pytest.mark.parametrize("key", ["normalization", "protocol", "label_manifest", "waveform_audit"])
def test_rewritten_bound_input_file_is_rejected(tmp_path: Path, key: str) -> None:
    root, choices, _, _ = build_bound_root(tmp_path)
    path = root / choices["bound_inputs"][key]["path"]
    path.write_text(path.read_text() + " ")

    with pytest.raises(RuntimeError, match=f"bound input {key} does not match the evaluation seal"):
        verify(root, choices)


@pytest.mark.parametrize(
    ("filename", "digest_key"),
    [
        ("signals.npy", "signals_sha256"),
        ("targets.npy", "targets_sha256"),
        ("metadata.parquet", "metadata_sha256"),
        ("status.npy", "status_sha256"),
    ],
)
def test_rebuilt_cohort_store_file_is_rejected(
    tmp_path: Path, filename: str, digest_key: str
) -> None:
    root, choices, _, _ = build_bound_root(tmp_path)
    choices["bound_inputs"]["cohorts"]["chapman_shaoxing"][digest_key] = "0" * 64

    with pytest.raises(RuntimeError, match=f"store file does not match.*{filename}"):
        verify(root, choices)


def test_runtime_path_outside_the_seal_is_rejected(tmp_path: Path) -> None:
    root, choices, _, _ = build_bound_root(tmp_path)
    elsewhere = root / "data/derived/other_normalization.json"
    elsewhere.write_text("{}")

    with pytest.raises(RuntimeError, match="--normalization points at"):
        inference.verify_bound_inputs(
            root,
            choices,
            "chapman_shaoxing",
            store_root=STORE_ROOT,
            normalization=Path("data/derived/other_normalization.json"),
        )


def test_wrong_record_index_digest_is_rejected(tmp_path: Path) -> None:
    _, choices, indices, _ = build_bound_root(tmp_path)
    entry = choices["bound_inputs"]["cohorts"]["chapman_shaoxing"]

    inference.verify_cohort_indices(indices, entry, "chapman_shaoxing")

    permuted = indices[::-1].copy()
    with pytest.raises(RuntimeError, match="record indices do not match the evaluation seal"):
        inference.verify_cohort_indices(permuted, entry, "chapman_shaoxing")


def test_wrong_record_count_is_rejected(tmp_path: Path) -> None:
    _, choices, indices, _ = build_bound_root(tmp_path)
    entry = choices["bound_inputs"]["cohorts"]["chapman_shaoxing"]

    with pytest.raises(RuntimeError, match="resolves to 5 records"):
        inference.verify_cohort_indices(indices[:-1], entry, "chapman_shaoxing")


def write_prediction(
    path: Path, probabilities: np.ndarray, targets: np.ndarray, indices: np.ndarray
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=probabilities, targets=targets, record_indices=indices)
    return path


def good_prediction(tmp_path: Path, targets: np.ndarray, indices: np.ndarray) -> Path:
    probabilities = np.full((len(indices), inference.TARGET_COUNT), 0.25, dtype=np.float32)
    return write_prediction(tmp_path / "cached.npz", probabilities, targets, indices)


def test_cached_prediction_is_accepted_when_the_receipt_vouches_for_it(tmp_path: Path) -> None:
    _, _, indices, targets = build_bound_root(tmp_path)
    path = good_prediction(tmp_path, targets, indices)

    probabilities, loaded_targets, loaded_indices = inference.load_cached_prediction(
        path,
        expected_digest=sha256_file(path),
        expected_indices=indices,
        expected_targets=targets,
    )

    assert probabilities.shape == (RECORDS, inference.TARGET_COUNT)
    assert np.array_equal(loaded_targets, targets)
    assert np.array_equal(loaded_indices, indices)


def test_cached_prediction_without_a_receipt_entry_is_rejected(tmp_path: Path) -> None:
    _, _, indices, targets = build_bound_root(tmp_path)
    path = good_prediction(tmp_path, targets, indices)

    with pytest.raises(RuntimeError, match="not recorded in the retained receipt"):
        inference.load_cached_prediction(
            path, expected_digest=None, expected_indices=indices, expected_targets=targets
        )


def test_cached_prediction_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    _, _, indices, targets = build_bound_root(tmp_path)
    path = good_prediction(tmp_path, targets, indices)

    with pytest.raises(RuntimeError, match="the retained receipt records"):
        inference.load_cached_prediction(
            path, expected_digest="0" * 64, expected_indices=indices, expected_targets=targets
        )


def test_cached_prediction_with_wrong_record_indices_is_rejected(tmp_path: Path) -> None:
    _, _, indices, targets = build_bound_root(tmp_path)
    probabilities = np.full((RECORDS, inference.TARGET_COUNT), 0.25, dtype=np.float32)
    path = write_prediction(tmp_path / "cached.npz", probabilities, targets, indices[::-1].copy())

    with pytest.raises(RuntimeError, match="not in the sealed cohort record order"):
        inference.load_cached_prediction(
            path,
            expected_digest=sha256_file(path),
            expected_indices=indices,
            expected_targets=targets,
        )


def test_cached_prediction_with_wrong_targets_is_rejected(tmp_path: Path) -> None:
    _, _, indices, targets = build_bound_root(tmp_path)
    probabilities = np.full((RECORDS, inference.TARGET_COUNT), 0.25, dtype=np.float32)
    corrupted = targets.copy()
    corrupted[0, 0] ^= 1
    path = write_prediction(tmp_path / "cached.npz", probabilities, corrupted, indices)

    with pytest.raises(RuntimeError, match="targets do not match the waveform store"):
        inference.load_cached_prediction(
            path,
            expected_digest=sha256_file(path),
            expected_indices=indices,
            expected_targets=targets,
        )


@pytest.mark.parametrize(
    ("probabilities", "pattern"),
    [
        (np.full((RECORDS, 3), 0.5, dtype=np.float32), "have shape"),
        (np.full((RECORDS, 16), np.nan, dtype=np.float32), "non-finite"),
        (np.full((RECORDS, 16), 1.5, dtype=np.float32), r"outside \[0, 1\]"),
    ],
)
def test_malformed_cached_probabilities_are_rejected(
    tmp_path: Path, probabilities: np.ndarray, pattern: str
) -> None:
    _, _, indices, targets = build_bound_root(tmp_path)
    path = write_prediction(tmp_path / "cached.npz", probabilities, targets, indices)

    with pytest.raises(RuntimeError, match=pattern):
        inference.load_cached_prediction(
            path,
            expected_digest=sha256_file(path),
            expected_indices=indices,
            expected_targets=targets,
        )


def receipt_arguments(**overrides: object) -> dict:
    arguments: dict[str, object] = {
        "cohort": "chapman_shaoxing",
        "records": RECORDS,
        "device": "cpu",
        "choices_commit": "d" * 40,
        "checkpoint_digests": {"s4d-seed-17": "a" * 64},
    }
    arguments.update(overrides)
    return arguments


def test_fresh_receipt_is_created(tmp_path: Path) -> None:
    receipt = inference.prepare_receipt(tmp_path / "receipt.json", **receipt_arguments())

    assert receipt["status"] == "in_progress"
    assert receipt["completed_seed_passes"] == []


def test_completed_receipt_refuses_a_second_pass(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"status": "complete", "cohort": "chapman_shaoxing"}))

    with pytest.raises(RuntimeError, match="already completed"):
        inference.prepare_receipt(path, **receipt_arguments())


def test_in_progress_receipt_is_retained_rather_than_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    original = inference.prepare_receipt(path, **receipt_arguments())
    inference.record_seed_pass(original, "s4d", 17, "e" * 64)
    path.write_text(json.dumps(original, indent=2, sort_keys=True))

    resumed = inference.prepare_receipt(path, **receipt_arguments())

    assert resumed["completed_seed_passes"] == original["completed_seed_passes"]
    assert inference.recorded_prediction_digests(resumed) == {("s4d", 17): "e" * 64}


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"cohort": "ningbo"}, "disagrees with this pass on cohort"),
        ({"choices_commit": "f" * 40}, "evaluation_seal_choices_commit"),
        ({"records": RECORDS + 1}, "disagrees with this pass on records"),
        ({"checkpoint_digests": {"s4d-seed-17": "b" * 64}}, "different sealed checkpoint digests"),
    ],
)
def test_disagreeing_receipt_aborts_instead_of_restarting(
    tmp_path: Path, overrides: dict, pattern: str
) -> None:
    path = tmp_path / "receipt.json"
    existing = inference.prepare_receipt(path, **receipt_arguments())
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))

    with pytest.raises(RuntimeError, match=pattern):
        inference.prepare_receipt(path, **receipt_arguments(**overrides))


def test_legacy_receipt_without_checkpoint_digests_aborts(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohort": "chapman_shaoxing",
                "records": RECORDS,
                "evaluation_seal_choices_commit": "d" * 40,
                "status": "in_progress",
                "completed_seed_passes": [],
            }
        )
    )

    with pytest.raises(RuntimeError, match="different sealed checkpoint digests"):
        inference.prepare_receipt(path, **receipt_arguments())


def test_recorded_seed_passes_are_updated_in_place(tmp_path: Path) -> None:
    receipt = inference.prepare_receipt(tmp_path / "receipt.json", **receipt_arguments())

    inference.record_seed_pass(receipt, "s4d", 17, "a" * 64)
    inference.record_seed_pass(receipt, "s4d", 17, "b" * 64)
    inference.record_seed_pass(receipt, "s4d", 29, "c" * 64)

    assert len(receipt["completed_seed_passes"]) == 2
    assert inference.recorded_prediction_digests(receipt) == {
        ("s4d", 17): "b" * 64,
        ("s4d", 29): "c" * 64,
    }


def test_sealed_checkpoint_digests_are_collected_per_run(tmp_path: Path) -> None:
    choices = {
        "architectures": {
            "s4d": {"runs": [{"seed": 17, "checkpoint_sha256": "a" * 64}]},
            "xresnet1d101": {"runs": [{"seed": 29, "checkpoint_sha256": "b" * 64}]},
        }
    }

    assert inference.sealed_checkpoint_digests(choices) == {
        "s4d-seed-17": "a" * 64,
        "xresnet1d101-seed-29": "b" * 64,
    }
