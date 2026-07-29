"""Unit tests for the validation-only freezing step.

These build fake cohort stores in ``tmp_path``. Crucially, no ``signals.npy`` is
written: the signal and target digests come from the waveform audit, so the
freezing step must produce a complete ``bound_inputs`` object without ever
opening a protected signal store, and these tests fail if it starts doing so.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import load_script, write_fake_store

freeze = load_script("freeze_evaluation_choices")

STORE_ROOT = Path("data/cache/waveforms")
COHORT_RECORDS = {"ptb_xl": 12, "chapman_shaoxing": 5, "ningbo": 7}


def build_project(tmp_path: Path) -> dict[str, object]:
    for relative in (
        "data/derived/training_normalization.json",
        "configs/preregistered_protocol.json",
        "data/derived/preregistration/harmonized_labels.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"file": relative}))

    # Twelve PTB-XL rows covering every fold once, with an extra fold-9 and
    # fold-10 record so the derived index arrays are not trivially contiguous.
    folds = np.concatenate([np.arange(1, 11), np.array([9, 10])]).astype(np.int64)
    contents = {}
    audit_cohorts = {}
    for store_name, records in COHORT_RECORDS.items():
        store = tmp_path / STORE_ROOT / store_name
        contents[store_name] = write_fake_store(
            store,
            records=records,
            folds=folds if store_name == "ptb_xl" else None,
        )
        audit_cohorts[store_name] = {
            "records": records,
            "signals_sha256": hashlib.sha256(f"{store_name}-signals".encode()).hexdigest(),
            "targets_sha256": hashlib.sha256(f"{store_name}-targets".encode()).hexdigest(),
        }
    audit_path = tmp_path / "data/derived/waveform_audit.json"
    audit_path.write_text(json.dumps({"schema_version": 1, "cohorts": audit_cohorts}))
    return {"audit_cohorts": audit_cohorts, "contents": contents, "folds": folds}


def build(tmp_path: Path) -> dict[str, object]:
    return freeze.build_bound_inputs(
        tmp_path,
        store_root=STORE_ROOT,
        normalization=Path("data/derived/training_normalization.json"),
        protocol=Path("configs/preregistered_protocol.json"),
        label_manifest=Path("data/derived/preregistration/harmonized_labels.json"),
        waveform_audit=Path("data/derived/waveform_audit.json"),
    )


def test_bound_inputs_carry_the_agreed_schema(tmp_path: Path) -> None:
    fixture = build_project(tmp_path)

    bound = build(tmp_path)

    assert set(bound) == {
        "normalization",
        "protocol",
        "label_manifest",
        "waveform_audit",
        "cohorts",
    }
    assert bound["normalization"]["path"] == "data/derived/training_normalization.json"
    assert set(bound["cohorts"]) == {"ptb_test", "chapman_shaoxing", "ningbo"}
    for cohort in bound["cohorts"].values():
        assert set(cohort) == {
            "store_root",
            "signals_sha256",
            "targets_sha256",
            "metadata_sha256",
            "status_sha256",
            "records",
            "record_indices_sha256",
        }
    audit = fixture["audit_cohorts"]
    assert bound["cohorts"]["ningbo"]["signals_sha256"] == audit["ningbo"]["signals_sha256"]
    assert bound["cohorts"]["ptb_test"]["targets_sha256"] == audit["ptb_xl"]["targets_sha256"]


def test_ptb_cohort_binds_the_fold_ten_records(tmp_path: Path) -> None:
    fixture = build_project(tmp_path)

    bound = build(tmp_path)

    expected = np.flatnonzero(np.asarray(fixture["folds"]) == 10).astype(np.int64)
    entry = bound["cohorts"]["ptb_test"]
    assert entry["records"] == len(expected)
    assert entry["record_indices_sha256"] == freeze.array_sha256(expected)


def test_external_cohorts_bind_every_record_in_order(tmp_path: Path) -> None:
    build_project(tmp_path)

    bound = build(tmp_path)

    for cohort, store_name in (("chapman_shaoxing", "chapman_shaoxing"), ("ningbo", "ningbo")):
        expected = np.arange(COHORT_RECORDS[store_name], dtype=np.int64)
        assert bound["cohorts"][cohort]["records"] == len(expected)
        assert bound["cohorts"][cohort]["record_indices_sha256"] == freeze.array_sha256(expected)


def test_audit_record_count_disagreeing_with_the_store_is_rejected(tmp_path: Path) -> None:
    build_project(tmp_path)
    audit_path = tmp_path / "data/derived/waveform_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["cohorts"]["ningbo"]["records"] = 999
    audit_path.write_text(json.dumps(audit))

    with pytest.raises(RuntimeError, match="waveform audit records 999"):
        build(tmp_path)


def test_missing_audit_digest_is_rejected(tmp_path: Path) -> None:
    build_project(tmp_path)
    audit_path = tmp_path / "data/derived/waveform_audit.json"
    audit = json.loads(audit_path.read_text())
    del audit["cohorts"]["chapman_shaoxing"]["signals_sha256"]
    audit_path.write_text(json.dumps(audit))

    with pytest.raises(RuntimeError, match="has no signals_sha256"):
        build(tmp_path)


def base_manifest() -> dict:
    return {
        "model": "s4d",
        "seed": 17,
        "status": "complete",
        "completed_epochs": 50,
        "protected_evaluation_accessed": False,
        "development_limits": {"train_limit": None, "validation_limit": None},
        "best_epoch": 13,
    }


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"model": "xresnet1d101"}, "does not match its directory architecture"),
        ({"seed": 29}, "does not match its directory seed"),
        ({"status": "in_progress"}, "run is not complete"),
        ({"completed_epochs": 49}, "completed epochs"),
        ({"protected_evaluation_accessed": True}, "protected evaluation data was untouched"),
        (
            {"development_limits": {"train_limit": 10, "validation_limit": None}},
            "development-limited run cannot be sealed",
        ),
    ],
)
def test_run_manifest_defects_are_rejected(tmp_path: Path, overrides: dict, pattern: str) -> None:
    manifest = base_manifest()
    manifest.update(overrides)

    with pytest.raises(RuntimeError, match=pattern):
        freeze.validate_run_manifest(manifest, architecture="s4d", seed=17, where=tmp_path)


def test_clean_run_manifest_passes(tmp_path: Path) -> None:
    freeze.validate_run_manifest(base_manifest(), architecture="s4d", seed=17, where=tmp_path)


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"model": "xresnet1d101"}, "checkpoint model"),
        ({"seed": 29}, "checkpoint seed"),
        ({"epoch": 30}, "but the manifest selects epoch 13"),
    ],
)
def test_checkpoint_defects_are_rejected(tmp_path: Path, overrides: dict, pattern: str) -> None:
    checkpoint = {"model": "s4d", "seed": 17, "epoch": 12}
    checkpoint.update(overrides)

    with pytest.raises(RuntimeError, match=pattern):
        freeze.validate_checkpoint(
            checkpoint,
            architecture="s4d",
            seed=17,
            manifest=base_manifest(),
            where=tmp_path,
        )


def test_selected_checkpoint_epoch_is_one_based_in_the_manifest(tmp_path: Path) -> None:
    freeze.validate_checkpoint(
        {"model": "s4d", "seed": 17, "epoch": 12},
        architecture="s4d",
        seed=17,
        manifest=base_manifest(),
        where=tmp_path,
    )


def validation_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.array([1, 4, 9], dtype=np.int64)
    targets = np.zeros((3, 16), dtype=np.uint8)
    targets[0, 0] = 1
    probabilities = np.full((3, 16), 0.5, dtype=np.float64)
    return probabilities, targets, indices


def test_clean_validation_predictions_pass(tmp_path: Path) -> None:
    probabilities, targets, indices = validation_fixture()

    freeze.validate_validation_predictions(
        probabilities,
        targets,
        indices,
        expected_indices=indices,
        expected_targets=targets,
        where=tmp_path,
    )


def test_validation_predictions_from_the_wrong_fold_are_rejected(tmp_path: Path) -> None:
    probabilities, targets, indices = validation_fixture()
    other_fold = np.array([2, 5, 11], dtype=np.int64)

    with pytest.raises(RuntimeError, match="not exactly the PTB-XL fold-9 records"):
        freeze.validate_validation_predictions(
            probabilities,
            targets,
            indices,
            expected_indices=other_fold,
            expected_targets=targets,
            where=tmp_path,
        )


def test_validation_targets_disagreeing_with_the_store_are_rejected(tmp_path: Path) -> None:
    probabilities, targets, indices = validation_fixture()
    store_targets = targets.copy()
    store_targets[2, 3] = 1

    with pytest.raises(RuntimeError, match="targets do not match the waveform store"):
        freeze.validate_validation_predictions(
            probabilities,
            targets,
            indices,
            expected_indices=indices,
            expected_targets=store_targets,
            where=tmp_path,
        )


@pytest.mark.parametrize(
    ("probabilities", "pattern"),
    [
        (np.full((3, 13), 0.5), "have shape"),
        (np.full((3, 16), np.inf), "non-finite"),
        (np.full((3, 16), -0.1), r"outside \[0, 1\]"),
    ],
)
def test_malformed_validation_probabilities_are_rejected(
    tmp_path: Path, probabilities: np.ndarray, pattern: str
) -> None:
    _, targets, indices = validation_fixture()

    with pytest.raises(RuntimeError, match=pattern):
        freeze.validate_validation_predictions(
            probabilities,
            targets,
            indices,
            expected_indices=indices,
            expected_targets=targets,
            where=tmp_path,
        )


def test_fold_indices_match_the_waveform_store_helper(tmp_path: Path) -> None:
    """``fold_indices`` must reproduce ``WaveformStore.indices_for_folds`` exactly."""

    from ecg_clinical.data import WaveformStore

    store_root = tmp_path / "store"
    folds = np.array([9, 10, 9, 1, 10], dtype=np.int64)
    write_fake_store(store_root, records=5, folds=folds)
    np.save(store_root / "signals.npy", np.zeros((5, 12, 1000), dtype=np.float16))

    store = WaveformStore(store_root)
    metadata = pd.read_parquet(store_root / "metadata.parquet")

    for fold in (9, 10):
        assert np.array_equal(freeze.fold_indices(metadata, fold), store.indices_for_folds((fold,)))
