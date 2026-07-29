"""End-to-end and provenance tests for the analysis step.

No protected prediction exists yet, so every fixture here is synthetic: a
throwaway sealed repository, small fake waveform stores, and fabricated
ensembles and receipts written into ``tmp_path``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import build_sealed_study, load_script

from ecg_clinical.metrics import apply_temperature

analysis = load_script("analyze_results")

LABEL_KEYS = [f"L{index:02d}" for index in range(16)]
HEADLINE_KEYS = LABEL_KEYS[:13]
ARCHITECTURES = ("xresnet1d101", "s4d")
TEMPERATURES = {"xresnet1d101": 1.3, "s4d": 0.85}
COHORT_STORE_NAMES = {
    "ptb_test": "ptb_xl",
    "chapman_shaoxing": "chapman_shaoxing",
    "ningbo": "ningbo",
}
STORE_RECORDS = {"ptb_test": 120, "chapman_shaoxing": 72, "ningbo": 72}
RECORD_SAMPLES = 1000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def write_store(root: Path, cohort: str, generator: np.random.Generator) -> np.ndarray:
    """Write a small but structurally valid waveform store."""

    records = STORE_RECORDS[cohort]
    root.mkdir(parents=True, exist_ok=True)
    ages = np.tile([31, 52, 71, 84], records // 4)
    if cohort == "ptb_test":
        folds = np.where(np.arange(records) < records // 2, 9, 10)
        sex: np.ndarray = np.tile([0, 1], records // 2)
        patients = [f"P{index // 2:04d}" for index in range(records)]
    else:
        folds = np.ones(records, dtype=np.int64)
        sex = np.tile(np.asarray(["Male", "Female"]), records // 2)
        patients = [f"P{index:04d}" for index in range(records)]
    metadata = pd.DataFrame(
        {
            "record_id": [f"{cohort}-{index:05d}" for index in range(records)],
            "patient_id": patients,
            "strat_fold": np.asarray(folds, dtype=np.int64),
            "age": ages.astype(np.int64),
            "sex": sex,
        }
    )
    metadata.to_parquet(root / "metadata.parquet")
    targets = generator.integers(0, 2, size=(records, len(LABEL_KEYS))).astype(np.uint8)
    np.save(root / "targets.npy", targets)
    np.save(root / "status.npy", np.ones(records, dtype=np.uint8))
    np.save(
        root / "signals.npy",
        np.zeros((records, 12, RECORD_SAMPLES), dtype=np.float16),
    )
    return targets


def cohort_record_indices(cohort: str, metadata: pd.DataFrame) -> np.ndarray:
    if cohort == "ptb_test":
        return np.flatnonzero((metadata["strat_fold"] == 10).to_numpy()).astype(np.int64)
    return np.arange(len(metadata), dtype=np.int64)


def build_synthetic_study(
    tmp_path: Path,
    *,
    bound_overrides: dict | None = None,
    receipt_overrides: dict | None = None,
) -> argparse.Namespace:
    """Build a sealed repository, fake stores, ensembles, and receipts."""

    generator = np.random.default_rng(41)
    store_root = tmp_path / "stores"
    prediction_root = tmp_path / "predictions"
    repository = tmp_path / "repo"
    repository.mkdir()

    store_targets: dict[str, np.ndarray] = {}
    bound_cohorts: dict[str, dict] = {}
    for cohort, store_name in COHORT_STORE_NAMES.items():
        cohort_store = store_root / store_name
        store_targets[cohort] = write_store(cohort_store, cohort, generator)
        metadata = pd.read_parquet(cohort_store / "metadata.parquet")
        indices = cohort_record_indices(cohort, metadata)
        bound_cohorts[cohort] = {
            "store_root": str(cohort_store),
            "signals_sha256": sha256_file(cohort_store / "signals.npy"),
            "targets_sha256": sha256_file(cohort_store / "targets.npy"),
            "metadata_sha256": sha256_file(cohort_store / "metadata.parquet"),
            "status_sha256": sha256_file(cohort_store / "status.npy"),
            "records": int(indices.size),
            "record_indices_sha256": array_sha256(indices),
        }
        bound_cohorts[cohort].update((bound_overrides or {}).get(cohort, {}))

    choices_overrides = {
        "label_keys": LABEL_KEYS,
        "headline_label_keys": HEADLINE_KEYS,
        "architectures": {
            architecture: {
                "runs": [],
                "thresholds_by_label": dict.fromkeys(LABEL_KEYS, 0.5),
                "temperature": TEMPERATURES[architecture],
            }
            for architecture in ARCHITECTURES
        },
        "bound_inputs": {"cohorts": bound_cohorts},
    }
    study = build_sealed_study(repository, choices_overrides=choices_overrides)

    for cohort, store_name in COHORT_STORE_NAMES.items():
        cohort_store = store_root / store_name
        metadata = pd.read_parquet(cohort_store / "metadata.parquet")
        indices = cohort_record_indices(cohort, metadata)
        targets = store_targets[cohort][indices]
        arrays: dict[str, np.ndarray] = {
            "targets": targets,
            "record_indices": indices,
        }
        for architecture in ARCHITECTURES:
            noise = generator.uniform(-0.25, 0.25, size=targets.shape)
            probabilities = np.clip(0.5 + 0.56 * (targets - 0.5) + noise, 0.0, 1.0)
            arrays[f"{architecture}_probabilities"] = probabilities.astype(np.float32)
            arrays[f"{architecture}_calibrated_probabilities"] = apply_temperature(
                probabilities, TEMPERATURES[architecture]
            ).astype(np.float32)
        cohort_predictions = prediction_root / cohort
        cohort_predictions.mkdir(parents=True, exist_ok=True)
        ensemble_path = cohort_predictions / "ensembles.npz"
        np.savez_compressed(ensemble_path, **arrays)
        receipt = {
            "schema_version": 1,
            "cohort": cohort,
            "records": int(indices.size),
            "device": "cpu",
            "evaluation_seal_choices_commit": study.choices_commit,
            "status": "complete",
            "completed_seed_passes": [],
            "targets_sha256": array_sha256(targets),
            "record_indices_sha256": array_sha256(indices),
            "ensemble_file": str(ensemble_path),
            "ensemble_sha256": sha256_file(ensemble_path),
        }
        receipt.update((receipt_overrides or {}).get(cohort, {}))
        (cohort_predictions / "receipt.json").write_text(json.dumps(receipt, indent=2))

    manifest_path = tmp_path / "harmonized_labels.json"
    manifest_path.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "label_key": key,
                        "diagnosis": f"diagnosis {key}",
                        "headline_eligible": key in HEADLINE_KEYS,
                    }
                    for key in LABEL_KEYS
                ],
                "headline_label_keys": HEADLINE_KEYS,
                "subgroup_common_label_keys": {
                    cohort: {
                        "age": HEADLINE_KEYS[:6],
                        "sex": HEADLINE_KEYS[:9],
                    }
                    for cohort in COHORT_STORE_NAMES
                },
            }
        )
    )

    return argparse.Namespace(
        prediction_root=prediction_root,
        store_root=store_root,
        label_manifest=manifest_path,
        output_root=tmp_path / "results_out",
        bootstrap_replicates=12,
        bootstrap_batch_size=5,
        root=repository,
    )


@pytest.fixture
def synthetic(tmp_path: Path) -> argparse.Namespace:
    return build_synthetic_study(tmp_path)


def test_analysis_writes_calibration_intervals_and_label_counts(
    synthetic: argparse.Namespace,
) -> None:
    uncertainty = analysis.run_analysis(synthetic, synthetic.root)
    output_root = synthetic.output_root

    cohorts = pd.read_csv(output_root / "cohort_metrics.csv")
    assert len(cohorts) == 6
    assert (cohorts.registered_headline_labels == len(HEADLINE_KEYS)).all()
    assert (cohorts.contributing_labels == len(HEADLINE_KEYS)).all()
    for variant in ("uncalibrated", "calibrated"):
        for metric in ("macro_per_label_ece", "macro_brier", "pooled_ece"):
            point = cohorts[f"{variant}_{metric}"]
            lower = cohorts[f"{variant}_{metric}_lower_95"]
            upper = cohorts[f"{variant}_{metric}_upper_95"]
            assert (lower <= upper).all()
            assert point.between(0, 1).all()

    subgroups = pd.read_csv(output_root / "subgroup_metrics.csv")
    assert set(subgroups.eligible_label_keys.unique()) == {
        ";".join(HEADLINE_KEYS[:6]),
        ";".join(HEADLINE_KEYS[:9]),
    }
    assert subgroups.eligible_label_digest.nunique() == 2
    assert (subgroups.contributing_labels == subgroups.eligible_labels).all()

    # Subgroup calibration carries intervals for the same reason discrimination does.
    # Only the ordering of the bounds is asserted: a percentile bootstrap interval is
    # not guaranteed to contain its own point estimate, and for a tightly paired
    # contrast it occasionally does not.
    for column in (
        "uncalibrated_macro_ece",
        "calibrated_macro_ece",
        "calibrated_minus_uncalibrated_macro_ece",
        "uncalibrated_brier",
        "calibrated_brier",
        "calibrated_minus_uncalibrated_brier",
    ):
        assert subgroups[f"{column}_lower_95"].notna().all()
        assert (subgroups[f"{column}_lower_95"] <= subgroups[f"{column}_upper_95"]).all()

    calibration = uncertainty["calibration"]["ningbo"]["s4d"]
    assert set(calibration) == {"uncalibrated", "calibrated", "calibrated_minus_uncalibrated"}
    for metric, record in calibration["calibrated_minus_uncalibrated"].items():
        assert metric in ("macro_per_label_ece", "macro_brier", "pooled_ece")
        assert record["lower_95"] <= record["upper_95"]
    assert set(uncertainty["calibration_shift_deltas"]) == {"chapman_shaoxing", "ningbo"}
    assert uncertainty["contributing_labels"]["cohort_replicates_below_registered"] == 0
    assert uncertainty["contributing_labels"]["subgroup_replicates_below_eligible"] == 0
    assert uncertainty["interval_type"] == "percentile"

    distributions = np.load(output_root / "bootstrap_distributions.npz")
    assert "ningbo__s4d__calibrated__macro_per_label_ece" in distributions
    assert "ningbo__s4d__calibrated_minus_uncalibrated__pooled_ece" in distributions
    assert "ningbo__s4d__uncalibrated__pooled_ece__shift_delta" in distributions
    assert "ningbo__s4d__shift_delta" in distributions
    assert len(distributions["ningbo__s4d__macro_auroc_labels"]) == 12


def test_calibration_contrast_distribution_is_paired(synthetic: argparse.Namespace) -> None:
    analysis.run_analysis(synthetic, synthetic.root)
    distributions = np.load(synthetic.output_root / "bootstrap_distributions.npz")
    for architecture in ARCHITECTURES:
        np.testing.assert_array_equal(
            distributions[f"ptb_test__{architecture}__calibrated_minus_uncalibrated__macro_brier"],
            distributions[f"ptb_test__{architecture}__calibrated__macro_brier"]
            - distributions[f"ptb_test__{architecture}__uncalibrated__macro_brier"],
        )


@pytest.mark.parametrize(
    ("receipt_overrides", "bound_overrides", "message"),
    [
        ({"ningbo": {"targets_sha256": "0" * 64}}, None, "target hash mismatch"),
        ({"ningbo": {"cohort": "chapman_shaoxing"}}, None, "does not match 'ningbo'"),
        (
            {"ningbo": {"evaluation_seal_choices_commit": "b" * 40}},
            None,
            "evaluation choices commit",
        ),
        ({"ningbo": {"ensemble_sha256": "0" * 64}}, None, "ensemble file hash mismatch"),
        ({"ningbo": {"status": "in_progress"}}, None, "protected inference incomplete"),
        ({"ningbo": {"records": 3}}, None, "predictions but the ensemble"),
        (None, {"ningbo": {"metadata_sha256": "0" * 64}}, "bound metadata_sha256 mismatch"),
        (None, {"ningbo": {"targets_sha256": "0" * 64}}, "bound targets_sha256 mismatch"),
        (None, {"ningbo": {"record_indices_sha256": "0" * 64}}, "bound record-index digest"),
        (None, {"ningbo": {"records": 5}}, "bound inputs expect"),
    ],
)
def test_provenance_failures_refuse_to_analyze(
    tmp_path: Path,
    receipt_overrides: dict | None,
    bound_overrides: dict | None,
    message: str,
) -> None:
    args = build_synthetic_study(
        tmp_path, receipt_overrides=receipt_overrides, bound_overrides=bound_overrides
    )
    with pytest.raises(RuntimeError, match=message):
        analysis.run_analysis(args, args.root)


def test_missing_bound_inputs_refuses_to_analyze(tmp_path: Path) -> None:
    args = build_synthetic_study(tmp_path)
    choices_path = args.root / "results/evaluation_choices.json"
    choices = json.loads(choices_path.read_text())
    del choices["bound_inputs"]
    choices_path.write_text(json.dumps(choices, indent=2, sort_keys=True) + "\n")
    seal_path = args.root / "results/EVALUATION_SEAL.json"
    seal = json.loads(seal_path.read_text())
    seal["choices_sha256"] = sha256_file(choices_path)
    seal_path.write_text(json.dumps(seal, indent=2))

    # The seal itself now fails first, which is the stronger guarantee; the
    # bound-input check is exercised directly against the parsed choices.
    with pytest.raises(RuntimeError):
        analysis.run_analysis(args, args.root)
    with pytest.raises(RuntimeError, match="no bound_inputs cohorts section"):
        analysis.verify_bound_cohort_inputs(
            cohort="ningbo",
            choices=choices,
            store_root=args.store_root,
            store=None,
            targets=np.zeros((1, 16), dtype=np.uint8),
            record_indices=np.zeros(1, dtype=np.int64),
        )


def test_out_of_range_probabilities_are_refused() -> None:
    with pytest.raises(RuntimeError, match=r"outside \[0, 1\]"):
        analysis.verify_probability_array(
            np.asarray([[0.5, 1.4]]),
            cohort="ningbo",
            name="s4d_probabilities",
            expected_shape=(1, 2),
        )
    with pytest.raises(RuntimeError, match="non-finite"):
        analysis.verify_probability_array(
            np.asarray([[0.5, np.nan]]),
            cohort="ningbo",
            name="s4d_probabilities",
            expected_shape=(1, 2),
        )
    with pytest.raises(RuntimeError, match="expected"):
        analysis.verify_probability_array(
            np.asarray([[0.5, 0.4]]),
            cohort="ningbo",
            name="s4d_probabilities",
            expected_shape=(2, 2),
        )
