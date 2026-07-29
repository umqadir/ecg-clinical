#!/usr/bin/env python3
"""Generate all preregistered metrics, intervals, and subgroup tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_clinical.bootstrap import (
    CALIBRATION_METRICS,
    paired_bootstrap_cohort_statistics,
    percentile_interval,
)
from ecg_clinical.data import WaveformStore
from ecg_clinical.integrity import (
    sha256_file,
    verify_evaluation_seal,
    verify_preregistration_seal,
)
from ecg_clinical.metrics import (
    calibration_summary,
    discrimination_summary,
    expected_calibration_error,
    per_label_metrics,
    threshold_metrics,
)

COHORTS = ("ptb_test", "chapman_shaoxing", "ningbo")
EXTERNAL_COHORTS = ("chapman_shaoxing", "ningbo")
ARCHITECTURES = ("xresnet1d101", "s4d")
COHORT_STORE_NAMES = {
    "ptb_test": "ptb_xl",
    "chapman_shaoxing": "chapman_shaoxing",
    "ningbo": "ningbo",
}
COHORT_SEEDS = {"ptb_test": 20260716, "chapman_shaoxing": 20260717, "ningbo": 20260718}
CALIBRATION_VARIANTS = ("uncalibrated", "calibrated")
CONTRAST_VARIANT = "calibrated_minus_uncalibrated"
MODEL_CONTRAST = f"{ARCHITECTURES[1]}_minus_{ARCHITECTURES[0]}"
# The small store files are re-hashed here; signals.npy is deliberately not,
# because protected inference already verified it against the same digests and
# re-reading 1.5 GB of protected waveforms adds no evidence at analysis time.
BOUND_STORE_FILES = (
    ("metadata.parquet", "metadata_sha256"),
    ("targets.npy", "targets_sha256"),
    ("status.npy", "status_sha256"),
)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def label_set_digest(label_keys: list[str]) -> str:
    """A short, order-independent fingerprint of an eligible label set."""

    joined = ";".join(sorted(label_keys))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root", type=Path, default=Path("artifacts/predictions/protected")
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/cache/waveforms"))
    parser.add_argument(
        "--label-manifest",
        type=Path,
        default=Path("data/derived/preregistration/harmonized_labels.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=100)
    return parser.parse_args()


def sealed_checkpoint_digests(choices: dict) -> dict[str, str]:
    """The checkpoint digests the frozen choices authorize, keyed by run id."""

    digests: dict[str, str] = {}
    for architecture, architecture_choices in choices.get("architectures", {}).items():
        for run in architecture_choices.get("runs", []):
            digests[f"{architecture}-seed-{int(run['seed'])}"] = str(run["checkpoint_sha256"])
    return digests


def verify_receipt_provenance(
    *,
    cohort: str,
    receipt: dict,
    evaluation_seal: dict,
    choices: dict,
    ensemble_path: Path,
    targets: np.ndarray,
    record_indices: np.ndarray,
) -> None:
    """Bind the loaded predictions to the receipt written by protected inference.

    The receipt already carried these digests; verifying only the targets left
    the ensemble file, the record order, the cohort identity, and the sealed
    choices commit unchecked. Fields whose presence depends on the receipt
    schema version are checked only when the receipt carries them, so an older
    receipt is read rather than rejected on a schema technicality.
    """

    if receipt.get("status") != "complete":
        raise RuntimeError(f"protected inference incomplete for {cohort}")
    if receipt.get("cohort") != cohort:
        raise RuntimeError(f"receipt cohort {receipt.get('cohort')!r} does not match {cohort!r}")
    sealed_commit = evaluation_seal.get("choices_commit")
    if receipt.get("evaluation_seal_choices_commit") != sealed_commit:
        raise RuntimeError(
            f"receipt for {cohort} names evaluation choices commit "
            f"{receipt.get('evaluation_seal_choices_commit')!r}, but the verified seal names "
            f"{sealed_commit!r}"
        )
    if array_sha256(targets) != receipt.get("targets_sha256"):
        raise RuntimeError(f"target hash mismatch for {cohort}")
    if array_sha256(record_indices) != receipt.get("record_indices_sha256"):
        raise RuntimeError(f"record-index hash mismatch for {cohort}")
    if sha256_file(ensemble_path) != receipt.get("ensemble_sha256"):
        raise RuntimeError(f"ensemble file hash mismatch for {cohort}")
    recorded_records = receipt.get("records")
    if recorded_records is not None and int(recorded_records) != len(record_indices):
        raise RuntimeError(
            f"receipt for {cohort} records {recorded_records} predictions but the ensemble "
            f"holds {len(record_indices)}"
        )
    recorded_checkpoints = receipt.get("sealed_checkpoint_sha256")
    if recorded_checkpoints is not None and recorded_checkpoints != sealed_checkpoint_digests(
        choices
    ):
        raise RuntimeError(
            f"receipt for {cohort} names checkpoint digests the sealed evaluation choices "
            "do not authorize"
        )


def verify_bound_cohort_inputs(
    *,
    cohort: str,
    choices: dict,
    store_root: Path,
    store: WaveformStore,
    targets: np.ndarray,
    record_indices: np.ndarray,
) -> dict:
    """Check the cohort store and record set against the sealed bound inputs."""

    bound = choices.get("bound_inputs")
    if not isinstance(bound, dict) or not isinstance(bound.get("cohorts"), dict):
        raise RuntimeError("sealed evaluation choices carry no bound_inputs cohorts section")
    entry = bound["cohorts"].get(cohort)
    if not isinstance(entry, dict):
        raise RuntimeError(f"sealed evaluation choices bind no inputs for cohort {cohort}")

    cohort_store = store_root / COHORT_STORE_NAMES[cohort]
    for filename, digest_key in BOUND_STORE_FILES:
        expected = entry.get(digest_key)
        if not isinstance(expected, str):
            raise RuntimeError(f"bound inputs for {cohort} carry no {digest_key}")
        if sha256_file(cohort_store / filename) != expected:
            raise RuntimeError(f"bound {digest_key} mismatch for {cohort}: {filename}")
    if int(entry.get("records", -1)) != len(record_indices):
        raise RuntimeError(
            f"bound inputs expect {entry.get('records')} records for {cohort} but the "
            f"ensemble holds {len(record_indices)}"
        )
    if array_sha256(record_indices) != entry.get("record_indices_sha256"):
        raise RuntimeError(f"bound record-index digest mismatch for {cohort}")
    if not np.array_equal(np.asarray(store.targets[record_indices]), targets):
        raise RuntimeError(f"protected targets do not match the bound store for {cohort}")
    return entry


def verify_probability_array(
    array: np.ndarray, *, cohort: str, name: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    """Refuse malformed, non-finite, or out-of-range probabilities."""

    values = np.asarray(array, dtype=np.float64)
    if values.shape != expected_shape:
        raise RuntimeError(
            f"{name} for {cohort} has shape {values.shape}, expected {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} for {cohort} contains non-finite probabilities")
    if values.min() < 0.0 or values.max() > 1.0:
        raise RuntimeError(f"{name} for {cohort} contains probabilities outside [0, 1]")
    return values


def load_verified_cohort(
    *,
    cohort: str,
    prediction_root: Path,
    store_root: Path,
    evaluation_seal: dict,
    choices: dict,
    label_count: int,
) -> dict[str, object]:
    """Load one cohort's sealed predictions after every provenance check passes."""

    cohort_root = prediction_root / cohort
    receipt = json.loads((cohort_root / "receipt.json").read_text())
    ensemble_path = cohort_root / "ensembles.npz"
    prediction = np.load(ensemble_path)
    targets = np.asarray(prediction["targets"], dtype=np.uint8)
    record_indices = np.asarray(prediction["record_indices"], dtype=np.int64)
    verify_receipt_provenance(
        cohort=cohort,
        receipt=receipt,
        evaluation_seal=evaluation_seal,
        choices=choices,
        ensemble_path=ensemble_path,
        targets=targets,
        record_indices=record_indices,
    )
    if targets.ndim != 2 or targets.shape[1] != label_count:
        raise RuntimeError(
            f"targets for {cohort} have shape {targets.shape}, expected (records, {label_count})"
        )
    store = WaveformStore(store_root / COHORT_STORE_NAMES[cohort])
    verify_bound_cohort_inputs(
        cohort=cohort,
        choices=choices,
        store_root=store_root,
        store=store,
        targets=targets,
        record_indices=record_indices,
    )
    probabilities = {
        architecture: verify_probability_array(
            prediction[f"{architecture}_probabilities"],
            cohort=cohort,
            name=f"{architecture}_probabilities",
            expected_shape=targets.shape,
        )
        for architecture in ARCHITECTURES
    }
    calibrated = {
        architecture: verify_probability_array(
            prediction[f"{architecture}_calibrated_probabilities"],
            cohort=cohort,
            name=f"{architecture}_calibrated_probabilities",
            expected_shape=targets.shape,
        )
        for architecture in ARCHITECTURES
    }
    metadata = store.metadata.iloc[record_indices].reset_index(drop=True)
    return {
        "targets": targets,
        "metadata": metadata,
        "probabilities": probabilities,
        "calibrated": calibrated,
    }


def reliability_rows(
    cohort: str,
    architecture: str,
    calibration: str,
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_indices: np.ndarray,
    label_keys: list[str],
    bins: int = 15,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources = [
        (
            "pooled",
            targets[:, label_indices].ravel(),
            probabilities[:, label_indices].ravel(),
        )
    ]
    sources.extend(
        (label_keys[index], targets[:, index], probabilities[:, index]) for index in label_indices
    )
    edges = np.linspace(0, 1, bins + 1)
    for label_key, truth, scores in sources:
        assignments = np.minimum(np.digitize(scores, edges[1:-1]), bins - 1)
        for bin_index in range(bins):
            selected = assignments == bin_index
            rows.append(
                {
                    "cohort": cohort,
                    "architecture": architecture,
                    "calibration": calibration,
                    "label_key": label_key,
                    "bin": bin_index + 1,
                    "lower": edges[bin_index],
                    "upper": edges[bin_index + 1],
                    "count": int(selected.sum()),
                    "mean_probability": (
                        float(scores[selected].mean()) if selected.any() else np.nan
                    ),
                    "observed_frequency": (
                        float(truth[selected].mean()) if selected.any() else np.nan
                    ),
                }
            )
    return rows


def normalized_sex(metadata: pd.DataFrame, cohort: str) -> pd.Series:
    if cohort == "ptb_test":
        numeric = pd.to_numeric(metadata.sex, errors="coerce")
        return numeric.map({0: "male", 1: "female"}).fillna("unknown")
    text = metadata.sex.astype(str).str.strip().str.lower()
    return text.map({"male": "male", "m": "male", "female": "female", "f": "female"}).fillna(
        "unknown"
    )


def normalized_age(metadata: pd.DataFrame) -> pd.Series:
    age = pd.to_numeric(metadata.age, errors="coerce")
    return pd.Series(
        np.select(
            [age < 40, (age >= 40) & (age < 65), age >= 65],
            ["under_40", "40_to_64", "65_plus"],
            default="unknown",
        ),
        index=metadata.index,
    )


def confidence_record(point: float, distribution: np.ndarray) -> dict[str, float]:
    lower, upper = percentile_interval(distribution)
    return {"point": point, "lower_95": lower, "upper_95": upper}


def subgroup_interval(
    distributions: dict[str, np.ndarray], key: str, column: str
) -> dict[str, float]:
    """Emit the percentile interval for one subgroup statistic as two named columns.

    Subgroup calibration carries intervals for the same reason discrimination does:
    the strata are small, the registered subgroup analysis is descriptive, and a bare
    point estimate invites a reader to over-read sampling noise as heterogeneity.
    """

    lower, upper = percentile_interval(distributions[key])
    return {f"{column}_lower_95": lower, f"{column}_upper_95": upper}


def calibration_point_estimates(
    targets: np.ndarray,
    probabilities: np.ndarray,
    calibrated: np.ndarray,
    label_indices: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Point estimates for every registered calibration quantity and its contrast."""

    summaries = {
        "uncalibrated": calibration_summary(targets, probabilities, label_indices),
        "calibrated": calibration_summary(targets, calibrated, label_indices),
    }
    points = {
        variant: {metric: float(summary[metric]) for metric in CALIBRATION_METRICS}
        for variant, summary in summaries.items()
    }
    points[CONTRAST_VARIANT] = {
        metric: points["calibrated"][metric] - points["uncalibrated"][metric]
        for metric in CALIBRATION_METRICS
    }
    return points


def run_analysis(args: argparse.Namespace, root: Path) -> dict[str, object]:
    registration = verify_preregistration_seal(root)
    evaluation_seal, choices = verify_evaluation_seal(root)
    label_manifest = json.loads(args.label_manifest.read_text())
    labels = label_manifest["labels"]
    label_keys = [label["label_key"] for label in labels]
    label_names = [label["diagnosis"] for label in labels]
    headline_indices = np.asarray(
        [label_keys.index(key) for key in label_manifest["headline_label_keys"]], dtype=np.int64
    )
    headline_count = len(headline_indices)
    args.output_root.mkdir(parents=True, exist_ok=True)

    cohort_rows: list[dict[str, object]] = []
    per_label_rows: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    point_macro_auc: dict[str, dict[str, float]] = {}
    point_calibration: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    bootstrap: dict[str, dict[str, np.ndarray]] = {}
    bootstrap_artifacts: dict[str, np.ndarray] = {}
    contributing_cohorts: dict[str, dict[str, dict[str, int]]] = {}
    contributing_subgroups: list[dict[str, object]] = []
    loaded: dict[str, dict[str, object]] = {}

    for cohort in COHORTS:
        cohort_data = load_verified_cohort(
            cohort=cohort,
            prediction_root=args.prediction_root,
            store_root=args.store_root,
            evaluation_seal=evaluation_seal,
            choices=choices,
            label_count=len(label_keys),
        )
        targets = cohort_data["targets"]
        metadata = cohort_data["metadata"]
        probabilities_by_model = cohort_data["probabilities"]
        calibrated_by_model = cohort_data["calibrated"]
        if cohort == "ptb_test":
            _, group_inverse = np.unique(metadata.patient_id.astype(str), return_inverse=True)
        else:
            group_inverse = np.arange(len(metadata), dtype=np.int64)
        cohort_bootstrap = paired_bootstrap_cohort_statistics(
            targets,
            probabilities_by_model,
            headline_indices,
            group_inverse,
            replicates=args.bootstrap_replicates,
            seed=COHORT_SEEDS[cohort],
            batch_size=args.bootstrap_batch_size,
            calibrated_by_model=calibrated_by_model,
        )
        bootstrap[cohort] = cohort_bootstrap
        for name, values in cohort_bootstrap.items():
            bootstrap_artifacts[f"{cohort}__{name}"] = values
        point_macro_auc[cohort] = {}
        point_calibration[cohort] = {}
        contributing_cohorts[cohort] = {}
        for architecture in ARCHITECTURES:
            probabilities = probabilities_by_model[architecture]
            calibrated = calibrated_by_model[architecture]
            discrimination = discrimination_summary(targets, probabilities, headline_indices)
            calibration_points = calibration_point_estimates(
                targets, probabilities, calibrated, headline_indices
            )
            thresholds = np.asarray(
                [
                    choices["architectures"][architecture]["thresholds_by_label"][key]
                    for key in label_keys
                ]
            )
            thresholded = threshold_metrics(targets, probabilities, thresholds, headline_indices)
            point_macro_auc[cohort][architecture] = discrimination["macro_auroc"]
            point_calibration[cohort][architecture] = calibration_points
            per_label = per_label_metrics(targets, probabilities)
            contributing = int(np.isfinite(per_label["auroc"][headline_indices]).sum())
            replicate_counts = cohort_bootstrap[f"{architecture}__macro_auroc_labels"]
            contributing_cohorts[cohort][architecture] = {
                "point": contributing,
                "replicates_below_registered": int((replicate_counts < headline_count).sum()),
                "minimum_replicate_count": int(replicate_counts.min()),
            }
            lower, upper = percentile_interval(cohort_bootstrap[architecture])
            calibration_columns: dict[str, object] = {}
            for variant in CALIBRATION_VARIANTS:
                for metric in CALIBRATION_METRICS:
                    calibration_columns[f"{variant}_{metric}"] = calibration_points[variant][metric]
                    interval = percentile_interval(
                        cohort_bootstrap[f"{architecture}__{variant}__{metric}"]
                    )
                    calibration_columns[f"{variant}_{metric}_lower_95"] = interval[0]
                    calibration_columns[f"{variant}_{metric}_upper_95"] = interval[1]
            cohort_rows.append(
                {
                    "cohort": cohort,
                    "architecture": architecture,
                    **discrimination,
                    "macro_auroc_lower_95": lower,
                    "macro_auroc_upper_95": upper,
                    "registered_headline_labels": headline_count,
                    "contributing_labels": contributing,
                    **calibration_columns,
                    **thresholded,
                }
            )
            for label_index, (label_key, label_name) in enumerate(
                zip(label_keys, label_names, strict=True)
            ):
                per_label_rows.append(
                    {
                        "cohort": cohort,
                        "architecture": architecture,
                        "label_key": label_key,
                        "diagnosis": label_name,
                        "headline": label_index in headline_indices,
                        "positives": int(targets[:, label_index].sum()),
                        "negatives": int(len(targets) - targets[:, label_index].sum()),
                        "auroc": per_label["auroc"][label_index],
                        "auprc": per_label["auprc"][label_index],
                        "uncalibrated_ece": expected_calibration_error(
                            targets[:, label_index], probabilities[:, label_index]
                        ),
                        "calibrated_ece": expected_calibration_error(
                            targets[:, label_index], calibrated[:, label_index]
                        ),
                        "uncalibrated_brier": float(
                            np.mean(
                                np.square(probabilities[:, label_index] - targets[:, label_index])
                            )
                        ),
                        "calibrated_brier": float(
                            np.mean(np.square(calibrated[:, label_index] - targets[:, label_index]))
                        ),
                    }
                )
            reliability.extend(
                reliability_rows(
                    cohort,
                    architecture,
                    "uncalibrated",
                    targets,
                    probabilities,
                    headline_indices,
                    label_keys,
                )
            )
            reliability.extend(
                reliability_rows(
                    cohort,
                    architecture,
                    "calibrated",
                    targets,
                    calibrated,
                    headline_indices,
                    label_keys,
                )
            )
        loaded[cohort] = cohort_data

    uncertainty: dict[str, object] = {
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed_base": 20260716,
        "interval_type": "percentile",
        "cohort_macro_auroc": {},
        "paired_model_contrasts": {},
        "shift_deltas": {},
        "calibration": {},
        "calibration_model_contrasts": {},
        "calibration_shift_deltas": {},
    }
    for cohort in COHORTS:
        uncertainty["cohort_macro_auroc"][cohort] = {
            architecture: confidence_record(
                point_macro_auc[cohort][architecture], bootstrap[cohort][architecture]
            )
            for architecture in ARCHITECTURES
        }
        point_contrast = point_macro_auc[cohort]["s4d"] - point_macro_auc[cohort]["xresnet1d101"]
        uncertainty["paired_model_contrasts"][cohort] = confidence_record(
            point_contrast, bootstrap[cohort][MODEL_CONTRAST]
        )
        uncertainty["calibration"][cohort] = {
            architecture: {
                variant: {
                    metric: confidence_record(
                        point_calibration[cohort][architecture][variant][metric],
                        bootstrap[cohort][f"{architecture}__{variant}__{metric}"],
                    )
                    for metric in CALIBRATION_METRICS
                }
                for variant in (*CALIBRATION_VARIANTS, CONTRAST_VARIANT)
            }
            for architecture in ARCHITECTURES
        }
        uncertainty["calibration_model_contrasts"][cohort] = {
            variant: {
                metric: confidence_record(
                    point_calibration[cohort]["s4d"][variant][metric]
                    - point_calibration[cohort]["xresnet1d101"][variant][metric],
                    bootstrap[cohort][f"{MODEL_CONTRAST}__{variant}__{metric}"],
                )
                for metric in CALIBRATION_METRICS
            }
            for variant in CALIBRATION_VARIANTS
        }
    for external in EXTERNAL_COHORTS:
        uncertainty["shift_deltas"][external] = {}
        uncertainty["calibration_shift_deltas"][external] = {}
        for architecture in ARCHITECTURES:
            point = (
                point_macro_auc[external][architecture] - point_macro_auc["ptb_test"][architecture]
            )
            distribution = bootstrap[external][architecture] - bootstrap["ptb_test"][architecture]
            bootstrap_artifacts[f"{external}__{architecture}__shift_delta"] = distribution
            uncertainty["shift_deltas"][external][architecture] = confidence_record(
                point, distribution
            )
            calibration_shift: dict[str, dict[str, dict[str, float]]] = {}
            for variant in (*CALIBRATION_VARIANTS, CONTRAST_VARIANT):
                calibration_shift[variant] = {}
                for metric in CALIBRATION_METRICS:
                    name = f"{architecture}__{variant}__{metric}"
                    delta = bootstrap[external][name] - bootstrap["ptb_test"][name]
                    bootstrap_artifacts[f"{external}__{name}__shift_delta"] = delta
                    calibration_shift[variant][metric] = confidence_record(
                        point_calibration[external][architecture][variant][metric]
                        - point_calibration["ptb_test"][architecture][variant][metric],
                        delta,
                    )
            uncertainty["calibration_shift_deltas"][external][architecture] = calibration_shift
    uncertainty["hospital_shift_difference_ningbo_minus_chapman"] = {}
    for architecture in ARCHITECTURES:
        point = (
            point_macro_auc["ningbo"][architecture]
            - point_macro_auc["chapman_shaoxing"][architecture]
        )
        distribution = (
            bootstrap["ningbo"][architecture] - bootstrap["chapman_shaoxing"][architecture]
        )
        uncertainty["hospital_shift_difference_ningbo_minus_chapman"][architecture] = (
            confidence_record(point, distribution)
        )

    for cohort in COHORTS:
        targets = loaded[cohort]["targets"]
        metadata = loaded[cohort]["metadata"]
        stratifications = {
            "age": (normalized_age(metadata), ("under_40", "40_to_64", "65_plus")),
            "sex": (normalized_sex(metadata, cohort), ("female", "male")),
        }
        manifest_cohort = "ptb_test" if cohort == "ptb_test" else cohort
        for stratification, (assignments, levels) in stratifications.items():
            eligible_keys = label_manifest["subgroup_common_label_keys"][manifest_cohort][
                stratification
            ]
            eligible_indices = np.asarray([label_keys.index(key) for key in eligible_keys])
            eligible_digest = label_set_digest(eligible_keys)
            for level in levels:
                selected = assignments.to_numpy() == level
                subgroup_targets = targets[selected]
                subgroup_metadata = metadata[selected]
                if cohort == "ptb_test":
                    _, subgroup_groups = np.unique(
                        subgroup_metadata.patient_id.astype(str), return_inverse=True
                    )
                else:
                    subgroup_groups = np.arange(selected.sum(), dtype=np.int64)
                subgroup_probability_models = {
                    architecture: loaded[cohort]["probabilities"][architecture][selected]
                    for architecture in ARCHITECTURES
                }
                subgroup_calibrated_models = {
                    architecture: loaded[cohort]["calibrated"][architecture][selected]
                    for architecture in ARCHITECTURES
                }
                subgroup_bootstrap = paired_bootstrap_cohort_statistics(
                    subgroup_targets,
                    subgroup_probability_models,
                    eligible_indices,
                    subgroup_groups,
                    replicates=args.bootstrap_replicates,
                    seed=COHORT_SEEDS[cohort]
                    + (100 if stratification == "age" else 200)
                    + levels.index(level),
                    batch_size=args.bootstrap_batch_size,
                    calibrated_by_model=subgroup_calibrated_models,
                    include_calibration=True,
                )
                for architecture in ARCHITECTURES:
                    probability = subgroup_probability_models[architecture]
                    calibrated = loaded[cohort]["calibrated"][architecture][selected]
                    discrimination = discrimination_summary(
                        subgroup_targets, probability, eligible_indices
                    )
                    raw_calibration = calibration_summary(
                        subgroup_targets, probability, eligible_indices
                    )
                    calibrated_summary = calibration_summary(
                        subgroup_targets, calibrated, eligible_indices
                    )
                    subgroup_per_label = per_label_metrics(subgroup_targets, probability)
                    contributing = int(
                        np.isfinite(subgroup_per_label["auroc"][eligible_indices]).sum()
                    )
                    replicate_counts = subgroup_bootstrap[f"{architecture}__macro_auroc_labels"]
                    contributing_subgroups.append(
                        {
                            "cohort": cohort,
                            "stratification": stratification,
                            "subgroup": level,
                            "architecture": architecture,
                            "eligible_labels": len(eligible_indices),
                            "eligible_label_digest": eligible_digest,
                            "point": contributing,
                            "replicates_below_eligible": int(
                                (replicate_counts < len(eligible_indices)).sum()
                            ),
                            "minimum_replicate_count": int(replicate_counts.min()),
                        }
                    )
                    lower, upper = percentile_interval(subgroup_bootstrap[architecture])
                    subgroup_rows.append(
                        {
                            "cohort": cohort,
                            "stratification": stratification,
                            "subgroup": level,
                            "architecture": architecture,
                            "records": int(selected.sum()),
                            "eligible_labels": len(eligible_indices),
                            "eligible_label_keys": ";".join(eligible_keys),
                            "eligible_label_digest": eligible_digest,
                            "contributing_labels": contributing,
                            "macro_auroc": discrimination["macro_auroc"],
                            "macro_auroc_lower_95": lower,
                            "macro_auroc_upper_95": upper,
                            "macro_auprc": discrimination["macro_auprc"],
                            "uncalibrated_macro_ece": raw_calibration["macro_per_label_ece"],
                            **subgroup_interval(
                                subgroup_bootstrap,
                                f"{architecture}__uncalibrated__macro_per_label_ece",
                                "uncalibrated_macro_ece",
                            ),
                            "calibrated_macro_ece": calibrated_summary["macro_per_label_ece"],
                            **subgroup_interval(
                                subgroup_bootstrap,
                                f"{architecture}__calibrated__macro_per_label_ece",
                                "calibrated_macro_ece",
                            ),
                            "calibrated_minus_uncalibrated_macro_ece": (
                                calibrated_summary["macro_per_label_ece"]
                                - raw_calibration["macro_per_label_ece"]
                            ),
                            **subgroup_interval(
                                subgroup_bootstrap,
                                f"{architecture}__calibrated_minus_uncalibrated"
                                "__macro_per_label_ece",
                                "calibrated_minus_uncalibrated_macro_ece",
                            ),
                            "uncalibrated_brier": raw_calibration["macro_brier"],
                            **subgroup_interval(
                                subgroup_bootstrap,
                                f"{architecture}__uncalibrated__macro_brier",
                                "uncalibrated_brier",
                            ),
                            "calibrated_brier": calibrated_summary["macro_brier"],
                            **subgroup_interval(
                                subgroup_bootstrap,
                                f"{architecture}__calibrated__macro_brier",
                                "calibrated_brier",
                            ),
                            "calibrated_minus_uncalibrated_brier": (
                                calibrated_summary["macro_brier"] - raw_calibration["macro_brier"]
                            ),
                            **subgroup_interval(
                                subgroup_bootstrap,
                                f"{architecture}__calibrated_minus_uncalibrated__macro_brier",
                                "calibrated_minus_uncalibrated_brier",
                            ),
                            "ptb_sex_encoding_correction": (
                                cohort == "ptb_test" and stratification == "sex"
                            ),
                        }
                    )

    uncertainty["contributing_labels"] = {
        "registered_headline_labels": headline_count,
        "cohorts": contributing_cohorts,
        "subgroups": contributing_subgroups,
        "cohort_replicates_below_registered": int(
            sum(
                entry["replicates_below_registered"]
                for cohort_entry in contributing_cohorts.values()
                for entry in cohort_entry.values()
            )
        ),
        "subgroup_replicates_below_eligible": int(
            sum(entry["replicates_below_eligible"] for entry in contributing_subgroups)
        ),
    }

    pd.DataFrame(cohort_rows).to_csv(args.output_root / "cohort_metrics.csv", index=False)
    pd.DataFrame(per_label_rows).to_csv(args.output_root / "per_label_metrics.csv", index=False)
    pd.DataFrame(reliability).to_csv(args.output_root / "reliability_bins.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(args.output_root / "subgroup_metrics.csv", index=False)
    temporary = args.output_root / "uncertainty.json.tmp"
    temporary.write_text(json.dumps(uncertainty, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output_root / "uncertainty.json")
    np.savez_compressed(args.output_root / "bootstrap_distributions.npz", **bootstrap_artifacts)
    analysis_manifest = {
        "schema_version": 2,
        "registration_commit": registration["registration_commit"],
        "evaluation_choices_commit": evaluation_seal["choices_commit"],
        "cohorts": list(COHORTS),
        "architectures": list(ARCHITECTURES),
        "headline_labels": headline_count,
        "all_targets": len(label_keys),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_statistics": "discrimination and calibration share one set of resamples",
        "calibration_bins": 15,
        "interval_type": "percentile",
        "ptb_cluster_unit": "patient_id",
        "external_cluster_unit": "record",
        "ptb_sex_encoding_deviation": "deviations/001_ptb_sex_encoding.md",
    }
    (args.output_root / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n"
    )
    return uncertainty


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    uncertainty = run_analysis(args, root)
    print(json.dumps(uncertainty, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
