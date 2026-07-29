#!/usr/bin/env python3
"""Decision curves and operating points for the sealed cohort predictions.

POST-SEAL EXPLORATORY ADDITION, 2026-07-26. This stage was not part of the frozen
analysis. It was written after the evaluation seal was broken and after the registered
results existed, and it is kept separate from `analyze_results.py` for that reason.

What it adds and why. The registered analysis reports discrimination, calibration, and
threshold metrics at one sealed operating point per label. None of those quantities says
what a reader has to know before deploying a flag: how many alerts a threshold raises per
true positive found, and whether acting on the model beats the two strategies that need no
model at all. Decision curve analysis answers the second question directly. Net benefit at
threshold probability t is TP/n minus (FP/n) times t/(1-t), which prices a false positive
against a true positive at the exchange rate a clinician who would act at t implies. The
comparators are treat everyone, whose net benefit is prevalence minus (1-prevalence) times
t/(1-t), and treat no one, whose net benefit is zero by construction. A model that fails to
clear both over a plausible threshold range carries no decision value at that range however
its AUROC reads, so this analysis can only qualify a reading of the registered results,
never strengthen one.

What it does not do. It changes no registered quantity, rewrites no sealed artifact, and
re-runs no inference. It recomputes nothing that `analyze_results.py` reports; it writes to
its own files.

Probability scale. Net benefit reads the score as a probability, because t/(1-t) is only
the right exchange rate when the score is calibrated. The primary curves therefore use the
temperature-calibrated probabilities, with the temperature fit on fold 9 and sealed before
any test cohort was touched. The uncalibrated curve travels beside it in the same row so
the effect of the calibration step on decision value is readable rather than assumed.

Operating points. Two threshold sets are reported, both derived from fold 9 alone. The
first is the sealed per-label F1 threshold set, applied to uncalibrated probabilities
exactly as the registered analysis applies it. The second is exploratory: per label, the
largest fold-9 uncalibrated probability whose rule attains fold-9 sensitivity of 0.90. It
exists because a screening deployment fixes sensitivity first and pays for it in alerts,
which is a different question than the F1 threshold answers. Neither set is derived from a
test or external cohort.

Faithfulness to the registered design. The resamples are not a new bootstrap. This stage
rebuilds the registered cluster resamples from the same per-cohort seeds, the same cluster
units (patient for PTB-XL, record externally), the same replicate count and the same batch
size, then proves it: macro-averaging the per-label replicate AUROCs over the headline
labels must reproduce the registered macro-AUROC distribution in
`results/bootstrap_distributions.npz` bit-for-bit. The run aborts if that check fails, so
an interval published here cannot come from a different resample than the registered
estimate it sits beside. Every count inside a replicate is a weighted count taken against
the multinomial cluster draw, so the treat-all comparator moves with the resampled
prevalence and the net benefit gain is paired within the replicate.

Multiplicity. These intervals are marginal and uncorrected, matching the registered plan,
which states that no multiplicity-adjusted tests were to be run. Thirteen labels by three
cohorts by two architectures is a large simultaneous family, so the intervals should be
read as descriptive rather than as a family of tests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from ecg_clinical.bootstrap import (
    bootstrap_record_weight_batches,
    percentile_interval,
    weighted_roc_auc_batch,
)
from ecg_clinical.integrity import verify_evaluation_seal, verify_preregistration_seal

EXPLORATORY_STAGE_DATE = "2026-07-26"
THRESHOLD_GRID = np.round(np.arange(1, 51, dtype=np.float64) / 100.0, 2)
INTERVAL_THRESHOLDS = (0.05, 0.10, 0.20)
SENSITIVITY_TARGET = 0.90
REGISTERED_THRESHOLD_SET = "registered_f1"
SENSITIVITY_THRESHOLD_SET = "validation_sensitivity_90"


def load_analysis_module() -> ModuleType:
    """Import the registered analysis stage to reuse its verified cohort loader.

    The provenance checks that guard every sealed prediction live there. Reusing them
    keeps one implementation of receipt, digest, and bound-input verification rather
    than a second copy that could drift from it.
    """

    path = Path(__file__).resolve().parent / "analyze_results.py"
    spec = importlib.util.spec_from_file_location("analyze_results", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the analysis stage from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root", type=Path, default=Path("artifacts/predictions/protected")
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path("artifacts/predictions/validation_ensembles.npz"),
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/cache/waveforms"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/derived/preregistration/harmonized_labels.json"),
    )
    parser.add_argument("--registered-bootstrap", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=100)
    return parser.parse_args()


def cluster_inverse(cohort: str, metadata: pd.DataFrame) -> np.ndarray:
    """Rebuild the registered resampling unit for a cohort."""

    if cohort == "ptb_test":
        _, group_inverse = np.unique(metadata.patient_id.astype(str), return_inverse=True)
        return group_inverse
    return np.arange(len(metadata), dtype=np.int64)


def macro_of_per_label(distribution: np.ndarray, label_indices: np.ndarray) -> np.ndarray:
    """Macro-average the per-label replicates over defined labels only.

    This is the registered macro rule, reproduced here purely so the result can be
    compared against the registered macro distribution as a proof of identical resamples.

    The copy is forced contiguous deliberately. Selecting columns with an integer array
    yields a non-contiguous result, and NumPy's pairwise summation blocks a non-contiguous
    reduction differently, which perturbs the last bit on a minority of replicates even
    when every input value is identical. Comparing against a bit-exact reference therefore
    requires matching the registered code's memory layout as well as its arithmetic.
    """

    selected = np.ascontiguousarray(distribution[:, label_indices])
    contributing = np.isfinite(selected).sum(axis=1)
    return np.divide(
        np.nansum(selected, axis=1),
        contributing,
        out=np.full(len(selected), np.nan),
        where=contributing > 0,
    )


def threshold_odds(threshold: np.ndarray | float) -> np.ndarray | float:
    """The exchange rate t/(1-t) a decision maker acting at threshold t implies."""

    return threshold / (1.0 - threshold)


def net_benefit(
    true_positives: np.ndarray | float,
    false_positives: np.ndarray | float,
    records: np.ndarray | float,
    threshold: np.ndarray | float,
) -> np.ndarray | float:
    """Net benefit of treating every record whose predicted probability is at least t."""

    return true_positives / records - (false_positives / records) * threshold_odds(threshold)


def net_benefit_treat_all(
    prevalence: np.ndarray | float, threshold: np.ndarray | float
) -> np.ndarray | float:
    """Net benefit of treating every record, the first of the two default strategies."""

    return prevalence - (1.0 - prevalence) * threshold_odds(threshold)


def net_benefit_reference(
    prevalence: np.ndarray | float, threshold: np.ndarray | float
) -> np.ndarray | float:
    """The better of the two default strategies: treat all, or treat none at zero."""

    return np.maximum(net_benefit_treat_all(prevalence, threshold), 0.0)


def decision_curve(
    targets: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict[str, np.ndarray]:
    """Counts, flagged fraction, and net benefit across a threshold grid for one label."""

    positive = np.asarray(targets, dtype=bool)
    scores = np.asarray(probabilities, dtype=np.float64)
    if positive.ndim != 1 or scores.shape != positive.shape:
        raise ValueError("targets and probabilities must be matching one-dimensional arrays")
    flagged = scores[:, None] >= np.asarray(thresholds, dtype=np.float64)[None, :]
    true_positives = np.logical_and(flagged, positive[:, None]).sum(axis=0)
    false_positives = np.logical_and(flagged, ~positive[:, None]).sum(axis=0)
    records = len(positive)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "flagged_fraction": flagged.sum(axis=0) / records,
        "net_benefit": net_benefit(true_positives, false_positives, records, thresholds),
    }


def superior_threshold_summary(
    thresholds: np.ndarray, model: np.ndarray, reference: np.ndarray
) -> dict[str, object]:
    """Where on the grid the model beats both defaults, and by how much at its best.

    The superior set is reported as a count with its endpoints rather than as a list, and
    separately as contiguous or not, because a set with holes in it means the model wins
    at some thresholds and loses at others inside the same range. Endpoints and contiguity
    are null when nothing on the grid clears both defaults.
    """

    gain = np.asarray(model, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    superior = np.flatnonzero(np.asarray(model) > np.asarray(reference))
    best = int(np.argmax(gain))
    empty = superior.size == 0
    return {
        "superior_threshold_count": int(superior.size),
        "superior_threshold_min": None if empty else float(thresholds[superior.min()]),
        "superior_threshold_max": None if empty else float(thresholds[superior.max()]),
        "superior_thresholds_contiguous": (
            None if empty else bool(superior.size == superior.max() - superior.min() + 1)
        ),
        "max_net_benefit_gain": float(gain[best]),
        "max_net_benefit_gain_threshold": float(thresholds[best]),
    }


def confusion_counts(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Weighted confusion counts at one threshold, flagging records with p >= t.

    Weights default to one per record, so the unweighted point estimate and a bootstrap
    replicate go through the same counting rule.
    """

    positive = np.asarray(targets, dtype=bool)
    flagged = np.asarray(probabilities, dtype=np.float64) >= threshold
    if weights is None:
        weights = np.ones(len(positive), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != positive.shape:
        raise ValueError("weights must carry one value per record")
    return {
        "tp": float(weights[flagged & positive].sum()),
        "fp": float(weights[flagged & ~positive].sum()),
        "tn": float(weights[~flagged & ~positive].sum()),
        "fn": float(weights[~flagged & positive].sum()),
    }


def defined_ratio(numerator: float, denominator: float) -> float | None:
    """Return None rather than a number when the denominator is zero."""

    if denominator == 0:
        return None
    return numerator / denominator


def operating_point_metrics(counts: dict[str, float]) -> dict[str, float | None]:
    """Screening arithmetic at one operating point, with undefined rates left undefined."""

    true_positive = counts["tp"]
    false_positive = counts["fp"]
    true_negative = counts["tn"]
    false_negative = counts["fn"]
    records = true_positive + false_positive + true_negative + false_negative
    alerts = true_positive + false_positive
    return {
        "sensitivity": defined_ratio(true_positive, true_positive + false_negative),
        "specificity": defined_ratio(true_negative, true_negative + false_positive),
        "ppv": defined_ratio(true_positive, alerts),
        "npv": defined_ratio(true_negative, true_negative + false_negative),
        "alerts_per_1000_records": defined_ratio(1000.0 * alerts, records),
        "alerts_per_true_positive": defined_ratio(alerts, true_positive),
        "records_screened_per_true_positive": defined_ratio(records, true_positive),
        "missed_cases_per_1000_records": defined_ratio(1000.0 * false_negative, records),
    }


def sensitivity_threshold(
    targets: np.ndarray, probabilities: np.ndarray, minimum_sensitivity: float
) -> tuple[float, bool]:
    """The largest observed probability whose rule attains a sensitivity floor.

    Sensitivity is nonincreasing in the threshold, so the qualifying candidates form a
    prefix of the sorted observed values and the largest qualifying value is the last of
    them. A label with no positives has no defined sensitivity and counts as unattained.
    The fallback threshold is 0.0, which flags every record, and the caller is told the
    floor was not attained rather than being handed a threshold that looks chosen.
    """

    positive = np.asarray(targets, dtype=bool)
    scores = np.asarray(probabilities, dtype=np.float64)
    positives = int(positive.sum())
    if positives == 0:
        return 0.0, False
    candidates = np.unique(scores)
    detected = (scores[positive][None, :] >= candidates[:, None]).sum(axis=1)
    qualifying = np.flatnonzero(detected / positives >= minimum_sensitivity)
    if qualifying.size == 0:
        return 0.0, False
    return float(candidates[int(qualifying.max())]), True


def utility_basis(
    targets: np.ndarray,
    calibrated: np.ndarray,
    uncalibrated: np.ndarray,
    registered_threshold: float,
    interval_thresholds: Sequence[float],
) -> np.ndarray:
    """Per-record design whose weighted column sums are every count the intervals need.

    Columns are the positive indicator, then a true-positive and a false-positive
    indicator at each interval threshold on the calibrated probabilities, then the same
    pair at the registered F1 threshold on the uncalibrated probabilities. One matrix
    product of a batch of weight rows against this design yields every weighted count for
    that batch without materializing a single resampled dataset.
    """

    positive = np.asarray(targets, dtype=bool)
    columns = [positive.astype(np.float64)]
    scored = [
        (np.asarray(calibrated, dtype=np.float64), threshold) for threshold in interval_thresholds
    ]
    scored.append((np.asarray(uncalibrated, dtype=np.float64), float(registered_threshold)))
    for scores, threshold in scored:
        flagged = scores >= threshold
        columns.append(np.logical_and(flagged, positive).astype(np.float64))
        columns.append(np.logical_and(flagged, ~positive).astype(np.float64))
    return np.stack(columns, axis=1)


def utility_basis_width(interval_thresholds: Sequence[float]) -> int:
    """Columns one label contributes to the design built by `utility_basis`."""

    return 1 + 2 * (len(interval_thresholds) + 1)


def weighted_totals(weights: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Accumulate weighted column totals for a batch of weight rows.

    The vectorized BLAS kernel behind `matmul` sets spurious floating-point flags on the
    padding lanes of the last block, so the flags are silenced here exactly as the
    registered calibration kernel silences them. Both operands are finite and
    non-negative by construction: weights are multinomial counts and the basis holds
    indicators.
    """

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return weights @ basis


def utility_statistics_from_totals(
    totals: np.ndarray, total_weight: np.ndarray, interval_thresholds: Sequence[float]
) -> dict[str, np.ndarray]:
    """Turn one label's weighted column totals into replicate-wise utility statistics.

    Prevalence is recomputed inside the replicate, so the treat-all comparator is drawn
    from the same resample as the model curve and the gain is a paired difference.
    """

    positive = totals[:, 0]
    prevalence = positive / total_weight
    negative = total_weight - positive
    statistics: dict[str, np.ndarray] = {}
    for position, threshold in enumerate(interval_thresholds):
        true_positive = totals[:, 1 + 2 * position]
        false_positive = totals[:, 2 + 2 * position]
        model = net_benefit(true_positive, false_positive, total_weight, threshold)
        statistics[f"net_benefit__{threshold:.2f}"] = model
        statistics[f"net_benefit_gain__{threshold:.2f}"] = model - net_benefit_reference(
            prevalence, threshold
        )
    true_positive = totals[:, 1 + 2 * len(interval_thresholds)]
    false_positive = totals[:, 2 + 2 * len(interval_thresholds)]
    statistics["sensitivity"] = np.divide(
        true_positive, positive, out=np.full(len(totals), np.nan), where=positive > 0
    )
    statistics["specificity"] = np.divide(
        negative - false_positive, negative, out=np.full(len(totals), np.nan), where=negative > 0
    )
    return statistics


def bootstrap_utility_and_auroc(
    targets: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    basis_by_model: dict[str, np.ndarray],
    label_indices: np.ndarray,
    group_inverse: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """One pass over the registered resamples for both the proof and the estimates.

    The AUROC block exists only to prove the resamples are the registered ones. The
    weighted utility totals ride on the same weight rows, so every interval published here
    comes from the draws the registered macro estimate came from.
    """

    label_count = len(label_indices)
    auroc = {
        name: np.empty((replicates, label_count), dtype=np.float64)
        for name in probabilities_by_model
    }
    totals = {
        name: np.empty((replicates, basis.shape[1]), dtype=np.float64)
        for name, basis in basis_by_model.items()
    }
    total_weight = np.empty(replicates, dtype=np.float64)
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        total_weight[start:end] = weights.sum(axis=1)
        for name, probabilities in probabilities_by_model.items():
            for column, label_index in enumerate(label_indices):
                auroc[name][start:end, column] = weighted_roc_auc_batch(
                    targets[:, label_index], probabilities[:, label_index], weights
                )
            totals[name][start:end] = weighted_totals(weights, basis_by_model[name])
    return auroc, totals, total_weight


def interval_record(distribution: np.ndarray, point: float) -> dict[str, float]:
    lower, upper = percentile_interval(distribution)
    return {"point": point, "lower_95": lower, "upper_95": upper}


def grid_position(grid: np.ndarray, threshold: float) -> int:
    """Locate a threshold on the grid, refusing a value the grid does not carry."""

    matches = np.flatnonzero(np.isclose(grid, threshold, rtol=0, atol=1e-9))
    if matches.size != 1:
        raise RuntimeError(f"threshold {threshold} is not a single point of the grid")
    return int(matches[0])


def interval_columns(
    intervals: dict[str, dict[str, dict[str, float]]], threshold: float
) -> dict[str, float | None]:
    """Bootstrap bounds for one grid row, empty where no interval was computed.

    Intervals are attached at three thresholds rather than all fifty, so the remaining
    rows carry nulls instead of a number that was never estimated.
    """

    entry = intervals.get(f"{threshold:.2f}")
    columns: dict[str, float | None] = {}
    for name in ("net_benefit_calibrated", "net_benefit_gain"):
        source = "net_benefit" if name == "net_benefit_calibrated" else "net_benefit_gain"
        columns[f"{name}_lower_95"] = None if entry is None else entry[source]["lower_95"]
        columns[f"{name}_upper_95"] = None if entry is None else entry[source]["upper_95"]
    return columns


def validation_sensitivity_thresholds(
    path: Path,
    *,
    choices: dict,
    analysis: ModuleType,
    headline_keys: list[str],
    label_order: list[str],
) -> dict[str, dict[str, dict[str, object]]]:
    """Choose a fold-9 sensitivity threshold per architecture and headline label.

    The fold-9 ensemble is bound to the sealed choices by the target and record-index
    digests that were frozen with it, so this exploratory threshold set cannot be fit on a
    record set other than the validation fold the registered thresholds were fit on. No
    test or external cohort enters this function.
    """

    ensemble = np.load(path)
    targets = np.asarray(ensemble["targets"], dtype=np.uint8)
    record_indices = np.asarray(ensemble["record_indices"], dtype=np.int64)
    if analysis.array_sha256(targets) != choices.get("validation_targets_sha256"):
        raise RuntimeError(f"fold-9 targets do not match the sealed digest: {path}")
    if analysis.array_sha256(record_indices) != choices.get("validation_record_indices_sha256"):
        raise RuntimeError(f"fold-9 record indices do not match the sealed digest: {path}")
    if int(choices.get("validation_records", -1)) != len(targets):
        raise RuntimeError(f"fold-9 ensemble holds {len(targets)} records, the seal disagrees")

    table: dict[str, dict[str, dict[str, object]]] = {}
    for architecture in analysis.ARCHITECTURES:
        probabilities = np.asarray(ensemble[f"{architecture}_probabilities"], dtype=np.float64)
        if probabilities.shape != targets.shape:
            raise RuntimeError(f"fold-9 {architecture} probabilities do not match the targets")
        table[architecture] = {}
        for key in headline_keys:
            label_index = label_order.index(key)
            threshold, attainable = sensitivity_threshold(
                targets[:, label_index], probabilities[:, label_index], SENSITIVITY_TARGET
            )
            table[architecture][key] = {
                "threshold": threshold,
                "attainable": attainable,
                "validation_positives": int(targets[:, label_index].sum()),
            }
    return table


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    verify_preregistration_seal(root)
    evaluation_seal, choices = verify_evaluation_seal(root)

    analysis = load_analysis_module()

    manifest = json.loads(args.labels.read_text())
    label_order = [entry["label_key"] for entry in manifest["labels"]]
    diagnosis = {entry["label_key"]: entry["diagnosis"] for entry in manifest["labels"]}
    headline_keys = list(manifest["headline_label_keys"])
    headline_indices = np.array([label_order.index(key) for key in headline_keys], dtype=np.int64)
    basis_columns = utility_basis_width(INTERVAL_THRESHOLDS)

    registered_thresholds = {
        architecture: {
            key: float(choices["architectures"][architecture]["thresholds_by_label"][key])
            for key in label_order
        }
        for architecture in analysis.ARCHITECTURES
    }
    sensitivity_thresholds = validation_sensitivity_thresholds(
        args.validation_predictions,
        choices=choices,
        analysis=analysis,
        headline_keys=headline_keys,
        label_order=label_order,
    )
    threshold_sets = {
        REGISTERED_THRESHOLD_SET: {
            architecture: {
                key: {"threshold": registered_thresholds[architecture][key], "attainable": True}
                for key in headline_keys
            }
            for architecture in analysis.ARCHITECTURES
        },
        SENSITIVITY_THRESHOLD_SET: sensitivity_thresholds,
    }

    registered_path = args.registered_bootstrap or (
        args.output_root / "bootstrap_distributions.npz"
    )
    registered = np.load(registered_path)

    curve_rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    label_summaries: list[dict[str, object]] = []
    cross_checks: list[dict[str, object]] = []

    for cohort in analysis.COHORTS:
        cohort_data = analysis.load_verified_cohort(
            cohort=cohort,
            prediction_root=args.prediction_root,
            store_root=args.store_root,
            evaluation_seal=evaluation_seal,
            choices=choices,
            label_count=len(label_order),
        )
        targets = cohort_data["targets"]
        uncalibrated_by_model = cohort_data["probabilities"]
        calibrated_by_model = cohort_data["calibrated"]
        group_inverse = cluster_inverse(cohort, cohort_data["metadata"])
        records = len(targets)

        basis_by_model = {
            architecture: np.concatenate(
                [
                    utility_basis(
                        targets[:, label_index],
                        calibrated_by_model[architecture][:, label_index],
                        uncalibrated_by_model[architecture][:, label_index],
                        registered_thresholds[architecture][headline_keys[column]],
                        INTERVAL_THRESHOLDS,
                    )
                    for column, label_index in enumerate(headline_indices)
                ],
                axis=1,
            )
            for architecture in analysis.ARCHITECTURES
        }
        auroc, totals, total_weight = bootstrap_utility_and_auroc(
            targets,
            uncalibrated_by_model,
            basis_by_model,
            headline_indices,
            group_inverse,
            replicates=args.bootstrap_replicates,
            seed=analysis.COHORT_SEEDS[cohort],
            batch_size=args.bootstrap_batch_size,
        )

        # Proof that these are the registered resamples, not a new bootstrap.
        for architecture in analysis.ARCHITECTURES:
            rebuilt = macro_of_per_label(
                auroc[architecture], np.arange(len(headline_indices), dtype=np.int64)
            )
            reference = registered[f"{cohort}__{architecture}"]
            identical = bool(np.array_equal(rebuilt, reference))
            largest = float(np.max(np.abs(rebuilt - reference)))
            cross_checks.append(
                {
                    "cohort": cohort,
                    "architecture": architecture,
                    "bit_identical_to_registered_macro": identical,
                    "largest_absolute_difference": largest,
                }
            )
            if not identical:
                raise RuntimeError(
                    f"rebuilt macro-AUROC replicates for {cohort}/{architecture} do not match "
                    f"the registered distribution (largest difference {largest:.3e}); "
                    "the resamples are not the registered ones and no interval may be published"
                )
        print(f"verified registered resamples reproduced for {cohort}")

        for architecture in analysis.ARCHITECTURES:
            for column, label_index in enumerate(headline_indices):
                key = headline_keys[column]
                truth = targets[:, label_index]
                positives = int(truth.sum())
                prevalence = positives / records
                calibrated_curve = decision_curve(
                    truth, calibrated_by_model[architecture][:, label_index], THRESHOLD_GRID
                )
                uncalibrated_curve = decision_curve(
                    truth, uncalibrated_by_model[architecture][:, label_index], THRESHOLD_GRID
                )
                treat_all = net_benefit_treat_all(prevalence, THRESHOLD_GRID)
                reference_curve = net_benefit_reference(prevalence, THRESHOLD_GRID)
                gain = calibrated_curve["net_benefit"] - reference_curve
                replicate_statistics = utility_statistics_from_totals(
                    totals[architecture][:, column * basis_columns : (column + 1) * basis_columns],
                    total_weight,
                    INTERVAL_THRESHOLDS,
                )
                intervals = {
                    f"{threshold:.2f}": {
                        "net_benefit": interval_record(
                            replicate_statistics[f"net_benefit__{threshold:.2f}"],
                            float(
                                calibrated_curve["net_benefit"][
                                    grid_position(THRESHOLD_GRID, threshold)
                                ]
                            ),
                        ),
                        "net_benefit_gain": interval_record(
                            replicate_statistics[f"net_benefit_gain__{threshold:.2f}"],
                            float(gain[grid_position(THRESHOLD_GRID, threshold)]),
                        ),
                        "net_benefit_treat_all": float(
                            treat_all[grid_position(THRESHOLD_GRID, threshold)]
                        ),
                    }
                    for threshold in INTERVAL_THRESHOLDS
                }
                for position, threshold in enumerate(THRESHOLD_GRID):
                    bounds = interval_columns(intervals, threshold)
                    curve_rows.append(
                        {
                            "cohort": cohort,
                            "architecture": architecture,
                            "label_key": key,
                            "diagnosis": diagnosis[key],
                            "records": records,
                            "positives": positives,
                            "prevalence": prevalence,
                            "threshold": float(threshold),
                            "net_benefit_calibrated": float(
                                calibrated_curve["net_benefit"][position]
                            ),
                            "net_benefit_uncalibrated": float(
                                uncalibrated_curve["net_benefit"][position]
                            ),
                            "net_benefit_treat_all": float(treat_all[position]),
                            "net_benefit_gain": float(gain[position]),
                            "standardized_net_benefit_calibrated": defined_ratio(
                                float(calibrated_curve["net_benefit"][position]), prevalence
                            ),
                            "flagged_fraction": float(
                                calibrated_curve["flagged_fraction"][position]
                            ),
                            "true_positives": int(calibrated_curve["true_positives"][position]),
                            "false_positives": int(calibrated_curve["false_positives"][position]),
                            **bounds,
                        }
                    )
                label_summaries.append(
                    {
                        "cohort": cohort,
                        "architecture": architecture,
                        "label_key": key,
                        "diagnosis": diagnosis[key],
                        "records": records,
                        "positives": positives,
                        "prevalence": prevalence,
                        **superior_threshold_summary(
                            THRESHOLD_GRID, calibrated_curve["net_benefit"], reference_curve
                        ),
                        "intervals": intervals,
                    }
                )

                for set_name, table in threshold_sets.items():
                    entry = table[architecture][key]
                    threshold = float(entry["threshold"])
                    counts = confusion_counts(
                        truth, uncalibrated_by_model[architecture][:, label_index], threshold
                    )
                    row: dict[str, object] = {
                        "cohort": cohort,
                        "architecture": architecture,
                        "threshold_set": set_name,
                        "label_key": key,
                        "diagnosis": diagnosis[key],
                        "threshold": threshold,
                        "threshold_attainable": bool(entry["attainable"]),
                        "records": records,
                        "positives": positives,
                        "negatives": records - positives,
                        "prevalence": prevalence,
                        "tp": int(counts["tp"]),
                        "fp": int(counts["fp"]),
                        "tn": int(counts["tn"]),
                        "fn": int(counts["fn"]),
                        **operating_point_metrics(counts),
                    }
                    if set_name == REGISTERED_THRESHOLD_SET:
                        for metric in ("sensitivity", "specificity"):
                            lower, upper = percentile_interval(replicate_statistics[metric])
                            row[f"{metric}_lower_95"] = lower
                            row[f"{metric}_upper_95"] = upper
                    else:
                        for metric in ("sensitivity", "specificity"):
                            row[f"{metric}_lower_95"] = None
                            row[f"{metric}_upper_95"] = None
                    operating_rows.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_root / "exploratory_decision_curves.csv"
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
    operating_path = args.output_root / "exploratory_operating_points.csv"
    pd.DataFrame(operating_rows).to_csv(operating_path, index=False)

    summary_path = args.output_root / "exploratory_clinical_utility.json"
    summary = {
        "stage": "post-seal exploratory addition",
        "stage_date": EXPLORATORY_STAGE_DATE,
        "changes_registered_quantities": False,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seeds": dict(analysis.COHORT_SEEDS),
        "interval_type": "percentile",
        "multiplicity_adjustment": "none; marginal intervals, matching the registered plan",
        "ptb_cluster_unit": "patient_id",
        "external_cluster_unit": "record",
        "threshold_grid": {
            "minimum": float(THRESHOLD_GRID.min()),
            "maximum": float(THRESHOLD_GRID.max()),
            "step": 0.01,
            "points": int(len(THRESHOLD_GRID)),
        },
        "interval_thresholds": list(INTERVAL_THRESHOLDS),
        "primary_probability_scale": "temperature-calibrated, temperature sealed from fold 9",
        "operating_point_probability_scale": "uncalibrated, as the registered thresholds are",
        "sensitivity_target": SENSITIVITY_TARGET,
        "validation_sensitivity_90_thresholds": sensitivity_thresholds,
        "registered_resample_cross_check": cross_checks,
        "decision_curve_summaries": label_summaries,
        "outputs": [curve_path.name, operating_path.name, summary_path.name],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"wrote {curve_path}")
    print(f"wrote {operating_path}")
    print(f"wrote {summary_path}")
    beating = sum(1 for entry in label_summaries if entry["superior_threshold_count"] > 0)
    print(f"{beating} of {len(label_summaries)} label curves clear both default strategies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
