"""Efficient cluster-bootstrap intervals for discrimination and calibration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

CALIBRATION_BINS = 15
CALIBRATION_METRICS = ("macro_per_label_ece", "macro_brier", "pooled_ece")


def bootstrap_record_weight_batches(
    group_inverse: np.ndarray,
    replicates: int,
    *,
    seed: int,
    batch_size: int = 100,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield record weights from a nonparametric cluster bootstrap."""

    group_inverse = np.asarray(group_inverse, dtype=np.int64)
    groups = int(group_inverse.max()) + 1
    if groups < 2 or np.unique(group_inverse).size != groups:
        raise ValueError("group_inverse must contain contiguous group indices")
    generator = np.random.default_rng(seed)
    probabilities = np.full(groups, 1 / groups)
    for start in range(0, replicates, batch_size):
        current_batch = min(batch_size, replicates - start)
        group_weights = generator.multinomial(groups, probabilities, size=current_batch)
        yield start, group_weights[:, group_inverse].astype(np.float64)


def weighted_roc_auc_batch(
    targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Compute exact tie-aware weighted AUROC for each row of bootstrap weights."""

    targets = np.asarray(targets, dtype=bool)
    probabilities = np.asarray(probabilities)
    weights = np.asarray(weights, dtype=np.float64)
    if targets.ndim != 1 or probabilities.shape != targets.shape:
        raise ValueError("targets and probabilities must be matching one-dimensional arrays")
    if weights.ndim != 2 or weights.shape[1] != len(targets):
        raise ValueError("weights must have shape (replicates, records)")

    order = np.argsort(probabilities, kind="stable")
    sorted_targets = targets[order]
    sorted_probabilities = probabilities[order]
    sorted_weights = weights[:, order]
    group_starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_probabilities) != 0) + 1))
    positive_by_score = np.add.reduceat(
        sorted_weights * sorted_targets[None, :], group_starts, axis=1
    )
    negative_by_score = np.add.reduceat(
        sorted_weights * (~sorted_targets)[None, :], group_starts, axis=1
    )
    negative_before = np.cumsum(negative_by_score, axis=1) - negative_by_score
    numerator = np.sum(positive_by_score * (negative_before + 0.5 * negative_by_score), axis=1)
    positive_total = positive_by_score.sum(axis=1)
    negative_total = negative_by_score.sum(axis=1)
    return np.divide(
        numerator,
        positive_total * negative_total,
        out=np.full(len(weights), np.nan),
        where=(positive_total > 0) & (negative_total > 0),
    )


def _macro_average(label_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average over defined labels only and report how many contributed.

    This is the registered macro rule: a label whose statistic is undefined in a
    replicate drops out of that replicate's average rather than voiding it, and
    the contributing-label count travels with the estimate.
    """

    contributing = np.isfinite(label_values).sum(axis=1)
    macro = np.divide(
        np.nansum(label_values, axis=1),
        contributing,
        out=np.full(len(label_values), np.nan),
        where=contributing > 0,
    )
    return macro, contributing


def calibration_bin_assignments(
    probabilities: np.ndarray, bins: int = CALIBRATION_BINS
) -> np.ndarray:
    """Assign probabilities to equal-width bins exactly as `metrics` does."""

    edges = np.linspace(0, 1, bins + 1)
    flat = np.asarray(probabilities, dtype=np.float64).ravel()
    return np.minimum(np.digitize(flat, edges[1:-1]), bins - 1)


def calibration_bin_basis(
    targets: np.ndarray, probabilities: np.ndarray, *, bins: int = CALIBRATION_BINS
) -> np.ndarray:
    """Build the per-record design whose weighted column sums are bin totals.

    Column blocks are, in order, the bin indicator, the bin indicator scaled by
    the target, the bin indicator scaled by the probability, and a final squared
    residual column. One matrix product of a batch of weight rows against this
    design therefore yields every quantity the weighted calibration statistics
    need, without materializing a single resampled dataset.
    """

    targets = np.asarray(targets, dtype=np.float64).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must be matching arrays")
    assignments = calibration_bin_assignments(probabilities, bins=bins)
    records = np.arange(len(assignments))
    basis = np.zeros((len(assignments), 3 * bins + 1), dtype=np.float64)
    basis[records, assignments] = 1.0
    basis[records, assignments + bins] = targets
    basis[records, assignments + 2 * bins] = probabilities
    basis[:, 3 * bins] = np.square(probabilities - targets)
    return basis


def _weighted_bin_totals(weights: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Accumulate per-bin weighted totals for a batch of weight rows.

    The vectorized BLAS kernel behind `matmul` sets spurious floating-point
    flags on the padding lanes of the last block, so this is the one place the
    flags are silenced. Both operands are finite and non-negative by
    construction: weights are multinomial counts and the basis holds
    indicators, targets, probabilities, and squared residuals.
    """

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return weights @ basis


def _expected_calibration_error_from_totals(totals: np.ndarray, bins: int) -> np.ndarray:
    weight = totals[:, :bins]
    target = totals[:, bins : 2 * bins]
    probability = totals[:, 2 * bins : 3 * bins]
    total_weight = weight.sum(axis=1)
    occupied = weight > 0
    mean_target = np.divide(target, weight, out=np.zeros_like(target), where=occupied)
    mean_probability = np.divide(
        probability, weight, out=np.zeros_like(probability), where=occupied
    )
    share = np.divide(
        weight,
        total_weight[:, None],
        out=np.zeros_like(weight),
        where=total_weight[:, None] > 0,
    )
    error = (share * np.abs(mean_target - mean_probability)).sum(axis=1)
    return np.where(total_weight > 0, error, np.nan)


def _brier_from_totals(totals: np.ndarray, bins: int) -> np.ndarray:
    total_weight = totals[:, :bins].sum(axis=1)
    return np.divide(
        totals[:, 3 * bins],
        total_weight,
        out=np.full(len(totals), np.nan),
        where=total_weight > 0,
    )


def weighted_expected_calibration_error_batch(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = CALIBRATION_BINS,
) -> np.ndarray:
    """Weighted expected calibration error for each row of bootstrap weights."""

    basis = calibration_bin_basis(targets, probabilities, bins=bins)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(basis):
        raise ValueError("weights must have shape (replicates, records)")
    return _expected_calibration_error_from_totals(_weighted_bin_totals(weights, basis), bins)


def weighted_brier_batch(
    targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted mean squared error for each row of bootstrap weights."""

    targets = np.asarray(targets, dtype=np.float64).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must be matching arrays")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(targets):
        raise ValueError("weights must have shape (replicates, records)")
    total_weight = weights.sum(axis=1)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        residual_total = weights @ np.square(probabilities - targets)
    return np.divide(
        residual_total,
        total_weight,
        out=np.full(len(weights), np.nan),
        where=total_weight > 0,
    )


def _weighted_calibration_statistics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_indices: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = CALIBRATION_BINS,
) -> dict[str, np.ndarray]:
    """Macro and pooled weighted calibration statistics for one probability set."""

    replicates = len(weights)
    label_error = np.empty((replicates, len(label_indices)), dtype=np.float64)
    label_brier = np.empty((replicates, len(label_indices)), dtype=np.float64)
    pooled_totals = np.zeros((replicates, 3 * bins + 1), dtype=np.float64)
    for column, label_index in enumerate(label_indices):
        basis = calibration_bin_basis(
            targets[:, label_index], probabilities[:, label_index], bins=bins
        )
        totals = _weighted_bin_totals(weights, basis)
        pooled_totals += totals
        label_error[:, column] = _expected_calibration_error_from_totals(totals, bins)
        label_brier[:, column] = _brier_from_totals(totals, bins)
    macro_error, contributing = _macro_average(label_error)
    macro_brier, _ = _macro_average(label_brier)
    return {
        "macro_per_label_ece": macro_error,
        "macro_brier": macro_brier,
        "pooled_ece": _expected_calibration_error_from_totals(pooled_totals, bins),
        "contributing_labels": contributing,
    }


def paired_bootstrap_macro_auroc(
    targets: np.ndarray,
    probabilities_by_model: Mapping[str, np.ndarray],
    label_indices: np.ndarray,
    group_inverse: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 100,
) -> dict[str, np.ndarray]:
    """Return paired macro-AUROC bootstrap distributions for multiple models."""

    model_names = list(probabilities_by_model)
    output = {name: np.empty(replicates, dtype=np.float64) for name in model_names}
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        for model_name, probabilities in probabilities_by_model.items():
            label_aurocs = np.stack(
                [
                    weighted_roc_auc_batch(
                        targets[:, label_index], probabilities[:, label_index], weights
                    )
                    for label_index in label_indices
                ],
                axis=1,
            )
            output[model_name][start:end], _ = _macro_average(label_aurocs)
    if len(model_names) == 2:
        output[f"{model_names[1]}_minus_{model_names[0]}"] = (
            output[model_names[1]] - output[model_names[0]]
        )
    return output


def paired_bootstrap_cohort_statistics(
    targets: np.ndarray,
    probabilities_by_model: Mapping[str, np.ndarray],
    label_indices: np.ndarray,
    group_inverse: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 100,
    calibrated_by_model: Mapping[str, np.ndarray] | None = None,
    include_calibration: bool = True,
    bins: int = CALIBRATION_BINS,
) -> dict[str, np.ndarray]:
    """Bootstrap discrimination and calibration on one shared set of resamples.

    Every statistic in a cohort rides on the same cluster resamples, so the
    calibrated-minus-uncalibrated contrast within an architecture and the
    model contrast within a cohort are both paired. The fold-9 temperature is
    never refitted inside a replicate: `calibrated_by_model` carries the
    probabilities that the frozen temperature already produced.

    Returned keys are, per model, the macro-AUROC distribution under the model's
    own name, `<model>__macro_auroc_labels`, and, for each calibration variant,
    `<model>__<variant>__<metric>` plus `<model>__<variant>__calibration_labels`.
    Contrasts add `<model>__calibrated_minus_uncalibrated__<metric>` and, when
    exactly two models are given, `<second>_minus_<first>` for macro-AUROC and
    `<second>_minus_<first>__<variant>__<metric>` for calibration.

    `include_calibration=False` restricts the pass to discrimination and its
    contributing-label counts, which is what the subgroup tables need.
    """

    label_indices = np.asarray(label_indices, dtype=np.int64)
    model_names = list(probabilities_by_model)
    variants: dict[str, Mapping[str, np.ndarray]] = {}
    if include_calibration:
        variants["uncalibrated"] = probabilities_by_model
    if calibrated_by_model is not None:
        if not include_calibration:
            raise ValueError("calibrated probabilities require include_calibration")
        if list(calibrated_by_model) != model_names:
            raise ValueError("calibrated probabilities must cover the same models in order")
        variants["calibrated"] = calibrated_by_model

    output: dict[str, np.ndarray] = {}
    for model_name in model_names:
        output[model_name] = np.empty(replicates, dtype=np.float64)
        output[f"{model_name}__macro_auroc_labels"] = np.empty(replicates, dtype=np.int64)
        for variant in variants:
            for metric in CALIBRATION_METRICS:
                output[f"{model_name}__{variant}__{metric}"] = np.empty(
                    replicates, dtype=np.float64
                )
            output[f"{model_name}__{variant}__calibration_labels"] = np.empty(
                replicates, dtype=np.int64
            )

    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        for model_name in model_names:
            label_aurocs = np.stack(
                [
                    weighted_roc_auc_batch(
                        targets[:, label_index],
                        probabilities_by_model[model_name][:, label_index],
                        weights,
                    )
                    for label_index in label_indices
                ],
                axis=1,
            )
            macro_auroc, contributing = _macro_average(label_aurocs)
            output[model_name][start:end] = macro_auroc
            output[f"{model_name}__macro_auroc_labels"][start:end] = contributing
            for variant, table in variants.items():
                statistics = _weighted_calibration_statistics(
                    targets, table[model_name], label_indices, weights, bins=bins
                )
                for metric in CALIBRATION_METRICS:
                    output[f"{model_name}__{variant}__{metric}"][start:end] = statistics[metric]
                output[f"{model_name}__{variant}__calibration_labels"][start:end] = statistics[
                    "contributing_labels"
                ]

    if "calibrated" in variants:
        for model_name in model_names:
            for metric in CALIBRATION_METRICS:
                output[f"{model_name}__calibrated_minus_uncalibrated__{metric}"] = (
                    output[f"{model_name}__calibrated__{metric}"]
                    - output[f"{model_name}__uncalibrated__{metric}"]
                )
    if len(model_names) == 2:
        reference, comparison = model_names
        output[f"{comparison}_minus_{reference}"] = output[comparison] - output[reference]
        for variant in variants:
            for metric in CALIBRATION_METRICS:
                output[f"{comparison}_minus_{reference}__{variant}__{metric}"] = (
                    output[f"{comparison}__{variant}__{metric}"]
                    - output[f"{reference}__{variant}__{metric}"]
                )
    return output


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper)
