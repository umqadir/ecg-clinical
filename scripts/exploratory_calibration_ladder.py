#!/usr/bin/env python3
"""Decompose the external calibration error into a ladder of score transforms.

POST-SEAL EXPLORATORY ADDITION, 2026-07-26. This stage was not part of the frozen
analysis. It was written after the evaluation seal was broken and after the registered
results existed, and it is kept separate from `analyze_results.py` for that reason.

The registered result is not under revision. The registered calibration procedure is one
scalar temperature per architecture, fitted on PTB-XL fold 9, and it moves external macro
per-label ECE by about 0.0005 against an error near 0.08. That null stands. Rungs 3 to 6
of this ladder fit their parameters on the evaluation cohort's own labels, so they are
oracle upper bounds and are not deployable. Rungs 7 to 9 are deployable but were not
registered and were selected after seeing the registered outcome. Nothing here converts
the registered null into a positive result, and no number here may be reported as a
calibration method the study evaluated.

What it measures. A review claimed the dominant external miscalibration mechanism is
prior shift, meaning a change in label prevalence, removable by one additive logit offset
per label rather than by any multiplicative temperature. The ladder measures that claim
instead of assuming it. Every rung acts on the logit of the uncalibrated ensemble
probability, and every rung is scored with the registered estimator, 15 equal-width bins
per label macro-averaged over the 13 headline labels, via
`ecg_clinical.metrics.calibration_summary`. Rung 1 must reproduce the registered
uncalibrated macro per-label ECE and rung 2 the registered calibrated one; the run aborts
if either fails, so the ladder cannot be read against a different estimator than the
registered numbers it sits beside.

Rungs.
  1 uncalibrated: no transform.
  2 registered_temperature: the sealed fold-9 scalar temperature.
  3 oracle_global_temperature: one temperature per cohort per architecture, fitted on the
    cohort's own labels over the 13 headline labels pooled.
  4 oracle_per_label_temperature: one temperature per label, fitted on that label's own
    cohort labels.
  5 oracle_per_label_offset: one additive logit offset per label, temperature fixed at 1,
    fitted on that label's own cohort labels.
  6 oracle_per_label_affine: slope and offset per label, both fitted on the cohort's own
    labels.
  7 prevalence_only_intercept: offset equal to the target log-odds minus the source
    log-odds. Needs the target prevalence, no target scores, no fitting.
  8 sld_em: Saerens-Latinne-Decaestecker expectation-maximization run per label on the
    target cohort's unlabeled scores, initialized at the source prevalence, then the rung 7
    offset using the estimated target prevalence. Uses no target labels.
  9 bbse_hard: black-box shift estimation from the fold-9 confusion matrix at the
    registered F1 threshold and the target cohort's flagged rate, then the rung 7 offset
    using the estimated target prevalence. Uses no target labels.

Source prevalence. Taken from the PTB-XL training folds 1 to 8 of
`data/cache/waveforms/ptb_xl`, read from `metadata.parquet` column `strat_fold` and
`targets.npy`, both of which the sealed bound inputs digest. The run asserts agreement
with `data/derived/preregistration/harmonized_label_counts.csv` column
`ptb_train_positives` and aborts on any disagreement, so the two possible sources are
proven to be the same numbers rather than chosen between.

Uncertainty. Intervals are attached to rungs 1, 2, 5 and 7 only, the four rungs the
argument rests on. They come from the registered cluster bootstrap, not a new one: same
per-cohort seeds, same cluster units, 2000 replicates, batch size 100. The transform
parameters are held fixed at their full-cohort fitted values inside every replicate and
are never refitted, so a rung 5 or rung 7 interval is the sampling variability of the
metric under a fixed transform and not the variability of a fitting procedure. It
understates the uncertainty of an oracle rung for that reason. Two proofs that the
resamples are the registered ones run before any interval is published: per-label AUROC
replicates for the headline labels, macro-averaged, must equal the registered macro-AUROC
distribution in `results/bootstrap_distributions.npz` bit-for-bit, and the rung 1 and
rung 2 macro ECE replicate distributions must equal the registered uncalibrated and
calibrated distributions bit-for-bit. The run aborts if either fails.

What it does not do. It changes no registered quantity, rewrites no sealed artifact, and
re-runs no inference. It writes to its own two files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from ecg_clinical.bootstrap import (
    bootstrap_record_weight_batches,
    percentile_interval,
    weighted_expected_calibration_error_batch,
    weighted_roc_auc_batch,
)
from ecg_clinical.integrity import (
    sha256_file,
    verify_evaluation_seal,
    verify_preregistration_seal,
)
from ecg_clinical.metrics import (
    apply_temperature,
    calibration_summary,
    expected_calibration_error,
    fit_scalar_temperature,
)

EXPLORATORY_STAGE_DATE = "2026-07-26"
CALIBRATION_BINS = 15
PROBABILITY_CLIP = 1e-7
PREVALENCE_CLIP = 1e-6
TRAINING_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
LADDER_RUNGS = (
    "uncalibrated",
    "registered_temperature",
    "oracle_global_temperature",
    "oracle_per_label_temperature",
    "oracle_per_label_offset",
    "oracle_per_label_affine",
    "prevalence_only_intercept",
    "sld_em",
    "bbse_hard",
)
INTERVAL_RUNGS = (
    "uncalibrated",
    "registered_temperature",
    "oracle_per_label_offset",
    "prevalence_only_intercept",
)
ORACLE_RUNGS = (
    "oracle_global_temperature",
    "oracle_per_label_temperature",
    "oracle_per_label_offset",
    "oracle_per_label_affine",
)
ESTIMATED_PREVALENCE_RUNGS = ("sld_em", "bbse_hard")
LADDER_METRICS = ("macro_per_label_ece", "pooled_ece", "macro_brier")
RECOVERY_SIZES = (100, 250, 500, 1000)
RECOVERY_REPEATS = 50
RECOVERY_SEED = 20260726
AFFINE_SLOPE_BOUNDS = (1e-3, 1e3)
AFFINE_OFFSET_BOUNDS = (-40.0, 40.0)


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
    parser = argparse.ArgumentParser(description="Post-seal exploratory calibration ladder")
    parser.add_argument(
        "--prediction-root", type=Path, default=Path("artifacts/predictions/protected")
    )
    parser.add_argument(
        "--validation-ensembles",
        type=Path,
        default=Path("artifacts/predictions/validation_ensembles.npz"),
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/cache/waveforms"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/derived/preregistration/harmonized_labels.json"),
    )
    parser.add_argument(
        "--label-counts",
        type=Path,
        default=Path("data/derived/preregistration/harmonized_label_counts.csv"),
    )
    parser.add_argument("--registered-bootstrap", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=100)
    return parser.parse_args()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


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


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    """Logit of probabilities clipped at the bound `apply_temperature` uses."""

    values = np.asarray(probabilities, dtype=np.float64)
    return logit(np.clip(values, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))


def log_odds(prevalence: float) -> float:
    return float(logit(min(max(float(prevalence), PREVALENCE_CLIP), 1 - PREVALENCE_CLIP)))


def prior_shift_offset(source_prevalence: float, target_prevalence: float) -> float:
    """The label-free prior-shift correction: target log-odds minus source log-odds."""

    return log_odds(target_prevalence) - log_odds(source_prevalence)


def binary_nll_and_gradient(
    parameters: np.ndarray, targets: np.ndarray, logits: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean binary negative log likelihood of `slope * logit + offset` and its gradient."""

    slope, offset = float(parameters[0]), float(parameters[1])
    scores = slope * logits + offset
    value = float(np.mean(np.logaddexp(0.0, scores) - targets * scores))
    residual = expit(scores) - targets
    return value, np.array([float(np.mean(residual * logits)), float(np.mean(residual))])


def binary_nll(targets: np.ndarray, logits: np.ndarray, slope: float, offset: float) -> float:
    return binary_nll_and_gradient(np.array([slope, offset]), targets, logits)[0]


def fit_logit_affine(
    targets: np.ndarray, logits: np.ndarray, *, fit_slope: bool
) -> tuple[float, float]:
    """Fit slope and offset, or offset alone, by minimizing binary NLL.

    Returns (1.0, 0.0) when the label has no positive or no negative record, because the
    likelihood is then unbounded and the minimizer runs to a bound that carries no
    information. Callers count how often that happens.
    """

    targets = np.asarray(targets, dtype=np.float64).ravel()
    logits = np.asarray(logits, dtype=np.float64).ravel()
    positives = float(targets.sum())
    if positives <= 0 or positives >= len(targets):
        return 1.0, 0.0
    if fit_slope:
        bounds = [AFFINE_SLOPE_BOUNDS, AFFINE_OFFSET_BOUNDS]
        result = minimize(
            binary_nll_and_gradient,
            np.array([1.0, 0.0]),
            args=(targets, logits),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-12},
        )
        return float(result.x[0]), float(result.x[1])

    def offset_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = binary_nll_and_gradient(
            np.array([1.0, float(parameters[0])]), targets, logits
        )
        return value, gradient[1:]

    result = minimize(
        offset_objective,
        np.array([0.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=[AFFINE_OFFSET_BOUNDS],
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-12},
    )
    return 1.0, float(result.x[0])


def sld_em_prevalence(
    logits: np.ndarray,
    source_prevalence: float,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
) -> tuple[float, int]:
    """Saerens-Latinne-Decaestecker EM prevalence estimate from unlabeled target scores.

    The E step applies the prior-shift offset for the current prevalence estimate to every
    score, the M step sets the next estimate to the mean adjusted posterior. Returns the
    estimate and the number of iterations run.
    """

    logits = np.asarray(logits, dtype=np.float64).ravel()
    source_log_odds = log_odds(source_prevalence)
    estimate = min(max(float(source_prevalence), PREVALENCE_CLIP), 1 - PREVALENCE_CLIP)
    for iteration in range(1, max_iterations + 1):
        posterior = expit(logits + log_odds(estimate) - source_log_odds)
        updated = min(max(float(posterior.mean()), PREVALENCE_CLIP), 1 - PREVALENCE_CLIP)
        change = abs(updated - estimate)
        estimate = updated
        if change < tolerance:
            return estimate, iteration
    return estimate, max_iterations


def confusion_rates(targets: np.ndarray, flags: np.ndarray) -> tuple[float, float]:
    """True positive rate and false positive rate of a hard predictor."""

    targets = np.asarray(targets).ravel().astype(bool)
    flags = np.asarray(flags).ravel().astype(bool)
    positives = int(targets.sum())
    negatives = int((~targets).sum())
    true_positive_rate = float(flags[targets].mean()) if positives else float("nan")
    false_positive_rate = float(flags[~targets].mean()) if negatives else float("nan")
    return true_positive_rate, false_positive_rate


def bbse_hard_prevalence(
    true_positive_rate: float, false_positive_rate: float, flagged_rate: float
) -> float:
    """Invert flagged_rate = tpr * pi + fpr * (1 - pi) for the target prevalence."""

    separation = true_positive_rate - false_positive_rate
    if not np.isfinite(separation) or separation == 0.0:
        return float("nan")
    estimate = (flagged_rate - false_positive_rate) / separation
    return float(min(max(estimate, PREVALENCE_CLIP), 1 - PREVALENCE_CLIP))


def equal_mass_bin_assignments(scores: np.ndarray, bins: int = CALIBRATION_BINS) -> np.ndarray:
    """Assign scores to quantile bins, deduplicating tied edges so bins do not overlap.

    Ties collapse edges, so the realized bin count is at most `bins`. Every record with an
    identical score lands in one bin, which is the only assignment a quantile rule can make
    without splitting a tie group arbitrarily.
    """

    scores = np.asarray(scores, dtype=np.float64).ravel()
    edges = np.quantile(scores, np.linspace(0.0, 1.0, bins + 1))
    interior = np.unique(edges[1:-1])
    return np.searchsorted(interior, scores, side="right")


def equal_mass_expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, bins: int = CALIBRATION_BINS
) -> float:
    """Expected calibration error under equal-mass bins instead of equal-width bins."""

    targets = np.asarray(targets, dtype=np.float64).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    assignments = equal_mass_bin_assignments(probabilities, bins=bins)
    error = 0.0
    for bin_index in range(int(assignments.max()) + 1):
        selected = assignments == bin_index
        if selected.any():
            error += selected.mean() * abs(
                targets[selected].mean() - probabilities[selected].mean()
            )
    return float(error)


def equal_mass_macro_per_label_ece(
    targets: np.ndarray,
    probabilities: np.ndarray,
    headline_indices: np.ndarray,
    bins: int = CALIBRATION_BINS,
) -> float:
    return float(
        np.mean(
            [
                equal_mass_expected_calibration_error(
                    targets[:, index], probabilities[:, index], bins=bins
                )
                for index in headline_indices
            ]
        )
    )


def apply_logit_affine(
    probabilities: np.ndarray,
    label_indices: np.ndarray,
    slopes: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Transform the named columns by slope * logit + offset, pass the rest through.

    Only the headline columns enter any reported statistic, so the remaining columns are
    left at their uncalibrated values rather than given parameters that nothing consumes.
    """

    transformed = np.array(probabilities, dtype=np.float64, copy=True)
    logits = safe_logit(probabilities[:, label_indices])
    transformed[:, label_indices] = expit(
        logits * np.asarray(slopes, dtype=np.float64)[None, :]
        + np.asarray(offsets, dtype=np.float64)[None, :]
    )
    return transformed


def sealed_temperature_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Reproduce the sealed calibrated probabilities from the uncalibrated ensemble.

    Protected inference stored both ensembles as float32 and applied the temperature to
    the float32 ensemble, so the sealed calibrated array is only reproduced bit-for-bit by
    doing the same. The float64 view returned here is what the registered analysis read.
    """

    single = np.asarray(probabilities, dtype=np.float32)
    return apply_temperature(single, temperature).astype(np.float32).astype(np.float64)


def constant_predictor_matrix(prevalences: np.ndarray, records: int) -> np.ndarray:
    """A predictor that emits one fixed probability per label for every record."""

    return np.tile(np.asarray(prevalences, dtype=np.float64)[None, :], (records, 1))


def recovery_seed(cohort_index: int, architecture_index: int, sample_size: int) -> list[int]:
    """Seed entropy for one recovery-curve cell.

    One fixed base seed, 20260726, extended by the cohort index, the architecture index and
    the labeled-sample size. Every cell therefore has an independent stream that is
    reproducible from the base seed alone.
    """

    return [RECOVERY_SEED, cohort_index, architecture_index, int(sample_size)]


def held_out_platt_recovery(
    targets: np.ndarray,
    probabilities: np.ndarray,
    headline_indices: np.ndarray,
    *,
    sample_size: int,
    repeats: int,
    seed_entropy: list[int],
) -> dict[str, float]:
    """Fit per-label Platt scaling on k labeled records, score the remaining records.

    The held-out score is the registered macro per-label ECE, assembled from the registered
    per-label primitive `expected_calibration_error` over the 13 headline labels rather than
    from `calibration_summary`, which would additionally compute three unused labels, the
    pooled error and the Brier score on every one of the 1200 held-out evaluations. The
    arithmetic on the headline labels is the same.

    A label with no positive or no negative record in the labeled sample keeps the identity
    transform, because its likelihood is then unbounded and the minimizer runs to a bound
    that carries no information. Those cases are counted and reported.
    """

    generator = np.random.default_rng(seed_entropy)
    records = len(targets)
    headline_targets = np.ascontiguousarray(targets[:, headline_indices])
    headline_logits = np.ascontiguousarray(safe_logit(probabilities[:, headline_indices]))
    values: list[float] = []
    degenerate = 0
    for _ in range(repeats):
        fit_index = generator.choice(records, size=sample_size, replace=False)
        held_out = np.ones(records, dtype=bool)
        held_out[fit_index] = False
        fit_targets = headline_targets[fit_index]
        fit_logits = headline_logits[fit_index]
        out_targets = headline_targets[held_out]
        out_logits = headline_logits[held_out]
        label_error: list[float] = []
        for column in range(len(headline_indices)):
            positives = int(fit_targets[:, column].sum())
            if positives == 0 or positives == sample_size:
                degenerate += 1
                slope, offset = 1.0, 0.0
            else:
                slope, offset = fit_logit_affine(
                    fit_targets[:, column], fit_logits[:, column], fit_slope=True
                )
            label_error.append(
                expected_calibration_error(
                    out_targets[:, column],
                    expit(slope * out_logits[:, column] + offset),
                    bins=CALIBRATION_BINS,
                )
            )
        values.append(float(np.mean(np.asarray(label_error))))
    array = np.asarray(values, dtype=np.float64)
    lower, upper = np.percentile(array, [2.5, 97.5])
    return {
        "sample_size": int(sample_size),
        "repeats": int(repeats),
        "mean": float(array.mean()),
        "percentile_2_5": float(lower),
        "percentile_97_5": float(upper),
        "degenerate_label_fits": int(degenerate),
        "degenerate_label_fit_share": float(degenerate / (repeats * len(headline_indices))),
    }


def bootstrap_registered_resamples(
    targets: np.ndarray,
    auroc_probabilities: np.ndarray,
    ece_probabilities_by_rung: dict[str, np.ndarray],
    headline_indices: np.ndarray,
    group_inverse: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One pass over the registered resamples for per-label AUROC and per-label ECE.

    Transform parameters are already baked into `ece_probabilities_by_rung` and are never
    refitted inside a replicate. The AUROC replicates exist only to prove the resamples are
    the registered ones.
    """

    label_count = len(headline_indices)
    auroc = np.empty((replicates, label_count), dtype=np.float64)
    ece = {
        rung: np.empty((replicates, label_count), dtype=np.float64)
        for rung in ece_probabilities_by_rung
    }
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        for column, label_index in enumerate(headline_indices):
            auroc[start:end, column] = weighted_roc_auc_batch(
                targets[:, label_index], auroc_probabilities[:, label_index], weights
            )
            for rung, probabilities in ece_probabilities_by_rung.items():
                ece[rung][start:end, column] = weighted_expected_calibration_error_batch(
                    targets[:, label_index],
                    probabilities[:, label_index],
                    weights,
                    bins=CALIBRATION_BINS,
                )
    return auroc, ece


def registered_cohort_metrics(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Read the sealed cohort metrics with an exact float round trip.

    The CSV holds shortest round-trip decimal text. `float` recovers the original double
    from it; the pandas parser does not always, and rungs 1 and 2 are compared for exact
    equality.
    """

    registered: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            registered[(row["cohort"], row["architecture"])] = {
                "uncalibrated_macro_per_label_ece": float(row["uncalibrated_macro_per_label_ece"]),
                "calibrated_macro_per_label_ece": float(row["calibrated_macro_per_label_ece"]),
            }
    return registered


def source_prevalences(
    store_root: Path, label_order: list[str], counts_path: Path, bound_entry: dict
) -> tuple[np.ndarray, dict[str, object]]:
    """Label prevalence in PTB-XL training folds 1 to 8, from the digest-bound store.

    The sealed bound inputs cover the whole PTB-XL store, not only the fold-10 rows, so
    re-verifying `metadata.parquet` and `targets.npy` against them binds the training-fold
    prevalence to the same digests the evaluation used. The result is asserted equal to the
    frozen `harmonized_label_counts.csv` counts.
    """

    for filename, digest_key in (
        ("metadata.parquet", "metadata_sha256"),
        ("targets.npy", "targets_sha256"),
    ):
        expected = bound_entry.get(digest_key)
        if not isinstance(expected, str):
            raise RuntimeError(f"sealed bound inputs carry no {digest_key} for ptb_test")
        if sha256_file(store_root / filename) != expected:
            raise RuntimeError(f"PTB-XL store {filename} does not match the sealed {digest_key}")

    metadata = pd.read_parquet(store_root / "metadata.parquet")
    store_targets = np.load(store_root / "targets.npy")
    training = np.isin(metadata.strat_fold.to_numpy(), np.asarray(TRAINING_FOLDS))
    records = int(training.sum())
    positives = store_targets[training].sum(axis=0).astype(np.int64)

    counts = pd.read_csv(counts_path).set_index("label_key")
    frozen_records = {int(value) for value in counts.loc[label_order, "ptb_train_records"]}
    frozen_positives = counts.loc[label_order, "ptb_train_positives"].to_numpy(dtype=np.int64)
    if frozen_records != {records}:
        raise RuntimeError(
            f"training-fold record count {records} disagrees with the frozen label counts "
            f"{sorted(frozen_records)}"
        )
    if not np.array_equal(positives, frozen_positives):
        raise RuntimeError(
            "training-fold positive counts disagree with the frozen harmonized label counts"
        )

    provenance = {
        "source": "data/cache/waveforms/ptb_xl metadata.parquet strat_fold in 1..8 and targets.npy",
        "verified_against": "data/derived/preregistration/harmonized_label_counts.csv "
        "columns ptb_train_records and ptb_train_positives",
        "records": records,
        "folds": list(TRAINING_FOLDS),
    }
    return positives.astype(np.float64) / records, provenance


def validation_confusion(
    path: Path,
    choices: dict,
    label_order: list[str],
    architectures: tuple[str, ...],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Fold-9 true and false positive rates at the registered per-label F1 thresholds."""

    prediction = np.load(path)
    targets = np.asarray(prediction["targets"], dtype=np.uint8)
    record_indices = np.asarray(prediction["record_indices"], dtype=np.int64)
    if array_sha256(targets) != choices["validation_targets_sha256"]:
        raise RuntimeError("fold-9 validation targets do not match the sealed digest")
    if array_sha256(record_indices) != choices["validation_record_indices_sha256"]:
        raise RuntimeError("fold-9 validation record indices do not match the sealed digest")
    if len(record_indices) != int(choices["validation_records"]):
        raise RuntimeError("fold-9 validation record count does not match the sealed count")

    rates: dict[str, dict[str, tuple[float, float]]] = {}
    for architecture in architectures:
        probabilities = np.asarray(prediction[f"{architecture}_probabilities"], dtype=np.float64)
        thresholds = choices["architectures"][architecture]["thresholds_by_label"]
        rates[architecture] = {}
        for index, key in enumerate(label_order):
            flags = probabilities[:, index] >= float(thresholds[key])
            rates[architecture][key] = confusion_rates(targets[:, index], flags)
    return rates


def build_ladder(
    *,
    targets: np.ndarray,
    probabilities: np.ndarray,
    headline_indices: np.ndarray,
    headline_keys: list[str],
    label_order: list[str],
    temperature: float,
    source_prevalence: np.ndarray,
    target_prevalence: np.ndarray,
    thresholds: dict[str, float],
    validation_rates: dict[str, tuple[float, float]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, float]]]]:
    """Transformed probability matrices and fitted parameters for every ladder rung."""

    logits = safe_logit(probabilities)
    headline_source = source_prevalence[headline_indices]
    headline_target = target_prevalence[headline_indices]
    ones = np.ones(len(headline_indices), dtype=np.float64)
    zeros = np.zeros(len(headline_indices), dtype=np.float64)

    matrices: dict[str, np.ndarray] = {}
    parameters: dict[str, dict[str, dict[str, float]]] = {}

    matrices["uncalibrated"] = np.array(probabilities, dtype=np.float64, copy=True)
    parameters["uncalibrated"] = {}

    matrices["registered_temperature"] = sealed_temperature_probabilities(
        probabilities, temperature
    )
    parameters["registered_temperature"] = {
        key: {"temperature": float(temperature)} for key in headline_keys
    }

    global_temperature = fit_scalar_temperature(
        targets[:, headline_indices], probabilities[:, headline_indices]
    )
    matrices["oracle_global_temperature"] = apply_logit_affine(
        probabilities, headline_indices, ones / global_temperature, zeros
    )
    parameters["oracle_global_temperature"] = {
        key: {"temperature": float(global_temperature), "slope": float(1.0 / global_temperature)}
        for key in headline_keys
    }

    label_temperatures = np.array(
        [
            fit_scalar_temperature(targets[:, index], probabilities[:, index])
            for index in headline_indices
        ],
        dtype=np.float64,
    )
    matrices["oracle_per_label_temperature"] = apply_logit_affine(
        probabilities, headline_indices, 1.0 / label_temperatures, zeros
    )
    parameters["oracle_per_label_temperature"] = {
        key: {"temperature": float(value), "slope": float(1.0 / value)}
        for key, value in zip(headline_keys, label_temperatures, strict=True)
    }

    offsets = np.array(
        [
            fit_logit_affine(targets[:, index], logits[:, index], fit_slope=False)[1]
            for index in headline_indices
        ],
        dtype=np.float64,
    )
    matrices["oracle_per_label_offset"] = apply_logit_affine(
        probabilities, headline_indices, ones, offsets
    )
    parameters["oracle_per_label_offset"] = {
        key: {"slope": 1.0, "offset": float(value)}
        for key, value in zip(headline_keys, offsets, strict=True)
    }

    affine = [
        fit_logit_affine(targets[:, index], logits[:, index], fit_slope=True)
        for index in headline_indices
    ]
    affine_slopes = np.array([entry[0] for entry in affine], dtype=np.float64)
    affine_offsets = np.array([entry[1] for entry in affine], dtype=np.float64)
    matrices["oracle_per_label_affine"] = apply_logit_affine(
        probabilities, headline_indices, affine_slopes, affine_offsets
    )
    parameters["oracle_per_label_affine"] = {
        key: {"slope": float(slope), "offset": float(offset)}
        for key, slope, offset in zip(headline_keys, affine_slopes, affine_offsets, strict=True)
    }

    prevalence_offsets = np.array(
        [
            prior_shift_offset(source, target)
            for source, target in zip(headline_source, headline_target, strict=True)
        ],
        dtype=np.float64,
    )
    matrices["prevalence_only_intercept"] = apply_logit_affine(
        probabilities, headline_indices, ones, prevalence_offsets
    )
    parameters["prevalence_only_intercept"] = {
        key: {
            "slope": 1.0,
            "offset": float(offset),
            "source_prevalence": float(source),
            "target_prevalence": float(target),
        }
        for key, offset, source, target in zip(
            headline_keys, prevalence_offsets, headline_source, headline_target, strict=True
        )
    }

    sld_estimates = np.empty(len(headline_indices), dtype=np.float64)
    sld_iterations = np.empty(len(headline_indices), dtype=np.int64)
    for column, index in enumerate(headline_indices):
        sld_estimates[column], sld_iterations[column] = sld_em_prevalence(
            logits[:, index], float(source_prevalence[index])
        )
    sld_offsets = np.array(
        [
            prior_shift_offset(source, estimate)
            for source, estimate in zip(headline_source, sld_estimates, strict=True)
        ],
        dtype=np.float64,
    )
    matrices["sld_em"] = apply_logit_affine(probabilities, headline_indices, ones, sld_offsets)
    parameters["sld_em"] = {
        key: {
            "slope": 1.0,
            "offset": float(offset),
            "source_prevalence": float(source),
            "estimated_target_prevalence": float(estimate),
            "true_target_prevalence": float(true_value),
            "prevalence_absolute_error": float(abs(estimate - true_value)),
            "em_iterations": int(iterations),
        }
        for key, offset, source, estimate, true_value, iterations in zip(
            headline_keys,
            sld_offsets,
            headline_source,
            sld_estimates,
            headline_target,
            sld_iterations,
            strict=True,
        )
    }

    bbse_estimates = np.empty(len(headline_indices), dtype=np.float64)
    flagged_rates = np.empty(len(headline_indices), dtype=np.float64)
    for column, index in enumerate(headline_indices):
        key = label_order[index]
        threshold = float(thresholds[key])
        flagged_rates[column] = float((probabilities[:, index] >= threshold).mean())
        true_positive_rate, false_positive_rate = validation_rates[key]
        bbse_estimates[column] = bbse_hard_prevalence(
            true_positive_rate, false_positive_rate, flagged_rates[column]
        )
    bbse_offsets = np.array(
        [
            prior_shift_offset(source, estimate)
            for source, estimate in zip(headline_source, bbse_estimates, strict=True)
        ],
        dtype=np.float64,
    )
    matrices["bbse_hard"] = apply_logit_affine(probabilities, headline_indices, ones, bbse_offsets)
    parameters["bbse_hard"] = {
        key: {
            "slope": 1.0,
            "offset": float(offset),
            "source_prevalence": float(source),
            "estimated_target_prevalence": float(estimate),
            "true_target_prevalence": float(true_value),
            "prevalence_absolute_error": float(abs(estimate - true_value)),
            "validation_true_positive_rate": float(validation_rates[key][0]),
            "validation_false_positive_rate": float(validation_rates[key][1]),
            "target_flagged_rate": float(flagged),
            "threshold": float(thresholds[key]),
        }
        for key, offset, source, estimate, true_value, flagged in zip(
            headline_keys,
            bbse_offsets,
            headline_source,
            bbse_estimates,
            headline_target,
            flagged_rates,
            strict=True,
        )
    }
    return matrices, parameters


def _row(
    section: str,
    *,
    cohort: str = "",
    architecture: str = "",
    rung: str = "",
    label_key: str = "",
    diagnosis: str = "",
    sample_size: object = "",
    metric: str,
    value: float,
    lower_95: float = float("nan"),
    upper_95: float = float("nan"),
) -> dict[str, object]:
    return {
        "section": section,
        "cohort": cohort,
        "architecture": architecture,
        "rung": rung,
        "label_key": label_key,
        "diagnosis": diagnosis,
        "sample_size": sample_size,
        "metric": metric,
        "value": value,
        "lower_95": lower_95,
        "upper_95": upper_95,
    }


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
    headline_columns = np.arange(len(headline_indices), dtype=np.int64)

    registered_points = registered_cohort_metrics(args.output_root / "cohort_metrics.csv")
    registered_bootstrap = np.load(
        args.registered_bootstrap or (args.output_root / "bootstrap_distributions.npz")
    )
    bound_entry = choices["bound_inputs"]["cohorts"]["ptb_test"]
    source_prevalence, prevalence_provenance = source_prevalences(
        args.store_root / analysis.COHORT_STORE_NAMES["ptb_test"],
        label_order,
        args.label_counts,
        bound_entry,
    )
    validation_rates = validation_confusion(
        args.validation_ensembles, choices, label_order, analysis.ARCHITECTURES
    )

    rows: list[dict[str, object]] = []
    cross_checks: list[dict[str, object]] = []
    reproduction: list[dict[str, object]] = []
    ladder_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    prevalence_accuracy: dict[str, dict[str, dict[str, object]]] = {}
    equal_mass: dict[str, dict[str, dict[str, float]]] = {}
    floors: dict[str, dict[str, float | dict[str, float]]] = {}
    recovery: dict[str, dict[str, list[dict[str, float]]]] = {}

    for cohort_index, cohort in enumerate(analysis.COHORTS):
        cohort_data = analysis.load_verified_cohort(
            cohort=cohort,
            prediction_root=args.prediction_root,
            store_root=args.store_root,
            evaluation_seal=evaluation_seal,
            choices=choices,
            label_count=len(label_order),
        )
        targets = cohort_data["targets"]
        group_inverse = cluster_inverse(cohort, cohort_data["metadata"])
        target_prevalence = targets.mean(axis=0).astype(np.float64)
        ladder_summary[cohort] = {}
        prevalence_accuracy[cohort] = {}
        equal_mass[cohort] = {}
        recovery[cohort] = {}

        constant_true = constant_predictor_matrix(target_prevalence, len(targets))
        constant_source = constant_predictor_matrix(source_prevalence, len(targets))
        true_floor = calibration_summary(
            targets, constant_true, headline_indices, bins=CALIBRATION_BINS
        )
        source_floor = calibration_summary(
            targets, constant_source, headline_indices, bins=CALIBRATION_BINS
        )
        floors[cohort] = {
            "true_target_prevalence": {
                metric: float(true_floor[metric]) for metric in LADDER_METRICS
            },
            "source_prevalence": {metric: float(source_floor[metric]) for metric in LADDER_METRICS},
            "mean_absolute_prevalence_shift": float(
                np.mean(
                    np.abs(
                        target_prevalence[headline_indices] - source_prevalence[headline_indices]
                    )
                )
            ),
        }
        for name, summary in (
            ("true_target_prevalence", true_floor),
            ("source_prevalence", source_floor),
        ):
            for metric in LADDER_METRICS:
                rows.append(
                    _row(
                        "constant_predictor_floor",
                        cohort=cohort,
                        rung=name,
                        metric=metric,
                        value=float(summary[metric]),
                    )
                )
        for column, index in enumerate(headline_indices):
            key = headline_keys[column]
            rows.append(
                _row(
                    "prevalence",
                    cohort=cohort,
                    label_key=key,
                    diagnosis=diagnosis[key],
                    metric="target_prevalence",
                    value=float(target_prevalence[index]),
                )
            )
            rows.append(
                _row(
                    "prevalence",
                    cohort=cohort,
                    label_key=key,
                    diagnosis=diagnosis[key],
                    metric="source_prevalence",
                    value=float(source_prevalence[index]),
                )
            )

        for architecture_index, architecture in enumerate(analysis.ARCHITECTURES):
            probabilities = cohort_data["probabilities"][architecture]
            architecture_choices = choices["architectures"][architecture]
            temperature = float(architecture_choices["temperature"])
            matrices, parameters = build_ladder(
                targets=targets,
                probabilities=probabilities,
                headline_indices=headline_indices,
                headline_keys=headline_keys,
                label_order=label_order,
                temperature=temperature,
                source_prevalence=source_prevalence,
                target_prevalence=target_prevalence,
                thresholds=architecture_choices["thresholds_by_label"],
                validation_rates=validation_rates[architecture],
            )

            if not np.array_equal(
                matrices["registered_temperature"], cohort_data["calibrated"][architecture]
            ):
                raise RuntimeError(
                    f"rung registered_temperature for {cohort}/{architecture} does not reproduce "
                    "the sealed calibrated probabilities bit-for-bit"
                )

            summaries = {
                rung: calibration_summary(
                    targets, matrices[rung], headline_indices, bins=CALIBRATION_BINS
                )
                for rung in LADDER_RUNGS
            }
            registered = registered_points[(cohort, architecture)]
            for rung, registered_key in (
                ("uncalibrated", "uncalibrated_macro_per_label_ece"),
                ("registered_temperature", "calibrated_macro_per_label_ece"),
            ):
                recomputed = float(summaries[rung]["macro_per_label_ece"])
                expected = registered[registered_key]
                matches = recomputed == expected
                reproduction.append(
                    {
                        "cohort": cohort,
                        "architecture": architecture,
                        "rung": rung,
                        "registered_column": registered_key,
                        "registered_value": expected,
                        "recomputed_value": recomputed,
                        "exactly_equal": bool(matches),
                    }
                )
                if not matches:
                    raise RuntimeError(
                        f"rung {rung} for {cohort}/{architecture} gives "
                        f"{recomputed!r} but results/cohort_metrics.csv column {registered_key} "
                        f"holds {expected!r}; the ladder is not on the registered estimator"
                    )

            auroc_replicates, ece_replicates = bootstrap_registered_resamples(
                targets,
                probabilities,
                {rung: matrices[rung] for rung in INTERVAL_RUNGS},
                headline_indices,
                group_inverse,
                replicates=args.bootstrap_replicates,
                seed=analysis.COHORT_SEEDS[cohort],
                batch_size=args.bootstrap_batch_size,
            )
            rebuilt = macro_of_per_label(auroc_replicates, headline_columns)
            reference = registered_bootstrap[f"{cohort}__{architecture}"]
            identical = bool(np.array_equal(rebuilt, reference))
            largest = float(np.max(np.abs(rebuilt - reference)))
            macro_ece = {
                rung: macro_of_per_label(values, headline_columns)
                for rung, values in ece_replicates.items()
            }
            registered_ece_match = {}
            for rung, registered_variant in (
                ("uncalibrated", "uncalibrated"),
                ("registered_temperature", "calibrated"),
            ):
                registered_ece_match[rung] = bool(
                    np.array_equal(
                        macro_ece[rung],
                        registered_bootstrap[
                            f"{cohort}__{architecture}__{registered_variant}__macro_per_label_ece"
                        ],
                    )
                )
            cross_checks.append(
                {
                    "cohort": cohort,
                    "architecture": architecture,
                    "bit_identical_to_registered_macro": identical,
                    "largest_absolute_difference": largest,
                    "bit_identical_macro_ece_distributions": registered_ece_match,
                }
            )
            if not identical:
                raise RuntimeError(
                    f"rebuilt macro-AUROC replicates for {cohort}/{architecture} do not match "
                    f"the registered distribution (largest difference {largest:.3e}); "
                    "the resamples are not the registered ones and no interval may be published"
                )
            for rung, matched in registered_ece_match.items():
                if not matched:
                    raise RuntimeError(
                        f"rebuilt macro per-label ECE replicates for rung {rung} on "
                        f"{cohort}/{architecture} do not match the registered distribution; "
                        "the resamples are not the registered ones"
                    )

            intervals = {rung: percentile_interval(values) for rung, values in macro_ece.items()}
            ladder_summary[cohort][architecture] = {}
            for rung in LADDER_RUNGS:
                entry = {metric: float(summaries[rung][metric]) for metric in LADDER_METRICS}
                if rung in intervals:
                    entry["macro_per_label_ece_lower_95"] = intervals[rung][0]
                    entry["macro_per_label_ece_upper_95"] = intervals[rung][1]
                ladder_summary[cohort][architecture][rung] = entry
                for metric in LADDER_METRICS:
                    bounds = (
                        intervals[rung]
                        if rung in intervals and metric == "macro_per_label_ece"
                        else (float("nan"), float("nan"))
                    )
                    rows.append(
                        _row(
                            "ladder",
                            cohort=cohort,
                            architecture=architecture,
                            rung=rung,
                            metric=metric,
                            value=float(summaries[rung][metric]),
                            lower_95=bounds[0],
                            upper_95=bounds[1],
                        )
                    )
                for key, values in parameters[rung].items():
                    for name, value in values.items():
                        rows.append(
                            _row(
                                "rung_parameters",
                                cohort=cohort,
                                architecture=architecture,
                                rung=rung,
                                label_key=key,
                                diagnosis=diagnosis[key],
                                metric=name,
                                value=float(value),
                            )
                        )

            uncalibrated_per_label = np.asarray(summaries["uncalibrated"]["per_label_ece"])[
                headline_indices
            ]
            total = float(uncalibrated_per_label.sum())
            for column, key in enumerate(headline_keys):
                rows.append(
                    _row(
                        "per_label_ece_share",
                        cohort=cohort,
                        architecture=architecture,
                        rung="uncalibrated",
                        label_key=key,
                        diagnosis=diagnosis[key],
                        metric="per_label_ece",
                        value=float(uncalibrated_per_label[column]),
                    )
                )
                rows.append(
                    _row(
                        "per_label_ece_share",
                        cohort=cohort,
                        architecture=architecture,
                        rung="uncalibrated",
                        label_key=key,
                        diagnosis=diagnosis[key],
                        metric="macro_share",
                        value=float(uncalibrated_per_label[column] / total),
                    )
                )

            prevalence_accuracy[cohort][architecture] = {}
            for rung in ESTIMATED_PREVALENCE_RUNGS:
                errors = np.array(
                    [parameters[rung][key]["prevalence_absolute_error"] for key in headline_keys]
                )
                prevalence_accuracy[cohort][architecture][rung] = {
                    "mean_absolute_error": float(errors.mean()),
                    "median_absolute_error": float(np.median(errors)),
                    "maximum_absolute_error": float(errors.max()),
                    "per_label": {
                        key: {
                            "estimated_target_prevalence": float(
                                parameters[rung][key]["estimated_target_prevalence"]
                            ),
                            "true_target_prevalence": float(
                                parameters[rung][key]["true_target_prevalence"]
                            ),
                            "absolute_error": float(
                                parameters[rung][key]["prevalence_absolute_error"]
                            ),
                        }
                        for key in headline_keys
                    },
                }
                rows.append(
                    _row(
                        "prevalence_estimation",
                        cohort=cohort,
                        architecture=architecture,
                        rung=rung,
                        metric="mean_absolute_error",
                        value=float(errors.mean()),
                    )
                )

            equal_mass[cohort][architecture] = {}
            for rung in ("uncalibrated", "registered_temperature"):
                equal_width = float(summaries[rung]["macro_per_label_ece"])
                mass = equal_mass_macro_per_label_ece(
                    targets, matrices[rung], headline_indices, bins=CALIBRATION_BINS
                )
                equal_mass[cohort][architecture][rung] = {
                    "equal_width_macro_per_label_ece": equal_width,
                    "equal_mass_macro_per_label_ece": mass,
                    "difference": mass - equal_width,
                }
                for metric, value in (
                    ("macro_per_label_ece_equal_width", equal_width),
                    ("macro_per_label_ece_equal_mass", mass),
                ):
                    rows.append(
                        _row(
                            "equal_mass_binning",
                            cohort=cohort,
                            architecture=architecture,
                            rung=rung,
                            metric=metric,
                            value=value,
                        )
                    )

            recovery[cohort][architecture] = []
            for sample_size in RECOVERY_SIZES:
                entry = held_out_platt_recovery(
                    targets,
                    probabilities,
                    headline_indices,
                    sample_size=sample_size,
                    repeats=RECOVERY_REPEATS,
                    seed_entropy=recovery_seed(cohort_index, architecture_index, sample_size),
                )
                recovery[cohort][architecture].append(entry)
                for metric in ("mean", "percentile_2_5", "percentile_97_5"):
                    rows.append(
                        _row(
                            "recovery_curve",
                            cohort=cohort,
                            architecture=architecture,
                            rung="held_out_platt",
                            sample_size=sample_size,
                            metric=metric,
                            value=float(entry[metric]),
                        )
                    )
            print(f"completed {cohort}/{architecture}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    table_path = args.output_root / "exploratory_calibration_ladder.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)

    summary = {
        "stage": "post-seal exploratory addition",
        "stage_date": EXPLORATORY_STAGE_DATE,
        "changes_registered_quantities": False,
        "registered_result_status": (
            "not under revision; the registered fold-9 temperature genuinely does not "
            "reduce external calibration error, and no rung here is a registered method"
        ),
        "estimator": "15 equal-width bins per label, macro-averaged over 13 headline labels",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_batch_size": args.bootstrap_batch_size,
        "bootstrap_seeds": dict(analysis.COHORT_SEEDS),
        "bootstrap_parameter_handling": (
            "transform parameters fixed at full-cohort fitted values inside every replicate; "
            "no refitting inside the bootstrap"
        ),
        "interval_rungs": list(INTERVAL_RUNGS),
        "interval_type": "percentile",
        "ptb_cluster_unit": "patient_id",
        "external_cluster_unit": "record",
        "oracle_rungs": list(ORACLE_RUNGS),
        "deployable_rungs": ["prevalence_only_intercept", "sld_em", "bbse_hard"],
        "deployable_rung_notes": (
            "prevalence_only_intercept needs the target prevalence and no target scores; "
            "sld_em and bbse_hard need neither target labels nor target prevalence"
        ),
        "registered_reproduction": reproduction,
        "registered_resample_cross_check": cross_checks,
        "source_prevalence_provenance": prevalence_provenance,
        "ladder": ladder_summary,
        "prevalence_estimation": prevalence_accuracy,
        "equal_mass_comparison": equal_mass,
        "constant_predictor_floors": floors,
        "recovery_curve": recovery,
        "recovery_curve_seed": {
            "base_seed": RECOVERY_SEED,
            "entropy": "[base_seed, cohort_index, architecture_index, sample_size]",
            "repeats": RECOVERY_REPEATS,
            "sample_sizes": list(RECOVERY_SIZES),
        },
        "outputs": [
            table_path.name,
            "exploratory_calibration_ladder.json",
        ],
    }
    summary_path = args.output_root / "exploratory_calibration_ladder.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"wrote {table_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
