#!/usr/bin/env python3
"""Diagnose why the bootstrap intervals for expected calibration error exclude their point.

Twenty-seven percentile intervals in the completed analysis exclude their own point
estimate, all of them expected-calibration-error quantities and all of them carrying
the PTB-XL fold-10 term. Twelve are cohort-level, five sitting above the point and
seven below it as shift deltas in which the internal term enters negatively; the other
fifteen are PTB-XL fold-10 subgroup rows sitting above their point. This script tests
one explanation for the whole set: expected
calibration error is a positively biased statistic whose bias grows as the effective
sample shrinks, and a nonparametric cluster bootstrap resample holds only about
sixty-three percent distinct clusters, so every replicate is computed on a smaller
effective sample than the full cohort and lands high.

The test has three parts per cohort and architecture. It reproduces the registered
bootstrap distribution and cross-checks its percentiles against the sealed result
file. It draws subsamples without replacement across a sweep of cluster fractions,
which isolates sample size from resampling with replacement. It counts how many
distinct clusters the registered resamples actually contain.

This is an exploratory diagnostic. It recomputes nothing that is reported, writes to
its own file, and leaves every registered quantity untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ecg_clinical.bootstrap import (
    CALIBRATION_BINS,
    bootstrap_record_weight_batches,
    paired_bootstrap_cohort_statistics,
    percentile_interval,
)
from ecg_clinical.data import WaveformStore
from ecg_clinical.metrics import calibration_summary, expected_calibration_error

COHORTS = ("ptb_test", "chapman_shaoxing", "ningbo")
ARCHITECTURES = ("xresnet1d101", "s4d")
COHORT_STORE_NAMES = {
    "ptb_test": "ptb_xl",
    "chapman_shaoxing": "chapman_shaoxing",
    "ningbo": "ningbo",
}
COHORT_SEEDS = {"ptb_test": 20260716, "chapman_shaoxing": 20260717, "ningbo": 20260718}
CLUSTER_UNITS = {"ptb_test": "patient_id", "chapman_shaoxing": "record", "ningbo": "record"}
DIAGNOSTIC_METRIC = "macro_per_label_ece"
DIAGNOSTIC_VARIANT = "uncalibrated"
# Offset 900 keeps the subsample draws clear of the registered cohort seeds and of the
# subgroup offsets of 100 and 200 that the analysis script already spends.
SUBSAMPLE_SEED_OFFSET = 900
SUBSAMPLE_FRACTIONS = (0.2, 0.4, 0.632, 0.8, 1.0)
MECHANISM_FRACTION = 0.632
PERCENTILE_TOLERANCE = 1e-9
POINT_TOLERANCE = 1e-12
NOTE = (
    "Exploratory diagnostic of bootstrap behaviour for expected calibration error; it is "
    "not a registered analysis and it changes no reported quantity."
)


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
    parser.add_argument("--uncertainty", type=Path, default=Path("results/uncertainty.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/ece_bootstrap_bias_diagnostic.json")
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=100)
    parser.add_argument("--subsample-draws", type=int, default=400)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def macro_per_label_ece(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Average the registered per-label calibration error over the supplied label columns.

    The caller restricts both arrays to the headline labels beforehand, so this is the
    same quantity `calibration_summary` reports under `macro_per_label_ece`, computed
    through the same binning function rather than a private reimplementation.
    """

    return float(
        np.mean(
            [
                expected_calibration_error(
                    targets[:, column], probabilities[:, column], bins=CALIBRATION_BINS
                )
                for column in range(targets.shape[1])
            ]
        )
    )


def cohort_group_inverse(cohort: str, store_root: Path, record_indices: np.ndarray) -> np.ndarray:
    """Rebuild the registered cluster assignment: PTB-XL patients, records elsewhere."""

    if cohort != "ptb_test":
        return np.arange(len(record_indices), dtype=np.int64)
    store = WaveformStore(store_root / COHORT_STORE_NAMES[cohort])
    metadata = store.metadata.iloc[record_indices].reset_index(drop=True)
    _, group_inverse = np.unique(metadata.patient_id.astype(str), return_inverse=True)
    return np.asarray(group_inverse, dtype=np.int64)


def distinct_cluster_summary(
    group_inverse: np.ndarray, *, replicates: int, seed: int, batch_size: int
) -> dict[str, float]:
    """Count the distinct clusters the registered resamples actually contain.

    The weight batches are regenerated from the same seed and batch size the analysis
    used, so these are the very draws the reported intervals rest on. A cluster is
    present in a replicate when its weight is nonzero, and one representative record
    per cluster carries that weight.
    """

    clusters = int(group_inverse.max()) + 1
    _, representatives = np.unique(group_inverse, return_index=True)
    present = np.empty(replicates, dtype=np.int64)
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        present[start:end] = (weights[:, representatives] > 0).sum(axis=1)
    return {
        "clusters": clusters,
        "mean_distinct_clusters": float(present.mean()),
        "mean_distinct_fraction": float(present.mean() / clusters),
        "minimum_distinct_clusters": int(present.min()),
        "maximum_distinct_clusters": int(present.max()),
    }


def subsample_curve(
    targets: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    group_inverse: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, list[dict[str, float]]]:
    """Trace macro calibration error against cluster count, sampling without replacement.

    Drawing clusters without replacement changes the sample size and nothing else, so
    a rising mean as the fraction falls is the bias term on its own, separated from
    whatever resampling with replacement contributes. Both architectures see the same
    draws at every fraction, and the full-cluster fraction returns the point estimate
    by construction.
    """

    clusters = int(group_inverse.max()) + 1
    curve: dict[str, list[dict[str, float]]] = {name: [] for name in probabilities_by_model}
    for position, fraction in enumerate(SUBSAMPLE_FRACTIONS):
        size = int(round(fraction * clusters))
        generator = np.random.default_rng(seed + position)
        values = {name: np.empty(draws, dtype=np.float64) for name in probabilities_by_model}
        record_counts = np.empty(draws, dtype=np.int64)
        for draw in range(draws):
            keep = np.zeros(clusters, dtype=bool)
            keep[generator.choice(clusters, size=size, replace=False)] = True
            records = np.flatnonzero(keep[group_inverse])
            record_counts[draw] = len(records)
            subset_targets = targets[records]
            for name, probabilities in probabilities_by_model.items():
                values[name][draw] = macro_per_label_ece(subset_targets, probabilities[records])
        for name in probabilities_by_model:
            curve[name].append(
                {
                    "fraction": float(fraction),
                    "clusters": int(size),
                    "mean_records": float(record_counts.mean()),
                    "mean_ece": float(values[name].mean()),
                    "median_ece": float(np.median(values[name])),
                }
            )
    return curve


def bootstrap_summary(distribution: np.ndarray, point: float) -> dict[str, float]:
    """Describe one reproduced bootstrap distribution relative to its point estimate."""

    finite = distribution[np.isfinite(distribution)]
    lower, upper = percentile_interval(distribution)
    return {
        "replicates": int(len(distribution)),
        "finite_replicates": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "percentile_2_5": lower,
        "percentile_97_5": upper,
        "fraction_above_point": float((finite > point).mean()),
    }


def cross_check_registered(
    registered: dict[str, float], point: float, summary: dict[str, float], label: str
) -> dict[str, float]:
    """Refuse to report unless the reproduced distribution is the registered one.

    A diagnosis of the reported intervals is worthless if it is run against a
    different set of resamples, so the reproduced percentiles and point estimate are
    matched against the sealed result file before anything else is believed.
    """

    differences = {
        "percentile_2_5": abs(summary["percentile_2_5"] - float(registered["lower_95"])),
        "percentile_97_5": abs(summary["percentile_97_5"] - float(registered["upper_95"])),
    }
    largest = max(differences.values())
    point_difference = abs(point - float(registered["point"]))
    if largest > PERCENTILE_TOLERANCE or point_difference > POINT_TOLERANCE:
        raise RuntimeError(
            f"reproduced bootstrap for {label} does not match results/uncertainty.json: "
            f"percentile difference {largest:.3e} against tolerance "
            f"{PERCENTILE_TOLERANCE:.0e}, point difference {point_difference:.3e} against "
            f"tolerance {POINT_TOLERANCE:.0e}"
        )
    return {
        "registered_point": float(registered["point"]),
        "registered_lower_95": float(registered["lower_95"]),
        "registered_upper_95": float(registered["upper_95"]),
        "largest_percentile_difference": float(largest),
        "point_difference": float(point_difference),
        "tolerance": PERCENTILE_TOLERANCE,
        "passed": True,
    }


def run_diagnostic(args: argparse.Namespace, root: Path) -> dict[str, object]:
    prediction_root = resolve(root, args.prediction_root)
    store_root = resolve(root, args.store_root)
    label_manifest = json.loads(resolve(root, args.label_manifest).read_text())
    registered_calibration = json.loads(resolve(root, args.uncertainty).read_text())["calibration"]
    label_keys = [label["label_key"] for label in label_manifest["labels"]]
    headline_indices = np.asarray(
        [label_keys.index(key) for key in label_manifest["headline_label_keys"]], dtype=np.int64
    )

    payload: dict[str, object] = {
        "note": NOTE,
        "hypothesis": (
            "Expected calibration error is positively biased and the bias grows as the "
            "effective sample falls, so bootstrap replicates holding roughly 63.2 percent "
            "distinct clusters sit above the full-sample point estimate."
        ),
        "settings": {
            "metric": DIAGNOSTIC_METRIC,
            "calibration_variant": DIAGNOSTIC_VARIANT,
            "calibration_bins": CALIBRATION_BINS,
            "headline_labels": int(len(headline_indices)),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_batch_size": args.bootstrap_batch_size,
            "cohort_seeds": COHORT_SEEDS,
            "subsample_draws": args.subsample_draws,
            "subsample_fractions": list(SUBSAMPLE_FRACTIONS),
            "subsample_seed_offset": SUBSAMPLE_SEED_OFFSET,
            "percentile_tolerance": PERCENTILE_TOLERANCE,
        },
        "cohorts": {},
    }

    for cohort in COHORTS:
        prediction = np.load(prediction_root / cohort / "ensembles.npz")
        targets = np.asarray(prediction["targets"], dtype=np.uint8)
        record_indices = np.asarray(prediction["record_indices"], dtype=np.int64)
        probabilities_by_model = {
            architecture: np.asarray(prediction[f"{architecture}_probabilities"], dtype=np.float64)
            for architecture in ARCHITECTURES
        }
        calibrated_by_model = {
            architecture: np.asarray(
                prediction[f"{architecture}_calibrated_probabilities"], dtype=np.float64
            )
            for architecture in ARCHITECTURES
        }
        group_inverse = cohort_group_inverse(cohort, store_root, record_indices)

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
        distinct = distinct_cluster_summary(
            group_inverse,
            replicates=args.bootstrap_replicates,
            seed=COHORT_SEEDS[cohort],
            batch_size=args.bootstrap_batch_size,
        )
        headline_targets = targets[:, headline_indices]
        headline_probabilities = {
            architecture: probabilities_by_model[architecture][:, headline_indices]
            for architecture in ARCHITECTURES
        }
        curve = subsample_curve(
            headline_targets,
            headline_probabilities,
            group_inverse,
            seed=COHORT_SEEDS[cohort] + SUBSAMPLE_SEED_OFFSET,
            draws=args.subsample_draws,
        )

        cohort_entry: dict[str, object] = {
            "records": int(len(record_indices)),
            "cluster_unit": CLUSTER_UNITS[cohort],
            "registered_resample_clusters": distinct,
            "architectures": {},
        }
        for architecture in ARCHITECTURES:
            summary = calibration_summary(
                targets, probabilities_by_model[architecture], headline_indices
            )
            point = float(summary[DIAGNOSTIC_METRIC])
            direct = macro_per_label_ece(headline_targets, headline_probabilities[architecture])
            if abs(direct - point) > POINT_TOLERANCE:
                raise RuntimeError(
                    f"headline-restricted macro ECE for {cohort}/{architecture} differs from "
                    f"calibration_summary by {abs(direct - point):.3e}"
                )
            distribution = cohort_bootstrap[
                f"{architecture}__{DIAGNOSTIC_VARIANT}__{DIAGNOSTIC_METRIC}"
            ]
            reproduced = bootstrap_summary(distribution, point)
            check = cross_check_registered(
                registered_calibration[cohort][architecture][DIAGNOSTIC_VARIANT][DIAGNOSTIC_METRIC],
                point,
                reproduced,
                f"{cohort}/{architecture}",
            )
            cohort_entry["architectures"][architecture] = {
                "point": point,
                "bootstrap": reproduced,
                "registered_cross_check": check,
                "subsample_curve": curve[architecture],
            }
        payload["cohorts"][cohort] = cohort_entry

    return payload


def curve_value(rows: list[dict[str, float]], fraction: float, field: str) -> float:
    for row in rows:
        if row["fraction"] == fraction:
            return float(row[field])
    raise KeyError(f"subsample fraction {fraction} was not swept")


def print_summary(payload: dict[str, object]) -> None:
    print(payload["note"])
    print()
    header = (
        f"{'cohort':<17}{'architecture':<14}{'point':>10}{'boot mean':>11}{'2.5%':>10}"
        f"{'97.5%':>10}{'above':>8}{'sub .632':>10}{'distinct':>10}"
    )
    print(header)
    print("-" * len(header))
    for cohort, entry in payload["cohorts"].items():
        distinct_fraction = entry["registered_resample_clusters"]["mean_distinct_fraction"]
        for architecture, values in entry["architectures"].items():
            bootstrap = values["bootstrap"]
            print(
                f"{cohort:<17}{architecture:<14}"
                f"{values['point']:>10.5f}{bootstrap['mean']:>11.5f}"
                f"{bootstrap['percentile_2_5']:>10.5f}{bootstrap['percentile_97_5']:>10.5f}"
                f"{bootstrap['fraction_above_point']:>8.3f}"
                f"{curve_value(values['subsample_curve'], MECHANISM_FRACTION, 'mean_ece'):>10.5f}"
                f"{distinct_fraction:>10.4f}"
            )
    print()
    fractions = list(SUBSAMPLE_FRACTIONS)
    curve_header = f"{'cohort':<17}{'architecture':<14}" + "".join(
        f"{fraction:>11.3f}" for fraction in fractions
    )
    print("mean macro ECE over cluster subsamples drawn without replacement")
    print(curve_header)
    print("-" * len(curve_header))
    for cohort, entry in payload["cohorts"].items():
        for architecture, values in entry["architectures"].items():
            cells = "".join(
                f"{curve_value(values['subsample_curve'], fraction, 'mean_ece'):>11.5f}"
                for fraction in fractions
            )
            print(f"{cohort:<17}{architecture:<14}{cells}")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    payload = run_diagnostic(args, root)
    output = resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print_summary(payload)
    print()
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
