#!/usr/bin/env python3
"""Attach bootstrap intervals to the per-label discrimination estimates.

POST-SEAL EXPLORATORY ADDITION, 2026-07-25. This stage was not part of the frozen
analysis. It was written after the evaluation seal was broken and after the registered
results existed, and it is kept separate from `analyze_results.py` for that reason.

What it adds and why. Per-label AUROC is a registered secondary metric, but the frozen
analysis emitted it as a bare point estimate while bootstrapping every macro quantity.
The writeup leads with the per-label transfer split rather than with the aggregate, so
the claim carrying the most weight was the one carrying no uncertainty. Supplying the
interval can only qualify a reading, never strengthen one, which is why adding it after
the fact is defensible where changing an estimator after the fact would not be.

What it does not do. It changes no registered quantity, rewrites no sealed artifact, and
re-runs no inference. It recomputes nothing that `analyze_results.py` reports; it writes
to its own files.

Faithfulness to the registered design. The resamples are not a new bootstrap. This stage
rebuilds the registered cluster resamples from the same per-cohort seeds, the same
cluster units (patient for PTB-XL, record externally), the same replicate count and the
same batch size, then proves it: macro-averaging the per-label replicate AUROCs must
reproduce the registered macro-AUROC distribution in `results/bootstrap_distributions.npz`
bit-for-bit. The run aborts if that check fails, so an interval published here cannot
come from a different resample than the registered macro estimate it sits beside.
Per-label shift deltas are formed replicate-wise from independent cohort draws and paired
with the true point difference, exactly as the registered macro shift delta is.

Multiplicity. These intervals are marginal and uncorrected, matching the registered plan,
which states that no multiplicity-adjusted tests were to be run. Thirteen labels by two
external cohorts is twenty-six simultaneous drops, and the writeup ranks them and
discusses the extremes, so the intervals should be read as descriptive rather than as a
family of tests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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

EXPLORATORY_STAGE_DATE = "2026-07-25"


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


def per_label_bootstrap(
    targets: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    group_inverse: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Per-label and micro AUROC replicate distributions on the registered resamples."""

    label_count = targets.shape[1]
    output = {
        name: np.empty((replicates, label_count), dtype=np.float64)
        for name in probabilities_by_model
    }
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        for name, probabilities in probabilities_by_model.items():
            for label_index in range(label_count):
                output[name][start:end, label_index] = weighted_roc_auc_batch(
                    targets[:, label_index], probabilities[:, label_index], weights
                )
    return output


def micro_bootstrap(
    targets: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
    label_indices: np.ndarray,
    group_inverse: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Micro-AUROC replicate distributions over the headline labels.

    The micro estimand pools every record-by-label pair, so a record's cluster weight
    has to be repeated once per headline label to keep the resampling unit intact. That
    tiled weight matrix is thirteen times the width of the per-label one, so this pass
    uses a smaller batch to bound peak memory on the largest cohort.
    """

    batch_size = min(batch_size, 25)
    output = {name: np.empty(replicates, dtype=np.float64) for name in probabilities_by_model}
    flat_targets = targets[:, label_indices].reshape(-1)
    flat_probabilities = {
        name: probabilities[:, label_indices].reshape(-1)
        for name, probabilities in probabilities_by_model.items()
    }
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        tiled = np.repeat(weights, len(label_indices), axis=1)
        for name in probabilities_by_model:
            output[name][start:end] = weighted_roc_auc_batch(
                flat_targets, flat_probabilities[name], tiled
            )
    return output


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


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    verify_preregistration_seal(root)
    verify_evaluation_seal(root)

    analysis = load_analysis_module()
    evaluation_seal = json.loads((root / "results/EVALUATION_SEAL.json").read_text())
    choices = json.loads((root / evaluation_seal["choices_path"]).read_text())

    manifest = json.loads(args.labels.read_text())
    label_order = [entry["label_key"] for entry in manifest["labels"]]
    diagnosis = {entry["label_key"]: entry["diagnosis"] for entry in manifest["labels"]}
    headline_keys = list(manifest["headline_label_keys"])
    headline_indices = np.array([label_order.index(key) for key in headline_keys], dtype=np.int64)

    registered_path = args.registered_bootstrap or (
        args.output_root / "bootstrap_distributions.npz"
    )
    registered = np.load(registered_path)

    per_label: dict[str, dict[str, np.ndarray]] = {}
    micro: dict[str, dict[str, np.ndarray]] = {}
    points: dict[str, dict[str, np.ndarray]] = {}
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
        probabilities_by_model = cohort_data["probabilities"]
        group_inverse = cluster_inverse(cohort, cohort_data["metadata"])

        per_label[cohort] = per_label_bootstrap(
            targets,
            probabilities_by_model,
            group_inverse,
            replicates=args.bootstrap_replicates,
            seed=analysis.COHORT_SEEDS[cohort],
            batch_size=args.bootstrap_batch_size,
        )
        micro[cohort] = micro_bootstrap(
            targets,
            probabilities_by_model,
            headline_indices,
            group_inverse,
            replicates=args.bootstrap_replicates,
            seed=analysis.COHORT_SEEDS[cohort],
            batch_size=args.bootstrap_batch_size,
        )

        unit_weights = np.ones((1, len(targets)), dtype=np.float64)
        points[cohort] = {
            architecture: np.array(
                [
                    weighted_roc_auc_batch(
                        targets[:, index], probabilities[:, index], unit_weights
                    )[0]
                    for index in range(len(label_order))
                ]
            )
            for architecture, probabilities in probabilities_by_model.items()
        }

        # Proof that these are the registered resamples, not a new bootstrap.
        for architecture in analysis.ARCHITECTURES:
            rebuilt = macro_of_per_label(per_label[cohort][architecture], headline_indices)
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

    rows: list[dict[str, object]] = []
    for cohort in analysis.COHORTS:
        for architecture in analysis.ARCHITECTURES:
            for index, key in enumerate(label_order):
                lower, upper = percentile_interval(per_label[cohort][architecture][:, index])
                rows.append(
                    {
                        "cohort": cohort,
                        "architecture": architecture,
                        "label_key": key,
                        "diagnosis": diagnosis[key],
                        "headline": key in headline_keys,
                        "auroc": float(points[cohort][architecture][index]),
                        "auroc_lower_95": lower,
                        "auroc_upper_95": upper,
                    }
                )
    label_frame = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    label_path = args.output_root / "exploratory_per_label_intervals.csv"
    label_frame.to_csv(label_path, index=False)

    drop_rows: list[dict[str, object]] = []
    for external in ("chapman_shaoxing", "ningbo"):
        for architecture in analysis.ARCHITECTURES:
            for index, key in enumerate(label_order):
                distribution = (
                    per_label[external][architecture][:, index]
                    - per_label["ptb_test"][architecture][:, index]
                )
                point = float(
                    points[external][architecture][index] - points["ptb_test"][architecture][index]
                )
                lower, upper = percentile_interval(distribution)
                drop_rows.append(
                    {
                        "external_cohort": external,
                        "architecture": architecture,
                        "label_key": key,
                        "diagnosis": diagnosis[key],
                        "headline": key in headline_keys,
                        "shift_delta": point,
                        "shift_delta_lower_95": lower,
                        "shift_delta_upper_95": upper,
                        "excludes_zero": bool(lower > 0 or upper < 0),
                    }
                )
    drop_frame = pd.DataFrame(drop_rows)
    drop_path = args.output_root / "exploratory_per_label_shift_deltas.csv"
    drop_frame.to_csv(drop_path, index=False)

    micro_summary: dict[str, dict[str, dict[str, float]]] = {}
    for cohort in analysis.COHORTS:
        micro_summary[cohort] = {}
        for architecture in analysis.ARCHITECTURES:
            lower, upper = percentile_interval(micro[cohort][architecture])
            micro_summary[cohort][architecture] = {
                "lower_95": lower,
                "upper_95": upper,
            }

    headline_drops = drop_frame[drop_frame.headline]
    summary = {
        "stage": "post-seal exploratory addition",
        "stage_date": EXPLORATORY_STAGE_DATE,
        "changes_registered_quantities": False,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seeds": dict(analysis.COHORT_SEEDS),
        "interval_type": "percentile",
        "multiplicity_adjustment": "none; marginal intervals, matching the registered plan",
        "headline_shift_deltas_excluding_zero": int(headline_drops.excludes_zero.sum()),
        "headline_shift_deltas_total": int(len(headline_drops)),
        "ptb_cluster_unit": "patient_id",
        "external_cluster_unit": "record",
        "registered_resample_cross_check": cross_checks,
        "micro_auroc_intervals": micro_summary,
        "outputs": [label_path.name, drop_path.name],
    }
    summary_path = args.output_root / "exploratory_per_label_intervals.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"wrote {label_path}")
    print(f"wrote {drop_path}")
    print(f"wrote {summary_path}")
    print(
        f"{summary['headline_shift_deltas_excluding_zero']} of "
        f"{summary['headline_shift_deltas_total']} headline shift deltas exclude zero"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
