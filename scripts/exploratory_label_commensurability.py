#!/usr/bin/env python3
"""Test whether the headline labels mean the same thing in the three cohorts.

POST-SEAL EXPLORATORY ADDITION, 2026-07-26. This stage was not part of the frozen
analysis. It was written after the evaluation seal was broken and after the registered
results existed, and it is kept separate from `analyze_results.py` for that reason.

What it adds and why. The registered shift estimand assumes the 13 headline labels
denote the same finding in PTB-XL, Chapman-Shaoxing and Ningbo. That assumption is
testable from the label matrices alone, without reference to any model, and it fails
for part of the label set. PTB-XL annotates sinus rhythm on 56.2 percent of its sinus
bradycardia records and 29.3 percent of its sinus tachycardia records; the two external
cohorts annotate it on 0.00 percent and 0.05 percent. The three codes partition the
same finding space differently, so the registered sinus-rhythm AUROC compares a model
trained on one partition against labels drawn from another.

What it does not do. It changes no registered quantity, rewrites no sealed artifact,
re-runs no inference, and recomputes nothing that `analyze_results.py` reports. The
registered macro-AUROC and the registered shift deltas stand exactly as computed. The
reconciled estimands below are reported beside the registered ones, never in place of
them, and the registered estimand remains the primary result of the study.

The two reconciled estimands answer different questions and are both reported.
`union` replaces the target with the disjunction of the label and its competing labels
in every cohort including PTB-XL, and answers whether the model lost the underlying
capability. `restricted` evaluates only on records carrying no competing label in every
cohort including PTB-XL, and answers whether the model transfers on the records where
the two annotation conventions cannot disagree. Neither is a correction to the model.
Both change the estimand, so both are reported with the internal reference recomputed
under the same definition.

The competing-label sets are declared in COMPETING_LABELS below and are fixed for all
three cohorts and both architectures. They are chosen on the clinical relation between
the SNOMED groups, not on which choice recovers the most AUROC. Two of the four are
negative controls: T-wave abnormality and nonspecific intraventricular conduction
disorder receive the same treatment as sinus rhythm and do not recover under it.

Faithfulness to the registered design. The resamples are not a new bootstrap. This
stage rebuilds the registered cluster resamples from the same per-cohort seeds, the
same cluster units, the same replicate count and the same batch size, then proves it:
macro-averaging its per-label replicate AUROCs must reproduce the registered
macro-AUROC distribution in `results/bootstrap_distributions.npz` bit for bit. The run
aborts if that check fails. The `restricted` estimand is applied by zeroing the weight
of excluded records inside the registered resample rather than by drawing a new one, so
the cluster design is preserved exactly.

Multiplicity. These intervals are marginal and uncorrected, matching the registered
plan, which states that no multiplicity-adjusted tests were to be run.
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

EXPLORATORY_STAGE_DATE = "2026-07-26"

SINUS_RHYTHM = "426783006"
SINUS_BRADYCARDIA = "426177001"
SINUS_TACHYCARDIA = "427084000"
ATRIAL_FLUTTER = "164890007"
T_WAVE_ABNORMALITY = "164934002"
T_WAVE_INVERSION = "59931005"
NONSPECIFIC_IVCD = "698252002"
LEFT_BUNDLE_BRANCH_BLOCK = "733534002|164909002"
RIGHT_BUNDLE_BRANCH_BLOCK = "713427006|59118001"

# Groups that describe the same finding space and can therefore be annotated in place
# of one another. Declared before any reconciled number was computed, on the clinical
# relation between the codes: the three sinus codes partition sinus rhythm by rate, the
# two T-wave codes partition repolarization deviation by whether the reader called the
# deflection inverted, and nonspecific intraventricular conduction disorder is defined
# as QRS prolongation without a specific block pattern, so a bundle branch block is the
# alternative annotation for the same widened QRS.
COMPETING_LABELS = {
    SINUS_RHYTHM: (SINUS_BRADYCARDIA, SINUS_TACHYCARDIA),
    T_WAVE_ABNORMALITY: (T_WAVE_INVERSION,),
    T_WAVE_INVERSION: (T_WAVE_ABNORMALITY,),
    NONSPECIFIC_IVCD: (LEFT_BUNDLE_BRANCH_BLOCK, RIGHT_BUNDLE_BRANCH_BLOCK),
}

# Codes that describe the underlying rhythm, used only for the label-structure table.
RHYTHM_LABELS = (SINUS_RHYTHM, SINUS_BRADYCARDIA, SINUS_TACHYCARDIA, ATRIAL_FLUTTER)

# Macro variants reported beside the registered 13-label macro. `drop_sinus_rhythm`
# removes the one label whose annotation convention provably differs; `union_reconciled`
# keeps all 13 but evaluates the four labels in COMPETING_LABELS against their union
# target in every cohort.
MACRO_VARIANTS = ("registered", "drop_sinus_rhythm", "union_reconciled")


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


def point_auroc(targets: np.ndarray, scores: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Tie-aware AUROC on the full cohort, optionally restricted to a record mask."""

    weights = np.ones((1, len(targets)), dtype=np.float64)
    if mask is not None:
        weights = weights * mask.astype(np.float64)[None, :]
    return float(weighted_roc_auc_batch(targets.astype(bool), scores, weights)[0])


def reconciled_targets(
    targets: np.ndarray, label_index: int, competing_indices: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the union target and the restricted-record mask for one label.

    The union target is the disjunction of the label with every competing label. The
    restricted mask keeps the records that carry no competing label. Both are computed
    identically in every cohort, so the internal reference moves with the estimand.
    """

    label = targets[:, label_index].astype(bool)
    if not competing_indices:
        return label, np.ones(len(targets), dtype=bool)
    competing = targets[:, competing_indices].any(axis=1)
    return label | competing, ~competing


def label_structure_row(cohort: str, targets: np.ndarray, headline: np.ndarray, rhythm: list[int]):
    """One row of the label-density table for a cohort."""

    all_targets = targets.sum(axis=1)
    headline_targets = targets[:, headline].sum(axis=1)
    rhythm_count = targets[:, rhythm].sum(axis=1)
    return {
        "cohort": cohort,
        "records": int(len(targets)),
        "mean_labels_per_record_all_targets": float(all_targets.mean()),
        "mean_labels_per_record_headline": float(headline_targets.mean()),
        "share_no_label_all_targets": float(np.mean(all_targets == 0)),
        "share_no_label_headline": float(np.mean(headline_targets == 0)),
        "share_more_than_one_rhythm_label": float(np.mean(rhythm_count > 1)),
    }


def cooccurrence_rows(cohort: str, targets: np.ndarray, indices: np.ndarray, names: list[str]):
    """Pairwise joint counts and conditional rates over one cohort's headline labels."""

    rows: list[dict[str, object]] = []
    for row_position, row_index in enumerate(indices):
        row_mask = targets[:, row_index].astype(bool)
        for column_position, column_index in enumerate(indices):
            column_mask = targets[:, column_index].astype(bool)
            joint = int(np.sum(row_mask & column_mask))
            rows.append(
                {
                    "cohort": cohort,
                    "row_label": names[row_position],
                    "column_label": names[column_position],
                    "row_positives": int(row_mask.sum()),
                    "column_positives": int(column_mask.sum()),
                    "joint_positives": joint,
                    "conditional_column_given_row": (
                        float(joint / row_mask.sum()) if row_mask.any() else float("nan")
                    ),
                }
            )
    return rows


def per_label_replicates(
    targets: np.ndarray,
    scores: np.ndarray,
    group_inverse: np.ndarray,
    label_indices: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Per-label AUROC replicate distributions on the registered resamples."""

    output = np.empty((replicates, scores.shape[1]), dtype=np.float64)
    output.fill(np.nan)
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        for label_index in label_indices:
            output[start:end, label_index] = weighted_roc_auc_batch(
                targets[:, label_index], scores[:, label_index], weights
            )
    return output


def reconciled_replicates(
    truth: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    group_inverse: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """AUROC replicates for one reconciled target on the registered resamples.

    The record mask multiplies the resample weights rather than subsetting the cohort,
    so the cluster draw is the registered one and a restricted estimand differs from an
    unrestricted one only by which records carry weight.

    This draws its own resamples and is used by the tests. The pipeline uses
    `cohort_replicate_pass`, which computes every estimand of a cohort inside one
    traversal of the same resamples, because generating a cluster draw over 34,905
    record clusters costs far more than the AUROC that rides on it.
    """

    output = np.empty(replicates, dtype=np.float64)
    selector = mask.astype(np.float64)[None, :]
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        output[start:end] = weighted_roc_auc_batch(truth, scores, weights * selector)
    return output


def cohort_replicate_pass(
    specifications: list[tuple[object, np.ndarray, np.ndarray, np.ndarray | None]],
    group_inverse: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> dict[object, np.ndarray]:
    """Evaluate many weighted AUROCs on one traversal of a cohort's registered resamples.

    Each specification is a key, a boolean target vector, a score vector, and an
    optional record mask. Every specification sees the same weight batch before the next
    batch is drawn, so one cluster draw serves all of them and the resamples are
    identical across estimands by construction rather than by repeating the same seed.
    """

    output = {key: np.empty(replicates, dtype=np.float64) for key, _, _, _ in specifications}
    selectors = {
        key: (None if mask is None else mask.astype(np.float64)[None, :])
        for key, _, _, mask in specifications
    }
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        for key, truth, scores, _ in specifications:
            selector = selectors[key]
            effective = weights if selector is None else weights * selector
            output[key][start:end] = weighted_roc_auc_batch(truth, scores, effective)
    return output


def main() -> int:  # noqa: C901
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
    headline_names = [diagnosis[key] for key in headline_keys]
    rhythm_indices = [label_order.index(key) for key in RHYTHM_LABELS]

    registered_path = args.registered_bootstrap or (
        args.output_root / "bootstrap_distributions.npz"
    )
    registered = np.load(registered_path)

    loaded: dict[str, dict[str, object]] = {}
    groups: dict[str, np.ndarray] = {}
    cross_checks: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []
    cooccurrence: list[dict[str, object]] = []
    estimand_rows: list[dict[str, object]] = []
    replicate_store: dict[tuple[str, str, str, str], np.ndarray] = {}
    point_store: dict[tuple[str, str, str, str], float] = {}
    masks: dict[tuple[str, str, str], np.ndarray] = {}
    truths: dict[tuple[str, str, str], np.ndarray] = {}

    for cohort in analysis.COHORTS:
        cohort_data = analysis.load_verified_cohort(
            cohort=cohort,
            prediction_root=args.prediction_root,
            store_root=args.store_root,
            evaluation_seal=evaluation_seal,
            choices=choices,
            label_count=len(label_order),
        )
        loaded[cohort] = cohort_data
        targets = cohort_data["targets"]
        groups[cohort] = cluster_inverse(cohort, cohort_data["metadata"])
        structure_rows.append(
            label_structure_row(cohort, targets, headline_indices, rhythm_indices)
        )
        cooccurrence.extend(cooccurrence_rows(cohort, targets, headline_indices, headline_names))

        # One traversal of this cohort's registered resamples serves the bit-for-bit
        # verification and all three reconciled estimands of every headline label and
        # both architectures. `as_registered` is the same statistic the verification
        # computes, so it is read back from the verification table rather than repeated.
        specifications: list[tuple[object, np.ndarray, np.ndarray, np.ndarray | None]] = []
        for label_key in headline_keys:
            label_index = label_order.index(label_key)
            competing_indices = [
                label_order.index(key) for key in COMPETING_LABELS.get(label_key, ())
            ]
            union_truth, restricted_mask = reconciled_targets(
                targets, label_index, competing_indices
            )
            label_truth = targets[:, label_index].astype(bool)
            for architecture in analysis.ARCHITECTURES:
                scores = cohort_data["probabilities"][architecture][:, label_index]
                specifications.append(
                    (("verify", architecture, label_key), label_truth, scores, None)
                )
                specifications.append(
                    (("union", architecture, label_key), union_truth, scores, None)
                )
                specifications.append(
                    (
                        ("restricted", architecture, label_key),
                        label_truth,
                        scores,
                        restricted_mask,
                    )
                )
            truths[(cohort, label_key, "as_registered")] = label_truth
            truths[(cohort, label_key, "union")] = union_truth
            truths[(cohort, label_key, "restricted")] = label_truth
            masks[(cohort, label_key, "as_registered")] = np.ones(len(targets), dtype=bool)
            masks[(cohort, label_key, "union")] = np.ones(len(targets), dtype=bool)
            masks[(cohort, label_key, "restricted")] = restricted_mask

        computed = cohort_replicate_pass(
            specifications,
            groups[cohort],
            replicates=args.bootstrap_replicates,
            seed=analysis.COHORT_SEEDS[cohort],
            batch_size=args.bootstrap_batch_size,
        )

        for architecture in analysis.ARCHITECTURES:
            verification = np.full((args.bootstrap_replicates, len(label_order)), np.nan)
            for label_key in headline_keys:
                verification[:, label_order.index(label_key)] = computed[
                    ("verify", architecture, label_key)
                ]
            rebuilt = macro_of_per_label(verification, headline_indices)
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

        for label_key in headline_keys:
            label_index = label_order.index(label_key)
            competing_keys = list(COMPETING_LABELS.get(label_key, ()))
            for architecture in analysis.ARCHITECTURES:
                scores = cohort_data["probabilities"][architecture][:, label_index]
                for variant, computed_key in (
                    ("as_registered", "verify"),
                    ("union", "union"),
                    ("restricted", "restricted"),
                ):
                    truth = truths[(cohort, label_key, variant)]
                    mask = masks[(cohort, label_key, variant)]
                    point = point_auroc(truth, scores, mask)
                    distribution = computed[(computed_key, architecture, label_key)]
                    lower, upper = percentile_interval(distribution)
                    replicate_store[(cohort, architecture, label_key, variant)] = distribution
                    point_store[(cohort, architecture, label_key, variant)] = point
                    estimand_rows.append(
                        {
                            "cohort": cohort,
                            "architecture": architecture,
                            "label_key": label_key,
                            "diagnosis": diagnosis[label_key],
                            "estimand": variant,
                            "competing_labels": ";".join(diagnosis[key] for key in competing_keys),
                            "records_evaluated": int(mask.sum()),
                            "positives": int(truth[mask].sum()),
                            "prevalence": float(truth[mask].mean()) if mask.any() else float("nan"),
                            "auroc": point,
                            "auroc_lower_95": lower,
                            "auroc_upper_95": upper,
                        }
                    )

    shift_rows: list[dict[str, object]] = []
    for label_key in headline_keys:
        for architecture in analysis.ARCHITECTURES:
            for external in analysis.EXTERNAL_COHORTS:
                for variant in ("as_registered", "union", "restricted"):
                    internal_key = ("ptb_test", architecture, label_key, variant)
                    external_key = (external, architecture, label_key, variant)
                    distribution = replicate_store[external_key] - replicate_store[internal_key]
                    point = point_store[external_key] - point_store[internal_key]
                    lower, upper = percentile_interval(distribution)
                    shift_rows.append(
                        {
                            "external_cohort": external,
                            "architecture": architecture,
                            "label_key": label_key,
                            "diagnosis": diagnosis[label_key],
                            "estimand": variant,
                            "internal_auroc": point_store[internal_key],
                            "external_auroc": point_store[external_key],
                            "shift_delta": point,
                            "shift_delta_lower_95": lower,
                            "shift_delta_upper_95": upper,
                            "excludes_zero": bool(lower > 0 or upper < 0),
                        }
                    )

    # ---- macro variants -----------------------------------------------------------------
    macro_points: dict[tuple[str, str, str], float] = {}
    macro_distributions: dict[tuple[str, str, str], np.ndarray] = {}
    for variant in MACRO_VARIANTS:
        if variant == "registered":
            member_keys = headline_keys
            estimand_by_label = dict.fromkeys(headline_keys, "as_registered")
        elif variant == "drop_sinus_rhythm":
            member_keys = [key for key in headline_keys if key != SINUS_RHYTHM]
            estimand_by_label = dict.fromkeys(member_keys, "as_registered")
        else:
            member_keys = headline_keys
            estimand_by_label = {
                key: ("union" if key in COMPETING_LABELS else "as_registered")
                for key in headline_keys
            }
        for cohort in analysis.COHORTS:
            for architecture in analysis.ARCHITECTURES:
                stacked = np.stack(
                    [
                        replicate_store[(cohort, architecture, key, estimand_by_label[key])]
                        for key in member_keys
                    ],
                    axis=1,
                )
                macro_distributions[(cohort, architecture, variant)] = np.nanmean(stacked, axis=1)
                macro_points[(cohort, architecture, variant)] = float(
                    np.nanmean(
                        [
                            point_store[(cohort, architecture, key, estimand_by_label[key])]
                            for key in member_keys
                        ]
                    )
                )

    macro_rows: list[dict[str, object]] = []
    for variant in MACRO_VARIANTS:
        for architecture in analysis.ARCHITECTURES:
            for cohort in analysis.COHORTS:
                lower, upper = percentile_interval(
                    macro_distributions[(cohort, architecture, variant)]
                )
                macro_rows.append(
                    {
                        "macro_variant": variant,
                        "architecture": architecture,
                        "cohort": cohort,
                        "quantity": "macro_auroc",
                        "value": macro_points[(cohort, architecture, variant)],
                        "lower_95": lower,
                        "upper_95": upper,
                    }
                )
            for external in analysis.EXTERNAL_COHORTS:
                distribution = (
                    macro_distributions[(external, architecture, variant)]
                    - macro_distributions[("ptb_test", architecture, variant)]
                )
                point = (
                    macro_points[(external, architecture, variant)]
                    - macro_points[("ptb_test", architecture, variant)]
                )
                lower, upper = percentile_interval(distribution)
                macro_rows.append(
                    {
                        "macro_variant": variant,
                        "architecture": architecture,
                        "cohort": external,
                        "quantity": "shift_delta",
                        "value": point,
                        "lower_95": lower,
                        "upper_95": upper,
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    structure_path = args.output_root / "exploratory_label_structure.csv"
    cooccurrence_path = args.output_root / "exploratory_label_cooccurrence.csv"
    estimand_path = args.output_root / "exploratory_reconciled_auroc.csv"
    shift_path = args.output_root / "exploratory_reconciled_shift_deltas.csv"
    macro_path = args.output_root / "exploratory_macro_variants.csv"
    pd.DataFrame(structure_rows).to_csv(structure_path, index=False)
    pd.DataFrame(cooccurrence).to_csv(cooccurrence_path, index=False)
    pd.DataFrame(estimand_rows).to_csv(estimand_path, index=False)
    pd.DataFrame(shift_rows).to_csv(shift_path, index=False)
    pd.DataFrame(macro_rows).to_csv(macro_path, index=False)

    sinus_conditional = {}
    for cohort in analysis.COHORTS:
        targets = loaded[cohort]["targets"]
        sinus = targets[:, label_order.index(SINUS_RHYTHM)].astype(bool)
        entry = {}
        for name, key in (
            ("sinus_bradycardia", SINUS_BRADYCARDIA),
            ("sinus_tachycardia", SINUS_TACHYCARDIA),
        ):
            other = targets[:, label_order.index(key)].astype(bool)
            entry[name] = {
                "positives": int(other.sum()),
                "also_sinus_rhythm": int(np.sum(other & sinus)),
                "conditional_sinus_rhythm": (
                    float(np.mean(sinus[other])) if other.any() else float("nan")
                ),
            }
        sinus_conditional[cohort] = entry

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
        "competing_label_sets": {
            diagnosis[key]: [diagnosis[other] for other in value]
            for key, value in COMPETING_LABELS.items()
        },
        "registered_resample_cross_check": cross_checks,
        "label_structure": structure_rows,
        "sinus_rhythm_conditional_annotation": sinus_conditional,
        "macro_variants": {
            variant: {
                architecture: {
                    "macro_auroc": {
                        cohort: macro_points[(cohort, architecture, variant)]
                        for cohort in analysis.COHORTS
                    },
                    "shift_delta": {
                        external: (
                            macro_points[(external, architecture, variant)]
                            - macro_points[("ptb_test", architecture, variant)]
                        )
                        for external in analysis.EXTERNAL_COHORTS
                    },
                }
                for architecture in analysis.ARCHITECTURES
            }
            for variant in MACRO_VARIANTS
        },
        "outputs": [
            structure_path.name,
            cooccurrence_path.name,
            estimand_path.name,
            shift_path.name,
            macro_path.name,
        ],
    }
    summary_path = args.output_root / "exploratory_label_commensurability.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    for path in (structure_path, cooccurrence_path, estimand_path, shift_path, macro_path):
        print(f"wrote {path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
