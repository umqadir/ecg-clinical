#!/usr/bin/env python3
"""Locate the transfer loss: which side of the comparison moved, and what did not move it.

POST-SEAL EXPLORATORY ADDITION, 2026-07-26. This stage was not part of the frozen
analysis. It was written after the evaluation seal was broken and after the registered
results existed, and it is kept separate from `analyze_results.py` for that reason.

What it adds and why. The registered analysis establishes that per-label AUROC falls on
transfer and by how much. It does not say which of the two classes moved. AUROC is a
comparison between the scores of positive records and the scores of negative records, so
a fall can come from external positives scoring lower, from external negatives scoring
higher, or from both. Those are different failures with different causes, and they are
separable from the sealed predictions at no additional inference cost.

The decomposition. Write A(x, y) for the AUROC of cohort x's positive records against
cohort y's negative records, with i for PTB-XL fold 10 and e for an external cohort.
A(i, i) is the registered internal AUROC and A(e, e) is the registered external AUROC.
The two mixed quantities A(e, i) and A(i, e) hold one class fixed while the other moves.
The Shapley value of each class over the two orderings of the two swaps is

    positive_side = 0.5 * [(A(e, i) - A(i, i)) + (A(e, e) - A(i, e))]
    negative_side = 0.5 * [(A(i, e) - A(i, i)) + (A(e, e) - A(e, i))]

and positive_side + negative_side = A(e, e) - A(i, i) exactly, which is the registered
per-label shift delta. The split is therefore an exact attribution of a quantity the
registered analysis already reports, not a new estimand.

The marginal-shift control. A(e, i) and A(i, e) compare scores across cohorts, so a
uniform shift of the external score distribution moves both of them without any change
in class membership. Two variants are reported. `raw` uses the sealed probabilities and
is contaminated by that shift. `rank_matched` replaces every score by its within-cohort
midrank divided by the cohort size, which is monotone within each cohort and therefore
leaves A(i, i) and A(e, e) exactly unchanged while forcing the two marginal score
distributions to agree. Any asymmetry that survives rank matching is class membership,
not a global score shift. The rank transform is computed once on the full cohort and
held fixed inside every bootstrap replicate.

Three candidate explanations are tested and reported whether or not they survive.
Prevalence shift is tested by correlating the absolute log prevalence ratio against the
per-label drop across the 13 headline labels. Demographic shift is tested by
standardizing each external cohort to the PTB-XL fold 10 joint distribution of age band
and sex and recomputing the macro estimand. Raw SNOMED code composition is tested for
the two bundle branch block groups, whose behaviour differs by hospital.

Seed variance. The registered cluster bootstrap resamples records and holds the
three-seed ensemble fixed, so it estimates evaluation-set sampling error and carries no
information about run-to-run variation in training. The 18 per-seed macro-AUROCs are
reported here so that the architecture contrast can be read against the spread of the
design that produced it.

What it does not do. It changes no registered quantity, rewrites no sealed artifact,
re-runs no inference, and recomputes nothing that `analyze_results.py` reports.

Faithfulness to the registered design. The resamples are not a new bootstrap. This
stage rebuilds the registered cluster resamples from the same per-cohort seeds, cluster
units, replicate count and batch size, and proves it bit for bit against
`results/bootstrap_distributions.npz` before publishing an interval. Cross-cohort
quantities pair the internal and external resamples at the same replicate index, which
is how the registered shift delta is formed.

Multiplicity. These intervals are marginal and uncorrected, matching the registered plan.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ecg_clinical.bootstrap import (
    bootstrap_record_weight_batches,
    percentile_interval,
    weighted_roc_auc_batch,
)
from ecg_clinical.integrity import verify_evaluation_seal, verify_preregistration_seal

EXPLORATORY_STAGE_DATE = "2026-07-26"
SEEDS = (17, 29, 43)
AGE_BANDS = ("under_40", "40_to_64", "65_plus")
SEX_LEVELS = ("female", "male")
BUNDLE_BRANCH_GROUPS = {
    "733534002|164909002": ("733534002", "164909002"),
    "713427006|59118001": ("713427006", "59118001"),
}


def load_analysis_module() -> ModuleType:
    """Import the registered analysis stage to reuse its verified cohort loader."""

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
    parser.add_argument(
        "--headers", type=Path, default=Path("data/cache/challenge2021_headers.csv")
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


def midrank_scores(scores: np.ndarray) -> np.ndarray:
    """Within-cohort midranks scaled to the unit interval.

    Monotone in the score and tie-preserving, so the AUROC of any subset of this cohort
    against any other subset of the same cohort is exactly unchanged. Applying it to
    both cohorts gives them the same marginal score distribution up to tie structure,
    which is what removes a global score shift from the cross-cohort comparisons.
    """

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    lower = np.searchsorted(sorted_scores, sorted_scores, side="left")
    upper = np.searchsorted(sorted_scores, sorted_scores, side="right")
    midranks = np.empty(len(scores), dtype=np.float64)
    midranks[order] = (lower + upper - 1) / 2.0
    return midranks / max(len(scores) - 1, 1)


def weighted_auroc(truth: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weighted_roc_auc_batch(truth, scores, weights)


def cross_cohort_auroc(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    positive_weights: np.ndarray,
    negative_weights: np.ndarray,
) -> np.ndarray:
    """AUROC of one cohort's positives against another cohort's negatives.

    The two weight blocks come from the two cohorts' own registered resamples at the
    same replicate index, which is how the registered shift delta pairs independent
    cohort draws.
    """

    scores = np.concatenate([positive_scores, negative_scores])
    truth = np.concatenate(
        [
            np.ones(len(positive_scores), dtype=bool),
            np.zeros(len(negative_scores), dtype=bool),
        ]
    )
    weights = np.concatenate([positive_weights, negative_weights], axis=1)
    return weighted_roc_auc_batch(truth, scores, weights)


def weight_batches(group_inverse: np.ndarray, *, replicates: int, seed: int, batch_size: int):
    """The registered resample weights for a cohort, one batch at a time.

    Drawing a multinomial over 34,905 record clusters costs far more than the AUROC that
    rides on it, so every statistic a pass needs is evaluated inside the batch loop
    rather than by re-drawing the same seed once per statistic. The batches are never
    materialized as a list: at 2,000 replicates the Ningbo weights alone would be 558 MB.
    """

    return bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    )


def shapley_split(
    a_ii: np.ndarray, a_ee: np.ndarray, a_ei: np.ndarray, a_ie: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact attribution of A(e, e) - A(i, i) to the positive and the negative class."""

    positive_side = 0.5 * ((a_ei - a_ii) + (a_ee - a_ie))
    negative_side = 0.5 * ((a_ie - a_ii) + (a_ee - a_ei))
    return positive_side, negative_side


def standardization_weights(
    internal_metadata: pd.DataFrame,
    external_metadata: pd.DataFrame,
    analysis: ModuleType,
    external_cohort: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reweight an external cohort to the PTB-XL fold 10 joint age-band and sex mix.

    Standardization is exact only on the cells both cohorts populate. Three exclusions
    are counted separately rather than merged, because they mean different things. A
    record missing either stratifier gets weight zero, matching the registered rule that
    a record is excluded only from the analysis whose stratifier it lacks. A record in a
    cell that PTB-XL fold 10 does not populate gets weight zero because there is no
    internal mass to match it to. An internal cell the external cohort does not populate
    cannot be represented at all, so the internal target is renormalized over the cells
    that are present and the unmatched internal mass is reported. Weights are then scaled
    so the reweighted cohort carries the same total mass as the records it retains.
    """

    def strata(metadata: pd.DataFrame, cohort: str) -> pd.Series:
        age = analysis.normalized_age(metadata)
        sex = analysis.normalized_sex(metadata, cohort)
        return pd.Series([f"{a}|{s}" for a, s in zip(age, sex, strict=True)], index=metadata.index)

    valid_cells = [f"{band}|{sex}" for band in AGE_BANDS for sex in SEX_LEVELS]
    internal_cells = strata(internal_metadata, "ptb_test")
    external_cells = strata(external_metadata, external_cohort)
    internal_share = internal_cells.value_counts(normalize=True)
    external_share = external_cells.value_counts(normalize=True)

    external_present = [cell for cell in valid_cells if float(external_share.get(cell, 0.0)) > 0.0]
    matched_internal_mass = sum(float(internal_share.get(cell, 0.0)) for cell in external_present)
    if matched_internal_mass <= 0.0:
        raise RuntimeError(
            f"no age-band and sex cell is populated in both PTB-XL fold 10 and {external_cohort}"
        )

    weights = np.zeros(len(external_metadata), dtype=np.float64)
    for cell in external_present:
        selected = (external_cells == cell).to_numpy()
        target = float(internal_share.get(cell, 0.0)) / matched_internal_mass
        weights[selected] = target / float(external_share.get(cell, 0.0))
    carried = weights > 0
    if carried.any():
        weights = weights * (carried.sum() / weights.sum())

    has_stratifiers = external_cells.isin(valid_cells).to_numpy()
    diagnostics = {
        "external_records": int(len(external_metadata)),
        "records_carrying_weight": int(carried.sum()),
        "records_missing_a_stratifier": int((~has_stratifiers).sum()),
        "records_in_a_stratum_absent_from_ptb_fold_10": int(np.sum(has_stratifiers & ~carried)),
        "internal_mass_in_strata_absent_externally": float(1.0 - matched_internal_mass),
        "internal_cell_shares": {
            cell: float(internal_share.get(cell, 0.0)) for cell in valid_cells
        },
        "external_cell_shares": {
            cell: float(external_share.get(cell, 0.0)) for cell in valid_cells
        },
    }
    return weights, diagnostics


def macro_from_weights(
    targets: np.ndarray,
    scores: np.ndarray,
    label_indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Macro-AUROC over the headline labels for a batch of record-weight rows."""

    per_label = np.stack(
        [
            weighted_roc_auc_batch(targets[:, index], scores[:, index], weights)
            for index in label_indices
        ],
        axis=1,
    )
    contributing = np.isfinite(per_label).sum(axis=1)
    return np.divide(
        np.nansum(per_label, axis=1),
        contributing,
        out=np.full(len(per_label), np.nan),
        where=contributing > 0,
    )


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

    registered_path = args.registered_bootstrap or (
        args.output_root / "bootstrap_distributions.npz"
    )
    registered = np.load(registered_path)

    loaded: dict[str, dict[str, object]] = {}
    groups: dict[str, np.ndarray] = {}
    per_label_replicates: dict[tuple[str, str], np.ndarray] = {}
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
        loaded[cohort] = cohort_data
        groups[cohort] = cluster_inverse(cohort, cohort_data["metadata"])
        targets = cohort_data["targets"]
        # One traversal of this cohort's registered resamples fills the per-label table
        # for both architectures at once.
        tables = {
            architecture: np.full((args.bootstrap_replicates, len(label_order)), np.nan)
            for architecture in analysis.ARCHITECTURES
        }
        for start, weights in weight_batches(
            groups[cohort],
            replicates=args.bootstrap_replicates,
            seed=analysis.COHORT_SEEDS[cohort],
            batch_size=args.bootstrap_batch_size,
        ):
            end = start + len(weights)
            for architecture in analysis.ARCHITECTURES:
                scores = cohort_data["probabilities"][architecture]
                for index in headline_indices:
                    tables[architecture][start:end, index] = weighted_auroc(
                        targets[:, index], scores[:, index], weights
                    )
        for architecture in analysis.ARCHITECTURES:
            table = tables[architecture]
            per_label_replicates[(cohort, architecture)] = table
            rebuilt = macro_of_per_label(table, headline_indices)
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

    unit = {cohort: np.ones((1, len(loaded[cohort]["targets"]))) for cohort in analysis.COHORTS}

    # ---- class-side decomposition -------------------------------------------------------
    # Point estimates first, for both variants. The bootstrap that follows traverses each
    # external cohort's resamples once, paired with PTB-XL fold 10 at the same replicate
    # index, and fills the cross-cohort terms for every architecture and label inside that
    # single traversal.
    decomposition_rows: list[dict[str, object]] = []
    rank_matched_scores: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for architecture in analysis.ARCHITECTURES:
        for external in analysis.EXTERNAL_COHORTS:
            internal_targets = loaded["ptb_test"]["targets"]
            external_targets = loaded[external]["targets"]
            for label_key in headline_keys:
                index = label_order.index(label_key)
                internal_truth = internal_targets[:, index].astype(bool)
                external_truth = external_targets[:, index].astype(bool)
                raw_internal = loaded["ptb_test"]["probabilities"][architecture][:, index]
                raw_external = loaded[external]["probabilities"][architecture][:, index]
                matched_internal = midrank_scores(raw_internal)
                matched_external = midrank_scores(raw_external)
                rank_matched_scores[(architecture, external, label_key)] = (
                    matched_internal,
                    matched_external,
                )
                variants = {
                    "raw": (raw_internal, raw_external),
                    "rank_matched": (matched_internal, matched_external),
                }
                marginal = float(
                    cross_cohort_auroc(
                        raw_external, raw_internal, unit[external], unit["ptb_test"]
                    )[0]
                )
                for variant, (internal_scores, external_scores) in variants.items():
                    point_ii = float(
                        weighted_auroc(internal_truth, internal_scores, unit["ptb_test"])[0]
                    )
                    point_ee = float(
                        weighted_auroc(external_truth, external_scores, unit[external])[0]
                    )
                    point_ei = float(
                        cross_cohort_auroc(
                            external_scores[external_truth],
                            internal_scores[~internal_truth],
                            unit[external][:, external_truth],
                            unit["ptb_test"][:, ~internal_truth],
                        )[0]
                    )
                    point_ie = float(
                        cross_cohort_auroc(
                            internal_scores[internal_truth],
                            external_scores[~external_truth],
                            unit["ptb_test"][:, internal_truth],
                            unit[external][:, ~external_truth],
                        )[0]
                    )
                    positive_point, negative_point = shapley_split(
                        np.array([point_ii]),
                        np.array([point_ee]),
                        np.array([point_ei]),
                        np.array([point_ie]),
                    )
                    decomposition_rows.append(
                        {
                            "external_cohort": external,
                            "architecture": architecture,
                            "label_key": label_key,
                            "diagnosis": diagnosis[label_key],
                            "variant": variant,
                            "marginal_score_shift": marginal,
                            "auroc_internal_positives_internal_negatives": point_ii,
                            "auroc_external_positives_external_negatives": point_ee,
                            "auroc_external_positives_internal_negatives": point_ei,
                            "auroc_internal_positives_external_negatives": point_ie,
                            "shift_delta": point_ee - point_ii,
                            "positive_side": float(positive_point[0]),
                            "negative_side": float(negative_point[0]),
                        }
                    )

    standardization_weight_by_cohort: dict[str, np.ndarray] = {}
    standardization_diagnostics: dict[str, object] = {}
    for external in analysis.EXTERNAL_COHORTS:
        weights, diagnostics = standardization_weights(
            loaded["ptb_test"]["metadata"], loaded[external]["metadata"], analysis, external
        )
        standardization_weight_by_cohort[external] = weights
        standardization_diagnostics[external] = diagnostics

    cross_replicates: dict[tuple[str, str, str, str], np.ndarray] = {}
    standardized_replicates: dict[tuple[str, str], np.ndarray] = {}
    for external in analysis.EXTERNAL_COHORTS:
        for architecture in analysis.ARCHITECTURES:
            for label_key in headline_keys:
                for term in ("ei", "ie"):
                    cross_replicates[(external, architecture, label_key, term)] = np.empty(
                        args.bootstrap_replicates, dtype=np.float64
                    )
            standardized_replicates[(external, architecture)] = np.empty(
                args.bootstrap_replicates, dtype=np.float64
            )
        selector = standardization_weight_by_cohort[external][None, :]
        paired = zip(
            weight_batches(
                groups["ptb_test"],
                replicates=args.bootstrap_replicates,
                seed=analysis.COHORT_SEEDS["ptb_test"],
                batch_size=args.bootstrap_batch_size,
            ),
            weight_batches(
                groups[external],
                replicates=args.bootstrap_replicates,
                seed=analysis.COHORT_SEEDS[external],
                batch_size=args.bootstrap_batch_size,
            ),
            strict=True,
        )
        for (start, internal_weights), (_, external_weights) in paired:
            end = start + len(internal_weights)
            for architecture in analysis.ARCHITECTURES:
                standardized_replicates[(external, architecture)][start:end] = macro_from_weights(
                    loaded[external]["targets"],
                    loaded[external]["probabilities"][architecture],
                    headline_indices,
                    external_weights * selector,
                )
                for label_key in headline_keys:
                    index = label_order.index(label_key)
                    internal_truth = loaded["ptb_test"]["targets"][:, index].astype(bool)
                    external_truth = loaded[external]["targets"][:, index].astype(bool)
                    internal_scores, external_scores = rank_matched_scores[
                        (architecture, external, label_key)
                    ]
                    cross_replicates[(external, architecture, label_key, "ei")][start:end] = (
                        cross_cohort_auroc(
                            external_scores[external_truth],
                            internal_scores[~internal_truth],
                            external_weights[:, external_truth],
                            internal_weights[:, ~internal_truth],
                        )
                    )
                    cross_replicates[(external, architecture, label_key, "ie")][start:end] = (
                        cross_cohort_auroc(
                            internal_scores[internal_truth],
                            external_scores[~external_truth],
                            internal_weights[:, internal_truth],
                            external_weights[:, ~external_truth],
                        )
                    )
        print(f"paired resample traversal complete for {external}")

    for row in decomposition_rows:
        if row["variant"] != "rank_matched":
            continue
        external = row["external_cohort"]
        architecture = row["architecture"]
        index = label_order.index(row["label_key"])
        positive_distribution, negative_distribution = shapley_split(
            per_label_replicates[("ptb_test", architecture)][:, index],
            per_label_replicates[(external, architecture)][:, index],
            cross_replicates[(external, architecture, row["label_key"], "ei")],
            cross_replicates[(external, architecture, row["label_key"], "ie")],
        )
        for name, distribution in (
            ("positive_side", positive_distribution),
            ("negative_side", negative_distribution),
        ):
            lower, upper = percentile_interval(distribution)
            row[f"{name}_lower_95"] = lower
            row[f"{name}_upper_95"] = upper

    # ---- how the macro drop is distributed over the 13 labels ---------------------------
    # The registered macro is an unweighted mean over the headline labels, so each label
    # contributes exactly its own shift delta divided by the label count. Reporting the
    # ranked contributions answers how much of the headline result rests on how few
    # labels, and the bootstrap p-value and its Benjamini-Hochberg q-value answer what
    # survives the multiplicity the registered plan declined to adjust for.
    contribution_rows: list[dict[str, object]] = []
    for architecture in analysis.ARCHITECTURES:
        collected: list[dict[str, object]] = []
        for external in analysis.EXTERNAL_COHORTS:
            internal_points = {
                key: float(
                    weighted_auroc(
                        loaded["ptb_test"]["targets"][:, label_order.index(key)],
                        loaded["ptb_test"]["probabilities"][architecture][
                            :, label_order.index(key)
                        ],
                        unit["ptb_test"],
                    )[0]
                )
                for key in headline_keys
            }
            external_points = {
                key: float(
                    weighted_auroc(
                        loaded[external]["targets"][:, label_order.index(key)],
                        loaded[external]["probabilities"][architecture][:, label_order.index(key)],
                        unit[external],
                    )[0]
                )
                for key in headline_keys
            }
            total_drop = float(
                np.mean([external_points[key] for key in headline_keys])
                - np.mean([internal_points[key] for key in headline_keys])
            )
            for key in headline_keys:
                index = label_order.index(key)
                distribution = (
                    per_label_replicates[(external, architecture)][:, index]
                    - per_label_replicates[("ptb_test", architecture)][:, index]
                )
                finite = distribution[np.isfinite(distribution)]
                # Add-one bounds on both tails so a replicate distribution lying wholly
                # on one side of zero reports 2 / (replicates + 1) rather than exactly 0.
                below = (float(np.sum(finite <= 0.0)) + 1.0) / (len(finite) + 1.0)
                above = (float(np.sum(finite >= 0.0)) + 1.0) / (len(finite) + 1.0)
                p_value = float(min(1.0, 2.0 * min(below, above)))
                drop = external_points[key] - internal_points[key]
                collected.append(
                    {
                        "architecture": architecture,
                        "external_cohort": external,
                        "label_key": key,
                        "diagnosis": diagnosis[key],
                        "shift_delta": drop,
                        "contribution_to_macro_drop": drop / len(headline_keys),
                        "share_of_macro_drop": (
                            (drop / len(headline_keys)) / total_drop
                            if total_drop != 0
                            else float("nan")
                        ),
                        "bootstrap_two_sided_p": p_value,
                    }
                )
        ordered = sorted(collected, key=lambda entry: entry["bootstrap_two_sided_p"])
        tests = len(ordered)
        running = 1.0
        for rank in range(tests, 0, -1):
            entry = ordered[rank - 1]
            running = min(running, entry["bootstrap_two_sided_p"] * tests / rank)
            entry["benjamini_hochberg_q"] = running
            entry["tests_in_family"] = tests
        contribution_rows.extend(collected)

    # ---- prevalence-shift test ----------------------------------------------------------
    prevalence_rows: list[dict[str, object]] = []
    decomposition = pd.DataFrame(decomposition_rows)
    for architecture in analysis.ARCHITECTURES:
        for external in analysis.EXTERNAL_COHORTS:
            subset = decomposition.query(
                "architecture == @architecture and external_cohort == @external "
                "and variant == 'rank_matched'"
            )
            internal_prevalence = np.array(
                [
                    loaded["ptb_test"]["targets"][:, label_order.index(key)].mean()
                    for key in subset.label_key
                ]
            )
            external_prevalence = np.array(
                [
                    loaded[external]["targets"][:, label_order.index(key)].mean()
                    for key in subset.label_key
                ]
            )
            log_ratio = np.abs(np.log2(external_prevalence / internal_prevalence))
            for statistic_name, quantity in (
                ("shift_delta", subset.shift_delta.to_numpy()),
                ("positive_side", subset.positive_side.to_numpy()),
                ("negative_side", subset.negative_side.to_numpy()),
            ):
                spearman = spearmanr(log_ratio, quantity)
                pearson = pearsonr(log_ratio, quantity)
                prevalence_rows.append(
                    {
                        "architecture": architecture,
                        "external_cohort": external,
                        "quantity": statistic_name,
                        "labels": int(len(quantity)),
                        "spearman_rho": float(spearman.statistic),
                        "spearman_p": float(spearman.pvalue),
                        "pearson_r": float(pearson.statistic),
                        "pearson_p": float(pearson.pvalue),
                    }
                )

    # ---- demographic standardization ----------------------------------------------------
    standardization_rows: list[dict[str, object]] = []
    for external in analysis.EXTERNAL_COHORTS:
        selector = standardization_weight_by_cohort[external][None, :]
        for architecture in analysis.ARCHITECTURES:
            targets = loaded[external]["targets"]
            scores = loaded[external]["probabilities"][architecture]
            observed = float(
                macro_from_weights(targets, scores, headline_indices, unit[external])[0]
            )
            standardized = float(macro_from_weights(targets, scores, headline_indices, selector)[0])
            replicates = standardized_replicates[(external, architecture)]
            internal_point = float(
                macro_from_weights(
                    loaded["ptb_test"]["targets"],
                    loaded["ptb_test"]["probabilities"][architecture],
                    headline_indices,
                    unit["ptb_test"],
                )[0]
            )
            internal_replicates = macro_of_per_label(
                per_label_replicates[("ptb_test", architecture)], headline_indices
            )
            lower, upper = percentile_interval(replicates)
            delta_lower, delta_upper = percentile_interval(replicates - internal_replicates)
            standardization_rows.append(
                {
                    "external_cohort": external,
                    "architecture": architecture,
                    "observed_macro_auroc": observed,
                    "standardized_macro_auroc": standardized,
                    "standardized_lower_95": lower,
                    "standardized_upper_95": upper,
                    "internal_macro_auroc": internal_point,
                    "observed_shift_delta": observed - internal_point,
                    "standardized_shift_delta": standardized - internal_point,
                    "standardized_shift_delta_lower_95": delta_lower,
                    "standardized_shift_delta_upper_95": delta_upper,
                }
            )

    # ---- bundle branch block code composition -------------------------------------------
    composition_rows: list[dict[str, object]] = []
    headers = pd.read_csv(args.headers, dtype={"diagnoses": str})
    header_cohorts = {
        "ptb_test": "ptb-xl",
        "chapman_shaoxing": "chapman_shaoxing",
        "ningbo": "ningbo",
    }
    for group_key, (first_code, second_code) in BUNDLE_BRANCH_GROUPS.items():
        for cohort, header_name in header_cohorts.items():
            source = headers[headers.cohort == header_name]
            codes = source.diagnoses.fillna("")
            first = codes.str.contains(rf"(?:^|\|){first_code}(?:\||$)", regex=True)
            second = codes.str.contains(rf"(?:^|\|){second_code}(?:\||$)", regex=True)
            composition_rows.append(
                {
                    "label_key": group_key,
                    "diagnosis": diagnosis[group_key],
                    "cohort": cohort,
                    "source_records": int(len(source)),
                    f"code_{first_code}": int(first.sum()),
                    f"code_{second_code}": int(second.sum()),
                    "both_codes": int(np.sum(first & second)),
                    "either_code": int(np.sum(first | second)),
                }
            )

    score_rows: list[dict[str, object]] = []
    for group_key in BUNDLE_BRANCH_GROUPS:
        index = label_order.index(group_key)
        other_key = next(key for key in BUNDLE_BRANCH_GROUPS if key != group_key)
        other_index = label_order.index(other_key)
        for architecture in analysis.ARCHITECTURES:
            for cohort in analysis.COHORTS:
                targets = loaded[cohort]["targets"]
                scores = loaded[cohort]["probabilities"][architecture][:, index]
                positive = targets[:, index].astype(bool)
                negative = ~positive
                both = positive & targets[:, other_index].astype(bool)
                keep = ~both
                threshold = float(np.percentile(scores[negative], 99))
                score_rows.append(
                    {
                        "label_key": group_key,
                        "diagnosis": diagnosis[group_key],
                        "architecture": architecture,
                        "cohort": cohort,
                        "positives": int(positive.sum()),
                        "median_score_positives": float(np.median(scores[positive])),
                        "median_score_negatives": float(np.median(scores[negative])),
                        "negative_99th_percentile": threshold,
                        "positives_below_negative_99th": int(np.sum(scores[positive] < threshold)),
                        "share_positives_below_negative_99th": float(
                            np.mean(scores[positive] < threshold)
                        ),
                        "records_with_both_blocks": int(both.sum()),
                        "auroc": float(weighted_auroc(positive, scores, unit[cohort])[0]),
                        "auroc_excluding_both_blocks": float(
                            weighted_auroc(
                                positive, scores, unit[cohort] * keep.astype(np.float64)[None, :]
                            )[0]
                        ),
                    }
                )

    # ---- per-seed spread ----------------------------------------------------------------
    seed_rows: list[dict[str, object]] = []
    for architecture in analysis.ARCHITECTURES:
        for cohort in analysis.COHORTS:
            targets = loaded[cohort]["targets"]
            per_seed = []
            for seed in SEEDS:
                path = args.prediction_root / cohort / f"{architecture}_seed_{seed}.npz"
                seed_scores = np.asarray(np.load(path)["probabilities"], dtype=np.float64)
                value = float(
                    macro_from_weights(targets, seed_scores, headline_indices, unit[cohort])[0]
                )
                per_seed.append(value)
                seed_rows.append(
                    {
                        "architecture": architecture,
                        "cohort": cohort,
                        "member": f"seed_{seed}",
                        "macro_auroc": value,
                    }
                )
            ensemble = float(
                macro_from_weights(
                    targets,
                    loaded[cohort]["probabilities"][architecture],
                    headline_indices,
                    unit[cohort],
                )[0]
            )
            seed_rows.append(
                {
                    "architecture": architecture,
                    "cohort": cohort,
                    "member": "ensemble",
                    "macro_auroc": ensemble,
                    "seed_standard_deviation": float(np.std(per_seed, ddof=1)),
                    "seed_range": float(max(per_seed) - min(per_seed)),
                    "gain_over_worst_seed": float(ensemble - min(per_seed)),
                    "gain_over_best_seed": float(ensemble - max(per_seed)),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    decomposition_path = args.output_root / "exploratory_class_side_decomposition.csv"
    standardization_path = args.output_root / "exploratory_demographic_standardization.csv"
    composition_path = args.output_root / "exploratory_bundle_branch_codes.csv"
    score_path = args.output_root / "exploratory_bundle_branch_scores.csv"
    seed_path = args.output_root / "exploratory_seed_spread.csv"
    contribution_path = args.output_root / "exploratory_macro_contributions.csv"
    pd.DataFrame(decomposition_rows).to_csv(decomposition_path, index=False)
    pd.DataFrame(standardization_rows).to_csv(standardization_path, index=False)
    pd.DataFrame(composition_rows).to_csv(composition_path, index=False)
    pd.DataFrame(score_rows).to_csv(score_path, index=False)
    pd.DataFrame(seed_rows).to_csv(seed_path, index=False)
    pd.DataFrame(contribution_rows).to_csv(contribution_path, index=False)

    summary = {
        "stage": "post-seal exploratory addition",
        "stage_date": EXPLORATORY_STAGE_DATE,
        "changes_registered_quantities": False,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seeds": dict(analysis.COHORT_SEEDS),
        "interval_type": "percentile",
        "multiplicity_adjustment": "none; marginal intervals, matching the registered plan",
        "registered_resample_cross_check": cross_checks,
        "decomposition_identity": (
            "positive_side + negative_side equals the registered per-label shift delta exactly"
        ),
        "largest_identity_residual": float(
            np.max(
                np.abs(
                    decomposition.positive_side
                    + decomposition.negative_side
                    - decomposition.shift_delta
                )
            )
        ),
        "prevalence_shift_test": prevalence_rows,
        "demographic_standardization": standardization_diagnostics,
        "multiplicity_note": (
            "the registered plan runs no multiplicity-adjusted tests; the "
            "Benjamini-Hochberg q-values in exploratory_macro_contributions.csv apply "
            "only to the post-seal exploratory per-label family of 26 shift deltas per "
            "architecture and adjust no registered quantity"
        ),
        "outputs": [
            decomposition_path.name,
            standardization_path.name,
            composition_path.name,
            score_path.name,
            seed_path.name,
            contribution_path.name,
        ],
    }
    summary_path = args.output_root / "exploratory_transfer_mechanism.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    for path in (
        decomposition_path,
        standardization_path,
        composition_path,
        score_path,
        seed_path,
        contribution_path,
        summary_path,
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
