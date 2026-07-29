"""Tests for the two post-seal exploratory shift-diagnosis stages.

The load-bearing properties are that the reconciled estimands change the target or the
evaluated record set and nothing else, that the class-side attribution sums exactly to
the registered per-label shift delta, and that rank matching leaves both within-cohort
AUROCs untouched. Everything here is synthetic; nothing reads the project's real
waveform stores or sealed predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import load_script
from sklearn.metrics import roc_auc_score

from ecg_clinical.bootstrap import bootstrap_record_weight_batches

commensurability = load_script("exploratory_label_commensurability")
mechanism = load_script("exploratory_transfer_mechanism")


def _cohort(records: int, labels: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    targets = (generator.uniform(size=(records, labels)) < 0.3).astype(np.uint8)
    scores = np.clip(
        0.2 * targets + generator.uniform(0.01, 0.85, size=(records, labels)), 0.01, 0.99
    )
    return targets, scores


# --------------------------------------------------------------------------------------
# reconciled estimands


def test_union_target_is_the_disjunction_and_restriction_removes_competing_records() -> None:
    targets = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )
    union, mask = commensurability.reconciled_targets(targets, 0, [1, 2])
    assert union.tolist() == [True, True, True, False, True]
    assert mask.tolist() == [True, False, False, True, False]


def test_reconciled_targets_without_competitors_is_the_identity() -> None:
    targets = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.uint8)
    union, mask = commensurability.reconciled_targets(targets, 0, [])
    assert union.tolist() == [True, False, False]
    assert mask.all()


def test_point_auroc_with_a_mask_equals_scoring_the_subset() -> None:
    targets, scores = _cohort(200, 3, seed=5)
    mask = np.zeros(200, dtype=bool)
    mask[:130] = True
    observed = commensurability.point_auroc(targets[:, 0], scores[:, 0], mask)
    expected = roc_auc_score(targets[:130, 0], scores[:130, 0])
    assert observed == pytest.approx(expected)


def test_restricted_replicates_equal_zero_weighting_the_excluded_records() -> None:
    """The restriction must ride on the registered resample, not on a fresh draw."""

    targets, scores = _cohort(150, 2, seed=7)
    group_inverse = np.arange(150, dtype=np.int64)
    mask = np.ones(150, dtype=bool)
    mask[::5] = False
    truth = targets[:, 0].astype(bool)

    observed = commensurability.reconciled_replicates(
        truth, scores[:, 0], mask, group_inverse, replicates=20, seed=31, batch_size=10
    )
    batches = list(bootstrap_record_weight_batches(group_inverse, 20, seed=31, batch_size=10))
    weights = np.concatenate([batch for _, batch in batches], axis=0)
    for replicate in range(20):
        combined = weights[replicate] * mask
        expected = roc_auc_score(truth, scores[:, 0], sample_weight=combined)
        assert observed[replicate] == pytest.approx(expected)


def test_label_structure_counts_rhythm_collisions() -> None:
    targets = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    row = commensurability.label_structure_row(
        "example", targets, np.array([0, 1, 2]), [0, 1, 2, 3]
    )
    assert row["records"] == 4
    assert row["mean_labels_per_record_all_targets"] == pytest.approx(1.25)
    assert row["mean_labels_per_record_headline"] == pytest.approx(1.0)
    assert row["share_no_label_all_targets"] == pytest.approx(0.25)
    # Only the third record is empty across the three headline columns; the fourth
    # carries the third column, so headline emptiness is one record in four.
    assert row["share_no_label_headline"] == pytest.approx(0.25)
    assert row["share_more_than_one_rhythm_label"] == pytest.approx(0.5)


def test_cooccurrence_reports_joint_counts_and_conditional_rates() -> None:
    targets = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=np.uint8)
    rows = commensurability.cooccurrence_rows(
        "example", targets, np.array([0, 1]), ["first", "second"]
    )
    lookup = {(row["row_label"], row["column_label"]): row for row in rows}
    assert lookup[("first", "second")]["joint_positives"] == 1
    assert lookup[("first", "second")]["conditional_column_given_row"] == pytest.approx(0.5)
    assert lookup[("first", "first")]["conditional_column_given_row"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# class-side decomposition


def test_shapley_split_sums_to_the_registered_shift_delta() -> None:
    generator = np.random.default_rng(3)
    a_ii, a_ee, a_ei, a_ie = (generator.uniform(0.5, 1.0, size=64) for _ in range(4))
    positive, negative = mechanism.shapley_split(a_ii, a_ee, a_ei, a_ie)
    np.testing.assert_allclose(positive + negative, a_ee - a_ii, atol=1e-12)


def test_shapley_split_assigns_a_pure_positive_move_to_the_positive_side() -> None:
    """Negatives held fixed means A(i, e) equals A(i, i) and the split is one-sided."""

    a_ii = np.array([0.90])
    a_ie = np.array([0.90])
    a_ei = np.array([0.70])
    a_ee = np.array([0.70])
    positive, negative = mechanism.shapley_split(a_ii, a_ee, a_ei, a_ie)
    assert positive[0] == pytest.approx(-0.20)
    assert negative[0] == pytest.approx(0.0)


def test_midranks_preserve_within_cohort_auroc_exactly() -> None:
    targets, scores = _cohort(300, 2, seed=11)
    transformed = mechanism.midrank_scores(scores[:, 0])
    before = roc_auc_score(targets[:, 0], scores[:, 0])
    after = roc_auc_score(targets[:, 0], transformed)
    assert after == pytest.approx(before, abs=1e-12)


def test_midranks_handle_ties_without_reordering() -> None:
    scores = np.array([0.2, 0.5, 0.5, 0.5, 0.9])
    transformed = mechanism.midrank_scores(scores)
    assert transformed[1] == transformed[2] == transformed[3]
    assert transformed[0] < transformed[1] < transformed[4]
    assert transformed[0] == pytest.approx(0.0)
    assert transformed[4] == pytest.approx(1.0)


def test_cross_cohort_auroc_matches_sklearn_on_the_concatenated_pair() -> None:
    generator = np.random.default_rng(17)
    positive_scores = generator.uniform(0.3, 1.0, size=40)
    negative_scores = generator.uniform(0.0, 0.7, size=90)
    positive_weights = generator.integers(0, 4, size=(3, 40)).astype(np.float64)
    negative_weights = generator.integers(0, 4, size=(3, 90)).astype(np.float64)
    observed = mechanism.cross_cohort_auroc(
        positive_scores, negative_scores, positive_weights, negative_weights
    )
    truth = np.concatenate([np.ones(40, dtype=bool), np.zeros(90, dtype=bool)])
    scores = np.concatenate([positive_scores, negative_scores])
    for replicate in range(3):
        weights = np.concatenate([positive_weights[replicate], negative_weights[replicate]])
        expected = roc_auc_score(truth, scores, sample_weight=weights)
        assert observed[replicate] == pytest.approx(expected)


def test_decomposition_identity_holds_on_a_synthetic_pair() -> None:
    """End to end: the four AUROCs of a real pair of cohorts satisfy the identity."""

    internal_targets, internal_scores = _cohort(400, 1, seed=21)
    external_targets, external_scores = _cohort(700, 1, seed=22)
    internal_truth = internal_targets[:, 0].astype(bool)
    external_truth = external_targets[:, 0].astype(bool)
    internal = mechanism.midrank_scores(internal_scores[:, 0])
    external = mechanism.midrank_scores(external_scores[:, 0])

    a_ii = roc_auc_score(internal_truth, internal)
    a_ee = roc_auc_score(external_truth, external)
    a_ei = mechanism.cross_cohort_auroc(
        external[external_truth],
        internal[~internal_truth],
        np.ones((1, external_truth.sum())),
        np.ones((1, (~internal_truth).sum())),
    )[0]
    a_ie = mechanism.cross_cohort_auroc(
        internal[internal_truth],
        external[~external_truth],
        np.ones((1, internal_truth.sum())),
        np.ones((1, (~external_truth).sum())),
    )[0]
    positive, negative = mechanism.shapley_split(
        np.array([a_ii]), np.array([a_ee]), np.array([a_ei]), np.array([a_ie])
    )
    assert positive[0] + negative[0] == pytest.approx(a_ee - a_ii, abs=1e-12)


# --------------------------------------------------------------------------------------
# demographic standardization


def _cell_labels(analysis, metadata: pd.DataFrame, cohort: str) -> np.ndarray:
    age = analysis.normalized_age(metadata)
    sex = analysis.normalized_sex(metadata, cohort)
    return np.asarray([f"{a}|{s}" for a, s in zip(age, sex, strict=True)])


def test_standardization_reproduces_the_internal_cell_shares_on_common_support() -> None:
    """With every cell populated in both cohorts, the reweighted joint mix must match."""

    analysis = load_script("analyze_results")
    internal = pd.DataFrame(
        {
            "age": [30] * 10 + [50] * 20 + [70] * 30 + [30] * 15 + [50] * 15 + [70] * 10,
            "sex": [0] * 60 + [1] * 40,
        }
    )
    external = pd.DataFrame(
        {
            "age": [30] * 40 + [50] * 10 + [70] * 10 + [30] * 10 + [50] * 10 + [70] * 20,
            "sex": ["Male"] * 60 + ["Female"] * 40,
        }
    )
    weights, diagnostics = mechanism.standardization_weights(internal, external, analysis, "ningbo")
    assert diagnostics["external_records"] == 100
    assert diagnostics["records_missing_a_stratifier"] == 0
    assert diagnostics["records_in_a_stratum_absent_from_ptb_fold_10"] == 0
    assert diagnostics["internal_mass_in_strata_absent_externally"] == pytest.approx(0.0)
    assert weights.sum() == pytest.approx(100.0)

    internal_cells = _cell_labels(analysis, internal, "ptb_test")
    external_cells = _cell_labels(analysis, external, "ningbo")
    for cell in np.unique(internal_cells):
        share = weights[external_cells == cell].sum() / weights.sum()
        assert share == pytest.approx(float(np.mean(internal_cells == cell)), abs=1e-9)


def test_standardization_separates_missing_stratifiers_from_unmatched_strata() -> None:
    """A record with no age and a record in an internally empty cell are not the same."""

    analysis = load_script("analyze_results")
    internal = pd.DataFrame({"age": [30, 50, 70, 50], "sex": [0, 1, 0, 1]})
    external = pd.DataFrame(
        {
            "age": [30, 50, 70, np.nan, 30],
            "sex": ["Male", "Female", "Male", "Female", "Female"],
        }
    )
    weights, diagnostics = mechanism.standardization_weights(
        internal, external, analysis, "chapman_shaoxing"
    )
    # Index 3 carries no age band. Index 4 is under-40 female, a cell this internal
    # frame does not populate, since its only under-40 record is male.
    assert weights[3] == 0.0
    assert weights[4] == 0.0
    assert diagnostics["records_missing_a_stratifier"] == 1
    assert diagnostics["records_in_a_stratum_absent_from_ptb_fold_10"] == 1
    assert diagnostics["records_carrying_weight"] == 3


def test_standardization_reports_internal_mass_it_cannot_represent() -> None:
    """An internal cell with no external counterpart is renormalized away and counted."""

    analysis = load_script("analyze_results")
    internal = pd.DataFrame({"age": [30, 30, 70, 70], "sex": [0, 0, 1, 1]})
    external = pd.DataFrame({"age": [30, 30, 30], "sex": ["Male", "Male", "Male"]})
    weights, diagnostics = mechanism.standardization_weights(internal, external, analysis, "ningbo")
    assert diagnostics["internal_mass_in_strata_absent_externally"] == pytest.approx(0.5)
    assert weights.sum() == pytest.approx(3.0)


def test_macro_from_weights_averages_over_defined_labels_only() -> None:
    targets = np.array([[1, 1], [0, 1], [1, 1], [0, 1]], dtype=np.uint8)
    scores = np.array([[0.9, 0.5], [0.1, 0.4], [0.8, 0.6], [0.2, 0.3]])
    observed = mechanism.macro_from_weights(targets, scores, np.array([0, 1]), np.ones((1, 4)))
    # The second label has no negatives, so it is undefined and drops out.
    assert observed[0] == pytest.approx(1.0)
