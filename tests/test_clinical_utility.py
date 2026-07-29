"""Tests for the post-seal exploratory decision-curve and operating-point stage.

Two properties carry the weight. The utility arithmetic has to be the textbook net benefit
rather than something that resembles it, and the stage has to ride on the registered
resamples rather than drawing its own. Everything here is synthetic; nothing reads the
project's waveform stores, sealed predictions, or fold-9 ensembles.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import load_script

from ecg_clinical.bootstrap import paired_bootstrap_cohort_statistics

stage = load_script("exploratory_clinical_utility")


def _uninformative_cohort(
    groups: int = 100, per_group: int = 10, positives_per_group: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """A score that carries no information about the label.

    Every distinct score value holds the same positive fraction, so at any threshold the
    flagged set has exactly the cohort prevalence. The construction is exact rather than
    random, which lets the treat-all comparison be asserted without a sampling tolerance.
    """

    scores = np.repeat((np.arange(groups) + 0.5) / groups, per_group)
    block = np.zeros(per_group, dtype=np.uint8)
    block[:positives_per_group] = 1
    targets = np.tile(block, groups)
    return targets, scores


def test_net_benefit_of_a_perfect_classifier_approaches_prevalence() -> None:
    """With no false positives the odds term vanishes and net benefit is prevalence."""

    targets = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    curve = stage.decision_curve(targets, targets.astype(np.float64), np.array([1e-9, 0.01]))
    assert curve["net_benefit"] == pytest.approx([0.3, 0.3])
    assert list(curve["true_positives"]) == [3, 3]
    assert list(curve["false_positives"]) == [0, 0]


def test_treat_all_net_benefit_matches_hand_computed_values() -> None:
    """NB_all(t) = prevalence - (1 - prevalence) * t / (1 - t)."""

    assert stage.net_benefit_treat_all(0.2, 0.2) == pytest.approx(0.0)
    assert stage.net_benefit_treat_all(0.1, 0.05) == pytest.approx(0.05263157894736842)
    assert stage.net_benefit_treat_all(0.5, 0.5) == pytest.approx(0.0)
    assert stage.net_benefit_treat_all(0.4, 0.1) == pytest.approx(0.4 - 0.6 / 9.0)


def test_treat_none_is_the_reference_once_treat_all_turns_negative() -> None:
    reference = stage.net_benefit_reference(0.1, np.array([0.05, 0.2, 0.4]))
    assert reference[0] > 0.0
    assert reference[1] == 0.0
    assert reference[2] == 0.0


def test_uninformative_scores_never_beat_the_better_default() -> None:
    """A score independent of the label earns a fraction of the treat-all net benefit."""

    targets, scores = _uninformative_cohort()
    prevalence = float(targets.mean())
    curve = stage.decision_curve(targets, scores, stage.THRESHOLD_GRID)
    reference = stage.net_benefit_reference(prevalence, stage.THRESHOLD_GRID)
    assert np.all(curve["net_benefit"] <= reference + 1e-12)
    assert (
        stage.superior_threshold_summary(stage.THRESHOLD_GRID, curve["net_benefit"], reference)[
            "superior_threshold_count"
        ]
        == 0
    )


def test_threshold_grid_covers_one_to_fifty_percent_in_hundredths() -> None:
    assert len(stage.THRESHOLD_GRID) == 50
    assert stage.THRESHOLD_GRID[0] == pytest.approx(0.01)
    assert stage.THRESHOLD_GRID[-1] == pytest.approx(0.50)
    assert np.allclose(np.diff(stage.THRESHOLD_GRID), 0.01)
    assert stage.grid_position(stage.THRESHOLD_GRID, 0.10) == 9


def test_superior_threshold_summary_reports_endpoints_gaps_and_the_best_gain() -> None:
    thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    reference = np.zeros(5)
    contiguous = stage.superior_threshold_summary(
        thresholds, np.array([-1.0, 0.2, 0.5, 0.1, -1.0]), reference
    )
    assert contiguous["superior_threshold_count"] == 3
    assert contiguous["superior_threshold_min"] == 0.2
    assert contiguous["superior_threshold_max"] == 0.4
    assert contiguous["superior_thresholds_contiguous"] is True
    assert contiguous["max_net_benefit_gain"] == pytest.approx(0.5)
    assert contiguous["max_net_benefit_gain_threshold"] == 0.3

    gapped = stage.superior_threshold_summary(
        thresholds, np.array([0.1, -1.0, 0.2, -1.0, 0.3]), reference
    )
    assert gapped["superior_threshold_count"] == 3
    assert gapped["superior_thresholds_contiguous"] is False

    empty = stage.superior_threshold_summary(thresholds, np.full(5, -1.0), reference)
    assert empty["superior_threshold_count"] == 0
    assert empty["superior_threshold_min"] is None
    assert empty["superior_threshold_max"] is None
    assert empty["superior_thresholds_contiguous"] is None


def test_sensitivity_threshold_takes_the_largest_qualifying_observed_value() -> None:
    """Sensitivity is nonincreasing in the threshold, so the largest one is the tightest."""

    targets = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.55, 0.45, 0.35, 0.25, 0.15])

    threshold, attainable = stage.sensitivity_threshold(targets, scores, 0.90)
    assert attainable is True
    assert threshold == pytest.approx(0.2)

    relaxed, relaxed_attainable = stage.sensitivity_threshold(targets, scores, 0.80)
    assert relaxed_attainable is True
    assert relaxed == pytest.approx(0.6)

    complete, complete_attainable = stage.sensitivity_threshold(targets, scores, 1.0)
    assert complete_attainable is True
    assert complete == pytest.approx(0.2)


def test_sensitivity_threshold_reports_an_unattainable_floor_without_imputing() -> None:
    """A label with no positives has no defined sensitivity and gets no chosen threshold."""

    targets = np.zeros(6, dtype=np.uint8)
    scores = np.linspace(0.1, 0.6, 6)
    threshold, attainable = stage.sensitivity_threshold(targets, scores, 0.90)
    assert attainable is False
    assert threshold == 0.0

    impossible, impossible_attainable = stage.sensitivity_threshold(
        np.array([1, 0, 1, 0], dtype=np.uint8), np.array([0.4, 0.3, 0.2, 0.1]), 1.5
    )
    assert impossible_attainable is False
    assert impossible == 0.0


def test_operating_point_metrics_match_a_hand_built_confusion_matrix() -> None:
    counts = {"tp": 30.0, "fp": 20.0, "tn": 940.0, "fn": 10.0}
    metrics = stage.operating_point_metrics(counts)
    assert metrics["sensitivity"] == pytest.approx(0.75)
    assert metrics["specificity"] == pytest.approx(940 / 960)
    assert metrics["ppv"] == pytest.approx(0.6)
    assert metrics["npv"] == pytest.approx(940 / 950)
    assert metrics["alerts_per_1000_records"] == pytest.approx(50.0)
    assert metrics["alerts_per_true_positive"] == pytest.approx(50 / 30)
    assert metrics["records_screened_per_true_positive"] == pytest.approx(1000 / 30)
    assert metrics["missed_cases_per_1000_records"] == pytest.approx(10.0)


def test_operating_point_metrics_leave_undefined_rates_undefined() -> None:
    """No true positives means no alerts per true positive, and no imputed zero."""

    metrics = stage.operating_point_metrics({"tp": 0.0, "fp": 5.0, "tn": 90.0, "fn": 5.0})
    assert metrics["alerts_per_true_positive"] is None
    assert metrics["records_screened_per_true_positive"] is None
    assert metrics["sensitivity"] == pytest.approx(0.0)
    assert metrics["ppv"] == pytest.approx(0.0)

    without_positives = stage.operating_point_metrics(
        {"tp": 0.0, "fp": 0.0, "tn": 100.0, "fn": 0.0}
    )
    assert without_positives["sensitivity"] is None
    assert without_positives["ppv"] is None
    assert without_positives["specificity"] == pytest.approx(1.0)


def test_confusion_counts_under_unit_weights_equal_unweighted_counts() -> None:
    generator = np.random.default_rng(20260726)
    targets = generator.integers(0, 2, size=400).astype(np.uint8)
    scores = generator.uniform(0.0, 1.0, size=400)
    for threshold in (0.05, 0.2, 0.5, 0.9):
        unweighted = stage.confusion_counts(targets, scores, threshold)
        weighted = stage.confusion_counts(
            targets, scores, threshold, weights=np.ones(len(targets), dtype=np.float64)
        )
        assert unweighted == weighted
        assert sum(unweighted.values()) == len(targets)


def test_confusion_counts_scale_with_repeated_records() -> None:
    """A weight of two must count a record exactly twice, as a resample does."""

    targets = np.array([1, 1, 0, 0], dtype=np.uint8)
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    doubled = stage.confusion_counts(targets, scores, 0.5, weights=np.full(4, 2.0))
    assert doubled == {"tp": 2.0, "fp": 2.0, "tn": 2.0, "fn": 2.0}


def test_weighted_utility_totals_under_unit_weights_match_the_point_estimates() -> None:
    """The bootstrap path and the point-estimate path must agree at weight one."""

    generator = np.random.default_rng(4242)
    targets = (generator.uniform(size=600) < 0.25).astype(np.uint8)
    calibrated = generator.uniform(0.0, 1.0, size=600)
    uncalibrated = generator.uniform(0.0, 1.0, size=600)
    registered_threshold = 0.37

    basis = stage.utility_basis(
        targets, calibrated, uncalibrated, registered_threshold, stage.INTERVAL_THRESHOLDS
    )
    assert basis.shape == (600, stage.utility_basis_width(stage.INTERVAL_THRESHOLDS))
    weights = np.ones((1, 600), dtype=np.float64)
    totals = stage.weighted_totals(weights, basis)
    statistics = stage.utility_statistics_from_totals(
        totals, weights.sum(axis=1), stage.INTERVAL_THRESHOLDS
    )

    prevalence = float(targets.mean())
    curve = stage.decision_curve(targets, calibrated, np.asarray(stage.INTERVAL_THRESHOLDS))
    for position, threshold in enumerate(stage.INTERVAL_THRESHOLDS):
        expected = curve["net_benefit"][position]
        assert statistics[f"net_benefit__{threshold:.2f}"][0] == pytest.approx(expected)
        assert statistics[f"net_benefit_gain__{threshold:.2f}"][0] == pytest.approx(
            expected - stage.net_benefit_reference(prevalence, threshold)
        )

    counts = stage.confusion_counts(targets, uncalibrated, registered_threshold)
    point = stage.operating_point_metrics(counts)
    assert statistics["sensitivity"][0] == pytest.approx(point["sensitivity"])
    assert statistics["specificity"][0] == pytest.approx(point["specificity"])


def test_bootstrap_pass_reproduces_the_registered_macro_distribution_bit_for_bit() -> None:
    """The whole point of the stage: same resamples as the registered macro-AUROC.

    If this ever fails, an interval published beside a registered estimate would have come
    from a different bootstrap, which is exactly what the stage refuses to do.
    """

    generator = np.random.default_rng(20260726)
    records, labels = 150, 5
    targets = generator.integers(0, 2, size=(records, labels)).astype(np.uint8)
    first = generator.uniform(0.02, 0.98, size=(records, labels))
    second = np.clip(first + generator.normal(0, 0.2, size=(records, labels)), 0.01, 0.99)
    probabilities = {"first": first, "second": second}
    label_indices = np.array([0, 1, 3, 4], dtype=np.int64)
    group_inverse = np.arange(records, dtype=np.int64)

    registered = paired_bootstrap_cohort_statistics(
        targets,
        probabilities,
        label_indices,
        group_inverse,
        replicates=200,
        seed=8123,
        batch_size=50,
        include_calibration=False,
    )
    basis_by_model = {
        name: np.concatenate(
            [
                stage.utility_basis(
                    targets[:, label_index],
                    scores[:, label_index],
                    scores[:, label_index],
                    0.5,
                    stage.INTERVAL_THRESHOLDS,
                )
                for label_index in label_indices
            ],
            axis=1,
        )
        for name, scores in probabilities.items()
    }
    auroc, totals, total_weight = stage.bootstrap_utility_and_auroc(
        targets,
        probabilities,
        basis_by_model,
        label_indices,
        group_inverse,
        replicates=200,
        seed=8123,
        batch_size=50,
    )
    for name in probabilities:
        rebuilt = stage.macro_of_per_label(auroc[name], np.arange(len(label_indices)))
        assert np.array_equal(rebuilt, registered[name])
        assert totals[name].shape == (
            200,
            len(label_indices) * stage.utility_basis_width(stage.INTERVAL_THRESHOLDS),
        )
    assert np.all(total_weight == records)


def test_macro_of_per_label_averages_over_defined_labels_only() -> None:
    values = np.array([[0.8, np.nan, 0.6], [0.9, 0.7, 0.5]])
    observed = stage.macro_of_per_label(values, np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_allclose(observed, [0.7, 0.7])


def test_cluster_inverse_uses_patients_internally_and_records_externally() -> None:
    import pandas as pd

    metadata = pd.DataFrame({"patient_id": ["a", "a", "b", "c", "c"]})
    internal = stage.cluster_inverse("ptb_test", metadata)
    assert internal[0] == internal[1]
    assert internal[3] == internal[4]
    assert len(np.unique(internal)) == 3
    assert np.array_equal(stage.cluster_inverse("ningbo", metadata), np.arange(5))


def test_interval_columns_are_empty_away_from_the_interval_thresholds() -> None:
    intervals = {
        "0.10": {
            "net_benefit": {"point": 0.02, "lower_95": 0.01, "upper_95": 0.03},
            "net_benefit_gain": {"point": 0.005, "lower_95": 0.001, "upper_95": 0.009},
        }
    }
    attached = stage.interval_columns(intervals, 0.10)
    assert attached["net_benefit_calibrated_lower_95"] == 0.01
    assert attached["net_benefit_gain_upper_95"] == 0.009
    absent = stage.interval_columns(intervals, 0.11)
    assert set(absent.values()) == {None}
