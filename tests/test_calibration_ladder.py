"""Tests for the post-seal exploratory calibration ladder.

The load-bearing properties are that the prior-shift algebra is exact where theory says it
is exact, that the two deployable prevalence estimators recover a known shifted prevalence,
and that the binning and floor constructions behave as the ladder's argument assumes.
Everything here is synthetic; nothing reads the project's real waveform stores or sealed
predictions.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import load_script
from scipy.special import expit, logit

from ecg_clinical.metrics import expected_calibration_error

stage = load_script("exploratory_calibration_ladder")

# A two-class problem on a five-point score support. The class-conditional distributions
# are fixed, so only the prior changes between source and target, which is exactly the
# label-shift assumption the ladder's rungs 7 to 9 rely on.
SUPPORT_POSITIVE = np.array([0.05, 0.10, 0.20, 0.25, 0.40])
SUPPORT_NEGATIVE = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
SOURCE_PREVALENCE = 0.12
TARGET_PREVALENCE = 0.37


def _source_posterior(prevalence: float = SOURCE_PREVALENCE) -> np.ndarray:
    """The exact posterior a perfectly calibrated source model would emit."""

    joint_positive = prevalence * SUPPORT_POSITIVE
    joint_negative = (1 - prevalence) * SUPPORT_NEGATIVE
    return joint_positive / (joint_positive + joint_negative)


def _target_marginal(prevalence: float = TARGET_PREVALENCE) -> np.ndarray:
    return prevalence * SUPPORT_POSITIVE + (1 - prevalence) * SUPPORT_NEGATIVE


def test_additive_offset_recovers_the_target_prevalence_exactly() -> None:
    """Rung 7 algebra: the offset turns the source posterior into the target posterior.

    Under label shift the corrected posterior is the true target posterior, so its mean
    under the target marginal is the target prevalence itself. This is the identity the
    prevalence-only intercept rung claims, checked without any sampling.
    """

    posterior = _source_posterior()
    offset = stage.prior_shift_offset(SOURCE_PREVALENCE, TARGET_PREVALENCE)
    corrected = expit(logit(posterior) + offset)
    marginal = _target_marginal()

    assert float(np.dot(marginal, corrected)) == pytest.approx(TARGET_PREVALENCE, abs=1e-12)


def test_additive_offset_is_the_identity_when_the_prevalence_does_not_move() -> None:
    assert stage.prior_shift_offset(0.3, 0.3) == pytest.approx(0.0, abs=1e-15)


def test_sld_em_recovers_a_known_shifted_prevalence() -> None:
    """Rung 8: EM on unlabeled target scores alone recovers the shifted prior.

    The sample is built to match the target marginal to within integer rounding at 200000
    records, so the estimate is compared at a tolerance of 1e-4 rather than at a sampling
    tolerance.
    """

    posterior = _source_posterior()
    counts = np.round(_target_marginal() * 200_000).astype(int)
    scores = np.repeat(posterior, counts)
    estimate, iterations = stage.sld_em_prevalence(logit(scores), SOURCE_PREVALENCE)

    assert estimate == pytest.approx(TARGET_PREVALENCE, abs=1e-4)
    assert 1 < iterations < 1000


def test_sld_em_stays_at_the_source_prevalence_with_no_shift() -> None:
    posterior = _source_posterior()
    counts = np.round(_target_marginal(SOURCE_PREVALENCE) * 200_000).astype(int)
    scores = np.repeat(posterior, counts)
    estimate, _ = stage.sld_em_prevalence(logit(scores), SOURCE_PREVALENCE)

    assert estimate == pytest.approx(SOURCE_PREVALENCE, abs=1e-4)


def test_bbse_hard_inverts_a_known_confusion_matrix() -> None:
    """Rung 9: given the source confusion matrix, the flagged rate fixes the prevalence."""

    true_positive_rate, false_positive_rate = 0.82, 0.11
    for prevalence in (0.02, 0.15, 0.5, 0.9):
        flagged = true_positive_rate * prevalence + false_positive_rate * (1 - prevalence)
        estimate = stage.bbse_hard_prevalence(true_positive_rate, false_positive_rate, flagged)
        assert estimate == pytest.approx(prevalence, abs=1e-12)


def test_bbse_hard_refuses_an_uninformative_predictor() -> None:
    assert np.isnan(stage.bbse_hard_prevalence(0.4, 0.4, 0.4))


def test_confusion_rates_match_a_hand_counted_table() -> None:
    targets = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    flags = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    true_positive_rate, false_positive_rate = stage.confusion_rates(targets, flags)

    assert true_positive_rate == pytest.approx(0.75)
    assert false_positive_rate == pytest.approx(1 / 6)


def test_equal_mass_bins_hold_near_equal_counts_on_continuous_scores() -> None:
    """Section (b): the quantile rule must actually equalize mass, not just edges."""

    scores = np.random.default_rng(20260726).uniform(0.0, 1.0, size=1500)
    assignments = stage.equal_mass_bin_assignments(scores, bins=15)
    counts = np.bincount(assignments, minlength=15)

    assert len(counts) == 15
    assert counts.sum() == 1500
    assert counts.min() >= 99
    assert counts.max() <= 101


def test_equal_mass_bins_keep_a_tie_group_in_one_bin() -> None:
    """Ties collapse edges, so the realized bin count drops rather than splitting a tie."""

    scores = np.concatenate([np.full(900, 0.2), np.linspace(0.3, 0.9, 600)])
    assignments = stage.equal_mass_bin_assignments(scores, bins=15)

    assert len(np.unique(assignments[:900])) == 1
    assert len(np.unique(assignments)) < 15


def test_equal_mass_and_equal_width_agree_on_a_perfectly_calibrated_score() -> None:
    generator = np.random.default_rng(11)
    scores = generator.uniform(0.05, 0.95, size=40_000)
    targets = (generator.uniform(size=40_000) < scores).astype(np.float64)

    equal_width = expected_calibration_error(targets, scores, bins=15)
    equal_mass = stage.equal_mass_expected_calibration_error(targets, scores, bins=15)

    assert equal_width < 0.01
    assert equal_mass < 0.01


def test_constant_predictor_at_the_true_prevalence_has_zero_error() -> None:
    """Section (c): the floor is exactly zero, and the naive baseline is the shift itself."""

    targets = np.concatenate([np.ones(37), np.zeros(63)])
    prevalence = float(targets.mean())
    matrix = stage.constant_predictor_matrix(np.array([prevalence]), len(targets))

    assert matrix.shape == (100, 1)
    assert len(np.unique(stage.equal_mass_bin_assignments(matrix[:, 0], bins=15))) == 1
    assert expected_calibration_error(targets, matrix[:, 0], bins=15) == pytest.approx(
        0.0, abs=1e-15
    )

    source = 0.10
    naive = stage.constant_predictor_matrix(np.array([source]), len(targets))
    assert expected_calibration_error(targets, naive[:, 0], bins=15) == pytest.approx(
        abs(prevalence - source), abs=1e-15
    )


def test_per_label_offset_fitting_reduces_the_negative_log_likelihood() -> None:
    """Rung 5: fitting the offset can only lower NLL, and it does so under prior shift."""

    generator = np.random.default_rng(4242)
    logits = generator.normal(-2.0, 1.5, size=6000)
    targets = (generator.uniform(size=6000) < expit(logits + 1.8)).astype(np.float64)

    slope, offset = stage.fit_logit_affine(targets, logits, fit_slope=False)
    fitted = stage.binary_nll(targets, logits, slope, offset)
    unfitted = stage.binary_nll(targets, logits, 1.0, 0.0)

    assert slope == 1.0
    assert fitted < unfitted
    assert offset == pytest.approx(1.8, abs=0.15)


def test_offset_fit_matches_the_mean_calibration_stationary_condition() -> None:
    """The offset-only optimum sets the mean predicted probability to the prevalence."""

    generator = np.random.default_rng(7)
    logits = generator.normal(-1.0, 2.0, size=4000)
    targets = (generator.uniform(size=4000) < expit(logits - 0.9)).astype(np.float64)

    _, offset = stage.fit_logit_affine(targets, logits, fit_slope=False)

    assert float(expit(logits + offset).mean()) == pytest.approx(float(targets.mean()), abs=1e-5)


def test_affine_fit_reaches_a_lower_likelihood_than_offset_or_slope_alone() -> None:
    generator = np.random.default_rng(2026)
    logits = generator.normal(0.0, 2.0, size=8000)
    targets = (generator.uniform(size=8000) < expit(0.6 * logits + 1.1)).astype(np.float64)

    _, offset_only = stage.fit_logit_affine(targets, logits, fit_slope=False)
    slope, offset = stage.fit_logit_affine(targets, logits, fit_slope=True)

    assert stage.binary_nll(targets, logits, slope, offset) <= stage.binary_nll(
        targets, logits, 1.0, offset_only
    )
    assert slope == pytest.approx(0.6, abs=0.1)
    assert offset == pytest.approx(1.1, abs=0.15)


def test_degenerate_label_keeps_the_identity_transform() -> None:
    logits = np.linspace(-3.0, 3.0, 40)
    for targets in (np.zeros(40), np.ones(40)):
        assert stage.fit_logit_affine(targets, logits, fit_slope=True) == (1.0, 0.0)
        assert stage.fit_logit_affine(targets, logits, fit_slope=False) == (1.0, 0.0)


def test_binary_nll_gradient_matches_a_finite_difference() -> None:
    generator = np.random.default_rng(5)
    logits = generator.normal(0.0, 1.5, size=500)
    targets = (generator.uniform(size=500) < 0.3).astype(np.float64)
    parameters = np.array([0.8, -0.4])

    _, gradient = stage.binary_nll_and_gradient(parameters, targets, logits)
    step = 1e-6
    for index in range(2):
        forward = parameters.copy()
        backward = parameters.copy()
        forward[index] += step
        backward[index] -= step
        numeric = (
            stage.binary_nll_and_gradient(forward, targets, logits)[0]
            - stage.binary_nll_and_gradient(backward, targets, logits)[0]
        ) / (2 * step)
        assert gradient[index] == pytest.approx(numeric, abs=1e-7)


def test_apply_logit_affine_transforms_only_the_named_columns() -> None:
    probabilities = np.array([[0.10, 0.40, 0.70], [0.25, 0.55, 0.85]])
    label_indices = np.array([0, 2], dtype=np.int64)
    transformed = stage.apply_logit_affine(
        probabilities, label_indices, np.array([1.0, 2.0]), np.array([0.5, -1.0])
    )

    assert np.array_equal(transformed[:, 1], probabilities[:, 1])
    np.testing.assert_allclose(transformed[:, 0], expit(logit(probabilities[:, 0]) + 0.5))
    np.testing.assert_allclose(transformed[:, 2], expit(2.0 * logit(probabilities[:, 2]) - 1.0))


def test_safe_logit_clips_at_the_bound_apply_temperature_uses() -> None:
    values = stage.safe_logit(np.array([0.0, 1.0, 0.5]))

    assert np.isfinite(values).all()
    assert values[0] == pytest.approx(float(logit(stage.PROBABILITY_CLIP)))
    assert values[2] == pytest.approx(0.0)


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

    external = stage.cluster_inverse("ningbo", metadata)
    assert np.array_equal(external, np.arange(5))


def test_recovery_seed_is_distinct_per_cell_and_fixed_by_the_base_seed() -> None:
    assert stage.recovery_seed(0, 0, 100)[0] == stage.RECOVERY_SEED
    assert stage.recovery_seed(1, 0, 100) != stage.recovery_seed(0, 1, 100)
    assert stage.recovery_seed(1, 1, 250) != stage.recovery_seed(1, 1, 500)

    first = np.random.default_rng(stage.recovery_seed(2, 1, 500)).integers(0, 10_000, size=8)
    second = np.random.default_rng(stage.recovery_seed(2, 1, 500)).integers(0, 10_000, size=8)
    assert np.array_equal(first, second)


def test_held_out_platt_recovery_scores_records_it_did_not_fit_on() -> None:
    """Fitting on k records and scoring the rest must beat leaving the shift uncorrected."""

    generator = np.random.default_rng(20260726)
    records, labels = 3000, 4
    logits = generator.normal(-1.5, 1.8, size=(records, labels))
    targets = (generator.uniform(size=(records, labels)) < expit(logits + 2.0)).astype(np.uint8)
    probabilities = expit(logits)
    headline_indices = np.arange(labels, dtype=np.int64)

    entry = stage.held_out_platt_recovery(
        targets,
        probabilities,
        headline_indices,
        sample_size=500,
        repeats=8,
        seed_entropy=stage.recovery_seed(0, 0, 500),
    )
    uncorrected = float(
        np.mean(
            [
                expected_calibration_error(targets[:, index], probabilities[:, index], bins=15)
                for index in headline_indices
            ]
        )
    )

    assert entry["sample_size"] == 500
    assert entry["repeats"] == 8
    assert entry["degenerate_label_fits"] == 0
    assert entry["percentile_2_5"] <= entry["mean"] <= entry["percentile_97_5"]
    assert entry["mean"] < uncorrected


def test_sealed_temperature_reproduces_a_float32_artifact_bit_for_bit() -> None:
    """Rung 2 must land on the sealed calibrated array, which was stored as float32."""

    from ecg_clinical.metrics import apply_temperature

    generator = np.random.default_rng(3)
    ensemble = generator.uniform(0.001, 0.999, size=(400, 5)).astype(np.float32)
    temperature = 1.0407298082428036
    sealed = apply_temperature(ensemble, temperature).astype(np.float32)

    rebuilt = stage.sealed_temperature_probabilities(
        np.asarray(ensemble, dtype=np.float64), temperature
    )
    assert rebuilt.dtype == np.float64
    assert np.array_equal(rebuilt, sealed.astype(np.float64))
