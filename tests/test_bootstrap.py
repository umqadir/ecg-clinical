import numpy as np
from sklearn.metrics import roc_auc_score

from ecg_clinical.bootstrap import (
    _weighted_calibration_statistics,
    bootstrap_record_weight_batches,
    paired_bootstrap_cohort_statistics,
    paired_bootstrap_macro_auroc,
    weighted_brier_batch,
    weighted_expected_calibration_error_batch,
    weighted_roc_auc_batch,
)
from ecg_clinical.metrics import calibration_summary, expected_calibration_error


def _example_cohort(records: int = 90, labels: int = 4) -> tuple[np.ndarray, dict, dict]:
    generator = np.random.default_rng(2026)
    targets = generator.integers(0, 2, size=(records, labels)).astype(np.uint8)
    first = generator.uniform(0.02, 0.98, size=(records, labels))
    second = np.clip(first + generator.normal(0, 0.1, size=(records, labels)), 0.01, 0.99)
    probabilities = {"first": first, "second": second}
    calibrated = {name: np.clip(value * 0.8 + 0.1, 0, 1) for name, value in probabilities.items()}
    return targets, probabilities, calibrated


def test_weighted_auc_matches_sklearn_with_ties() -> None:
    targets = np.asarray([0, 1, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.4, 0.8, 0.9])
    weights = np.asarray([[1, 2, 3, 1, 2], [2, 1, 0, 4, 1]])
    observed = weighted_roc_auc_batch(targets, probabilities, weights)
    expected = np.asarray(
        [
            roc_auc_score(targets, probabilities, sample_weight=replicate_weights)
            for replicate_weights in weights
        ]
    )
    np.testing.assert_allclose(observed, expected)


def test_cluster_bootstrap_assigns_equal_weights_within_cluster() -> None:
    groups = np.asarray([0, 0, 1, 2, 2])
    _, weights = next(bootstrap_record_weight_batches(groups, 3, seed=5, batch_size=3))
    assert np.array_equal(weights[:, 0], weights[:, 1])
    assert np.array_equal(weights[:, 3], weights[:, 4])


def test_paired_bootstrap_returns_model_difference() -> None:
    targets = np.asarray([[0], [0], [1], [1]])
    first = np.asarray([[0.1], [0.2], [0.8], [0.9]])
    second = np.asarray([[0.2], [0.3], [0.7], [0.8]])
    output = paired_bootstrap_macro_auroc(
        targets,
        {"first": first, "second": second},
        np.asarray([0]),
        np.arange(4),
        replicates=10,
        seed=7,
        batch_size=5,
    )
    np.testing.assert_allclose(output["second_minus_first"], output["second"] - output["first"])


def test_weighted_calibration_reduces_to_unweighted_at_unit_weights() -> None:
    targets, probabilities, _ = _example_cohort()
    unit = np.ones((1, len(targets)))
    for label_index in range(targets.shape[1]):
        observed_ece = weighted_expected_calibration_error_batch(
            targets[:, label_index], probabilities["first"][:, label_index], unit
        )
        observed_brier = weighted_brier_batch(
            targets[:, label_index], probabilities["first"][:, label_index], unit
        )
        np.testing.assert_allclose(
            observed_ece,
            expected_calibration_error(
                targets[:, label_index], probabilities["first"][:, label_index]
            ),
        )
        np.testing.assert_allclose(
            observed_brier,
            np.mean(np.square(probabilities["first"][:, label_index] - targets[:, label_index])),
        )


def test_weighted_calibration_matches_duplicated_dataset_under_integer_weights() -> None:
    targets, probabilities, _ = _example_cohort(records=40, labels=1)
    truth = targets[:, 0]
    scores = probabilities["first"][:, 0]
    generator = np.random.default_rng(11)
    counts = generator.integers(0, 4, size=len(truth))
    counts[0] = 3
    duplicated_truth = np.repeat(truth, counts)
    duplicated_scores = np.repeat(scores, counts)

    observed_ece = weighted_expected_calibration_error_batch(
        truth, scores, counts[None, :].astype(np.float64)
    )
    observed_brier = weighted_brier_batch(truth, scores, counts[None, :].astype(np.float64))

    np.testing.assert_allclose(
        observed_ece, expected_calibration_error(duplicated_truth, duplicated_scores)
    )
    np.testing.assert_allclose(
        observed_brier, np.mean(np.square(duplicated_scores - duplicated_truth))
    )


def test_cohort_statistics_macro_calibration_matches_metrics_at_unit_weights() -> None:
    targets, probabilities, calibrated = _example_cohort()
    label_indices = np.asarray([0, 1, 3])
    output = paired_bootstrap_cohort_statistics(
        targets,
        probabilities,
        label_indices,
        np.arange(len(targets)),
        replicates=1,
        seed=3,
        batch_size=1,
        calibrated_by_model=calibrated,
    )
    assert set(output) >= {
        "first__uncalibrated__macro_per_label_ece",
        "second__calibrated__pooled_ece",
        "second_minus_first__calibrated__macro_brier",
    }

    # The statistic itself is checked against the registered unweighted
    # implementation by evaluating the same code path at unit weights.
    unit = np.ones((1, len(targets)))
    for table in (probabilities, calibrated):
        for values in table.values():
            statistics = _weighted_calibration_statistics(targets, values, label_indices, unit)
            reference = calibration_summary(targets, values, label_indices)
            for metric in ("macro_per_label_ece", "macro_brier", "pooled_ece"):
                np.testing.assert_allclose(statistics[metric], reference[metric])
            assert statistics["contributing_labels"][0] == len(label_indices)


def test_cohort_statistics_reproduce_macro_auroc_exactly() -> None:
    targets, probabilities, calibrated = _example_cohort()
    label_indices = np.asarray([0, 1, 2, 3])
    groups = np.repeat(np.arange(len(targets) // 3), 3)
    reference = paired_bootstrap_macro_auroc(
        targets,
        probabilities,
        label_indices,
        groups,
        replicates=40,
        seed=20260716,
        batch_size=13,
    )
    observed = paired_bootstrap_cohort_statistics(
        targets,
        probabilities,
        label_indices,
        groups,
        replicates=40,
        seed=20260716,
        batch_size=13,
        calibrated_by_model=calibrated,
    )
    for name, values in reference.items():
        np.testing.assert_array_equal(observed[name], values)
    np.testing.assert_array_equal(
        observed["first__macro_auroc_labels"], np.full(40, len(label_indices))
    )


def test_cohort_statistics_contrasts_are_paired() -> None:
    targets, probabilities, calibrated = _example_cohort()
    label_indices = np.asarray([0, 2])
    output = paired_bootstrap_cohort_statistics(
        targets,
        probabilities,
        label_indices,
        np.arange(len(targets)),
        replicates=30,
        seed=5,
        batch_size=10,
        calibrated_by_model=calibrated,
    )
    for metric in ("macro_per_label_ece", "macro_brier", "pooled_ece"):
        for model_name in ("first", "second"):
            np.testing.assert_array_equal(
                output[f"{model_name}__calibrated_minus_uncalibrated__{metric}"],
                output[f"{model_name}__calibrated__{metric}"]
                - output[f"{model_name}__uncalibrated__{metric}"],
            )
        np.testing.assert_array_equal(
            output[f"second_minus_first__uncalibrated__{metric}"],
            output["second__uncalibrated__" + metric] - output["first__uncalibrated__" + metric],
        )
    # A paired contrast has to be strictly narrower than an unpaired one here,
    # which only holds if both terms rode on the same resample.
    assert np.std(output["second__calibrated_minus_uncalibrated__macro_brier"]) < np.std(
        output["second__calibrated__macro_brier"]
    )


def test_cohort_statistics_without_calibration_yields_only_discrimination() -> None:
    targets, probabilities, _ = _example_cohort()
    label_indices = np.asarray([0, 1])
    groups = np.arange(len(targets))
    output = paired_bootstrap_cohort_statistics(
        targets,
        probabilities,
        label_indices,
        groups,
        replicates=10,
        seed=1,
        batch_size=10,
        include_calibration=False,
    )
    assert not [name for name in output if "ece" in name or "brier" in name]
    reference = paired_bootstrap_macro_auroc(
        targets, probabilities, label_indices, groups, replicates=10, seed=1, batch_size=10
    )
    for name, values in reference.items():
        np.testing.assert_array_equal(output[name], values)
    assert output["first__macro_auroc_labels"].tolist() == [2] * 10
