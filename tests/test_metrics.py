import numpy as np

from ecg_clinical.metrics import (
    apply_temperature,
    calibration_summary,
    fit_scalar_temperature,
    select_f1_thresholds,
)


def test_threshold_tie_chooses_smallest_value() -> None:
    targets = np.asarray([[0], [1], [1], [0]])
    probabilities = np.asarray([[0.1], [0.4], [0.8], [0.9]])
    thresholds = select_f1_thresholds(targets, probabilities)
    assert thresholds.shape == (1,)
    assert thresholds[0] == 0.4


def test_temperature_fit_improves_overconfident_nll() -> None:
    targets = np.asarray([[0], [0], [1], [1]])
    probabilities = np.asarray([[0.01], [0.9], [0.1], [0.99]])
    temperature = fit_scalar_temperature(targets, probabilities)
    assert temperature > 1
    calibrated = apply_temperature(probabilities, temperature)
    assert calibrated[1, 0] < probabilities[1, 0]


def test_calibration_summary_is_finite() -> None:
    targets = np.asarray([[0, 1], [1, 0], [0, 1]])
    probabilities = np.asarray([[0.1, 0.8], [0.7, 0.2], [0.3, 0.9]])
    summary = calibration_summary(targets, probabilities, np.asarray([0, 1]))
    assert 0 <= summary["macro_per_label_ece"] <= 1
    assert 0 <= summary["pooled_ece"] <= 1
