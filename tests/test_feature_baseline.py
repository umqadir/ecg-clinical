"""Tests for the post-seal handcrafted-feature baseline stage.

The load-bearing properties are that the estimators measure what they are named after, that
nothing outside the PTB-XL fitting split touches a fitted constant, and that a failed
detection is reported rather than quietly filled in. Everything here is synthetic apart from
the committed label manifest; nothing reads the project's waveform stores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import load_script

from ecg_clinical.bootstrap import bootstrap_record_weight_batches, weighted_roc_auc_batch

stage = load_script("exploratory_feature_baseline")

FS = stage.SAMPLING_HZ
MANIFEST_PATH = Path(__file__).parents[1] / "data/derived/preregistration/harmonized_labels.json"


def _beat_template(*, with_p: bool = True, with_t: bool = True, qrs_width: float = 0.012):
    """One beat sampled around its R peak, in millivolts."""

    offsets = np.arange(-int(0.40 * FS), int(0.60 * FS)) / FS
    wave = 1.20 * np.exp(-0.5 * np.square(offsets / qrs_width))
    wave = wave - 0.28 * np.exp(-0.5 * np.square((offsets - 3 * qrs_width) / (1.2 * qrs_width)))
    wave = wave - 0.10 * np.exp(-0.5 * np.square((offsets + 3 * qrs_width) / qrs_width))
    if with_p:
        wave = wave + 0.18 * np.exp(-0.5 * np.square((offsets + 0.16) / 0.022))
    if with_t:
        wave = wave + 0.30 * np.exp(-0.5 * np.square((offsets - 0.28) / 0.045))
    return offsets, wave


def synthetic_record(
    rate_bpm: float = 75.0,
    *,
    with_p: bool = True,
    with_t: bool = True,
    qrs_width: float = 0.012,
    lead_scales: np.ndarray | None = None,
    seconds: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A twelve-lead record built from a repeating beat, plus the true R-peak samples."""

    samples = int(seconds * FS)
    period = 60.0 * FS / rate_bpm
    peaks = np.rint(np.arange(0.6 * period, samples - 0.4 * period, period)).astype(np.int64)
    offsets, wave = _beat_template(with_p=with_p, with_t=with_t, qrs_width=qrs_width)
    trace = np.zeros(samples, dtype=np.float64)
    positions = np.rint(offsets * FS).astype(np.int64)
    for peak in peaks:
        indices = peak + positions
        inside = (indices >= 0) & (indices < samples)
        trace[indices[inside]] += wave[inside]
    if lead_scales is None:
        lead_scales = np.array([0.6, 1.0, 0.4, -0.8, 0.1, 0.7, -0.5, 0.3, 0.6, 0.9, 1.0, 0.8])
    return lead_scales[:, None] * trace[None, :], peaks


def test_r_peak_detection_recovers_a_known_heart_rate() -> None:
    """The detector has to find the planted beats, not merely find something periodic."""

    for rate in (48.0, 75.0, 132.0):
        record, peaks = synthetic_record(rate)
        detected = stage.detect_r_peaks(record[stage.LEAD_INDEX["II"]], FS)
        assert len(detected) == len(peaks)
        assert np.max(np.abs(detected - peaks)) <= 1
        assert stage.heart_rate_bpm(detected, FS) == pytest.approx(rate, abs=1.5)


def test_r_peak_detection_reports_nothing_on_a_flat_lead() -> None:
    assert stage.detect_r_peaks(np.zeros(1000), FS).size == 0
    assert np.isnan(stage.heart_rate_bpm(np.array([], dtype=np.int64), FS))


def test_detection_falls_back_to_the_largest_lead_when_lead_two_is_flat() -> None:
    scales = np.zeros(12)
    scales[stage.LEAD_INDEX["V4"]] = 1.0
    record, peaks = synthetic_record(72.0, lead_scales=scales)
    band = stage.detection_band(record, FS)
    assert stage.detection_lead_index(band) == stage.LEAD_INDEX["V4"]
    detected = stage.detect_r_peaks_in_band(band[stage.detection_lead_index(band)], FS)
    assert len(detected) == len(peaks)


@pytest.mark.parametrize(
    ("area_lead_i", "area_avf", "expected"),
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 90.0),
        (0.0, -1.0, -90.0),
        (-1.0, 0.0, 180.0),
        (1.0, 1.0, 45.0),
        (-1.0, -1.0, -135.0),
        (0.5, -0.5, -45.0),
    ],
)
def test_frontal_axis_follows_the_clinical_sign_convention(
    area_lead_i: float, area_avf: float, expected: float
) -> None:
    assert stage.frontal_axis_degrees(area_lead_i, area_avf) == pytest.approx(expected)


def test_frontal_axis_is_scale_free() -> None:
    assert stage.frontal_axis_degrees(0.3, 0.3) == pytest.approx(
        stage.frontal_axis_degrees(3.0, 3.0)
    )


def test_extracted_axis_matches_the_planted_lead_geometry() -> None:
    """Lead I positive with aVF silent must read near zero degrees, and the reverse near 90."""

    scales = np.zeros(12)
    scales[stage.LEAD_INDEX["I"]] = 1.0
    scales[stage.LEAD_INDEX["II"]] = 1.0
    record, _ = synthetic_record(70.0, lead_scales=scales)
    features = stage.extract_record_features(record, FS)
    assert features[stage.FEATURE_INDEX["frontal_axis_degrees"]] == pytest.approx(0.0, abs=12.0)

    scales = np.zeros(12)
    scales[stage.LEAD_INDEX["aVF"]] = 1.0
    scales[stage.LEAD_INDEX["II"]] = 1.0
    record, _ = synthetic_record(70.0, lead_scales=scales)
    features = stage.extract_record_features(record, FS)
    assert features[stage.FEATURE_INDEX["frontal_axis_degrees"]] == pytest.approx(90.0, abs=12.0)


def test_qrs_duration_grows_with_a_wider_planted_complex() -> None:
    narrow, _ = synthetic_record(70.0, qrs_width=0.010)
    wide, _ = synthetic_record(70.0, qrs_width=0.030)
    narrow_ms = stage.extract_record_features(narrow, FS)[stage.FEATURE_INDEX["qrs_duration_ms"]]
    wide_ms = stage.extract_record_features(wide, FS)[stage.FEATURE_INDEX["qrs_duration_ms"]]
    assert np.isfinite(narrow_ms) and np.isfinite(wide_ms)
    assert wide_ms > narrow_ms + 20.0


def test_heart_rate_feature_tracks_the_planted_rate() -> None:
    for rate in (50.0, 90.0, 140.0):
        record, _ = synthetic_record(rate)
        features = stage.extract_record_features(record, FS)
        assert features[stage.FEATURE_INDEX["heart_rate_bpm"]] == pytest.approx(rate, abs=2.0)
        assert features[stage.FEATURE_INDEX["r_peak_detection_failed"]] == 0.0


def test_missing_p_wave_is_flagged_rather_than_imputed() -> None:
    """A record with no P wave must report NaN and raise its indicator, not borrow a value."""

    absent, _ = synthetic_record(70.0, with_p=False)
    features = stage.extract_record_features(absent, FS)
    assert np.isnan(features[stage.FEATURE_INDEX["pr_interval_ms"]])
    assert features[stage.FEATURE_INDEX["pr_interval_missing"]] == 1.0

    present, _ = synthetic_record(70.0, with_p=True)
    features = stage.extract_record_features(present, FS)
    assert np.isfinite(features[stage.FEATURE_INDEX["pr_interval_ms"]])
    assert features[stage.FEATURE_INDEX["pr_interval_missing"]] == 0.0


def test_unusable_record_returns_all_indicators_raised() -> None:
    features = stage.extract_record_features(np.zeros((12, 1000)), FS)
    assert features[stage.FEATURE_INDEX["r_peak_detection_failed"]] == 1.0
    assert features[stage.FEATURE_INDEX["pr_interval_missing"]] == 1.0
    assert features[stage.FEATURE_INDEX["qt_interval_missing"]] == 1.0
    measured = [name for name in stage.FEATURE_NAMES if not name.endswith(("failed", "missing"))]
    assert all(np.isnan(features[stage.FEATURE_INDEX[name]]) for name in measured)


def test_every_feature_carries_a_one_line_estimator_description() -> None:
    assert tuple(stage.ESTIMATOR_DESCRIPTIONS) == stage.FEATURE_NAMES


def test_standardization_constants_come_from_the_fitting_split_only() -> None:
    """The holdout is transformed by the fitting constants and never contributes to them."""

    fitting = np.array([[0.0, 10.0], [2.0, 30.0]])
    holdout = np.array([[100.0, 1000.0], [200.0, 2000.0]])
    centre, scale = stage.fit_standardizer(fitting)
    np.testing.assert_allclose(centre, [1.0, 20.0])
    np.testing.assert_allclose(scale, [1.0, 10.0])

    combined_centre, combined_scale = stage.fit_standardizer(np.vstack([fitting, holdout]))
    assert not np.allclose(centre, combined_centre)
    assert not np.allclose(scale, combined_scale)

    standardized = stage.apply_standardizer(holdout, centre, scale)
    np.testing.assert_allclose(standardized, [[99.0, 98.0], [199.0, 198.0]])
    np.testing.assert_allclose(
        stage.apply_standardizer(fitting, centre, scale), [[-1.0, -1.0], [1.0, 1.0]]
    )


def test_standardizer_ignores_missing_entries_and_degenerate_columns() -> None:
    fitting = np.array([[1.0, 4.0], [np.nan, 4.0], [3.0, 4.0]])
    centre, scale = stage.fit_standardizer(fitting)
    np.testing.assert_allclose(centre, [2.0, 4.0])
    np.testing.assert_allclose(scale, [1.0, 1.0])


def test_missing_entries_are_replaced_by_the_fitting_centre_not_by_the_column_itself() -> None:
    centre = np.array([5.0, 7.0])
    scale = np.array([2.0, 4.0])
    matrix = np.array([[np.nan, 11.0], [9.0, np.nan]])
    standardized = stage.apply_standardizer(matrix, centre, scale)
    np.testing.assert_allclose(standardized, [[0.0, 1.0], [2.0, 0.0]])


def test_label_column_order_matches_the_manifest_and_the_store_builder() -> None:
    """The stage reads targets.npy positionally, so its column order must be the built one."""

    manifest = json.loads(MANIFEST_PATH.read_text())
    order = stage.label_column_order(manifest)
    assert order == [entry["label_key"] for entry in manifest["labels"]]
    assert len(order) == len(set(order))

    positions = stage.headline_positions(manifest)
    assert [order[index] for index in positions] == list(manifest["headline_label_keys"])
    assert sorted(stage.FINDING_CLASSES) == sorted(manifest["headline_label_keys"])

    builder = load_script("build_waveform_store")
    diagnoses = pd.Series([entry["snomed_codes"][0] for entry in manifest["labels"]])
    matrix = builder.label_matrix(diagnoses, MANIFEST_PATH)
    np.testing.assert_array_equal(matrix, np.eye(len(order), dtype=np.uint8))


def test_univariate_auroc_orients_the_feature_and_drops_missing_values() -> None:
    scores = np.array([1.0, 2.0, 3.0, np.nan])
    targets = np.array([0, 0, 1, 1])
    assert stage.univariate_auroc(scores, targets, 1.0) == pytest.approx(1.0)
    assert stage.univariate_auroc(scores, targets, -1.0) == pytest.approx(0.0)
    assert np.isnan(stage.univariate_auroc(scores, np.zeros(4, dtype=np.int64), 1.0))


def test_cluster_inverse_uses_patients_internally_and_records_externally() -> None:
    metadata = pd.DataFrame({"patient_id": ["a", "a", "b", "c", "c"]})
    internal = stage.cluster_inverse("ptb_test", metadata)
    assert internal[0] == internal[1]
    assert internal[3] == internal[4]
    assert len(np.unique(internal)) == 3
    np.testing.assert_array_equal(stage.cluster_inverse("ningbo", metadata), np.arange(5))


def test_competing_rhythm_mask_matches_whole_codes_only() -> None:
    """A substring hit would silently drop records from the rate self-check's negatives."""

    metadata = pd.DataFrame(
        {
            "diagnoses": [
                "164889003|426783006",  # atrial fibrillation
                "426783006",  # sinus rhythm alone
                "427084000|164934002",  # sinus tachycardia, no competing rhythm
                None,
                "1164889003",  # a longer code that merely contains one
            ]
        }
    )
    observed = stage.competing_rhythm_mask(metadata)
    np.testing.assert_array_equal(observed, [True, False, False, False, False])


def test_rate_checks_cover_the_two_rate_defined_diagnoses() -> None:
    labels = {label for label, _, _, _ in stage.RATE_CHECKS}
    assert labels == {"426177001", "427084000"}
    assert all(side in ("below", "above") for *_, side in stage.RATE_CHECKS)


def test_macro_over_labels_averages_defined_labels_only() -> None:
    values = np.array([[0.8, np.nan, 0.6], [0.9, 0.7, 0.5]])
    observed = stage.macro_over_labels(values, np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_allclose(observed, [0.7, 0.7])


def test_per_label_bootstrap_rides_on_the_registered_resamples() -> None:
    """The replicate values must be the registered weights applied to this model's scores."""

    generator = np.random.default_rng(20260726)
    targets = generator.integers(0, 2, size=(80, 3)).astype(np.uint8)
    scores = generator.uniform(0.01, 0.99, size=(80, 3))
    group_inverse = np.arange(80, dtype=np.int64)
    observed = stage.per_label_bootstrap(
        targets, scores, group_inverse, replicates=40, seed=20260716, batch_size=20
    )
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, 40, seed=20260716, batch_size=20
    ):
        for index in range(3):
            expected = weighted_roc_auc_batch(targets[:, index], scores[:, index], weights)
            np.testing.assert_array_equal(
                observed[start : start + len(weights), index], expected
            )


def test_point_auroc_matches_the_unweighted_estimator() -> None:
    targets = np.array([[0], [0], [1], [1]], dtype=np.uint8)
    scores = np.array([[0.1], [0.4], [0.35], [0.8]])
    assert stage.point_auroc(targets, scores)[0] == pytest.approx(0.75)


def test_regularization_selection_stays_on_the_grid_and_uses_the_selection_split() -> None:
    generator = np.random.default_rng(7)
    fit_features = generator.normal(size=(200, 4))
    fit_targets = (fit_features[:, 0] + 0.2 * generator.normal(size=200) > 0).astype(np.uint8)
    select_features = generator.normal(size=(120, 4))
    select_targets = (select_features[:, 0] > 0).astype(np.uint8)
    strength, model, auroc = stage.select_regularization(
        fit_features, fit_targets, select_features, select_targets
    )
    assert strength in stage.REGULARIZATION_GRID
    assert auroc > 0.9
    assert model.predict_proba(select_features).shape == (120, 2)


def test_row_helper_emits_every_declared_column() -> None:
    record = stage.row(section="macro_performance", cohort="ningbo", value=0.5)
    assert tuple(record) == stage.CSV_COLUMNS
    assert record["section"] == "macro_performance"
    assert record["label_key"] == ""
