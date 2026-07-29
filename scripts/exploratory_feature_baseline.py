#!/usr/bin/env python3
"""Handcrafted-feature logistic-regression baseline for the cross-hospital transfer split.

POST-SEAL EXPLORATORY ADDITION, 2026-07-26. This stage was not part of the frozen analysis.
It changes no registered quantity, rewrites no sealed artifact, and re-runs no sealed
inference. It reads waveforms and targets, fits its own transparent model, and writes only
results/exploratory_feature_baseline.csv and results/exploratory_feature_baseline.json. The
preregistration seal is verified before any waveform is opened. The evaluation seal is not
required because no sealed prediction file is read; the committed
results/exploratory_per_label_shift_deltas.csv is read for the model comparison and is left
untouched.

What it tests. The study reports that geometric findings transport across hospitals and that
interpretive findings do not. That statement is an interpretation of two deep models'
per-label behaviour. A logistic regression over explicit geometric measurements tests it
directly: if the same split appears in a model whose only inputs are a rate, an axis, a
width, an interval, and named amplitudes, the split belongs to the finding rather than to
the network. The stage also supplies the simple reference model the repository lacks, so the
external macro-AUROC of the deep models can be read against a floor.

Fitting discipline. Features are extracted by one code path with no per-cohort tuning.
Standardization constants, per-label regularization strength, and every fitted coefficient
come from PTB-XL only: folds 1 to 8 fit, fold 9 selects. Fold 10 and the two external
cohorts are scored once. The external cohorts never enter fitting, standardization, or
selection, and are never pooled with each other.

Uncertainty. Intervals come from the registered cluster bootstrap design: 2000 replicates,
batch size 100, per-cohort seeds from analyze_results.COHORT_SEEDS, cluster unit patient_id
for PTB-XL fold 10 and record for the external cohorts, drawn with
bootstrap_record_weight_batches and evaluated with weighted_roc_auc_batch. These are the
registered seeds, cluster units, replicate count and batch size, so the intervals are
commensurable with the deep models'. They are not bit-for-bit the registered resamples of
any sealed model, because a different model's scores are being bootstrapped on the same
records.

Estimators, one line each.
heart_rate_bpm: 60 divided by the median RR interval of the accepted R peaks.
rr_standard_deviation_ms: standard deviation of the accepted RR intervals.
rr_interquartile_range_ms: 75th minus 25th percentile of the accepted RR intervals.
qrs_duration_ms: width of the median-beat QRS energy envelope at 5 percent of its peak
    height above the envelope noise floor, with linearly interpolated crossings.
frontal_axis_degrees: atan2 of the baseline-corrected median-beat QRS area in aVF against
    the same area in lead I, in degrees.
frontal_axis_sin: sine of the frontal axis, so the regression sees a continuous circle.
frontal_axis_cos: cosine of the frontal axis, so the regression sees a continuous circle.
pr_interval_ms: median-beat lead II P-wave onset, taken where the P deflection first exceeds
    20 percent of its peak inside a 220 ms search window, to the QRS onset.
qt_interval_ms: QRS onset to the tangent-method T-wave end on the median-beat lead II, the
    tangent taken at the steepest post-peak T slope and extended to the isoelectric line.
r_amplitude_*: largest positive median-beat deflection inside the QRS bounds, floored at zero.
s_amplitude_*: largest negative median-beat deflection inside the QRS bounds, capped at zero.
t_amplitude_*: signed median-beat value at the largest absolute deflection of the T window.
r_peak_detection_failed: 1 when fewer than two R peaks were accepted, so no RR interval exists.
pr_interval_missing: 1 when no P wave passed the amplitude and interval checks.
qt_interval_missing: 1 when the T-wave tangent gave no usable baseline intersection.

Missing values are emitted as NaN and carried by the three indicator features above. They are
never silently imputed: the design matrix replaces a standardized NaN with the fitting-split
mean, which is zero after standardization, and the indicator tells the regression that the
replacement happened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from ecg_clinical.bootstrap import (
    bootstrap_record_weight_batches,
    percentile_interval,
    weighted_roc_auc_batch,
)
from ecg_clinical.data import WaveformStore
from ecg_clinical.integrity import verify_preregistration_seal

EXPLORATORY_STAGE_DATE = "2026-07-26"
SAMPLING_HZ = 100
LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LEAD_INDEX = {name: index for index, name in enumerate(LEAD_NAMES)}

BASELINE_HIGHPASS_HZ = 0.5
DETECTION_BAND_HZ = (5.0, 25.0)
FLAT_LEAD_VARIANCE_MV2 = 1e-4
MIN_RR_SECONDS = 0.24
INTEGRATION_SECONDS = 0.15
QRS_ENERGY_FRACTION = 0.30
FIDUCIAL_SEARCH_SECONDS = 0.12

MEDIAN_BEAT_PRE = 40
MEDIAN_BEAT_POST = 60
ENVELOPE_SMOOTHING_SECONDS = 0.03
QRS_THRESHOLD_FRACTION = 0.05
QRS_HALF_LIMIT_SECONDS = 0.14
ISOELECTRIC_SECONDS = 0.04

P_SEARCH_SECONDS = 0.22
P_GUARD_SECONDS = 0.03
P_MIN_AMPLITUDE_MV = 0.05
P_ONSET_FRACTION = 0.20
PR_RANGE_MS = (60.0, 400.0)

T_GUARD_SECONDS = 0.04
T_SEARCH_SECONDS = 0.50
QT_RANGE_MS = (200.0, 700.0)

AMPLITUDE_LEADS = ("I", "aVF", "V1", "V6")
T_AMPLITUDE_LEADS = ("II", "V5")

FEATURE_NAMES = (
    "heart_rate_bpm",
    "rr_standard_deviation_ms",
    "rr_interquartile_range_ms",
    "qrs_duration_ms",
    "frontal_axis_degrees",
    "frontal_axis_sin",
    "frontal_axis_cos",
    "pr_interval_ms",
    "qt_interval_ms",
    "r_amplitude_i_mv",
    "s_amplitude_i_mv",
    "r_amplitude_avf_mv",
    "s_amplitude_avf_mv",
    "r_amplitude_v1_mv",
    "s_amplitude_v1_mv",
    "r_amplitude_v6_mv",
    "s_amplitude_v6_mv",
    "t_amplitude_ii_mv",
    "t_amplitude_v5_mv",
    "r_peak_detection_failed",
    "pr_interval_missing",
    "qt_interval_missing",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

ESTIMATOR_DESCRIPTIONS = {
    "heart_rate_bpm": "60 divided by the median RR interval of the accepted R peaks",
    "rr_standard_deviation_ms": "standard deviation of the accepted RR intervals",
    "rr_interquartile_range_ms": "75th minus 25th percentile of the accepted RR intervals",
    "qrs_duration_ms": (
        "width of the median-beat QRS energy envelope at 5 percent of its peak height "
        "above the envelope noise floor, with linearly interpolated crossings"
    ),
    "frontal_axis_degrees": (
        "atan2 of the baseline-corrected median-beat QRS area in aVF against the same "
        "area in lead I, in degrees"
    ),
    "frontal_axis_sin": "sine of the frontal axis",
    "frontal_axis_cos": "cosine of the frontal axis",
    "pr_interval_ms": (
        "median-beat lead II P-wave onset, taken where the P deflection first exceeds 20 "
        "percent of its peak inside a 220 ms search window, to the QRS onset"
    ),
    "qt_interval_ms": (
        "QRS onset to the tangent-method T-wave end on the median-beat lead II, the tangent "
        "taken at the steepest post-peak T slope and extended to the isoelectric line"
    ),
    "r_amplitude_i_mv": "largest positive median-beat lead I deflection inside the QRS bounds",
    "s_amplitude_i_mv": "largest negative median-beat lead I deflection inside the QRS bounds",
    "r_amplitude_avf_mv": "largest positive median-beat aVF deflection inside the QRS bounds",
    "s_amplitude_avf_mv": "largest negative median-beat aVF deflection inside the QRS bounds",
    "r_amplitude_v1_mv": "largest positive median-beat V1 deflection inside the QRS bounds",
    "s_amplitude_v1_mv": "largest negative median-beat V1 deflection inside the QRS bounds",
    "r_amplitude_v6_mv": "largest positive median-beat V6 deflection inside the QRS bounds",
    "s_amplitude_v6_mv": "largest negative median-beat V6 deflection inside the QRS bounds",
    "t_amplitude_ii_mv": "signed median-beat lead II value at the largest T-window deflection",
    "t_amplitude_v5_mv": "signed median-beat V5 value at the largest T-window deflection",
    "r_peak_detection_failed": "1 when fewer than two R peaks were accepted",
    "pr_interval_missing": "1 when no P wave passed the amplitude and interval checks",
    "qt_interval_missing": "1 when the T-wave tangent gave no usable baseline intersection",
}

COHORTS = ("ptb_test", "chapman_shaoxing", "ningbo")
EXTERNAL_COHORTS = ("chapman_shaoxing", "ningbo")
COHORT_STORE_NAMES = {
    "ptb_test": "ptb_xl",
    "chapman_shaoxing": "chapman_shaoxing",
    "ningbo": "ningbo",
}
FIT_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
SELECT_FOLD = (9,)
TEST_FOLD = (10,)
REGULARIZATION_GRID = (0.01, 0.1, 1.0, 10.0)
COMPARISON_ARCHITECTURE = "xresnet1d101"

# Declared before the features were extracted. Each pair names a feature, the label whose
# definition that feature encodes, and the sign that orients the feature so that a larger
# score means a more positive record.
VALIDATION_CHECKS = (
    ("heart_rate_bpm", "426177001", -1.0),
    ("heart_rate_bpm", "427084000", 1.0),
    ("frontal_axis_degrees", "39732003", -1.0),
    ("frontal_axis_degrees", "47665007", 1.0),
    ("qrs_duration_ms", "733534002|164909002", 1.0),
    ("qrs_duration_ms", "713427006|59118001", 1.0),
    ("pr_interval_ms", "164947007", 1.0),
    ("qt_interval_ms", "111975006", 1.0),
)

# Raw CinC codes for fast rhythms that are not sinus tachycardia. A record carrying one of
# these is fast by a mechanism no rate measurement can tell from sinus tachycardia, so the
# rate self-check reports its discrimination with and without those records among the
# negatives. Without the split, a low rate AUROC reads as a broken detector when it is a
# property of what the label names.
COMPETING_TACHYARRHYTHMIA_CODES = frozenset(
    {
        "164889003",  # atrial fibrillation
        "164890007",  # atrial flutter
        "195080001",  # atrial fibrillation and flutter
        "426749004",  # chronic atrial fibrillation
        "426761007",  # supraventricular tachycardia
        "713422000",  # atrial tachycardia
        "233896004",  # av node reentrant tachycardia
        "233897008",  # av reentrant tachycardia
        "164895002",  # ventricular tachycardia
    }
)

# Label, feature orientation, the rate in beats per minute that defines the diagnosis, and
# the side of it a positive record has to be measured on.
RATE_CHECKS = (
    ("426177001", -1.0, 60.0, "below"),
    ("427084000", 1.0, 100.0, "above"),
)

# The split the study asserts, written down so the comparison can be scored rather than
# eyeballed. Geometric findings name a measurement; interpretive findings name a reading.
# The two headline labels that are neither are left unclassified rather than forced.
FINDING_CLASSES = {
    "733534002|164909002": "geometric",
    "713427006|59118001": "geometric",
    "270492004": "geometric",
    "39732003": "geometric",
    "47665007": "geometric",
    "426177001": "geometric",
    "427084000": "geometric",
    "698252002": "interpretive",
    "426783006": "interpretive",
    "164934002": "interpretive",
    "59931005": "interpretive",
    "284470004|63593006": "unclassified",
    "164917005": "unclassified",
}

CSV_COLUMNS = (
    "section",
    "cohort",
    "external_cohort",
    "label_key",
    "diagnosis",
    "headline",
    "finding_class",
    "feature",
    "orientation",
    "metric",
    "value",
    "lower_95",
    "upper_95",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Handcrafted-feature baseline, post-seal.")
    parser.add_argument("--store-root", type=Path, default=Path("data/cache/waveforms"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/derived/preregistration/harmonized_labels.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--feature-cache", type=Path, default=Path("data/cache/exploratory_features.npz")
    )
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=100)
    parser.add_argument("--extraction-batch", type=int, default=256)
    return parser.parse_args()


def load_analysis_module() -> ModuleType:
    """Import the registered analysis stage to reuse its per-cohort bootstrap seeds."""

    path = Path(__file__).resolve().parent / "analyze_results.py"
    spec = importlib.util.spec_from_file_location("analyze_results", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the analysis stage from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=8)
def _highpass_coefficients(fs: int) -> tuple[np.ndarray, np.ndarray]:
    return butter(2, BASELINE_HIGHPASS_HZ / (fs / 2), btype="highpass")


@lru_cache(maxsize=8)
def _bandpass_coefficients(fs: int) -> tuple[np.ndarray, np.ndarray]:
    low, high = DETECTION_BAND_HZ
    return butter(2, [low / (fs / 2), high / (fs / 2)], btype="bandpass")


def remove_baseline(signals: np.ndarray, fs: int = SAMPLING_HZ) -> np.ndarray:
    """Zero-phase 0.5 Hz highpass, applied along the last axis."""

    numerator, denominator = _highpass_coefficients(fs)
    return filtfilt(numerator, denominator, signals, axis=-1)


def detection_band(signals: np.ndarray, fs: int = SAMPLING_HZ) -> np.ndarray:
    """Zero-phase 5 to 25 Hz bandpass, applied along the last axis."""

    numerator, denominator = _bandpass_coefficients(fs)
    return filtfilt(numerator, denominator, signals, axis=-1)


def detection_lead_index(band: np.ndarray) -> int:
    """Lead II unless it is flat, in which case the lead with the largest QRS-band energy."""

    energies = band.var(axis=-1)
    lead_two = LEAD_INDEX["II"]
    if energies[lead_two] < FLAT_LEAD_VARIANCE_MV2:
        return int(np.argmax(energies))
    return lead_two


def detect_r_peaks_in_band(band_lead: np.ndarray, fs: int = SAMPLING_HZ) -> np.ndarray:
    """Pan-Tompkins style R-peak indices from one already bandpassed lead.

    The squared derivative is smoothed over 150 ms, candidate maxima are separated by at
    least 240 ms, and a candidate is accepted when its integrated energy reaches 30 percent
    of the median energy of the upper quartile of candidates. Each accepted candidate is
    then moved to the largest absolute deflection of the bandpassed lead within 120 ms,
    which is the fiducial point the median beat is built on.
    """

    if band_lead.ndim != 1:
        raise ValueError("detection expects a single lead")
    derivative = np.diff(band_lead, prepend=band_lead[:1])
    integrated = uniform_filter1d(
        np.square(derivative), size=max(1, int(round(INTEGRATION_SECONDS * fs)))
    )
    candidates, _ = find_peaks(integrated, distance=max(1, int(round(MIN_RR_SECONDS * fs))))
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    heights = integrated[candidates]
    upper = heights[heights >= np.percentile(heights, 75.0)]
    reference = float(np.median(upper)) if upper.size else float(np.max(heights))
    if not np.isfinite(reference) or reference <= 0.0:
        return np.empty(0, dtype=np.int64)
    accepted = candidates[heights >= QRS_ENERGY_FRACTION * reference]
    if accepted.size == 0:
        return np.empty(0, dtype=np.int64)
    half = max(1, int(round(FIDUCIAL_SEARCH_SECONDS * fs)))
    magnitude = np.abs(band_lead)
    located = []
    for candidate in accepted:
        start = max(0, int(candidate) - half)
        stop = min(len(band_lead), int(candidate) + half + 1)
        located.append(start + int(np.argmax(magnitude[start:stop])))
    return np.unique(np.asarray(located, dtype=np.int64))


def detect_r_peaks(lead: np.ndarray, fs: int = SAMPLING_HZ) -> np.ndarray:
    """R-peak indices for one raw lead, bandpassing it first."""

    return detect_r_peaks_in_band(detection_band(np.asarray(lead, dtype=np.float64), fs), fs)


def rr_intervals_ms(peaks: np.ndarray, fs: int = SAMPLING_HZ) -> np.ndarray:
    """RR intervals in milliseconds; empty when fewer than two peaks were accepted."""

    if len(peaks) < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(np.asarray(peaks, dtype=np.float64)) * (1000.0 / fs)


def heart_rate_bpm(peaks: np.ndarray, fs: int = SAMPLING_HZ) -> float:
    """Heart rate from the median RR interval."""

    intervals = rr_intervals_ms(peaks, fs)
    if intervals.size == 0:
        return float("nan")
    median = float(np.median(intervals))
    if median <= 0.0:
        return float("nan")
    return 60000.0 / median


def median_beat(record: np.ndarray, peaks: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Ensemble median over the detected beats, aligned on the fiducial point.

    Beats whose window falls outside the record are dropped. When no beat has a complete
    window the indices are clipped to the record instead, so a record with one edge beat
    still yields a morphology estimate rather than nothing.
    """

    samples = record.shape[-1]
    offsets = np.arange(-pre, post, dtype=np.int64)
    peaks = np.asarray(peaks, dtype=np.int64)
    complete = peaks[(peaks - pre >= 0) & (peaks + post <= samples)]
    if complete.size:
        windows = record[:, complete[:, None] + offsets[None, :]]
    else:
        clipped = np.clip(peaks[:, None] + offsets[None, :], 0, samples - 1)
        windows = record[:, clipped]
    return np.median(windows, axis=1)


def qrs_energy_envelope(beat: np.ndarray, fs: int = SAMPLING_HZ) -> np.ndarray:
    """Smoothed root-sum-square of the twelve-lead median-beat derivative."""

    derivative = np.diff(beat, axis=-1, prepend=beat[:, :1])
    energy = np.sqrt(np.square(derivative).sum(axis=0))
    return uniform_filter1d(energy, size=max(1, int(round(ENVELOPE_SMOOTHING_SECONDS * fs))))


def _threshold_crossing(
    values: np.ndarray, start: int, step: int, threshold: float, limit: int, *, strict: bool = False
) -> float:
    """Fractional index where `values` first falls below `threshold` walking from `start`.

    The crossing is linearly interpolated between the bracketing samples. When no crossing
    occurs within `limit` samples, a saturating search returns the boundary and a strict
    search returns NaN. QRS bounds saturate, because a very wide complex is still a wide
    complex. The P-wave onset is strict, because a deflection that never returns toward the
    isoelectric line inside the search window is not a measured P wave.
    """

    index = start
    for _ in range(limit):
        following = index + step
        if following < 0 or following >= len(values):
            return float("nan") if strict else float(index)
        if values[following] < threshold:
            span = values[index] - values[following]
            if span <= 0.0:
                return float(following)
            fraction = (values[index] - threshold) / span
            return float(index) + step * float(fraction)
        index = following
    return float("nan") if strict else float(index)


def qrs_bounds(
    envelope: np.ndarray, fiducial: int, fs: int = SAMPLING_HZ
) -> tuple[float, float]:
    """Fractional QRS onset and offset indices from the energy envelope."""

    half = max(1, int(round(0.08 * fs)))
    start = max(0, fiducial - half)
    stop = min(len(envelope), fiducial + half + 1)
    if stop <= start:
        return float("nan"), float("nan")
    peak = start + int(np.argmax(envelope[start:stop]))
    floor = float(np.percentile(envelope, 20.0))
    amplitude = float(envelope[peak]) - floor
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        return float("nan"), float("nan")
    threshold = floor + QRS_THRESHOLD_FRACTION * amplitude
    limit = max(1, int(round(QRS_HALF_LIMIT_SECONDS * fs)))
    onset = _threshold_crossing(envelope, peak, -1, threshold, limit)
    offset = _threshold_crossing(envelope, peak, 1, threshold, limit)
    if not (onset < offset):
        return float("nan"), float("nan")
    return onset, offset


def isoelectric_levels(beat: np.ndarray, onset: float, fs: int = SAMPLING_HZ) -> np.ndarray:
    """Per-lead isoelectric level from the 40 ms of median beat before the QRS onset."""

    width = max(1, int(round(ISOELECTRIC_SECONDS * fs)))
    stop = max(1, int(np.floor(onset)))
    start = max(0, stop - width)
    return np.median(beat[:, start:stop], axis=1)


def frontal_axis_degrees(area_lead_i: float, area_avf: float) -> float:
    """Frontal-plane QRS axis in degrees from the lead I and aVF net QRS areas.

    Lead I positive with no aVF component is 0 degrees, aVF positive alone is +90, aVF
    negative alone is -90, and lead I negative alone is 180.
    """

    return float(np.degrees(np.arctan2(area_avf, area_lead_i)))


def p_wave_onset(
    lead_beat: np.ndarray, onset: float, lookback: int, fs: int = SAMPLING_HZ
) -> float:
    """Fractional P-wave onset index, or NaN when no P wave is present.

    The search runs from the QRS onset back by `lookback` samples, stopping 30 ms short of
    the QRS. The largest absolute deflection in that window is the P peak; it must clear
    0.05 mV and must not sit on either edge of the window, which is what a leading T wave or
    a trailing QRS shoulder would produce. The onset is where the deflection first reaches
    20 percent of the P peak, walking back from the peak, and the walk must reach that level
    inside the window for the interval to count.
    """

    guard = max(1, int(round(P_GUARD_SECONDS * fs)))
    stop = int(np.floor(onset)) - guard
    start = max(0, stop - lookback)
    if stop - start < 4:
        return float("nan")
    segment = lead_beat[start:stop]
    magnitude = np.abs(segment)
    local = int(np.argmax(magnitude))
    if local == 0 or local == len(segment) - 1:
        return float("nan")
    amplitude = float(magnitude[local])
    if not np.isfinite(amplitude) or amplitude < P_MIN_AMPLITUDE_MV:
        return float("nan")
    threshold = P_ONSET_FRACTION * amplitude
    crossing = _threshold_crossing(magnitude, local, -1, threshold, local, strict=True)
    if not np.isfinite(crossing):
        return float("nan")
    return float(start) + crossing


def t_wave_end(
    lead_beat: np.ndarray, offset: float, search_stop: int, fs: int = SAMPLING_HZ
) -> tuple[float, float]:
    """T peak index and tangent-method T-wave end index, either of which may be NaN.

    The T peak is the largest absolute deflection between 40 ms after the QRS offset and
    `search_stop`. The tangent is taken at the steepest slope after the peak and extended to
    the isoelectric line; the intersection is the T end.
    """

    guard = max(1, int(round(T_GUARD_SECONDS * fs)))
    start = int(np.ceil(offset)) + guard
    stop = min(len(lead_beat), search_stop)
    if stop - start < 4:
        return float("nan"), float("nan")
    segment = lead_beat[start:stop]
    peak = int(np.argmax(np.abs(segment)))
    if peak >= len(segment) - 2:
        return float(start + peak), float("nan")
    tail = segment[peak:]
    slopes = np.diff(tail)
    descending = -np.sign(tail[0]) * slopes
    step = int(np.argmax(descending))
    slope = float(slopes[step])
    value = float(tail[step])
    if slope == 0.0 or not np.isfinite(slope):
        return float(start + peak), float("nan")
    intersection = float(start + peak + step) - value / slope
    if intersection <= float(start + peak):
        return float(start + peak), float("nan")
    return float(start + peak), intersection


def extract_record_features(record: np.ndarray, fs: int = SAMPLING_HZ) -> np.ndarray:
    """Feature vector in FEATURE_NAMES order for one twelve-lead record in millivolts."""

    features = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float64)
    features[FEATURE_INDEX["r_peak_detection_failed"]] = 1.0
    features[FEATURE_INDEX["pr_interval_missing"]] = 1.0
    features[FEATURE_INDEX["qt_interval_missing"]] = 1.0

    signal = np.asarray(record, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] != len(LEAD_NAMES):
        raise ValueError("a record must be twelve leads by samples")
    if not np.isfinite(signal).all():
        return features

    corrected = remove_baseline(signal, fs)
    band = detection_band(signal, fs)
    lead = detection_lead_index(band)
    peaks = detect_r_peaks_in_band(band[lead], fs)
    if peaks.size == 0:
        return features

    intervals = rr_intervals_ms(peaks, fs)
    if intervals.size:
        features[FEATURE_INDEX["r_peak_detection_failed"]] = 0.0
        features[FEATURE_INDEX["heart_rate_bpm"]] = heart_rate_bpm(peaks, fs)
        features[FEATURE_INDEX["rr_standard_deviation_ms"]] = float(np.std(intervals))
        features[FEATURE_INDEX["rr_interquartile_range_ms"]] = float(
            np.percentile(intervals, 75.0) - np.percentile(intervals, 25.0)
        )
        median_rr = float(np.median(intervals)) * fs / 1000.0
    else:
        median_rr = float(MEDIAN_BEAT_POST)

    beat = median_beat(corrected, peaks, MEDIAN_BEAT_PRE, MEDIAN_BEAT_POST)
    envelope = qrs_energy_envelope(beat, fs)
    onset, offset = qrs_bounds(envelope, MEDIAN_BEAT_PRE, fs)
    if not np.isfinite(onset) or not np.isfinite(offset):
        return features

    beat = beat - isoelectric_levels(beat, onset, fs)[:, None]
    features[FEATURE_INDEX["qrs_duration_ms"]] = (offset - onset) * 1000.0 / fs

    first = int(np.floor(onset))
    last = int(np.ceil(offset)) + 1
    window = beat[:, first:last]
    areas = window.sum(axis=1) / fs
    axis = frontal_axis_degrees(float(areas[LEAD_INDEX["I"]]), float(areas[LEAD_INDEX["aVF"]]))
    features[FEATURE_INDEX["frontal_axis_degrees"]] = axis
    features[FEATURE_INDEX["frontal_axis_sin"]] = float(np.sin(np.radians(axis)))
    features[FEATURE_INDEX["frontal_axis_cos"]] = float(np.cos(np.radians(axis)))

    for name in AMPLITUDE_LEADS:
        trace = window[LEAD_INDEX[name]]
        suffix = name.lower()
        features[FEATURE_INDEX[f"r_amplitude_{suffix}_mv"]] = float(max(trace.max(), 0.0))
        features[FEATURE_INDEX[f"s_amplitude_{suffix}_mv"]] = float(min(trace.min(), 0.0))

    lookback = min(
        int(round(P_SEARCH_SECONDS * fs)), max(1, int(round(0.5 * median_rr))), MEDIAN_BEAT_PRE
    )
    p_onset = p_wave_onset(beat[LEAD_INDEX["II"]], onset, lookback, fs)
    if np.isfinite(p_onset):
        interval = (onset - p_onset) * 1000.0 / fs
        if PR_RANGE_MS[0] <= interval <= PR_RANGE_MS[1]:
            features[FEATURE_INDEX["pr_interval_ms"]] = interval
            features[FEATURE_INDEX["pr_interval_missing"]] = 0.0

    search_stop = min(
        beat.shape[1],
        int(np.ceil(offset)) + int(round(T_SEARCH_SECONDS * fs)),
        MEDIAN_BEAT_PRE + max(4, int(round(0.70 * median_rr))),
    )
    for name in T_AMPLITUDE_LEADS:
        peak_index, _ = t_wave_end(beat[LEAD_INDEX[name]], offset, search_stop, fs)
        if np.isfinite(peak_index):
            features[FEATURE_INDEX[f"t_amplitude_{name.lower()}_mv"]] = float(
                beat[LEAD_INDEX[name], int(peak_index)]
            )
    _, t_end = t_wave_end(beat[LEAD_INDEX["II"]], offset, search_stop, fs)
    if np.isfinite(t_end):
        interval = (t_end - onset) * 1000.0 / fs
        if QT_RANGE_MS[0] <= interval <= QT_RANGE_MS[1]:
            features[FEATURE_INDEX["qt_interval_ms"]] = interval
            features[FEATURE_INDEX["qt_interval_missing"]] = 0.0
    return features


def extract_cohort_features(
    store: WaveformStore, *, batch: int, fs: int = SAMPLING_HZ, label: str = ""
) -> np.ndarray:
    """Run the extractor over every record of one store, reading the signals in blocks."""

    records = len(store.metadata)
    output = np.full((records, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    for start in range(0, records, batch):
        stop = min(records, start + batch)
        block = np.asarray(store.signals[start:stop], dtype=np.float64)
        for offset in range(stop - start):
            output[start + offset] = extract_record_features(block[offset], fs)
        if (start // batch) % 20 == 0:
            print(f"  {label} {stop}/{records} records", flush=True)
    return output


def store_digest(root: Path) -> str:
    """Digest of the signal matrix a cached feature block was extracted from."""

    digest = hashlib.sha256()
    with (root / "signals.npy").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_standardizer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column centre and scale from the fitting split only, ignoring missing entries."""

    with np.errstate(invalid="ignore"):
        centre = np.nanmean(matrix, axis=0)
        scale = np.nanstd(matrix, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return centre, scale


def apply_standardizer(
    matrix: np.ndarray, centre: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Standardize and replace a missing entry with the fitting-split centre.

    The replacement is visible to the model: every column that can go missing has its own
    indicator feature, so a substituted value is never mistaken for a measured one.
    """

    standardized = (np.asarray(matrix, dtype=np.float64) - centre) / scale
    return np.where(np.isfinite(standardized), standardized, 0.0)


def univariate_auroc(scores: np.ndarray, targets: np.ndarray, orientation: float) -> float:
    """AUROC of one oriented feature, with missing values dropped."""

    scores = np.asarray(scores, dtype=np.float64) * orientation
    targets = np.asarray(targets, dtype=np.int64)
    usable = np.isfinite(scores)
    scores = scores[usable]
    targets = targets[usable]
    if targets.size == 0 or np.unique(targets).size < 2:
        return float("nan")
    weights = np.ones((1, len(scores)), dtype=np.float64)
    return float(weighted_roc_auc_batch(targets, scores, weights)[0])


def point_auroc(targets: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per-label AUROC point estimates through the registered weighted estimator."""

    weights = np.ones((1, targets.shape[0]), dtype=np.float64)
    return np.asarray(
        [
            weighted_roc_auc_batch(targets[:, index], scores[:, index], weights)[0]
            for index in range(targets.shape[1])
        ]
    )


def per_label_bootstrap(
    targets: np.ndarray,
    scores: np.ndarray,
    group_inverse: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Per-label AUROC replicates on the registered cluster resampling design."""

    output = np.empty((replicates, targets.shape[1]), dtype=np.float64)
    for start, weights in bootstrap_record_weight_batches(
        group_inverse, replicates, seed=seed, batch_size=batch_size
    ):
        end = start + len(weights)
        for index in range(targets.shape[1]):
            output[start:end, index] = weighted_roc_auc_batch(
                targets[:, index], scores[:, index], weights
            )
    return output


def macro_over_labels(distribution: np.ndarray, label_indices: np.ndarray) -> np.ndarray:
    """Registered macro rule: average over the labels defined in each replicate."""

    selected = np.ascontiguousarray(distribution[:, label_indices])
    contributing = np.isfinite(selected).sum(axis=1)
    return np.divide(
        np.nansum(selected, axis=1),
        contributing,
        out=np.full(len(selected), np.nan),
        where=contributing > 0,
    )


def competing_rhythm_mask(metadata: pd.DataFrame) -> np.ndarray:
    """Records whose diagnosis list names a fast rhythm other than sinus tachycardia."""

    codes = metadata.diagnoses.fillna("").map(lambda value: set(str(value).split("|")))
    return np.asarray(
        [bool(COMPETING_TACHYARRHYTHMIA_CODES & entry) for entry in codes], dtype=bool
    )


def label_column_order(manifest: dict) -> list[str]:
    """Target column order, which is the manifest label order the stores were built in."""

    return [entry["label_key"] for entry in manifest["labels"]]


def headline_positions(manifest: dict) -> np.ndarray:
    """Column positions of the headline labels inside the target matrix."""

    order = label_column_order(manifest)
    return np.array(
        [order.index(key) for key in manifest["headline_label_keys"]], dtype=np.int64
    )


def cluster_inverse(cohort: str, metadata: pd.DataFrame) -> np.ndarray:
    """The registered resampling unit: patient inside PTB-XL, record externally."""

    if cohort == "ptb_test":
        _, group_inverse = np.unique(metadata.patient_id.astype(str), return_inverse=True)
        return group_inverse
    return np.arange(len(metadata), dtype=np.int64)


def select_regularization(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    select_features: np.ndarray,
    select_targets: np.ndarray,
    grid: tuple[float, ...] = REGULARIZATION_GRID,
) -> tuple[float, LogisticRegression, float]:
    """Fit one label at every grid value and keep the best fold-9 AUROC.

    Selection is per label, so the macro-AUROC of a one-label set reduces to that label's
    own validation AUROC. Ties keep the smaller inverse strength, which is the more
    regularized fit.
    """

    best: tuple[float, LogisticRegression, float] | None = None
    for strength in grid:
        # l1_ratio=0 is the current spelling of a pure L2 penalty in scikit-learn 1.8.
        model = LogisticRegression(l1_ratio=0.0, C=strength, solver="lbfgs", max_iter=5000)
        model.fit(fit_features, fit_targets)
        scores = model.predict_proba(select_features)[:, 1]
        auroc = univariate_auroc(scores, select_targets, 1.0)
        if not np.isfinite(auroc):
            auroc = 0.0
        if best is None or auroc > best[2]:
            best = (strength, model, auroc)
    assert best is not None
    return best


def row(**values: object) -> dict[str, object]:
    """One tidy row, with every unused column left empty."""

    record: dict[str, object] = dict.fromkeys(CSV_COLUMNS, "")
    record.update(values)
    return record


def build_features(args: argparse.Namespace, root: Path) -> dict[str, np.ndarray]:
    """Extract or reload the per-cohort feature blocks, keyed by cohort name."""

    stores = {
        cohort: WaveformStore(root / args.store_root / COHORT_STORE_NAMES[cohort])
        for cohort in COHORTS
    }
    digests = {
        cohort: store_digest(root / args.store_root / COHORT_STORE_NAMES[cohort])
        for cohort in COHORTS
    }
    cache_path = root / args.feature_cache
    if cache_path.is_file() and not args.refresh_features:
        cached = np.load(cache_path, allow_pickle=False)
        names = [str(name) for name in cached["feature_names"]]
        matches = names == list(FEATURE_NAMES) and all(
            str(cached[f"{cohort}__digest"]) == digests[cohort] for cohort in COHORTS
        )
        if matches:
            print(f"reusing cached features from {cache_path}")
            return {cohort: cached[cohort] for cohort in COHORTS}
        print(f"cached features at {cache_path} are stale; re-extracting")

    features: dict[str, np.ndarray] = {}
    for cohort in COHORTS:
        print(f"extracting features for {cohort}", flush=True)
        features[cohort] = extract_cohort_features(
            stores[cohort], batch=args.extraction_batch, label=cohort
        )
    payload: dict[str, np.ndarray] = dict(features)
    payload["feature_names"] = np.asarray(FEATURE_NAMES)
    for cohort in COHORTS:
        payload[f"{cohort}__digest"] = np.asarray(digests[cohort])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(f"cached features to {cache_path}")
    return features


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_preregistration_seal(root)
    analysis = load_analysis_module()

    manifest = json.loads((root / args.labels).read_text())
    label_order = label_column_order(manifest)
    diagnosis = {entry["label_key"]: entry["diagnosis"] for entry in manifest["labels"]}
    headline_keys = list(manifest["headline_label_keys"])
    headline_indices = headline_positions(manifest)

    features = build_features(args, root)
    stores = {
        cohort: WaveformStore(root / args.store_root / COHORT_STORE_NAMES[cohort])
        for cohort in COHORTS
    }
    ptb = stores["ptb_test"]
    fit_rows = ptb.indices_for_folds(FIT_FOLDS)
    select_rows = ptb.indices_for_folds(SELECT_FOLD)
    test_rows = ptb.indices_for_folds(TEST_FOLD)
    ptb_features = features["ptb_test"]
    ptb_targets = np.asarray(ptb.targets)

    centre, scale = fit_standardizer(ptb_features[fit_rows])
    fit_design = apply_standardizer(ptb_features[fit_rows], centre, scale)
    select_design = apply_standardizer(ptb_features[select_rows], centre, scale)

    evaluation: dict[str, dict[str, np.ndarray]] = {
        "ptb_test": {
            "design": apply_standardizer(ptb_features[test_rows], centre, scale),
            "targets": ptb_targets[test_rows],
            "metadata": ptb.metadata.iloc[test_rows].reset_index(drop=True),
            "raw": ptb_features[test_rows],
        }
    }
    for cohort in EXTERNAL_COHORTS:
        evaluation[cohort] = {
            "design": apply_standardizer(features[cohort], centre, scale),
            "targets": np.asarray(stores[cohort].targets),
            "metadata": stores[cohort].metadata.reset_index(drop=True),
            "raw": features[cohort],
        }

    rows: list[dict[str, object]] = []
    failure_rates: dict[str, dict[str, float]] = {}
    validation: dict[str, dict[str, float]] = {}
    rate_check: dict[str, dict[str, dict[str, float]]] = {}
    for cohort in COHORTS:
        raw = evaluation[cohort]["raw"]
        targets = evaluation[cohort]["targets"]
        failure_rates[cohort] = {
            "r_peak": float(np.mean(raw[:, FEATURE_INDEX["r_peak_detection_failed"]] > 0.5)),
            "p_wave": float(np.mean(raw[:, FEATURE_INDEX["pr_interval_missing"]] > 0.5)),
            "t_wave_end": float(np.mean(raw[:, FEATURE_INDEX["qt_interval_missing"]] > 0.5)),
        }
        for name, rate in failure_rates[cohort].items():
            rows.append(
                row(
                    section="detection_failure",
                    cohort=cohort,
                    feature=name,
                    metric="failure_rate",
                    value=rate,
                )
            )
        validation[cohort] = {}
        for feature_name, label_key, orientation in VALIDATION_CHECKS:
            index = label_order.index(label_key)
            auroc = univariate_auroc(
                raw[:, FEATURE_INDEX[feature_name]], targets[:, index], orientation
            )
            validation[cohort][f"{feature_name}__{label_key}"] = auroc
            rows.append(
                row(
                    section="feature_validation",
                    cohort=cohort,
                    label_key=label_key,
                    diagnosis=diagnosis[label_key],
                    headline=label_key in headline_keys,
                    finding_class=FINDING_CLASSES.get(label_key, ""),
                    feature=feature_name,
                    orientation=orientation,
                    metric="univariate_auroc",
                    value=auroc,
                )
            )
        rate = raw[:, FEATURE_INDEX["heart_rate_bpm"]]
        competing = competing_rhythm_mask(evaluation[cohort]["metadata"])
        rate_check[cohort] = {}
        for label_key, orientation, threshold, side in RATE_CHECKS:
            positive = targets[:, label_order.index(label_key)].astype(bool)
            measured = rate[positive]
            defining = measured < threshold if side == "below" else measured > threshold
            keep = positive | ~competing
            restricted = univariate_auroc(
                rate[keep], targets[keep, label_order.index(label_key)], orientation
            )
            entry = {
                "positives": int(positive.sum()),
                "share_of_positives_on_the_defining_side": float(np.nanmean(defining)),
                "median_measured_rate_bpm": float(np.nanmedian(measured)),
                "auroc_excluding_competing_rhythm_negatives": restricted,
                "records_after_exclusion": int(keep.sum()),
            }
            rate_check[cohort][label_key] = entry
            for metric, value in entry.items():
                rows.append(
                    row(
                        section="rate_detector_check",
                        cohort=cohort,
                        label_key=label_key,
                        diagnosis=diagnosis[label_key],
                        headline=label_key in headline_keys,
                        finding_class=FINDING_CLASSES.get(label_key, ""),
                        feature="heart_rate_bpm",
                        orientation=orientation,
                        metric=metric,
                        value=value,
                    )
                )
    print(json.dumps({"detection_failure_rates": failure_rates}, indent=2, sort_keys=True))
    print(json.dumps({"feature_validation": validation}, indent=2, sort_keys=True))
    print(json.dumps({"rate_detector_check": rate_check}, indent=2, sort_keys=True))

    chosen: dict[str, float] = {}
    validation_auroc: dict[str, float] = {}
    scores = {
        cohort: np.zeros_like(evaluation[cohort]["targets"], dtype=np.float64)
        for cohort in COHORTS
    }
    for index, label_key in enumerate(label_order):
        strength, model, auroc = select_regularization(
            fit_design,
            ptb_targets[fit_rows][:, index],
            select_design,
            ptb_targets[select_rows][:, index],
        )
        chosen[label_key] = float(strength)
        validation_auroc[label_key] = float(auroc)
        for cohort in COHORTS:
            scores[cohort][:, index] = model.predict_proba(evaluation[cohort]["design"])[:, 1]
    print(json.dumps({"chosen_inverse_regularization": chosen}, indent=2, sort_keys=True))

    points: dict[str, np.ndarray] = {}
    replicates: dict[str, np.ndarray] = {}
    macro_points: dict[str, float] = {}
    macro_replicates: dict[str, np.ndarray] = {}
    for cohort in COHORTS:
        targets = evaluation[cohort]["targets"]
        points[cohort] = point_auroc(targets, scores[cohort])
        group_inverse = cluster_inverse(cohort, evaluation[cohort]["metadata"])
        print(f"bootstrapping {cohort}", flush=True)
        replicates[cohort] = per_label_bootstrap(
            targets,
            scores[cohort],
            group_inverse,
            replicates=args.bootstrap_replicates,
            seed=analysis.COHORT_SEEDS[cohort],
            batch_size=args.bootstrap_batch_size,
        )
        macro_points[cohort] = float(np.nanmean(points[cohort][headline_indices]))
        macro_replicates[cohort] = macro_over_labels(replicates[cohort], headline_indices)

    macro_summary: dict[str, dict[str, float]] = {}
    for cohort in COHORTS:
        lower, upper = percentile_interval(macro_replicates[cohort])
        macro_summary[cohort] = {
            "point": macro_points[cohort],
            "lower_95": lower,
            "upper_95": upper,
        }
        rows.append(
            row(
                section="macro_performance",
                cohort=cohort,
                metric="macro_auroc",
                value=macro_points[cohort],
                lower_95=lower,
                upper_95=upper,
            )
        )
        for index, label_key in enumerate(label_order):
            lower, upper = percentile_interval(replicates[cohort][:, index])
            rows.append(
                row(
                    section="label_performance",
                    cohort=cohort,
                    label_key=label_key,
                    diagnosis=diagnosis[label_key],
                    headline=label_key in headline_keys,
                    finding_class=FINDING_CLASSES.get(label_key, ""),
                    metric="auroc",
                    value=float(points[cohort][index]),
                    lower_95=lower,
                    upper_95=upper,
                )
            )

    macro_shift: dict[str, dict[str, float]] = {}
    feature_shift: dict[str, dict[str, float]] = {}
    for external in EXTERNAL_COHORTS:
        distribution = macro_replicates[external] - macro_replicates["ptb_test"]
        lower, upper = percentile_interval(distribution)
        point = macro_points[external] - macro_points["ptb_test"]
        macro_shift[external] = {"point": point, "lower_95": lower, "upper_95": upper}
        rows.append(
            row(
                section="macro_shift_delta",
                external_cohort=external,
                metric="macro_shift_delta",
                value=point,
                lower_95=lower,
                upper_95=upper,
            )
        )
        feature_shift[external] = {}
        for index, label_key in enumerate(label_order):
            delta = replicates[external][:, index] - replicates["ptb_test"][:, index]
            lower, upper = percentile_interval(delta)
            point = float(points[external][index] - points["ptb_test"][index])
            feature_shift[external][label_key] = point
            rows.append(
                row(
                    section="shift_delta",
                    external_cohort=external,
                    label_key=label_key,
                    diagnosis=diagnosis[label_key],
                    headline=label_key in headline_keys,
                    finding_class=FINDING_CLASSES.get(label_key, ""),
                    metric="shift_delta",
                    value=point,
                    lower_95=lower,
                    upper_95=upper,
                )
            )

    committed = pd.read_csv(root / args.output_root / "exploratory_per_label_shift_deltas.csv")
    deep = committed[committed.architecture == COMPARISON_ARCHITECTURE]
    correlations: dict[str, dict[str, float]] = {}
    class_means: dict[str, dict[str, dict[str, float]]] = {}
    for external in EXTERNAL_COHORTS:
        table = deep[deep.external_cohort == external].set_index("label_key")
        feature_values = []
        deep_values = []
        for label_key in headline_keys:
            feature_value = feature_shift[external][label_key]
            deep_value = float(table.loc[label_key, "shift_delta"])
            feature_values.append(feature_value)
            deep_values.append(deep_value)
            common = {
                "external_cohort": external,
                "label_key": label_key,
                "diagnosis": diagnosis[label_key],
                "headline": True,
                "finding_class": FINDING_CLASSES.get(label_key, ""),
            }
            rows.append(
                row(
                    section="model_comparison",
                    metric="feature_model_shift_delta",
                    value=feature_value,
                    **common,
                )
            )
            rows.append(
                row(
                    section="model_comparison",
                    metric=f"{COMPARISON_ARCHITECTURE}_shift_delta",
                    value=deep_value,
                    lower_95=float(table.loc[label_key, "shift_delta_lower_95"]),
                    upper_95=float(table.loc[label_key, "shift_delta_upper_95"]),
                    **common,
                )
            )
        result = spearmanr(feature_values, deep_values)
        correlations[external] = {
            "spearman_rho": float(result.statistic),
            "p_value": float(result.pvalue),
            "labels": len(headline_keys),
        }
        rows.append(
            row(
                section="rank_correlation",
                external_cohort=external,
                metric="spearman_rho",
                value=correlations[external]["spearman_rho"],
            )
        )
        rows.append(
            row(
                section="rank_correlation",
                external_cohort=external,
                metric="p_value",
                value=correlations[external]["p_value"],
            )
        )
        class_means[external] = {}
        for class_name in ("geometric", "interpretive", "unclassified"):
            keys = [key for key in headline_keys if FINDING_CLASSES.get(key) == class_name]
            feature_mean = float(np.mean([feature_shift[external][key] for key in keys]))
            deep_mean = float(np.mean([float(table.loc[key, "shift_delta"]) for key in keys]))
            class_means[external][class_name] = {
                "labels": len(keys),
                "feature_model_mean_shift_delta": feature_mean,
                f"{COMPARISON_ARCHITECTURE}_mean_shift_delta": deep_mean,
            }
            rows.append(
                row(
                    section="finding_class_summary",
                    external_cohort=external,
                    finding_class=class_name,
                    metric="feature_model_mean_shift_delta",
                    value=feature_mean,
                )
            )
            rows.append(
                row(
                    section="finding_class_summary",
                    external_cohort=external,
                    finding_class=class_name,
                    metric=f"{COMPARISON_ARCHITECTURE}_mean_shift_delta",
                    value=deep_mean,
                )
            )

    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "exploratory_feature_baseline.csv"
    pd.DataFrame(rows, columns=list(CSV_COLUMNS)).to_csv(csv_path, index=False)

    summary = {
        "stage": "post-seal exploratory addition",
        "stage_date": EXPLORATORY_STAGE_DATE,
        "changes_registered_quantities": False,
        "model": "one L2-penalized logistic regression per target over standardized features",
        "features": list(FEATURE_NAMES),
        "estimators": ESTIMATOR_DESCRIPTIONS,
        "sampling_frequency_hz": SAMPLING_HZ,
        "fitting_split": "ptb_xl strat_fold 1 to 8",
        "selection_split": "ptb_xl strat_fold 9",
        "evaluation_splits": {
            "ptb_test": "ptb_xl strat_fold 10",
            "chapman_shaoxing": "all records",
            "ningbo": "all records",
        },
        "standardization": "centre and scale from the fitting split only",
        "regularization_grid": list(REGULARIZATION_GRID),
        "chosen_inverse_regularization": chosen,
        "selection_auroc_fold_9": validation_auroc,
        "trained_targets": len(label_order),
        "headline_labels": headline_keys,
        "detection_failure_rates": failure_rates,
        "feature_validation_auroc": validation,
        "feature_validation_orientation": {
            f"{feature}__{label}": orientation
            for feature, label, orientation in VALIDATION_CHECKS
        },
        "rate_detector_check": rate_check,
        "rate_detector_check_note": (
            "the rate AUROC for sinus tachycardia is bounded by what the label names, not "
            "by the detector: a record in atrial fibrillation, flutter or another "
            "supraventricular tachycardia is fast and is a negative for sinus tachycardia. "
            "auroc_excluding_competing_rhythm_negatives drops those records from the "
            "negatives and is the number that reflects rate measurement alone"
        ),
        "macro_auroc": macro_summary,
        "macro_shift_deltas": macro_shift,
        "rank_correlation_with_xresnet1d101_shift_deltas": correlations,
        "finding_class_mean_shift_deltas": class_means,
        "finding_classes": FINDING_CLASSES,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "batch_size": args.bootstrap_batch_size,
            "seeds": dict(analysis.COHORT_SEEDS),
            "ptb_cluster_unit": "patient_id",
            "external_cluster_unit": "record",
            "interval_type": "percentile",
            "design": (
                "the registered cluster bootstrap design: same seeds, cluster units, "
                "replicate count and batch size as the registered analysis, so the "
                "intervals are commensurable with the deep models'"
            ),
            "registered_cross_check": (
                "not applicable; a different model's scores are bootstrapped on the same "
                "records, so the replicate values cannot equal the registered ones"
            ),
        },
        "comparison_source": "results/exploratory_per_label_shift_deltas.csv, read only",
        "outputs": [csv_path.name, "exploratory_feature_baseline.json"],
    }
    json_path = output_root / "exploratory_feature_baseline.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(json.dumps({"macro_auroc": macro_summary}, indent=2, sort_keys=True))
    print(json.dumps({"macro_shift_deltas": macro_shift}, indent=2, sort_keys=True))
    print(json.dumps({"rank_correlations": correlations}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
