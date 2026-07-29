"""Signal decoding and validation for the frozen ECG cohorts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

EXPECTED_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
GAIN_PATTERN = re.compile(
    r"^(?P<gain>[0-9]+(?:\.[0-9]+)?)(?:\((?P<baseline>-?[0-9]+)\))?/(?P<unit>\S+)$"
)


@dataclass(frozen=True)
class SignalSpecification:
    record_id: str
    num_leads: int
    sampling_frequency_hz: int
    num_samples: int
    gains: np.ndarray
    baselines: np.ndarray
    units: tuple[str, ...]
    leads: tuple[str, ...]


def parse_signal_specification(header: str) -> SignalSpecification:
    """Parse gain, baseline, unit, and lead order from a WFDB header."""

    lines = [line.strip() for line in header.splitlines() if line.strip()]
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError("malformed WFDB record line")
    record_id = first[0]
    num_leads = int(first[1])
    sampling_frequency_hz = int(float(first[2].split("/")[0]))
    num_samples = int(first[3])
    signal_lines = [line.split() for line in lines[1 : num_leads + 1]]
    if len(signal_lines) != num_leads or any(len(parts) < 9 for parts in signal_lines):
        raise ValueError(f"{record_id}: malformed signal specification")

    gains: list[float] = []
    baselines: list[float] = []
    units: list[str] = []
    leads: list[str] = []
    for parts in signal_lines:
        match = GAIN_PATTERN.match(parts[2])
        if match is None:
            raise ValueError(f"{record_id}: unsupported gain specification {parts[2]!r}")
        gains.append(float(match.group("gain")))
        baselines.append(float(match.group("baseline") or 0))
        units.append(match.group("unit"))
        leads.append(parts[-1].upper())
    specification = SignalSpecification(
        record_id=record_id,
        num_leads=num_leads,
        sampling_frequency_hz=sampling_frequency_hz,
        num_samples=num_samples,
        gains=np.asarray(gains, dtype=np.float32),
        baselines=np.asarray(baselines, dtype=np.float32),
        units=tuple(units),
        leads=tuple(leads),
    )
    validate_signal_specification(specification)
    return specification


def validate_signal_specification(specification: SignalSpecification) -> None:
    if specification.num_leads != 12:
        raise ValueError(f"{specification.record_id}: expected 12 leads")
    if specification.leads != EXPECTED_LEADS:
        raise ValueError(f"{specification.record_id}: unexpected leads {specification.leads}")
    if specification.num_samples != specification.sampling_frequency_hz * 10:
        raise ValueError(f"{specification.record_id}: expected 10 second duration")
    if any(unit.lower() != "mv" for unit in specification.units):
        raise ValueError(f"{specification.record_id}: expected mV units")
    if not np.isfinite(specification.gains).all() or (specification.gains <= 0).any():
        raise ValueError(f"{specification.record_id}: invalid gains")


def physical_signal(digital: np.ndarray, specification: SignalSpecification) -> np.ndarray:
    """Convert a lead-by-time digital signal to physical millivolts."""

    expected = (specification.num_leads, specification.num_samples)
    if digital.shape != expected:
        raise ValueError(
            f"{specification.record_id}: signal shape {digital.shape}, expected {expected}"
        )
    signal = (digital.astype(np.float32) - specification.baselines[:, None]) / specification.gains[
        :, None
    ]
    if not np.isfinite(signal).all():
        raise ValueError(f"{specification.record_id}: non-finite physical samples")
    return signal


def decode_ptb_dat(payload: bytes, specification: SignalSpecification) -> np.ndarray:
    if specification.sampling_frequency_hz != 100:
        raise ValueError(f"{specification.record_id}: PTB-XL store requires 100 Hz")
    digital = np.frombuffer(payload, dtype="<i2")
    expected_size = specification.num_leads * specification.num_samples
    if digital.size != expected_size:
        raise ValueError(
            f"{specification.record_id}: {digital.size} digital samples, expected {expected_size}"
        )
    return physical_signal(
        digital.reshape(specification.num_samples, specification.num_leads).T,
        specification,
    )


def decode_external_mat(payload: bytes, specification: SignalSpecification) -> np.ndarray:
    if specification.sampling_frequency_hz != 500:
        raise ValueError(f"{specification.record_id}: external source must be 500 Hz")
    loaded = loadmat(BytesIO(payload), variable_names=["val"])
    if "val" not in loaded:
        raise ValueError(f"{specification.record_id}: MATLAB file lacks 'val'")
    physical = physical_signal(np.asarray(loaded["val"]), specification)
    resampled = resample_poly(physical, up=1, down=5, axis=1).astype(np.float32)
    if resampled.shape != (12, 1000) or not np.isfinite(resampled).all():
        raise ValueError(f"{specification.record_id}: invalid resampled signal")
    return resampled
