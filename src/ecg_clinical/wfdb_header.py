"""Small, dependency-light parser for Challenge 2021 WFDB headers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class HeaderMetadata:
    """Fields needed for cohort auditing and later waveform loading."""

    record_id: str
    num_leads: int
    sampling_frequency_hz: float
    num_samples: int
    signal_file: str
    signal_format: str
    lead_names: tuple[str, ...]
    age: str
    sex: str
    diagnoses: tuple[str, ...]
    header_sha256: str

    def as_row(self) -> dict[str, str | int | float]:
        return {
            "record_id": self.record_id,
            "num_leads": self.num_leads,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "num_samples": self.num_samples,
            "signal_file": self.signal_file,
            "signal_format": self.signal_format,
            "lead_names": "|".join(self.lead_names),
            "age": self.age,
            "sex": self.sex,
            "diagnoses": "|".join(self.diagnoses),
            "header_sha256": self.header_sha256,
        }


def parse_header(text: str) -> HeaderMetadata:
    """Parse the deterministic subset of a WFDB header used by the challenge release."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty WFDB header")

    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"invalid first WFDB line: {lines[0]!r}")
    record_id = first[0]
    num_leads = int(first[1])
    sampling_frequency_hz = float(first[2].split("/")[0])
    num_samples = int(first[3])

    signal_lines = lines[1 : 1 + num_leads]
    if len(signal_lines) != num_leads:
        raise ValueError(f"{record_id}: expected {num_leads} signal lines")
    signal_parts = [line.split() for line in signal_lines]
    if any(len(parts) < 2 for parts in signal_parts):
        raise ValueError(f"{record_id}: malformed signal line")
    signal_files = {parts[0] for parts in signal_parts}
    if len(signal_files) != 1:
        raise ValueError(f"{record_id}: multiple signal files: {sorted(signal_files)}")

    comments: dict[str, str] = {}
    for line in lines[1 + num_leads :]:
        if not line.startswith("#") or ":" not in line:
            continue
        key, value = line[1:].split(":", maxsplit=1)
        comments[key.strip().lower()] = value.strip()

    diagnoses = tuple(
        sorted(
            {code.strip() for code in comments.get("dx", "").split(",") if code.strip()},
            key=lambda code: (not code.isdigit(), code),
        )
    )
    if not diagnoses:
        raise ValueError(f"{record_id}: missing #Dx labels")

    return HeaderMetadata(
        record_id=record_id,
        num_leads=num_leads,
        sampling_frequency_hz=sampling_frequency_hz,
        num_samples=num_samples,
        signal_file=next(iter(signal_files)),
        signal_format=signal_parts[0][1],
        lead_names=tuple(parts[-1] for parts in signal_parts),
        age=comments.get("age", "Unknown"),
        sex=comments.get("sex", "Unknown"),
        diagnoses=diagnoses,
        header_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
