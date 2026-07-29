#!/usr/bin/env python3
"""Fetch and aggregate waveform-free PhysioNet Challenge 2021 headers.

This command intentionally fetches only ``.hea`` files. It is safe to run before the
preregistration seal because it never requests signal-bearing ``.mat`` files.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urljoin

import requests

from ecg_clinical.wfdb_header import parse_header

BASE_URL = "https://physionet.org/files/challenge-2021/1.0.3/training/"
COHORTS = ("ptb-xl", "chapman_shaoxing", "ningbo")
EXPECTED_COUNTS = {"ptb-xl": 21_837, "chapman_shaoxing": 10_247, "ningbo": 34_905}
GROUP_PATTERN = re.compile(r'href="(g\d+/)"')
HEADER_PATTERN = re.compile(r'href="([^"/]+\.hea)"')
THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    current = getattr(THREAD_LOCAL, "session", None)
    if current is None:
        current = requests.Session()
        current.headers["User-Agent"] = "ecg-clinical-metadata-audit/0.1"
        THREAD_LOCAL.session = current
    return current


def get_text(url: str, *, attempts: int = 6) -> str:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session().get(url, timeout=(10, 45))
            response.raise_for_status()
            return response.text
        except (requests.RequestException, UnicodeDecodeError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(0.5 * 2**attempt, 8.0))
    raise RuntimeError(f"failed after {attempts} attempts: {url}") from error


def discover_headers(cohort: str) -> list[tuple[str, str, str]]:
    cohort_url = urljoin(BASE_URL, f"{cohort}/")
    groups = sorted(set(GROUP_PATTERN.findall(get_text(cohort_url))))
    if not groups:
        raise RuntimeError(f"no group directories found for {cohort_url}")

    discovered: list[tuple[str, str, str]] = []
    for group in groups:
        group_url = urljoin(cohort_url, group)
        names = sorted(set(HEADER_PATTERN.findall(get_text(group_url))))
        if not names:
            raise RuntimeError(f"no headers found for {group_url}")
        discovered.extend((group.rstrip("/"), name, urljoin(group_url, name)) for name in names)
    expected = EXPECTED_COUNTS[cohort]
    if len(discovered) != expected:
        raise RuntimeError(f"{cohort}: discovered {len(discovered)} headers, expected {expected}")
    return discovered


def fetch_one(
    cohort: str,
    group: str,
    name: str,
    url: str,
    raw_root: Path | None,
) -> dict[str, str | int | float]:
    destination = raw_root / cohort / group / name if raw_root is not None else None
    if destination is not None and destination.is_file():
        text = destination.read_text(encoding="utf-8")
    else:
        text = get_text(url)
    parsed = parse_header(text)
    if name != f"{parsed.record_id}.hea":
        raise ValueError(f"URL/header ID mismatch: {name} vs {parsed.record_id}")
    if destination is not None and not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".hea.part")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    row = parsed.as_row()
    row.update(
        {
            "cohort": cohort,
            "group": group,
            "relative_record_path": f"training/{cohort}/{group}/{parsed.record_id}",
            "source_header_url": url,
        }
    )
    return row


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cohort",
        "group",
        "record_id",
        "relative_record_path",
        "num_leads",
        "sampling_frequency_hz",
        "num_samples",
        "signal_file",
        "signal_format",
        "lead_names",
        "age",
        "sex",
        "diagnoses",
        "header_sha256",
        "source_header_url",
    ]
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, action="append", dest="cohorts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cohorts = args.cohorts or list(COHORTS)
    if args.workers < 1 or args.workers > 64:
        raise ValueError("--workers must be between 1 and 64")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    jobs: list[tuple[str, str, str, str]] = []
    for cohort in cohorts:
        cohort_jobs = [(cohort, *item) for item in discover_headers(cohort)]
        logging.info("discovered %d %s headers", len(cohort_jobs), cohort)
        jobs.extend(cohort_jobs)

    rows: list[dict[str, object]] = []
    pending_jobs = jobs
    for round_number in range(1, 7):
        failed_jobs: list[tuple[str, str, str, str]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch_one, *job, args.raw_root): job for job in pending_jobs}
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    rows.append(future.result())
                except Exception as exc:
                    job = futures[future]
                    logging.warning("round %d failed %s: %s", round_number, job[3], exc)
                    failed_jobs.append(job)
                if index % 1_000 == 0 or index == len(pending_jobs):
                    logging.info(
                        "round %d processed %d/%d headers", round_number, index, len(pending_jobs)
                    )
        if not failed_jobs:
            break
        logging.warning("retrying %d failed headers after round %d", len(failed_jobs), round_number)
        pending_jobs = failed_jobs
        time.sleep(min(2**round_number, 30))
    else:
        logging.error("failed to fetch %d headers after all rounds", len(pending_jobs))
        return 1

    rows.sort(key=lambda row: (str(row["cohort"]), str(row["group"]), str(row["record_id"])))
    write_csv(rows, args.output)
    logging.info("wrote %d rows to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
