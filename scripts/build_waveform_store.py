#!/usr/bin/env python3
"""Stream checksummed source waveforms into compact, validated 100 Hz stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from numpy.lib.format import open_memmap

from ecg_clinical.integrity import verify_preregistration_seal
from ecg_clinical.waveforms import (
    decode_external_mat,
    decode_ptb_dat,
    parse_signal_specification,
)

PTB_BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
EXTERNAL_BASE_URL = "https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/1.0.0/"
THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class DownloadJob:
    row_index: int
    record_id: str
    header_path: Path | None
    header_url: str | None
    header_sha256: str | None
    signal_url: str
    signal_sha256: str
    decoder: str


def session() -> requests.Session:
    current = getattr(THREAD_LOCAL, "session", None)
    if current is None:
        current = requests.Session()
        current.headers["User-Agent"] = "ecg-clinical-waveform-builder/0.1"
        THREAD_LOCAL.session = current
    return current


def download(url: str, expected_sha256: str, attempts: int = 8) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session().get(url, timeout=(10, 60))
            response.raise_for_status()
            payload = response.content
            observed = hashlib.sha256(payload).hexdigest()
            if observed != expected_sha256:
                raise ValueError(f"checksum mismatch for {url}: {observed} != {expected_sha256}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(0.25 * 2**attempt, 8.0))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from error


def load_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative_path = line.split(maxsplit=1)
        checksums[relative_path.strip()] = digest
    return checksums


def label_matrix(diagnoses: pd.Series, label_manifest: Path) -> np.ndarray:
    labels = json.loads(label_manifest.read_text())["labels"]
    output = np.zeros((len(diagnoses), len(labels)), dtype=np.uint8)
    diagnosis_sets = diagnoses.fillna("").map(lambda value: set(str(value).split("|")))
    for column, label in enumerate(labels):
        codes = set(label["snomed_codes"])
        output[:, column] = diagnosis_sets.map(
            lambda observed, label_codes=codes: bool(label_codes & observed)
        )
    return output


def build_ptb_jobs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[DownloadJob]]:
    database = pd.read_csv(args.ptb_database)
    headers = pd.read_csv(args.headers, dtype={"record_id": str, "diagnoses": str})
    ptb_labels = headers[headers.cohort == "ptb-xl"][["record_id", "diagnoses"]].copy()
    ptb_labels["ecg_id"] = ptb_labels.record_id.str.removeprefix("HR").astype(int)
    metadata = database.merge(
        ptb_labels[["ecg_id", "diagnoses"]], on="ecg_id", validate="one_to_one"
    )
    if len(metadata) != 21_799:
        raise ValueError("PTB-XL metadata join did not produce 21,799 current records")
    metadata = metadata.sort_values("ecg_id").reset_index(drop=True)
    metadata["record_id"] = metadata.ecg_id.map(lambda value: f"HR{value:05d}")
    metadata["cohort"] = "ptb_xl"
    metadata["patient_id"] = metadata.patient_id.astype(str)
    metadata = metadata[
        [
            "record_id",
            "cohort",
            "patient_id",
            "strat_fold",
            "age",
            "sex",
            "diagnoses",
            "filename_lr",
        ]
    ]

    checksums = load_checksums(args.ptb_checksums)
    jobs: list[DownloadJob] = []
    for index, row in metadata.iterrows():
        base_path = str(row.filename_lr)
        header_relative = f"{base_path}.hea"
        signal_relative = f"{base_path}.dat"
        jobs.append(
            DownloadJob(
                row_index=index,
                record_id=str(row.record_id),
                header_path=None,
                header_url=f"{PTB_BASE_URL}{header_relative}",
                header_sha256=checksums[header_relative],
                signal_url=f"{PTB_BASE_URL}{signal_relative}",
                signal_sha256=checksums[signal_relative],
                decoder="ptb",
            )
        )
    return metadata, jobs


def external_mat_index(checksums: dict[str, str]) -> dict[str, tuple[str, str]]:
    output: dict[str, tuple[str, str]] = {}
    for relative_path, digest in checksums.items():
        if not relative_path.endswith(".mat"):
            continue
        record_id = Path(relative_path).stem
        if record_id in output:
            raise ValueError(f"duplicate external matrix for {record_id}")
        output[record_id] = (relative_path, digest)
    return output


def build_external_jobs(
    args: argparse.Namespace, cohort: str
) -> tuple[pd.DataFrame, list[DownloadJob]]:
    headers = pd.read_csv(args.headers, dtype={"record_id": str, "diagnoses": str})
    metadata = headers[headers.cohort == cohort].copy()
    metadata["patient_id"] = metadata.record_id
    metadata["strat_fold"] = 0
    metadata = metadata.sort_values("record_id").reset_index(drop=True)
    metadata = metadata[
        [
            "record_id",
            "cohort",
            "patient_id",
            "strat_fold",
            "age",
            "sex",
            "diagnoses",
            "group",
            "signal_file",
            "header_sha256",
        ]
    ]
    checksums = load_checksums(args.external_checksums)
    matrices = external_mat_index(checksums)
    jobs: list[DownloadJob] = []
    for index, row in metadata.iterrows():
        # The Challenge release renames one Ningbo record header to S23074 while
        # retaining the source matrix name JS23074.mat. The signal filename in
        # the official header is the authoritative join key for every record.
        matrix_id = Path(str(row.signal_file)).stem
        relative_path, digest = matrices[matrix_id]
        header_path = args.header_root / cohort / str(row.group) / f"{row.record_id}.hea"
        if not header_path.is_file():
            raise FileNotFoundError(header_path)
        jobs.append(
            DownloadJob(
                row_index=index,
                record_id=str(row.record_id),
                header_path=header_path,
                header_url=None,
                header_sha256=str(row.header_sha256),
                signal_url=f"{EXTERNAL_BASE_URL}{relative_path}",
                signal_sha256=digest,
                decoder="external",
            )
        )
    return metadata, jobs


def process_job(job: DownloadJob) -> tuple[int, np.ndarray, dict[str, object]]:
    if job.header_path is not None:
        header_payload = job.header_path.read_bytes()
        observed = hashlib.sha256(header_payload).hexdigest()
        if observed != job.header_sha256:
            raise ValueError(f"{job.record_id}: cached header checksum mismatch")
    else:
        if job.header_url is None or job.header_sha256 is None:
            raise ValueError(f"{job.record_id}: missing header source")
        header_payload = download(job.header_url, job.header_sha256)
    specification = parse_signal_specification(header_payload.decode("utf-8"))
    payload = download(job.signal_url, job.signal_sha256)
    if job.decoder == "ptb":
        signal = decode_ptb_dat(payload, specification)
    else:
        signal = decode_external_mat(payload, specification)
    summary = {
        "record_id": job.record_id,
        "minimum_mv": float(signal.min()),
        "maximum_mv": float(signal.max()),
        "absolute_maximum_mv": float(np.abs(signal).max()),
    }
    return job.row_index, signal, summary


def open_or_create(path: Path, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.dtype != np.dtype(dtype) or array.shape != shape:
            raise ValueError(f"incompatible resumable array {path}: {array.dtype} {array.shape}")
        return array
    return open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["ptb_xl", "chapman_shaoxing", "ningbo"], required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/cache/waveforms"))
    parser.add_argument(
        "--headers", type=Path, default=Path("data/cache/challenge2021_headers.csv")
    )
    parser.add_argument(
        "--header-root", type=Path, default=Path("data/cache/challenge2021_headers")
    )
    parser.add_argument(
        "--ptb-database",
        type=Path,
        default=Path("data/raw-metadata/ptb-xl/1.0.3/ptbxl_database.csv"),
    )
    parser.add_argument(
        "--ptb-checksums",
        type=Path,
        default=Path("data/raw-metadata/ptb-xl/1.0.3/SHA256SUMS.txt"),
    )
    parser.add_argument(
        "--external-checksums",
        type=Path,
        default=Path("data/raw-metadata/ecg-arrhythmia/1.0.0/SHA256SUMS.txt"),
    )
    parser.add_argument(
        "--label-manifest",
        type=Path,
        default=Path("data/derived/preregistration/harmonized_labels.json"),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, help="development-only prefix limit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > 64:
        raise ValueError("--workers must be between 1 and 64")
    root = Path(__file__).resolve().parents[1]
    verify_preregistration_seal(root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cohort == "ptb_xl":
        metadata, jobs = build_ptb_jobs(args)
    else:
        metadata, jobs = build_external_jobs(args, args.cohort)
    if args.limit is not None:
        metadata = metadata.iloc[: args.limit].copy()
        jobs = jobs[: args.limit]

    cohort_root = args.output_root / args.cohort
    cohort_root.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(cohort_root / "metadata.parquet", index=False)
    targets = label_matrix(metadata.diagnoses, args.label_manifest)
    np.save(cohort_root / "targets.npy", targets)
    signals = open_or_create(cohort_root / "signals.npy", "float16", (len(metadata), 12, 1000))
    status = open_or_create(cohort_root / "status.npy", "uint8", (len(metadata),))
    pending = [job for job in jobs if status[job.row_index] == 0]
    logging.info("%s: %d total, %d pending", args.cohort, len(jobs), len(pending))

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_job, job): job for job in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                index, signal, summary = future.result()
                signals[index] = signal.astype(np.float16)
                status[index] = 1
                summaries.append(summary)
            except Exception as exc:
                failures.append(
                    {
                        "record_id": job.record_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                logging.error("%s failed: %s", job.record_id, exc)
            if completed % 1_000 == 0 or completed == len(pending):
                signals.flush()
                status.flush()
                logging.info("processed %d/%d pending records", completed, len(pending))

    failure_path = cohort_root / "build_failures.json"
    failure_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")
    if failures:
        logging.error("%d records failed; rerun will retry them", len(failures))
        return 1
    if not np.all(status == 1):
        raise RuntimeError("store contains incomplete records")

    absolute_maximum = max((float(row["absolute_maximum_mv"]) for row in summaries), default=None)
    manifest = {
        "schema_version": 1,
        "cohort": args.cohort,
        "records": len(metadata),
        "shape": [len(metadata), 12, 1000],
        "dtype": "float16",
        "units": "mV",
        "sampling_frequency_hz": 100,
        "source_checksums_verified": True,
        "valid_records": int((status == 1).sum()),
        "invalid_records": 0,
        "maximum_observed_absolute_mv_in_last_build_batch": absolute_maximum,
        "signals_file": "signals.npy",
        "targets_file": "targets.npy",
        "metadata_file": "metadata.parquet",
    }
    temporary = cohort_root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, cohort_root / "manifest.json")
    logging.info("completed %s waveform store", args.cohort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
