#!/usr/bin/env python3
"""Fetch the five source metadata files the preregistration and waveform stages read.

These files are not committed. They are the PTB-XL 1.0.3 record database, the PTB-XL
1.0.3 and ECG Arrhythmia 1.0.0 release checksum manifests, and the PhysioNet/CinC 2021
scored-code weights matrix and diagnosis mapping table pinned at the mapping commit
recorded in METHODOLOGY.md section 3.

The command fetches metadata only. It requests no signal-bearing file and is safe to run
before the preregistration seal. Downloads are verified against the SHA-256 digests
recorded in the sealed label manifest where those digests exist, and against the digests
observed at registration otherwise. The command is idempotent: a file already present
with the expected digest is left alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

PHYSIONET_BASE = "https://physionet.org/files/"
MAPPING_COMMIT = "e2a75fc01f729cb74cc4e853e054ce81e28381fc"
MAPPING_BASE = (
    f"https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/{MAPPING_COMMIT}/"
)


@dataclass(frozen=True)
class Source:
    relative_path: str
    url: str
    observed_sha256: str
    purpose: str


SOURCES = (
    Source(
        relative_path="data/raw-metadata/ptb-xl/1.0.3/ptbxl_database.csv",
        url=f"{PHYSIONET_BASE}ptb-xl/1.0.3/ptbxl_database.csv",
        observed_sha256="7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
        purpose="PTB-XL records, folds, demographics and SCP statements",
    ),
    Source(
        relative_path="data/raw-metadata/ptb-xl/1.0.3/SHA256SUMS.txt",
        url=f"{PHYSIONET_BASE}ptb-xl/1.0.3/SHA256SUMS.txt",
        observed_sha256="b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d",
        purpose="PTB-XL release checksums, verified per waveform file during store construction",
    ),
    Source(
        relative_path="data/raw-metadata/ecg-arrhythmia/1.0.0/SHA256SUMS.txt",
        url=f"{PHYSIONET_BASE}ecg-arrhythmia/1.0.0/SHA256SUMS.txt",
        observed_sha256="dfad0c26a276ce7e450e5002051f4c749304552d9add4d592928bbdcb6e4f31c",
        purpose="external release checksums, verified per waveform file during store construction",
    ),
    Source(
        relative_path="data/raw-metadata/cinc2021_weights.csv",
        url=f"{MAPPING_BASE}weights.csv",
        observed_sha256="668cb04555a97e8977488cb4bb6088d0d2bf9e59557945deedf29d5065c14d2b",
        purpose="the 26 scored diagnosis groups and their official equivalence sets",
    ),
    Source(
        relative_path="data/raw-metadata/cinc2021_dx_mapping_scored.csv",
        url=f"{MAPPING_BASE}dx_mapping_scored.csv",
        observed_sha256="fad13ad9f7ca230e7e6392ac8a264cb7cd157879525129f964c5f708eabb41d0",
        purpose="SNOMED CT code to diagnosis name mapping for the scored codes",
    ),
)

LABEL_MANIFEST = "data/derived/preregistration/harmonized_labels.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_digests(root: Path) -> dict[str, str]:
    """Return the source digests sealed into the label manifest."""

    manifest_path = root / LABEL_MANIFEST
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(manifest.get("source_sha256", {}))


def download(url: str, destination: Path, *, attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with requests.get(url, timeout=(10, 120), stream=True) as response:
                response.raise_for_status()
                with NamedTemporaryFile(
                    mode="wb", dir=destination.parent, delete=False
                ) as temporary:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        temporary.write(chunk)
                    temporary_path = Path(temporary.name)
            os.replace(temporary_path, destination)
            return
        except requests.RequestException as exc:
            error = exc
            logging.warning("attempt %d failed for %s: %s", attempt + 1, url, exc)
    raise RuntimeError(f"failed after {attempts} attempts: {url}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root that the source paths are resolved against",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download files that are already present and already verify",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.root.resolve()
    sealed = registered_digests(root)

    failures: list[str] = []
    for source in SOURCES:
        destination = root / source.relative_path
        expected = sealed.get(source.relative_path)
        binding = "sealed label manifest" if expected else "registration-time observation"
        if expected is None:
            expected = source.observed_sha256

        if destination.is_file() and not args.force and sha256_file(destination) == expected:
            logging.info("present and verified: %s", source.relative_path)
            continue

        logging.info("fetching %s from %s", source.relative_path, source.url)
        download(source.url, destination)
        actual = sha256_file(destination)
        if actual == expected:
            logging.info("verified against %s: %s", binding, source.relative_path)
            continue
        message = (
            f"{source.relative_path}: digest {actual} does not match the {binding} "
            f"digest {expected}"
        )
        if binding == "sealed label manifest":
            failures.append(message)
            logging.error("%s. The sealed analysis is bound to the registered digest.", message)
        else:
            logging.warning(
                "%s. The upstream release has changed since registration; the file is kept "
                "so the difference can be inspected.",
                message,
            )

    if failures:
        logging.error("%d source file(s) do not match their registered digest", len(failures))
        return 1
    logging.info("all %d source metadata files are present under %s", len(SOURCES), root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
