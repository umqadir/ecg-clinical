"""The source metadata fetcher must agree with the sealed label manifest.

These tests read the committed manifest and never touch the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import load_script

ROOT = Path(__file__).resolve().parents[1]
fetch = load_script("fetch_source_metadata")


def test_sources_cover_every_pipeline_input() -> None:
    relative_paths = {source.relative_path for source in fetch.SOURCES}
    assert relative_paths == {
        "data/raw-metadata/ptb-xl/1.0.3/ptbxl_database.csv",
        "data/raw-metadata/ptb-xl/1.0.3/SHA256SUMS.txt",
        "data/raw-metadata/ecg-arrhythmia/1.0.0/SHA256SUMS.txt",
        "data/raw-metadata/cinc2021_weights.csv",
        "data/raw-metadata/cinc2021_dx_mapping_scored.csv",
    }


def test_pinned_digests_match_the_sealed_manifest() -> None:
    sealed = fetch.registered_digests(ROOT)
    assert sealed, "the committed label manifest must record source digests"
    checked = 0
    for source in fetch.SOURCES:
        expected = sealed.get(source.relative_path)
        if expected is None:
            continue
        assert source.observed_sha256 == expected, source.relative_path
        checked += 1
    assert checked == 3


def test_mapping_urls_use_the_registered_commit() -> None:
    manifest = json.loads(
        (ROOT / "data/derived/preregistration/harmonized_labels.json").read_text(encoding="utf-8")
    )
    assert manifest["source_versions"]["ptb_xl"] == "1.0.3"
    mapping_sources = [source for source in fetch.SOURCES if "cinc2021" in source.relative_path]
    assert len(mapping_sources) == 2
    for source in mapping_sources:
        assert fetch.MAPPING_COMMIT in source.url


def test_registered_digests_is_empty_without_a_manifest(tmp_path: Path) -> None:
    assert fetch.registered_digests(tmp_path) == {}
