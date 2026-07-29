from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]


def _markdown_count_rows() -> list[tuple[str, list[int], bool]]:
    text = (ROOT / "preregistration/PREREGISTRATION.md").read_text()
    rows: list[tuple[str, list[int], bool]] = []
    for line in text.splitlines():
        if not re.match(r"^\| [0-9]", line):
            continue
        # Escaped pipes appear inside grouped SNOMED keys. Counts and eligibility
        # occupy the final six columns and are therefore parsed from the right.
        fields = [field.strip() for field in line.strip("|").split("|")]
        key_fields = fields[:-7]
        key = "|".join(key_fields).replace("\\", "")
        count_fields = fields[-6:-1]
        rows.append((key, [int(value) for value in count_fields], fields[-1] == "yes"))
    return rows


def test_preregistration_hash_and_counts_match_generated_artifacts() -> None:
    manifest_path = ROOT / "data/derived/preregistration/harmonized_labels.json"
    protocol = json.loads((ROOT / "configs/preregistered_protocol.json").read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert digest == protocol["data"]["label_manifest_sha256"]

    counts = pd.read_csv(
        ROOT / "data/derived/preregistration/harmonized_label_counts.csv"
    ).set_index("label_key")
    rendered = _markdown_count_rows()
    assert len(rendered) == len(counts) == 16

    columns = [
        "ptb_train_positives",
        "ptb_validation_positives",
        "ptb_test_positives",
        "chapman_shaoxing_positives",
        "ningbo_positives",
    ]
    for key, observed_counts, headline in rendered:
        assert key in counts.index
        expected = counts.loc[key]
        assert observed_counts == [int(expected[column]) for column in columns]
        assert headline is bool(expected["headline_eligible"])
