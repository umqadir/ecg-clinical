#!/usr/bin/env python3
"""Audit completed waveform stores and reconcile labels with preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

COHORTS = ("ptb_xl", "chapman_shaoxing", "ningbo")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-root", type=Path, default=Path("data/cache/waveforms"))
    parser.add_argument(
        "--registered-counts",
        type=Path,
        default=Path("data/derived/preregistration/harmonized_label_counts.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/derived/waveform_audit.json"))
    parser.add_argument(
        "--count-output", type=Path, default=Path("data/derived/waveform_target_counts.csv")
    )
    parser.add_argument("--chunk-records", type=int, default=256)
    return parser.parse_args()


def signal_summary(
    signals: np.ndarray, metadata: pd.DataFrame, chunk_records: int
) -> dict[str, object]:
    minimum = float("inf")
    maximum = -float("inf")
    absolute_maximum = -float("inf")
    absolute_maximum_record = ""
    finite_samples = 0
    total_samples = int(signals.size)
    for start in range(0, len(signals), chunk_records):
        chunk = np.asarray(signals[start : start + chunk_records], dtype=np.float32)
        finite_samples += int(np.isfinite(chunk).sum())
        minimum = min(minimum, float(np.nanmin(chunk)))
        maximum = max(maximum, float(np.nanmax(chunk)))
        per_record = np.nanmax(np.abs(chunk), axis=(1, 2))
        local_index = int(np.argmax(per_record))
        if float(per_record[local_index]) > absolute_maximum:
            absolute_maximum = float(per_record[local_index])
            absolute_maximum_record = str(metadata.iloc[start + local_index].record_id)
    return {
        "minimum_mv": minimum,
        "maximum_mv": maximum,
        "absolute_maximum_mv": absolute_maximum,
        "absolute_maximum_record_id": absolute_maximum_record,
        "finite_samples": finite_samples,
        "total_samples": total_samples,
        "all_samples_finite": finite_samples == total_samples,
    }


def main() -> int:
    args = parse_args()
    registered = pd.read_csv(args.registered_counts).set_index("label_key")
    label_keys = registered.index.tolist()
    target_rows: list[dict[str, object]] = []
    audit: dict[str, object] = {
        "schema_version": 1,
        "validation_rule": (
            "12 expected leads, 10 seconds, finite physical mV samples, "
            "interpretable gains and units"
        ),
        "cohorts": {},
        "attrition": {},
        "known_source_anomalies": {
            "S23074": (
                "Ningbo Challenge header ID is S23074 but its declared and checksummed signal "
                "matrix is JS23074.mat; the header ID remains the evaluation record ID"
            )
        },
    }
    metadata_by_cohort: dict[str, pd.DataFrame] = {}
    targets_by_cohort: dict[str, np.ndarray] = {}
    for cohort in COHORTS:
        root = args.store_root / cohort
        signals = np.load(root / "signals.npy", mmap_mode="r")
        targets = np.load(root / "targets.npy", mmap_mode="r")
        status = np.load(root / "status.npy", mmap_mode="r")
        metadata = pd.read_parquet(root / "metadata.parquet")
        metadata_by_cohort[cohort] = metadata
        targets_by_cohort[cohort] = targets
        if metadata.record_id.duplicated().any():
            raise ValueError(f"{cohort}: duplicate record IDs")
        if signals.shape != (len(metadata), 12, 1000) or targets.shape != (len(metadata), 16):
            raise ValueError(f"{cohort}: incompatible store shapes")
        valid = int((status == 1).sum())
        cohort_summary = {
            "records": len(metadata),
            "valid_records": valid,
            "invalid_records": int(len(metadata) - valid),
            "signal_shape": list(signals.shape),
            "signal_dtype": str(signals.dtype),
            "signals_sha256": sha256(root / "signals.npy"),
            "targets_sha256": sha256(root / "targets.npy"),
            **signal_summary(signals, metadata, args.chunk_records),
        }
        if not cohort_summary["all_samples_finite"] or valid != len(metadata):
            raise ValueError(f"{cohort}: waveform validation failed")
        audit["cohorts"][cohort] = cohort_summary
        audit["attrition"][cohort] = {
            "source_records": len(metadata),
            "included_records": valid,
            "excluded_records": int(len(metadata) - valid),
            "reasons": {},
        }

    ptb_metadata = metadata_by_cohort["ptb_xl"]
    patient_fold_counts = ptb_metadata.groupby("patient_id").strat_fold.nunique()
    if int((patient_fold_counts > 1).sum()) != 0:
        raise ValueError("PTB-XL patient identity crosses folds")
    audit["ptb_patient_split"] = {
        "patients": int(ptb_metadata.patient_id.nunique()),
        "patients_crossing_folds": 0,
    }

    cohort_slices = {
        "ptb_train": ("ptb_xl", ptb_metadata.strat_fold.between(1, 8).to_numpy()),
        "ptb_validation": ("ptb_xl", (ptb_metadata.strat_fold == 9).to_numpy()),
        "ptb_test": ("ptb_xl", (ptb_metadata.strat_fold == 10).to_numpy()),
        "chapman_shaoxing": (
            "chapman_shaoxing",
            np.ones(len(metadata_by_cohort["chapman_shaoxing"]), dtype=bool),
        ),
        "ningbo": ("ningbo", np.ones(len(metadata_by_cohort["ningbo"]), dtype=bool)),
    }
    for label_index, label_key in enumerate(label_keys):
        row: dict[str, object] = {"label_key": label_key}
        for split, (cohort, mask) in cohort_slices.items():
            positives = int(np.asarray(targets_by_cohort[cohort][mask, label_index]).sum())
            row[f"{split}_positives"] = positives
            registered_value = int(registered.loc[label_key, f"{split}_positives"])
            if positives != registered_value:
                raise ValueError(
                    f"{label_key} {split}: store count {positives} != registered {registered_value}"
                )
        target_rows.append(row)
    args.count_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(target_rows).to_csv(args.count_output, index=False)
    audit["target_count_reconciliation"] = {
        "labels": len(label_keys),
        "all_counts_match_preregistration": True,
        "output": str(args.count_output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
