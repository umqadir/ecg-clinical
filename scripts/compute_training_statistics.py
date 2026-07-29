#!/usr/bin/env python3
"""Compute frozen per-lead normalization statistics from PTB-XL folds 1-8."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/cache/waveforms/ptb_xl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/derived/training_normalization.json")
    )
    parser.add_argument("--chunk-records", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signals = np.load(args.store / "signals.npy", mmap_mode="r")
    status = np.load(args.store / "status.npy", mmap_mode="r")
    metadata = pd.read_parquet(args.store / "metadata.parquet")
    indices = np.flatnonzero(metadata.strat_fold.between(1, 8).to_numpy() & (status == 1))
    sums = np.zeros(12, dtype=np.float64)
    squared_sums = np.zeros(12, dtype=np.float64)
    for start in range(0, len(indices), args.chunk_records):
        chunk = np.asarray(signals[indices[start : start + args.chunk_records]], dtype=np.float64)
        sums += chunk.sum(axis=(0, 2))
        squared_sums += np.square(chunk).sum(axis=(0, 2))
    samples_per_lead = len(indices) * signals.shape[-1]
    means = sums / samples_per_lead
    variances = squared_sums / samples_per_lead - np.square(means)
    standard_deviations = np.sqrt(np.maximum(variances, 0))
    payload = {
        "schema_version": 1,
        "source": "PTB-XL 1.0.3 folds 1-8 only",
        "training_records": len(indices),
        "samples_per_lead": samples_per_lead,
        "units": "mV",
        "lead_order": [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        "per_lead_mean_mv": means.tolist(),
        "per_lead_standard_deviation_mv": standard_deviations.tolist(),
        "calculation": "population moments over every 100 Hz sample in valid training records",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
