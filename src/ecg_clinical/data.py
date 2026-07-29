"""Memory-mapped datasets for deterministic ECG training and inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WINDOW_SAMPLES = 250
RECORD_SAMPLES = 1000
INFERENCE_WINDOWS = 10


def inference_window_starts() -> np.ndarray:
    return np.rint(np.linspace(0, RECORD_SAMPLES - WINDOW_SAMPLES, INFERENCE_WINDOWS)).astype(
        np.int64
    )


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    mean = np.asarray(payload["per_lead_mean_mv"], dtype=np.float32)
    standard_deviation = np.asarray(payload["per_lead_standard_deviation_mv"], dtype=np.float32)
    if mean.shape != (12,) or standard_deviation.shape != (12,):
        raise ValueError("normalization statistics must contain 12 leads")
    if not np.isfinite(mean).all() or not np.isfinite(standard_deviation).all():
        raise ValueError("normalization statistics must be finite")
    if (standard_deviation <= 0).any():
        raise ValueError("normalization standard deviations must be positive")
    return mean[:, None], standard_deviation[:, None]


class WaveformStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.signals = np.load(root / "signals.npy", mmap_mode="r")
        self.targets = np.load(root / "targets.npy", mmap_mode="r")
        self.metadata = pd.read_parquet(root / "metadata.parquet")
        self.status = np.load(root / "status.npy", mmap_mode="r")
        expected_records = len(self.metadata)
        if self.signals.shape != (expected_records, 12, RECORD_SAMPLES):
            raise ValueError(f"invalid signal store shape: {self.signals.shape}")
        if self.targets.shape != (expected_records, 16):
            raise ValueError(f"invalid target store shape: {self.targets.shape}")
        if self.status.shape != (expected_records,) or not np.all(self.status == 1):
            raise ValueError("waveform store is incomplete")

    def indices_for_folds(self, folds: tuple[int, ...]) -> np.ndarray:
        return np.flatnonzero(self.metadata.strat_fold.isin(folds).to_numpy())


class TrainingCropDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        store: WaveformStore,
        indices: np.ndarray,
        mean: np.ndarray,
        standard_deviation: np.ndarray,
        seed: int,
    ) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index = int(self.indices[item])
        generator = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, record_index])
        )
        start = int(generator.integers(0, RECORD_SAMPLES - WINDOW_SAMPLES + 1))
        signal = np.asarray(
            self.store.signals[record_index, :, start : start + WINDOW_SAMPLES], dtype=np.float32
        )
        normalized = (signal - self.mean) / self.standard_deviation
        target = np.asarray(self.store.targets[record_index], dtype=np.float32)
        return torch.from_numpy(normalized.copy()), torch.from_numpy(target.copy())


class InferenceRecordDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        store: WaveformStore,
        indices: np.ndarray,
        mean: np.ndarray,
        standard_deviation: np.ndarray,
    ) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.starts = inference_window_starts()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        record_index = int(self.indices[item])
        signal = np.asarray(self.store.signals[record_index], dtype=np.float32)
        normalized = (signal - self.mean) / self.standard_deviation
        windows = np.stack(
            [normalized[:, start : start + WINDOW_SAMPLES] for start in self.starts], axis=0
        )
        target = np.asarray(self.store.targets[record_index], dtype=np.float32)
        return torch.from_numpy(windows), torch.from_numpy(target.copy()), record_index
