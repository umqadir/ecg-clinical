"""Shared deterministic training and record-level inference routines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_clinical.data import InferenceRecordDataset


def predict_records(
    model: nn.Module,
    dataset: InferenceRecordDataset,
    *,
    device: torch.device,
    record_batch_size: int,
    num_workers: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average probabilities from the ten registered windows per record."""

    loader = DataLoader(
        dataset,
        batch_size=record_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    record_indices: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, (windows, batch_targets, indices) in enumerate(loader, start=1):
            batch_records, windows_per_record, channels, samples = windows.shape
            inputs = windows.reshape(batch_records * windows_per_record, channels, samples).to(
                device
            )
            batch_probabilities = model(inputs).sigmoid()
            batch_probabilities = batch_probabilities.reshape(
                batch_records, windows_per_record, -1
            ).mean(dim=1)
            probabilities.append(batch_probabilities.cpu().numpy())
            targets.append(batch_targets.numpy())
            record_indices.append(indices.numpy())
            if progress is not None:
                progress(batch_index, len(loader))
    return (
        np.concatenate(probabilities),
        np.concatenate(targets).astype(np.uint8),
        np.concatenate(record_indices),
    )
