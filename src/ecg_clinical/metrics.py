"""Registered discrimination, threshold, and calibration metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float("nan")
    return float(roc_auc_score(targets, probabilities))


def safe_average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if targets.sum() == 0:
        return float("nan")
    return float(average_precision_score(targets, probabilities))


def per_label_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, np.ndarray]:
    if targets.shape != probabilities.shape or targets.ndim != 2:
        raise ValueError("targets and probabilities must be matching 2D arrays")
    auroc = np.asarray(
        [
            safe_roc_auc(targets[:, index], probabilities[:, index])
            for index in range(targets.shape[1])
        ]
    )
    auprc = np.asarray(
        [
            safe_average_precision(targets[:, index], probabilities[:, index])
            for index in range(targets.shape[1])
        ]
    )
    return {"auroc": auroc, "auprc": auprc}


def discrimination_summary(
    targets: np.ndarray, probabilities: np.ndarray, headline_indices: np.ndarray
) -> dict[str, float]:
    per_label = per_label_metrics(targets, probabilities)
    headline_auc = per_label["auroc"][headline_indices]
    headline_auprc = per_label["auprc"][headline_indices]
    return {
        "macro_auroc": float(np.nanmean(headline_auc)),
        "macro_auprc": float(np.nanmean(headline_auprc)),
        "micro_auroc": safe_roc_auc(
            targets[:, headline_indices].ravel(),
            probabilities[:, headline_indices].ravel(),
        ),
        "micro_auprc": safe_average_precision(
            targets[:, headline_indices].ravel(), probabilities[:, headline_indices].ravel()
        ),
    }


def select_f1_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Choose the smallest observed threshold attaining maximum validation F1."""

    thresholds = np.empty(targets.shape[1], dtype=np.float64)
    for label_index in range(targets.shape[1]):
        observed = np.unique(probabilities[:, label_index])
        candidates = np.concatenate(([0.0], observed, [1.0]))
        predictions = probabilities[:, label_index, None] >= candidates[None, :]
        truth = targets[:, label_index, None].astype(bool)
        true_positive = np.logical_and(predictions, truth).sum(axis=0)
        false_positive = np.logical_and(predictions, ~truth).sum(axis=0)
        false_negative = np.logical_and(~predictions, truth).sum(axis=0)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator > 0,
        )
        best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
        thresholds[label_index] = candidates[best].min()
    return thresholds


def threshold_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    label_indices: np.ndarray,
) -> dict[str, float]:
    truth = targets[:, label_indices].astype(bool)
    predictions = probabilities[:, label_indices] >= thresholds[label_indices][None, :]
    true_positive = np.logical_and(predictions, truth).sum(axis=0)
    true_negative = np.logical_and(~predictions, ~truth).sum(axis=0)
    false_positive = np.logical_and(predictions, ~truth).sum(axis=0)
    false_negative = np.logical_and(~predictions, truth).sum(axis=0)

    f1_per_label = np.divide(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
        out=np.full(len(label_indices), np.nan),
        where=(2 * true_positive + false_positive + false_negative) > 0,
    )
    sensitivity = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.full(len(label_indices), np.nan),
        where=(true_positive + false_negative) > 0,
    )
    specificity = np.divide(
        true_negative,
        true_negative + false_positive,
        out=np.full(len(label_indices), np.nan),
        where=(true_negative + false_positive) > 0,
    )
    micro_tp = true_positive.sum()
    micro_fp = false_positive.sum()
    micro_fn = false_negative.sum()
    return {
        "macro_f1": float(np.nanmean(f1_per_label)),
        "micro_f1": float(2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn)),
        "macro_sensitivity": float(np.nanmean(sensitivity)),
        "macro_specificity": float(np.nanmean(specificity)),
    }


def expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    targets = np.asarray(targets).ravel()
    probabilities = np.asarray(probabilities).ravel()
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for bin_index in range(bins):
        selected = assignments == bin_index
        if selected.any():
            error += selected.mean() * abs(
                targets[selected].mean() - probabilities[selected].mean()
            )
    return float(error)


def calibration_summary(
    targets: np.ndarray, probabilities: np.ndarray, headline_indices: np.ndarray, bins: int = 15
) -> dict[str, float | list[float]]:
    label_ece = [
        expected_calibration_error(targets[:, index], probabilities[:, index], bins=bins)
        for index in range(targets.shape[1])
    ]
    headline_targets = targets[:, headline_indices]
    headline_probabilities = probabilities[:, headline_indices]
    return {
        "macro_per_label_ece": float(np.mean(np.asarray(label_ece)[headline_indices])),
        "pooled_ece": expected_calibration_error(
            headline_targets, headline_probabilities, bins=bins
        ),
        "macro_brier": float(np.mean(np.square(headline_probabilities - headline_targets))),
        "per_label_ece": label_ece,
    }


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return expit(logit(clipped) / temperature)


def binary_negative_log_likelihood(targets: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped)))


def fit_scalar_temperature(targets: np.ndarray, probabilities: np.ndarray) -> float:
    result = minimize_scalar(
        lambda log_temperature: binary_negative_log_likelihood(
            targets, apply_temperature(probabilities, float(np.exp(log_temperature)))
        ),
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(np.exp(result.x))
