"""Tests for the post-seal exploratory per-label interval stage.

The load-bearing property is that this stage rides on the registered resamples rather
than drawing its own. Everything here is synthetic; nothing reads the project's real
waveform stores or sealed predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import load_script
from sklearn.metrics import roc_auc_score

from ecg_clinical.bootstrap import (
    bootstrap_record_weight_batches,
    paired_bootstrap_cohort_statistics,
)

stage = load_script("compute_per_label_intervals")


def _example_cohort(records: int = 120, labels: int = 6) -> tuple[np.ndarray, dict]:
    generator = np.random.default_rng(20260725)
    targets = generator.integers(0, 2, size=(records, labels)).astype(np.uint8)
    first = generator.uniform(0.02, 0.98, size=(records, labels))
    second = np.clip(first + generator.normal(0, 0.15, size=(records, labels)), 0.01, 0.99)
    return targets, {"first": first, "second": second}


def test_per_label_macro_reproduces_registered_distribution_bit_for_bit() -> None:
    """The whole point of the stage: same resamples as the registered macro-AUROC.

    If this ever fails, an interval published beside a registered macro estimate would
    have come from a different bootstrap, which is exactly what the stage refuses to do.
    """

    targets, probabilities = _example_cohort()
    label_indices = np.array([0, 2, 3, 5], dtype=np.int64)
    group_inverse = np.arange(len(targets), dtype=np.int64)

    registered = paired_bootstrap_cohort_statistics(
        targets,
        probabilities,
        label_indices,
        group_inverse,
        replicates=200,
        seed=4321,
        batch_size=50,
        include_calibration=False,
    )
    per_label = stage.per_label_bootstrap(
        targets,
        probabilities,
        group_inverse,
        replicates=200,
        seed=4321,
        batch_size=50,
    )
    for model in probabilities:
        rebuilt = stage.macro_of_per_label(per_label[model], label_indices)
        assert np.array_equal(rebuilt, registered[model])


def test_macro_of_per_label_is_insensitive_to_column_layout() -> None:
    """Selecting columns yields a non-contiguous array whose reduction can differ.

    The stage forces contiguity for that reason; this pins the behaviour so a future
    refactor cannot silently reintroduce last-bit drift into the identity check.
    """

    values = np.random.default_rng(11).uniform(0.5, 1.0, size=(500, 9))
    label_indices = np.array([1, 2, 4, 6, 8], dtype=np.int64)
    contiguous = np.ascontiguousarray(values[:, label_indices])
    assert np.array_equal(
        stage.macro_of_per_label(values, label_indices),
        contiguous.sum(axis=1) / len(label_indices),
    )


def test_macro_of_per_label_averages_over_defined_labels_only() -> None:
    values = np.array([[0.8, np.nan, 0.6], [0.9, 0.7, 0.5]])
    observed = stage.macro_of_per_label(values, np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_allclose(observed, [0.7, 0.7])


def test_cluster_inverse_uses_patients_internally_and_records_externally() -> None:
    metadata = pd.DataFrame({"patient_id": ["a", "a", "b", "c", "c"]})
    internal = stage.cluster_inverse("ptb_test", metadata)
    assert internal[0] == internal[1]
    assert internal[3] == internal[4]
    assert len(np.unique(internal)) == 3

    external = stage.cluster_inverse("ningbo", metadata)
    assert np.array_equal(external, np.arange(5))


def test_micro_bootstrap_matches_sklearn_on_pooled_pairs() -> None:
    """The tiled weights must keep each record's cluster weight on all its labels."""

    targets, probabilities = _example_cohort(records=60, labels=5)
    label_indices = np.array([0, 1, 3], dtype=np.int64)
    group_inverse = np.arange(len(targets), dtype=np.int64)

    observed = stage.micro_bootstrap(
        targets,
        probabilities,
        label_indices,
        group_inverse,
        replicates=4,
        seed=99,
        batch_size=4,
    )
    batches = bootstrap_record_weight_batches(group_inverse, 4, seed=99, batch_size=4)
    weights = next(iter(batches))[1]
    flat_targets = targets[:, label_indices].reshape(-1)
    for model, probability in probabilities.items():
        flat_probability = probability[:, label_indices].reshape(-1)
        for replicate in range(4):
            tiled = np.repeat(weights[replicate], len(label_indices))
            expected = roc_auc_score(flat_targets, flat_probability, sample_weight=tiled)
            assert observed[model][replicate] == pytest.approx(expected)
