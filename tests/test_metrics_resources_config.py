from __future__ import annotations

import numpy as np
import pytest

from quantum_risk.config import ExperimentConfig
from quantum_risk.metrics import classification_metrics, mean_pairwise_jaccard
from quantum_risk.resources import estimate_statevector, preflight_quantum_run
from quantum_risk.scoring import normalize_minmax


def test_metrics_from_known_confusion_matrix() -> None:
    y = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.2, 0.8, 0.1])
    metrics, labels = classification_metrics(y, probabilities)
    assert labels.tolist() == [1, 0, 1, 0]
    assert metrics["tp"] == metrics["tn"] == metrics["fp"] == metrics["fn"] == 1
    assert metrics["recall_30day"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["f1"] == 0.5


def test_constant_mi_normalization_is_safe() -> None:
    normalized, warning = normalize_minmax(np.ones(4))
    assert np.array_equal(normalized, np.zeros(4))
    assert warning is not None


def test_stability_is_pairwise_jaccard() -> None:
    observed = mean_pairwise_jaccard([{"a", "b"}, {"b", "c"}])
    assert observed == pytest.approx(1 / 3)


def test_n_k_qubit_contract() -> None:
    with pytest.raises(ValueError, match="q_budget"):
        ExperimentConfig(n_candidates=13, q_budget=12)
    with pytest.raises(ValueError, match="every k"):
        ExperimentConfig(n_candidates=12, q_budget=12, k_values=[13])


def test_statevector_n40_guard() -> None:
    estimate = estimate_statevector(40)
    assert estimate.statevector_bytes == 16 * 2**40
    assert not estimate.allowed
    with pytest.raises(MemoryError):
        preflight_quantum_run(40, 54, "statevector_ideal", 0.5)
