from __future__ import annotations

import numpy as np

from quantum_risk.classical import _build_l1_logistic_model, l1_ranked_top_k


def _deterministic_binary_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(42)
    informative_1 = rng.normal(size=240)
    informative_2 = rng.normal(size=240)
    irrelevant = rng.normal(size=240)
    constant_zero = np.zeros(240)
    target = (2.5 * informative_1 - 1.5 * informative_2 > 0).astype(int)
    train = np.column_stack(
        [informative_1, informative_2, irrelevant, constant_zero]
    )
    return train, target, ["informative_1", "informative_2", "irrelevant", "constant_zero"]


def test_l1_model_uses_explicit_l1_penalty_with_liblinear() -> None:
    model = _build_l1_logistic_model(0.3, seed=42, class_weight="balanced")
    assert model.penalty == "l1"
    assert model.solver == "liblinear"


def test_l1_model_produces_sparse_coefficients() -> None:
    train, target, _ = _deterministic_binary_data()
    model = _build_l1_logistic_model(0.1, seed=42, class_weight=None)
    model.fit(train, target)
    coefficients = np.abs(model.coef_[0])
    assert np.count_nonzero(coefficients <= 1e-10) >= 1
    assert coefficients[0] > 1e-10
    assert coefficients[1] > 1e-10


def test_l1_ranked_top_k_has_requested_cardinality() -> None:
    train, target, names = _deterministic_binary_data()
    bits, selected_c, status = l1_ranked_top_k(
        train, target, names, k=2, seed=42, class_weight=None
    )
    assert status == "completed"
    assert selected_c is not None
    assert bits is not None
    assert int(bits.sum()) == 2


def test_l1_ranked_top_k_reports_insufficient_active_features() -> None:
    train = np.zeros((40, 3))
    target = np.tile([0, 1], 20)
    bits, selected_c, status = l1_ranked_top_k(
        train, target, ["a", "b", "c"], k=1, seed=42, class_weight=None
    )
    assert bits is None
    assert selected_c is None
    assert status == "insufficient_active_features"


def test_l1_ranked_top_k_is_deterministic_at_fixed_seed() -> None:
    train, target, names = _deterministic_binary_data()
    first = l1_ranked_top_k(train, target, names, k=2, seed=42, class_weight="balanced")
    second = l1_ranked_top_k(train, target, names, k=2, seed=42, class_weight="balanced")
    assert first[1:] == second[1:]
    assert first[0] is not None
    assert second[0] is not None
    assert np.array_equal(first[0], second[0])
