from __future__ import annotations

from itertools import combinations, product

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from sklearn.linear_model import LogisticRegression

from quantum_risk.qubo import QuboModel

L1_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)


def _build_l1_logistic_model(
    c_value: float, seed: int, class_weight: str | None
) -> LogisticRegression:
    return LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=c_value,
        class_weight=class_weight,
        max_iter=2000,
        random_state=seed,
    )


def top_k_bits(scores: np.ndarray, names: list[str], k: int) -> np.ndarray:
    order = np.lexsort((np.asarray(names), -np.asarray(scores)))
    bits = np.zeros(len(names), dtype=np.int8)
    bits[order[:k]] = 1
    return bits


def l1_ranked_top_k(
    train: np.ndarray,
    y_train: np.ndarray,
    names: list[str],
    k: int,
    seed: int,
    class_weight: str | None,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray | None, float | None, str]:
    for c_value in L1_C_GRID:
        model = _build_l1_logistic_model(c_value, seed, class_weight)
        model.fit(train, y_train)
        coefficients = np.abs(model.coef_[0])
        if int((coefficients > tolerance).sum()) >= k:
            return top_k_bits(coefficients, names, k), c_value, "completed"
    return None, None, "insufficient_active_features"


def simulated_annealing_select(
    qubo: QuboModel,
    num_reads: int,
    num_sweeps: int,
    seed: int,
) -> tuple[np.ndarray | None, list[dict[str, object]], str]:
    linear = {i: float(value) for i, value in enumerate(qubo.linear)}
    quadratic = {
        (i, j): float(qubo.quadratic[i, j])
        for i, j in combinations(range(qubo.n), 2)
        if abs(qubo.quadratic[i, j]) > 1e-15
    }
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, qubo.offset, dimod.BINARY)
    sampleset = SimulatedAnnealingSampler().sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )
    raw: list[dict[str, object]] = []
    feasible: list[tuple[float, tuple[int, ...], np.ndarray]] = []
    for record in sampleset.data(fields=["sample", "energy", "num_occurrences"]):
        bits = np.asarray([record.sample[i] for i in range(qubo.n)], dtype=np.int8)
        row = {
            "bitstring_q0_first": "".join(map(str, bits.tolist())),
            "energy": float(record.energy),
            "cardinality": int(bits.sum()),
            "num_occurrences": int(record.num_occurrences),
        }
        raw.append(row)
        if int(bits.sum()) == qubo.k:
            feasible.append((float(record.energy), tuple(bits.tolist()), bits))
    feasible.sort(key=lambda item: (item[0], item[1]))
    if not feasible:
        return None, raw, "no_feasible_sample"
    return feasible[0][2], raw, "completed"


def exhaustive_select(qubo: QuboModel, exact_cardinality: bool = True) -> tuple[np.ndarray, float]:
    if qubo.n > 20:
        raise ValueError("exhaustive search is limited to N<=20")
    candidates = []
    for values in product((0, 1), repeat=qubo.n):
        bits = np.asarray(values, dtype=np.int8)
        if exact_cardinality and int(bits.sum()) != qubo.k:
            continue
        candidates.append((qubo.energy(bits), values, bits))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2], float(candidates[0][0])


def random_k(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = np.zeros(n, dtype=np.int8)
    bits[rng.choice(n, size=k, replace=False)] = 1
    return bits
