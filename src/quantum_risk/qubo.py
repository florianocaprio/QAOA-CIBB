from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class QuboModel:
    linear: np.ndarray
    quadratic: np.ndarray
    offset: float
    relevance: np.ndarray
    redundancy: np.ndarray
    k: int
    alpha_relevance: float
    beta_redundancy: float
    gamma_cardinality: float

    @property
    def n(self) -> int:
        return len(self.linear)

    def energy(self, bits: np.ndarray) -> float:
        x = np.asarray(bits, dtype=float)
        return float(
            self.offset
            + x @ self.linear
            + sum(self.quadratic[i, j] * x[i] * x[j] for i, j in combinations(range(self.n), 2))
        )

    def direct_energy(self, bits: np.ndarray) -> float:
        x = np.asarray(bits, dtype=float)
        relevance_term = -self.alpha_relevance * float(self.relevance @ x)
        redundancy_term = self.beta_redundancy * sum(
            self.redundancy[i, j] * x[i] * x[j] for i, j in combinations(range(self.n), 2)
        )
        cardinality_term = self.gamma_cardinality * (float(x.sum()) - self.k) ** 2
        return float(relevance_term + redundancy_term + cardinality_term)

    def contributions(self, bits: np.ndarray) -> dict[str, float]:
        x = np.asarray(bits, dtype=float)
        return {
            "relevance": -self.alpha_relevance * float(self.relevance @ x),
            "redundancy": self.beta_redundancy
            * sum(self.redundancy[i, j] * x[i] * x[j] for i, j in combinations(range(self.n), 2)),
            "cardinality": self.gamma_cardinality * (float(x.sum()) - self.k) ** 2,
        }

    def symmetric_matrix(self) -> np.ndarray:
        matrix = np.diag(self.linear.copy())
        for i, j in combinations(range(self.n), 2):
            matrix[i, j] = matrix[j, i] = self.quadratic[i, j] / 2.0
        return matrix


def build_qubo(
    relevance: np.ndarray,
    redundancy: np.ndarray,
    k: int,
    alpha_relevance: float = 1.0,
    beta_redundancy: float = 0.5,
    gamma_cardinality: float = 2.0,
) -> QuboModel:
    relevance = np.asarray(relevance, dtype=float)
    redundancy = np.asarray(redundancy, dtype=float)
    n = len(relevance)
    if redundancy.shape != (n, n):
        raise ValueError("redundancy matrix has incompatible shape")
    if not np.allclose(redundancy, redundancy.T) or not np.allclose(np.diag(redundancy), 0):
        raise ValueError("redundancy must be symmetric with a zero diagonal")
    linear = -alpha_relevance * relevance + gamma_cardinality * (1 - 2 * k)
    quadratic = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        quadratic[i, j] = beta_redundancy * redundancy[i, j] + 2 * gamma_cardinality
    return QuboModel(
        linear=linear,
        quadratic=quadratic,
        offset=float(gamma_cardinality * k * k),
        relevance=relevance,
        redundancy=redundancy,
        k=k,
        alpha_relevance=alpha_relevance,
        beta_redundancy=beta_redundancy,
        gamma_cardinality=gamma_cardinality,
    )
