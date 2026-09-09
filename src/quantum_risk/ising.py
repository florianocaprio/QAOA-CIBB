from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from quantum_risk.qubo import QuboModel


@dataclass(frozen=True)
class IsingModel:
    h: np.ndarray
    j: np.ndarray
    offset_identity: float

    @property
    def n(self) -> int:
        return len(self.h)

    def energy_from_bits(self, bits: np.ndarray) -> float:
        z = 1.0 - 2.0 * np.asarray(bits, dtype=float)
        return float(
            self.offset_identity
            + self.h @ z
            + sum(self.j[i, j] * z[i] * z[j] for i, j in combinations(range(self.n), 2))
        )

    def sparse_pauli_op(self, include_identity: bool = True) -> SparsePauliOp:
        terms: list[tuple[str, complex]] = []
        if include_identity:
            terms.append(("I" * self.n, complex(self.offset_identity)))
        for i, value in enumerate(self.h):
            if abs(value) > 1e-15:
                label = ["I"] * self.n
                label[self.n - 1 - i] = "Z"
                terms.append(("".join(label), complex(value)))
        for i, j in combinations(range(self.n), 2):
            value = self.j[i, j]
            if abs(value) > 1e-15:
                label = ["I"] * self.n
                label[self.n - 1 - i] = "Z"
                label[self.n - 1 - j] = "Z"
                terms.append(("".join(label), complex(value)))
        return SparsePauliOp.from_list(terms or [("I" * self.n, 0.0)])


def qubo_to_ising(qubo: QuboModel) -> IsingModel:
    n = qubo.n
    interactions = qubo.quadratic
    h = np.empty(n, dtype=float)
    for i in range(n):
        incident = sum(interactions[min(i, j), max(i, j)] for j in range(n) if i != j)
        h[i] = -qubo.linear[i] / 2.0 - incident / 4.0
    j_matrix = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        j_matrix[i, j] = interactions[i, j] / 4.0
    identity = (
        qubo.offset
        + float(qubo.linear.sum()) / 2.0
        + sum(interactions[i, j] for i, j in combinations(range(n), 2)) / 4.0
    )
    return IsingModel(h=h, j=j_matrix, offset_identity=float(identity))
