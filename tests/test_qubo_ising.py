from __future__ import annotations

from itertools import product

import numpy as np
from qiskit.quantum_info import Statevector

from quantum_risk.ising import qubo_to_ising
from quantum_risk.qubo import build_qubo


def test_all_bitstrings_match_direct_qubo_symmetric_and_ising() -> None:
    rng = np.random.default_rng(7)
    n = 7
    relevance = rng.random(n)
    redundancy = rng.random((n, n))
    redundancy = (redundancy + redundancy.T) / 2
    np.fill_diagonal(redundancy, 0)
    qubo = build_qubo(relevance, redundancy, k=3)
    ising = qubo_to_ising(qubo)
    symmetric = qubo.symmetric_matrix()
    for values in product((0, 1), repeat=n):
        bits = np.asarray(values)
        expected = qubo.direct_energy(bits)
        assert np.isclose(qubo.energy(bits), expected, atol=1e-10)
        assert np.isclose(qubo.offset + bits @ symmetric @ bits, expected, atol=1e-10)
        assert np.isclose(ising.energy_from_bits(bits), expected, atol=1e-10)


def test_sparse_pauli_basis_expectation_matches_qubo() -> None:
    relevance = np.array([0.2, 0.7, 1.0])
    redundancy = np.array([[0, 0.1, 0.3], [0.1, 0, 0.2], [0.3, 0.2, 0]])
    qubo = build_qubo(relevance, redundancy, k=2)
    operator = qubo_to_ising(qubo).sparse_pauli_op()
    for index in range(2**qubo.n):
        # Statevector.from_int uses qubit 0 as the least-significant bit.
        bits = np.asarray([(index >> i) & 1 for i in range(qubo.n)])
        state = Statevector.from_int(index, 2**qubo.n)
        observed = float(np.real(state.expectation_value(operator)))
        assert np.isclose(observed, qubo.energy(bits), atol=1e-10)


def test_qubo_does_not_depend_on_paper_reference(tmp_path) -> None:
    reference = tmp_path / "paper.csv"
    reference.write_text("auc\n0.1\n", encoding="utf-8")
    q1 = build_qubo(np.array([0.0, 1.0]), np.zeros((2, 2)), 1)
    reference.write_text("auc\n0.999\n", encoding="utf-8")
    q2 = build_qubo(np.array([0.0, 1.0]), np.zeros((2, 2)), 1)
    assert np.array_equal(q1.linear, q2.linear)
    assert np.array_equal(q1.quadratic, q2.quadratic)
