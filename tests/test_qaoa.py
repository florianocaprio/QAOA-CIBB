from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator

from quantum_risk.ising import qubo_to_ising
from quantum_risk.qaoa import bitstring_to_bits, build_qaoa_circuit, counts_mean_energy, run_qaoa
from quantum_risk.qubo import build_qubo


def small_qubo():
    relevance = np.array([0.1, 0.8, 1.0, 0.4])
    redundancy = np.array(
        [
            [0, 0.2, 0.1, 0.4],
            [0.2, 0, 0.5, 0.1],
            [0.1, 0.5, 0, 0.2],
            [0.4, 0.1, 0.2, 0],
        ]
    )
    return build_qubo(relevance, redundancy, k=2)


def test_qiskit_bit_ordering_is_explicit() -> None:
    assert bitstring_to_bits("1001", 4).tolist() == [1, 0, 0, 1]
    assert bitstring_to_bits("0010", 4).tolist() == [0, 1, 0, 0]


def test_counts_energy_uses_all_shots() -> None:
    qubo = small_qubo()
    counts = {"0011": 3, "1100": 1}
    expected = (
        3 * qubo.energy(bitstring_to_bits("0011", 4)) + qubo.energy(bitstring_to_bits("1100", 4))
    ) / 4
    assert np.isclose(counts_mean_energy(counts, qubo), expected)


def test_explicit_circuit_structure_and_real_aer_sampling() -> None:
    qubo = small_qubo()
    ising = qubo_to_ising(qubo)
    circuit, costs, mixers = build_qaoa_circuit(ising, p=1)
    assert len(costs) == len(mixers) == 1
    assert circuit.num_qubits == 4
    result = run_qaoa(
        qubo,
        ising,
        p=1,
        mode="statevector_ideal",
        restarts=1,
        optimization_shots=64,
        final_shots=256,
        max_evaluations=4,
        seed=123,
        max_memory_mb=256,
    )
    assert sum(result.counts.values()) == 256
    assert result.status == "completed"
    assert result.bits is not None and int(result.bits.sum()) == 2
    assert result.transpiled_depth > 0


def test_explicit_circuit_matches_independent_qiskit_ansatz() -> None:
    qubo = small_qubo()
    ising = qubo_to_ising(qubo)
    explicit, costs, mixers = build_qaoa_circuit(ising, p=1, measured=False)
    reference = qaoa_ansatz(ising.sparse_pauli_op(include_identity=False), reps=1, flatten=True)
    theta_cost = 0.37
    theta_mix = -0.21
    explicit_state = Statevector.from_instruction(
        explicit.assign_parameters({costs[0]: theta_cost, mixers[0]: theta_mix})
    )
    parameters = {str(parameter): parameter for parameter in reference.parameters}
    reference_state = Statevector.from_instruction(
        reference.assign_parameters({parameters["γ[0]"]: theta_cost, parameters["β[0]"]: theta_mix})
    )
    assert state_fidelity(explicit_state, reference_state) == pytest.approx(1.0, abs=1e-12)


def test_mps_matches_statevector_distribution_on_small_instance() -> None:
    ising = qubo_to_ising(small_qubo())
    circuit, costs, mixers = build_qaoa_circuit(ising, p=1, measured=True)
    bound = circuit.assign_parameters({costs[0]: 0.3, mixers[0]: -0.2})
    shots = 4096
    statevector_counts = (
        AerSimulator(method="statevector")
        .run(bound, shots=shots, seed_simulator=99)
        .result()
        .get_counts()
    )
    mps_counts = (
        AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=64,
            matrix_product_state_truncation_threshold=1e-14,
            mps_sample_measure_algorithm="mps_apply_measure",
        )
        .run(bound, shots=shots, seed_simulator=100)
        .result()
        .get_counts()
    )
    keys = set(statevector_counts) | set(mps_counts)
    total_variation = 0.5 * sum(
        abs(statevector_counts.get(key, 0) / shots - mps_counts.get(key, 0) / shots) for key in keys
    )
    assert total_variation < 0.08
