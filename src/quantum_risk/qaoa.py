from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize

from quantum_risk.ising import IsingModel
from quantum_risk.qubo import QuboModel


@dataclass(frozen=True)
class QaoaResult:
    status: str
    bits: np.ndarray | None
    bitstring: str | None
    energy: float | None
    feasible_fraction: float
    counts: dict[str, int]
    optimal_angles: list[float]
    trace: list[dict[str, float | int]]
    depth: int
    gate_counts: dict[str, int]
    transpiled_depth: int
    transpiled_gate_counts: dict[str, int]
    duration_seconds: float
    backend_method: str
    seed_optimization: int
    seed_final: int


def build_qaoa_circuit(
    ising: IsingModel, p: int, measured: bool = True
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    if p < 1:
        raise ValueError("p must be >= 1")
    theta_cost = ParameterVector("theta_cost", p)
    theta_mix = ParameterVector("theta_mix", p)
    circuit = QuantumCircuit(ising.n)
    circuit.h(range(ising.n))
    for layer in range(p):
        for i, value in enumerate(ising.h):
            if abs(value) > 1e-15:
                circuit.rz(2.0 * theta_cost[layer] * value, i)
        for i, j in combinations(range(ising.n), 2):
            value = ising.j[i, j]
            if abs(value) > 1e-15:
                circuit.rzz(2.0 * theta_cost[layer] * value, i, j)
        for qubit in range(ising.n):
            circuit.rx(2.0 * theta_mix[layer], qubit)
    if measured:
        circuit.measure_all()
    return circuit, theta_cost, theta_mix


def bitstring_to_bits(bitstring: str, n: int) -> np.ndarray:
    compact = bitstring.replace(" ", "")
    if len(compact) != n or set(compact) - {"0", "1"}:
        raise ValueError(f"invalid Qiskit bitstring: {bitstring!r}")
    # Qiskit displays classical bit n-1 on the left; feature i maps to qubit i.
    return np.fromiter((int(char) for char in reversed(compact)), dtype=np.int8)


def counts_mean_energy(counts: dict[str, int], qubo: QuboModel) -> float:
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("counts are empty")
    return (
        sum(qubo.energy(bitstring_to_bits(key, qubo.n)) * count for key, count in counts.items())
        / shots
    )


def _backend(mode: str, max_memory_mb: int, mps_bond: int, mps_threshold: float) -> AerSimulator:
    common = {"max_memory_mb": max_memory_mb, "precision": "double"}
    if mode == "statevector_ideal":
        return AerSimulator(method="statevector", **common)
    if mode == "statevector_noisy":
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["h", "rz", "rx"])
        noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["rzz"])
        return AerSimulator(method="statevector", noise_model=noise, **common)
    if mode == "mps":
        return AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=mps_bond,
            matrix_product_state_truncation_threshold=mps_threshold,
            mps_sample_measure_algorithm="mps_apply_measure",
            mps_log_data=True,
            **common,
        )
    raise ValueError(f"unsupported simulator mode: {mode}")


def run_qaoa(
    qubo: QuboModel,
    ising: IsingModel,
    p: int,
    mode: str,
    restarts: int,
    optimization_shots: int,
    final_shots: int,
    max_evaluations: int,
    seed: int,
    max_memory_mb: int,
    mps_bond: int = 128,
    mps_threshold: float = 1e-12,
) -> QaoaResult:
    started = perf_counter()
    circuit, cost_params, mix_params = build_qaoa_circuit(ising, p, measured=True)
    backend = _backend(mode, max_memory_mb, mps_bond, mps_threshold)
    compiled = transpile(circuit, backend, optimization_level=1, seed_transpiler=seed + 31)
    parameters = list(cost_params) + list(mix_params)
    trace: list[dict[str, float | int]] = []
    best_value = np.inf
    best_angles: np.ndarray | None = None
    eval_index = 0

    for restart in range(restarts):
        restart_index = restart
        rng = np.random.default_rng(seed + 1009 * restart)
        initial = np.concatenate((rng.uniform(-1, 1, p), rng.uniform(-np.pi / 2, np.pi / 2, p)))

        def objective(angles: np.ndarray, restart_index: int = restart_index) -> float:
            nonlocal eval_index
            bound = compiled.assign_parameters(dict(zip(parameters, angles, strict=True)))
            result = backend.run(
                bound,
                shots=optimization_shots,
                seed_simulator=seed + 10_000 + eval_index,
            ).result()
            counts = result.get_counts()
            energy = float(counts_mean_energy(counts, qubo))
            trace.append({"restart": restart_index, "evaluation": eval_index, "energy": energy})
            eval_index += 1
            return energy

        optimized = minimize(
            objective,
            initial,
            method="COBYLA",
            options={"maxiter": max_evaluations, "rhobeg": 0.5, "tol": 1e-3},
        )
        if float(optimized.fun) < best_value:
            best_value = float(optimized.fun)
            best_angles = np.asarray(optimized.x)

    if best_angles is None:
        raise RuntimeError("QAOA optimizer produced no result")
    final_seed = seed + 999_983
    bound = compiled.assign_parameters(dict(zip(parameters, best_angles, strict=True)))
    final_counts = (
        backend.run(bound, shots=final_shots, seed_simulator=final_seed).result().get_counts()
    )
    if sum(final_counts.values()) != final_shots:
        raise RuntimeError("QAOA counts do not sum to requested final shots")
    candidates: list[tuple[float, str, np.ndarray, int]] = []
    feasible_shots = 0
    for bitstring, count in final_counts.items():
        bits = bitstring_to_bits(bitstring, qubo.n)
        if int(bits.sum()) == qubo.k:
            feasible_shots += count
            candidates.append((qubo.energy(bits), bitstring, bits, count))
    candidates.sort(key=lambda item: (item[0], item[1]))
    winner = candidates[0] if candidates else None
    return QaoaResult(
        status="completed" if winner else "no_feasible_sample",
        bits=winner[2] if winner else None,
        bitstring=winner[1] if winner else None,
        energy=float(winner[0]) if winner else None,
        feasible_fraction=feasible_shots / final_shots,
        counts={str(k): int(v) for k, v in final_counts.items()},
        optimal_angles=best_angles.tolist(),
        trace=trace,
        depth=circuit.depth(),
        gate_counts={str(k): int(v) for k, v in circuit.count_ops().items()},
        transpiled_depth=compiled.depth(),
        transpiled_gate_counts={str(k): int(v) for k, v in compiled.count_ops().items()},
        duration_seconds=perf_counter() - started,
        backend_method=mode,
        seed_optimization=seed,
        seed_final=final_seed,
    )
