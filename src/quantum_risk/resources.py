from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ResourceEstimate:
    qubits: int
    statevector_bytes: int
    available_bytes: int
    allowed_bytes: int
    allowed: bool
    reason: str | None


def estimate_statevector(
    qubits: int, memory_fraction: float = 0.5, overhead: float = 1.5
) -> ResourceEstimate:
    raw = 16 * (2**qubits)
    required = int(raw * overhead)
    available = int(psutil.virtual_memory().available)
    allowed = int(available * memory_fraction)
    ok = required <= allowed
    reason = (
        None
        if ok
        else (
            f"statevector requires about {required / 2**30:.2f} GiB including overhead, "
            f"above the configured allowance of {allowed / 2**30:.2f} GiB"
        )
    )
    return ResourceEstimate(qubits, raw, available, allowed, ok, reason)


def preflight_quantum_run(
    qubits: int, q_budget: int, simulator: str, memory_fraction: float
) -> ResourceEstimate:
    if qubits > q_budget:
        raise ValueError(f"q_used={qubits} exceeds q_budget={q_budget}")
    estimate = estimate_statevector(qubits, memory_fraction)
    if simulator.startswith("statevector") and not estimate.allowed:
        raise MemoryError(estimate.reason)
    if simulator == "mps" and not estimate.allowed:
        return ResourceEstimate(
            qubits=estimate.qubits,
            statevector_bytes=estimate.statevector_bytes,
            available_bytes=estimate.available_bytes,
            allowed_bytes=estimate.allowed_bytes,
            allowed=True,
            reason=(
                f"{estimate.reason}; MPS is permitted but remains approximate and may exceed "
                "time or bond-dimension limits"
            ),
        )
    return estimate
