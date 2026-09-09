from __future__ import annotations

import os
from dataclasses import dataclass


class QpuNotConfiguredError(RuntimeError):
    """Raised when physical-QPU execution is requested without credentials."""


@dataclass(frozen=True)
class QpuIntegrationStatus:
    provider: str
    configured: bool
    execution_enabled: bool
    message: str


def iqm_integration_status() -> QpuIntegrationStatus:
    configured = bool(os.environ.get("IQM_TOKEN", "").strip())
    if configured:
        return QpuIntegrationStatus(
            provider="IQM",
            configured=True,
            execution_enabled=False,
            message=(
                "Credenziale rilevata, ma l’esecuzione remota diretta non è abilitata in questa "
                "release. Importare soltanto run hardware con provenienza completa."
            ),
        )
    return QpuIntegrationStatus(
        provider="IQM",
        configured=False,
        execution_enabled=False,
        message=(
            "Credenziali IQM non configurate. Nessun job QPU verrà eseguito e non è previsto "
            "alcun fallback silenzioso verso Aer."
        ),
    )


def require_direct_qpu_execution() -> None:
    status = iqm_integration_status()
    raise QpuNotConfiguredError(status.message)
