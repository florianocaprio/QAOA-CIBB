import pytest

from quantum_risk.qpu import (
    QpuNotConfiguredError,
    iqm_integration_status,
    require_direct_qpu_execution,
)


def test_qpu_missing_credentials_is_explicit_and_never_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("IQM_TOKEN", raising=False)
    status = iqm_integration_status()
    assert not status.configured
    assert not status.execution_enabled
    assert "fallback silenzioso" in status.message
    with pytest.raises(QpuNotConfiguredError, match="Credenziali IQM non configurate"):
        require_direct_qpu_execution()


def test_qpu_token_is_never_returned(monkeypatch) -> None:
    secret = "do-not-expose-this-token"
    monkeypatch.setenv("IQM_TOKEN", secret)
    status = iqm_integration_status()
    assert status.configured
    assert not status.execution_enabled
    assert secret not in repr(status)
