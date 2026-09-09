from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantum_risk.external_qpu import load_external_qpu_json
from quantum_risk.jobs import JobManager


def test_external_qpu_requires_complete_provenance(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    payload = {
        "backend": "example-backend",
        "run_id": "run-1",
        "N": 2,
        "k": 1,
        "p": 1,
        "shots": 10,
        "counts": {"01": 6, "10": 4},
        "feature_order": ["a", "b"],
        "qubo_hash": "a" * 64,
        "preprocessing_hash": "b" * 64,
        "split_hash": "c" * 64,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    imported = load_external_qpu_json(path)
    assert imported.source == "external_qpu"
    payload["shots"] = 11
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to shots"):
        load_external_qpu_json(path)


def test_job_cancellation_persists_state(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path)
    job_id = "job"
    directory = tmp_path / job_id
    directory.mkdir()
    (directory / "status.json").write_text(
        json.dumps({"status": "running", "pid": 99999}), encoding="utf-8"
    )
    observed = []
    monkeypatch.setattr("os.killpg", lambda pid, signal: observed.append((pid, signal)))
    status = manager.cancel(job_id)
    assert status["status"] == "cancelled"
    assert observed and observed[0][0] == 99999
