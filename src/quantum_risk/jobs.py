from __future__ import annotations

import os
import signal
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from quantum_risk.artifacts import atomic_json


class JobManager:
    def __init__(self, root: str | Path = "runs/jobs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def launch(self, config_path: str | Path) -> str:
        job_id = uuid.uuid4().hex
        directory = self.root / job_id
        directory.mkdir(parents=True, exist_ok=False)
        log_path = directory / "worker.log"
        initial = {
            "status": "queued",
            "job_id": job_id,
            "config_path": str(config_path),
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(directory / "status.json", initial)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "quantum_risk.cli",
                    "worker",
                    str(config_path),
                    str(directory),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        atomic_json(
            directory / "status.json",
            {
                **initial,
                "status": "running",
                "pid": process.pid,
                "started_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return job_id

    def status(self, job_id: str) -> dict[str, object]:
        path = self.root / job_id / "status.json"
        if not path.exists():
            raise FileNotFoundError(f"unknown job: {job_id}")
        import json

        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def cancel(self, job_id: str) -> dict[str, object]:
        status = self.status(job_id)
        if status.get("status") != "running":
            return status
        pid = int(status["pid"])
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        cancelled = {
            **status,
            "status": "cancelled",
            "cancelled_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(self.root / job_id / "status.json", cancelled)
        return cancelled
