from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from quantum_risk.artifacts import atomic_json
from quantum_risk.config import load_config
from quantum_risk.jobs import JobManager
from quantum_risk.resources import preflight_quantum_run
from quantum_risk.validation import run_experiment


def _worker(config_path: str, job_directory: str) -> int:
    job_dir = Path(job_directory)
    import json as json_module

    with (job_dir / "status.json").open(encoding="utf-8") as handle:
        status = json_module.load(handle)
    try:
        output = run_experiment(load_config(config_path))
        atomic_json(
            job_dir / "status.json",
            {
                **status,
                "status": "completed",
                "run_directory": str(output),
                "completed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        return 0
    except Exception as error:
        atomic_json(
            job_dir / "status.json",
            {
                **status,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantum-risk")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "preflight", "launch"):
        command = sub.add_parser(name)
        command.add_argument("config")
    status = sub.add_parser("status")
    status.add_argument("job_id")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("job_id")
    worker = sub.add_parser("worker")
    worker.add_argument("config")
    worker.add_argument("job_directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        return _worker(args.config, args.job_directory)
    if args.command == "run":
        output = run_experiment(load_config(args.config))
        print(output)
        return 0
    if args.command == "preflight":
        config = load_config(args.config)
        result = preflight_quantum_run(
            config.n_candidates,
            config.q_budget,
            config.simulator,
            config.statevector_memory_fraction,
        )
        payload = {
            **result.__dict__,
            "estimated_objective_circuits": (
                config.n_splits
                * len(config.k_values)
                * len(config.p_values)
                * config.qaoa_restarts
                * config.max_objective_evaluations
            ),
        }
        print(json.dumps(payload, indent=2))
        return 0
    manager = JobManager()
    if args.command == "launch":
        print(manager.launch(args.config))
        return 0
    if args.command == "status":
        print(json.dumps(manager.status(args.job_id), indent=2))
        return 0
    if args.command == "cancel":
        print(json.dumps(manager.cancel(args.job_id), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
