from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantum_risk.config import ExperimentConfig
from quantum_risk.validation import run_experiment


def test_real_uci_end_to_end_reduced(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="pytest-e2e",
        dataset_path=Path("data/raw/diabetic_data.csv"),
        mapping_path=Path("data/raw/IDS_mapping.csv"),
        output_root=tmp_path,
        seed=42,
        n_candidates=4,
        q_budget=4,
        k_values=[2],
        p_values=[1],
        n_splits=2,
        max_patients=300,
        qaoa_restarts=1,
        optimization_shots=32,
        final_shots=128,
        max_objective_evaluations=4,
        sa_num_reads=20,
        sa_num_sweeps=20,
    )
    run_dir = run_experiment(config)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    summary = pd.read_csv(run_dir / "summary_metrics.csv")
    assert {"computed_classical", "computed_qaoa_simulation"} <= set(summary["source"])
    assert (run_dir / "predictions.csv").stat().st_size > 0
    assert (run_dir / "summary_metrics.tex").exists()
    assert (run_dir / "auc_by_method.png").exists()
    for fold in (0, 1):
        detail = json.loads((run_dir / f"fold_{fold}.json").read_text())
        assert detail["patient_overlap"] == 0
        assert len(detail["candidate_manifest"]) == 4
