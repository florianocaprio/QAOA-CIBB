from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ExperimentConfig(BaseModel):
    name: str = "quick"
    dataset_path: Path = Path("data/raw/diabetic_data.csv")
    mapping_path: Path | None = Path("data/raw/IDS_mapping.csv")
    output_root: Path = Path("runs")
    seed: int = 42
    n_candidates: int = Field(12, ge=2)
    q_budget: int = Field(12, ge=2)
    k_values: list[int] = [5]
    p_values: list[int] = [1]
    n_splits: int = Field(3, ge=2)
    max_patients: int | None = Field(2000, ge=100)
    alpha_relevance: float = Field(1.0, ge=0)
    beta_redundancy: float = Field(0.5, ge=0)
    gamma_cardinality: float = Field(2.0, ge=0)
    simulator: Literal["statevector_ideal", "statevector_noisy", "mps"] = "statevector_ideal"
    qaoa_restarts: int = Field(1, ge=1)
    optimization_shots: int = Field(512, ge=1)
    final_shots: int = Field(2048, ge=1)
    max_objective_evaluations: int = Field(60, ge=1)
    statevector_memory_fraction: float = Field(0.5, gt=0, le=0.9)
    mps_max_bond_dimension: int = Field(128, ge=2)
    mps_truncation_threshold: float = Field(1e-12, gt=0)
    sa_num_reads: int = Field(1000, ge=1)
    sa_num_sweeps: int = Field(1000, ge=1)
    class_weight: Literal["balanced"] | None = "balanced"
    mi_n_neighbors: int = Field(3, ge=1)
    require_original_manifest: bool = False
    run_qaoa: bool = True

    @model_validator(mode="after")
    def validate_contract(self) -> ExperimentConfig:
        if self.n_candidates > self.q_budget:
            raise ValueError("n_candidates must be <= q_budget because q_used=N")
        if not self.k_values or any(k < 1 or k > self.n_candidates for k in self.k_values):
            raise ValueError("every k must satisfy 1 <= k <= n_candidates")
        if not self.p_values or any(p < 1 for p in self.p_values):
            raise ValueError("every QAOA depth p must be >= 1")
        minimum_evaluations = 2 * max(self.p_values) + 2
        if self.max_objective_evaluations < minimum_evaluations:
            raise ValueError("max_objective_evaluations must be at least 2*max(p)+2 for COBYLA")
        if self.require_original_manifest and self.run_qaoa:
            raise ValueError(
                "paper_reference requires an original feature manifest before computed QAOA runs"
            )
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=True)
