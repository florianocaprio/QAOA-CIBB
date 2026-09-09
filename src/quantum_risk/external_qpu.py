from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class ExternalQpuRun(BaseModel):
    source: str = "external_qpu"
    backend: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    N: int = Field(ge=1)
    k: int = Field(ge=1)
    p: int = Field(ge=1)
    shots: int = Field(ge=1)
    counts: dict[str, int]
    feature_order: list[str]
    qubo_hash: str = Field(min_length=64, max_length=64)
    preprocessing_hash: str = Field(min_length=64, max_length=64)
    split_hash: str = Field(min_length=64, max_length=64)

    @field_validator("counts")
    @classmethod
    def validate_counts(cls, counts: dict[str, int]) -> dict[str, int]:
        if not counts or any(value < 0 for value in counts.values()):
            raise ValueError("counts must be non-empty non-negative integers")
        return counts

    @field_validator("feature_order")
    @classmethod
    def unique_features(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("feature_order contains duplicates")
        return values

    @model_validator(mode="after")
    def validate_run_contract(self) -> ExternalQpuRun:
        if len(self.feature_order) != self.N:
            raise ValueError("feature_order length must equal N")
        if sum(self.counts.values()) != self.shots:
            raise ValueError("counts must sum to shots")
        if any(len(key.replace(" ", "")) != self.N for key in self.counts):
            raise ValueError("every bitstring must have length N")
        return self


def load_external_qpu_json(path: str | Path) -> ExternalQpuRun:
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("external QPU imports must be validated JSON, never pickle/joblib")
    with source.open(encoding="utf-8") as handle:
        return ExternalQpuRun.model_validate(json.load(handle))
