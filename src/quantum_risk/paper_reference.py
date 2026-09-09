from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_predictive_reference(root: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(root) / "reference" / "paper_predictive_metrics.csv")


def load_quality_reference(root: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(root) / "reference" / "paper_subset_quality.csv")


def load_selected_reference(root: str | Path = ".") -> pd.DataFrame:
    return pd.read_csv(Path(root) / "reference" / "paper_selected_features_k10.csv")
