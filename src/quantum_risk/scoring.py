from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_classif


@dataclass(frozen=True)
class CandidatePool:
    names: list[str]
    source_variables: list[str]
    train: np.ndarray
    test: np.ndarray
    discrete_mask: np.ndarray
    mi_raw: np.ndarray
    relevance: np.ndarray
    redundancy: np.ndarray
    warning: str | None = None


def normalize_minmax(values: np.ndarray, tolerance: float = 1e-12) -> tuple[np.ndarray, str | None]:
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    if span <= tolerance:
        return np.zeros_like(
            values
        ), "all mutual-information scores are equal; relevance set to zero"
    return (values - values.min()) / span, None


def build_candidate_pool(
    train: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
    names: list[str],
    source_variables: list[str],
    discrete_mask: np.ndarray,
    n_candidates: int,
    seed: int,
    n_neighbors: int = 3,
) -> CandidatePool:
    if train.shape[1] < n_candidates:
        raise ValueError(
            f"only {train.shape[1]} encoded candidates available, requested N={n_candidates}"
        )
    mi_all = mutual_info_classif(
        train,
        y_train,
        discrete_features=discrete_mask,
        n_neighbors=n_neighbors,
        random_state=seed,
    )
    # Primary key: descending MI. Secondary key: stable encoded name.
    order = np.lexsort((np.asarray(names), -mi_all))[:n_candidates]
    selected_train = train[:, order]
    selected_test = test[:, order]
    selected_mi = mi_all[order]
    relevance, warning = normalize_minmax(selected_mi)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(selected_train, rowvar=False)
    redundancy = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(redundancy, 0.0)
    return CandidatePool(
        names=[names[i] for i in order],
        source_variables=[source_variables[i] for i in order],
        train=selected_train,
        test=selected_test,
        discrete_mask=discrete_mask[order],
        mi_raw=selected_mi,
        relevance=relevance,
        redundancy=redundancy,
        warning=warning,
    )
