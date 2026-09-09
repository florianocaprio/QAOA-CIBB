from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class PredictionResult:
    metrics: dict[str, float | int | None]
    probabilities: np.ndarray
    predictions: np.ndarray


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict[str, float | int | None], np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y_true, probabilities)) if np.unique(y_true).size == 2 else None
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    metrics: dict[str, float | int | None] = {
        "auc": auc,
        "recall_30day": float(recall_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, predicted


def fit_evaluate_logistic(
    train: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_bits: np.ndarray,
    seed: int,
    class_weight: str | None = "balanced",
) -> PredictionResult:
    columns = np.flatnonzero(selected_bits)
    if columns.size == 0:
        raise ValueError("at least one feature must be selected")
    model = LogisticRegression(
        l1_ratio=0.0,
        C=1.0,
        class_weight=class_weight,
        max_iter=2000,
        random_state=seed,
    )
    model.fit(train[:, columns], y_train)
    probabilities = model.predict_proba(test[:, columns])[:, 1]
    metrics, predictions = classification_metrics(y_test, probabilities)
    return PredictionResult(metrics, probabilities, predictions)


def average_pairwise_redundancy(bits: np.ndarray, redundancy: np.ndarray) -> float | None:
    selected = np.flatnonzero(bits)
    if len(selected) < 2:
        return None
    values = [redundancy[i, j] for index, i in enumerate(selected) for j in selected[index + 1 :]]
    return float(np.mean(values))


def mean_pairwise_jaccard(subsets: list[set[str]]) -> float | None:
    if len(subsets) < 2:
        return None
    values = []
    for index, left in enumerate(subsets):
        for right in subsets[index + 1 :]:
            union = left | right
            values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values))
