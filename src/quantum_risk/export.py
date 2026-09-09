from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qaoa-cibb-matplotlib"))

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def export_summary(run_dir: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(run_dir)
    frame = pd.DataFrame(rows)
    frame.to_latex(destination / "summary_metrics.tex", index=False, float_format="%.4f")
    chart = frame.dropna(subset=["auc_mean"]).copy()
    if chart.empty:
        return
    labels = [
        f"{method}\nk={k}" + (f", p={int(p)}" if pd.notna(p) else "")
        for method, k, p in zip(chart["method"], chart["k"], chart["p"], strict=True)
    ]
    colors = [
        "#1f77b4" if source == "computed_classical" else "#9467bd" for source in chart["source"]
    ]
    figure, axis = plt.subplots(figsize=(max(8, len(chart) * 1.1), 5.5))
    axis.bar(labels, chart["auc_mean"], yerr=chart["auc_std"].fillna(0), color=colors, capsize=3)
    axis.set_ylabel("Mean ROC AUC across folds")
    axis.set_title("Computed discrimination by selection method")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    figure.autofmt_xdate(rotation=35, ha="right")
    figure.tight_layout()
    figure.savefig(destination / "auc_by_method.png", dpi=200)
    figure.savefig(destination / "auc_by_method.svg")
    plt.close(figure)
