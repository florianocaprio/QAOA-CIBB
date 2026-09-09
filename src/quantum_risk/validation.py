from __future__ import annotations

import importlib.metadata
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from quantum_risk.artifacts import atomic_json, stable_hash, write_csv
from quantum_risk.classical import (
    l1_ranked_top_k,
    random_k,
    simulated_annealing_select,
    top_k_bits,
)
from quantum_risk.config import ExperimentConfig, save_config
from quantum_risk.data import deterministic_patient_subsample, load_uci_csv
from quantum_risk.export import export_summary
from quantum_risk.ising import qubo_to_ising
from quantum_risk.metrics import (
    average_pairwise_redundancy,
    classification_metrics,
    fit_evaluate_logistic,
    mean_pairwise_jaccard,
)
from quantum_risk.preprocessing import ICD9_MAPPING_VERSION, ClinicalPreprocessor
from quantum_risk.qaoa import run_qaoa
from quantum_risk.qubo import build_qubo
from quantum_risk.resources import preflight_quantum_run
from quantum_risk.scoring import build_candidate_pool

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "qiskit",
    "qiskit-aer",
    "dimod",
    "dwave-samplers",
    "streamlit",
    "plotly",
    "pydantic",
    "PyYAML",
    "psutil",
)


def _git_revision() -> dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _versions() -> dict[str, str]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _selected_names(bits: np.ndarray, names: list[str]) -> list[str]:
    return [name for name, selected in zip(names, bits, strict=True) if selected]


def _evaluate(
    *,
    method: str,
    source: str,
    bits: np.ndarray,
    pool,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_indices: np.ndarray,
    encounter_ids: np.ndarray,
    fold: int,
    k: int,
    p: int | None,
    seed: int,
    class_weight: str | None,
    extra: dict[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    prediction = fit_evaluate_logistic(
        pool.train, pool.test, y_train, y_test, bits, seed, class_weight
    )
    row: dict[str, object] = {
        "source": source,
        "method": method,
        "fold": fold,
        "N": len(pool.names),
        "k": k,
        "k_actual": int(bits.sum()),
        "p": p,
        "selected_features": _selected_names(bits, pool.names),
        "redundancy": average_pairwise_redundancy(bits, pool.redundancy),
        **prediction.metrics,
    }
    if extra:
        row.update(extra)
    predictions = [
        {
            "source": source,
            "method": method,
            "fold": fold,
            "N": len(pool.names),
            "k": k,
            "p": p,
            "row_index": int(row_index),
            "encounter_id": str(encounter),
            "y_true": int(truth),
            "probability": float(probability),
            "prediction": int(label),
        }
        for row_index, encounter, truth, probability, label in zip(
            test_indices,
            encounter_ids,
            y_test,
            prediction.probabilities,
            prediction.predictions,
            strict=True,
        )
    ]
    return row, predictions


def _aggregate(
    fold_rows: list[dict[str, object]], prediction_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    metrics = (
        "auc",
        "recall_30day",
        "precision",
        "f1",
        "specificity",
        "balanced_accuracy",
        "redundancy",
    )
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in fold_rows:
        if row.get("status", "completed") == "completed":
            groups[(row["source"], row["method"], row["N"], row["k"], row["p"])].append(row)
    prediction_groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        prediction_groups[(row["source"], row["method"], row["N"], row["k"], row["p"])].append(row)
    output = []
    for key, rows in groups.items():
        source, method, n_value, k_value, p_value = key
        summary: dict[str, object] = {
            "source": source,
            "method": method,
            "N": n_value,
            "k": k_value,
            "p": p_value,
            "folds_completed": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        subsets = [set(row["selected_features"]) for row in rows]
        summary["stability_jaccard"] = mean_pairwise_jaccard(subsets)
        predictions = prediction_groups[key]
        if predictions:
            oof, _ = classification_metrics(
                np.asarray([row["y_true"] for row in predictions]),
                np.asarray([row["probability"] for row in predictions]),
            )
            summary.update({f"oof_{name}": value for name, value in oof.items()})
        output.append(summary)
    return sorted(output, key=lambda row: (str(row["method"]), int(row["k"]), int(row["p"] or 0)))


def run_experiment(config: ExperimentConfig) -> Path:
    started = datetime.now(UTC)
    bundle = load_uci_csv(config.dataset_path)
    if config.max_patients:
        bundle = deterministic_patient_subsample(bundle, config.max_patients, config.seed)
    identity = {
        "config": config.model_dump(mode="json"),
        "dataset_sha256": bundle.metadata["sha256"],
        "started_at_utc": started.isoformat(),
    }
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{stable_hash(identity)[:10]}"
    run_dir = config.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_config(config, run_dir / "config.yaml")
    atomic_json(run_dir / "status.json", {"status": "running", "run_id": run_id})
    atomic_json(run_dir / "dataset_metadata.json", bundle.metadata)
    atomic_json(run_dir / "environment.json", {"git": _git_revision(), "packages": _versions()})

    try:
        splitter = StratifiedGroupKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed + 1
        )
        fold_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        fold_artifacts: list[dict[str, object]] = []
        split_iterator = splitter.split(bundle.frame, bundle.y, groups=bundle.groups)
        for fold, (train_idx, test_idx) in enumerate(split_iterator):
            train_groups = set(bundle.groups.iloc[train_idx])
            test_groups = set(bundle.groups.iloc[test_idx])
            if train_groups & test_groups:
                raise RuntimeError("patient leakage detected between train and test")
            processor = ClinicalPreprocessor()
            encoded = processor.fit_transform_pair(
                bundle.frame.iloc[train_idx], bundle.frame.iloc[test_idx]
            )
            pool = build_candidate_pool(
                encoded.train,
                encoded.test,
                bundle.y.iloc[train_idx].to_numpy(),
                encoded.names,
                encoded.source_variables,
                encoded.discrete_mask,
                config.n_candidates,
                config.seed + 101 + fold,
                config.mi_n_neighbors,
            )
            fold_detail: dict[str, object] = {
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_patients": len(train_groups),
                "test_patients": len(test_groups),
                "patient_overlap": 0,
                "split_hash": stable_hash(
                    {"train_indices": train_idx.tolist(), "test_indices": test_idx.tolist()}
                ),
                "icd9_mapping_version": ICD9_MAPPING_VERSION,
                "excluded": encoded.excluded,
                "candidate_manifest": [
                    {
                        "qubit": i,
                        "encoded_feature": name,
                        "source_variable": source,
                        "discrete": bool(discrete),
                        "mi_raw": float(mi),
                        "relevance": float(relevance),
                    }
                    for i, (name, source, discrete, mi, relevance) in enumerate(
                        zip(
                            pool.names,
                            pool.source_variables,
                            pool.discrete_mask,
                            pool.mi_raw,
                            pool.relevance,
                            strict=True,
                        )
                    )
                ],
                "redundancy": pool.redundancy,
                "warning": pool.warning,
                "qubo_by_k": {},
                "qaoa": [],
                "sa": [],
            }
            y_train = bundle.y.iloc[train_idx].to_numpy()
            y_test = bundle.y.iloc[test_idx].to_numpy()
            encounters = bundle.encounter_ids.iloc[test_idx].to_numpy()

            all_bits = np.ones(config.n_candidates, dtype=np.int8)
            row, predictions = _evaluate(
                method="All candidates",
                source="computed_classical",
                bits=all_bits,
                pool=pool,
                y_train=y_train,
                y_test=y_test,
                test_indices=test_idx,
                encounter_ids=encounters,
                fold=fold,
                k=config.n_candidates,
                p=None,
                seed=config.seed + 200 + fold,
                class_weight=config.class_weight,
            )
            fold_rows.append(row)
            prediction_rows.extend(predictions)

            for k_value in config.k_values:
                qubo = build_qubo(
                    pool.relevance,
                    pool.redundancy,
                    k_value,
                    config.alpha_relevance,
                    config.beta_redundancy,
                    config.gamma_cardinality,
                )
                ising = qubo_to_ising(qubo)
                fold_detail["qubo_by_k"][str(k_value)] = {
                    "linear": qubo.linear,
                    "quadratic_upper": qubo.quadratic,
                    "offset": qubo.offset,
                    "ising_h": ising.h,
                    "ising_j_upper": ising.j,
                    "ising_offset_identity": ising.offset_identity,
                    "qubo_hash": stable_hash(
                        {
                            "linear": qubo.linear,
                            "quadratic": qubo.quadratic,
                            "offset": qubo.offset,
                        }
                    ),
                }
                mi_bits = top_k_bits(pool.mi_raw, pool.names, k_value)
                row, predictions = _evaluate(
                    method="MI top-k",
                    source="computed_classical",
                    bits=mi_bits,
                    pool=pool,
                    y_train=y_train,
                    y_test=y_test,
                    test_indices=test_idx,
                    encounter_ids=encounters,
                    fold=fold,
                    k=k_value,
                    p=None,
                    seed=config.seed + 300 + fold,
                    class_weight=config.class_weight,
                )
                fold_rows.append(row)
                prediction_rows.extend(predictions)

                l1_bits, selected_c, l1_status = l1_ranked_top_k(
                    pool.train,
                    y_train,
                    pool.names,
                    k_value,
                    config.seed + 400 + fold,
                    config.class_weight,
                )
                if l1_bits is not None:
                    row, predictions = _evaluate(
                        method="L1-ranked-top-k",
                        source="computed_classical",
                        bits=l1_bits,
                        pool=pool,
                        y_train=y_train,
                        y_test=y_test,
                        test_indices=test_idx,
                        encounter_ids=encounters,
                        fold=fold,
                        k=k_value,
                        p=None,
                        seed=config.seed + 401 + fold,
                        class_weight=config.class_weight,
                        extra={"l1_C": selected_c, "status": l1_status},
                    )
                    fold_rows.append(row)
                    prediction_rows.extend(predictions)
                else:
                    fold_rows.append(
                        {
                            "source": "computed_classical",
                            "method": "L1-ranked-top-k",
                            "fold": fold,
                            "N": config.n_candidates,
                            "k": k_value,
                            "p": None,
                            "status": l1_status,
                        }
                    )

                sa_bits, sa_raw, sa_status = simulated_annealing_select(
                    qubo,
                    config.sa_num_reads,
                    config.sa_num_sweeps,
                    config.seed + 500 + fold + k_value,
                )
                fold_detail["sa"].append({"k": k_value, "status": sa_status, "samples": sa_raw})
                if sa_bits is not None:
                    row, predictions = _evaluate(
                        method="SA-QUBO",
                        source="computed_classical",
                        bits=sa_bits,
                        pool=pool,
                        y_train=y_train,
                        y_test=y_test,
                        test_indices=test_idx,
                        encounter_ids=encounters,
                        fold=fold,
                        k=k_value,
                        p=None,
                        seed=config.seed + 501 + fold,
                        class_weight=config.class_weight,
                        extra={
                            "status": sa_status,
                            "qubo_energy": qubo.energy(sa_bits),
                            **qubo.contributions(sa_bits),
                        },
                    )
                    fold_rows.append(row)
                    prediction_rows.extend(predictions)

                random_bits = random_k(
                    config.n_candidates, k_value, config.seed + 600 + fold + k_value
                )
                row, predictions = _evaluate(
                    method="Random-k control",
                    source="computed_classical",
                    bits=random_bits,
                    pool=pool,
                    y_train=y_train,
                    y_test=y_test,
                    test_indices=test_idx,
                    encounter_ids=encounters,
                    fold=fold,
                    k=k_value,
                    p=None,
                    seed=config.seed + 601 + fold,
                    class_weight=config.class_weight,
                )
                fold_rows.append(row)
                prediction_rows.extend(predictions)

                if config.run_qaoa:
                    estimate = preflight_quantum_run(
                        config.n_candidates,
                        config.q_budget,
                        config.simulator,
                        config.statevector_memory_fraction,
                    )
                    for p_value in config.p_values:
                        qaoa_result = run_qaoa(
                            qubo,
                            ising,
                            p_value,
                            config.simulator,
                            config.qaoa_restarts,
                            config.optimization_shots,
                            config.final_shots,
                            config.max_objective_evaluations,
                            config.seed + 700 + fold * 100 + k_value * 10 + p_value,
                            max(1, estimate.allowed_bytes // 2**20),
                            config.mps_max_bond_dimension,
                            config.mps_truncation_threshold,
                        )
                        qaoa_payload = {"k": k_value, "p": p_value, **asdict(qaoa_result)}
                        fold_detail["qaoa"].append(qaoa_payload)
                        if qaoa_result.bits is not None:
                            row, predictions = _evaluate(
                                method="QAOA simulation",
                                source="computed_qaoa_simulation",
                                bits=qaoa_result.bits,
                                pool=pool,
                                y_train=y_train,
                                y_test=y_test,
                                test_indices=test_idx,
                                encounter_ids=encounters,
                                fold=fold,
                                k=k_value,
                                p=p_value,
                                seed=config.seed + 701 + fold,
                                class_weight=config.class_weight,
                                extra={
                                    "status": qaoa_result.status,
                                    "qubo_energy": qaoa_result.energy,
                                    "feasible_fraction": qaoa_result.feasible_fraction,
                                    **qubo.contributions(qaoa_result.bits),
                                },
                            )
                            fold_rows.append(row)
                            prediction_rows.extend(predictions)
                        else:
                            fold_rows.append(
                                {
                                    "source": "computed_qaoa_simulation",
                                    "method": "QAOA simulation",
                                    "fold": fold,
                                    "N": config.n_candidates,
                                    "k": k_value,
                                    "p": p_value,
                                    "status": qaoa_result.status,
                                    "feasible_fraction": qaoa_result.feasible_fraction,
                                }
                            )
            fold_artifacts.append(fold_detail)
            atomic_json(run_dir / f"fold_{fold}.json", fold_detail)

        aggregate_rows = _aggregate(fold_rows, prediction_rows)
        write_csv(run_dir / "fold_metrics.csv", fold_rows)
        write_csv(run_dir / "predictions.csv", prediction_rows)
        write_csv(run_dir / "summary_metrics.csv", aggregate_rows)
        atomic_json(run_dir / "fold_metrics.json", fold_rows)
        atomic_json(run_dir / "summary_metrics.json", aggregate_rows)
        export_summary(run_dir, aggregate_rows)
        completed = datetime.now(UTC)
        atomic_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "run_id": run_id,
                "started_at_utc": started.isoformat(),
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - started).total_seconds(),
            },
        )
        return run_dir
    except Exception as error:
        atomic_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "run_id": run_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
