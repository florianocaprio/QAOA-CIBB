from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from quantum_risk.config import load_config, save_config
from quantum_risk.data import deterministic_patient_subsample, load_uci_csv
from quantum_risk.jobs import JobManager
from quantum_risk.paper_reference import load_predictive_reference, load_quality_reference
from quantum_risk.resources import preflight_quantum_run

st.set_page_config(page_title="Quantum Risk Lab", page_icon="⚛️", layout="wide")
st.title("Quantum Risk Lab")
st.caption("Selezione QUBO/QAOA di predittori associati alla riammissione entro 30 giorni")

preset = st.sidebar.selectbox("Modalità", ["quick", "benchmark", "paper_reference"])
config_path = Path("configs") / f"{preset}.yaml"
config = load_config(config_path)
upload = st.sidebar.file_uploader("Dataset CSV locale", type=["csv"])
dataset_path = config.dataset_path
if upload is not None:
    content = upload.getvalue()
    if len(content) > 50 * 1024 * 1024:
        st.sidebar.error("File oltre il limite di 50 MiB.")
        st.stop()
    upload_hash = hashlib.sha256(content).hexdigest()
    dataset_path = Path("runs/uploads") / f"{upload_hash}.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        dataset_path.write_bytes(content)
st.sidebar.markdown(f"**Dataset:** `{dataset_path}`")
q_budget = st.sidebar.number_input("Budget qubit", min_value=2, max_value=54, value=config.q_budget)
n_candidates = st.sidebar.number_input(
    "N candidate", min_value=2, max_value=int(q_budget), value=min(config.n_candidates, q_budget)
)
k_defaults = [k for k in config.k_values if k <= n_candidates] or [min(5, n_candidates)]
k_values = st.sidebar.multiselect(
    "k feature selezionate", options=list(range(1, int(n_candidates) + 1)), default=k_defaults
)
p_values = st.sidebar.multiselect("Profondità p", options=[1, 2, 3], default=config.p_values)
simulator = st.sidebar.selectbox(
    "Simulatore",
    ["statevector_ideal", "statevector_noisy", "mps"],
    index=["statevector_ideal", "statevector_noisy", "mps"].index(config.simulator),
)
with st.sidebar.expander("Avanzate"):
    optimization_shots = st.number_input(
        "Shot ottimizzazione", min_value=32, value=config.optimization_shots
    )
    final_shots = st.number_input("Shot finali", min_value=32, value=config.final_shots)
    restarts = st.number_input("Restart", min_value=1, value=config.qaoa_restarts)
    max_evaluations = st.number_input(
        "Valutazioni per restart", min_value=1, value=config.max_objective_evaluations
    )
    alpha = st.number_input("Alpha rilevanza", min_value=0.0, value=config.alpha_relevance)
    beta = st.number_input("Beta ridondanza", min_value=0.0, value=config.beta_redundancy)
    gamma = st.number_input("Gamma cardinalità", min_value=0.0, value=config.gamma_cardinality)
config = config.model_copy(
    update={
        "dataset_path": dataset_path,
        "q_budget": int(q_budget),
        "n_candidates": int(n_candidates),
        "k_values": [int(value) for value in k_values],
        "p_values": [int(value) for value in p_values],
        "simulator": simulator,
        "optimization_shots": int(optimization_shots),
        "final_shots": int(final_shots),
        "qaoa_restarts": int(restarts),
        "max_objective_evaluations": int(max_evaluations),
        "alpha_relevance": float(alpha),
        "beta_redundancy": float(beta),
        "gamma_cardinality": float(gamma),
    }
)
configuration_error = None
try:
    config = type(config).model_validate(config.model_dump())
except ValueError as error:
    configuration_error = str(error)
    st.sidebar.error(configuration_error)
st.sidebar.write(f"N = {config.n_candidates} candidate · qubit usati = {config.n_candidates}")
st.sidebar.write(f"k = {config.k_values} · p = {config.p_values}")
st.sidebar.caption("k conta le feature selezionate; p conta le coppie di livelli costo/mixer.")

manager = JobManager()
if "job_id" not in st.session_state:
    st.session_state.job_id = None

run_col, stop_col = st.sidebar.columns(2)
if run_col.button(
    "Esegui",
    type="primary",
    disabled=config.require_original_manifest or configuration_error is not None,
):
    runtime_config = Path("runs/ui-configs") / "latest.yaml"
    save_config(config, runtime_config)
    st.session_state.job_id = manager.launch(runtime_config)
if stop_col.button("Interrompi", disabled=not st.session_state.job_id):
    manager.cancel(st.session_state.job_id)

tabs = st.tabs(["Esperimento", "Selezione", "QAOA", "Risultati", "Esportazioni"])
with tabs[0]:
    bundle = load_uci_csv(config.dataset_path)
    if config.max_patients:
        bundle = deterministic_patient_subsample(bundle, config.max_patients, config.seed)
    estimate = preflight_quantum_run(
        config.n_candidates, config.q_budget, config.simulator, config.statevector_memory_fraction
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ricoveri", f"{len(bundle.frame):,}")
    c2.metric("Pazienti", f"{bundle.groups.nunique():,}")
    c3.metric("Prevalenza <30", f"{bundle.y.mean():.1%}")
    c4.metric("RAM statevector", f"{estimate.statevector_bytes / 2**20:.1f} MiB")
    if estimate.allowed and estimate.reason:
        st.warning(estimate.reason)
    elif estimate.allowed:
        st.info("Preflight superato.")
    else:
        st.error(estimate.reason)
    st.json(config.model_dump(mode="json"))
    if st.session_state.job_id:
        status = manager.status(st.session_state.job_id)
        st.write("Stato job", status)


def latest_completed_run() -> Path | None:
    runs = []
    for path in Path("runs").glob("*/status.json"):
        if path.parent.name == "jobs":
            continue
        try:
            if json.loads(path.read_text())["status"] == "completed":
                runs.append(path.parent)
        except (KeyError, json.JSONDecodeError):
            continue
    return max(runs, key=lambda path: path.name) if runs else None


latest = latest_completed_run()
with tabs[1]:
    st.subheader("Selezione delle colonne candidate")
    st.caption("Le feature sono predittori associati all’esito, non cause dimostrate.")
    if latest:
        fold = json.loads((latest / "fold_0.json").read_text())
        manifest = pd.DataFrame(fold["candidate_manifest"])
        st.dataframe(manifest, width="stretch")
        fig = px.bar(
            manifest,
            x="relevance",
            y="encoded_feature",
            orientation="h",
            title="Rilevanza MI normalizzata",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Non eseguito: avvia un esperimento per visualizzare valori calcolati.")

with tabs[2]:
    st.subheader("QAOA su simulatore")
    st.write(
        "L’energia QUBO non è una probabilità clinica e la frequenza di una "
        "bitstring non è il rischio di un paziente."
    )
    if latest:
        fold = json.loads((latest / "fold_0.json").read_text())
        qaoa_runs = fold.get("qaoa", [])
        if qaoa_runs:
            selected = qaoa_runs[0]
            st.json(
                {
                    key: selected.get(key)
                    for key in (
                        "status",
                        "k",
                        "p",
                        "bitstring",
                        "energy",
                        "feasible_fraction",
                        "depth",
                        "transpiled_depth",
                        "gate_counts",
                        "transpiled_gate_counts",
                        "backend_method",
                        "duration_seconds",
                    )
                }
            )
            trace = pd.DataFrame(selected["trace"])
            st.plotly_chart(
                px.line(trace, x="evaluation", y="energy", color="restart", markers=True),
                width="stretch",
            )
        else:
            st.warning("QAOA non eseguito per questo run.")
    else:
        st.warning("Non eseguito.")

with tabs[3]:
    st.subheader("Metriche calcolate")
    st.caption(
        "AUC misura discriminazione, non la percentuale di pazienti classificati "
        "correttamente. Differenze piccole non dimostrano superiorità clinica."
    )
    if latest:
        summary = pd.read_csv(latest / "summary_metrics.csv")
        st.dataframe(summary, width="stretch")
        if summary["auc_mean"].notna().any():
            st.plotly_chart(
                px.bar(
                    summary,
                    x="method",
                    y="auc_mean",
                    color="source",
                    facet_col="k",
                    error_y="auc_std",
                ),
                width="stretch",
            )
        if st.toggle("Mostra riferimento del paper"):
            st.markdown("**Riferimento storico, non risultato ricalcolato**")
            st.dataframe(load_predictive_reference(), width="stretch")
            st.dataframe(load_quality_reference(), width="stretch")
    else:
        st.warning("Nessuna metrica calcolata disponibile.")

with tabs[4]:
    st.subheader("Artefatti riproducibili")
    if latest:
        st.code(str(latest.resolve()))
        for filename in (
            "config.yaml",
            "summary_metrics.csv",
            "fold_metrics.csv",
            "predictions.csv",
            "environment.json",
        ):
            path = latest / filename
            if path.exists():
                st.download_button(f"Scarica {filename}", path.read_bytes(), file_name=filename)
    else:
        st.warning("Le esportazioni saranno disponibili dopo un run completato.")
