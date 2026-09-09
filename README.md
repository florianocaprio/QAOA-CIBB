# QAOA-CIBB Quantum Risk Lab

Applicazione scientifica riproducibile a supporto dell’articolo **Quantum-Assisted Clinical Risk Factor Selection for Diabetes Readmission Prediction**. Il motore seleziona colonne cliniche con un obiettivo QUBO, esegue realmente QAOA tramite Qiskit Aer, confronta metodi classici e valuta i subset mediante regressione logistica.

> Stato: prima release sperimentale. Il software produce nuovi risultati calcolati e non promette di ricostruire esattamente i numeri del manoscritto. Non è un dispositivo medico e non è validato per decisioni cliniche.

## Che cosa calcola

Il target è `y=1` per `readmitted == "<30"` e `y=0` per `">30"` o `"NO"`. `patient_nbr` determina i gruppi della cross-validation e non entra mai nei predittori; `encounter_id` è conservato per la tracciabilità.

Per ogni training fold il motore:

1. imputa e codifica i dati senza osservare il test;
2. raggruppa `diag_1`, `diag_2` e `diag_3` con una mappatura ICD-9 versionata;
3. costruisce un pool esplorativo di `N` colonne codificate mediante mutual information;
4. calcola rilevanza MI min-max e ridondanza come correlazione assoluta di Pearson;
5. costruisce il QUBO e la conversione Ising;
6. seleziona subset con MI, L1-ranked-top-k, simulated annealing, random-k e QAOA;
7. addestra una regressione logistica L2 sulle feature selezionate;
8. salva predizioni out-of-fold, metriche, manifest, QUBO, Ising, counts e log QAOA.

L’obiettivo è:

```text
E(x) = -alpha * sum_i(r_i*x_i)
       + beta * sum_{i<j}(c_ij*x_i*x_j)
       + gamma * (sum_i(x_i)-k)^2
```

con default confermati `alpha=1.0`, `beta=0.5`, `gamma=2.0`. L’energia è un criterio di selezione, non una probabilità clinica. QAOA seleziona le variabili; la regressione logistica stima il rischio.

## Installazione riproducibile

Sono supportati Python 3.11 e 3.12. Le versioni effettivamente verificate il 9 settembre 2026 sono bloccate in `requirements.lock`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.lock
pip install -e . --no-deps
```

Versioni centrali verificate: Qiskit 2.5.2, Qiskit Aer 0.17.2, scikit-learn 1.9.0, NumPy 2.4.6, pandas 2.3.3, SciPy 1.17.1, D-Wave samplers 1.8.0 e Streamlit 1.63.0.

## Comandi

```bash
# Controllo preventivo di memoria, qubit e costo
quantum-risk preflight configs/quick.yaml

# Smoke test reale ridotto sul dataset UCI
quantum-risk run configs/smoke.yaml

# Benchmark rapido previsto dalla specifica
quantum-risk run configs/quick.yaml

# Processo separato, stato e cancellazione
quantum-risk launch configs/quick.yaml
quantum-risk status ID_JOB
quantum-risk cancel ID_JOB

# Dashboard italiana
streamlit run app.py

# Test e qualità
ruff check src app.py tests
pytest -q
```

## Preset

- `smoke`: N=6, k=3, due fold e budget minimo; verifica l’intera catena.
- `quick`: N=12, k=5, p=1, tre fold, massimo 2.000 pazienti campionati deterministicamente con tutti i relativi ricoveri.
- `benchmark`: cinque fold, k=5/10/15, p=1/2 e simulazione MPS; è oneroso e non garantisce scalabilità per QUBO densi.
- `paper_reference`: carica N=40 e i riferimenti storici, ma non esegue QAOA senza il manifest originale.

N è il numero di variabili binarie e coincide con i qubit usati nella codifica diretta. Selezionare k=5 da N=40 richiede comunque 40 qubit. Uno statevector N=40 richiederebbe 16 TiB per il solo stato complesso in doppia precisione ed è bloccato dal preflight.

## Output di un run

Ogni cartella sotto `runs/` contiene configurazione, stato atomico, hash del dataset e degli split, versioni software, manifest per fold, MI, matrice di ridondanza, QUBO/Ising, campioni SA, counts e traccia QAOA, predizioni, metriche per fold, riepilogo media/deviazione standard, metriche OOF aggregate, tabella LaTeX e grafici PNG/SVG. `runs/` è ignorata da Git per evitare di committare output locali pesanti.

I valori sono separati tramite `source`:

- `computed_classical`;
- `computed_qaoa_simulation`;
- `external_qpu` per import hardware validato;
- `paper_reference` per valori trascritti dal manoscritto.

Una simulazione Aer, anche rumorosa, non è un’esecuzione IQM. Un import hardware richiede backend, run ID, N, k, p, shot, counts, ordine delle feature e hash di QUBO, preprocessing e split.

## Dashboard

La dashboard contiene cinque schede: Esperimento, Selezione, QAOA, Risultati ed Esportazioni. Prima di un run non mostra metriche dimostrative. I riferimenti storici sono disattivati per default e visivamente identificati come tali. Le esecuzioni partono in un processo separato dal rendering Streamlit e possono essere interrotte.

## Dataset e documenti

Il dataset raw ufficiale UCI 296 è conservato senza modifiche sotto `data/raw/`. Provenienza, DOI, licenza CC BY 4.0, checksum e limiti sono descritti in [`data/README.md`](data/README.md).

- [`docs/paper/Quantum_Risk.pdf`](docs/paper/Quantum_Risk.pdf): manoscritto di lavoro.
- [`docs/paper/Quantum_Risk.docx`](docs/paper/Quantum_Risk.docx): sorgente editabile, non modificato dal software.
- [`docs/template/CIBB_Template_6884.pdf`](docs/template/CIBB_Template_6884.pdf): template editoriale CIBB, non fonte dei risultati.
- [`docs/reproducibility_status.md`](docs/reproducibility_status.md): fatti confermati, default e informazioni mancanti.
- [`docs/reproducibility-plan.md`](docs/reproducibility-plan.md): criteri di verifica.

## Limiti scientifici

Il manifest originale delle 40 feature, gli split, i seed, gli shot, l’ottimizzatore e i raw counts hardware non sono disponibili. La modalità automatica seleziona colonne one-hot, non gruppi di variabili cliniche. I risultati sono quindi una ricostruzione esplorativa controllata, non una replica certificata.

Il dataset riguarda ospedali statunitensi nel periodo 1999–2008, contiene attributi demografici sensibili ed è sbilanciato. Associazione predittiva non implica causalità, un’AUC leggermente maggiore non prova superiorità e nessun subset è automaticamente utile in clinica.

## Fonti tecniche

Fonti ufficiali consultate il 9 settembre 2026:

- [UCI dataset 296](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008)
- [Qiskit qaoa_ansatz](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.qaoa_ansatz)
- [Qiskit bit ordering](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering)
- [Qiskit AerSimulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html)
- [scikit-learn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [D-Wave SimulatedAnnealingSampler](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html)

Le fonti descrivono algoritmi e API; non certificano i risultati prodotti da questo repository.
