# Rapporto di validazione locale

Data della verifica: 9 settembre 2026. Ambiente: macOS arm64, Python 3.11.4, dipendenze bloccate in `requirements.lock`.

## Verifiche automatizzate

Il comando `pytest -q` ha completato 33 test. Le verifiche coprono target e marcatori mancanti, mappatura ICD-9, categorie test non viste, separazione dei pazienti, normalizzazione MI, metriche note, contratto N/k/qubit, protezione RAM, import QPU validato, assenza di fallback QPU silenzioso, cancellazione job, ordine dei bit, counts, circuito Aer, confronto con `qaoa_ansatz`, confronto MPS/statevector, rendering della dashboard, un end-to-end UCI reale ridotto ed equivalenza matematica completa.

La misura `pytest --cov=src/quantum_risk --cov-report=term-missing` ha rilevato una copertura complessiva dell’84% sul pacchetto scientifico.

Per N=7 sono state enumerate tutte le 128 bitstring e sono risultate equivalenti, entro tolleranza assoluta `1e-10`, le seguenti rappresentazioni: formula diretta, QUBO triangolare, matrice simmetrica e Hamiltoniana Ising. Il circuito QAOA esplicito coincide con l’ansatz Qiskit indipendente con fidelity 1 entro `1e-12` sul caso testato.

`ruff check src app.py tests` è terminato senza errori. La dashboard Streamlit è stata avviata sulla porta locale 8510 e l’endpoint `/_stcore/health` ha risposto HTTP 200; il processo è stato quindi arrestato.

## End-to-end smoke

Il preset `configs/smoke.yaml` ha eseguito due fold su un campione deterministico di 300 pazienti, N=6, k=3, p=1, includendo MI, L1-ranked-top-k, SA-QUBO, random-k, QAOA Aer e regressione logistica. Due run indipendenti con seed 42 hanno prodotto `summary_metrics.csv` e `predictions.csv` byte-per-byte identici.

## End-to-end quick

Run locale: `20260909T194133Z-8b1f59ea64`. Il preset ha usato 2.000 pazienti e tutti i relativi 2.893 ricoveri, con prevalenza positiva 10,58%, N=12, k=5, p=1 e tre fold stratificati per paziente. Durata osservata: 3,17 secondi su questa macchina.

| Metodo | Fonte | AUC media ± DS | Recall media ± DS | F1 media ± DS | Balanced accuracy media ± DS |
|---|---|---:|---:|---:|---:|
| Tutte le candidate | computed_classical | 0,6665 ± 0,0486 | 0,4703 ± 0,0905 | 0,2768 ± 0,0602 | 0,6199 ± 0,0567 |
| MI top-k | computed_classical | 0,6411 ± 0,0726 | 0,4015 ± 0,0933 | 0,2626 ± 0,0770 | 0,5999 ± 0,0653 |
| L1-ranked-top-k | computed_classical | 0,6560 ± 0,0494 | 0,5253 ± 0,1830 | 0,2470 ± 0,0369 | 0,6030 ± 0,0445 |
| SA-QUBO | computed_classical | 0,6463 ± 0,0726 | 0,4081 ± 0,0913 | 0,2663 ± 0,0772 | 0,6030 ± 0,0655 |
| QAOA simulation | computed_qaoa_simulation | 0,6411 ± 0,0726 | 0,4015 ± 0,0933 | 0,2626 ± 0,0770 | 0,5999 ± 0,0653 |
| Random-k | computed_classical | 0,6227 ± 0,0613 | 0,5852 ± 0,2220 | 0,2236 ± 0,0316 | 0,5755 ± 0,0448 |

La frazione di shot QAOA con cardinalità esatta k è stata 0,3354, 0,2651 e 0,0942 nei tre fold. Tutti i run QAOA hanno trovato almeno un campione ammissibile. La profondità transpiled registrata è 25 nel preset quick.

Questi valori dimostrano il funzionamento della pipeline, non superiorità quantistica né validità clinica. Il campione è ridotto, il pool è ricostruito automaticamente a livello di colonne codificate e il protocollo non coincide con quello originale, che non è disponibile in dettaglio.
