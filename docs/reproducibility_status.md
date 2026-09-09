# Stato di riproducibilità

## Confermato dal paper

- Dataset UCI Diabetes 130-US Hospitals, ID 296.
- Target: riammissione `<30` contro `>30` e `NO`.
- Formulazione QUBO con rilevanza, ridondanza e penalità di cardinalità.
- Pool dichiarato `N=40`, target `k=5,10,15` e classificatore logistico.
- Confronti con tutte le candidate, mutual information, LASSO e simulated annealing.
- Tabelle numeriche conservate esclusivamente sotto `reference/` con `source=paper_reference`.
- Il manoscritto afferma compatibilità/esecuzione hardware a 54 qubit, ma non fornisce raw counts, run ID o manifest completo.

## Confermato dall'autore

- `alpha=1.0`, `beta=0.5`, `gamma=2.0`.
- Rilevanza: mutual information feature-target normalizzata min-max.
- Ridondanza: valore assoluto della correlazione di Pearson sulle colonne preprocessate.
- Profondità investigate: `p=1` e `p=2`; le etichette shallow e low-depth sono convenzioni del progetto.
- Le righe aggregate storiche non sono attribuibili a una specifica `p`, quindi `p=null`.

## Default implementativo del nuovo motore

- `StratifiedGroupKFold`, seed radice 42, raggruppamento per `patient_nbr`.
- Imputazione fitted sul training, one-hot encoding, scaling delle numeriche.
- Pool automatico di colonne codificate ordinato per MI del training con pareggio per nome.
- Logistica L2, `C=1`, `class_weight=balanced`, soglia 0.5.
- L1-ranked-top-k con griglia di `C` dichiarata; SA con 1000 reads e 1000 sweeps.
- COBYLA, inizializzazioni casuali registrate, QAOA esplicito su Aer.
- Stabilità: Jaccard medio tra fold su nomi canonici.
- Raggruppamento ICD-9 implementato come ricostruzione versionata, non come ricetta certificata degli esperimenti originali.

## Informazione originale mancante

- Manifest ordinato e ricette esatte delle 40 candidate.
- Coorte ed esclusioni definitive, split e seed originali.
- Shot, ottimizzatore, inizializzazioni e dettagli del classificatore originale.
- Definizione completa dello stability index storico.
- Backend QPU, calibrazione, transpilation, layout, counts e identificativo dei run.

Di conseguenza il motore produce una ricostruzione esplorativa riproducibile, non una replica certificata dei numeri storici. Il PDF sorgente effettivo del manoscritto è `docs/paper/Quantum_Risk.pdf`; `docs/template/CIBB_Template_6884.pdf` è un template editoriale.

