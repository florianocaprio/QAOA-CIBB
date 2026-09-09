# QAOA-CIBB

Research repository for **Quantum-Assisted Clinical Risk Factor Selection for Diabetes Readmission Prediction**.

The project will provide a reproducible software implementation of the manuscript's QUBO/QAOA feature-selection framework for predicting hospital readmission within 30 days on the UCI Diabetes 130-US Hospitals dataset.

> Status: repository scaffold. The manuscript is a working draft and its numerical results have not yet been reproduced by code in this repository.

## Research question

Can a QUBO objective balancing feature relevance, pairwise redundancy, and subset cardinality identify compact and interpretable clinical feature sets while retaining competitive predictive performance?

The intended objective is

```text
min_x  -alpha * sum_i(r_i x_i)
       + beta * sum_{i<j}(c_ij x_i x_j)
       + gamma * (sum_i(x_i) - k)^2,
```

where `x_i` indicates whether feature `i` is selected, `r_i` is its relevance to early readmission, `c_ij` measures pairwise redundancy, and `k` is the target subset size.

## Repository contents

```text
QAOA-CIBB/
├── configs/              experiment configurations (to be implemented)
├── data/
│   ├── README.md         provenance, license, target and handling notes
│   └── raw/              unmodified official UCI files
├── docs/
│   ├── paper/            manuscript source and PDF export
│   ├── template/         supplied CIBB publication template
│   └── reproducibility-plan.md
├── notebooks/            exploratory analyses (to be implemented)
├── results/
│   ├── figures/          generated figures only
│   └── tables/           generated tables only
├── scripts/              command-line entry points (to be implemented)
├── src/qaoa_cibb/        reusable implementation (to be implemented)
└── tests/                automated tests (to be implemented)
```

## Primary endpoint

The original `readmitted` field has three values. The planned binary target is:

- positive: `<30`;
- negative: `>30` or `NO`.

Any feature selection, preprocessing, imputation, encoding and scaling must be fitted only on the training portion of each split to prevent leakage. Patient-level splitting should be evaluated because the same `patient_nbr` can occur in more than one encounter.

## Planned experimental workflow

1. Validate the raw data against the recorded checksums.
2. Audit missing values, repeated patients, outcome imbalance and excluded variables.
3. Define a leakage-safe candidate pool of 40 clinically interpretable variables.
4. Fit preprocessing and relevance/redundancy estimators inside each training split.
5. Build the QUBO and verify its energy against the mathematical objective.
6. Solve the same instances with QAOA and classical baselines.
7. Train downstream classifiers on each selected subset.
8. Report AUC, balanced accuracy, F1, precision, recall, specificity, redundancy, compactness and selection stability.
9. Record seeds, split identifiers, package versions, solver settings and QPU metadata.

Planned feature-selection baselines are all candidate features, mutual-information top-k, LASSO-based selection and simulated annealing on the same QUBO. Planned target sizes are `k = 5`, `10` and `15`.

## Documents

- [`docs/paper/Quantum_Risk.pdf`](docs/paper/Quantum_Risk.pdf): PDF export of the supplied manuscript.
- [`docs/paper/Quantum_Risk.docx`](docs/paper/Quantum_Risk.docx): supplied editable manuscript.
- [`docs/template/CIBB_Template_6884.pdf`](docs/template/CIBB_Template_6884.pdf): supplied CIBB template/reference.

## Dataset

The raw dataset is included under `data/raw/` exactly as distributed in the official UCI archive. See [`data/README.md`](data/README.md) for provenance, attribution, license, checksums and responsible-use notes.

Official source: [UCI Machine Learning Repository, dataset 296](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008), DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J).

## Reproducibility status

No implementation or executable environment has been selected yet. Dependency files and run commands will be added with the software implementation, rather than prematurely fixing a stack. The expected controls and acceptance criteria are documented in [`docs/reproducibility-plan.md`](docs/reproducibility-plan.md).

## Licensing and citation

The UCI dataset is distributed under CC BY 4.0 and retains its own attribution requirements. The manuscript and CIBB template are included as research inputs; no separate reuse license is asserted for them. A software license will be added when the implementation is introduced and the copyright holders are confirmed.

