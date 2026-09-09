# Dataset provenance and use

## Identification

- Name: Diabetes 130-US Hospitals for Years 1999-2008
- UCI dataset ID: 296
- DOI: https://doi.org/10.24432/C5230J
- Official page: https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
- Downloaded: 2026-09-09
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Creators credited by UCI: John Clore, Krzysztof Cios, Jon DeShazo and Beata Strack
- Introductory paper: Strack et al., *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*, 2014

Suggested dataset citation:

> Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

## Raw files

`raw/diabetic_data.csv` contains 101,766 encounter rows plus the header. `raw/IDS_mapping.csv` contains lookup values for coded admission, discharge and source fields. Both files were extracted without modification from the official UCI ZIP archive.

SHA-256 checksums:

```text
0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97  raw/diabetic_data.csv
f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e  raw/IDS_mapping.csv
```

Official ZIP SHA-256 at download time:

```text
f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a
```

## Target definition

The planned binary early-readmission target maps `readmitted == "<30"` to 1 and both `">30"` and `"NO"` to 0. The raw target remains unchanged in this directory.

## Data governance

UCI flags the dataset as containing potentially sensitive demographic attributes, including age, gender and race. Although public and de-identified, it should be handled as clinical research data:

- do not attempt re-identification;
- do not present the resulting model as a clinical device or as validated for patient care;
- report subgroup performance and selection stability where sample sizes permit;
- document exclusions, unknown values and discharge categories carefully;
- keep raw files immutable and write derived data outside `data/raw/`;
- avoid encounter-level leakage from repeated patients and post-outcome variables.

The dataset is historically and geographically specific (US hospitals, 1999-2008), so external validity and current clinical applicability cannot be assumed.

