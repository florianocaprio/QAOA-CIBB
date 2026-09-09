from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IDENTIFIERS = {"encounter_id", "patient_nbr", "readmitted"}
NUMERIC_COLUMNS = {
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
}
DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")
MISSING_MARKERS = {"", "?"}
ICD9_MAPPING_VERSION = "strack-style-v1"


def map_icd9(value: object) -> str:
    raw = str(value).strip()
    if raw in MISSING_MARKERS:
        return "unknown"
    upper = raw.upper()
    if upper.startswith(("E", "V")):
        return "other"
    try:
        code = float(raw)
    except ValueError:
        return "other"
    if 250 <= code < 251:
        return "diabetes"
    if 390 <= code <= 459 or int(code) == 785:
        return "circulatory"
    if 460 <= code <= 519 or int(code) == 786:
        return "respiratory"
    if 520 <= code <= 579 or int(code) == 787:
        return "digestive"
    if 800 <= code <= 999:
        return "injury"
    if 710 <= code <= 739:
        return "musculoskeletal"
    if 580 <= code <= 629 or int(code) == 788:
        return "genitourinary"
    if 140 <= code <= 239:
        return "neoplasms"
    return "other"


@dataclass(frozen=True)
class EncodedData:
    train: np.ndarray
    test: np.ndarray
    names: list[str]
    source_variables: list[str]
    discrete_mask: np.ndarray
    excluded: list[dict[str, str]]


class ClinicalPreprocessor:
    def __init__(self) -> None:
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        self.excluded: list[dict[str, str]] = []

    @staticmethod
    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.drop(columns=[c for c in IDENTIFIERS if c in frame], errors="ignore").copy()
        for column in DIAGNOSIS_COLUMNS:
            if column in result:
                result[column] = result[column].map(map_icd9)
        return result

    def fit(self, train: pd.DataFrame) -> ClinicalPreprocessor:
        prepared = self._prepare(train)
        self.excluded = []
        for column in list(prepared.columns):
            values = prepared[column].replace(list(MISSING_MARKERS), np.nan)
            if values.isna().all():
                self.excluded.append({"feature": column, "reason": "all_missing_in_training"})
                prepared = prepared.drop(columns=column)
        self.numeric_columns = [c for c in prepared.columns if c in NUMERIC_COLUMNS]
        self.categorical_columns = [c for c in prepared.columns if c not in self.numeric_columns]
        if self.numeric_columns:
            numeric = prepared[self.numeric_columns].replace(list(MISSING_MARKERS), np.nan)
            numeric = numeric.apply(pd.to_numeric, errors="coerce")
            self.scaler.fit(self.numeric_imputer.fit_transform(numeric))
        if self.categorical_columns:
            categorical = prepared[self.categorical_columns].replace(list(MISSING_MARKERS), np.nan)
            categorical = categorical.fillna("__MISSING__").astype(str)
            self.encoder.fit(categorical)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
        prepared = self._prepare(frame)
        parts: list[np.ndarray] = []
        names: list[str] = []
        sources: list[str] = []
        discrete: list[bool] = []
        if self.numeric_columns:
            numeric = prepared[self.numeric_columns].replace(list(MISSING_MARKERS), np.nan)
            numeric = numeric.apply(pd.to_numeric, errors="coerce")
            parts.append(self.scaler.transform(self.numeric_imputer.transform(numeric)))
            names.extend(self.numeric_columns)
            sources.extend(self.numeric_columns)
            discrete.extend([False] * len(self.numeric_columns))
        if self.categorical_columns:
            categorical = prepared[self.categorical_columns].replace(list(MISSING_MARKERS), np.nan)
            categorical = categorical.fillna("__MISSING__").astype(str)
            parts.append(self.encoder.transform(categorical))
            cat_names = self.encoder.get_feature_names_out(self.categorical_columns).tolist()
            names.extend(cat_names)
            for column, categories in zip(
                self.categorical_columns, self.encoder.categories_, strict=True
            ):
                sources.extend([column] * len(categories))
            discrete.extend([True] * len(cat_names))
        matrix = np.column_stack(parts) if parts else np.empty((len(frame), 0))
        return matrix, names, sources, np.asarray(discrete, dtype=bool)

    def fit_transform_pair(self, train: pd.DataFrame, test: pd.DataFrame) -> EncodedData:
        self.fit(train)
        train_matrix, names, sources, discrete = self.transform(train)
        test_matrix, test_names, _, _ = self.transform(test)
        if names != test_names:
            raise RuntimeError("train/test encoded feature order mismatch")
        variable = np.nanvar(train_matrix, axis=0) > 1e-12
        excluded = list(self.excluded)
        excluded.extend(
            {"feature": name, "reason": "zero_variance_in_training"}
            for name, keep in zip(names, variable, strict=True)
            if not keep
        )
        return EncodedData(
            train=train_matrix[:, variable],
            test=test_matrix[:, variable],
            names=[name for name, keep in zip(names, variable, strict=True) if keep],
            source_variables=[name for name, keep in zip(sources, variable, strict=True) if keep],
            discrete_mask=discrete[variable],
            excluded=excluded,
        )
