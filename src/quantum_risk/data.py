from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

EXPECTED_TARGETS = {"<30", ">30", "NO"}
TRACE_COLUMNS = ("encounter_id", "patient_nbr")


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    encounter_ids: pd.Series
    metadata: dict[str, object]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_uci_csv(path: str | Path) -> DatasetBundle:
    source = Path(path)
    # keep_default_na=False preserves clinically meaningful literal "None".
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {"encounter_id", "patient_nbr", "readmitted"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    observed = set(frame["readmitted"].unique())
    unexpected = observed - EXPECTED_TARGETS
    if unexpected or frame["readmitted"].eq("").any():
        raise ValueError(f"invalid or missing readmitted values: {sorted(unexpected)}")
    y = frame["readmitted"].map({"<30": 1, ">30": 0, "NO": 0}).astype("int8")
    metadata: dict[str, object] = {
        "source": str(source),
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256_file(source),
        "rows": int(len(frame)),
        "raw_columns": int(frame.shape[1]),
        "patients": int(frame["patient_nbr"].nunique()),
        "target_distribution": frame["readmitted"].value_counts().sort_index().to_dict(),
        "positive_prevalence": float(y.mean()),
    }
    return DatasetBundle(
        frame=frame,
        y=y,
        groups=frame["patient_nbr"].copy(),
        encounter_ids=frame["encounter_id"].copy(),
        metadata=metadata,
    )


def fetch_with_ucimlrepo(dataset_id: int = 296) -> DatasetBundle:
    """Fetch UCI data while retaining encounter and patient identifiers."""
    if dataset_id != 296:
        raise ValueError("this project supports UCI dataset 296 only")
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=dataset_id)
    original = getattr(dataset.data, "original", None)
    if original is None:
        parts = [getattr(dataset.data, name, None) for name in ("ids", "features", "targets")]
        original = pd.concat([part for part in parts if part is not None], axis=1)
    missing_ids = set(TRACE_COLUMNS) - set(original.columns)
    if missing_ids:
        raise ValueError(f"ucimlrepo response omitted identifiers: {sorted(missing_ids)}")
    frame = original.copy().fillna("").astype(str)
    observed = set(frame["readmitted"].unique())
    unexpected = observed - EXPECTED_TARGETS
    if unexpected:
        raise ValueError(f"invalid target values from ucimlrepo: {sorted(unexpected)}")
    y = frame["readmitted"].map({"<30": 1, ">30": 0, "NO": 0}).astype("int8")
    metadata: dict[str, object] = {
        "source": "ucimlrepo.fetch_ucirepo(id=296)",
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "rows": int(len(frame)),
        "raw_columns": int(frame.shape[1]),
        "patients": int(frame["patient_nbr"].nunique()),
        "target_distribution": frame["readmitted"].value_counts().sort_index().to_dict(),
        "positive_prevalence": float(y.mean()),
    }
    return DatasetBundle(frame, y, frame["patient_nbr"], frame["encounter_id"], metadata)


def deterministic_patient_subsample(
    bundle: DatasetBundle, maximum: int, seed: int
) -> DatasetBundle:
    unique = bundle.groups.drop_duplicates()
    if len(unique) <= maximum:
        return bundle
    selected = set(unique.sample(n=maximum, random_state=seed).tolist())
    mask = bundle.groups.isin(selected)
    frame = bundle.frame.loc[mask].reset_index(drop=True)
    y = bundle.y.loc[mask].reset_index(drop=True)
    groups = frame["patient_nbr"].copy()
    encounters = frame["encounter_id"].copy()
    metadata = dict(bundle.metadata)
    metadata.update(
        {
            "subsample_policy": "deterministic_patient_sample_with_all_encounters",
            "subsample_max_patients": maximum,
            "subsample_seed": seed,
            "rows_after_subsample": int(len(frame)),
            "patients_after_subsample": int(groups.nunique()),
            "positive_prevalence_after_subsample": float(y.mean()),
        }
    )
    return DatasetBundle(frame, y, groups, encounters, metadata)
