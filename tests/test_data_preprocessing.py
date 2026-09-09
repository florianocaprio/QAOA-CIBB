from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from quantum_risk.data import load_uci_csv
from quantum_risk.preprocessing import ClinicalPreprocessor, map_icd9


def test_target_mapping_preserves_literal_none(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "encounter_id": ["1", "2", "3"],
            "patient_nbr": ["10", "20", "30"],
            "A1Cresult": ["None", ">7", "Norm"],
            "readmitted": ["<30", ">30", "NO"],
        }
    ).to_csv(path, index=False)
    bundle = load_uci_csv(path)
    assert bundle.y.tolist() == [1, 0, 0]
    assert bundle.frame["A1Cresult"].tolist()[0] == "None"


def test_invalid_target_is_not_silently_negative(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"encounter_id": ["1"], "patient_nbr": ["10"], "readmitted": ["UNKNOWN"]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="invalid"):
        load_uci_csv(path)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("250.13", "diabetes"),
        ("414", "circulatory"),
        ("786", "respiratory"),
        ("787", "digestive"),
        ("820", "injury"),
        ("715", "musculoskeletal"),
        ("788", "genitourinary"),
        ("174", "neoplasms"),
        ("V45", "other"),
        ("?", "unknown"),
    ],
)
def test_versioned_icd9_mapping(code: str, expected: str) -> None:
    assert map_icd9(code) == expected


def test_unseen_test_category_does_not_change_training_vocabulary() -> None:
    train = pd.DataFrame(
        {
            "encounter_id": ["1", "2", "3", "4"],
            "patient_nbr": ["1", "2", "3", "4"],
            "race": ["A", "B", "A", "B"],
            "time_in_hospital": ["1", "2", "3", "4"],
            "readmitted": ["NO", "<30", "NO", "<30"],
        }
    )
    test = train.iloc[:1].copy()
    test.loc[:, "race"] = "UNSEEN"
    encoded = ClinicalPreprocessor().fit_transform_pair(train, test)
    assert not any("UNSEEN" in name for name in encoded.names)
    assert encoded.train.shape[1] == encoded.test.shape[1]
    assert np.isfinite(encoded.test).all()


def test_group_split_has_no_patient_overlap() -> None:
    groups = np.repeat(np.arange(30), 2)
    y = np.tile([0, 1], 30)
    frame = np.zeros((60, 1))
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in splitter.split(frame, y, groups):
        assert not (set(groups[train]) & set(groups[test]))
