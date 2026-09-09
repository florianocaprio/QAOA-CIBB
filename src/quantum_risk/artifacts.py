from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd


class ScientificJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def stable_hash(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, cls=ScientificJSONEncoder).encode()
    return hashlib.sha256(serialized).hexdigest()


def atomic_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, cls=ScientificJSONEncoder)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    destination.chmod(0o644)


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
