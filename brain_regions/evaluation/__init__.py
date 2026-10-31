"""Evaluation helpers for articulator artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def write_json_artifact(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
