from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_policy(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)
    if not isinstance(policy, dict):
        raise ValueError(f"Policy YAML must be a mapping: {path}")
    return policy
