"""Configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    defaults_path = Path(__file__).with_name("defaults.yaml")
    with defaults_path.open("r", encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle) or {}
    with config_path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}
    merged = _deep_merge(defaults, override)
    merged["config_path"] = str(config_path)
    return merged
