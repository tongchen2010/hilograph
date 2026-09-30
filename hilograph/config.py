"""YAML configuration loading with single-parent inheritance and CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config. A top-level ``inherits: <relative path>`` key merges
    this file on top of its parent."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherits", None)
    if parent is not None:
        cfg = _deep_update(load_config(path.parent / parent), cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides; values are parsed as YAML scalars."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # YAML reads "1e-4" as a string
            try:
                value = float(value)
            except ValueError:
                pass
        node[keys[-1]] = value
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
