"""Load the one public configuration file and resolve repository paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(
    os.environ.get("BRIGHTRATE_CONFIG", REPO_ROOT / "configs/default.yaml")
).expanduser()


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {source}")
    payload["_source"] = str(source)
    return payload


CONFIG = load_config()


def repo_path(name: str) -> Path:
    """Return a configured path, relative to the repository when needed."""
    override = os.environ.get(f"BRIGHTRATE_{name.upper()}")
    configured = override if override is not None else CONFIG["paths"][name]
    path = Path(configured).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def value(section: str, name: str) -> Any:
    return CONFIG[section][name]
