from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML configs. Install `pyyaml`."
            ) from exc
        return yaml.safe_load(text) or {}

    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


def write_markdown(text: str, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(text, encoding="utf-8")
    return output_path

