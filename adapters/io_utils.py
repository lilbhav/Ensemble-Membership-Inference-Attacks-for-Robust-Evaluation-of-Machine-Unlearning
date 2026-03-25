from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def project_root() -> Path:
    # Repository root (one level above adapters/)
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict[str, Any]:
    # Allow relative paths from project root for convenience
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config files. Install with: pip install pyyaml") from exc
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if suffix == ".json":
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config format: {cfg_path}")


def resolve_path(path_value: str | Path) -> Path:
    # Normalize project-relative paths from config into absolute paths
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def ensure_dir(path: str | Path) -> Path:
    # Create directory if needed and return it for chaining
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    # Small helper used by scripts to persist metadata blobs
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    # Centralized CSV writer to keep output formatting consistent
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_subprocess(cmd: list[str], cwd: str | Path | None = None) -> None:
    # Run external engine command and raise with full logs on failure
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    if proc.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(
            f"Subprocess failed ({proc.returncode}): {joined}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
