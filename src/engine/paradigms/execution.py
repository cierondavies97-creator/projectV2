from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from engine.research.snapshots import load_snapshot_manifest


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"Expected mapping in YAML: {path}")
    return obj


@lru_cache(maxsize=128)
def resolve_active_paradigm_and_principle(
    snapshot_id: str,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    """
    Resolve active (paradigm_id, principle_id) for a snapshot_id and load configs.

    Phase A contract:
      - Uses snapshot manifest as the source of truth for config file paths.
      - Does not mutate live config.
      - Cached for determinism and speed within a process.
    """
    manifest = load_snapshot_manifest(snapshot_id)

    # Find paradigm config path for "ict" (or whichever you set active).
    # For Phase A: choose the first paradigm in manifest list if present; else default to "ict".
    # If your snapshot JSON includes explicit active paradigm fields, we will add them later.
    if manifest.paradigms:
        paradigm_id = manifest.paradigms[0].paradigm_id
        paradigm_path = Path(manifest.paradigms[0].paradigm_config_path)
    else:
        paradigm_id = "ict"
        paradigm_path = Path(manifest.config_refs.paradigms_dir) / "ict.yaml"

    paradigm_cfg = _read_yaml(paradigm_path)

    # Principle selection: prefer paradigm default_principle_id; fallback.
    principle_id = str(paradigm_cfg.get("default_principle_id") or "ict_all_windows")

    # Find matching principle config path in manifest if available; else compute path.
    principle_path: Path | None = None
    for pr in manifest.principles:
        if pr.paradigm_id == paradigm_id and pr.principle_id == principle_id:
            principle_path = Path(pr.principle_config_path)
            break
    if principle_path is None:
        principle_path = Path(manifest.config_refs.principles_dir) / paradigm_id / f"{principle_id}.yaml"

    principle_cfg = _read_yaml(principle_path)

    return str(paradigm_id), str(principle_id), dict(principle_cfg), dict(paradigm_cfg)
