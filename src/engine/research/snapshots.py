from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.core.schema import (
    SnapshotConfigRefs,
    SnapshotDataSlice,
    SnapshotManifest,
    SnapshotParadigmRef,
    SnapshotPrincipleRef,
)


def snapshot_manifest_path(snapshot_id: str, project_root: Path | None = None) -> Path:
    """
    Return the path to the snapshot manifest JSON for a given snapshot_id.

    By default this assumes the current working directory is the project root
    and that manifests live under:

        snapshots/<snapshot_id>.json
    """
    root = project_root or Path(".")
    return root / "snapshots" / f"{snapshot_id}.json"


def _load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file into a Python dict.

    Raises FileNotFoundError if the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {path}")

    # utf-8-sig handles files that have a BOM or not.
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _parse_config_refs(obj: dict[str, Any]) -> SnapshotConfigRefs:
    return SnapshotConfigRefs(
        retail_config_path=obj["retail_config_path"],
        features_registry_path=obj["features_registry_path"],
        features_auto_path=obj["features_auto_path"],
        rails_auto_path=obj["rails_auto_path"],
        portfolio_auto_path=obj["portfolio_auto_path"],
        path_filters_auto_path=obj["path_filters_auto_path"],
        paradigms_dir=obj["paradigms_dir"],
        principles_dir=obj["principles_dir"],
    )


def _parse_paradigms(arr: list[dict[str, Any]]) -> list[SnapshotParadigmRef]:
    return [
        SnapshotParadigmRef(
            paradigm_id=item["paradigm_id"],
            paradigm_config_path=item["paradigm_config_path"],
        )
        for item in arr
    ]


def _parse_principles(arr: list[dict[str, Any]]) -> list[SnapshotPrincipleRef]:
    return [
        SnapshotPrincipleRef(
            paradigm_id=item["paradigm_id"],
            principle_id=item["principle_id"],
            principle_config_path=item["principle_config_path"],
        )
        for item in arr
    ]


def _parse_data_slice(obj: dict[str, Any]) -> SnapshotDataSlice:
    return SnapshotDataSlice(
        start_dt=obj["start_dt"],
        end_dt=obj["end_dt"],
        instruments=list(obj.get("instruments", [])),
        anchor_tfs=list(obj.get("anchor_tfs", [])),
        tf_entries=list(obj.get("tf_entries", [])),
        contexts_filter=obj.get("contexts_filter", ""),
    )


def load_snapshot_manifest(
    snapshot_id: str,
    project_root: Path | None = None,
) -> SnapshotManifest:
    """
    Load a SnapshotManifest from snapshots/<snapshot_id>.json.

    This is the single canonical way the engine and research lane should
    read snapshot manifests.
    """
    path = snapshot_manifest_path(snapshot_id, project_root=project_root)
    raw = _load_json(path)

    config_refs = _parse_config_refs(raw["config_refs"])
    paradigms = _parse_paradigms(raw.get("paradigms", []))
    principles = _parse_principles(raw.get("principles", []))
    data_slice = _parse_data_slice(raw["data_slice"])

    return SnapshotManifest(
        snapshot_id=raw["snapshot_id"],
        description=raw.get("description", ""),
        created_ts=raw["created_ts"],
        config_refs=config_refs,
        paradigms=paradigms,
        principles=principles,
        data_slice=data_slice,
    )
