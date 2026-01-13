from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.core.ids import RunContext
from engine.core.schema import SnapshotManifest, load_snapshot_manifest
from engine.io import paths as io_paths


@dataclass(frozen=True)
class EngineLayout:
    """
    Resolved repository layout.

    All paths are absolute, derived deterministically from the repo root.
    """
    repo_root: Path
    conf_root: Path
    snapshots_root: Path
    proposals_root: Path


@dataclass(frozen=True)
class ResolvedSnapshot:
    """
    Snapshot manifest + resolved absolute paths for config references.
    """
    snapshot_id: str
    manifest_path: Path
    manifest: SnapshotManifest

    retail_config_path: Path
    features_registry_path: Path
    features_auto_path: Path
    rails_auto_path: Path
    portfolio_auto_path: Path
    path_filters_auto_path: Path
    paradigms_dir: Path
    principles_dir: Path


def repo_root() -> Path:
    """
    Resolve repo root from the installed source tree:
      <repo>/src/engine/core/context.py -> parents[3] = <repo>

    This avoids depending on current working directory.
    """
    return Path(__file__).resolve().parents[3]


def layout(root: Optional[Path] = None) -> EngineLayout:
    """
    Compute absolute layout roots.
    """
    rr = (root or repo_root()).resolve()
    return EngineLayout(
        repo_root=rr,
        conf_root=(rr / "conf").resolve(),
        snapshots_root=(rr / io_paths.snapshots_root()).resolve(),
        proposals_root=(rr / io_paths.proposals_root()).resolve(),
    )


def resolve_repo_path(p: Path | str, *, root: Optional[Path] = None) -> Path:
    """
    Interpret p as absolute if already absolute, else relative to repo root.
    """
    rr = (root or repo_root()).resolve()
    pp = Path(p)
    return pp if pp.is_absolute() else (rr / pp).resolve()


def resolve_snapshot_manifest_path(snapshot_id_or_path: str, *, root: Optional[Path] = None) -> Path:
    """
    Accept either:
      - a bare snapshot id (e.g. "dev_retail_v1")
      - or a file path (absolute or repo-relative)

    Resolution:
      1) If it looks like a file path (has suffix), resolve as repo-relative/absolute.
      2) Else, look in snapshots_root for <id>.json, <id>.yaml, <id>.yml (in that order).
    """
    rr = (root or repo_root()).resolve()
    candidate = Path(snapshot_id_or_path)

    # If user provided an explicit filename (e.g. endswith .json/.yaml), treat as path.
    if candidate.suffix.lower() in (".json", ".yaml", ".yml"):
        p = resolve_repo_path(candidate, root=rr)
        if not p.exists():
            raise FileNotFoundError(f"Snapshot manifest path not found: {p}")
        return p

    # Otherwise treat as an ID under snapshots_root
    snaps = (rr / io_paths.snapshots_root()).resolve()
    for ext in (".json", ".yaml", ".yml"):
        p = (snaps / f"{snapshot_id_or_path}{ext}").resolve()
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Snapshot manifest not found for snapshot_id={snapshot_id_or_path!r} "
        f"under {snaps} (tried .json/.yaml/.yml)."
    )


def load_snapshot_for_ctx(ctx: RunContext, *, root: Optional[Path] = None) -> ResolvedSnapshot:
    """
    Load the snapshot manifest for ctx.snapshot_id and resolve referenced config paths
    to absolute repo-rooted paths.

    Side-effect free: read-only.
    """
    rr = (root or repo_root()).resolve()
    manifest_path = resolve_snapshot_manifest_path(ctx.snapshot_id, root=rr)
    manifest = load_snapshot_manifest(manifest_path)

    # Resolve config ref paths relative to repo root
    refs = manifest.config_refs
    return ResolvedSnapshot(
        snapshot_id=ctx.snapshot_id,
        manifest_path=manifest_path,
        manifest=manifest,

        retail_config_path=resolve_repo_path(refs.retail_config_path, root=rr),
        features_registry_path=resolve_repo_path(refs.features_registry_path, root=rr),
        features_auto_path=resolve_repo_path(refs.features_auto_path, root=rr),
        rails_auto_path=resolve_repo_path(refs.rails_auto_path, root=rr),
        portfolio_auto_path=resolve_repo_path(refs.portfolio_auto_path, root=rr),
        path_filters_auto_path=resolve_repo_path(refs.path_filters_auto_path, root=rr),
        paradigms_dir=resolve_repo_path(refs.paradigms_dir, root=rr),
        principles_dir=resolve_repo_path(refs.principles_dir, root=rr),
    )
