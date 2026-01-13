from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

from engine.core.config_models import load_retail_config
from engine.core.ids import RunContext


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def project_root() -> Path:
    """
    Resolve the repository/project root robustly.

    Resolution order:
      1) REALIGN_PROJECT_ROOT env var (explicit, recommended for tooling)
      2) Walk upwards from CWD looking for conf/retail.yaml
      3) Fallback to relative to this file (src/engine/io/paths.py -> project root)
    """
    env_root = os.getenv("REALIGN_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for p in (cwd, *cwd.parents):
        if (p / "conf" / "retail.yaml").exists():
            return p

    # .../project/src/engine/io/paths.py -> parents[3] == project
    return Path(__file__).resolve().parents[3]


def _resolve_cfg_path(value: str) -> Path:
    """
    Interpret configured roots as project-root relative unless absolute.
    """
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (project_root() / p).resolve()


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _retail_paths_root() -> tuple[Path, Path, Path]:
    """
    Read global roots from conf/retail.yaml:

      - data_root
      - sandbox_root
      - logs_root

    Notes:
      - If these are relative paths in YAML, we resolve relative to project_root().
      - EnvId currently does not change these roots; sandbox mode is explicit.
    """
    cfg = load_retail_config()

    data_root = _resolve_cfg_path(cfg.paths.data_root)
    sandbox_root = _resolve_cfg_path(cfg.paths.sandbox_root)
    logs_root = _resolve_cfg_path(cfg.paths.logs_root)

    return data_root, sandbox_root, logs_root


def data_root(ctx: RunContext, *, sandbox: bool = False) -> Path:
    """
    Root for Parquet-backed fact tables.

    By default returns the "real" data root (e.g. '<project>/data').
    If sandbox=True, returns the sandbox mirror (e.g. '<project>/sandbox_data/data').

    This preserves your current convention: sandbox_root contains 'data/' and 'logs/' subfolders.
    """
    data_root_path, sandbox_root_path, _ = _retail_paths_root()
    if sandbox:
        return (sandbox_root_path / "data").resolve()
    return data_root_path.resolve()


def logs_root(ctx: RunContext, *, sandbox: bool = False) -> Path:
    """
    Root for logs.

    If sandbox=True, logs under sandbox_root/logs; otherwise logs_root directly.
    """
    _, sandbox_root_path, logs_root_path = _retail_paths_root()
    if sandbox:
        return (sandbox_root_path / "logs").resolve()
    return logs_root_path.resolve()


def snapshots_root() -> Path:
    """
    Root for snapshot manifests.

    Use project-root absolute paths to avoid CWD surprises.
    """
    return (project_root() / "snapshots").resolve()


def proposals_root() -> Path:
    """
    Root for proposals.

    Use project-root absolute paths to avoid CWD surprises.
    """
    return (project_root() / "proposals").resolve()


# ---------------------------------------------------------------------------
# Helper formatting / safety
# ---------------------------------------------------------------------------

def _fmt_dt(trading_day: date) -> str:
    return trading_day.strftime("%Y-%m-%d")


def _sanitize_partition_value(value: str) -> str:
    """
    Make partition values safe for filesystem paths.
    """
    v = str(value)
    return v.replace("\\", "_").replace("/", "_").replace(":", "_").replace("..", "_")


def _part(key: str, value: str) -> str:
    if value is None or str(value) == "":
        raise ValueError(f"Partition '{key}' cannot be empty.")
    return f"{key}={_sanitize_partition_value(value)}"


def _candidate_partition(candidate_id: Optional[str]) -> str:
    """
    Candidate partition value.

    Use a stable sentinel when candidate_id is absent so Phase B directory layouts
    remain partition-compatible across runs where candidate_id is not yet used.
    """
    return candidate_id if candidate_id else "∅"


def _eval_partitions(
    *,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> list[str]:
    """
    Return ordered partition components for evaluation-aware layouts.

    Backward compatible behavior:
      - if paradigm_id is None -> return []
      - if paradigm_id is provided, include paradigm_id partition
      - include principle_id when provided
      - ALWAYS include candidate_id when in eval-mode (stable shape); use the "∅" sentinel from _candidate_partition if absent
      - include experiment_id only when provided (can be introduced later)

    Robustness:
      - treat blank/whitespace experiment_id as absent (do not create partition)
    """
    parts: list[str] = []
    if paradigm_id is None:
        return parts

    parts.append(_part("paradigm_id", paradigm_id))

    if principle_id is not None:
        parts.append(_part("principle_id", principle_id))

    parts.append(_part("candidate_id", _candidate_partition(candidate_id)))

    if experiment_id is not None:
        exp = str(experiment_id).strip()
        if exp != "":
            parts.append(_part("experiment_id", exp))

    return parts

def _resolve_instrument_kw(*, instrument_id: Optional[str], instrument: Optional[str]) -> str:
    """
    Allow both instrument_id= and instrument= callers for candles/ticks, without breaking
    existing call sites.
    """
    inst = instrument_id or instrument
    if not inst:
        raise ValueError("instrument_id or instrument must be provided.")
    return str(inst)


# ---------------------------------------------------------------------------
# Candles / ticks / macro
# ---------------------------------------------------------------------------

def candles_dir(
    ctx: RunContext,
    trading_day: date,
    *,
    instrument_id: Optional[str] = None,
    instrument: Optional[str] = None,
    sandbox: bool = False,
) -> Path:
    """
    data/candles/mode=<MODE>/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/
    """
    inst = _resolve_instrument_kw(instrument_id=instrument_id, instrument=instrument)
    root = data_root(ctx, sandbox=sandbox)
    return root / "candles" / _part("mode", ctx.mode) / _part("instrument", inst) / _part("dt", _fmt_dt(trading_day))


def ticks_dir(
    ctx: RunContext,
    trading_day: date,
    *,
    instrument_id: Optional[str] = None,
    instrument: Optional[str] = None,
    sandbox: bool = False,
) -> Path:
    """
    data/ticks/mode=<MODE>/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/
    """
    inst = _resolve_instrument_kw(instrument_id=instrument_id, instrument=instrument)
    root = data_root(ctx, sandbox=sandbox)
    return root / "ticks" / _part("mode", ctx.mode) / _part("instrument", inst) / _part("dt", _fmt_dt(trading_day))


def macro_dir(
    ctx: RunContext,
    trading_day: date,
    *,
    sandbox: bool = False,
) -> Path:
    """
    data/macro/dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    return root / "macro" / _part("dt", _fmt_dt(trading_day))


# ---------------------------------------------------------------------------
# Shared upstream artifacts (run-scoped by run_id, not evaluation-scoped)
# ---------------------------------------------------------------------------

def features_dir(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    anchor_tf: str,
    *,
    sandbox: bool = False,
) -> Path:
    """
    data/features/run_id=<RUN_ID>/instrument=<INSTRUMENT>/anchor_tf=<ANCHOR_TF>/dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    return (
        root
        / "features"
        / _part("run_id", ctx.run_id)
        / _part("instrument", instrument)
        / _part("anchor_tf", anchor_tf)
        / _part("dt", _fmt_dt(trading_day))
    )


def windows_dir(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    anchor_tf: str,
    *,
    sandbox: bool = False,
) -> Path:
    """
    data/windows/run_id=<RUN_ID>/instrument=<INSTRUMENT>/anchor_tf=<ANCHOR_TF>/dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    return (
        root
        / "windows"
        / _part("run_id", ctx.run_id)
        / _part("instrument", instrument)
        / _part("anchor_tf", anchor_tf)
        / _part("dt", _fmt_dt(trading_day))
    )


def zones_state_dir(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    anchor_tf: str,
    *,
    sandbox: bool = False,
    include_run_id: bool = False,
) -> Path:
    """
    Zones state directory.

    Backward-compatible default (include_run_id=False):
      data/zones_state/instrument=<INSTRUMENT>/anchor_tf=<ANCHOR_TF>/dt=<YYYY-MM-DD>/
      (no run_id; long-lived memory)

    Phase B / reproducibility option (include_run_id=True):
      data/zones_state/run_id=<RUN_ID>/instrument=<INSTRUMENT>/anchor_tf=<ANCHOR_TF>/dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    if include_run_id:
        return (
            root
            / "zones_state"
            / _part("run_id", ctx.run_id)
            / _part("instrument", instrument)
            / _part("anchor_tf", anchor_tf)
            / _part("dt", _fmt_dt(trading_day))
        )
    return root / "zones_state" / _part("instrument", instrument) / _part("anchor_tf", anchor_tf) / _part("dt", _fmt_dt(trading_day))


def pcra_dir(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    anchor_tf: str,
    *,
    sandbox: bool = False,
    include_run_id: bool = False,
) -> Path:
    """
    PCrA directory.

    Backward-compatible default (include_run_id=False):
      data/pcr_a/instrument=<INSTRUMENT>/anchor_tf=<ANCHOR_TF>/dt=<YYYY-MM-DD>/
      (no run_id; long-lived microstructure memory)

    Phase B / reproducibility option (include_run_id=True):
      data/pcr_a/run_id=<RUN_ID>/instrument=<INSTRUMENT>/anchor_tf=<ANCHOR_TF>/dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    if include_run_id:
        return (
            root
            / "pcr_a"
            / _part("run_id", ctx.run_id)
            / _part("instrument", instrument)
            / _part("anchor_tf", anchor_tf)
            / _part("dt", _fmt_dt(trading_day))
        )
    return root / "pcr_a" / _part("instrument", instrument) / _part("anchor_tf", anchor_tf) / _part("dt", _fmt_dt(trading_day))


# ---------------------------------------------------------------------------
# Evaluation-aware downstream artifacts (Phase B)
# ---------------------------------------------------------------------------

def decisions_dir(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    stage: str,
    *,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> Path:
    """
    Canonical on-disk path for decisions_* tables.

    Backward compatible layout (when paradigm_id is None):
      data/decisions/run_id=<RUN_ID>/instrument=<INSTRUMENT>/stage=<STAGE>/dt=<YYYY-MM-DD>/

    Phase B layout (when paradigm_id is provided):
      data/decisions/run_id=<RUN_ID>/
        paradigm_id=<PARADIGM>/
          principle_id=<PRINCIPLE>/        (optional but recommended)
            candidate_id=<CANDIDATE>/      (always present in eval-mode; "Ã¢Ë†â€¦" allowed)
              experiment_id=<EXPERIMENT>/  (optional)
                instrument=<INSTRUMENT>/
                  stage=<STAGE>/
                    dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    base = root / "decisions" / _part("run_id", ctx.run_id)

    for p in _eval_partitions(
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
    ):
        base = base / p

    return base / _part("instrument", instrument) / _part("stage", stage) / _part("dt", _fmt_dt(trading_day))


def trade_paths_dir(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    *,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> Path:
    """
    Backward compatible layout (when paradigm_id is None):
      data/trade_paths/run_id=<RUN_ID>/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/

    Phase B layout (when paradigm_id is provided):
      data/trade_paths/run_id=<RUN_ID>/paradigm_id=.../principle_id=.../candidate_id=.../(experiment_id=...)/
        instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    base = root / "trade_paths" / _part("run_id", ctx.run_id)

    for p in _eval_partitions(
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
    ):
        base = base / p

    return base / _part("instrument", instrument) / _part("dt", _fmt_dt(trading_day))


def brackets_dir(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    *,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> Path:
    """
    Backward compatible layout (when paradigm_id is None):
      data/brackets/run_id=<RUN_ID>/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/

    Phase B layout (when paradigm_id is provided):
      data/brackets/run_id=<RUN_ID>/paradigm_id=.../principle_id=.../candidate_id=.../(experiment_id=...)/
        instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    base = root / "brackets" / _part("run_id", ctx.run_id)

    for p in _eval_partitions(
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
    ):
        base = base / p

    return base / _part("instrument", instrument) / _part("dt", _fmt_dt(trading_day))


def orders_dir(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    *,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> Path:
    """
    Backward compatible layout (when paradigm_id is None):
      data/orders/run_id=<RUN_ID>/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/

    Phase B layout (when paradigm_id is provided):
      data/orders/run_id=<RUN_ID>/paradigm_id=.../principle_id=.../candidate_id=.../(experiment_id=...)/
        instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    base = root / "orders" / _part("run_id", ctx.run_id)

    for p in _eval_partitions(
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
    ):
        base = base / p

    return base / _part("instrument", instrument) / _part("dt", _fmt_dt(trading_day))


def fills_dir(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    *,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> Path:
    """
    Backward compatible layout (when paradigm_id is None):
      data/fills/run_id=<RUN_ID>/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/

    Phase B layout (when paradigm_id is provided):
      data/fills/run_id=<RUN_ID>/paradigm_id=.../principle_id=.../candidate_id=.../(experiment_id=...)/
        instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/
    """
    root = data_root(ctx, sandbox=sandbox)
    base = root / "fills" / _part("run_id", ctx.run_id)

    for p in _eval_partitions(
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
    ):
        base = base / p

    return base / _part("instrument", instrument) / _part("dt", _fmt_dt(trading_day))


def run_reports_dir(
    ctx: RunContext,
    trading_day: date,
    *,
    cluster_id: str,
    sandbox: bool = False,
) -> Path:
    """
    Run reports directory (kept run-scoped).

    data/run_reports/run_id=<RUN_ID>/dt=<YYYY-MM-DD>/cluster_id=<CLUSTER>/
    """
    root = data_root(ctx, sandbox=sandbox)
    return (
        root
        / "run_reports"
        / _part("run_id", ctx.run_id)
        / _part("dt", _fmt_dt(trading_day))
        / _part("cluster_id", cluster_id)
    )


__all__ = [
    "project_root",
    "data_root",
    "logs_root",
    "snapshots_root",
    "proposals_root",
    "candles_dir",
    "ticks_dir",
    "macro_dir",
    "features_dir",
    "windows_dir",
    "zones_state_dir",
    "pcra_dir",
    "decisions_dir",
    "trade_paths_dir",
    "brackets_dir",
    "orders_dir",
    "fills_dir",
    "run_reports_dir",
]

# ---------------------------------------------------------------------------
# Phase B: cluster-level evaluation artifacts
# ---------------------------------------------------------------------------

def principles_context_dir(
    ctx: RunContext,
    trading_day: date,
    *,
    cluster_id: str,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> Path:
    """
    Canonical on-disk path for principles_context (cluster-level).

    Base:
      data/principles_context/run_id=<RUN_ID>/(eval...)/dt=<YYYY-MM-DD>/cluster_id=<CLUSTER>/

    Evaluation partitions are included when paradigm_id is provided.
    """
    root = data_root(ctx, sandbox=sandbox)
    base = root / "principles_context" / f"run_id={ctx.run_id}"

    eval_parts = _eval_partitions(
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
    )
    for p in eval_parts:
        base = base / p

    return base / f"dt={_fmt_dt(trading_day)}" / f"cluster_id={cluster_id}"


def trade_clusters_dir(
    ctx: RunContext,
    trading_day: date,
    *,
    cluster_id: str,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> Path:
    """
    Canonical on-disk path for trade_clusters (cluster-level).

    Base:
      data/trade_clusters/run_id=<RUN_ID>/(eval...)/dt=<YYYY-MM-DD>/cluster_id=<CLUSTER>/

    Evaluation partitions are included when paradigm_id is provided.
    """
    root = data_root(ctx, sandbox=sandbox)
    base = root / "trade_clusters" / f"run_id={ctx.run_id}"

    eval_parts = _eval_partitions(
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
    )
    for p in eval_parts:
        base = base / p

    return base / f"dt={_fmt_dt(trading_day)}" / f"cluster_id={cluster_id}"
