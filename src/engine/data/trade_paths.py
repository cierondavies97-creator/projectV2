# src/engine/data/trade_paths.py
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

log = logging.getLogger(__name__)


def write_trade_paths_for_day(
    *,
    ctx: Any,
    df: pl.DataFrame | None = None,
    trade_paths: pl.DataFrame | None = None,
    instrument: str | None = None,
    trading_day: date | None = None,
    sandbox: bool = False,
    **kwargs: Any,
) -> Path | None:
    """
    Persist the trade_paths partition for one trading day (optionally per-instrument).

    Compatibility:
      - hypotheses_step calls this with df=... (see hypotheses_step.py). :contentReference[oaicite:1]{index=1}
      - Some older callers may pass trade_paths=...

    Determinism:
      - No fitting/training; only writing.
      - Does not mutate configs.

    Returns:
      - Path to written output if written; else None.
    """
    # Accept either 'df' or 'trade_paths'
    if df is None:
        df = trade_paths

    if df is None or df.is_empty():
        log.info("write_trade_paths_for_day: empty; skipping write")
        return None

    # Prefer canonical schema-aware writer if present
    try:
        from engine.core.schema import TRADE_PATHS_SCHEMA  # type: ignore
        from engine.data.writer import write_table  # type: ignore

        # write_table should handle partitioning based on schema.partition_cols
        out = write_table(ctx=ctx, schema=TRADE_PATHS_SCHEMA, df=df, trading_day=trading_day, sandbox=sandbox)
        log.info("write_trade_paths_for_day: wrote via write_table -> %s", out)
        return out
    except Exception as e:
        log.debug("write_trade_paths_for_day: canonical write_table not available (%s); using fallback", e)

    # Fallback: deterministic parquet path under ctx.artifacts_root (if available)
    root = getattr(ctx, "artifacts_root", None) or getattr(ctx, "root", None)
    if not root:
        log.warning("write_trade_paths_for_day: no writer available and ctx has no artifacts_root; no-op")
        return None

    root_path = Path(root)
    run_id = getattr(ctx, "run_id", "unknown_run")
    snap = getattr(ctx, "snapshot_id", "unknown_snapshot")
    mode = getattr(ctx, "mode", "unknown_mode")
    dt = trading_day.isoformat() if trading_day else "unknown_dt"
    inst = instrument or "unknown_instrument"

    # Keep layout stable; avoid silently changing existing partitioning rules
    out_dir = root_path / "data" / "trade_paths" / f"snapshot_id={snap}" / f"run_id={run_id}" / f"mode={mode}" / f"instrument={inst}" / f"dt={dt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "part-0000.parquet"
    df.write_parquet(out_file)
    log.info("write_trade_paths_for_day: wrote fallback parquet -> %s", out_file)
    return out_file
