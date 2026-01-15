# src/engine/data/trade_paths.py
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from engine.core.schema import enforce_table
from engine.io.parquet_io import write_parquet
from engine.io.paths import trade_paths_dir

log = logging.getLogger(__name__)

_EVAL_COLS = ("paradigm_id", "principle_id", "candidate_id", "experiment_id")


def _as_str(v: object | None, *, default: str = "∅") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _ensure_identity_columns(ctx: Any, trading_day: date, df: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    if "snapshot_id" not in df.columns:
        exprs.append(pl.lit(getattr(ctx, "snapshot_id", "")).alias("snapshot_id"))
    if "run_id" not in df.columns:
        exprs.append(pl.lit(getattr(ctx, "run_id", "")).alias("run_id"))
    if "mode" not in df.columns:
        exprs.append(pl.lit(getattr(ctx, "mode", "")).alias("mode"))
    if "dt" not in df.columns:
        exprs.append(pl.lit(trading_day).cast(pl.Date).alias("dt"))
    else:
        exprs.append(pl.col("dt").cast(pl.Date, strict=False).alias("dt"))
    return df.with_columns(exprs) if exprs else df


def _normalize_eval_cols(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for col in _EVAL_COLS:
        if col not in out.columns:
            out = out.with_columns(pl.lit("∅").cast(pl.Utf8).alias(col))
        else:
            out = out.with_columns(
                pl.when(pl.col(col).is_null() | (pl.col(col).cast(pl.Utf8).str.strip_chars() == ""))
                .then(pl.lit("∅"))
                .otherwise(pl.col(col).cast(pl.Utf8))
                .alias(col)
            )
    return out


def _resolve_instrument(df: pl.DataFrame, instrument: str | None) -> str:
    if instrument:
        return instrument
    if "instrument" not in df.columns:
        raise ValueError("write_trade_paths_for_day requires instrument or an instrument column in df")
    unique = df.select(pl.col("instrument").unique()).to_series().to_list()
    unique = [u for u in unique if u is not None]
    if len(unique) != 1:
        raise ValueError(f"write_trade_paths_for_day expected a single instrument, got {unique}")
    return str(unique[0])


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
      - hypotheses_step calls this with df=...
      - Some older callers may pass trade_paths=...
    """
    if df is None:
        df = trade_paths

    if df is None or df.is_empty():
        log.info("write_trade_paths_for_day: empty; skipping write")
        return None

    if trading_day is None:
        raise ValueError("write_trade_paths_for_day requires trading_day")

    out = _normalize_eval_cols(df)
    out = _ensure_identity_columns(ctx, trading_day, out)
    out = enforce_table(out, "trade_paths", allow_extra=True, reorder=True)

    inst = _resolve_instrument(out, instrument)
    eval_vals = {col: _as_str(out.get_column(col)[0]) for col in _EVAL_COLS}

    out_dir = trade_paths_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=inst,
        paradigm_id=eval_vals["paradigm_id"],
        principle_id=eval_vals["principle_id"],
        candidate_id=eval_vals["candidate_id"],
        experiment_id=eval_vals["experiment_id"],
        sandbox=sandbox,
    )
    write_parquet(out, out_dir, file_name="0000.parquet")
    log.info("write_trade_paths_for_day: wrote %d rows to %s", out.height, out_dir / "0000.parquet")
    return out_dir / "0000.parquet"
