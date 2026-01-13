from __future__ import annotations

from datetime import date

import polars as pl

from engine.core.ids import RunContext
from engine.io.parquet_io import write_parquet
from engine.io.paths import features_dir


def _ensure_identity_columns(ctx: RunContext, trading_day: date, df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure core identity columns exist for features tables.
    Adds dt (preferred) and trading_day (legacy-friendly).
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = df
    exprs: list[pl.Expr] = []

    if "snapshot_id" not in out.columns:
        exprs.append(pl.lit(ctx.snapshot_id).alias("snapshot_id"))
    if "run_id" not in out.columns:
        exprs.append(pl.lit(ctx.run_id).alias("run_id"))
    if "mode" not in out.columns:
        exprs.append(pl.lit(ctx.mode).alias("mode"))

    if "dt" not in out.columns:
        exprs.append(pl.lit(trading_day).cast(pl.Date).alias("dt"))
    if "trading_day" not in out.columns:
        exprs.append(pl.lit(trading_day).cast(pl.Date).alias("trading_day"))

    if exprs:
        out = out.with_columns(exprs)

    return out


def validate_features_frame(df: pl.DataFrame) -> None:
    """
    Minimal validation for features frames before writing.
    """
    if df is None or df.is_empty():
        return

    required = ["instrument", "anchor_tf", "ts"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"features frame is missing required columns: {missing}")


def write_features_for_instrument_tf_day(
    ctx: RunContext,
    df: pl.DataFrame,
    instrument: str,
    anchor_tf: str,
    trading_day: date,
    *,
    sandbox: bool = False,
) -> None:
    """
    Write features for a single (instrument, anchor_tf, trading_day) to Parquet.
    """
    if df is None or df.is_empty():
        return

    validate_features_frame(df)
    df = _ensure_identity_columns(ctx, trading_day, df)

    out_dir = features_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        anchor_tf=anchor_tf,
        sandbox=sandbox,
    )
    write_parquet(df, out_dir, file_name="0000.parquet")
