from __future__ import annotations

from datetime import date

import polars as pl

from engine.core.ids import RunContext
from engine.core.schema import enforce_table
from engine.io.parquet_io import write_parquet
from engine.io.paths import pcra_dir


def _ensure_identity_columns(
    ctx: RunContext,
    trading_day: date,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Ensure core identity columns exist for pcr_a tables.
    """
    exprs: list[pl.Expr] = []

    if "snapshot_id" not in df.columns:
        exprs.append(pl.lit(ctx.snapshot_id).alias("snapshot_id"))
    if "run_id" not in df.columns:
        exprs.append(pl.lit(ctx.run_id).alias("run_id"))
    if "mode" not in df.columns:
        exprs.append(pl.lit(ctx.mode).alias("mode"))
    if "dt" not in df.columns:
        exprs.append(pl.lit(trading_day).cast(pl.Date).alias("dt"))
    else:
        exprs.append(pl.col("dt").cast(pl.Date, strict=False).alias("dt"))
    if "trading_day" not in df.columns:
        exprs.append(pl.lit(trading_day).alias("trading_day"))

    if not exprs:
        return df

    return df.with_columns(exprs)


def validate_pcra_frame(df: pl.DataFrame) -> None:
    """
    Minimal validation for pcr_a frames before writing.
    """
    if df.is_empty():
        return

    required = ["instrument", "anchor_tf", "pcr_window_ts"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"pcr_a frame is missing required columns: {missing}")


def write_pcra_for_instrument_tf_day(
    ctx: RunContext,
    df: pl.DataFrame,
    instrument: str,
    anchor_tf: str,
    trading_day: date,
    *,
    sandbox: bool = False,
) -> None:
    """
    Write pcr_a rows for a single (instrument, anchor_tf, trading_day).
    """
    df2 = df if df is not None else pl.DataFrame()
    if not df2.is_empty():
        validate_pcra_frame(df2)
    df2 = _ensure_identity_columns(ctx, trading_day, df2)
    df2 = enforce_table(df2, "pcr_a", allow_extra=True, reorder=True)

    out_dir = pcra_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        anchor_tf=anchor_tf,
        sandbox=sandbox,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "0000.parquet"
    write_parquet(df2, out_path.parent, file_name=out_path.name)
