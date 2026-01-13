from __future__ import annotations

import logging
from datetime import date

import polars as pl

from engine.core.ids import RunContext
from engine.io.parquet_io import write_parquet
from engine.io.paths import windows_dir

log = logging.getLogger(__name__)

# Minimal required columns for windows, per Engine Pipeline & Contracts.
# IMPORTANT: tf_entry is part of the key (Phase B supports multiple tf_entry).
REQUIRED_WINDOWS_COLUMNS: list[str] = [
    "instrument",
    "anchor_tf",
    "anchor_ts",
    "tf_entry",
]


def validate_windows_frame(df: pl.DataFrame) -> None:
    """
    Minimal structural validation for windows frames.
    Type tightening is handled via schema alignment in steps.
    """
    missing = [c for c in REQUIRED_WINDOWS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"windows frame is missing required columns: {missing}")


def _ensure_identity_columns(ctx: RunContext, trading_day: date, df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure core identity columns exist for windows tables.

    Note:
      - dt is stored as a Date for auditability (even though the table is partitioned by dt).
      - We do not fabricate paradigm/principle identity here; windows are paradigm-neutral.
    """
    if df is None or df.is_empty():
        return df

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

    return df if not exprs else df.with_columns(exprs)


def _canonicalise_windows_slice(df: pl.DataFrame, instrument: str, anchor_tf: str, trading_day: date) -> pl.DataFrame:
    """
    Defensive normalisation before write:
      - Filter to instrument/anchor_tf
      - Cast anchor_ts to Datetime(us)
      - Cast dt to Date if present
      - Filter to UTC date == trading_day (by anchor_ts date)
      - Sort deterministically
      - De-duplicate on (instrument, anchor_tf, anchor_ts, tf_entry)
    """
    if df is None or df.is_empty():
        return df

    df2 = df.filter((pl.col("instrument") == instrument) & (pl.col("anchor_tf") == anchor_tf))
    if df2.is_empty():
        return df2

    df2 = df2.with_columns(
        pl.col("anchor_ts").cast(pl.Datetime("us"), strict=False).alias("anchor_ts"),
    )

    if "dt" in df2.columns:
        df2 = df2.with_columns(pl.col("dt").cast(pl.Date, strict=False).alias("dt"))

    df2 = df2.filter(pl.col("anchor_ts").dt.date() == pl.lit(trading_day))
    if df2.is_empty():
        return df2

    df2 = df2.sort(["instrument", "anchor_tf", "tf_entry", "anchor_ts"]).unique(
        subset=["instrument", "anchor_tf", "anchor_ts", "tf_entry"],
        keep="first",
    )
    return df2


def write_windows_for_instrument_tf_day(
    ctx: RunContext,
    df: pl.DataFrame,
    instrument: str,
    anchor_tf: str,
    trading_day: date,
    *,
    sandbox: bool = False,
) -> None:
    """
    Write windows for (instrument, anchor_tf, trading_day) to disk.

    Layout:
      data/windows/run_id=<RUN_ID>/instrument=<INSTRUMENT>/anchor_tf=<ANCHOR_TF>/dt=<YYYY-MM-DD>/0000.parquet
    """
    if df is None or df.is_empty():
        log.info(
            "windows.write_windows_for_instrument_tf_day: empty frame for instrument=%s anchor_tf=%s dt=%s; nothing to write",
            instrument,
            anchor_tf,
            trading_day,
        )
        return

    validate_windows_frame(df)

    df_slice = _canonicalise_windows_slice(df=df, instrument=instrument, anchor_tf=anchor_tf, trading_day=trading_day)
    if df_slice.is_empty():
        log.info(
            "windows.write_windows_for_instrument_tf_day: no rows for instrument=%s anchor_tf=%s dt=%s after canonicalise; nothing to write",
            instrument,
            anchor_tf,
            trading_day,
        )
        return

    df_slice = _ensure_identity_columns(ctx, trading_day, df_slice)

    out_dir = windows_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        anchor_tf=anchor_tf,
        sandbox=sandbox,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "0000.parquet"
    write_parquet(df_slice, out_path.parent, file_name=out_path.name)

    log.info("windows.write_windows_for_instrument_tf_day: wrote %d rows to %s", df_slice.height, out_path)
