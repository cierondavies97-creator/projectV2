from __future__ import annotations

from datetime import date

import polars as pl

from engine.core.ids import RunContext
from engine.core.schema import enforce_table
from engine.io.parquet_io import write_parquet
from engine.io.paths import features_dir


# -----------------------------------------------------------------------------
# data/features writer
#
# Contract:
# - Always writes a parquet file for the requested partition (even if df is empty)
# - Adds engine identity columns (snapshot_id/run_id/mode/dt) if missing
# - Expands df to full schema contract via enforce_table("features")
# -----------------------------------------------------------------------------


def _as_str(v: object) -> str:
    if v is None:
        return ""
    # handle enums / pydantic types that carry a `.value`
    val = getattr(v, "value", None)
    return str(val) if val is not None else str(v)


def _ensure_identity_columns(ctx: RunContext, trading_day: date, df: pl.DataFrame) -> pl.DataFrame:
    out = df if df is not None else pl.DataFrame()

    add_exprs: list[pl.Expr] = []

    if "snapshot_id" not in out.columns:
        add_exprs.append(pl.lit(_as_str(getattr(ctx, "snapshot_id", ""))).alias("snapshot_id"))
    if "run_id" not in out.columns:
        add_exprs.append(pl.lit(_as_str(getattr(ctx, "run_id", ""))).alias("run_id"))
    if "mode" not in out.columns:
        add_exprs.append(pl.lit(_as_str(getattr(ctx, "mode", ""))).alias("mode"))

    # Canonical partition key
    if "dt" not in out.columns:
        add_exprs.append(pl.lit(trading_day).cast(pl.Date).alias("dt"))

    # Legacy-friendly (optional). Keep if present; add if absent.
    if "trading_day" not in out.columns:
        add_exprs.append(pl.lit(trading_day).cast(pl.Date).alias("trading_day"))

    if add_exprs:
        out = out.with_columns(add_exprs)

    return out


def validate_features_frame(df: pl.DataFrame) -> None:
    """
    Minimal validation: when df has rows, require the keyed columns exist.
    Schema enforcement will add missing columns, but keys should exist for non-empty frames.
    """
    if df is None or df.is_empty():
        return
    missing = [c for c in ("instrument", "anchor_tf", "ts") if c not in df.columns]
    if missing:
        raise ValueError(
            "data/features frame missing key columns: "
            f"{missing} columns_present={sorted(df.columns)} rows={df.height}"
        )


def write_features_for_instrument_tf_day(
    *,
    ctx: RunContext,
    df: pl.DataFrame | None,
    instrument: str,
    anchor_tf: str,
    trading_day: date,
    sandbox: bool,
) -> None:
    """
    Write one parquet file for a single (instrument, anchor_tf, trading_day) partition.

    The caller may pass an empty frame; we will still materialize an empty parquet with
    the full schema contract (useful for reproducible auditing/drift detection).
    """
    out = df if df is not None else pl.DataFrame()

    # Ensure identity columns, then contract-expand to include all declared schema cols.
    out = _ensure_identity_columns(ctx, trading_day, out)

    # Validate only when non-empty
    validate_features_frame(out)

    # Expand to compiled contract (adds typed NULL placeholders)
    out = enforce_table(out, "features", allow_extra=True, reorder=True)

    out_dir = features_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        anchor_tf=anchor_tf,
        sandbox=sandbox,
    )
    write_parquet(out, out_dir, file_name="0000.parquet")
