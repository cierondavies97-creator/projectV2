from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from datetime import date
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# Output feature contract
OUT_COLS = [
    "instrument",
    "anchor_tf",
    "ts",
    "macro_is_blackout",
    "macro_blackout_max_impact",
]

# Calendar minimum contract (as stored on disk)
REQUIRED_CALENDAR_COLS = {
    "event_id",
    "event_ts",
    "currency",
    "impact",
    "blackout_start_ts",
    "blackout_end_ts",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _tf_to_truncate_rule(tf: str) -> str:
    """
    Map engine TF strings to Polars truncate rules.

    Examples:
      M1 -> "1m"
      M5 -> "5m"
      H1 -> "1h"
      D1 -> "1d"
    """
    tf = (tf or "").strip().upper()
    if tf.startswith("M"):
        return f"{int(tf[1:])}m"
    if tf.startswith("H"):
        return f"{int(tf[1:])}h"
    if tf in ("D1", "1D", "D"):
        return "1d"
    raise ValueError(f"macro_calendar: unsupported anchor_tf={tf!r} (expected M#, H#, D1)")


def _macro_calendar_path_for_day(trading_day: date) -> str:
    base = os.path.join("data", "macro", "calendar")
    return os.path.join(base, f"dt={trading_day:%Y-%m-%d}", "calendar.parquet")


def _impact_to_i32(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize vendor impact to Int32 for aggregation.

    Accepted inputs in calendar parquet:
      - labels: LOW / MEDIUM / HIGH (or L/M/H)
      - numeric strings: "0"/"1"/"2"/"3"
      - numeric dtypes

    Output:
      - impact: Int32 in {0,1,2,3}
    """
    if df.is_empty() or "impact" not in df.columns:
        return df

    impact_dtype = df.schema.get("impact")
    if impact_dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        return df.with_columns(pl.col("impact").cast(pl.Int32, strict=False))

    s = pl.col("impact").cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()
    impact_i32 = (
        pl.when(s.is_in(["HIGH", "H", "3"]))
        .then(pl.lit(3))
        .when(s.is_in(["MEDIUM", "MED", "M", "2"]))
        .then(pl.lit(2))
        .when(s.is_in(["LOW", "L", "1"]))
        .then(pl.lit(1))
        .otherwise(s.cast(pl.Int32, strict=False).fill_null(0))
        .cast(pl.Int32)
        .alias("impact")
    )
    return df.with_columns(impact_i32)


def _coerce_calendar_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize calendar types for engine consumption:
      - timestamps: Datetime("us"), tz-naive (UTC)
      - currency: uppercase string
      - impact: Int32 severity
    """
    if df.is_empty():
        return df

    missing = REQUIRED_CALENDAR_COLS - set(df.columns)
    if missing:
        raise ValueError(f"macro_calendar: missing columns in calendar parquet: {missing}")

    def _to_utc_naive_us(col: str) -> pl.Expr:
        e = pl.col(col)
        e = e.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        return e.cast(pl.Datetime("us"), strict=False).alias(col)

    df = df.with_columns(
        pl.col("event_id").cast(pl.Utf8, strict=False),
        _to_utc_naive_us("event_ts"),
        _to_utc_naive_us("blackout_start_ts"),
        _to_utc_naive_us("blackout_end_ts"),
        pl.col("currency").cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase().alias("currency"),
    )

    df = _impact_to_i32(df)

    # Deterministic ordering for audit/debug
    return df.sort(["event_ts", "currency", "impact", "event_id"])


def _load_macro_calendar_for_day(trading_day: date) -> pl.DataFrame:
    """
    Load the macro calendar for a given trading_day.
    """
    path = _macro_calendar_path_for_day(trading_day)
    if not os.path.exists(path):
        logger.info("macro_calendar: no calendar file for %s at %s", trading_day, path)
        return pl.DataFrame([])

    df = pl.read_parquet(path)

    missing = REQUIRED_CALENDAR_COLS - set(df.columns)
    if missing:
        raise ValueError(f"macro_calendar: missing columns in {path}: {missing}")

    return _coerce_calendar_schema(df)


def _default_features_for_bars(bars_df: pl.DataFrame) -> pl.DataFrame:
    """
    Always return bar-keyed rows with default values.
    """
    return bars_df.with_columns(
        pl.lit(False).alias("macro_is_blackout"),
        pl.lit(0).cast(pl.Int32).alias("macro_blackout_max_impact"),
    ).select(OUT_COLS)


# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------
def build_feature_frame(
    *,
    ctx,
    candles: pl.DataFrame | None = None,
    ticks: pl.DataFrame | None = None,  # accepted for API compatibility
    macro: pl.DataFrame | None = None,  # accepted for API compatibility
    external: pl.DataFrame | None = None,  # accepted for API compatibility
    family_cfg: Mapping[str, Any] | None = None,
    registry_entry: Mapping[str, Any] | None = None,
) -> pl.DataFrame:
    """
    Build macro_calendar features keyed on ['instrument', 'anchor_tf', 'ts'].

    Behavior:
      - Derive the bar index from `candles`, but publish on the canonical
        anchor grid by truncating timestamps to anchor_tf.
      - Load events for ctx.trading_day.
      - Mark blackout bars and compute max impact per bar.

    Notes:
      - Stored under data/macro/calendar/dt=.../calendar.parquet
      - Blackout interval semantics are half-open: [blackout_start_ts, blackout_end_ts)
      - Phase A: single anchor_tf per microbatch assumed (ctx.cluster.anchor_tfs[0]).
    """
    if candles is None or candles.is_empty():
        logger.info(
            "macro_calendar: no candles available for snapshot_id=%s run_id=%s; returning empty frame",
            getattr(ctx, "snapshot_id", None),
            getattr(ctx, "run_id", None),
        )
        return pl.DataFrame([])

    cols = set(candles.columns)
    missing_core = [c for c in ("instrument", "ts") if c not in cols]
    if missing_core:
        logger.warning("macro_calendar: candles missing core columns %s; returning empty frame", missing_core)
        return pl.DataFrame([])

    # Canonical anchor_tf for this microbatch
    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    anchor_tf_value = anchor_tfs[0] if anchor_tfs else "M5"
    if not anchor_tfs:
        logger.warning("macro_calendar: ctx.cluster.anchor_tfs empty; defaulting anchor_tf=%s", anchor_tf_value)

    rule = _tf_to_truncate_rule(anchor_tf_value)

    # Build bars_df on the anchor grid
    bars_df = (
        candles.select(["instrument", "ts"])
        .with_columns(pl.col("ts").cast(pl.Datetime("us"), strict=False))
        .with_columns(pl.col("ts").dt.truncate(rule).alias("ts"))
        .unique()
        .with_columns(pl.lit(anchor_tf_value).cast(pl.Utf8).alias("anchor_tf"))
        .select(["instrument", "anchor_tf", "ts"])
        .sort(["instrument", "anchor_tf", "ts"])
    )

    if bars_df.is_empty():
        return pl.DataFrame([])

    # Determine trading_day
    trading_day = getattr(ctx, "trading_day", None)
    if trading_day is None:
        first_ts = bars_df.select(pl.col("ts").min()).item()
        if first_ts is None:
            return pl.DataFrame([])
        trading_day = first_ts.date()

    events_df = _load_macro_calendar_for_day(trading_day)
    if events_df.is_empty():
        logger.info(
            "macro_calendar: no events for trading_day=%s; defaults for %d bars",
            trading_day,
            bars_df.height,
        )
        return _default_features_for_bars(bars_df)

    # Cross join bars with events, then filter to blackout windows (half-open interval)
    cross = bars_df.join(events_df, how="cross")
    blackout_cross = cross.filter(
        (pl.col("ts") >= pl.col("blackout_start_ts")) & (pl.col("ts") < pl.col("blackout_end_ts"))
    )

    if blackout_cross.is_empty():
        logger.info("macro_calendar: built features for %d bars (blackout rows=0)", bars_df.height)
        return _default_features_for_bars(bars_df)

    # Aggregate per bar: max impact across overlapping events
    agg = (
        blackout_cross.group_by(["instrument", "anchor_tf", "ts"])
        .agg(pl.col("impact").max().cast(pl.Int32).alias("macro_blackout_max_impact"))
        .with_columns(pl.lit(True).alias("macro_is_blackout"))
    )

    # Join back to bars and fill defaults
    features = (
        bars_df.join(agg, on=["instrument", "anchor_tf", "ts"], how="left")
        .with_columns(
            pl.col("macro_is_blackout").fill_null(False),
            pl.col("macro_blackout_max_impact").fill_null(0).cast(pl.Int32),
        )
        .select(OUT_COLS)
    )

    logger.info(
        "macro_calendar: built features for %d bars (blackout bars=%d)",
        features.height,
        features.filter(pl.col("macro_is_blackout")).height,
    )

    return features
