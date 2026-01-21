from __future__ import annotations

import logging

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry, safe_div

log = logging.getLogger(__name__)


def _empty_keyed_frame(registry_entry: dict | None) -> pl.DataFrame:
    if registry_entry and isinstance(registry_entry, dict):
        columns = registry_entry.get("columns")
        if isinstance(columns, dict) and columns:
            return pl.DataFrame({col: pl.Series([], dtype=pl.Null) for col in columns})
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "ts": pl.Series([], dtype=pl.Datetime("us")),
        }
    )


def _base_keys_from_candles(candles: pl.DataFrame) -> pl.DataFrame:
    if candles is None or candles.is_empty():
        return pl.DataFrame()
    if not {"instrument", "ts"}.issubset(set(candles.columns)):
        return pl.DataFrame()
    return (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
        )
        .drop_nulls(["instrument", "ts"])
        .unique()
        .sort(["instrument", "ts"])
    )


def build_feature_frame(
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    registry_entry: dict | None = None,
) -> pl.DataFrame:
    """
    Higher-timeframe correlation proxy using a longer rolling window.

    Table: data/features_corr
    Keys : instrument, ts
    """
    if candles is None or candles.is_empty():
        log.warning("corr_htf: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    if not {"instrument", "ts", "close"}.issubset(set(candles.columns)):
        log.warning("corr_htf: missing required columns; returning null-filled frame")
        base = _base_keys_from_candles(candles)
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        out = conform_to_registry(
            base,
            registry_entry=registry_entry,
            key_cols=["instrument", "ts"],
            where="corr_htf",
            allow_extra=False,
        )
        return out

    window = 200
    c = candles.select(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
        pl.col("close").cast(pl.Float64),
    ).drop_nulls(["instrument", "ts"])

    c = c.sort(["instrument", "ts"]).with_columns(
        pl.col("close").pct_change().over("instrument").alias("_ret"),
    )

    market_ret = (
        c.group_by("ts")
        .agg(pl.col("_ret").mean().alias("_market_ret"))
        .sort("ts")
    )
    c = c.join(market_ret, on="ts", how="left")

    by = ["instrument"]
    mean_ret = pl.col("_ret").rolling_mean(window_size=window, min_periods=window).over(by)
    mean_mkt = pl.col("_market_ret").rolling_mean(window_size=window, min_periods=window).over(by)
    mean_prod = (pl.col("_ret") * pl.col("_market_ret")).rolling_mean(window_size=window, min_periods=window).over(by)
    std_ret = pl.col("_ret").rolling_std(window_size=window, min_periods=window).over(by)
    std_mkt = pl.col("_market_ret").rolling_std(window_size=window, min_periods=window).over(by)

    corr = safe_div(mean_prod - (mean_ret * mean_mkt), std_ret * std_mkt, default=0.0).alias("corr_htf_proxy")

    out = c.with_columns(corr).select("instrument", "ts")
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "ts"],
        where="corr_htf",
        allow_extra=False,
    )

    log.info("corr_htf: built rows=%d window=%d", out.height, window)
    return out
