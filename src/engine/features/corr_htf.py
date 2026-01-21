from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry, safe_div

log = logging.getLogger(__name__)


def _empty_keyed_frame(registry_entry: Mapping[str, object] | None) -> pl.DataFrame:
    if registry_entry and isinstance(registry_entry, Mapping):
        columns = registry_entry.get("columns")
        if isinstance(columns, Mapping) and columns:
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


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    cfg = dict(auto_cfg.get("corr_htf", {}) if isinstance(auto_cfg, Mapping) else {})
    if isinstance(family_cfg, Mapping):
        cfg.update(family_cfg)
    return cfg


def build_feature_frame(
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
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
        return conform_to_registry(
            base,
            registry_entry=registry_entry,
            key_cols=["instrument", "ts"],
            where="corr_htf",
            allow_extra=False,
        )

    cfg = _merge_cfg(ctx, family_cfg)
    window = max(5, int(cfg.get("corr_htf_window_bars", 500)))
    min_periods = int(cfg.get("corr_htf_min_periods", 100))
    clip_abs = float(cfg.get("corr_htf_clip_abs", 0.999))
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

    corr_raw = safe_div(mean_prod - (mean_ret * mean_mkt), std_ret * std_mkt, default=None)
    corr = corr_raw.clip(-clip_abs, clip_abs).alias("corr_htf_ref1")

    out = c.with_columns(corr).with_columns(
        pl.lit("MARKET").alias("corr_htf_ref1_id"),
        pl.lit(None).cast(pl.Utf8).alias("corr_htf_ref2_id"),
        pl.lit(None).cast(pl.Float64).alias("corr_htf_ref2"),
        pl.lit(None).cast(pl.Utf8).alias("corr_htf_ref3_id"),
        pl.lit(None).cast(pl.Float64).alias("corr_htf_ref3"),
        pl.when(pl.col("corr_htf_ref1").is_null())
        .then(pl.lit(None).cast(pl.Utf8))
        .when(pl.col("corr_htf_ref1").abs() < pl.lit(0.2))
        .then(pl.lit("unstable"))
        .when(pl.col("corr_htf_ref1") >= 0)
        .then(pl.lit("aligned"))
        .otherwise(pl.lit("divergent"))
        .alias("htf_corr_regime"),
        pl.col("corr_htf_ref1").abs().alias("htf_corr_confidence"),
        pl.lit(None).cast(pl.Utf8).alias("corr_cluster_id_htf"),
        pl.lit(None).cast(pl.Float64).alias("corr_cluster_confidence_htf"),
        pl.lit(None).cast(pl.Float64).alias("corr_cluster_stability_htf"),
    ).select(
        "instrument",
        "ts",
        "corr_htf_ref1_id",
        "corr_htf_ref1",
        "corr_htf_ref2_id",
        "corr_htf_ref2",
        "corr_htf_ref3_id",
        "corr_htf_ref3",
        "htf_corr_regime",
        "htf_corr_confidence",
        "corr_cluster_id_htf",
        "corr_cluster_confidence_htf",
        "corr_cluster_stability_htf",
    )

    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "ts"],
        where="corr_htf",
        allow_extra=False,
    )

    log.info("corr_htf: built rows=%d window=%d", out.height, window)
    return out
