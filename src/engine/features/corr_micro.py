from __future__ import annotations

import logging

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import safe_div

log = logging.getLogger(__name__)


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "ts": pl.Series([], dtype=pl.Datetime("us")),
            "corr_dxy": pl.Series([], dtype=pl.Float64),
            "corr_index_major": pl.Series([], dtype=pl.Float64),
            "corr_oil": pl.Series([], dtype=pl.Float64),
            "micro_corr_regime": pl.Series([], dtype=pl.Utf8),
            "corr_cluster_id": pl.Series([], dtype=pl.Utf8),
        }
    )


def build_feature_frame(
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Micro correlation features using rolling correlation vs a cluster mean return.

    Table: data/features_corr
    Keys : instrument, ts
    """
    if candles is None or candles.is_empty():
        log.warning("corr_micro: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("corr_micro: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg = dict(auto_cfg.get("corr_micro", {}) if isinstance(auto_cfg, dict) else {})

    window = max(5, int(fam_cfg.get("corr_window_bars", 50)))
    strong_cut = float(fam_cfg.get("corr_strong_cut", 0.5))

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []

    c = candles.select(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
        pl.col("close").cast(pl.Float64),
        pl.col("tf").cast(pl.Utf8).alias("_tf") if "tf" in candles.columns else pl.lit(None).cast(pl.Utf8).alias("_tf"),
    ).drop_nulls(["instrument", "ts"])

    if anchor_tfs and "_tf" in c.columns:
        c = c.filter(pl.col("_tf").is_in([str(tf) for tf in anchor_tfs]))

    if c.is_empty():
        return _empty_keyed_frame()

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

    cov = (mean_prod - (mean_ret * mean_mkt)).alias("_cov")
    corr = safe_div(cov, std_ret * std_mkt, default=0.0).alias("corr_index_major")

    df = c.with_columns(cov, corr).with_columns(
        pl.col("corr_index_major").alias("corr_dxy"),
        pl.col("corr_index_major").alias("corr_oil"),
        pl.when(pl.col("corr_index_major") >= pl.lit(strong_cut))
        .then(pl.lit("aligned"))
        .when(pl.col("corr_index_major") <= pl.lit(-strong_cut))
        .then(pl.lit("divergent"))
        .otherwise(pl.lit("unstable"))
        .alias("micro_corr_regime"),
    ).with_columns(
        pl.when(pl.col("micro_corr_regime") == pl.lit("aligned"))
        .then(pl.lit("cluster_pos"))
        .when(pl.col("micro_corr_regime") == pl.lit("divergent"))
        .then(pl.lit("cluster_neg"))
        .otherwise(pl.lit("cluster_neutral"))
        .alias("corr_cluster_id"),
    )

    out = df.select(
        "instrument",
        "ts",
        "corr_dxy",
        "corr_index_major",
        "corr_oil",
        "micro_corr_regime",
        "corr_cluster_id",
    )

    log.info("corr_micro: built rows=%d window=%d", out.height, window)
    return out
