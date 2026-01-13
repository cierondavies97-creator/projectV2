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
            "anchor_tf": pl.Series([], dtype=pl.Utf8),
            "pcr_window_ts": pl.Series([], dtype=pl.Datetime("us")),
            "pcr_tick_total_volume_units": pl.Series([], dtype=pl.Float64),
            "pcr_tick_delta_units": pl.Series([], dtype=pl.Float64),
            "pcr_tick_delta_z": pl.Series([], dtype=pl.Float64),
            "pcr_tick_imbalance_ratio": pl.Series([], dtype=pl.Float64),
            "pcr_tick_imbalance_flag": pl.Series([], dtype=pl.Boolean),
            "pcr_tick_concentration_ratio_at": pl.Series([], dtype=pl.Float64),
            "pcr_tick_concentration_flag_at": pl.Series([], dtype=pl.Boolean),
            "pcr_tick_sweep_flag": pl.Series([], dtype=pl.Boolean),
            "pcr_tick_absorption_flag": pl.Series([], dtype=pl.Boolean),
            "pcr_footprint_pattern": pl.Series([], dtype=pl.Utf8),
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
    Tick-level microstructure features using rolling tick aggregates.

    Table: data/pcr_a
    Keys : instrument, anchor_tf, pcr_window_ts
    """
    if ticks is None or ticks.is_empty():
        log.warning("pcra_tick: ticks empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "ts", "price"}
    missing = sorted(required - set(ticks.columns))
    if missing:
        log.warning("pcra_tick: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg = dict(auto_cfg.get("pcra_tick", {}) if isinstance(auto_cfg, dict) else {})

    imbalance_cut = float(fam_cfg.get("tick_imbalance_ratio_threshold", 0.6))
    concentration_cut = float(fam_cfg.get("tick_concentration_ratio_at_threshold", 2.0))
    sweep_range_cut = float(fam_cfg.get("tick_sweep_range_ticks_threshold", 3.0))
    absorption_vol_cut = float(fam_cfg.get("tick_absorption_volume_threshold", 2.0))
    absorption_range_cut = float(fam_cfg.get("tick_absorption_range_ticks_threshold", 1.0))
    delta_z_strong = float(fam_cfg.get("tick_delta_z_strong", 2.0))

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    anchor_tf = str(anchor_tfs[0]) if anchor_tfs else "tick"

    df = ticks.select(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
        pl.col("price").cast(pl.Float64),
        pl.col("size").cast(pl.Float64).alias("_size") if "size" in ticks.columns else pl.lit(1.0).alias("_size"),
        pl.col("side").cast(pl.Utf8).alias("_side") if "side" in ticks.columns else pl.lit(None).cast(pl.Utf8).alias("_side"),
    ).drop_nulls(["instrument", "ts"]).sort(["instrument", "ts"])

    if df.is_empty():
        return _empty_keyed_frame()

    side_dir = (
        pl.when(pl.col("_side").is_in(["buy", "bid", "long"]))
        .then(pl.lit(1.0))
        .when(pl.col("_side").is_in(["sell", "ask", "short"]))
        .then(pl.lit(-1.0))
        .otherwise(
            pl.when(pl.col("price") >= pl.col("price").shift(1).over("instrument"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(-1.0))
        )
        .alias("_dir")
    )

    df = df.with_columns(
        side_dir,
        (pl.col("_size") * pl.col("_dir")).alias("_delta"),
    )

    by = ["instrument"]
    window = 50
    total_vol = pl.col("_size").rolling_sum(window_size=window, min_periods=window).over(by).alias("pcr_tick_total_volume_units")
    delta_sum = pl.col("_delta").rolling_sum(window_size=window, min_periods=window).over(by).alias("pcr_tick_delta_units")

    delta_mean = pl.col("_delta").rolling_mean(window_size=window, min_periods=window).over(by)
    delta_std = pl.col("_delta").rolling_std(window_size=window, min_periods=window).over(by)
    delta_z = safe_div(pl.col("_delta") - delta_mean, delta_std, default=0.0).alias("pcr_tick_delta_z")

    imbalance_ratio = safe_div(delta_sum.abs(), total_vol, default=0.0).alias("pcr_tick_imbalance_ratio")
    concentration_ratio = safe_div(pl.col("_size"), pl.col("_size").rolling_mean(window_size=window, min_periods=window).over(by), default=0.0).alias("pcr_tick_concentration_ratio_at")

    df = df.with_columns(
        total_vol,
        delta_sum,
        delta_z,
        imbalance_ratio,
        concentration_ratio,
        (pl.col("pcr_tick_imbalance_ratio") >= pl.lit(imbalance_cut)).alias("pcr_tick_imbalance_flag"),
        (pl.col("pcr_tick_concentration_ratio_at") >= pl.lit(concentration_cut)).alias("pcr_tick_concentration_flag_at"),
    )

    price_range = (pl.col("price") - pl.col("price").shift(1).over(by)).abs().alias("_price_range")
    df = df.with_columns(price_range)

    df = df.with_columns(
        (pl.col("_price_range") >= pl.lit(sweep_range_cut)).alias("pcr_tick_sweep_flag"),
        ((pl.col("_size") >= pl.lit(absorption_vol_cut)) & (pl.col("_price_range") <= pl.lit(absorption_range_cut))).alias("pcr_tick_absorption_flag"),
        pl.when(pl.col("pcr_tick_delta_z") >= pl.lit(delta_z_strong))
        .then(pl.lit("aggressive_buy"))
        .when(pl.col("pcr_tick_delta_z") <= pl.lit(-delta_z_strong))
        .then(pl.lit("aggressive_sell"))
        .otherwise(pl.lit("balanced"))
        .alias("pcr_footprint_pattern"),
    )

    out = df.select(
        pl.col("instrument"),
        pl.lit(anchor_tf).alias("anchor_tf"),
        pl.col("ts").alias("pcr_window_ts"),
        "pcr_tick_total_volume_units",
        "pcr_tick_delta_units",
        "pcr_tick_delta_z",
        "pcr_tick_imbalance_ratio",
        "pcr_tick_imbalance_flag",
        "pcr_tick_concentration_ratio_at",
        "pcr_tick_concentration_flag_at",
        "pcr_tick_sweep_flag",
        "pcr_tick_absorption_flag",
        "pcr_footprint_pattern",
    )

    log.info("pcra_tick: built rows=%d", out.height)
    return out
