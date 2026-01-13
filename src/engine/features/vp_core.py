from __future__ import annotations

import logging

import polars as pl

from engine.features import FeatureBuildContext

log = logging.getLogger(__name__)


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "anchor_tf": pl.Series([], dtype=pl.Utf8),
            "zone_id": pl.Series([], dtype=pl.Utf8),
            "ts": pl.Series([], dtype=pl.Datetime("us")),
            "zone_vp_total_volume_units": pl.Series([], dtype=pl.Float64),
            "zone_vp_poc_price": pl.Series([], dtype=pl.Float64),
            "zone_vp_poc_relative_position": pl.Series([], dtype=pl.Float64),
            "zone_vp_volume_lower_units": pl.Series([], dtype=pl.Float64),
            "zone_vp_volume_upper_units": pl.Series([], dtype=pl.Float64),
            "zone_vp_skew": pl.Series([], dtype=pl.Float64),
            "zone_vp_hvn_flag": pl.Series([], dtype=pl.Boolean),
            "zone_vp_lvn_flag": pl.Series([], dtype=pl.Boolean),
            "zone_vp_type": pl.Series([], dtype=pl.Utf8),
            "zone_vp_type_bucket": pl.Series([], dtype=pl.Utf8),
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
    Volume-profile proxy features derived from bar data.

    Table: data/zones_state
    Keys : instrument, anchor_tf, zone_id, ts
    """
    if candles is None or candles.is_empty():
        log.warning("vp_core: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "tf", "ts", "high", "low", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("vp_core: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg = dict(auto_cfg.get("vp_core", {}) if isinstance(auto_cfg, dict) else {})

    hvn_lvn_ratio_cut = float(fam_cfg.get("vp_hvn_lvn_ratio_cut", 1.5))
    thin_vol_cut = float(fam_cfg.get("vp_thin_total_volume_cut_units", 0.0))
    skew_strong_cut = float(fam_cfg.get("vp_skew_abs_strong_cut", 0.35))

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("vp_core: ctx.cluster.anchor_tfs empty; returning empty keyed frame")
        return _empty_keyed_frame()

    c = (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("tf").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64).alias("_volume") if "volume" in candles.columns else pl.lit(0.0).alias("_volume"),
        )
        .drop_nulls(["instrument", "tf", "ts"])
        .sort(["instrument", "tf", "ts"])
    )

    out_frames: list[pl.DataFrame] = []
    for anchor_tf in anchor_tfs:
        tf_str = str(anchor_tf)
        df = c.filter(pl.col("tf") == pl.lit(tf_str))
        if df.is_empty():
            continue

        df = df.sort(["instrument", "ts"]).with_columns(
            pl.lit(tf_str).alias("anchor_tf"),
            pl.concat_str([pl.lit("vp_core"), pl.col("instrument"), pl.lit(tf_str)], separator="::").alias("zone_id"),
        )

        mid = ((pl.col("high") + pl.col("low")) / pl.lit(2.0)).alias("_mid")
        rng = (pl.col("high") - pl.col("low")).alias("_range")
        df = df.with_columns(mid, rng)

        total_vol = pl.col("_volume").alias("zone_vp_total_volume_units")
        poc_price = pl.col("_mid").alias("zone_vp_poc_price")
        rel_pos = pl.when(pl.col("_range") == 0).then(pl.lit(0.5)).otherwise((pl.col("zone_vp_poc_price") - pl.col("low")) / pl.col("_range")).alias("zone_vp_poc_relative_position")
        vol_lower = (pl.col("_volume") * 0.5).alias("zone_vp_volume_lower_units")
        vol_upper = (pl.col("_volume") * 0.5).alias("zone_vp_volume_upper_units")
        skew = pl.when(pl.col("_range") == 0).then(pl.lit(0.0)).otherwise((pl.col("close") - pl.col("_mid")) / pl.col("_range")).alias("zone_vp_skew")

        df = df.with_columns(total_vol, poc_price, rel_pos, vol_lower, vol_upper, skew)

        mean_vol = pl.col("zone_vp_total_volume_units").rolling_mean(window_size=50, min_periods=10).over("instrument")
        hvn_flag = (pl.col("zone_vp_total_volume_units") >= mean_vol * pl.lit(hvn_lvn_ratio_cut)).alias("zone_vp_hvn_flag")
        lvn_flag = (pl.col("zone_vp_total_volume_units") <= mean_vol / pl.lit(hvn_lvn_ratio_cut)).alias("zone_vp_lvn_flag")

        df = df.with_columns(hvn_flag, lvn_flag)

        df = df.with_columns(
            pl.when(pl.col("zone_vp_total_volume_units") <= pl.lit(thin_vol_cut))
            .then(pl.lit("thin"))
            .when(pl.col("zone_vp_skew") >= pl.lit(skew_strong_cut))
            .then(pl.lit("skewed_high"))
            .when(pl.col("zone_vp_skew") <= pl.lit(-skew_strong_cut))
            .then(pl.lit("skewed_low"))
            .when(pl.col("zone_vp_hvn_flag"))
            .then(pl.lit("balanced_hvn"))
            .when(pl.col("zone_vp_lvn_flag"))
            .then(pl.lit("rejection_lvn"))
            .otherwise(pl.lit("balanced_hvn"))
            .alias("zone_vp_type"),
        )

        df = df.with_columns(
            pl.col("zone_vp_type")
            .replace(
                {
                    "balanced_hvn": "hvn",
                    "rejection_lvn": "lvn",
                    "skewed_high": "skew_high",
                    "skewed_low": "skew_low",
                    "thin": "thin",
                },
                default="unknown",
            )
            .alias("zone_vp_type_bucket"),
        )

        out_frames.append(
            df.select(
                "instrument",
                "anchor_tf",
                "zone_id",
                "ts",
                "zone_vp_total_volume_units",
                "zone_vp_poc_price",
                "zone_vp_poc_relative_position",
                "zone_vp_volume_lower_units",
                "zone_vp_volume_upper_units",
                "zone_vp_skew",
                "zone_vp_hvn_flag",
                "zone_vp_lvn_flag",
                "zone_vp_type",
                "zone_vp_type_bucket",
            )
        )

    if not out_frames:
        return _empty_keyed_frame()

    out = pl.concat(out_frames, how="vertical").sort(["instrument", "anchor_tf", "zone_id", "ts"])
    log.info("vp_core: built rows=%d", out.height)
    return out
