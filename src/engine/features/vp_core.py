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

    Deterministic Phase-A proxy:
      - POC price proxy = bar mid (high+low)/2
      - POC relative position = (mid-low)/(high-low) in [0,1]
      - Total volume = bar volume (or 0 if absent)
      - Upper/Lower volume proxy = 50/50 split (kept simple)
      - Skew proxy = (close-mid)/(high-low)
      - HVN/LVN proxy = compare total volume to rolling mean volume
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

    # Optional hardening knobs (safe defaults if not in registry yet)
    mean_vol_window = int(fam_cfg.get("vp_mean_volume_window_bars", 50))
    mean_vol_min_periods = int(fam_cfg.get("vp_mean_volume_min_periods", 10))
    mean_vol_window = max(1, mean_vol_window)
    mean_vol_min_periods = max(1, min(mean_vol_window, mean_vol_min_periods))

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
            (pl.col("volume").cast(pl.Float64) if "volume" in candles.columns else pl.lit(0.0, dtype=pl.Float64)).alias("_volume"),
        )
        .drop_nulls(["instrument", "tf", "ts"])
        .sort(["instrument", "tf", "ts"])
        .unique(subset=["instrument", "tf", "ts"], keep="last")
    )

    eps = 1e-12

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

        df = df.with_columns(
            ((pl.col("high") + pl.col("low")) / 2.0).alias("_mid"),
            (pl.col("high") - pl.col("low")).alias("_range"),
        )

        # IMPORTANT: rel_pos uses _mid directly; do NOT reference zone_vp_poc_price before it exists.
        df = df.with_columns(
            pl.col("_volume").alias("zone_vp_total_volume_units"),
            pl.col("_mid").alias("zone_vp_poc_price"),
            pl.when(pl.col("_range") == 0)
            .then(pl.lit(0.5))
            .otherwise(((pl.col("_mid") - pl.col("low")) / pl.max_horizontal(pl.col("_range"), pl.lit(eps))).clip(0.0, 1.0))
            .alias("zone_vp_poc_relative_position"),
            (pl.col("_volume") * 0.5).alias("zone_vp_volume_lower_units"),
            (pl.col("_volume") * 0.5).alias("zone_vp_volume_upper_units"),
            pl.when(pl.col("_range") == 0)
            .then(pl.lit(0.0))
            .otherwise((pl.col("close") - pl.col("_mid")) / pl.max_horizontal(pl.col("_range"), pl.lit(eps)))
            .alias("zone_vp_skew"),
        )

        mean_vol = (
            pl.col("zone_vp_total_volume_units")
            .rolling_mean(window_size=mean_vol_window, min_periods=mean_vol_min_periods)
            .over("instrument")
        )

        df = df.with_columns(
            (pl.col("zone_vp_total_volume_units") >= mean_vol * pl.lit(hvn_lvn_ratio_cut))
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("zone_vp_hvn_flag"),
            (pl.col("zone_vp_total_volume_units") <= mean_vol / pl.lit(hvn_lvn_ratio_cut))
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("zone_vp_lvn_flag"),
        )

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
            df.select(_empty_keyed_frame().columns)
        )

    if not out_frames:
        return _empty_keyed_frame()

    out = pl.concat(out_frames, how="vertical").sort(["instrument", "anchor_tf", "zone_id", "ts"])
    log.info("vp_core: built rows=%d", out.height)
    return out
