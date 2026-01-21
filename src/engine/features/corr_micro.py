from __future__ import annotations

import logging

import polars as pl

from collections.abc import Mapping

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
    cfg = dict(auto_cfg.get("corr_micro", {}) if isinstance(auto_cfg, Mapping) else {})
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
    Micro correlation features using rolling correlation vs a cluster mean return.

    Table: data/features_corr
    Keys : instrument, ts
    """
    if candles is None or candles.is_empty():
        log.warning("corr_micro: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    required = {"instrument", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("corr_micro: missing columns=%s; returning null-filled frame", missing)
        base = _base_keys_from_candles(candles)
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        out = conform_to_registry(
            base,
            registry_entry=registry_entry,
            key_cols=["instrument", "ts"],
            where="corr_micro",
            allow_extra=False,
        )
        return out

    fam_cfg = _merge_cfg(ctx, family_cfg)

    window = max(5, int(fam_cfg.get("corr_window_bars", 50)))
    min_periods = int(fam_cfg.get("corr_min_periods", max(10, window // 2)))
    strong_cut = float(fam_cfg.get("corr_strong_cut", 0.5))
    clip_abs = float(fam_cfg.get("corr_clip_abs", 0.999))
    stability_window = max(5, int(fam_cfg.get("corr_stability_window_bars", 200)))
    flip_cut = float(fam_cfg.get("corr_flip_abs_delta_cut", 0.5))
    unstable_cut = float(fam_cfg.get("corr_unstable_std_cut", 0.25))

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
        return _empty_keyed_frame(registry_entry)

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
    corr_raw = safe_div(cov, std_ret * std_mkt, default=None)
    corr = corr_raw.clip(-clip_abs, clip_abs).alias("corr_index_major")

    df = c.with_columns(cov, corr)
    df = df.with_columns(
        pl.col("corr_index_major").alias("corr_dxy"),
        pl.col("corr_index_major").alias("corr_oil"),
        pl.lit("MARKET").alias("corr_ref1_id"),
        pl.col("corr_index_major").alias("corr_ref1"),
        pl.lit(None).cast(pl.Utf8).alias("corr_ref2_id"),
        pl.lit(None).cast(pl.Float64).alias("corr_ref2"),
        pl.lit(None).cast(pl.Utf8).alias("corr_ref3_id"),
        pl.lit(None).cast(pl.Float64).alias("corr_ref3"),
    )
    df = df.with_columns(
        pl.when(pl.col("corr_ref1").is_null())
        .then(pl.lit(None).cast(pl.Utf8))
        .when(pl.col("corr_ref1") >= 0)
        .then(pl.lit("pos"))
        .otherwise(pl.lit("neg"))
        .alias("micro_corr_sign"),
        pl.col("corr_ref1").abs().clip(0.0, 1.0).alias("micro_corr_strength"),
    )
    corr_count = (
        pl.col("_ret")
        .rolling_count(window_size=window, min_periods=1)
        .over("instrument")
        .alias("_corr_count")
    )
    df = df.with_columns(corr_count)
    df = df.with_columns(
        pl.when(pl.col("_corr_count") >= pl.lit(min_periods))
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .alias("micro_corr_confidence")
    )
    corr_std = (
        pl.col("corr_ref1")
        .rolling_std(window_size=stability_window, min_periods=min_periods)
        .over("instrument")
        .alias("micro_corr_std")
    )
    df = df.with_columns(corr_std)
    df = df.with_columns(
        (pl.col("micro_corr_std") >= pl.lit(unstable_cut)).alias("micro_corr_unstable_flag"),
        (
            (pl.col("corr_ref1") * pl.col("corr_ref1").shift(1).over("instrument") < 0)
            | ((pl.col("corr_ref1") - pl.col("corr_ref1").shift(1).over("instrument")).abs() >= pl.lit(flip_cut))
        ).alias("micro_corr_flip_flag"),
    )
    df = df.with_columns(
        pl.when(pl.col("corr_ref1") >= pl.lit(strong_cut))
        .then(pl.lit("aligned"))
        .when(pl.col("corr_ref1") <= pl.lit(-strong_cut))
        .then(pl.lit("divergent"))
        .otherwise(pl.lit("unstable"))
        .alias("micro_corr_regime")
    )
    df = df.with_columns(
        pl.when(pl.col("micro_corr_regime") == pl.lit("aligned"))
        .then(pl.lit("cluster_pos"))
        .when(pl.col("micro_corr_regime") == pl.lit("divergent"))
        .then(pl.lit("cluster_neg"))
        .otherwise(pl.lit("cluster_neutral"))
        .alias("corr_cluster_id"),
        pl.col("corr_ref1").abs().alias("corr_ref1_xcorr_max"),
        pl.lit(0).cast(pl.Int64).alias("corr_ref1_xcorr_lag_bars"),
        pl.lit(None).cast(pl.Float64).alias("corr_vol_ref1"),
        pl.lit(None).cast(pl.Float64).alias("corr_vol_ref2"),
        pl.lit(None).cast(pl.Float64).alias("corr_vol_ref3"),
        pl.lit(None).cast(pl.Utf8).alias("corr_topk_neighbors_json"),
    )

    out = df.select(
        "instrument",
        "ts",
        "corr_dxy",
        "corr_index_major",
        "corr_oil",
        "micro_corr_regime",
        "corr_cluster_id",
        "corr_ref1_id",
        "corr_ref1",
        "corr_ref2_id",
        "corr_ref2",
        "corr_ref3_id",
        "corr_ref3",
        "micro_corr_sign",
        "micro_corr_strength",
        "micro_corr_confidence",
        "micro_corr_std",
        "micro_corr_unstable_flag",
        "micro_corr_flip_flag",
        "corr_ref1_xcorr_max",
        "corr_ref1_xcorr_lag_bars",
        "corr_vol_ref1",
        "corr_vol_ref2",
        "corr_vol_ref3",
        "corr_topk_neighbors_json",
    )

    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "ts"],
        where="corr_micro",
        allow_extra=False,
    )

    log.info("corr_micro: built rows=%d window=%d", out.height, window)
    return out
