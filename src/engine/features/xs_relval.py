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
            "xs_relval_spread_level": pl.Series([], dtype=pl.Float64),
            "xs_relval_spread_zscore": pl.Series([], dtype=pl.Float64),
            "xs_relval_carry_rank": pl.Series([], dtype=pl.Float64),
            "xs_relval_momo_rank": pl.Series([], dtype=pl.Float64),
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
    cfg = dict(auto_cfg.get("xs_relval", {}) if isinstance(auto_cfg, Mapping) else {})
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
    Cross-sectional relative value features from per-bar returns.

    Table: data/features_corr
    Keys : instrument, ts
    """
    if candles is None or candles.is_empty():
        log.warning("xs_relval: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    required = {"instrument", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("xs_relval: missing columns=%s; returning null-filled frame", missing)
        base = _base_keys_from_candles(candles)
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        return conform_to_registry(
            base,
            registry_entry=registry_entry,
            key_cols=["instrument", "ts"],
            where="xs_relval",
            allow_extra=False,
        )

    cfg = _merge_cfg(ctx, family_cfg)
    spread_type = str(cfg.get("xs_relval_spread_type", "log_ratio"))
    z_window = max(5, int(cfg.get("xs_relval_spread_z_window", 60)))
    z_clip = float(cfg.get("xs_relval_z_clip_abs", 8.0))
    mild_cut = float(cfg.get("xs_relval_bucket_mild_cut", 1.5))
    strong_cut = float(cfg.get("xs_relval_bucket_strong_cut", 2.5))
    peer_set_used = str(cfg.get("xs_relval_peer_set_id", "cluster_peers"))

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

    by_ts = ["ts"]
    mean_ret = pl.col("_ret").mean().over(by_ts)
    mean_close = pl.col("close").mean().over(by_ts)
    mean_log_close = pl.col("close").log().mean().over(by_ts)
    if spread_type == "price_ratio":
        spread_level = (safe_div(pl.col("close"), mean_close, default=None) - pl.lit(1.0)).alias(
            "xs_relval_spread_level"
        )
    elif spread_type == "log_ratio":
        spread_level = (pl.col("close").log() - mean_log_close).alias("xs_relval_spread_level")
    else:
        spread_level = (pl.col("_ret") - mean_ret).alias("xs_relval_spread_level")

    ret_rank = pl.col("_ret").rank("average").over(by_ts)
    count = pl.count().over(by_ts)
    carry_rank = safe_div(ret_rank - 1, count - 1, default=0.0).alias("xs_relval_carry_rank")

    momo = pl.col("_ret").rolling_mean(window_size=5, min_periods=2).over("instrument").alias("_momo")
    momo_rank = safe_div(pl.col("_momo").rank("average").over(by_ts) - 1, count - 1, default=0.0).alias("xs_relval_momo_rank")

    out = c.with_columns(spread_level, momo, carry_rank, momo_rank)
    spread_mean = (
        pl.col("xs_relval_spread_level")
        .rolling_mean(window_size=z_window, min_periods=max(3, z_window // 3))
        .over("instrument")
    )
    spread_std = (
        pl.col("xs_relval_spread_level")
        .rolling_std(window_size=z_window, min_periods=max(3, z_window // 3))
        .over("instrument")
    )
    spread_z = safe_div(
        pl.col("xs_relval_spread_level") - spread_mean,
        spread_std,
        default=0.0,
    ).clip(-z_clip, z_clip).alias("xs_relval_spread_zscore")
    out = out.with_columns(spread_z)

    out = out.with_columns(
        pl.when(pl.col("xs_relval_spread_zscore") <= pl.lit(-strong_cut))
        .then(pl.lit("far_undervalued"))
        .when(pl.col("xs_relval_spread_zscore") <= pl.lit(-mild_cut))
        .then(pl.lit("undervalued"))
        .when(pl.col("xs_relval_spread_zscore") >= pl.lit(strong_cut))
        .then(pl.lit("far_overvalued"))
        .when(pl.col("xs_relval_spread_zscore") >= pl.lit(mild_cut))
        .then(pl.lit("overvalued"))
        .otherwise(pl.lit("fair"))
        .alias("xs_relval_spread_bucket"),
        pl.when(pl.col("xs_relval_spread_zscore") <= pl.lit(-mild_cut))
        .then(pl.lit("long"))
        .when(pl.col("xs_relval_spread_zscore") >= pl.lit(mild_cut))
        .then(pl.lit("short"))
        .otherwise(pl.lit("neutral"))
        .alias("xs_relval_signal"),
        pl.when(pl.lit(strong_cut) > 0)
        .then((pl.col("xs_relval_spread_zscore").abs() / pl.lit(strong_cut)).clip(0.0, 1.0))
        .otherwise(pl.lit(0.0))
        .alias("xs_relval_signal_strength"),
        pl.count().over(by_ts).cast(pl.Int64).alias("xs_relval_peer_count"),
        pl.lit(peer_set_used).alias("xs_relval_peer_set_used"),
        pl.lit(None).cast(pl.Utf8).alias("xs_relval_primary_peer"),
        pl.lit(None).cast(pl.Float64).alias("xs_relval_primary_peer_corr"),
        pl.lit(None).cast(pl.Float64).alias("xs_relval_primary_peer_beta"),
        pl.lit("neutral").alias("xs_relval_primary_peer_role"),
        pl.lit(None).cast(pl.Float64).alias("xs_relval_coint_pvalue"),
        pl.lit(None).cast(pl.Int64).alias("xs_relval_half_life_bars"),
        pl.lit(None).cast(pl.Float64).alias("xs_relval_residual_z"),
    ).select(
        "instrument",
        "ts",
        "xs_relval_spread_level",
        "xs_relval_spread_zscore",
        "xs_relval_carry_rank",
        "xs_relval_momo_rank",
        "xs_relval_spread_bucket",
        "xs_relval_signal",
        "xs_relval_signal_strength",
        "xs_relval_peer_count",
        "xs_relval_peer_set_used",
        "xs_relval_primary_peer",
        "xs_relval_primary_peer_corr",
        "xs_relval_primary_peer_beta",
        "xs_relval_primary_peer_role",
        "xs_relval_coint_pvalue",
        "xs_relval_half_life_bars",
        "xs_relval_residual_z",
    )

    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "ts"],
        where="xs_relval",
        allow_extra=False,
    )

    log.info("xs_relval: built rows=%d", out.height)
    return out
