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
            "anchor_tf": pl.Series([], dtype=pl.Utf8),
            "anchor_ts": pl.Series([], dtype=pl.Datetime("us")),
            "unsup_regime_id": pl.Series([], dtype=pl.Utf8),
            "unsup_regime_confidence": pl.Series([], dtype=pl.Float64),
        }
    )


def _base_keys_from_candles(candles: pl.DataFrame, anchor_tfs: list[str]) -> pl.DataFrame:
    if candles is None or candles.is_empty():
        return pl.DataFrame()
    if not {"instrument", "tf", "ts"}.issubset(set(candles.columns)):
        return pl.DataFrame()
    df = candles.select(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("tf").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
    ).drop_nulls(["instrument", "tf", "ts"])
    if anchor_tfs:
        df = df.filter(pl.col("tf").is_in(anchor_tfs))
    return (
        df.select(
            pl.col("instrument"),
            pl.col("tf").alias("anchor_tf"),
            pl.col("ts").alias("anchor_ts"),
        )
        .unique()
        .sort(["instrument", "anchor_tf", "anchor_ts"])
    )


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg: dict[str, object] = {}
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("unsup_regime", {}) or {})
    if isinstance(family_cfg, Mapping):
        cfg.update(family_cfg)
    return cfg


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Unsupervised regime proxy using rolling volatility buckets.

    Table: data/windows
    Keys : instrument, anchor_tf, anchor_ts
    """
    if candles is None or candles.is_empty():
        log.warning("unsup_regime: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    required = {"instrument", "tf", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("unsup_regime: missing columns=%s; returning null-filled frame", missing)
        base = _base_keys_from_candles(candles, [])
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        out = base.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("unsup_regime_id"),
            pl.lit(None).cast(pl.Float64).alias("unsup_regime_confidence"),
            pl.lit(None).cast(pl.Utf8).alias("unsup_method_used"),
            pl.lit(None).cast(pl.Utf8).alias("unsup_feature_set_used"),
            pl.lit(None).cast(pl.Utf8).alias("unsup_model_id"),
            pl.lit(None).cast(pl.Datetime("us")).alias("unsup_fit_ts"),
            pl.lit(None).cast(pl.Float64).alias("unsup_distance_to_centroid"),
            pl.lit(None).cast(pl.Float64).alias("unsup_logprob"),
            pl.lit(None).cast(pl.Float64).alias("unsup_entropy"),
            pl.lit(None).cast(pl.Boolean).alias("unsup_outlier_flag"),
        )
        return conform_to_registry(
            out,
            registry_entry=registry_entry,
            key_cols=["instrument", "anchor_tf", "anchor_ts"],
            where="unsup_regime",
            allow_extra=False,
        )

    fam_cfg = _merge_cfg(ctx, family_cfg)

    n_clusters = max(2, int(fam_cfg.get("unsup_n_clusters", 8)))
    min_confidence = float(fam_cfg.get("unsup_min_confidence", 0.55))
    clip_abs = float(fam_cfg.get("unsup_clip_abs", 8.0))
    conf_temp = float(fam_cfg.get("unsup_conf_softmax_temp", 1.0))
    entropy_high_cut = float(fam_cfg.get("unsup_entropy_high_cut", 1.5))
    z_window = 200
    vol_window = 20

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs and "tf" in candles.columns:
        anchor_tfs = (
            candles.select(pl.col("tf").cast(pl.Utf8))
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
    if not anchor_tfs:
        log.warning("unsup_regime: no anchor_tfs resolved; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    c = (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("tf").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col("close").cast(pl.Float64),
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

        df = df.sort(["instrument", "ts"]).with_columns(pl.lit(tf_str).alias("anchor_tf"))
        df = df.with_columns(
            pl.col("close").pct_change().over("instrument").alias("_ret"),
        )

        by = ["instrument", "anchor_tf"]
        vol = pl.col("_ret").rolling_std(window_size=vol_window, min_periods=vol_window).over(by).alias("_vol")
        vol_mean = pl.col("_vol").rolling_mean(window_size=z_window, min_periods=z_window).over(by)
        vol_std = pl.col("_vol").rolling_std(window_size=z_window, min_periods=z_window).over(by)

        df = df.with_columns(vol).with_columns(
            safe_div(pl.col("_vol") - vol_mean, vol_std, default=0.0).alias("_vol_z"),
        )
        df = df.with_columns(pl.col("_vol_z").clip(-clip_abs, clip_abs).alias("_vol_z"))

        rank = pl.col("_vol").rank("average").over(by)
        count = pl.count().over(by)
        pct = safe_div(rank - 1, count - 1, default=0.0).alias("_pct")

        df = df.with_columns(
            pct,
            (pl.col("_pct") * pl.lit(n_clusters)).floor().cast(pl.Int64).alias("_bucket"),
        ).with_columns(
            pl.when(pl.col("_bucket") >= pl.lit(n_clusters))
            .then(pl.lit(n_clusters - 1))
            .otherwise(pl.col("_bucket"))
            .alias("_bucket"),
        )

        df = df.with_columns(
            pl.col("_bucket").cast(pl.Utf8).alias("unsup_regime_id"),
            pl.col("_vol_z").abs().clip(0.0, 3.0).alias("unsup_regime_confidence"),
            pl.lit(str(fam_cfg.get("unsup_method", "kmeans"))).alias("unsup_method_used"),
            pl.lit(str(fam_cfg.get("unsup_feature_set_id", "windows_core_v1"))).alias("unsup_feature_set_used"),
            pl.lit(None).cast(pl.Utf8).alias("unsup_model_id"),
            pl.lit(None).cast(pl.Datetime("us")).alias("unsup_fit_ts"),
            pl.lit(None).cast(pl.Float64).alias("unsup_distance_to_centroid"),
            pl.lit(None).cast(pl.Float64).alias("unsup_logprob"),
            pl.lit(None).cast(pl.Float64).alias("unsup_entropy"),
            pl.lit(None).cast(pl.Boolean).alias("unsup_outlier_flag"),
        )

        out_frames.append(
            df.select(
                pl.col("instrument"),
                pl.col("anchor_tf"),
                pl.col("ts").alias("anchor_ts"),
                "unsup_regime_id",
                "unsup_regime_confidence",
                "unsup_method_used",
                "unsup_feature_set_used",
                "unsup_model_id",
                "unsup_fit_ts",
                "unsup_distance_to_centroid",
                "unsup_logprob",
                "unsup_entropy",
                "unsup_outlier_flag",
            )
        )

    if not out_frames:
        base = _base_keys_from_candles(candles, [str(tf) for tf in anchor_tfs])
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        out = base.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("unsup_regime_id"),
            pl.lit(None).cast(pl.Float64).alias("unsup_regime_confidence"),
            pl.lit(str(fam_cfg.get("unsup_method", "kmeans"))).alias("unsup_method_used"),
            pl.lit(str(fam_cfg.get("unsup_feature_set_id", "windows_core_v1"))).alias("unsup_feature_set_used"),
            pl.lit(None).cast(pl.Utf8).alias("unsup_model_id"),
            pl.lit(None).cast(pl.Datetime("us")).alias("unsup_fit_ts"),
            pl.lit(None).cast(pl.Float64).alias("unsup_distance_to_centroid"),
            pl.lit(None).cast(pl.Float64).alias("unsup_logprob"),
            pl.lit(None).cast(pl.Float64).alias("unsup_entropy"),
            pl.lit(None).cast(pl.Boolean).alias("unsup_outlier_flag"),
        )
        return conform_to_registry(
            out,
            registry_entry=registry_entry,
            key_cols=["instrument", "anchor_tf", "anchor_ts"],
            where="unsup_regime",
            allow_extra=False,
        )

    out = pl.concat(out_frames, how="vertical").sort(["instrument", "anchor_tf", "anchor_ts"])
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "anchor_ts"],
        where="unsup_regime",
        allow_extra=False,
    )

    log.info("unsup_regime: built rows=%d clusters=%d", out.height, n_clusters)
    return out
