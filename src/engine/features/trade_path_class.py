from __future__ import annotations

import logging

import polars as pl

from engine.features import FeatureBuildContext

log = logging.getLogger(__name__)


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_id": pl.Series([], dtype=pl.Utf8),
            "path_shape": pl.Series([], dtype=pl.Utf8),
            "path_cluster_id": pl.Series([], dtype=pl.Utf8),
            "path_family_id": pl.Series([], dtype=pl.Utf8),
            "path_filter_primary": pl.Series([], dtype=pl.Utf8),
            "path_filter_tags_json": pl.Series([], dtype=pl.Utf8),
            "time_to_1R_bars": pl.Series([], dtype=pl.Int64),
            "time_to_2R_bars": pl.Series([], dtype=pl.Int64),
            "mae_R": pl.Series([], dtype=pl.Float64),
            "mae_R_bucket": pl.Series([], dtype=pl.Utf8),
            "mfe_R": pl.Series([], dtype=pl.Float64),
            "exit_reason": pl.Series([], dtype=pl.Utf8),
        }
    )


def build_feature_frame(
    ctx: FeatureBuildContext,
    candles: pl.DataFrame | None = None,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    trade_paths: pl.DataFrame | None = None,
    **_,
) -> pl.DataFrame:
    """
    Trade-path classification features derived from existing trade_paths rows.

    Table: data/trade_paths
    Keys : trade_id
    """
    df = trade_paths or external
    if df is None or df.is_empty():
        log.warning("trade_path_class: trade_paths empty; returning empty keyed frame")
        return _empty_keyed_frame()

    if "trade_id" not in df.columns:
        log.warning("trade_path_class: missing trade_id column; returning empty keyed frame")
        return _empty_keyed_frame()

    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg = dict(auto_cfg.get("trade_path_class", {}) if isinstance(auto_cfg, dict) else {})
    n_clusters = max(2, int(fam_cfg.get("path_cluster_n_clusters", 8)))

    base = df.select(pl.col("trade_id").cast(pl.Utf8))

    realised_r = pl.col("realised_R").cast(pl.Float64) if "realised_R" in df.columns else pl.lit(None).cast(pl.Float64)
    mae_r = pl.col("mae_R").cast(pl.Float64) if "mae_R" in df.columns else pl.lit(None).cast(pl.Float64)

    path_shape = (
        pl.when(realised_r.is_null())
        .then(pl.lit("unknown"))
        .when(realised_r >= pl.lit(1.0))
        .then(
            pl.when(mae_r <= pl.lit(0.5)).then(pl.lit("straight_runner")).otherwise(pl.lit("dip_then_go"))
        )
        .when(realised_r > pl.lit(0.0))
        .then(pl.lit("grind_then_go"))
        .otherwise(pl.lit("straight_fail"))
        .alias("path_shape")
    )

    cluster_id = (
        (pl.col("trade_id").hash(seed=0) % pl.lit(n_clusters))
        .cast(pl.Utf8)
        .alias("path_cluster_id")
    )

    time_to_1r = pl.col("time_to_1R_bars").cast(pl.Int64) if "time_to_1R_bars" in df.columns else pl.lit(None).cast(pl.Int64)
    time_to_2r = pl.col("time_to_2R_bars").cast(pl.Int64) if "time_to_2R_bars" in df.columns else pl.lit(None).cast(pl.Int64)
    mfe_r = pl.col("mfe_R").cast(pl.Float64) if "mfe_R" in df.columns else pl.lit(None).cast(pl.Float64)
    exit_reason = pl.col("exit_reason").cast(pl.Utf8) if "exit_reason" in df.columns else pl.lit(None).cast(pl.Utf8)

    mae_bucket = (
        pl.when(mae_r.is_null())
        .then(pl.lit("unknown"))
        .when(mae_r <= pl.lit(0.25))
        .then(pl.lit("tiny"))
        .when(mae_r <= pl.lit(0.5))
        .then(pl.lit("small"))
        .when(mae_r <= pl.lit(1.0))
        .then(pl.lit("medium"))
        .otherwise(pl.lit("large"))
        .alias("mae_R_bucket")
    )

    out = base.with_columns(
        path_shape,
        cluster_id,
        pl.lit("[]").alias("path_filter_tags_json"),
        time_to_1r.alias("time_to_1R_bars"),
        time_to_2r.alias("time_to_2R_bars"),
        mae_r.alias("mae_R"),
        mae_bucket,
        mfe_r.alias("mfe_R"),
        exit_reason.alias("exit_reason"),
    ).with_columns(
        pl.col("path_shape").alias("path_family_id"),
        pl.col("path_shape").alias("path_filter_primary"),
    )

    log.info("trade_path_class: built rows=%d clusters=%d", out.height, n_clusters)
    return out.select(_empty_keyed_frame().columns)
