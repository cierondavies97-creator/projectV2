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
            "anchor_ts": pl.Series([], dtype=pl.Datetime("us")),
            "vol_range_lookback_bars": pl.Series([], dtype=pl.Int64),
            "vol_range_realised_range": pl.Series([], dtype=pl.Float64),
            "vol_range_range_z": pl.Series([], dtype=pl.Float64),
            "vol_range_state": pl.Series([], dtype=pl.Utf8),
            "vol_range_compress_flag": pl.Series([], dtype=pl.Boolean),
            "vol_range_expand_flag": pl.Series([], dtype=pl.Boolean),
            "vol_range_regime_id": pl.Series([], dtype=pl.Utf8),
            "vol_range_regime_confidence": pl.Series([], dtype=pl.Float64),
        }
    )


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame | None = None,
    windows: pl.DataFrame | None = None,
    **_,
) -> pl.DataFrame:
    """
    Table: data/windows
    Keys : instrument, anchor_tf, anchor_ts

    Outputs (must match features_registry.yaml vol_range columns):
      - vol_range_lookback_bars
      - vol_range_realised_range
      - vol_range_range_z
      - vol_range_state
      - vol_range_compress_flag
      - vol_range_expand_flag
      - vol_range_regime_id
      - vol_range_regime_confidence
    """
    if candles is None or candles.is_empty():
        log.warning("vol_range: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "tf", "ts", "high", "low"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("vol_range: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("vol_range: ctx.cluster.anchor_tfs empty; returning empty keyed frame")
        return _empty_keyed_frame()

    base_keys = None
    if windows is not None and not windows.is_empty():
        required_win = {"instrument", "anchor_tf", "anchor_ts"}
        if required_win.issubset(set(windows.columns)):
            base_keys = (
                windows.select(
                    pl.col("instrument").cast(pl.Utf8),
                    pl.col("anchor_tf").cast(pl.Utf8),
                    pl.col("anchor_ts").cast(pl.Datetime("us")),
                )
                .drop_nulls(["instrument", "anchor_tf", "anchor_ts"])
                .unique()
            )

    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg = dict(auto_cfg.get("vol_range", {}) if isinstance(auto_cfg, dict) else {})

    lookback = max(2, int(fam_cfg.get("vol_range_lookback_bars", 200)))
    z_window = max(2, int(fam_cfg.get("vol_range_zscore_window_bars", 200)))
    compress_cut = float(fam_cfg.get("vol_range_compress_z_cut", -1.0))
    expand_cut = float(fam_cfg.get("vol_range_expand_z_cut", 1.0))
    conf_cut = float(fam_cfg.get("vol_range_regime_conf_cut", 0.55))

    max_cut = max(abs(compress_cut), abs(expand_cut), 1e-9)

    c = (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("tf").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
        )
        .drop_nulls(["instrument", "tf", "ts"])
        .sort(["instrument", "tf", "ts"])
        .unique(subset=["instrument", "tf", "ts"], keep="last")
    )

    out_frames: list[pl.DataFrame] = []
    for anchor_tf in anchor_tfs:
        tf_str = str(anchor_tf)
        df = c.filter(pl.col("tf") == pl.lit(tf_str))
        if df.is_empty():
            continue

        df = df.sort(["instrument", "ts"]).with_columns(pl.lit(tf_str).alias("anchor_tf"))
        df = df.with_columns((pl.col("high") - pl.col("low")).alias("_range"))

        by = ["instrument", "anchor_tf"]
        df = df.with_columns(
            pl.col("_range")
            .rolling_mean(window_size=lookback, min_periods=lookback)
            .over(by)
            .alias("vol_range_realised_range"),
        )

        range_mean = pl.col("_range").rolling_mean(window_size=z_window, min_periods=z_window).over(by)
        range_std = pl.col("_range").rolling_std(window_size=z_window, min_periods=z_window).over(by)
        df = df.with_columns(
            safe_div(pl.col("_range") - range_mean, range_std, default=0.0).alias("vol_range_range_z"),
        )

        df = df.with_columns(
            pl.lit(lookback).cast(pl.Int64).alias("vol_range_lookback_bars"),
            (pl.col("vol_range_range_z") <= pl.lit(compress_cut)).alias("vol_range_compress_flag"),
            (pl.col("vol_range_range_z") >= pl.lit(expand_cut)).alias("vol_range_expand_flag"),
        )

        df = df.with_columns(
            pl.when(pl.col("vol_range_compress_flag"))
            .then(pl.lit("compress"))
            .when(pl.col("vol_range_expand_flag"))
            .then(pl.lit("expand"))
            .otherwise(pl.lit("neutral"))
            .alias("vol_range_state"),
        )

        df = df.with_columns(
            pl.col("vol_range_state").alias("vol_range_regime_id"),
            pl.when(pl.col("vol_range_range_z").abs() < pl.lit(conf_cut))
            .then(pl.lit(0.0))
            .otherwise((pl.col("vol_range_range_z").abs() / pl.lit(max_cut)).clip(0.0, 1.0))
            .alias("vol_range_regime_confidence"),
        )

        out_frames.append(
            df.select(
                pl.col("instrument"),
                pl.col("anchor_tf"),
                pl.col("ts").alias("anchor_ts"),
                "vol_range_lookback_bars",
                "vol_range_realised_range",
                "vol_range_range_z",
                "vol_range_state",
                "vol_range_compress_flag",
                "vol_range_expand_flag",
                "vol_range_regime_id",
                "vol_range_regime_confidence",
            )
        )

    if not out_frames:
        return _empty_keyed_frame()

    out_all = pl.concat(out_frames, how="vertical").sort(["instrument", "anchor_tf", "anchor_ts"])

    if base_keys is not None and not base_keys.is_empty():
        out_all = base_keys.join(out_all, on=["instrument", "anchor_tf", "anchor_ts"], how="left")

    log.info(
        "vol_range: built rows=%d instruments=%d anchor_tfs=%d lookback=%d z_window=%d",
        out_all.height,
        out_all.select("instrument").n_unique(),
        out_all.select("anchor_tf").n_unique(),
        lookback,
        z_window,
    )
    return out_all.select(_empty_keyed_frame().columns)
