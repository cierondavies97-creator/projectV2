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
            "ts": pl.Series([], dtype=pl.Datetime("us")),
            # Match features_registry.yaml column names for ta_trend:
            "rsi_value": pl.Series([], dtype=pl.Float64),
            "rsi_bucket": pl.Series([], dtype=pl.Utf8),
            "ema_fast": pl.Series([], dtype=pl.Float64),
            "ema_slow": pl.Series([], dtype=pl.Float64),
        }
    )


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    **_,
) -> pl.DataFrame:
    if candles is None or candles.is_empty():
        log.warning("ta_trend: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "tf", "ts", "open", "high", "low", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("ta_trend: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("ta_trend: ctx.cluster.anchor_tfs empty; returning empty keyed frame")
        return _empty_keyed_frame()

    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg = dict(auto_cfg.get("ta_trend", {}) if isinstance(auto_cfg, dict) else {})

    rsi_period = max(1, int(fam_cfg.get("rsi_period", 14)))
    rsi_oversold_cut = float(fam_cfg.get("rsi_oversold_cut", 30.0))
    rsi_overbought_cut = float(fam_cfg.get("rsi_overbought_cut", 70.0))
    ema_fast_period = max(1, int(fam_cfg.get("ema_fast_period", 20)))
    ema_slow_period = max(1, int(fam_cfg.get("ema_slow_period", 50)))

    out_frames: list[pl.DataFrame] = []

    for anchor_tf in anchor_tfs:
        df = candles.filter(pl.col("tf") == pl.lit(str(anchor_tf)))
        if df.is_empty():
            continue

        df = df.sort(["instrument", "ts"]).with_columns(pl.lit(str(anchor_tf)).alias("anchor_tf"))

        df = df.with_columns(
            pl.col("close").cast(pl.Float64).ewm_mean(span=ema_fast_period, adjust=False).alias("_ema_fast"),
            pl.col("close").cast(pl.Float64).ewm_mean(span=ema_slow_period, adjust=False).alias("_ema_slow"),
        )

        diff = pl.col("close").cast(pl.Float64).diff().over("instrument")
        df = df.with_columns(
            pl.when(diff > 0).then(diff).otherwise(pl.lit(0.0)).alias("_gain"),
            pl.when(diff < 0).then(-diff).otherwise(pl.lit(0.0)).alias("_loss"),
        )

        alpha = 1.0 / float(rsi_period)
        df = df.with_columns(
            pl.col("_gain").ewm_mean(alpha=alpha, adjust=False).over("instrument").alias("_avg_gain"),
            pl.col("_loss").ewm_mean(alpha=alpha, adjust=False).over("instrument").alias("_avg_loss"),
        )

        df = df.with_columns((pl.col("_avg_gain") / pl.col("_avg_loss")).alias("_rs"))

        df = df.with_columns(
            pl.when(pl.col("_avg_loss") == 0)
            .then(pl.lit(100.0))
            .otherwise(100.0 - (100.0 / (1.0 + pl.col("_rs"))))
            .alias("_rsi")
        )

        df = df.with_columns(
            pl.when(pl.col("_rsi") <= pl.lit(rsi_oversold_cut))
            .then(pl.lit("oversold"))
            .when(pl.col("_rsi") >= pl.lit(rsi_overbought_cut))
            .then(pl.lit("overbought"))
            .otherwise(pl.lit("neutral"))
            .alias("_rsi_bucket")
        )

        out = df.select(
            "instrument",
            "anchor_tf",
            "ts",
            pl.col("_rsi").alias("rsi_value"),
            pl.col("_rsi_bucket").alias("rsi_bucket"),
            pl.col("_ema_fast").alias("ema_fast"),
            pl.col("_ema_slow").alias("ema_slow"),
        )

        out_frames.append(out)

    if not out_frames:
        return _empty_keyed_frame()

    out_all = pl.concat(out_frames, how="vertical")

    log.info(
        "ta_trend: built features rows=%d instruments=%s anchor_tfs=%s",
        out_all.height,
        out_all.select("instrument").unique().to_series().to_list(),
        out_all.select("anchor_tf").unique().to_series().to_list(),
    )
    return out_all
