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
            "stat_ts_return": pl.Series([], dtype=pl.Float64),
            "stat_ts_vol": pl.Series([], dtype=pl.Float64),
            "stat_ts_range": pl.Series([], dtype=pl.Float64),
            "stat_ts_return_zscore": pl.Series([], dtype=pl.Float64),
            "stat_ts_vol_zscore": pl.Series([], dtype=pl.Float64),
            "stat_ts_range_zscore": pl.Series([], dtype=pl.Float64),
        }
    )


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    **_,
) -> pl.DataFrame:
    """
    Table: data/features
    Keys : instrument, anchor_tf, ts

    Emits:
      - stat_ts_return          : close-to-close return (pct)
      - stat_ts_vol             : |return|
      - stat_ts_range           : (high - low)
      - stat_ts_*_zscore        : per-(instrument, anchor_tf) z-scores over the trading-day slice
    """
    if candles is None or candles.is_empty():
        log.warning("stat_ts: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "tf", "ts", "open", "high", "low", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("stat_ts: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("stat_ts: ctx.cluster.anchor_tfs empty; cannot determine anchor tf; returning empty keyed frame")
        return _empty_keyed_frame()

    out_frames: list[pl.DataFrame] = []
    for anchor_tf in anchor_tfs:
        df = candles.filter(pl.col("tf") == pl.lit(str(anchor_tf)))
        if df.is_empty():
            continue

        # deterministic order
        df = df.sort(["instrument", "ts"]).with_columns(pl.lit(str(anchor_tf)).alias("anchor_tf"))

        # raw stats
        df = df.with_columns(
            pl.col("close").pct_change().over("instrument").alias("_stat_ret"),
            (pl.col("high") - pl.col("low")).alias("_stat_range"),
        )
        df = df.with_columns(pl.col("_stat_ret").abs().alias("_stat_vol"))

        df = df.with_columns(
            pl.col("_stat_ret").alias("stat_ts_return"),
            pl.col("_stat_vol").alias("stat_ts_vol"),
            pl.col("_stat_range").alias("stat_ts_range"),
        )

        # per-(instrument, anchor_tf) z-scores (day-slice)
        by_keys = ["instrument", "anchor_tf"]
        df = df.with_columns(
            ((pl.col("stat_ts_return") - pl.col("stat_ts_return").mean().over(by_keys)) / pl.col("stat_ts_return").std(ddof=1).over(by_keys)).alias("stat_ts_return_zscore"),
            ((pl.col("stat_ts_range") - pl.col("stat_ts_range").mean().over(by_keys)) / pl.col("stat_ts_range").std(ddof=1).over(by_keys)).alias("stat_ts_range_zscore"),
            ((pl.col("stat_ts_vol") - pl.col("stat_ts_vol").mean().over(by_keys)) / pl.col("stat_ts_vol").std(ddof=1).over(by_keys)).alias("stat_ts_vol_zscore"),
        )

        out = df.select(
            "instrument",
            "anchor_tf",
            "ts",
            "stat_ts_return",
            "stat_ts_vol",
            "stat_ts_range",
            "stat_ts_return_zscore",
            "stat_ts_vol_zscore",
            "stat_ts_range_zscore",
        )
        out_frames.append(out)

    if not out_frames:
        return _empty_keyed_frame()

    out_all = pl.concat(out_frames, how="vertical")

    log.info(
        "stat_ts: built features rows=%d instruments=%s anchor_tfs=%s",
        out_all.height,
        out_all.select("instrument").unique().to_series().to_list(),
        out_all.select("anchor_tf").unique().to_series().to_list(),
    )
    return out_all