from __future__ import annotations

from collections.abc import Mapping

import polars as pl


def _cfg_value(cfg: Mapping[str, float], key: str, default: float) -> float:
    val = cfg.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def compute_event_flags(df: pl.DataFrame, *, cfg: Mapping[str, float]) -> pl.DataFrame:
    """
    Compute deterministic dealing-range evidence flags on a single instrument+tf frame.

    Expected columns: high, low, close, dr_low, dr_high, dr_mid, dr_width, atr
    """
    if df.is_empty():
        return df

    lookback = int(_cfg_value(cfg, "lookback_bars", 50))
    lookback = max(2, lookback)

    test_atr_mult = _cfg_value(cfg, "test_atr_mult", 0.2)
    probe_atr_mult = _cfg_value(cfg, "probe_atr_mult", 0.5)
    reclaim_atr_mult = _cfg_value(cfg, "reclaim_atr_mult", 0.2)
    accept_atr_mult = _cfg_value(cfg, "accept_atr_mult", 0.5)
    trend_atr_mult = _cfg_value(cfg, "trend_atr_mult", 1.5)
    trend_width_mult = _cfg_value(cfg, "trend_width_mult", 1.0)

    test_band = (pl.col("atr") * pl.lit(test_atr_mult)).alias("_test_band")
    probe_min = (pl.col("atr") * pl.lit(probe_atr_mult)).alias("_probe_min")
    reclaim_band = (pl.col("atr") * pl.lit(reclaim_atr_mult)).alias("_reclaim_band")
    accept_dist = (pl.col("atr") * pl.lit(accept_atr_mult)).alias("_accept_dist")

    inside = (
        (pl.col("close") >= pl.col("dr_low"))
        & (pl.col("close") <= pl.col("dr_high"))
    ).alias("_inside")

    inside_ratio = (
        pl.col("_inside")
        .cast(pl.Int64)
        .rolling_mean(window_size=lookback, min_periods=lookback)
        .alias("_inside_ratio")
    )

    test_high = (pl.col("high") >= (pl.col("dr_high") - pl.col("_test_band"))).alias("_test_high")
    test_low = (pl.col("low") <= (pl.col("dr_low") + pl.col("_test_band"))).alias("_test_low")
    tests_count = (
        (pl.col("_test_high").cast(pl.Int64) + pl.col("_test_low").cast(pl.Int64))
        .rolling_sum(window_size=lookback, min_periods=lookback)
        .alias("_tests_count")
    )

    pierce_low = (pl.col("dr_low") - pl.col("low")).alias("_pierce_low")
    pierce_high = (pl.col("high") - pl.col("dr_high")).alias("_pierce_high")
    pierce_low = pl.when(pl.col("_pierce_low") < 0).then(0.0).otherwise(pl.col("_pierce_low")).alias("_pierce_low")
    pierce_high = pl.when(pl.col("_pierce_high") < 0).then(0.0).otherwise(pl.col("_pierce_high")).alias("_pierce_high")

    probe_low = (pl.col("_pierce_low") >= pl.col("_probe_min")).alias("_probe_low")
    probe_high = (pl.col("_pierce_high") >= pl.col("_probe_min")).alias("_probe_high")

    reclaim_from_low = (pl.col("close") >= (pl.col("dr_low") + pl.col("_reclaim_band"))).alias("_reclaim_from_low")
    reclaim_from_high = (pl.col("close") <= (pl.col("dr_high") - pl.col("_reclaim_band"))).alias("_reclaim_from_high")

    outside_up = (pl.col("close") >= (pl.col("dr_high") + pl.col("_accept_dist"))).alias("_outside_up")
    outside_dn = (pl.col("close") <= (pl.col("dr_low") - pl.col("_accept_dist"))).alias("_outside_dn")

    dist_mid = (pl.col("close") - pl.col("dr_mid")).abs().alias("_dist_mid")
    trend_dist = pl.max_horizontal(
        (pl.col("atr") * pl.lit(trend_atr_mult)),
        (pl.col("dr_width") * pl.lit(trend_width_mult)),
    ).alias("_trend_dist")
    trend_far = (pl.col("_dist_mid") >= pl.col("_trend_dist")).alias("_trend_far")

    return (
        df.with_columns(test_band, probe_min, reclaim_band, accept_dist)
        .with_columns(inside)
        .with_columns(inside_ratio)
        .with_columns(test_high, test_low)
        .with_columns(tests_count)
        .with_columns(pierce_low, pierce_high)
        .with_columns(probe_low, probe_high)
        .with_columns(reclaim_from_low, reclaim_from_high)
        .with_columns(outside_up, outside_dn)
        .with_columns(dist_mid, trend_dist, trend_far)
    )
