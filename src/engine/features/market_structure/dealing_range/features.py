from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from engine.features._shared import safe_div


def _cfg_value(cfg: Mapping[str, float], key: str, default: float) -> float:
    val = cfg.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def compute_range_descriptors(df: pl.DataFrame, *, cfg: Mapping[str, float]) -> pl.DataFrame:
    """
    Compute range descriptors on a single instrument+tf candle frame.

    Expected columns: ts, high, low, close
    Adds: atr, dr_low, dr_high, dr_mid, dr_width, dr_width_atr, range_position
    """
    if df.is_empty():
        return df

    lookback = int(_cfg_value(cfg, "lookback_bars", 50))
    lookback = max(2, lookback)
    atr_window = int(_cfg_value(cfg, "atr_window", 14))
    atr_window = max(2, atr_window)

    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        (pl.col("high") - pl.col("low")),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    ).alias("atr_tr")

    atr = pl.col("atr_tr").rolling_mean(window_size=atr_window, min_periods=atr_window).alias("atr")

    dr_high = pl.col("high").rolling_max(window_size=lookback, min_periods=lookback).alias("dr_high")
    dr_low = pl.col("low").rolling_min(window_size=lookback, min_periods=lookback).alias("dr_low")
    dr_width = (pl.col("dr_high") - pl.col("dr_low")).alias("dr_width")
    dr_mid = ((pl.col("dr_high") + pl.col("dr_low")) / 2.0).alias("dr_mid")
    dr_width_atr = safe_div(pl.col("dr_width"), pl.col("atr"), default=None).alias("dr_width_atr")
    range_position = safe_div(
        pl.col("close") - pl.col("dr_low"),
        pl.col("dr_width"),
        default=None,
    ).alias("range_position")

    return (
        df.with_columns(tr)
        .with_columns(atr)
        .with_columns(dr_high, dr_low, dr_width, dr_mid, dr_width_atr, range_position)
    )
