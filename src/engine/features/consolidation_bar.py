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
            "consolidation_flag": pl.Series([], dtype=pl.Boolean),
            "consolidation_score": pl.Series([], dtype=pl.Float64),
            "consolidation_range_pct": pl.Series([], dtype=pl.Float64),
            "consolidation_vol_z": pl.Series([], dtype=pl.Float64),
            "consolidation_method": pl.Series([], dtype=pl.Utf8),
        }
    )


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    windows: pl.DataFrame | None = None,
    candles: pl.DataFrame | None = None,
    family_cfg: dict | None = None,
    **_,
) -> pl.DataFrame:
    """
    Table: data/windows
    Keys : instrument, anchor_tf, anchor_ts

    Outputs:
      - consolidation_flag
      - consolidation_score
      - consolidation_range_pct
      - consolidation_vol_z
      - consolidation_method

    Threshold keys (read only from family_cfg or ctx.features_auto_cfg["consolidation_bar"]):
      - lookback_bars
      - vol_z_window_bars
      - range_pct_max
      - vol_z_abs_max
    """
    if windows is None or windows.is_empty():
        log.warning("consolidation_bar: windows empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required_win = {"instrument", "anchor_tf", "anchor_ts"}
    missing = sorted(required_win - set(windows.columns))
    if missing:
        log.warning("consolidation_bar: windows missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    base = (
        windows.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("anchor_tf").cast(pl.Utf8),
            pl.col("anchor_ts").cast(pl.Datetime("us")),
        )
        .drop_nulls(["instrument", "anchor_tf", "anchor_ts"])
        .unique()
    )

    if candles is None or candles.is_empty():
        return base.with_columns(
            pl.lit(False).cast(pl.Boolean).alias("consolidation_flag"),
            pl.lit(None).cast(pl.Float64).alias("consolidation_score"),
            pl.lit(None).cast(pl.Float64).alias("consolidation_range_pct"),
            pl.lit(None).cast(pl.Float64).alias("consolidation_vol_z"),
            pl.lit("range_vol_squeeze").cast(pl.Utf8).alias("consolidation_method"),
        ).select(_empty_keyed_frame().columns)

    required_c = {"instrument", "tf", "ts", "high", "low", "close"}
    missing_c = sorted(required_c - set(candles.columns))
    if missing_c:
        log.warning("consolidation_bar: candles missing columns=%s; using placeholders", missing_c)
        return base.with_columns(
            pl.lit(False).cast(pl.Boolean).alias("consolidation_flag"),
            pl.lit(None).cast(pl.Float64).alias("consolidation_score"),
            pl.lit(None).cast(pl.Float64).alias("consolidation_range_pct"),
            pl.lit(None).cast(pl.Float64).alias("consolidation_vol_z"),
            pl.lit("range_vol_squeeze").cast(pl.Utf8).alias("consolidation_method"),
        ).select(_empty_keyed_frame().columns)

    cfg = dict(family_cfg or {})
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    cfg.update(auto_cfg.get("consolidation_bar", {}) if isinstance(auto_cfg, dict) else {})

    lookback_bars = int(cfg.get("lookback_bars", 20))
    lookback_bars = max(2, lookback_bars)
    vol_z_window_bars = int(cfg.get("vol_z_window_bars", 200))
    vol_z_window_bars = max(5, vol_z_window_bars)

    range_pct_max = float(cfg.get("range_pct_max", 0.003))
    vol_z_abs_max = float(cfg.get("vol_z_abs_max", 0.5))

    anchor_tfs = base.select("anchor_tf").unique().to_series().to_list()

    c = (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("tf").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
        )
        .drop_nulls(["instrument", "tf", "ts"])
        .sort(["instrument", "tf", "ts"])
        .unique(subset=["instrument", "tf", "ts"], keep="last")
        .filter(pl.col("tf").is_in(anchor_tfs))
        .with_columns(pl.col("tf").alias("anchor_tf"))
        .sort(["instrument", "anchor_tf", "ts"])
    )

    if c.is_empty():
        return base.with_columns(
            pl.lit(False).cast(pl.Boolean).alias("consolidation_flag"),
            pl.lit(None).cast(pl.Float64).alias("consolidation_score"),
            pl.lit(None).cast(pl.Float64).alias("consolidation_range_pct"),
            pl.lit(None).cast(pl.Float64).alias("consolidation_vol_z"),
            pl.lit("range_vol_squeeze").cast(pl.Utf8).alias("consolidation_method"),
        ).select(_empty_keyed_frame().columns)

    group_cols = ["instrument", "anchor_tf"]

    returns = (
        (pl.col("close") / pl.col("close").shift(1).over(group_cols) - 1)
        .alias("_ret")
    )

    rolling_high = pl.col("high").rolling_max(window_size=lookback_bars, min_periods=lookback_bars).over(group_cols)
    rolling_low = pl.col("low").rolling_min(window_size=lookback_bars, min_periods=lookback_bars).over(group_cols)
    range_pct = safe_div(rolling_high - rolling_low, pl.col("close"), default=0.0).alias("consolidation_range_pct")

    vol = pl.col("_ret").rolling_std(window_size=lookback_bars, min_periods=lookback_bars).over(group_cols).alias("_vol")
    vol_mean = pl.col("_vol").rolling_mean(window_size=vol_z_window_bars, min_periods=vol_z_window_bars).over(group_cols)
    vol_std = pl.col("_vol").rolling_std(window_size=vol_z_window_bars, min_periods=vol_z_window_bars).over(group_cols)
    vol_z = safe_div(pl.col("_vol") - vol_mean, vol_std, default=0.0).alias("consolidation_vol_z")

    range_score = (
        pl.when(range_pct.is_null())
        .then(pl.lit(None))
        .otherwise((1.0 - (range_pct / pl.lit(range_pct_max))).clip(0.0, 1.0))
        .alias("_range_score")
    )

    vol_score = (
        pl.when(vol_z.is_null())
        .then(pl.lit(None))
        .otherwise((1.0 - (vol_z.abs() / pl.lit(vol_z_abs_max))).clip(0.0, 1.0))
        .alias("_vol_score")
    )

    score = (
        pl.when(range_score.is_null() & vol_score.is_null())
        .then(pl.lit(None))
        .otherwise(pl.min_horizontal(range_score.fill_null(0.0), vol_score.fill_null(0.0)))
        .alias("consolidation_score")
    )

    flag = (
        (range_pct <= pl.lit(range_pct_max))
        & (vol_z.abs() <= pl.lit(vol_z_abs_max))
    ).fill_null(False).alias("consolidation_flag")

    out = (
        c.with_columns(returns)
        .with_columns(range_pct)
        .with_columns(vol)
        .with_columns(vol_z)
        .with_columns(range_score, vol_score, score, flag)
        .select(
            pl.col("instrument"),
            pl.col("anchor_tf"),
            pl.col("ts").alias("anchor_ts"),
            pl.col("consolidation_flag"),
            pl.col("consolidation_score"),
            pl.col("consolidation_range_pct"),
            pl.col("consolidation_vol_z"),
            pl.lit("range_vol_squeeze").cast(pl.Utf8).alias("consolidation_method"),
        )
    )

    return base.join(out, on=["instrument", "anchor_tf", "anchor_ts"], how="left").select(_empty_keyed_frame().columns)
