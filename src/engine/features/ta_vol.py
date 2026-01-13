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
            "atr_value": pl.Series([], dtype=pl.Float64),
            "ta_vol__atr": pl.Series([], dtype=pl.Float64),
            "ta_vol__ret1_std": pl.Series([], dtype=pl.Float64),
            "ta_vol__range_mean": pl.Series([], dtype=pl.Float64),
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

    Outputs (must match features_registry.yaml ta_vol):
      - atr_value
      - ta_vol__atr
      - ta_vol__ret1_std
      - ta_vol__range_mean

    Threshold keys (ONLY these are read from ctx.features_auto_cfg["ta_vol"]):
      - atr_period
      - atr_smoothing        ("wilder" | "sma")
      - ret_window
      - ret_type             ("pct" | "log")
      - min_periods_atr
      - min_periods_ret_std
      - min_periods_range_mean
    """
    if candles is None or candles.is_empty():
        log.warning("ta_vol: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "tf", "ts", "open", "high", "low", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("ta_vol: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("ta_vol: ctx.cluster.anchor_tfs empty; returning empty keyed frame")
        return _empty_keyed_frame()

    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg = dict(auto_cfg.get("ta_vol", {}) if isinstance(auto_cfg, dict) else {})

    atr_period = max(1, int(fam_cfg.get("atr_period", 14)))
    atr_smoothing = str(fam_cfg.get("atr_smoothing", "wilder")).strip().lower()
    if atr_smoothing not in {"wilder", "sma"}:
        atr_smoothing = "wilder"

    ret_window = max(2, int(fam_cfg.get("ret_window", 20)))
    ret_type = str(fam_cfg.get("ret_type", "pct")).strip().lower()
    if ret_type not in {"pct", "log"}:
        ret_type = "pct"

    min_periods_atr = max(1, int(fam_cfg.get("min_periods_atr", 1)))
    min_periods_ret_std = max(2, int(fam_cfg.get("min_periods_ret_std", 2)))
    min_periods_range_mean = max(1, int(fam_cfg.get("min_periods_range_mean", 1)))

    # Normalize inputs; dedupe (instrument, tf, ts) deterministically
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
    )

    out_frames: list[pl.DataFrame] = []

    for anchor_tf in anchor_tfs:
        tf_str = str(anchor_tf)
        df = c.filter(pl.col("tf") == pl.lit(tf_str))
        if df.is_empty():
            continue

        df = df.sort(["instrument", "ts"]).with_columns(pl.lit(tf_str).alias("anchor_tf"))

        # True Range: max(high-low, abs(high-prev_close), abs(low-prev_close))
        prev_close = pl.col("close").shift(1).over("instrument")
        tr = pl.max_horizontal(
            (pl.col("high") - pl.col("low")),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        ).alias("_tr")

        df = df.with_columns(tr)

        # ATR
        if atr_smoothing == "wilder":
            # Wilder ATR is an RMA of TR, alpha=1/period
            alpha = 1.0 / float(atr_period)
            df = df.with_columns(
                pl.col("_tr").ewm_mean(alpha=alpha, adjust=False).over("instrument").alias("atr_value")
            )
        else:
            # Simple moving average of TR
            df = df.with_columns(
                pl.col("_tr")
                .rolling_mean(window_size=atr_period, min_periods=min_periods_atr)
                .over("instrument")
                .alias("atr_value")
            )

        df = df.with_columns(pl.col("atr_value").alias("ta_vol__atr"))

        # Returns + rolling std
        if ret_type == "log":
            ret1 = (pl.col("close").log() - pl.col("close").shift(1).log()).over("instrument").alias("_ret1")
        else:
            ret1 = pl.col("close").pct_change().over("instrument").alias("_ret1")

        df = df.with_columns(
            ret1,
            (pl.col("high") - pl.col("low")).alias("_range"),
        )

        df = df.with_columns(
            pl.col("_ret1")
            .rolling_std(window_size=ret_window, min_periods=min_periods_ret_std)
            .over("instrument")
            .alias("ta_vol__ret1_std"),
            pl.col("_range")
            .rolling_mean(window_size=ret_window, min_periods=min_periods_range_mean)
            .over("instrument")
            .alias("ta_vol__range_mean"),
        )

        out_frames.append(
            df.select(
                "instrument",
                "anchor_tf",
                "ts",
                "atr_value",
                "ta_vol__atr",
                "ta_vol__ret1_std",
                "ta_vol__range_mean",
            )
        )

    if not out_frames:
        return _empty_keyed_frame()

    out_all = pl.concat(out_frames, how="vertical").sort(["instrument", "anchor_tf", "ts"])

    log.info(
        "ta_vol: built rows=%d instruments=%d anchor_tfs=%d atr_period=%d atr_smoothing=%s ret_window=%d ret_type=%s",
        out_all.height,
        out_all.select("instrument").n_unique(),
        out_all.select("anchor_tf").n_unique(),
        atr_period,
        atr_smoothing,
        ret_window,
        ret_type,
    )
    return out_all
