from __future__ import annotations

from typing import Mapping, Any

import polars as pl

from engine.features._shared import (
    require_cols,
    ensure_sorted,
    to_anchor_tf,
    rolling_atr,
    zscore,
    bucket_by_edges,
    safe_div,
    conform_to_registry,
)


def build_feature_frame(
    *,
    ctx,
    candles: pl.DataFrame,
    family_cfg: Mapping[str, Any] | None = None,
    registry_entry: Mapping[str, Any] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Family: ta_vol

    Target table:
      - data/features

    Keys:
      - instrument
      - anchor_tf
      - ts

    Outputs (recommended canonical set):
      - ta_vol__atr      : ATR(period) on anchor bars
      - ta_vol__atr_z    : rolling z-score of atr_anchor
      - ta_vol__vol_bucket : low/medium/high from atr_z

    Notes:
      - This implementation is causal (rolling windows only; no full-day stats).
      - Keep all feature names stable; if you rename, update conf/features_registry.yaml.
    """
    if candles is None or candles.is_empty():
        return pl.DataFrame()

    cfg = dict(family_cfg or {})
    atr_period = int(cfg.get("atr_period", 14))
    z_window = int(cfg.get("atr_z_window", 200))
    ret_window = int(cfg.get("ret_window", 50))
    vol_z_medium_high_cut = float(cfg.get("vol_z_medium_high_cut", 1.0))
    vol_z_low_high_cut = float(cfg.get("vol_z_low_high_cut", 2.0))

    require_cols(
        candles,
        ["instrument", "tf", "ts", "open", "high", "low", "close", "volume"],
        where="ta_vol",
    )

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = list(getattr(cluster, "anchor_tfs", []) or [])
    if not anchor_tfs:
        anchor_tfs = sorted(candles.get_column("tf").unique().to_list())

    out_frames: list[pl.DataFrame] = []

    for anchor_tf in anchor_tfs:
        df = to_anchor_tf(candles, anchor_tf=str(anchor_tf), where="ta_vol")
        if df.is_empty():
            continue

        df = ensure_sorted(df, by=["instrument", "ts"])

        # ATR + z-score regime
        df = rolling_atr(df, group_cols=["instrument"], period=atr_period, out_col="ta_vol__atr", tr_col="_tr")

        mean_atr = pl.col("ta_vol__atr").rolling_mean(z_window, min_periods=z_window).over("instrument")
        std_atr = pl.col("ta_vol__atr").rolling_std(z_window, min_periods=z_window).over("instrument")
        df = df.with_columns(zscore(pl.col("ta_vol__atr"), mean_expr=mean_atr, std_expr=std_atr).alias("ta_vol__atr_z"))

        df = df.with_columns(
            bucket_by_edges(
                pl.col("ta_vol__atr_z"),
                edges=[vol_z_medium_high_cut, vol_z_low_high_cut],
                labels=["low", "medium", "high"],
                default="unknown",
            ).alias("ta_vol__vol_bucket")
        )

        # Optional extra vol stats (useful for research; keep names namespaced)
        df = df.with_columns(
            safe_div(pl.col("close"), pl.col("close").shift(1), default=1.0).over("instrument").alias("_ratio"),
        ).with_columns(
            (pl.col("_ratio") - 1.0).alias("_ret1"),
            (pl.col("high") - pl.col("low")).alias("_range"),
        ).with_columns(
            pl.col("_ret1").rolling_std(ret_window, min_periods=ret_window).over("instrument").alias("ta_vol__ret1_std"),
            pl.col("_range").rolling_mean(ret_window, min_periods=ret_window).over("instrument").alias("ta_vol__range_mean"),
        )

        out = df.select(
            [
                pl.col("instrument"),
                pl.lit(str(anchor_tf)).alias("anchor_tf"),
                pl.col("ts"),
                pl.col("ta_vol__atr"),
                pl.col("ta_vol__atr_z"),
                pl.col("ta_vol__vol_bucket"),
                pl.col("ta_vol__ret1_std"),
                pl.col("ta_vol__range_mean"),
            ]
        )

        out = conform_to_registry(
            out,
            registry_entry=registry_entry,
            key_cols=["instrument", "anchor_tf", "ts"],
            where="ta_vol",
            allow_extra=False,
        )

        out_frames.append(out)

    return pl.concat(out_frames, how="vertical") if out_frames else pl.DataFrame()
