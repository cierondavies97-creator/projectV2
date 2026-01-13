from __future__ import annotations

from typing import Mapping, Any

import polars as pl

from engine.features._shared import (
    require_cols,
    ensure_sorted,
    to_anchor_tf,
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
    Family: pcra_bar (bar-proxy PCRA)

    Target table:
      - data/pcr_a

    Keys:
      - instrument
      - anchor_tf
      - pcr_window_ts

    Output columns (canonical):
      - pcr_ofi_value, pcr_ofi_bucket
      - pcr_micro_vol_value, pcr_micro_vol_bucket
      - pcr_range_value, pcr_range_bucket
      - pcr_clv_bucket
    """
    if candles is None or candles.is_empty():
        return pl.DataFrame()

    cfg = dict(family_cfg or {})
    vol_z_window = int(cfg.get("vol_z_window", 100))
    vol_z_medium_high_cut = float(cfg.get("vol_z_medium_high_cut", 1.0))
    vol_z_low_high_cut = float(cfg.get("vol_z_low_high_cut", 2.0))

    ofi_strong_cut = float(cfg.get("ofi_strong_cut", 0.5))

    range_bucket_window = int(cfg.get("range_bucket_window", 250))
    range_q_low = float(cfg.get("range_q_low", 0.33))
    range_q_high = float(cfg.get("range_q_high", 0.66))

    require_cols(
        candles,
        ["instrument", "tf", "ts", "open", "high", "low", "close", "volume"],
        where="pcra_bar",
    )

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = list(getattr(cluster, "anchor_tfs", []) or [])
    if not anchor_tfs:
        anchor_tfs = sorted(candles.get_column("tf").unique().to_list())

    out_frames: list[pl.DataFrame] = []

    for anchor_tf in anchor_tfs:
        df = to_anchor_tf(candles, anchor_tf=str(anchor_tf), where="pcra_bar")
        if df.is_empty():
            continue

        df = ensure_sorted(df, by=["instrument", "ts"])

        # Rolling stats for volume z-score (causal).
        mean_v = pl.col("volume").rolling_mean(vol_z_window, min_periods=vol_z_window).over("instrument")
        std_v = pl.col("volume").rolling_std(vol_z_window, min_periods=vol_z_window).over("instrument")

        # Proxy OFI: signed volume by bar direction.
        ofi = (
            pl.when(pl.col("close") >= pl.col("open"))
            .then(pl.col("volume"))
            .otherwise(-pl.col("volume"))
            .alias("_ofi")
        )

        # CLV: close location within the bar (0..1).
        clv = _clamp01(
            safe_div(
                pl.col("close") - pl.col("low"),
                pl.col("high") - pl.col("low"),
                default=0.5,
            )
        ).alias("_clv")

        df = df.with_columns(
            ofi,
            zscore(pl.col("volume"), mean_expr=mean_v, std_expr=std_v).alias("pcr_micro_vol_value"),
            (pl.col("high") - pl.col("low")).alias("pcr_range_value"),
            clv,
        ).with_columns(
            # Scale OFI by rolling mean volume for stability.
            safe_div(pl.col("_ofi"), mean_v, default=0.0).alias("pcr_ofi_value")
        )

        # Causal range bucket via rolling quantiles.
        q_lo = pl.col("pcr_range_value").rolling_quantile(range_q_low, "nearest", range_bucket_window, min_periods=range_bucket_window).over("instrument")
        q_hi = pl.col("pcr_range_value").rolling_quantile(range_q_high, "nearest", range_bucket_window, min_periods=range_bucket_window).over("instrument")

        df = df.with_columns(
            bucket_by_edges(
                pl.col("pcr_micro_vol_value"),
                edges=[vol_z_medium_high_cut, vol_z_low_high_cut],
                labels=["low", "medium", "high"],
                default="unknown",
            ).alias("pcr_micro_vol_bucket"),
            bucket_by_edges(
                pl.col("pcr_ofi_value"),
                edges=[-ofi_strong_cut, ofi_strong_cut],
                labels=["strong_sell", "balanced", "strong_buy"],
                default="unknown",
            ).alias("pcr_ofi_bucket"),
            bucket_by_edges(
                pl.col("pcr_range_value"),
                edges=[q_lo, q_hi],
                labels=["narrow", "normal", "wide"],
                default="unknown",
            ).alias("pcr_range_bucket"),
            bucket_by_edges(
                pl.col("_clv"),
                edges=[0.33, 0.66],
                labels=["lower", "mid", "upper"],
                default="unknown",
            ).alias("pcr_clv_bucket"),
        )

        out = df.select(
            [
                pl.col("instrument"),
                pl.lit(str(anchor_tf)).alias("anchor_tf"),
                pl.col("ts").alias("pcr_window_ts"),
                pl.col("pcr_ofi_value"),
                pl.col("pcr_ofi_bucket"),
                pl.col("pcr_micro_vol_value"),
                pl.col("pcr_micro_vol_bucket"),
                pl.col("pcr_range_value"),
                pl.col("pcr_range_bucket"),
                pl.col("pcr_clv_bucket"),
            ]
        )

        out = conform_to_registry(
            out,
            registry_entry=registry_entry,
            key_cols=["instrument", "anchor_tf", "pcr_window_ts"],
            where="pcra_bar",
            allow_extra=False,
        )

        out_frames.append(out)

    return pl.concat(out_frames, how="vertical") if out_frames else pl.DataFrame()


def _clamp01(x: pl.Expr) -> pl.Expr:
    return pl.when(x < 0).then(0.0).when(x > 1).then(1.0).otherwise(x)
