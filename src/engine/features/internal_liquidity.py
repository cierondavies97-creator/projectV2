from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry, rolling_atr, safe_div

log = logging.getLogger(__name__)


def _empty_keyed_frame(registry_entry: Mapping[str, object] | None) -> pl.DataFrame:
    if registry_entry and isinstance(registry_entry, Mapping):
        columns = registry_entry.get("columns")
        if isinstance(columns, Mapping) and columns:
            return pl.DataFrame({col: pl.Series([], dtype=pl.Null) for col in columns})
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "anchor_tf": pl.Series([], dtype=pl.Utf8),
            "anchor_ts": pl.Series([], dtype=pl.Datetime("us")),
        }
    )


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg: dict[str, object] = {}
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("internal_liquidity", {}) or {})
    if isinstance(family_cfg, Mapping):
        cfg.update(family_cfg)
    return cfg


def _microstructure_frame(external: pl.DataFrame | None) -> pl.DataFrame:
    if external is None or external.is_empty():
        return pl.DataFrame()
    required = {"instrument", "anchor_tf", "ts", "ofi", "agg_imbalance", "intensity"}
    if not required.issubset(set(external.columns)):
        return pl.DataFrame()
    return (
        external.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("anchor_tf").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")).alias("anchor_ts"),
            pl.col("ofi").cast(pl.Float64),
            pl.col("agg_imbalance").cast(pl.Float64),
            pl.col("intensity").cast(pl.Float64),
        )
        .drop_nulls(["instrument", "anchor_tf", "anchor_ts"])
        .sort(["instrument", "anchor_tf", "anchor_ts"])
    )


def _sign_expr(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr > 0)
        .then(pl.lit(1))
        .when(expr < 0)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
    )


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Internal liquidity proxy based on range-interior swing levels.

    Table: data/windows
    Keys : instrument, anchor_tf, anchor_ts
    """
    if candles is None or candles.is_empty():
        log.warning("internal_liquidity: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    required = {"instrument", "tf", "ts", "high", "low", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("internal_liquidity: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame(registry_entry)

    cfg = _merge_cfg(ctx, family_cfg)
    lookback = max(5, int(cfg.get("internal_liq_lookback_bars", 50)))
    range_method = str(cfg.get("internal_liq_range_method", "donchian"))
    range_min_atr = float(cfg.get("internal_liq_range_min_atr", 0.0))
    range_max_atr = float(cfg.get("internal_liq_range_max_atr", 1.0e9))
    buffer_mode = str(cfg.get("internal_liq_interior_buffer_mode", "atr"))
    buffer_ticks = float(cfg.get("internal_liq_interior_buffer_ticks", 0.0))
    atr_window = max(5, int(cfg.get("internal_liq_atr_window", 14)))
    edge_atr_mult = float(cfg.get("internal_liq_edge_atr_mult", 0.25))
    min_count = max(1, int(cfg.get("internal_liq_min_count", 2)))
    pivot_left = max(1, int(cfg.get("internal_liq_pivot_left_bars", 2)))
    pivot_right = max(1, int(cfg.get("internal_liq_pivot_right_bars", 2)))
    pivot_prom_atr = float(cfg.get("internal_liq_pivot_prominence_atr_min", 0.0))
    level_merge_mode = str(cfg.get("internal_liq_level_merge_mode", "atr"))
    level_merge_ticks = float(cfg.get("internal_liq_level_merge_ticks", 0.0))
    level_merge_atr = float(cfg.get("internal_liq_level_merge_atr_mult", 0.0))
    max_levels = max(1, int(cfg.get("internal_liq_max_levels_per_side", 1)))
    level_rank_method = str(cfg.get("internal_liq_level_rank_method", "count"))
    touch_mode = str(cfg.get("internal_liq_touch_mode", "wick"))
    touch_tol_mode = str(cfg.get("internal_liq_touch_tolerance_mode", "atr"))
    touch_tol_ticks = float(cfg.get("internal_liq_touch_tolerance_ticks", 0.0))
    touch_tol_atr = float(cfg.get("internal_liq_touch_tolerance_atr_mult", 0.0))
    min_bars_between = max(0, int(cfg.get("internal_liq_min_bars_between_touches", 0)))
    nearest_dist_atr_max = float(cfg.get("internal_liq_nearest_dist_atr_max", 1.0e9))
    require_both_sides = bool(cfg.get("internal_liq_require_both_sides", False))
    source_used = str(cfg.get("internal_liq_source", "range_interior"))
    micro_window = max(5, int(cfg.get("internal_liq_micro_window_bars", 50)))
    micro_intensity_cut = float(cfg.get("internal_liq_micro_intensity_z_cut", 2.0))

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("internal_liquidity: ctx.cluster.anchor_tfs empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

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
    )

    micro_df = _microstructure_frame(external)
    out_frames: list[pl.DataFrame] = []
    for anchor_tf in anchor_tfs:
        tf_str = str(anchor_tf)
        df = c.filter(pl.col("tf") == pl.lit(tf_str))
        if df.is_empty():
            continue

        df = df.sort(["instrument", "ts"]).with_columns(pl.lit(tf_str).alias("anchor_tf"))
        df = rolling_atr(
            df,
            group_cols=["instrument", "anchor_tf"],
            period=atr_window,
            ts_col="ts",
            high_col="high",
            low_col="low",
            close_col="close",
            out_col="_atr",
        )

        by = ["instrument", "anchor_tf"]
        if range_method != "donchian":
            range_high = pl.lit(None).cast(pl.Float64)
            range_low = pl.lit(None).cast(pl.Float64)
        else:
            range_high = pl.col("high").rolling_max(window_size=lookback, min_periods=lookback).over(by)
            range_low = pl.col("low").rolling_min(window_size=lookback, min_periods=lookback).over(by)
        edge_buffer = (
            pl.lit(buffer_ticks)
            if buffer_mode == "ticks"
            else (pl.col("_atr") * pl.lit(edge_atr_mult))
        )
        inner_high = range_high - edge_buffer
        inner_low = range_low + edge_buffer

        high_in = pl.when((pl.col("high") <= inner_high) & (pl.col("high") >= inner_low)).then(pl.col("high"))
        low_in = pl.when((pl.col("low") >= inner_low) & (pl.col("low") <= inner_high)).then(pl.col("low"))

        df = df.with_columns(
            range_high.alias("_range_high"),
            range_low.alias("_range_low"),
            inner_high.alias("_inner_high"),
            inner_low.alias("_inner_low"),
            high_in.alias("_high_in"),
            low_in.alias("_low_in"),
        )

        pivot_window = pivot_left + pivot_right + 1
        hi_window = pl.col("high").rolling_max(window_size=pivot_window, min_periods=pivot_window).shift(
            -pivot_right
        ).over(by)
        lo_window = pl.col("low").rolling_min(window_size=pivot_window, min_periods=pivot_window).shift(
            -pivot_right
        ).over(by)
        pivot_hi = pl.col("high") == hi_window
        pivot_lo = pl.col("low") == lo_window
        prom_hi = safe_div(pl.col("high") - lo_window, pl.col("_atr"), default=0.0) >= pl.lit(pivot_prom_atr)
        prom_lo = safe_div(hi_window - pl.col("low"), pl.col("_atr"), default=0.0) >= pl.lit(pivot_prom_atr)

        pivot_high_level = pl.when(pivot_hi & prom_hi & (pl.col("high") <= inner_high) & (pl.col("high") >= inner_low)).then(
            pl.col("high")
        )
        pivot_low_level = pl.when(pivot_lo & prom_lo & (pl.col("low") >= inner_low) & (pl.col("low") <= inner_high)).then(
            pl.col("low")
        )

        if level_rank_method == "recency_weighted":
            internal_high = pivot_high_level.fill_null(strategy="forward").over(by)
            internal_low = pivot_low_level.fill_null(strategy="forward").over(by)
        else:
            internal_high = pivot_high_level.rolling_max(window_size=lookback, min_periods=lookback).over(by)
            internal_low = pivot_low_level.rolling_min(window_size=lookback, min_periods=lookback).over(by)

        merge_tol = (
            pl.lit(level_merge_ticks)
            if level_merge_mode == "ticks"
            else (pl.col("_atr") * pl.lit(level_merge_atr))
        )
        touch_tol = (
            pl.lit(touch_tol_ticks)
            if touch_tol_mode == "ticks"
            else (pl.col("_atr") * pl.lit(touch_tol_atr))
        )

        if touch_mode == "close":
            touch_val_hi = pl.col("close")
            touch_val_lo = pl.col("close")
        elif touch_mode == "hlc3":
            touch_val_hi = (pl.col("high") + pl.col("low") + pl.col("close")) / pl.lit(3.0)
            touch_val_lo = touch_val_hi
        else:
            touch_val_hi = pl.col("high")
            touch_val_lo = pl.col("low")

        hi_touch = (internal_high.is_not_null()) & ((touch_val_hi - internal_high).abs() <= (touch_tol + merge_tol))
        lo_touch = (internal_low.is_not_null()) & ((touch_val_lo - internal_low).abs() <= (touch_tol + merge_tol))

        if min_bars_between > 0:
            hi_recent = (
                hi_touch.shift(1)
                .rolling_max(window_size=min_bars_between, min_periods=1)
                .over(by)
                .fill_null(0)
            )
            lo_recent = (
                lo_touch.shift(1)
                .rolling_max(window_size=min_bars_between, min_periods=1)
                .over(by)
                .fill_null(0)
            )
            hi_touch = hi_touch & (hi_recent == 0)
            lo_touch = lo_touch & (lo_recent == 0)

        high_count = (
            pl.when(hi_touch).then(pl.lit(1)).otherwise(pl.lit(0))
            .rolling_sum(window_size=lookback, min_periods=lookback)
            .over(by)
        )
        low_count = (
            pl.when(lo_touch).then(pl.lit(1)).otherwise(pl.lit(0))
            .rolling_sum(window_size=lookback, min_periods=lookback)
            .over(by)
        )

        dist_high = safe_div((internal_high - pl.col("close")).abs(), pl.col("_atr"), default=0.0)
        dist_low = safe_div((pl.col("close") - internal_low).abs(), pl.col("_atr"), default=0.0)
        nearest_side = (
            pl.when(internal_high.is_null() & internal_low.is_null())
            .then(pl.lit("none"))
            .when(internal_high.is_null())
            .then(pl.lit("low"))
            .when(internal_low.is_null())
            .then(pl.lit("high"))
            .when(dist_high <= dist_low)
            .then(pl.lit("high"))
            .otherwise(pl.lit("low"))
        )
        nearest_dist = pl.when(nearest_side == pl.lit("high")).then(dist_high).otherwise(dist_low)

        width_atr = safe_div(range_high - range_low, pl.col("_atr"), default=0.0)
        flag = (high_count >= pl.lit(min_count)) | (low_count >= pl.lit(min_count))
        if require_both_sides:
            flag = flag & (high_count >= pl.lit(min_count)) & (low_count >= pl.lit(min_count))
        if nearest_dist_atr_max < 1.0e9:
            flag = flag & (nearest_dist <= pl.lit(nearest_dist_atr_max))
        reason = (
            pl.when(pl.lit(range_method) != pl.lit("donchian"))
            .then(pl.lit("UNSUPPORTED_RANGE_METHOD"))
            .when(pl.col("_atr").is_null() | pl.col("_range_high").is_null() | pl.col("_range_low").is_null())
            .then(pl.lit("INSUFFICIENT_LOOKBACK"))
            .when((width_atr < pl.lit(range_min_atr)) | (width_atr > pl.lit(range_max_atr)))
            .then(pl.lit("RANGE_OUT_OF_BOUNDS"))
            .when(pl.lit(max_levels) < pl.lit(1))
            .then(pl.lit("LEVEL_LIMIT_INVALID"))
            .when(flag)
            .then(pl.lit("INTERNAL_LIQ_PRESENT"))
            .otherwise(pl.lit("INTERNAL_LIQ_ABSENT"))
        )

        out = df.with_columns(
            internal_high.alias("internal_liq_level_high"),
            internal_low.alias("internal_liq_level_low"),
            high_count.cast(pl.Int64).alias("internal_liq_high_count_L"),
            low_count.cast(pl.Int64).alias("internal_liq_low_count_L"),
            nearest_side.alias("internal_liq_nearest_side"),
            nearest_dist.alias("internal_liq_nearest_dist_atr"),
            flag.cast(pl.Boolean).alias("internal_liq_flag"),
            reason.alias("internal_liq_reason_code"),
            pl.lit(source_used).alias("internal_liq_source_used"),
            pl.lit(None).cast(pl.Boolean).alias("internal_liq_fvg_overlap_flag"),
        )

        if not micro_df.is_empty():
            micro_tf = micro_df.filter(pl.col("anchor_tf") == pl.lit(tf_str))
            if not micro_tf.is_empty():
                by_micro = ["instrument", "anchor_tf"]
                ofi_sign = _sign_expr(pl.col("ofi")).alias("_ofi_sign")
                agg_sign = _sign_expr(pl.col("agg_imbalance")).alias("_agg_sign")
                micro_tf = micro_tf.with_columns(ofi_sign, agg_sign)
                micro_tf = micro_tf.with_columns(
                    (
                        (pl.col("_ofi_sign") != pl.col("_ofi_sign").shift(1).over(by_micro))
                        & (pl.col("_ofi_sign") != 0)
                        & (pl.col("_ofi_sign").shift(1).over(by_micro) != 0)
                    ).alias("internal_liq_ofi_flip_flag"),
                    (
                        (pl.col("_agg_sign") != pl.col("_agg_sign").shift(1).over(by_micro))
                        & (pl.col("_agg_sign") != 0)
                        & (pl.col("_agg_sign").shift(1).over(by_micro) != 0)
                    ).alias("internal_liq_agg_flip_flag"),
                )
                intensity_mean = pl.col("intensity").rolling_mean(
                    window_size=micro_window, min_periods=micro_window
                ).over(by_micro)
                intensity_std = pl.col("intensity").rolling_std(
                    window_size=micro_window, min_periods=micro_window
                ).over(by_micro)
                intensity_z = safe_div(pl.col("intensity") - intensity_mean, intensity_std, default=0.0).alias(
                    "internal_liq_intensity_z"
                )
                micro_tf = micro_tf.with_columns(intensity_z).with_columns(
                    (pl.col("internal_liq_intensity_z").abs() >= pl.lit(micro_intensity_cut)).alias(
                        "internal_liq_intensity_spike_flag"
                    )
                )
                micro_tf = micro_tf.with_columns(
                    (
                        pl.col("internal_liq_ofi_flip_flag")
                        | pl.col("internal_liq_agg_flip_flag")
                        | pl.col("internal_liq_intensity_spike_flag")
                    ).alias("internal_liq_micro_confirm_flag"),
                    pl.lit("microstructure_flow").alias("internal_liq_micro_source_used"),
                ).select(
                    "instrument",
                    "anchor_tf",
                    "anchor_ts",
                    "internal_liq_micro_confirm_flag",
                    "internal_liq_ofi_flip_flag",
                    "internal_liq_agg_flip_flag",
                    "internal_liq_intensity_spike_flag",
                    "internal_liq_intensity_z",
                    "internal_liq_micro_source_used",
                )
                out = out.join(micro_tf, on=["instrument", "anchor_tf", "anchor_ts"], how="left")

        if "internal_liq_micro_confirm_flag" not in out.columns:
            out = out.with_columns(
                pl.lit(None).cast(pl.Boolean).alias("internal_liq_micro_confirm_flag"),
                pl.lit(None).cast(pl.Boolean).alias("internal_liq_ofi_flip_flag"),
                pl.lit(None).cast(pl.Boolean).alias("internal_liq_agg_flip_flag"),
                pl.lit(None).cast(pl.Boolean).alias("internal_liq_intensity_spike_flag"),
                pl.lit(None).cast(pl.Float64).alias("internal_liq_intensity_z"),
                pl.lit("none").alias("internal_liq_micro_source_used"),
            )

        out_frames.append(
            out.select(
                "instrument",
                "anchor_tf",
                pl.col("ts").alias("anchor_ts"),
                "internal_liq_high_count_L",
                "internal_liq_low_count_L",
                "internal_liq_level_high",
                "internal_liq_level_low",
                "internal_liq_nearest_side",
                "internal_liq_nearest_dist_atr",
                "internal_liq_flag",
                "internal_liq_reason_code",
                "internal_liq_source_used",
                "internal_liq_fvg_overlap_flag",
                "internal_liq_micro_confirm_flag",
                "internal_liq_ofi_flip_flag",
                "internal_liq_agg_flip_flag",
                "internal_liq_intensity_spike_flag",
                "internal_liq_intensity_z",
                "internal_liq_micro_source_used",
            )
        )

    if not out_frames:
        return _empty_keyed_frame(registry_entry)

    out = pl.concat(out_frames, how="vertical").sort(["instrument", "anchor_tf", "anchor_ts"])
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "anchor_ts"],
        where="internal_liquidity",
        allow_extra=False,
    )

    log.info("internal_liquidity: built rows=%d lookback=%d", out.height, lookback)
    return out
