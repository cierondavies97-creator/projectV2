from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import logging
import polars as pl

from engine.features._shared import (
    require_cols,
    ensure_sorted,
    to_anchor_tf,
    safe_div,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dev-stub: ICT structural features (bar-level)
#
# Key contract (features_step) for data/features is (instrument, anchor_tf, ts).
# features_step will also day-filter on ts and validate the anchor grid.
#
# Important Polars rule: do NOT reference a column created in the same with_columns call.
# This implementation avoids that by using pure expressions and/or sequential with_columns.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IctStructCfg:
    # Equal-high/low detection
    eq_lookback: int = 20
    eq_tol_atr_frac: float = 0.10  # tolerance = atr*frac + min_tick

    # ATR (for tolerances; also exported for now)
    atr_window: int = 14

    # FVG detection and location bucket
    fvg_location_window: int = 20
    fvg_min_gap_ticks: float = 8.0
    fvg_max_fill_bars: int = 6
    fvg_partial_fill_cut: float = 0.5
    fvg_origin_impulse_range_atr_min: float = 1.0
    fvg_origin_body_ratio_min: float = 0.55
    fvg_max_origin_age_bars: int = 200
    fvg_min_reaction_atr: float = 0.3
    displacement_followthrough_bars: int = 2


def _cfg_from(family_cfg: Mapping[str, Any] | None, registry_entry: Mapping[str, Any] | None) -> IctStructCfg:
    # Lightweight merge: registry_entry overrides family_cfg overrides defaults.
    base = {}
    if isinstance(family_cfg, Mapping):
        base.update(dict(family_cfg))
    if isinstance(registry_entry, Mapping):
        base.update(dict(registry_entry))

    def _get_int(k: str, default: int) -> int:
        v = base.get(k, default)
        try:
            return int(v)
        except Exception:
            return default

    def _get_float(k: str, default: float) -> float:
        v = base.get(k, default)
        try:
            return float(v)
        except Exception:
            return default

    return IctStructCfg(
        eq_lookback=_get_int("eq_lookback", 20),
        eq_tol_atr_frac=_get_float("eq_tol_atr_frac", 0.10),
        atr_window=_get_int("atr_window", 14),
        fvg_location_window=_get_int("fvg_location_window", 20),
        fvg_min_gap_ticks=_get_float("fvg_min_gap_ticks", 8.0),
        fvg_max_fill_bars=_get_int("fvg_max_fill_bars", 6),
        fvg_partial_fill_cut=_get_float("fvg_partial_fill_cut", 0.5),
        fvg_origin_impulse_range_atr_min=_get_float("fvg_origin_impulse_range_atr_min", 1.0),
        fvg_origin_body_ratio_min=_get_float("fvg_origin_body_ratio_min", 0.55),
        fvg_max_origin_age_bars=_get_int("fvg_max_origin_age_bars", 200),
        fvg_min_reaction_atr=_get_float("fvg_min_reaction_atr", 0.3),
        displacement_followthrough_bars=_get_int("displacement_followthrough_bars", 2),
    )


def _min_tick_expr(ctx, default_min_tick: float = 0.0001) -> pl.Expr:
    """
    Build an Expr that maps instrument -> min_tick.
    Falls back to default_min_tick if the plan doesn't provide ticks.
    """
    mp: dict[str, float] = {}
    cluster = getattr(ctx, "cluster", None)

    # cluster.instruments may be list[str] or list[InstrumentSpec]
    for inst in getattr(cluster, "instruments", []) or []:
        if isinstance(inst, str):
            continue
        name = getattr(inst, "instrument", None) or getattr(inst, "symbol", None)
        mt = getattr(inst, "min_tick", None) or getattr(inst, "tick_size", None)
        if name and mt is not None:
            try:
                mp[str(name)] = float(mt)
            except Exception:
                pass

    if not mp:
        return pl.lit(float(default_min_tick))

    return (
        pl.col("instrument")
        .cast(pl.Utf8, strict=False)
        .replace(mp, default=float(default_min_tick))
        .cast(pl.Float64, strict=False)
    )


def build_feature_frame(
    *,
    ctx,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    external_df: pl.DataFrame | None = None,  # tolerated alias
    family_cfg: Mapping[str, Any] | None = None,
    registry_entry: Mapping[str, Any] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Emit bar-level ICT structure dev-stub features for data/features.

    Output keys (required): instrument, anchor_tf, ts
    """
    if candles is None or candles.is_empty():
        return pl.DataFrame()

    cfg = _cfg_from(family_cfg, registry_entry)
    require_cols(candles, ["instrument", "tf", "ts", "open", "high", "low", "close"], where="ict_struct")
    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = list(getattr(cluster, "anchor_tfs", []) or [])
    if not anchor_tfs:
        # If not provided, treat existing candles tfs as anchors (dev-safe)
        anchor_tfs = sorted(set(candles.get_column("tf").cast(pl.Utf8, strict=False).to_list()))

    tick_expr = _min_tick_expr(ctx)

    frames: list[pl.DataFrame] = []

    for anchor_tf in anchor_tfs:
        df = candles.filter(pl.col("tf") == pl.lit(str(anchor_tf)))
        if df.is_empty():
            # allow M1-only input; aggregate to anchor if needed
            df = to_anchor_tf(candles, anchor_tf=str(anchor_tf), where="ict_struct")
        if df.is_empty():
            continue

        df = ensure_sorted(df, by=["instrument", "ts"]).with_columns(
            pl.lit(str(anchor_tf)).alias("anchor_tf"),
            pl.col("ts").cast(pl.Datetime("us"), strict=False).alias("ts"),
        )
        df = df.with_columns(pl.int_range(0, pl.len()).over("instrument").alias("_bar_index"))

        # -----------------------------
        # ATR/TR (dev stub)
        # -----------------------------
        prev_close = pl.col("close").shift(1).over("instrument")
        tr_expr = (
            pl.max_horizontal(
                (pl.col("high") - pl.col("low")).abs(),
                (pl.col("high") - prev_close).abs(),
                (pl.col("low") - prev_close).abs(),
            )
            .fill_null(0.0)
            .cast(pl.Float64, strict=False)
        )

        df = df.with_columns(tr_expr.alias("_tr"))        # Compute ATR as a concrete column first (no nested windows)
        df = df.with_columns(
            pl.col("_tr")
            .rolling_mean(window_size=int(cfg.atr_window), min_periods=1)
            .over("instrument")
            .cast(pl.Float64, strict=False)
            .alias("atr_anchor")
        )

        # Simple z-score of ATR via rolling stats (computed from the column)
        atr_mean = pl.col("atr_anchor").rolling_mean(window_size=50, min_periods=10).over("instrument")
        atr_std = pl.col("atr_anchor").rolling_std(window_size=50, min_periods=10).over("instrument")

        df = df.with_columns(
            safe_div((pl.col("atr_anchor") - atr_mean), atr_std, default=0.0)
            .cast(pl.Float64, strict=False)
            .alias("atr_z")
        )
        # -----------------------------
        # Equal highs/lows + liquidity grab (dev stub)
        # -----------------------------
        lookback = int(cfg.eq_lookback)
        prev_max = (
            pl.col("high")
            .shift(1)
            .rolling_max(window_size=lookback, min_periods=1)
            .over("instrument")
        )
        prev_min = (
            pl.col("low")
            .shift(1)
            .rolling_min(window_size=lookback, min_periods=1)
            .over("instrument")
        )

        tol_expr = (pl.col("atr_anchor").fill_null(0.0) * pl.lit(float(cfg.eq_tol_atr_frac))) + tick_expr

        eqh_expr = ((pl.col("high") - prev_max).abs() <= tol_expr).fill_null(False).alias("eqh_flag")
        eql_expr = ((pl.col("low") - prev_min).abs() <= tol_expr).fill_null(False).alias("eql_flag")

        liq_grab_expr = (
            ((pl.col("high") > (prev_max + tol_expr)) | (pl.col("low") < (prev_min - tol_expr)))
            .fill_null(False)
            .alias("liq_grab_flag")
        )

        # -----------------------------
        # FVG (dev stub, 3-bar gap)
        #   bullish: low_t > high_{t-2}
        #   bearish: high_t < low_{t-2}
        # -----------------------------
        hi_2 = pl.col("high").shift(2).over("instrument")
        lo_2 = pl.col("low").shift(2).over("instrument")

        bull_gap_px = (pl.col("low") - hi_2).cast(pl.Float64, strict=False)
        bear_gap_px = (lo_2 - pl.col("high")).cast(pl.Float64, strict=False)

        fvg_dir_raw = (
            pl.when(bull_gap_px > 0)
            .then(pl.lit("bull"))
            .when(bear_gap_px > 0)
            .then(pl.lit("bear"))
            .otherwise(pl.lit("none"))
        )

        fvg_lower_raw = (
            pl.when(fvg_dir_raw == pl.lit("bull"))
            .then(hi_2)
            .when(fvg_dir_raw == pl.lit("bear"))
            .then(pl.col("high"))
            .otherwise(pl.lit(None))
        ).cast(pl.Float64, strict=False)

        fvg_upper_raw = (
            pl.when(fvg_dir_raw == pl.lit("bull"))
            .then(pl.col("low"))
            .when(fvg_dir_raw == pl.lit("bear"))
            .then(lo_2)
            .otherwise(pl.lit(None))
        ).cast(pl.Float64, strict=False)

        fvg_gap_px_expr = (
            pl.when(fvg_lower_raw.is_not_null() & fvg_upper_raw.is_not_null())
            .then((fvg_upper_raw - fvg_lower_raw).abs())
            .otherwise(pl.lit(0.0))
            .cast(pl.Float64, strict=False)
        )

        fvg_gap_ticks_expr = safe_div(fvg_gap_px_expr, tick_expr, default=0.0).alias("fvg_gap_ticks")

        fvg_dir_expr = (
            pl.when(fvg_gap_ticks_expr >= pl.lit(float(cfg.fvg_min_gap_ticks)))
            .then(fvg_dir_raw)
            .otherwise(pl.lit("none"))
            .alias("fvg_direction")
        )

        fvg_gap_ticks_expr = (
            pl.when(fvg_gap_ticks_expr >= pl.lit(float(cfg.fvg_min_gap_ticks)))
            .then(fvg_gap_ticks_expr)
            .otherwise(pl.lit(0.0))
            .cast(pl.Float64, strict=False)
            .alias("fvg_gap_ticks")
        )

        fvg_fill_state_expr = pl.lit("none").alias("fvg_fill_state")

        mid_expr = ((pl.col("high") + pl.col("low")) / pl.lit(2.0)).cast(pl.Float64, strict=False)
        mid_mean_expr = (
            mid_expr.rolling_mean(window_size=int(cfg.fvg_location_window), min_periods=int(cfg.fvg_location_window))
            .over("instrument")
        )

        fvg_loc_expr = (
            pl.when(mid_mean_expr.is_null())
            .then(pl.lit("unknown"))
            .when(pl.col("close") > mid_mean_expr)
            .then(pl.lit("upper"))
            .when(pl.col("close") < mid_mean_expr)
            .then(pl.lit("lower"))
            .otherwise(pl.lit("mid"))
            .alias("fvg_location_bucket")
        )

        # -----------------------------
        # Order block (dev stub)
        #   flip events between bull/bear candles; assigns prior bar as "OB"
        # -----------------------------
        is_bull = (pl.col("close") >= pl.col("open")).fill_null(False)
        prev_is_bull = is_bull.shift(1).over("instrument").fill_null(is_bull)

        flip_to_bull = (is_bull & (~prev_is_bull)).fill_null(False)
        flip_to_bear = ((~is_bull) & prev_is_bull).fill_null(False)
        ob_active = (flip_to_bull | flip_to_bear).fill_null(False)

        ob_type_expr = (
            pl.when(flip_to_bull)
            .then(pl.lit("bullish"))
            .when(flip_to_bear)
            .then(pl.lit("bearish"))
            .otherwise(pl.lit("none"))
            .alias("ob_type")
        )

        ob_high_expr = (
            pl.when(ob_active)
            .then(pl.col("high").shift(1).over("instrument"))
            .otherwise(pl.lit(None))
            .cast(pl.Float64, strict=False)
            .alias("ob_high")
        )

        ob_low_expr = (
            pl.when(ob_active)
            .then(pl.col("low").shift(1).over("instrument"))
            .otherwise(pl.lit(None))
            .cast(pl.Float64, strict=False)
            .alias("ob_low")
        )

        ob_origin_ts_expr = (
            pl.when(ob_active)
            .then(pl.col("ts").shift(1).over("instrument"))
            .otherwise(pl.lit(None))
            .cast(pl.Datetime("us"), strict=False)
            .alias("ob_origin_ts")
        )

        ob_fresh_expr = (
            pl.when(ob_active)
            .then(pl.lit("fresh"))
            .otherwise(pl.lit("none"))
            .alias("ob_freshness_bucket")
        )

        df = df.with_columns(
            eqh_expr,
            eql_expr,
            liq_grab_expr,

            # FVG outputs
            fvg_dir_expr,
            fvg_gap_ticks_expr,
            fvg_fill_state_expr,
            fvg_loc_expr,

            # OB outputs
            ob_type_expr,
            ob_high_expr,
            ob_low_expr,
            ob_origin_ts_expr,
            ob_fresh_expr,
        )

        # -----------------------------
        # Dev-stub annotations (full registry surface)
        # -----------------------------
        range_expr = (pl.col("high") - pl.col("low")).abs().cast(pl.Float64, strict=False)
        body_expr = (pl.col("close") - pl.col("open")).abs().cast(pl.Float64, strict=False)
        close_loc_expr = safe_div((pl.col("close") - pl.col("low")), range_expr, default=0.5).alias(
            "displacement_close_loc"
        )
        body_ratio_expr = safe_div(body_expr, range_expr, default=0.0).alias("displacement_body_ratio")
        displacement_range_atr_expr = safe_div(range_expr, pl.col("atr_anchor"), default=0.0).alias(
            "displacement_range_atr"
        )

        df = df.with_columns(
            range_expr.alias("_range"),
            body_expr.alias("_body"),
            close_loc_expr,
            body_ratio_expr,
            displacement_range_atr_expr,
        )

        df = df.with_columns(
            pl.when(pl.col("displacement_range_atr") > pl.lit(1.0))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("displacement_flag"),
            pl.when(pl.col("close") > pl.col("open"))
            .then(pl.lit("up"))
            .when(pl.col("close") < pl.col("open"))
            .then(pl.lit("down"))
            .otherwise(pl.lit("none"))
            .alias("displacement_dir"),
            (
                (pl.col("displacement_body_ratio") + pl.col("displacement_close_loc")) / pl.lit(2.0)
            ).cast(pl.Float64, strict=False)
            .alias("displacement_quality"),
        )

        df = df.with_columns(
            pl.col("atr_z")
            .map_elements(
                lambda v: "low" if v is not None and v < 0 else ("high" if v is not None and v > 1 else "medium")
            )
            .cast(pl.Utf8, strict=False)
            .alias("vol_regime"),
        )

        df = df.with_columns(
            pl.col("high")
            .rolling_max(window_size=lookback, min_periods=1)
            .over("instrument")
            .cast(pl.Float64, strict=False)
            .alias("ict_struct_dealing_range_high"),
            pl.col("low")
            .rolling_min(window_size=lookback, min_periods=1)
            .over("instrument")
            .cast(pl.Float64, strict=False)
            .alias("ict_struct_dealing_range_low"),
        )

        df = df.with_columns(
            ((pl.col("ict_struct_dealing_range_high") + pl.col("ict_struct_dealing_range_low")) / pl.lit(2.0))
            .cast(pl.Float64, strict=False)
            .alias("ict_struct_dealing_range_mid")
        )

        df = df.with_columns(
            safe_div(
                (pl.col("close") - pl.col("ict_struct_dealing_range_low")),
                (pl.col("ict_struct_dealing_range_high") - pl.col("ict_struct_dealing_range_low")),
                default=0.5,
            )
            .cast(pl.Float64, strict=False)
            .alias("pd_location_frac")
        )

        df = df.with_columns(
            pl.when(pl.col("pd_location_frac") <= pl.lit(0.30))
            .then(pl.lit("discount"))
            .when(pl.col("pd_location_frac") >= pl.lit(0.70))
            .then(pl.lit("premium"))
            .otherwise(pl.lit("equilibrium"))
            .alias("pd_location_bucket"),
            pl.lit(0).cast(pl.Int32).alias("dealing_range_age_bars"),
        )

        df = df.with_columns(
            pl.col("high").shift(1).over("instrument").cast(pl.Float64, strict=False).alias("ict_struct_swing_high"),
            pl.col("low").shift(1).over("instrument").cast(pl.Float64, strict=False).alias("ict_struct_swing_low"),
            pl.lit(0).cast(pl.Int32).alias("ict_struct_pd_index"),
            pl.lit(0).cast(pl.Int32).alias("ict_struct_swing_strength"),
            pl.lit("none").cast(pl.Utf8).alias("ict_struct_swing_trend_dir"),
        )

        df = df.with_columns(
            pl.lit(False).alias("bos_flag"),
            pl.lit("none").cast(pl.Utf8).alias("bos_dir"),
            pl.lit(None).cast(pl.Float64).alias("bos_level_px"),
            pl.lit(0.0).cast(pl.Float64).alias("bos_distance_ticks"),
            pl.lit(0).cast(pl.Int32).alias("bos_age_bars"),
            pl.lit(False).alias("choch_flag"),
            pl.lit("none").cast(pl.Utf8).alias("choch_dir"),
            pl.lit(None).cast(pl.Float64).alias("choch_level_px"),
            pl.lit(0.0).cast(pl.Float64).alias("choch_distance_ticks"),
            pl.lit(0).cast(pl.Int32).alias("choch_age_bars"),
            pl.lit("none").cast(pl.Utf8).alias("struct_state"),
            pl.lit(0.0).cast(pl.Float64).alias("struct_trend_strength"),
        )

        df = df.with_columns(
            pl.when(pl.col("eqh_flag"))
            .then(prev_max.cast(pl.Float64, strict=False))
            .otherwise(pl.lit(None).cast(pl.Float64))
            .alias("eqh_level_px"),
            pl.when(pl.col("eql_flag"))
            .then(prev_min.cast(pl.Float64, strict=False))
            .otherwise(pl.lit(None).cast(pl.Float64))
            .alias("eql_level_px"),
            (pl.col("eqh_flag").cast(pl.Int32) + pl.col("eql_flag").cast(pl.Int32)).alias("eq_level_hit_count"),
            pl.lit(lookback).cast(pl.Int32).alias("eq_level_span_bars"),
        )

        df = df.with_columns(
            pl.when(pl.col("high") > (prev_max + tol_expr))
            .then(pl.lit("buyside"))
            .when(pl.col("low") < (prev_min - tol_expr))
            .then(pl.lit("sellside"))
            .otherwise(pl.lit("none"))
            .alias("liq_sweep_side")
        )

        df = df.with_columns(
            pl.col("liq_grab_flag").cast(pl.Boolean).alias("liq_sweep_flag"),
            pl.when(pl.col("liq_sweep_side") == pl.lit("buyside"))
            .then(prev_max.cast(pl.Float64, strict=False))
            .when(pl.col("liq_sweep_side") == pl.lit("sellside"))
            .then(prev_min.cast(pl.Float64, strict=False))
            .otherwise(pl.lit(None).cast(pl.Float64))
            .alias("liq_sweep_level_px"),
            pl.when(pl.col("liq_sweep_side") == pl.lit("buyside"))
            .then(safe_div((pl.col("high") - prev_max), tick_expr, default=0.0))
            .when(pl.col("liq_sweep_side") == pl.lit("sellside"))
            .then(safe_div((prev_min - pl.col("low")), tick_expr, default=0.0))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float64, strict=False)
            .alias("liq_sweep_depth_ticks"),
            pl.lit(0).cast(pl.Int32).alias("liq_sweep_reclaim_bars"),
            pl.when(pl.col("liq_grab_flag"))
            .then(pl.lit(0.5))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float64, strict=False)
            .alias("liq_sweep_quality"),
            pl.when(pl.col("liq_grab_flag"))
            .then(pl.lit("grab"))
            .otherwise(pl.lit("none"))
            .alias("ict_struct_liquidity_tag"),
        )

        df = df.with_columns(
            ((pl.col("ob_high") + pl.col("ob_low")) / pl.lit(2.0)).cast(pl.Float64, strict=False).alias("ob_mid"),
            safe_div((pl.col("ob_high") - pl.col("ob_low")), tick_expr, default=0.0)
            .cast(pl.Float64, strict=False)
            .alias("ob_height_ticks"),
            pl.when(ob_active).then(pl.lit(0)).otherwise(pl.lit(None)).cast(pl.Int32).alias("ob_age_bars"),
        )

        df = df.with_columns(
            safe_div((pl.col("close") - pl.col("ob_mid")).abs(), tick_expr, default=0.0)
            .cast(pl.Float64, strict=False)
            .alias("ob_distance_ticks"),
            pl.when(ob_active).then(pl.lit(0.5)).otherwise(pl.lit(0.0)).cast(pl.Float64).alias("ob_quality"),
            pl.lit(False).alias("ob_breaker_flag"),
            pl.lit("none").cast(pl.Utf8).alias("ob_breaker_dir"),
            pl.lit(None).cast(pl.Datetime("us")).alias("ob_breaker_ts"),
        )

        df = df.with_columns(
            pl.when(pl.col("fvg_direction") == pl.lit("none"))
            .then(pl.lit(0))
            .otherwise(pl.lit(0))
            .cast(pl.Int32)
            .alias("fvg_fill_duration_bars"),
            pl.lit(False).alias("fvg_was_mitigated_flag"),
            pl.lit("none").cast(pl.Utf8).alias("bos_type"),
            pl.lit("none").cast(pl.Utf8).alias("ict_struct_context_tag"),
            pl.lit("none").cast(pl.Utf8).alias("mms_phase"),
            pl.lit(0.0).cast(pl.Float64).alias("mms_confidence"),
        )

        out = df.select(
            [
                pl.col("instrument").cast(pl.Utf8, strict=False),
                pl.col("anchor_tf").cast(pl.Utf8, strict=False),
                pl.col("ts").cast(pl.Datetime("us"), strict=False),

                pl.col("fvg_direction"),
                pl.col("fvg_gap_ticks"),
                pl.col("fvg_origin_tf"),
                pl.col("fvg_origin_ts"),
                pl.col("fvg_fill_state"),
                pl.col("fvg_location_bucket"),
                pl.col("fvg_upper"),
                pl.col("fvg_lower"),
                pl.col("fvg_mid"),
                pl.col("fvg_age_bars"),
                pl.col("fvg_fill_frac"),
                pl.col("fvg_quality"),
                pl.col("fvg_origin_impulse_atr"),
                pl.col("fvg_origin_body_ratio"),

                pl.col("ob_type"),
                pl.col("ob_high"),
                pl.col("ob_low"),
                pl.col("ob_origin_ts"),
                pl.col("ob_freshness_bucket"),
                pl.col("ob_mid"),
                pl.col("ob_height_ticks"),
                pl.col("ob_age_bars"),
                pl.col("ob_distance_ticks"),
                pl.col("ob_quality"),
                pl.col("ob_breaker_flag"),
                pl.col("ob_breaker_dir"),
                pl.col("ob_breaker_ts"),

                pl.col("eqh_flag"),
                pl.col("eql_flag"),
                pl.col("liq_grab_flag"),
                pl.col("ict_struct_liquidity_tag"),
                pl.col("eqh_level_px"),
                pl.col("eql_level_px"),
                pl.col("eq_level_hit_count"),
                pl.col("eq_level_span_bars"),
                pl.col("liq_sweep_flag"),
                pl.col("liq_sweep_side"),
                pl.col("liq_sweep_level_px"),
                pl.col("liq_sweep_depth_ticks"),
                pl.col("liq_sweep_reclaim_bars"),
                pl.col("liq_sweep_quality"),

                pl.col("ict_struct_swing_high"),
                pl.col("ict_struct_swing_low"),
                pl.col("ict_struct_pd_index"),
                pl.col("ict_struct_swing_strength"),
                pl.col("ict_struct_swing_trend_dir"),

                pl.col("bos_flag"),
                pl.col("bos_dir"),
                pl.col("bos_level_px"),
                pl.col("bos_distance_ticks"),
                pl.col("bos_age_bars"),
                pl.col("choch_flag"),
                pl.col("choch_dir"),
                pl.col("choch_level_px"),
                pl.col("choch_distance_ticks"),
                pl.col("choch_age_bars"),
                pl.col("struct_state"),
                pl.col("struct_trend_strength"),

                pl.col("ict_struct_dealing_range_high"),
                pl.col("ict_struct_dealing_range_low"),
                pl.col("ict_struct_dealing_range_mid"),
                pl.col("pd_location_frac"),
                pl.col("pd_location_bucket"),
                pl.col("dealing_range_age_bars"),

                pl.col("fvg_fill_duration_bars"),
                pl.col("fvg_was_mitigated_flag"),
                pl.col("bos_type"),
                pl.col("ict_struct_context_tag"),

                pl.col("displacement_flag"),
                pl.col("displacement_dir"),
                pl.col("displacement_range_atr"),
                pl.col("displacement_body_ratio"),
                pl.col("displacement_close_loc"),
                pl.col("displacement_quality"),

                # exported for now; revisit ownership when ta_vol is implemented
                pl.col("atr_anchor"),
                pl.col("atr_z"),
                pl.col("vol_regime"),

                pl.col("mms_phase"),
                pl.col("mms_confidence"),
            ]
        )

        frames.append(out)

    return pl.concat(frames, how="vertical") if frames else pl.DataFrame()
