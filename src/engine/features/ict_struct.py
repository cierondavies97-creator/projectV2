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
    displacement_range_atr_min: float = 1.2
    displacement_body_ratio_min: float = 0.6
    displacement_close_loc_min: float = 0.7

    ob_lookback_bars: int = 50
    ob_height_ticks_min: float = 8.0
    ob_mitigation_bars_max: int = 200
    ob_min_impulse_range_atr: float = 1.0
    ob_min_body_ratio: float = 0.55
    ob_max_origin_age_bars: int = 300
    ob_min_reaction_atr: float = 0.3
    ob_breaker_confirm_bars: int = 2


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
        displacement_range_atr_min=_get_float("displacement_range_atr_min", 1.2),
        displacement_body_ratio_min=_get_float("displacement_body_ratio_min", 0.6),
        displacement_close_loc_min=_get_float("displacement_close_loc_min", 0.7),
        ob_lookback_bars=_get_int("ob_lookback_bars", 50),
        ob_height_ticks_min=_get_float("ob_height_ticks_min", 8.0),
        ob_mitigation_bars_max=_get_int("ob_mitigation_bars_max", 200),
        ob_min_impulse_range_atr=_get_float("ob_min_impulse_range_atr", 1.0),
        ob_min_body_ratio=_get_float("ob_min_body_ratio", 0.55),
        ob_max_origin_age_bars=_get_int("ob_max_origin_age_bars", 300),
        ob_min_reaction_atr=_get_float("ob_min_reaction_atr", 0.3),
        ob_breaker_confirm_bars=_get_int("ob_breaker_confirm_bars", 2),
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


def _min_tick_map(ctx, default_min_tick: float = 0.0001) -> dict[str, float]:
    mp: dict[str, float] = {}
    cluster = getattr(ctx, "cluster", None)
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
        return {"__default__": float(default_min_tick)}
    mp["__default__"] = float(default_min_tick)
    return mp


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
    tick_map = _min_tick_map(ctx)

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
        # FVG location bucket
        # -----------------------------
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

        df = df.with_columns(
            eqh_expr,
            eql_expr,
            liq_grab_expr,
            fvg_loc_expr,
        )

        # -----------------------------
        # FVG/OB full logic (per instrument)
        # -----------------------------
        feature_frames: list[pl.DataFrame] = []
        eps = 1e-9
        for inst, grp in df.partition_by("instrument", maintain_order=True, as_dict=True).items():
            tick_size = tick_map.get(str(inst), tick_map.get("__default__", 0.0001))
            high = grp.get_column("high").to_list()
            low = grp.get_column("low").to_list()
            open_ = grp.get_column("open").to_list()
            close = grp.get_column("close").to_list()
            atr = grp.get_column("atr_anchor").to_list()
            ts_list = grp.get_column("ts").to_list()
            n = len(high)

            displacement_flag = [False] * n
            displacement_dir = ["none"] * n
            displacement_range_atr = [0.0] * n
            displacement_body_ratio = [0.0] * n
            displacement_close_loc = [0.0] * n

            for i in range(n):
                bar_range = float(high[i]) - float(low[i])
                atr_val = float(atr[i]) if atr[i] is not None else 0.0
                range_atr = bar_range / atr_val if atr_val > eps else 0.0
                body_ratio = abs(float(close[i]) - float(open_[i])) / max(bar_range, eps)
                close_loc = (float(close[i]) - float(low[i])) / max(bar_range, eps)
                direction = "up" if float(close[i]) >= float(open_[i]) else "down"
                displacement_range_atr[i] = range_atr
                displacement_body_ratio[i] = body_ratio
                displacement_close_loc[i] = close_loc
                displacement_dir[i] = direction
                if range_atr >= cfg.displacement_range_atr_min and body_ratio >= cfg.displacement_body_ratio_min:
                    if direction == "up" and close_loc >= cfg.displacement_close_loc_min:
                        displacement_flag[i] = True
                    elif direction == "down" and close_loc <= (1.0 - cfg.displacement_close_loc_min):
                        displacement_flag[i] = True

            fvg_direction = ["none"] * n
            fvg_gap_ticks = [0.0] * n
            fvg_upper = [None] * n
            fvg_lower = [None] * n
            fvg_mid = [None] * n
            fvg_origin_ts = [None] * n
            fvg_age_bars = [None] * n
            fvg_fill_frac = [0.0] * n
            fvg_fill_state = ["none"] * n
            fvg_fill_duration_bars = [None] * n
            fvg_was_mitigated_flag = [False] * n
            fvg_origin_impulse_atr = [None] * n
            fvg_origin_body_ratio = [None] * n
            fvg_quality = [0.0] * n

            ob_type = ["none"] * n
            ob_high = [None] * n
            ob_low = [None] * n
            ob_mid = [None] * n
            ob_height_ticks = [None] * n
            ob_origin_ts = [None] * n
            ob_age_bars = [None] * n
            ob_distance_ticks = [None] * n
            ob_freshness_bucket = ["none"] * n
            ob_quality = [0.0] * n
            ob_breaker_flag = [False] * n
            ob_breaker_dir = [None] * n
            ob_breaker_ts = [None] * n

            for i in range(n):
                lookback_start = max(0, i - int(cfg.ob_lookback_bars))
                disp_idx = None
                for j in range(i, lookback_start - 1, -1):
                    if displacement_flag[j]:
                        disp_idx = j
                        break
                if disp_idx is None:
                    continue

                disp_direction = displacement_dir[disp_idx]
                if disp_direction == "up":
                    ob_idx = None
                    for j in range(disp_idx - 1, -1, -1):
                        if float(close[j]) < float(open_[j]):
                            ob_idx = j
                            break
                    ob_dir = "bullish"
                else:
                    ob_idx = None
                    for j in range(disp_idx - 1, -1, -1):
                        if float(close[j]) > float(open_[j]):
                            ob_idx = j
                            break
                    ob_dir = "bearish"

                if ob_idx is None:
                    continue

                if displacement_range_atr[disp_idx] < cfg.ob_min_impulse_range_atr or (
                    displacement_body_ratio[disp_idx] < cfg.ob_min_body_ratio
                ):
                    continue

                ob_high_val = float(high[ob_idx])
                ob_low_val = float(low[ob_idx])
                ob_height = ob_high_val - ob_low_val
                ob_height_ticks_val = ob_height / tick_size if tick_size > eps else 0.0
                if ob_height_ticks_val < cfg.ob_height_ticks_min:
                    continue

                age_bars = i - ob_idx
                if age_bars > cfg.ob_max_origin_age_bars:
                    continue

                ob_type[i] = ob_dir
                ob_high[i] = ob_high_val
                ob_low[i] = ob_low_val
                ob_mid_val = (ob_high_val + ob_low_val) / 2.0
                ob_mid[i] = ob_mid_val
                ob_height_ticks[i] = ob_height_ticks_val
                ob_origin_ts[i] = ts_list[ob_idx]
                ob_age_bars[i] = age_bars
                ob_distance_ticks[i] = abs(float(close[i]) - ob_mid_val) / tick_size if tick_size > eps else 0.0
                ob_quality[i] = 1.0

                first_touch_index = None
                touch_price = None
                atr_at_touch = None
                for j in range(ob_idx + 1, i + 1):
                    if float(low[j]) <= ob_high_val and float(high[j]) >= ob_low_val:
                        first_touch_index = j
                        if ob_dir == "bullish":
                            touch_price = min(float(high[j]), ob_high_val)
                        else:
                            touch_price = max(float(low[j]), ob_low_val)
                        atr_at_touch = float(atr[j]) if atr[j] is not None else 0.0
                        break

                if first_touch_index is None:
                    if age_bars > cfg.ob_mitigation_bars_max:
                        ob_freshness_bucket[i] = "old"
                    else:
                        ob_freshness_bucket[i] = "fresh"
                else:
                    ob_freshness_bucket[i] = "mitigated"
                    follow_end = min(n - 1, first_touch_index + int(cfg.displacement_followthrough_bars))
                    if atr_at_touch is None or atr_at_touch <= eps or touch_price is None:
                        ob_quality[i] = 0.0
                    else:
                        if ob_dir == "bullish":
                            max_future_high = max(float(val) for val in high[first_touch_index : follow_end + 1])
                            reaction = max_future_high - float(touch_price)
                        else:
                            min_future_low = min(float(val) for val in low[first_touch_index : follow_end + 1])
                            reaction = float(touch_price) - min_future_low
                        if reaction < cfg.ob_min_reaction_atr * atr_at_touch:
                            ob_quality[i] = 0.0

                confirm_bars = int(cfg.ob_breaker_confirm_bars)
                if confirm_bars > 0:
                    count = 0
                    breaker_start = None
                    for j in range(ob_idx + 1, i + 1):
                        if ob_dir == "bullish":
                            breaker = float(close[j]) < ob_low_val
                        else:
                            breaker = float(close[j]) > ob_high_val
                        if breaker:
                            count += 1
                            if count >= confirm_bars:
                                breaker_start = j - confirm_bars + 1
                                break
                        else:
                            count = 0
                    if breaker_start is not None:
                        ob_breaker_flag[i] = True
                        ob_breaker_dir[i] = "down" if ob_dir == "bullish" else "up"
                        ob_breaker_ts[i] = ts_list[breaker_start]

            for i in range(2, n):
                bull_gap = low[i] > high[i - 2]
                bear_gap = high[i] < low[i - 2]
                if not bull_gap and not bear_gap:
                    continue

                if bull_gap:
                    lower = float(high[i - 2])
                    upper = float(low[i])
                    direction = "bull"
                else:
                    upper = float(low[i - 2])
                    lower = float(high[i])
                    direction = "bear"

                gap_px = upper - lower
                if gap_px <= 0:
                    continue
                gap_ticks = gap_px / tick_size if tick_size > eps else 0.0
                if gap_ticks < cfg.fvg_min_gap_ticks:
                    continue

                origin_range = float(high[i - 1]) - float(low[i - 1])
                atr_val = float(atr[i - 1]) if atr[i - 1] is not None else 0.0
                origin_range_atr = origin_range / atr_val if atr_val > eps else 0.0
                body_ratio = abs(float(close[i - 1]) - float(open_[i - 1])) / max(origin_range, eps)
                if (origin_range_atr < cfg.fvg_origin_impulse_range_atr_min) or (
                    body_ratio < cfg.fvg_origin_body_ratio_min
                ):
                    continue

                fvg_direction[i] = direction
                fvg_gap_ticks[i] = gap_ticks
                fvg_upper[i] = upper
                fvg_lower[i] = lower
                fvg_mid[i] = (upper + lower) / 2.0
                fvg_origin_ts[i] = ts_list[i]
                fvg_age_bars[i] = 0
                fvg_origin_impulse_atr[i] = origin_range_atr
                fvg_origin_body_ratio[i] = body_ratio
                fvg_quality[i] = 1.0

                max_origin_index = min(n - 1, i + int(cfg.fvg_max_origin_age_bars))
                max_fill_frac = 0.0
                first_touch_index: int | None = None
                touch_price = None
                atr_at_touch = None

                for j in range(i + 1, max_origin_index + 1):
                    overlap = max(0.0, min(float(high[j]), upper) - max(float(low[j]), lower))
                    fill_frac = overlap / gap_px
                    if fill_frac > max_fill_frac:
                        max_fill_frac = fill_frac
                    if fill_frac > 0 and first_touch_index is None:
                        first_touch_index = j
                        if direction == "bull":
                            touch_price = min(float(high[j]), upper)
                        else:
                            touch_price = max(float(low[j]), lower)
                        atr_at_touch = float(atr[j]) if atr[j] is not None else 0.0

                max_fill_frac = max(0.0, min(max_fill_frac, 1.0))
                fvg_fill_frac[i] = max_fill_frac

                if max_fill_frac == 0:
                    fvg_fill_state[i] = "unfilled"
                elif max_fill_frac >= 1.0:
                    fvg_fill_state[i] = "filled"
                elif max_fill_frac >= cfg.fvg_partial_fill_cut:
                    fvg_fill_state[i] = "partially_filled"
                else:
                    fvg_fill_state[i] = "unfilled"

                if first_touch_index is not None:
                    duration = first_touch_index - i
                    fvg_fill_duration_bars[i] = duration
                    fvg_was_mitigated_flag[i] = True
                    if duration > cfg.fvg_max_fill_bars:
                        fvg_quality[i] = 0.0
                    else:
                        follow_end = min(n - 1, first_touch_index + int(cfg.displacement_followthrough_bars))
                        if atr_at_touch is None or atr_at_touch <= eps or touch_price is None:
                            fvg_quality[i] = 0.0
                        else:
                            if direction == "bull":
                                max_future_high = max(float(val) for val in high[first_touch_index : follow_end + 1])
                                reaction = max_future_high - float(touch_price)
                            else:
                                min_future_low = min(float(val) for val in low[first_touch_index : follow_end + 1])
                                reaction = float(touch_price) - min_future_low
                            if reaction < cfg.fvg_min_reaction_atr * atr_at_touch:
                                fvg_quality[i] = 0.0

            feature_frames.append(
                grp.with_columns(
                    pl.Series(name="fvg_direction", values=fvg_direction, dtype=pl.Utf8),
                    pl.Series(name="fvg_gap_ticks", values=fvg_gap_ticks, dtype=pl.Float64),
                    pl.Series(name="fvg_upper", values=fvg_upper, dtype=pl.Float64),
                    pl.Series(name="fvg_lower", values=fvg_lower, dtype=pl.Float64),
                    pl.Series(name="fvg_mid", values=fvg_mid, dtype=pl.Float64),
                    pl.Series(name="fvg_origin_ts", values=fvg_origin_ts, dtype=pl.Datetime("us")),
                    pl.Series(name="fvg_age_bars", values=fvg_age_bars, dtype=pl.Int32),
                    pl.Series(name="fvg_fill_frac", values=fvg_fill_frac, dtype=pl.Float64),
                    pl.Series(name="fvg_fill_state", values=fvg_fill_state, dtype=pl.Utf8),
                    pl.Series(name="fvg_fill_duration_bars", values=fvg_fill_duration_bars, dtype=pl.Int32),
                    pl.Series(
                        name="fvg_was_mitigated_flag",
                        values=fvg_was_mitigated_flag,
                        dtype=pl.Boolean,
                    ),
                    pl.Series(name="fvg_origin_impulse_atr", values=fvg_origin_impulse_atr, dtype=pl.Float64),
                    pl.Series(name="fvg_origin_body_ratio", values=fvg_origin_body_ratio, dtype=pl.Float64),
                    pl.Series(name="fvg_quality", values=fvg_quality, dtype=pl.Float64),
                    pl.Series(name="ob_type", values=ob_type, dtype=pl.Utf8),
                    pl.Series(name="ob_high", values=ob_high, dtype=pl.Float64),
                    pl.Series(name="ob_low", values=ob_low, dtype=pl.Float64),
                    pl.Series(name="ob_mid", values=ob_mid, dtype=pl.Float64),
                    pl.Series(name="ob_height_ticks", values=ob_height_ticks, dtype=pl.Float64),
                    pl.Series(name="ob_origin_ts", values=ob_origin_ts, dtype=pl.Datetime("us")),
                    pl.Series(name="ob_age_bars", values=ob_age_bars, dtype=pl.Int32),
                    pl.Series(name="ob_distance_ticks", values=ob_distance_ticks, dtype=pl.Float64),
                    pl.Series(name="ob_freshness_bucket", values=ob_freshness_bucket, dtype=pl.Utf8),
                    pl.Series(name="ob_quality", values=ob_quality, dtype=pl.Float64),
                    pl.Series(name="ob_breaker_flag", values=ob_breaker_flag, dtype=pl.Boolean),
                    pl.Series(name="ob_breaker_dir", values=ob_breaker_dir, dtype=pl.Utf8),
                    pl.Series(name="ob_breaker_ts", values=ob_breaker_ts, dtype=pl.Datetime("us")),
                )
            )

        df = pl.concat(feature_frames, how="vertical") if feature_frames else df
        df = df.with_columns(pl.lit(str(anchor_tf)).alias("fvg_origin_tf"))

        out = df.select(
            [
                pl.col("instrument").cast(pl.Utf8, strict=False),
                pl.col("anchor_tf").cast(pl.Utf8, strict=False),
                pl.col("ts").cast(pl.Datetime("us"), strict=False),

                pl.col("fvg_direction"),
                pl.col("fvg_gap_ticks"),
                pl.col("fvg_upper"),
                pl.col("fvg_lower"),
                pl.col("fvg_mid"),
                pl.col("fvg_origin_tf"),
                pl.col("fvg_origin_ts"),
                pl.col("fvg_age_bars"),
                pl.col("fvg_fill_frac"),
                pl.col("fvg_fill_state"),
                pl.col("fvg_fill_duration_bars"),
                pl.col("fvg_was_mitigated_flag"),
                pl.col("fvg_origin_impulse_atr"),
                pl.col("fvg_origin_body_ratio"),
                pl.col("fvg_quality"),
                pl.col("fvg_location_bucket"),

                pl.col("ob_type"),
                pl.col("ob_high"),
                pl.col("ob_low"),
                pl.col("ob_mid"),
                pl.col("ob_height_ticks"),
                pl.col("ob_origin_ts"),
                pl.col("ob_age_bars"),
                pl.col("ob_distance_ticks"),
                pl.col("ob_freshness_bucket"),
                pl.col("ob_quality"),
                pl.col("ob_breaker_flag"),
                pl.col("ob_breaker_dir"),
                pl.col("ob_breaker_ts"),

                pl.col("eqh_flag"),
                pl.col("eql_flag"),
                pl.col("liq_grab_flag"),

                # exported for now; revisit ownership when ta_vol is implemented
                pl.col("atr_anchor"),
                pl.col("atr_z"),
            ]
        )

        frames.append(out)

    return pl.concat(frames, how="vertical") if frames else pl.DataFrame()
