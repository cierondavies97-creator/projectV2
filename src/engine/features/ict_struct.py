from __future__ import annotations

from dataclasses import dataclass
import json
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
    eq_level_ticks_max: float = 4.0
    eq_lookback_bars: int = 50
    liq_grab_wick_ratio_cut: float = 0.6
    liq_grab_body_ratio_max: float = 0.3
    liq_grab_followthrough_bars: int = 3
    eq_min_hits: int = 2
    eq_min_sep_bars: int = 3
    sweep_depth_ticks_min: float = 6.0
    sweep_reclaim_bars_max: int = 3
    sweep_requires_displacement: int = 0
    sweep_displacement_range_atr_min: float = 1.0
    dealing_range_lookback_bars: int = 100
    pd_index_bins: Any = 10
    pd_location_discount_max_frac: float = 0.3
    pd_location_premium_min_frac: float = 0.7
    dealing_range_min_height_atr: float = 1.0

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

    swing_lookback_bars: int = 3
    swing_strength_min_bars: int = 5
    swing_trend_dir_window: int = 6
    bos_window_bars_min: int = 2
    bos_window_bars_max: int = 20
    bos_min_distance_ticks: float = 4.0
    choch_z_cut: float = 1.0
    choch_min_distance_ticks: float = 4.0


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

    def _get_pd_bins(k: str, default: Any) -> Any:
        v = base.get(k, default)
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed
            except Exception:
                return v
        return v

    return IctStructCfg(
        eq_lookback=_get_int("eq_lookback", 20),
        eq_tol_atr_frac=_get_float("eq_tol_atr_frac", 0.10),
        eq_level_ticks_max=_get_float("eq_level_ticks_max", 4.0),
        eq_lookback_bars=_get_int("eq_lookback_bars", 50),
        liq_grab_wick_ratio_cut=_get_float("liq_grab_wick_ratio_cut", 0.6),
        liq_grab_body_ratio_max=_get_float("liq_grab_body_ratio_max", 0.3),
        liq_grab_followthrough_bars=_get_int("liq_grab_followthrough_bars", 3),
        eq_min_hits=_get_int("eq_min_hits", 2),
        eq_min_sep_bars=_get_int("eq_min_sep_bars", 3),
        sweep_depth_ticks_min=_get_float("sweep_depth_ticks_min", 6.0),
        sweep_reclaim_bars_max=_get_int("sweep_reclaim_bars_max", 3),
        sweep_requires_displacement=_get_int("sweep_requires_displacement", 0),
        sweep_displacement_range_atr_min=_get_float("sweep_displacement_range_atr_min", 1.0),
        dealing_range_lookback_bars=_get_int("dealing_range_lookback_bars", 100),
        pd_index_bins=_get_pd_bins("pd_index_bins", 10),
        pd_location_discount_max_frac=_get_float("pd_location_discount_max_frac", 0.3),
        pd_location_premium_min_frac=_get_float("pd_location_premium_min_frac", 0.7),
        dealing_range_min_height_atr=_get_float("dealing_range_min_height_atr", 1.0),
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
        swing_lookback_bars=_get_int("swing_lookback_bars", 3),
        swing_strength_min_bars=_get_int("swing_strength_min_bars", 5),
        swing_trend_dir_window=_get_int("swing_trend_dir_window", 6),
        bos_window_bars_min=_get_int("bos_window_bars_min", 2),
        bos_window_bars_max=_get_int("bos_window_bars_max", 20),
        bos_min_distance_ticks=_get_float("bos_min_distance_ticks", 4.0),
        choch_z_cut=_get_float("choch_z_cut", 1.0),
        choch_min_distance_ticks=_get_float("choch_min_distance_ticks", 4.0),
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

        df = df.with_columns(
            eqh_expr,
            eql_expr,
            liq_grab_expr,
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
            atr_z = grp.get_column("atr_z").to_list()
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

            eqh_flag = [False] * n
            eql_flag = [False] * n
            eqh_level_px = [None] * n
            eql_level_px = [None] * n
            eq_level_hit_count = [None] * n
            eq_level_span_bars = [None] * n
            liq_grab_flag = [False] * n
            liq_sweep_flag = [False] * n
            liq_sweep_side = ["none"] * n
            liq_sweep_level_px = [None] * n
            liq_sweep_depth_ticks = [None] * n
            liq_sweep_reclaim_bars = [None] * n
            liq_sweep_quality = [0.0] * n
            ict_struct_liquidity_tag = ["none"] * n
            dealing_range_high = [None] * n
            dealing_range_low = [None] * n
            dealing_range_mid = [None] * n
            pd_location_frac = [0.5] * n
            pd_location_bucket = ["equilibrium"] * n
            pd_index = [None] * n
            dealing_range_age_bars = [0] * n
            fvg_location_bucket = ["equilibrium"] * n

            swing_high = [None] * n
            swing_low = [None] * n
            swing_strength = [None] * n
            swing_trend_dir = ["range"] * n
            bos_flag = [False] * n
            bos_dir = ["none"] * n
            bos_level_px = [None] * n
            bos_distance_ticks = [None] * n
            bos_age_bars = [None] * n
            choch_flag = [False] * n
            choch_dir = ["none"] * n
            choch_level_px = [None] * n
            choch_distance_ticks = [None] * n
            choch_age_bars = [None] * n
            struct_state = ["range"] * n
            struct_trend_strength = [0.0] * n
            bos_type = ["none"] * n

            def _cluster_hits(hit_indices: list[int], levels: list[float]) -> list[dict[str, Any]]:
                clusters: list[dict[str, Any]] = []
                max_gap = cfg.eq_level_ticks_max * tick_size
                min_sep = int(cfg.eq_min_sep_bars)
                for idx, level in zip(hit_indices, levels):
                    if not clusters:
                        clusters.append({"levels": [level], "indices": [idx]})
                        continue
                    last = clusters[-1]
                    last_level = sum(last["levels"]) / len(last["levels"])
                    if abs(level - last_level) <= max_gap:
                        if idx - last["indices"][-1] >= min_sep:
                            last["levels"].append(level)
                            last["indices"].append(idx)
                    else:
                        clusters.append({"levels": [level], "indices": [idx]})
                return clusters

            def _pd_index_from_bins(location: float) -> int | None:
                bins = cfg.pd_index_bins
                if isinstance(bins, int):
                    if bins <= 0:
                        return None
                    idx = int(location * bins)
                    return max(0, min(bins - 1, idx))
                if isinstance(bins, list):
                    for edge_idx in range(len(bins) - 1):
                        if bins[edge_idx] <= location < bins[edge_idx + 1]:
                            return edge_idx
                    if bins and location == bins[-1]:
                        return len(bins) - 2
                return None

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

            eq_lookback_bars = int(cfg.eq_lookback_bars)
            sweep_reclaim_max = int(cfg.sweep_reclaim_bars_max)
            followthrough_bars = int(cfg.liq_grab_followthrough_bars)
            dealing_range_lookback = int(cfg.dealing_range_lookback_bars)
            prev_range_high = None
            prev_range_low = None
            prev_range_age = 0
            for i in range(n):
                window_start = max(0, i - eq_lookback_bars + 1)
                local_high_idx = []
                local_high_levels = []
                local_low_idx = []
                local_low_levels = []
                for j in range(window_start + 1, i):
                    if float(high[j]) >= float(high[j - 1]) and float(high[j]) >= float(high[j + 1]):
                        local_high_idx.append(j)
                        local_high_levels.append(float(high[j]))
                    if float(low[j]) <= float(low[j - 1]) and float(low[j]) <= float(low[j + 1]):
                        local_low_idx.append(j)
                        local_low_levels.append(float(low[j]))

                high_clusters = _cluster_hits(local_high_idx, local_high_levels)
                low_clusters = _cluster_hits(local_low_idx, local_low_levels)

                def _best_cluster(clusters: list[dict[str, Any]]) -> dict[str, Any] | None:
                    best = None
                    for cluster in clusters:
                        hit_count = len(cluster["levels"])
                        if hit_count < int(cfg.eq_min_hits):
                            continue
                        if best is None:
                            best = cluster
                            continue
                        if hit_count > len(best["levels"]):
                            best = cluster
                        elif hit_count == len(best["levels"]):
                            if cluster["indices"][-1] > best["indices"][-1]:
                                best = cluster
                    return best

                best_high = _best_cluster(high_clusters)
                best_low = _best_cluster(low_clusters)
                best_hit_count = None
                best_span = None

                if best_high:
                    level = sum(best_high["levels"]) / len(best_high["levels"])
                    eqh_level_px[i] = level
                    eqh_flag[i] = True
                    span = best_high["indices"][-1] - best_high["indices"][0]
                    best_hit_count = len(best_high["levels"])
                    best_span = span

                if best_low:
                    level = sum(best_low["levels"]) / len(best_low["levels"])
                    eql_level_px[i] = level
                    eql_flag[i] = True
                    span = best_low["indices"][-1] - best_low["indices"][0]
                    if best_hit_count is None or len(best_low["levels"]) > best_hit_count:
                        best_hit_count = len(best_low["levels"])
                        best_span = span

                eq_level_hit_count[i] = best_hit_count
                eq_level_span_bars[i] = best_span

                if eqh_flag[i] or eql_flag[i]:
                    if eqh_flag[i] and eql_flag[i]:
                        ict_struct_liquidity_tag[i] = "pool_both"
                    elif eqh_flag[i]:
                        ict_struct_liquidity_tag[i] = "pool_eqh"
                    else:
                        ict_struct_liquidity_tag[i] = "pool_eql"

                sweep_depth_min = cfg.sweep_depth_ticks_min * tick_size
                sweep_requires_disp = int(cfg.sweep_requires_displacement) == 1
                if eqh_level_px[i] is not None:
                    depth_px = float(high[i]) - float(eqh_level_px[i])
                    if depth_px >= sweep_depth_min:
                        reclaim_idx = None
                        for j in range(i + 1, min(n, i + sweep_reclaim_max + 1)):
                            if float(close[j]) < float(eqh_level_px[i]):
                                reclaim_idx = j
                                break
                        if reclaim_idx is not None:
                            liq_sweep_flag[i] = True
                            liq_sweep_side[i] = "eqh"
                            liq_sweep_level_px[i] = eqh_level_px[i]
                            depth_ticks = depth_px / tick_size if tick_size > eps else 0.0
                            liq_sweep_depth_ticks[i] = depth_ticks
                            liq_sweep_reclaim_bars[i] = reclaim_idx - i
                            depth_score = min(1.0, depth_ticks / max(cfg.sweep_depth_ticks_min, eps))
                            reclaim_score = 1.0 - (liq_sweep_reclaim_bars[i] / (sweep_reclaim_max + 1))
                            base_quality = max(0.0, min(1.0, (depth_score + reclaim_score) / 2.0))
                            displacement_ok = True
                            if sweep_requires_disp:
                                displacement_ok = False
                                for k in range(reclaim_idx + 1, min(n, reclaim_idx + followthrough_bars + 1)):
                                    if displacement_range_atr[k] >= cfg.sweep_displacement_range_atr_min:
                                        displacement_ok = True
                                        break
                            liq_sweep_quality[i] = base_quality if displacement_ok else 0.0

                if not liq_sweep_flag[i] and eql_level_px[i] is not None:
                    depth_px = float(eql_level_px[i]) - float(low[i])
                    if depth_px >= sweep_depth_min:
                        reclaim_idx = None
                        for j in range(i + 1, min(n, i + sweep_reclaim_max + 1)):
                            if float(close[j]) > float(eql_level_px[i]):
                                reclaim_idx = j
                                break
                        if reclaim_idx is not None:
                            liq_sweep_flag[i] = True
                            liq_sweep_side[i] = "eql"
                            liq_sweep_level_px[i] = eql_level_px[i]
                            depth_ticks = depth_px / tick_size if tick_size > eps else 0.0
                            liq_sweep_depth_ticks[i] = depth_ticks
                            liq_sweep_reclaim_bars[i] = reclaim_idx - i
                            depth_score = min(1.0, depth_ticks / max(cfg.sweep_depth_ticks_min, eps))
                            reclaim_score = 1.0 - (liq_sweep_reclaim_bars[i] / (sweep_reclaim_max + 1))
                            base_quality = max(0.0, min(1.0, (depth_score + reclaim_score) / 2.0))
                            displacement_ok = True
                            if sweep_requires_disp:
                                displacement_ok = False
                                for k in range(reclaim_idx + 1, min(n, reclaim_idx + followthrough_bars + 1)):
                                    if displacement_range_atr[k] >= cfg.sweep_displacement_range_atr_min:
                                        displacement_ok = True
                                        break
                            liq_sweep_quality[i] = base_quality if displacement_ok else 0.0

                if liq_sweep_flag[i]:
                    ict_struct_liquidity_tag[i] = f"sweep_{liq_sweep_side[i]}"

                bar_range = float(high[i]) - float(low[i])
                body_len = abs(float(close[i]) - float(open_[i]))
                wick_len = (float(high[i]) - max(float(open_[i]), float(close[i]))) + (
                    min(float(open_[i]), float(close[i])) - float(low[i])
                )
                wick_ratio = wick_len / max(bar_range, eps)
                body_ratio = body_len / max(bar_range, eps)
                if wick_ratio >= cfg.liq_grab_wick_ratio_cut and body_ratio <= cfg.liq_grab_body_ratio_max:
                    if liq_sweep_side[i] == "eqh":
                        expected_dir = "down"
                    elif liq_sweep_side[i] == "eql":
                        expected_dir = "up"
                    else:
                        expected_dir = "down" if float(close[i]) < float(open_[i]) else "up"
                    follow_ok = False
                    for j in range(i + 1, min(n, i + followthrough_bars + 1)):
                        if expected_dir == "down" and float(close[j]) < float(close[i]):
                            follow_ok = True
                            break
                        if expected_dir == "up" and float(close[j]) > float(close[i]):
                            follow_ok = True
                            break
                    if follow_ok:
                        liq_grab_flag[i] = True

                dr_start = max(0, i - dealing_range_lookback + 1)
                range_high = max(float(val) for val in high[dr_start : i + 1])
                range_low = min(float(val) for val in low[dr_start : i + 1])
                dealing_range_high[i] = range_high
                dealing_range_low[i] = range_low
                dealing_range_mid[i] = (range_high + range_low) / 2.0

                range_height = range_high - range_low
                atr_val = float(atr[i]) if atr[i] is not None else 0.0
                if atr_val > eps and range_height / atr_val >= cfg.dealing_range_min_height_atr:
                    location = (float(close[i]) - range_low) / max(range_height, eps)
                    location = max(0.0, min(1.0, location))
                    pd_location_frac[i] = location
                    if location <= cfg.pd_location_discount_max_frac:
                        pd_location_bucket[i] = "discount"
                    elif location >= cfg.pd_location_premium_min_frac:
                        pd_location_bucket[i] = "premium"
                    else:
                        pd_location_bucket[i] = "equilibrium"
                else:
                    pd_location_frac[i] = 0.5
                    pd_location_bucket[i] = "equilibrium"

                pd_index[i] = _pd_index_from_bins(pd_location_frac[i])
                if prev_range_high is not None and prev_range_low is not None:
                    if range_high == prev_range_high and range_low == prev_range_low:
                        prev_range_age += 1
                    else:
                        prev_range_age = 0
                dealing_range_age_bars[i] = prev_range_age
                prev_range_high = range_high
                prev_range_low = range_low
                fvg_location_bucket[i] = pd_location_bucket[i]

            swing_window = max(1, int(cfg.swing_lookback_bars))
            swing_min_bars = int(cfg.swing_strength_min_bars)
            trend_window = int(cfg.swing_trend_dir_window)
            last_swing_high_idx = None
            last_swing_low_idx = None
            last_bos_idx = None
            last_choch_idx = None
            struct_state_current = "range"
            swing_points: list[tuple[int, str, float]] = []

            def _trend_from_swings(points: list[tuple[int, str, float]]) -> tuple[str, float]:
                if len(points) < 2:
                    return "range", 0.0
                highs = [p for p in points if p[1] == "high"]
                lows = [p for p in points if p[1] == "low"]
                if len(highs) < 2 or len(lows) < 2:
                    return "range", 0.0
                high_dirs = [1 if highs[i][2] > highs[i - 1][2] else -1 for i in range(1, len(highs))]
                low_dirs = [1 if lows[i][2] > lows[i - 1][2] else -1 for i in range(1, len(lows))]
                high_score = sum(high_dirs)
                low_score = sum(low_dirs)
                if high_score > 0 and low_score > 0:
                    trend = "up"
                elif high_score < 0 and low_score < 0:
                    trend = "down"
                else:
                    trend = "range"

                xs = [p[0] for p in points]
                ys = [p[2] for p in points]
                x_mean = sum(xs) / len(xs)
                y_mean = sum(ys) / len(ys)
                cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
                var = sum((x - x_mean) ** 2 for x in xs)
                slope = cov / var if var > eps else 0.0
                y_var = sum((y - y_mean) ** 2 for y in ys) / len(ys)
                strength = slope / (y_var**0.5 + eps)
                return trend, strength

            for i in range(n):
                if i < swing_window or i + swing_window >= n:
                    if last_bos_idx is not None:
                        bos_age_bars[i] = i - last_bos_idx
                    if last_choch_idx is not None:
                        choch_age_bars[i] = i - last_choch_idx
                    struct_state[i] = struct_state_current
                    continue

                high_slice = high[i - swing_window : i + swing_window + 1]
                low_slice = low[i - swing_window : i + swing_window + 1]
                is_swing_high = float(high[i]) == max(high_slice)
                is_swing_low = float(low[i]) == min(low_slice)
                if is_swing_high:
                    if last_swing_low_idx is None or (i - last_swing_low_idx) >= swing_min_bars:
                        swing_high[i] = float(high[i])
                        strength = i - last_swing_low_idx if last_swing_low_idx is not None else None
                        swing_strength[i] = strength
                        last_swing_high_idx = i
                        swing_points.append((i, "high", float(high[i])))
                if is_swing_low:
                    if last_swing_high_idx is None or (i - last_swing_high_idx) >= swing_min_bars:
                        swing_low[i] = float(low[i])
                        strength = i - last_swing_high_idx if last_swing_high_idx is not None else None
                        swing_strength[i] = strength
                        last_swing_low_idx = i
                        swing_points.append((i, "low", float(low[i])))

                window_points = swing_points[-trend_window:] if trend_window > 0 else swing_points
                trend_dir, trend_strength = _trend_from_swings(window_points)
                swing_trend_dir[i] = trend_dir
                struct_trend_strength[i] = trend_strength

                last_high_idx = last_swing_high_idx
                last_low_idx = last_swing_low_idx
                if last_high_idx is not None:
                    bars_since_high = i - last_high_idx
                    if int(cfg.bos_window_bars_min) <= bars_since_high <= int(cfg.bos_window_bars_max):
                        bos_level = float(high[last_high_idx])
                        if float(close[i]) > bos_level + cfg.bos_min_distance_ticks * tick_size:
                            bos_flag[i] = True
                            bos_dir[i] = "bullish"
                            bos_level_px[i] = bos_level
                            bos_distance_ticks[i] = (float(close[i]) - bos_level) / tick_size if tick_size > eps else 0.0
                            bos_type[i] = "bos_up"
                            last_bos_idx = i
                            if struct_state_current in {"range", "bullish"}:
                                struct_state_current = "bullish"
                            else:
                                struct_state_current = "bullish"
                if last_low_idx is not None:
                    bars_since_low = i - last_low_idx
                    if int(cfg.bos_window_bars_min) <= bars_since_low <= int(cfg.bos_window_bars_max):
                        bos_level = float(low[last_low_idx])
                        if float(close[i]) < bos_level - cfg.bos_min_distance_ticks * tick_size:
                            bos_flag[i] = True
                            bos_dir[i] = "bearish"
                            bos_level_px[i] = bos_level
                            bos_distance_ticks[i] = (bos_level - float(close[i])) / tick_size if tick_size > eps else 0.0
                            bos_type[i] = "bos_down"
                            last_bos_idx = i
                            if struct_state_current in {"range", "bearish"}:
                                struct_state_current = "bearish"
                            else:
                                struct_state_current = "bearish"

                if struct_state_current == "bullish" and last_low_idx is not None:
                    bos_level = float(low[last_low_idx])
                    if float(close[i]) < bos_level - cfg.choch_min_distance_ticks * tick_size:
                        atr_z_val = float(atr_z[i]) if atr_z[i] is not None else 0.0
                        if atr_z_val >= cfg.choch_z_cut:
                            choch_flag[i] = True
                            choch_dir[i] = "bearish"
                            choch_level_px[i] = bos_level
                            choch_distance_ticks[i] = (bos_level - float(close[i])) / tick_size if tick_size > eps else 0.0
                            last_choch_idx = i
                            struct_state_current = "transition"
                if struct_state_current == "bearish" and last_high_idx is not None:
                    bos_level = float(high[last_high_idx])
                    if float(close[i]) > bos_level + cfg.choch_min_distance_ticks * tick_size:
                        atr_z_val = float(atr_z[i]) if atr_z[i] is not None else 0.0
                        if atr_z_val >= cfg.choch_z_cut:
                            choch_flag[i] = True
                            choch_dir[i] = "bullish"
                            choch_level_px[i] = bos_level
                            choch_distance_ticks[i] = (float(close[i]) - bos_level) / tick_size if tick_size > eps else 0.0
                            last_choch_idx = i
                            struct_state_current = "transition"

                if last_bos_idx is not None:
                    bos_age_bars[i] = i - last_bos_idx
                if last_choch_idx is not None:
                    choch_age_bars[i] = i - last_choch_idx
                struct_state[i] = struct_state_current

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
                    pl.Series(name="eqh_flag", values=eqh_flag, dtype=pl.Boolean),
                    pl.Series(name="eql_flag", values=eql_flag, dtype=pl.Boolean),
                    pl.Series(name="eqh_level_px", values=eqh_level_px, dtype=pl.Float64),
                    pl.Series(name="eql_level_px", values=eql_level_px, dtype=pl.Float64),
                    pl.Series(name="eq_level_hit_count", values=eq_level_hit_count, dtype=pl.Int32),
                    pl.Series(name="eq_level_span_bars", values=eq_level_span_bars, dtype=pl.Int32),
                    pl.Series(name="liq_grab_flag", values=liq_grab_flag, dtype=pl.Boolean),
                    pl.Series(name="liq_sweep_flag", values=liq_sweep_flag, dtype=pl.Boolean),
                    pl.Series(name="liq_sweep_side", values=liq_sweep_side, dtype=pl.Utf8),
                    pl.Series(name="liq_sweep_level_px", values=liq_sweep_level_px, dtype=pl.Float64),
                    pl.Series(name="liq_sweep_depth_ticks", values=liq_sweep_depth_ticks, dtype=pl.Float64),
                    pl.Series(name="liq_sweep_reclaim_bars", values=liq_sweep_reclaim_bars, dtype=pl.Int32),
                    pl.Series(name="liq_sweep_quality", values=liq_sweep_quality, dtype=pl.Float64),
                    pl.Series(name="ict_struct_liquidity_tag", values=ict_struct_liquidity_tag, dtype=pl.Utf8),
                    pl.Series(name="ict_struct_dealing_range_high", values=dealing_range_high, dtype=pl.Float64),
                    pl.Series(name="ict_struct_dealing_range_low", values=dealing_range_low, dtype=pl.Float64),
                    pl.Series(name="ict_struct_dealing_range_mid", values=dealing_range_mid, dtype=pl.Float64),
                    pl.Series(name="pd_location_frac", values=pd_location_frac, dtype=pl.Float64),
                    pl.Series(name="pd_location_bucket", values=pd_location_bucket, dtype=pl.Utf8),
                    pl.Series(name="ict_struct_pd_index", values=pd_index, dtype=pl.Int32),
                    pl.Series(name="dealing_range_age_bars", values=dealing_range_age_bars, dtype=pl.Int32),
                    pl.Series(name="fvg_location_bucket", values=fvg_location_bucket, dtype=pl.Utf8),
                    pl.Series(name="ict_struct_swing_high", values=swing_high, dtype=pl.Float64),
                    pl.Series(name="ict_struct_swing_low", values=swing_low, dtype=pl.Float64),
                    pl.Series(name="ict_struct_swing_strength", values=swing_strength, dtype=pl.Float64),
                    pl.Series(name="ict_struct_swing_trend_dir", values=swing_trend_dir, dtype=pl.Utf8),
                    pl.Series(name="bos_flag", values=bos_flag, dtype=pl.Boolean),
                    pl.Series(name="bos_dir", values=bos_dir, dtype=pl.Utf8),
                    pl.Series(name="bos_level_px", values=bos_level_px, dtype=pl.Float64),
                    pl.Series(name="bos_distance_ticks", values=bos_distance_ticks, dtype=pl.Float64),
                    pl.Series(name="bos_age_bars", values=bos_age_bars, dtype=pl.Int32),
                    pl.Series(name="choch_flag", values=choch_flag, dtype=pl.Boolean),
                    pl.Series(name="choch_dir", values=choch_dir, dtype=pl.Utf8),
                    pl.Series(name="choch_level_px", values=choch_level_px, dtype=pl.Float64),
                    pl.Series(name="choch_distance_ticks", values=choch_distance_ticks, dtype=pl.Float64),
                    pl.Series(name="choch_age_bars", values=choch_age_bars, dtype=pl.Int32),
                    pl.Series(name="struct_state", values=struct_state, dtype=pl.Utf8),
                    pl.Series(name="struct_trend_strength", values=struct_trend_strength, dtype=pl.Float64),
                    pl.Series(name="bos_type", values=bos_type, dtype=pl.Utf8),
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
                pl.col("eqh_level_px"),
                pl.col("eql_level_px"),
                pl.col("eq_level_hit_count"),
                pl.col("eq_level_span_bars"),
                pl.col("liq_grab_flag"),
                pl.col("liq_sweep_flag"),
                pl.col("liq_sweep_side"),
                pl.col("liq_sweep_level_px"),
                pl.col("liq_sweep_depth_ticks"),
                pl.col("liq_sweep_reclaim_bars"),
                pl.col("liq_sweep_quality"),
                pl.col("ict_struct_liquidity_tag"),
                pl.col("ict_struct_dealing_range_high"),
                pl.col("ict_struct_dealing_range_low"),
                pl.col("ict_struct_dealing_range_mid"),
                pl.col("pd_location_frac"),
                pl.col("pd_location_bucket"),
                pl.col("ict_struct_pd_index"),
                pl.col("dealing_range_age_bars"),
                pl.col("ict_struct_swing_high"),
                pl.col("ict_struct_swing_low"),
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
                pl.col("bos_type"),

                # exported for now; revisit ownership when ta_vol is implemented
                pl.col("atr_anchor"),
                pl.col("atr_z"),
            ]
        )

        frames.append(out)

    return pl.concat(frames, how="vertical") if frames else pl.DataFrame()
