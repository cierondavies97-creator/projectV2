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
            fvg_loc_expr,

            # OB outputs
            ob_type_expr,
            ob_high_expr,
            ob_low_expr,
            ob_origin_ts_expr,
            ob_fresh_expr,
        )

        # -----------------------------
        # FVG full logic (per instrument)
        # -----------------------------
        fvg_frames: list[pl.DataFrame] = []
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

            fvg_frames.append(
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
                )
            )

        df = pl.concat(fvg_frames, how="vertical") if fvg_frames else df
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
                pl.col("ob_origin_ts"),
                pl.col("ob_freshness_bucket"),

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
