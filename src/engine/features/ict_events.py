from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry
from engine.features.ict_struct import build_feature_frame as build_ict_struct

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
            "ts": pl.Series([], dtype=pl.Datetime("us")),
            "event_type": pl.Series([], dtype=pl.Utf8),
        }
    )


def _event_frame(
    df: pl.DataFrame,
    *,
    event_type: str,
    event_side: pl.Expr,
    filter_expr: pl.Expr,
    ref_level: pl.Expr | None = None,
    ref_low: pl.Expr | None = None,
    ref_high: pl.Expr | None = None,
    event_strength: pl.Expr | None = None,
    event_quality: pl.Expr | None = None,
    extra_cols: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    base_cols: list[pl.Expr] = [
        pl.col("instrument"),
        pl.col("anchor_tf"),
        pl.col("ts"),
        pl.lit(event_type).alias("event_type"),
        event_side.alias("event_side"),
        (ref_level if ref_level is not None else pl.lit(None).cast(pl.Float64)).alias("ref_level"),
        (ref_low if ref_low is not None else pl.lit(None).cast(pl.Float64)).alias("ref_low"),
        (ref_high if ref_high is not None else pl.lit(None).cast(pl.Float64)).alias("ref_high"),
        (event_strength if event_strength is not None else pl.lit(None).cast(pl.Float64)).alias("event_strength"),
        (event_quality if event_quality is not None else pl.lit(None).cast(pl.Float64)).alias("event_quality"),
        pl.lit("ict_struct").alias("event_source"),
    ]
    if extra_cols:
        base_cols.extend(extra_cols)
    return df.filter(filter_expr).select(base_cols)


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    external_df: pl.DataFrame | None = None,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Canonical ICT event stream derived from ict_struct evidence.

    Table: data/market_events
    Keys : instrument, anchor_tf, ts, event_type
    """
    if candles is None or candles.is_empty():
        log.warning("ict_events: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    struct_df = build_ict_struct(
        ctx=ctx,
        candles=candles,
        ticks=ticks,
        macro=macro,
        external=external,
        external_df=external_df,
        family_cfg=family_cfg,
        registry_entry=None,
    )
    if struct_df.is_empty():
        return _empty_keyed_frame(registry_entry)

    events: list[pl.DataFrame] = []

    events.append(
        _event_frame(
            struct_df,
            event_type="FVG",
            event_side=pl.col("fvg_direction"),
            filter_expr=pl.col("fvg_direction") != pl.lit("none"),
            ref_level=pl.col("fvg_mid"),
            ref_low=pl.col("fvg_lower"),
            ref_high=pl.col("fvg_upper"),
            event_strength=pl.col("fvg_quality"),
            event_quality=pl.col("fvg_quality"),
            extra_cols=[
                pl.col("fvg_gap_ticks"),
                pl.col("fvg_fill_state"),
                pl.col("fvg_fill_frac"),
                pl.col("fvg_age_bars"),
                pl.col("fvg_origin_ts"),
            ],
        )
    )

    events.append(
        _event_frame(
            struct_df,
            event_type="OB",
            event_side=pl.col("ob_type"),
            filter_expr=pl.col("ob_type") != pl.lit("none"),
            ref_level=pl.col("ob_mid"),
            ref_low=pl.col("ob_low"),
            ref_high=pl.col("ob_high"),
            event_strength=pl.col("ob_quality"),
            event_quality=pl.col("ob_quality"),
            extra_cols=[
                pl.col("ob_height_ticks"),
                pl.col("ob_age_bars"),
                pl.col("ob_freshness_bucket"),
                pl.col("ob_breaker_flag"),
            ],
        )
    )

    events.append(
        _event_frame(
            struct_df,
            event_type="LIQ_SWEEP",
            event_side=pl.col("liq_sweep_side"),
            filter_expr=pl.col("liq_sweep_flag") == pl.lit(True),
            ref_level=pl.col("liq_sweep_level_px"),
            event_strength=pl.col("liq_sweep_quality"),
            event_quality=pl.col("liq_sweep_quality"),
            extra_cols=[
                pl.col("liq_sweep_depth_ticks"),
                pl.col("liq_sweep_reclaim_bars"),
            ],
        )
    )

    events.append(
        _event_frame(
            struct_df,
            event_type="EQ_HIGH",
            event_side=pl.lit("high"),
            filter_expr=pl.col("eqh_flag") == pl.lit(True),
            ref_level=pl.col("eqh_level_px"),
            event_strength=pl.col("eq_level_hit_count").cast(pl.Float64),
            event_quality=pl.col("eq_level_hit_count").cast(pl.Float64),
            extra_cols=[
                pl.col("eq_level_hit_count"),
                pl.col("eq_level_span_bars"),
            ],
        )
    )

    events.append(
        _event_frame(
            struct_df,
            event_type="EQ_LOW",
            event_side=pl.lit("low"),
            filter_expr=pl.col("eql_flag") == pl.lit(True),
            ref_level=pl.col("eql_level_px"),
            event_strength=pl.col("eq_level_hit_count").cast(pl.Float64),
            event_quality=pl.col("eq_level_hit_count").cast(pl.Float64),
            extra_cols=[
                pl.col("eq_level_hit_count"),
                pl.col("eq_level_span_bars"),
            ],
        )
    )

    events.append(
        _event_frame(
            struct_df,
            event_type="BOS",
            event_side=pl.col("bos_dir"),
            filter_expr=pl.col("bos_flag") == pl.lit(True),
            ref_level=pl.col("bos_level_px"),
            event_strength=pl.col("bos_distance_ticks"),
            event_quality=pl.col("bos_distance_ticks"),
            extra_cols=[
                pl.col("bos_distance_ticks"),
                pl.col("bos_age_bars"),
            ],
        )
    )

    events.append(
        _event_frame(
            struct_df,
            event_type="CHOCH",
            event_side=pl.col("choch_dir"),
            filter_expr=pl.col("choch_flag") == pl.lit(True),
            ref_level=pl.col("choch_level_px"),
            event_strength=pl.col("choch_distance_ticks"),
            event_quality=pl.col("choch_distance_ticks"),
            extra_cols=[
                pl.col("choch_distance_ticks"),
                pl.col("choch_age_bars"),
            ],
        )
    )

    events.append(
        _event_frame(
            struct_df,
            event_type="DISPLACEMENT",
            event_side=pl.col("displacement_dir"),
            filter_expr=pl.col("displacement_flag") == pl.lit(True),
            event_strength=pl.col("displacement_quality"),
            event_quality=pl.col("displacement_quality"),
            extra_cols=[
                pl.col("displacement_range_atr"),
                pl.col("displacement_body_ratio"),
                pl.col("displacement_close_loc"),
            ],
        )
    )

    events = [evt for evt in events if not evt.is_empty()]
    if not events:
        return _empty_keyed_frame(registry_entry)

    out = pl.concat(events, how="vertical").sort(["instrument", "anchor_tf", "ts", "event_type"])
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "ts", "event_type"],
        where="ict_events",
        allow_extra=False,
    )

    log.info("ict_events: built rows=%d", out.height)
    return out
