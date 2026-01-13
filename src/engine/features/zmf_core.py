from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import polars as pl

from engine.core.timegrid import build_anchor_grid
from engine.features import FeatureBuildContext
from engine.features._shared import require_cols, conform_to_registry

log = logging.getLogger(__name__)


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    family_cfg: Mapping[str, Any] | None = None,
    registry_entry: Mapping[str, Any] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Phase A: deterministic "DAY_RANGE" zone for data/zones_state.
    """
    if candles is None or candles.is_empty():
        return pl.DataFrame()

    require_cols(candles, ["instrument", "ts"], where="zmf_core")

    cfg = dict(family_cfg or {})
    price_hi_col = str(cfg.get("price_hi_col", "high"))
    price_lo_col = str(cfg.get("price_lo_col", "low"))

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = list(getattr(cluster, "anchor_tfs", []) or [])
    if not anchor_tfs:
        anchor_tfs = sorted(candles.get_column("tf").unique().to_list()) if "tf" in candles.columns else ["M5"]

    out_frames: list[pl.DataFrame] = []

    for anchor_tf in anchor_tfs:
        bars = build_anchor_grid(candles, anchor_tf=str(anchor_tf), ts_col="ts", instrument_col="instrument")
        if bars.is_empty():
            continue

        zone_id = f"DAY_RANGE::{ctx.trading_day}::{anchor_tf}"

        if price_hi_col not in candles.columns or price_lo_col not in candles.columns:
            out = (
                bars.with_columns(
                    pl.lit(str(anchor_tf)).alias("anchor_tf"),
                    pl.lit(zone_id).cast(pl.Utf8).alias("zone_id"),
                    pl.lit("DAY_RANGE").cast(pl.Utf8).alias("zmf_zone_kind"),
                    pl.lit(None).cast(pl.Float64).alias("zmf_zone_lo"),
                    pl.lit(None).cast(pl.Float64).alias("zmf_zone_hi"),
                    pl.lit(None).cast(pl.Float64).alias("zmf_zone_mid"),
                    pl.lit(None).cast(pl.Float64).alias("zmf_zone_width"),
                    pl.lit("mixed").cast(pl.Utf8).alias("zmf_zone_behaviour_type_bucket"),
                    pl.lit("fresh").cast(pl.Utf8).alias("zmf_zone_freshness_bucket"),
                    pl.lit("single").cast(pl.Utf8).alias("zmf_zone_stack_depth_bucket"),
                    pl.lit("single").cast(pl.Utf8).alias("zmf_zone_htf_confluence_bucket"),
                )
                .select(
                    [
                        "instrument",
                        "anchor_tf",
                        "ts",
                        "zone_id",
                        "zmf_zone_kind",
                        "zmf_zone_lo",
                        "zmf_zone_hi",
                        "zmf_zone_mid",
                        "zmf_zone_width",
                        "zmf_zone_behaviour_type_bucket",
                        "zmf_zone_freshness_bucket",
                        "zmf_zone_stack_depth_bucket",
                        "zmf_zone_htf_confluence_bucket",
                    ]
                )
            )

            out = conform_to_registry(
                out,
                registry_entry=registry_entry,
                key_cols=["instrument", "anchor_tf", "ts", "zone_id"],
                where="zmf_core",
                allow_extra=False,
            )

            out_frames.append(out)
            continue

        c = candles
        if "tf" in c.columns:
            c_tf = c.filter(pl.col("tf") == pl.lit(str(anchor_tf)))
            c = c_tf if not c_tf.is_empty() else candles

        day_hilo = (
            c.group_by("instrument")
            .agg(
                pl.col(price_hi_col).max().cast(pl.Float64).alias("zmf_zone_hi"),
                pl.col(price_lo_col).min().cast(pl.Float64).alias("zmf_zone_lo"),
            )
            .with_columns(
                ((pl.col("zmf_zone_hi") + pl.col("zmf_zone_lo")) / 2.0).alias("zmf_zone_mid"),
                (pl.col("zmf_zone_hi") - pl.col("zmf_zone_lo")).alias("zmf_zone_width"),
            )
        )

        out = (
            bars.join(day_hilo, on="instrument", how="left")
            .with_columns(
                pl.lit(str(anchor_tf)).alias("anchor_tf"),
                pl.lit(zone_id).cast(pl.Utf8).alias("zone_id"),
                pl.lit("DAY_RANGE").cast(pl.Utf8).alias("zmf_zone_kind"),
                pl.lit("mixed").cast(pl.Utf8).alias("zmf_zone_behaviour_type_bucket"),
                pl.lit("fresh").cast(pl.Utf8).alias("zmf_zone_freshness_bucket"),
                pl.lit("single").cast(pl.Utf8).alias("zmf_zone_stack_depth_bucket"),
                pl.lit("single").cast(pl.Utf8).alias("zmf_zone_htf_confluence_bucket"),
            )
            .select(
                [
                    "instrument",
                    "anchor_tf",
                    "ts",
                    "zone_id",
                    "zmf_zone_kind",
                    "zmf_zone_lo",
                    "zmf_zone_hi",
                    "zmf_zone_mid",
                    "zmf_zone_width",
                    "zmf_zone_behaviour_type_bucket",
                    "zmf_zone_freshness_bucket",
                    "zmf_zone_stack_depth_bucket",
                    "zmf_zone_htf_confluence_bucket",
                ]
            )
        )

        out = conform_to_registry(
            out,
            registry_entry=registry_entry,
            key_cols=["instrument", "anchor_tf", "ts", "zone_id"],
            where="zmf_core",
            allow_extra=False,
        )

        out_frames.append(out)

    return pl.concat(out_frames, how="vertical") if out_frames else pl.DataFrame()
