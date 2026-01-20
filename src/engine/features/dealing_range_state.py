from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry
from engine.features.market_structure.dealing_range import fold_state_machine

log = logging.getLogger(__name__)


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg: dict[str, object] = {}
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("dealing_range_state", {}) or {})
    if isinstance(family_cfg, Mapping):
        cfg.update(family_cfg)
    return cfg


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Table: data/features
    Keys : instrument, anchor_tf, ts
    """
    if candles is None or candles.is_empty():
        log.warning("dealing_range_state: candles empty; returning empty frame")
        return pl.DataFrame()

    required = {"instrument", "tf", "ts", "high", "low", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("dealing_range_state: missing columns=%s; returning empty frame", missing)
        return pl.DataFrame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("dealing_range_state: ctx.cluster.anchor_tfs empty; returning empty frame")
        return pl.DataFrame()

    cfg = _merge_cfg(ctx, family_cfg)

    frames: list[pl.DataFrame] = []
    for anchor_tf in anchor_tfs:
        tf_str = str(anchor_tf)
        c = (
            candles.filter(pl.col("tf") == pl.lit(tf_str))
            .select(
                pl.col("instrument").cast(pl.Utf8),
                pl.col("tf").cast(pl.Utf8),
                pl.col("ts").cast(pl.Datetime("us")),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
            )
            .drop_nulls(["instrument", "ts"])
        )
        if c.is_empty():
            continue

        windows = (
            c.select(
                pl.col("instrument"),
                pl.lit(tf_str).alias("anchor_tf"),
                pl.col("ts").alias("anchor_ts"),
            )
            .unique()
            .sort(["instrument", "anchor_tf", "anchor_ts"])
        )

        dr = fold_state_machine(windows, c, cfg=cfg)
        if dr.is_empty():
            continue

        out = dr.rename({"anchor_ts": "ts"})
        frames.append(out)

    if not frames:
        return pl.DataFrame()

    out = pl.concat(frames, how="vertical").sort(["instrument", "anchor_tf", "ts"])
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "ts"],
        where="dealing_range_state",
        allow_extra=False,
    )

    log.info("dealing_range_state: built rows=%d", out.height)
    return out
