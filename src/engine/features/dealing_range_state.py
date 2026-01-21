from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry
from engine.features.market_structure.dealing_range import fold_state_machine

log = logging.getLogger(__name__)


_CFG_KEY_MAP = {
    "dealing_range_lookback_bars": "lookback_bars",
    "width_min_atr_mult": "width_min_atr",
    "test_band_atr_mult": "test_atr_mult",
    "spring_penetration_atr_mult": "probe_atr_mult",
    "trend_distance_atr": "trend_atr_mult",
}


def _normalize_cfg_keys(cfg: Mapping[str, object]) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for key, value in cfg.items():
        if key in _CFG_KEY_MAP:
            mapped[_CFG_KEY_MAP[key]] = value
        else:
            mapped[key] = value
    return mapped


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg: dict[str, object] = {}
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("dealing_range_state", {}) or {})
    if isinstance(family_cfg, Mapping):
        cfg.update(family_cfg)
    return _normalize_cfg_keys(cfg)


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


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Table: data/windows
    Keys : instrument, anchor_tf, anchor_ts
    """
    if candles is None or candles.is_empty():
        log.warning("dealing_range_state: candles empty; returning empty frame")
        return _empty_keyed_frame(registry_entry)

    required = {"instrument", "tf", "ts", "high", "low", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("dealing_range_state: missing columns=%s; returning empty frame", missing)
        return _empty_keyed_frame(registry_entry)

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("dealing_range_state: ctx.cluster.anchor_tfs empty; returning empty frame")
        return _empty_keyed_frame(registry_entry)

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

        frames.append(dr)

    if not frames:
        return _empty_keyed_frame(registry_entry)

    out = pl.concat(frames, how="vertical").sort(["instrument", "anchor_tf", "anchor_ts"])
    if "ts" in out.columns and "anchor_ts" not in out.columns:
        out = out.rename({"ts": "anchor_ts"})
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "anchor_ts"],
        where="dealing_range_state",
        allow_extra=False,
    )

    log.info("dealing_range_state: built rows=%d", out.height)
    return out
