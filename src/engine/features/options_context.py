from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry

log = logging.getLogger(__name__)


DEFAULT_CFG: dict[str, object] = {
    "phase_version": "options_context_v1",
    "threshold_bundle_id": "options_context_thresholds_v1",
    "micro_policy_id": "micro_policy_v1",
    "jump_policy_id": "jump_policy_v1",
    "impact_policy_id": "impact_policy_v1",
    "options_policy_id": "options_policy_v1",
}


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg = dict(DEFAULT_CFG)
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("options_context", {}) or {})
    if isinstance(family_cfg, Mapping):
        cfg.update(family_cfg)
    return cfg


def _policy_columns(cfg: Mapping[str, object]) -> list[pl.Expr]:
    return [
        pl.lit(str(cfg.get("phase_version"))).cast(pl.Utf8).alias("phase_version"),
        pl.lit(str(cfg.get("threshold_bundle_id"))).cast(pl.Utf8).alias("threshold_bundle_id"),
        pl.lit(str(cfg.get("micro_policy_id"))).cast(pl.Utf8).alias("micro_policy_id"),
        pl.lit(str(cfg.get("jump_policy_id"))).cast(pl.Utf8).alias("jump_policy_id"),
        pl.lit(str(cfg.get("impact_policy_id"))).cast(pl.Utf8).alias("impact_policy_id"),
        pl.lit(str(cfg.get("options_policy_id"))).cast(pl.Utf8).alias("options_policy_id"),
    ]


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
        log.warning("options_context: candles empty; returning empty frame")
        return pl.DataFrame()

    required = {"instrument", "tf", "ts"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("options_context: missing columns=%s; returning empty frame", missing)
        return pl.DataFrame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("options_context: ctx.cluster.anchor_tfs empty; returning empty frame")
        return pl.DataFrame()

    cfg = _merge_cfg(ctx, family_cfg)

    frames: list[pl.DataFrame] = []
    for anchor_tf in anchor_tfs:
        tf_str = str(anchor_tf)
        df = (
            candles.filter(pl.col("tf") == pl.lit(tf_str))
            .select(
                pl.col("instrument").cast(pl.Utf8),
                pl.col("ts").cast(pl.Datetime("us")),
            )
            .drop_nulls(["instrument", "ts"])
            .unique()
            .with_columns(pl.lit(tf_str).alias("anchor_tf"))
        )
        if df.is_empty():
            continue
        df = df.with_columns(_policy_columns(cfg))
        frames.append(df)

    if not frames:
        return pl.DataFrame()

    out = pl.concat(frames, how="vertical").sort(["instrument", "anchor_tf", "ts"])
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "ts"],
        where="options_context",
        allow_extra=False,
    )

    log.info("options_context: built rows=%d", out.height)
    return out
