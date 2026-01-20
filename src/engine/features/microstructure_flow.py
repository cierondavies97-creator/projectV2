from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.core.timegrid import build_anchor_grid, tf_to_truncate_rule
from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry, safe_div

log = logging.getLogger(__name__)


DEFAULT_CFG: dict[str, object] = {
    "micro_window_ticks": 200,
    "micro_intensity_low_cut": 25,
    "micro_intensity_high_cut": 250,
    "micro_agg_imbalance_abs_cut": 0.2,
    "phase_version": "micro_flow_v1",
    "threshold_bundle_id": "micro_flow_thresholds_v1",
    "micro_policy_id": "micro_policy_v1",
    "jump_policy_id": "jump_policy_v1",
    "impact_policy_id": "impact_policy_v1",
    "options_policy_id": "options_policy_v1",
}


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg = dict(DEFAULT_CFG)
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("microstructure_flow", {}) or {})
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


def _base_grid_for_anchor(candles: pl.DataFrame, anchor_tf: str) -> pl.DataFrame:
    if candles.is_empty():
        return pl.DataFrame()
    if "tf" in candles.columns:
        candles = candles.filter(pl.col("tf") == pl.lit(anchor_tf))
    return build_anchor_grid(candles, anchor_tf=anchor_tf, ts_col="ts", instrument_col="instrument")


def _aggregate_ticks(
    ticks: pl.DataFrame,
    *,
    anchor_tf: str,
    cfg: Mapping[str, object],
) -> pl.DataFrame:
    if ticks.is_empty():
        return pl.DataFrame()

    required = {"instrument", "ts"}
    if not required.issubset(set(ticks.columns)):
        return pl.DataFrame()

    if "side" not in ticks.columns and "price" not in ticks.columns:
        return pl.DataFrame()

    rule = tf_to_truncate_rule(anchor_tf)

    df = ticks.select(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
        pl.col("price").cast(pl.Float64) if "price" in ticks.columns else pl.lit(None).cast(pl.Float64),
        pl.col("size").cast(pl.Float64) if "size" in ticks.columns else pl.lit(1.0).cast(pl.Float64),
        pl.col("side").cast(pl.Utf8) if "side" in ticks.columns else pl.lit(None).cast(pl.Utf8),
    ).drop_nulls(["instrument", "ts"])

    if df.is_empty():
        return pl.DataFrame()

    dir_expr = (
        pl.when(pl.col("side").is_in(["buy", "bid", "long"]))
        .then(pl.lit(1.0))
        .when(pl.col("side").is_in(["sell", "ask", "short"]))
        .then(pl.lit(-1.0))
        .otherwise(
            pl.when(pl.col("price") >= pl.col("price").shift(1).over("instrument"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(-1.0))
        )
        .alias("_dir")
    )

    df = df.with_columns(dir_expr)
    df = df.with_columns(
        (pl.col("size") * pl.col("_dir")).alias("_signed_vol"),
        pl.col("ts").dt.truncate(rule).alias("_anchor_ts"),
        pl.lit(anchor_tf).cast(pl.Utf8).alias("anchor_tf"),
    )

    grouped = (
        df.sort(["instrument", "_anchor_ts"])
        .group_by(["instrument", "anchor_tf", "_anchor_ts"], maintain_order=True)
        .agg(
            pl.col("size").sum().alias("_total_vol"),
            pl.col("_signed_vol").sum().alias("_delta_vol"),
            pl.col("_signed_vol").count().alias("_event_count"),
        )
    )

    agg_imb = safe_div(pl.col("_delta_vol"), pl.col("_total_vol"), default=None).alias("micro_agg_imbalance")
    ofi_value = pl.col("_delta_vol").alias("micro_ofi_value")
    intensity = pl.col("_event_count").cast(pl.Float64).alias("micro_intensity_events")

    imb_cut = float(cfg.get("micro_agg_imbalance_abs_cut", 0.2))
    intensity_low = float(cfg.get("micro_intensity_low_cut", 25))
    intensity_high = float(cfg.get("micro_intensity_high_cut", 250))

    agg_bucket = (
        pl.when(pl.col("micro_agg_imbalance") >= pl.lit(imb_cut))
        .then(pl.lit("buy_dominant"))
        .when(pl.col("micro_agg_imbalance") <= pl.lit(-imb_cut))
        .then(pl.lit("sell_dominant"))
        .otherwise(pl.lit("balanced"))
        .alias("micro_agg_imbalance_bucket")
    )

    intensity_bucket = (
        pl.when(pl.col("micro_intensity_events") >= pl.lit(intensity_high))
        .then(pl.lit("high"))
        .when(pl.col("micro_intensity_events") <= pl.lit(intensity_low))
        .then(pl.lit("low"))
        .otherwise(pl.lit("normal"))
        .alias("micro_intensity_bucket")
    )

    out = grouped.with_columns(ofi_value, agg_imb, intensity).with_columns(agg_bucket, intensity_bucket)

    return out.rename({"_anchor_ts": "ts"}).select(
        "instrument",
        "anchor_tf",
        "ts",
        "micro_ofi_value",
        "micro_agg_imbalance",
        "micro_agg_imbalance_bucket",
        "micro_intensity_events",
        "micro_intensity_bucket",
    )


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Table: data/features
    Keys : instrument, anchor_tf, ts
    """
    if candles is None or candles.is_empty():
        log.warning("microstructure_flow: candles empty; returning empty frame")
        return pl.DataFrame()

    required = {"instrument", "tf", "ts"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("microstructure_flow: missing columns=%s; returning empty frame", missing)
        return pl.DataFrame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("microstructure_flow: ctx.cluster.anchor_tfs empty; returning empty frame")
        return pl.DataFrame()

    cfg = _merge_cfg(ctx, family_cfg)

    frames: list[pl.DataFrame] = []

    for anchor_tf in anchor_tfs:
        base = _base_grid_for_anchor(candles, str(anchor_tf))
        if base.is_empty():
            continue

        if ticks is None or ticks.is_empty():
            agg = base
        else:
            agg_ticks = _aggregate_ticks(ticks, anchor_tf=str(anchor_tf), cfg=cfg)
            agg = base.join(agg_ticks, on=["instrument", "anchor_tf", "ts"], how="left")

        agg = agg.with_columns(_policy_columns(cfg))
        frames.append(agg)

    if not frames:
        return pl.DataFrame()

    out = pl.concat(frames, how="vertical").sort(["instrument", "anchor_tf", "ts"])
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "ts"],
        where="microstructure_flow",
        allow_extra=False,
    )

    log.info("microstructure_flow: built rows=%d", out.height)
    return out
