from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry, safe_div

log = logging.getLogger(__name__)


DEFAULT_CFG: dict[str, object] = {
    "impact_window_bars": 50,
    "impact_min_periods": 10,
    "impact_lambda_low_cut": 1e-6,
    "impact_lambda_high_cut": 1e-4,
    "phase_version": "impact_lambda_v1",
    "threshold_bundle_id": "impact_lambda_thresholds_v1",
    "micro_policy_id": "micro_policy_v1",
    "jump_policy_id": "jump_policy_v1",
    "impact_policy_id": "impact_policy_v1",
    "options_policy_id": "options_policy_v1",
}


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg = dict(DEFAULT_CFG)
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("impact_lambda", {}) or {})
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


def _compute_for_tf(df: pl.DataFrame, *, anchor_tf: str, cfg: Mapping[str, object]) -> pl.DataFrame:
    df = df.with_columns(pl.lit(anchor_tf).alias("anchor_tf")).sort(["instrument", "ts"])

    window = int(cfg.get("impact_window_bars", 50))
    min_periods = int(cfg.get("impact_min_periods", 10))
    window = max(2, window)
    min_periods = max(1, min_periods)

    dp = (pl.col("close") - pl.col("close").shift(1).over("instrument")).alias("_dp")
    df = df.with_columns(dp)

    sign = (
        pl.when(pl.col("_dp") > 0)
        .then(pl.lit(1.0))
        .when(pl.col("_dp") < 0)
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0))
        .alias("_sign")
    )

    df = df.with_columns(sign)

    q = (pl.col("volume") * pl.col("_sign")).alias("_q")
    df = df.with_columns(q)

    num = (
        (pl.col("_dp") * pl.col("_q"))
        .rolling_sum(window_size=window, min_periods=min_periods)
        .over("instrument")
        .alias("_num")
    )
    den = (
        (pl.col("_q") ** 2)
        .rolling_sum(window_size=window, min_periods=min_periods)
        .over("instrument")
        .alias("_den")
    )

    df = df.with_columns(num, den)

    lam = safe_div(pl.col("_num"), pl.col("_den"), default=None).alias("impact_kyle_lambda")

    avg_abs_q = (
        pl.col("_q").abs().rolling_mean(window_size=window, min_periods=min_periods).over("instrument").alias("_abs_q")
    )
    df = df.with_columns(avg_abs_q)
    cost = (pl.col("impact_kyle_lambda") * pl.col("_abs_q")).alias("impact_cost_unit")

    low_cut = float(cfg.get("impact_lambda_low_cut", 1e-6))
    high_cut = float(cfg.get("impact_lambda_high_cut", 1e-4))

    regime = (
        pl.when(pl.col("impact_kyle_lambda").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("impact_kyle_lambda") <= pl.lit(low_cut))
        .then(pl.lit("low"))
        .when(pl.col("impact_kyle_lambda") >= pl.lit(high_cut))
        .then(pl.lit("high"))
        .otherwise(pl.lit("medium"))
        .alias("impact_lambda_regime")
    )

    df = df.with_columns(lam, cost, regime)

    out = df.select(
        "instrument",
        "anchor_tf",
        "ts",
        "impact_kyle_lambda",
        "impact_lambda_regime",
        "impact_cost_unit",
    )
    return out


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
        log.warning("impact_lambda: candles empty; returning empty frame")
        return pl.DataFrame()

    required = {"instrument", "tf", "ts", "close", "volume"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("impact_lambda: missing columns=%s; returning empty frame", missing)
        return pl.DataFrame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("impact_lambda: ctx.cluster.anchor_tfs empty; returning empty frame")
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
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            )
            .drop_nulls(["instrument", "ts"])
        )
        if df.is_empty():
            continue
        out = _compute_for_tf(df, anchor_tf=tf_str, cfg=cfg)
        out = out.with_columns(_policy_columns(cfg))
        frames.append(out)

    if not frames:
        return pl.DataFrame()

    out = pl.concat(frames, how="vertical").sort(["instrument", "anchor_tf", "ts"])
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "ts"],
        where="impact_lambda",
        allow_extra=False,
    )

    log.info("impact_lambda: built rows=%d", out.height)
    return out
