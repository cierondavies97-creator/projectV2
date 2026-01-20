from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry

log = logging.getLogger(__name__)


DEFAULT_CFG: dict[str, object] = {
    "jump_window_bars": 50,
    "jump_min_periods": 10,
    "jump_ret_type": "pct",
    "jump_bv_mu1": 2.0 / 3.141592653589793,
    "phase_version": "jump_variation_v1",
    "threshold_bundle_id": "jump_variation_thresholds_v1",
    "micro_policy_id": "micro_policy_v1",
    "jump_policy_id": "jump_policy_v1",
    "impact_policy_id": "impact_policy_v1",
    "options_policy_id": "options_policy_v1",
}


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg = dict(DEFAULT_CFG)
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("jump_variation", {}) or {})
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

    ret_type = str(cfg.get("jump_ret_type", "pct")).lower()
    if ret_type == "log":
        ret = (pl.col("close").log() - pl.col("close").shift(1).over("instrument")).alias("_ret")
    else:
        ret = pl.col("close").pct_change().over("instrument").alias("_ret")

    window = int(cfg.get("jump_window_bars", 50))
    min_periods = int(cfg.get("jump_min_periods", 10))
    window = max(2, window)
    min_periods = max(1, min_periods)

    mu1 = float(cfg.get("jump_bv_mu1", 2.0 / 3.141592653589793))

    df = df.with_columns(ret)

    r2 = (pl.col("_ret") ** 2).alias("_r2")
    abs_r = pl.col("_ret").abs().alias("_abs_r")
    abs_r_lag = pl.col("_abs_r").shift(1).over("instrument").alias("_abs_r_lag")
    bp = (pl.col("_abs_r") * pl.col("_abs_r_lag")).alias("_bp")

    df = df.with_columns(r2, abs_r, abs_r_lag, bp)

    rv = pl.col("_r2").rolling_sum(window_size=window, min_periods=min_periods).over("instrument").alias("jump_rv")
    bv = (
        pl.col("_bp").rolling_sum(window_size=window, min_periods=min_periods).over("instrument")
        * pl.lit(mu1)
    ).alias("jump_bv")

    rs_plus = (
        pl.when(pl.col("_ret") > 0)
        .then(pl.col("_r2"))
        .otherwise(pl.lit(0.0))
        .rolling_sum(window_size=window, min_periods=min_periods)
        .over("instrument")
        .alias("jump_rs_plus")
    )

    rs_minus = (
        pl.when(pl.col("_ret") < 0)
        .then(pl.col("_r2"))
        .otherwise(pl.lit(0.0))
        .rolling_sum(window_size=window, min_periods=min_periods)
        .over("instrument")
        .alias("jump_rs_minus")
    )

    df = df.with_columns(rv, bv, rs_plus, rs_minus)

    jv = (
        pl.when(pl.col("jump_rv").is_null() | pl.col("jump_bv").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise((pl.col("jump_rv") - pl.col("jump_bv")).clip_min(0.0))
        .alias("jump_jv")
    )

    df = df.with_columns(jv)

    out = df.select(
        "instrument",
        "anchor_tf",
        "ts",
        "jump_rv",
        "jump_bv",
        "jump_jv",
        "jump_rs_plus",
        "jump_rs_minus",
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
        log.warning("jump_variation: candles empty; returning empty frame")
        return pl.DataFrame()

    required = {"instrument", "tf", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("jump_variation: missing columns=%s; returning empty frame", missing)
        return pl.DataFrame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("jump_variation: ctx.cluster.anchor_tfs empty; returning empty frame")
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
        where="jump_variation",
        allow_extra=False,
    )

    log.info("jump_variation: built rows=%d", out.height)
    return out
