from __future__ import annotations

import logging

import polars as plhttps://github.com/cierondavies97-creator/projectV2/pulls

from engine.features import FeatureBuildContext
from engine.features._shared import safe_div

log = logging.getLogger(__name__)


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "ts": pl.Series([], dtype=pl.Datetime("us")),
            "xs_relval_spread_level": pl.Series([], dtype=pl.Float64),
            "xs_relval_spread_zscore": pl.Series([], dtype=pl.Float64),
            "xs_relval_carry_rank": pl.Series([], dtype=pl.Float64),
            "xs_relval_momo_rank": pl.Series([], dtype=pl.Float64),
        }
    )


def build_feature_frame(
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Cross-sectional relative value features from per-bar returns.

    Table: data/features_corr
    Keys : instrument, ts
    """
    if candles is None or candles.is_empty():
        log.warning("xs_relval: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("xs_relval: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []

    c = candles.select(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
        pl.col("close").cast(pl.Float64),
        pl.col("tf").cast(pl.Utf8).alias("_tf") if "tf" in candles.columns else pl.lit(None).cast(pl.Utf8).alias("_tf"),
    ).drop_nulls(["instrument", "ts"])

    if anchor_tfs and "_tf" in c.columns:
        c = c.filter(pl.col("_tf").is_in([str(tf) for tf in anchor_tfs]))

    if c.is_empty():
        return _empty_keyed_frame()

    c = c.sort(["instrument", "ts"]).with_columns(
        pl.col("close").pct_change().over("instrument").alias("_ret"),
    )

    by_ts = ["ts"]
    mean_ret = pl.col("_ret").mean().over(by_ts)
    std_ret = pl.col("_ret").std(ddof=1).over(by_ts)

    spread_level = (pl.col("_ret") - mean_ret).alias("xs_relval_spread_level")
    spread_z = safe_div(pl.col("_ret") - mean_ret, std_ret, default=0.0).alias("xs_relval_spread_zscore")

    ret_rank = pl.col("_ret").rank("average").over(by_ts)
    count = pl.count().over(by_ts)
    carry_rank = safe_div(ret_rank - 1, count - 1, default=0.0).alias("xs_relval_carry_rank")

    momo = pl.col("_ret").rolling_mean(window_size=5, min_periods=2).over("instrument").alias("_momo")
    momo_rank = safe_div(pl.col("_momo").rank("average").over(by_ts) - 1, count - 1, default=0.0).alias("xs_relval_momo_rank")

    out = c.with_columns(
        spread_level,
        spread_z,
        momo,
        carry_rank,
        momo_rank,
    ).select(
        "instrument",
        "ts",
        "xs_relval_spread_level",
        "xs_relval_spread_zscore",
        "xs_relval_carry_rank",
        "xs_relval_momo_rank",
    )

    log.info("xs_relval: built rows=%d", out.height)
    return out
