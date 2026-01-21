from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry, safe_div

log = logging.getLogger(__name__)


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "anchor_tf": pl.Series([], dtype=pl.Utf8),
            "anchor_ts": pl.Series([], dtype=pl.Datetime("us")),
            "unsup_regime_id": pl.Series([], dtype=pl.Utf8),
            "unsup_regime_confidence": pl.Series([], dtype=pl.Float64),
        }
    )


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg: dict[str, object] = {}
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("unsup_regime", {}) or {})
    if isinstance(family_cfg, Mapping):
        cfg.update(family_cfg)
    return cfg


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Unsupervised regime proxy using rolling volatility buckets.

    Table: data/windows
    Keys : instrument, anchor_tf, anchor_ts
    """
    if candles is None or candles.is_empty():
        log.warning("unsup_regime: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "tf", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("unsup_regime: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    fam_cfg = _merge_cfg(ctx, family_cfg)

    n_clusters = max(2, int(fam_cfg.get("unsup_n_clusters", 8)))
    z_window = max(10, int(fam_cfg.get("unsup_zscore_window_bars", 200)))
    vol_window = max(5, int(fam_cfg.get("unsup_vol_window_bars", 20)))

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("unsup_regime: ctx.cluster.anchor_tfs empty; returning empty keyed frame")
        return _empty_keyed_frame()

    c = (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("tf").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col("close").cast(pl.Float64),
        )
        .drop_nulls(["instrument", "tf", "ts"])
        .sort(["instrument", "tf", "ts"])
    )

    out_frames: list[pl.DataFrame] = []
    for anchor_tf in anchor_tfs:
        tf_str = str(anchor_tf)
        df = c.filter(pl.col("tf") == pl.lit(tf_str))
        if df.is_empty():
            continue

        df = df.sort(["instrument", "ts"]).with_columns(pl.lit(tf_str).alias("anchor_tf"))
        df = df.with_columns(
            pl.col("close").pct_change().over("instrument").alias("_ret"),
        )

        by = ["instrument", "anchor_tf"]
        vol = pl.col("_ret").rolling_std(window_size=vol_window, min_periods=vol_window).over(by).alias("_vol")
        vol_mean = pl.col("_vol").rolling_mean(window_size=z_window, min_periods=z_window).over(by)
        vol_std = pl.col("_vol").rolling_std(window_size=z_window, min_periods=z_window).over(by)

        df = df.with_columns(vol).with_columns(
            safe_div(pl.col("_vol") - vol_mean, vol_std, default=0.0).alias("_vol_z"),
        )

        rank = pl.col("_vol").rank("average").over(by)
        count = pl.count().over(by)
        pct = safe_div(rank - 1, count - 1, default=0.0).alias("_pct")

        df = df.with_columns(
            pct,
            (pl.col("_pct") * pl.lit(n_clusters)).floor().cast(pl.Int64).alias("_bucket"),
        ).with_columns(
            pl.when(pl.col("_bucket") >= pl.lit(n_clusters))
            .then(pl.lit(n_clusters - 1))
            .otherwise(pl.col("_bucket"))
            .alias("_bucket"),
        )

        df = df.with_columns(
            pl.col("_bucket").cast(pl.Utf8).alias("unsup_regime_id"),
            pl.col("_vol_z").abs().clip(0.0, 3.0).alias("unsup_regime_confidence"),
        )

        out_frames.append(
            df.select(
                pl.col("instrument"),
                pl.col("anchor_tf"),
                pl.col("ts").alias("anchor_ts"),
                "unsup_regime_id",
                "unsup_regime_confidence",
            )
        )

    if not out_frames:
        return _empty_keyed_frame()

    out = pl.concat(out_frames, how="vertical").sort(["instrument", "anchor_tf", "anchor_ts"])
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "anchor_ts"],
        where="unsup_regime",
        allow_extra=False,
    )

    log.info("unsup_regime: built rows=%d clusters=%d", out.height, n_clusters)
    return out
