from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry, safe_div

log = logging.getLogger(__name__)


def _corrcoef_pair(x: np.ndarray, y: np.ndarray, min_periods: int) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < min_periods:
        return None
    xv = x[mask]
    yv = y[mask]
    if xv.size < 2:
        return None
    x_std = xv.std()
    y_std = yv.std()
    if x_std == 0 or y_std == 0:
        return None
    return float(np.corrcoef(xv, yv)[0, 1])


def _compute_peer_refs(
    wide_returns: pl.DataFrame,
    window: int,
    min_periods: int,
    max_refs: int,
    min_abs: float,
) -> pl.DataFrame:
    if wide_returns is None or wide_returns.is_empty():
        return pl.DataFrame(
            {
                "instrument": pl.Series([], dtype=pl.Utf8),
                "ts": pl.Series([], dtype=pl.Datetime("us")),
            }
        )

    instruments = [col for col in wide_returns.columns if col != "ts"]
    if not instruments:
        return pl.DataFrame(
            {
                "instrument": pl.Series([], dtype=pl.Utf8),
                "ts": pl.Series([], dtype=pl.Datetime("us")),
            }
        )

    ts_list = wide_returns.get_column("ts").to_list()
    arr = wide_returns.select(instruments).to_numpy()
    rows: list[dict[str, object]] = []
    n = len(instruments)
    max_refs = max(0, max_refs)

    for idx, ts in enumerate(ts_list):
        start = max(0, idx - window + 1)
        window_arr = arr[start : idx + 1]
        if window_arr.shape[0] < min_periods:
            for inst in instruments:
                rows.append(
                    {
                        "instrument": inst,
                        "ts": ts,
                        "corr_htf_ref2_id": None,
                        "corr_htf_ref2": None,
                        "corr_htf_ref3_id": None,
                        "corr_htf_ref3": None,
                    }
                )
            continue

        corr_matrix = np.full((n, n), np.nan, dtype=float)
        for j in range(n):
            x = window_arr[:, j]
            for k in range(j, n):
                y = window_arr[:, k]
                corr_val = _corrcoef_pair(x, y, min_periods=min_periods)
                if corr_val is None:
                    continue
                corr_matrix[j, k] = corr_val
                corr_matrix[k, j] = corr_val

        for j, inst in enumerate(instruments):
            peers: list[tuple[str, float]] = []
            for k, peer in enumerate(instruments):
                if k == j:
                    continue
                corr_val = corr_matrix[j, k]
                if np.isnan(corr_val) or abs(corr_val) < min_abs:
                    continue
                peers.append((peer, float(corr_val)))
            peers.sort(key=lambda item: (-abs(item[1]), item[0]))
            top = peers[:max_refs]
            ref2_id, ref2_val = (top[0][0], top[0][1]) if len(top) > 0 else (None, None)
            ref3_id, ref3_val = (top[1][0], top[1][1]) if len(top) > 1 else (None, None)
            rows.append(
                {
                    "instrument": inst,
                    "ts": ts,
                    "corr_htf_ref2_id": ref2_id,
                    "corr_htf_ref2": ref2_val,
                    "corr_htf_ref3_id": ref3_id,
                    "corr_htf_ref3": ref3_val,
                }
            )

    return pl.DataFrame(rows)


def _empty_keyed_frame(registry_entry: Mapping[str, object] | None) -> pl.DataFrame:
    if registry_entry and isinstance(registry_entry, Mapping):
        columns = registry_entry.get("columns")
        if isinstance(columns, Mapping) and columns:
            return pl.DataFrame({col: pl.Series([], dtype=pl.Null) for col in columns})
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "ts": pl.Series([], dtype=pl.Datetime("us")),
        }
    )


def _base_keys_from_candles(candles: pl.DataFrame) -> pl.DataFrame:
    if candles is None or candles.is_empty():
        return pl.DataFrame()
    if not {"instrument", "ts"}.issubset(set(candles.columns)):
        return pl.DataFrame()
    return (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
        )
        .drop_nulls(["instrument", "ts"])
        .unique()
        .sort(["instrument", "ts"])
    )


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    cfg = dict(auto_cfg.get("corr_htf", {}) if isinstance(auto_cfg, Mapping) else {})
    if isinstance(family_cfg, Mapping):
        cfg.update(family_cfg)
    return cfg


def build_feature_frame(
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
) -> pl.DataFrame:
    """
    Higher-timeframe correlation proxy using a longer rolling window.

    Table: data/features_corr
    Keys : instrument, ts
    """
    if candles is None or candles.is_empty():
        log.warning("corr_htf: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    if not {"instrument", "ts", "close"}.issubset(set(candles.columns)):
        log.warning("corr_htf: missing required columns; returning null-filled frame")
        base = _base_keys_from_candles(candles)
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        return conform_to_registry(
            base,
            registry_entry=registry_entry,
            key_cols=["instrument", "ts"],
            where="corr_htf",
            allow_extra=False,
        )

    cfg = _merge_cfg(ctx, family_cfg)
    window = max(5, int(cfg.get("corr_htf_window_bars", 500)))
    min_periods = int(cfg.get("corr_htf_min_periods", 100))
    clip_abs = float(cfg.get("corr_htf_clip_abs", 0.999))
    cluster_stability_cut = float(cfg.get("corr_htf_cluster_stability_cut", 0.70))
    ref_slots = 3
    topk_min_abs = 0.5
    c = candles.select(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
        pl.col("close").cast(pl.Float64),
    ).drop_nulls(["instrument", "ts"])

    c = c.sort(["instrument", "ts"]).with_columns(
        pl.col("close").pct_change().over("instrument").alias("_ret"),
    )

    market_ret = (
        c.group_by("ts")
        .agg(pl.col("_ret").mean().alias("_market_ret"))
        .sort("ts")
    )
    c = c.join(market_ret, on="ts", how="left")

    by = ["instrument"]
    mean_ret = pl.col("_ret").rolling_mean(window_size=window, min_periods=window).over(by)
    mean_mkt = pl.col("_market_ret").rolling_mean(window_size=window, min_periods=window).over(by)
    mean_prod = (pl.col("_ret") * pl.col("_market_ret")).rolling_mean(window_size=window, min_periods=window).over(by)
    std_ret = pl.col("_ret").rolling_std(window_size=window, min_periods=window).over(by)
    std_mkt = pl.col("_market_ret").rolling_std(window_size=window, min_periods=window).over(by)

    corr_raw = safe_div(mean_prod - (mean_ret * mean_mkt), std_ret * std_mkt, default=None)
    corr = corr_raw.clip(-clip_abs, clip_abs).alias("corr_htf_ref1")

    out = c.with_columns(corr)
    wide = (
        out.select("ts", "instrument", "_ret")
        .sort("ts")
        .pivot(values="_ret", index="ts", columns="instrument")
    )
    peer_refs = _compute_peer_refs(
        wide_returns=wide,
        window=window,
        min_periods=min_periods,
        max_refs=max(0, ref_slots - 1),
        min_abs=topk_min_abs,
    )
    out = out.join(peer_refs, on=["instrument", "ts"], how="left")
    out = out.with_columns(
        pl.lit("MARKET").alias("corr_htf_ref1_id"),
        pl.when(pl.col("corr_htf_ref1").is_null())
        .then(pl.lit(None).cast(pl.Utf8))
        .when(pl.col("corr_htf_ref1").abs() < pl.lit(0.2))
        .then(pl.lit("unstable"))
        .when(pl.col("corr_htf_ref1") >= 0)
        .then(pl.lit("aligned"))
        .otherwise(pl.lit("divergent"))
        .alias("htf_corr_regime"),
        pl.col("corr_htf_ref1").abs().alias("htf_corr_confidence"),
    )
    corr_std = (
        pl.col("corr_htf_ref1")
        .rolling_std(window_size=window, min_periods=min_periods)
        .over("instrument")
        .alias("_corr_htf_std")
    )
    out = out.with_columns(corr_std).with_columns(
        pl.when(pl.col("htf_corr_regime") == pl.lit("aligned"))
        .then(pl.lit("cluster_pos"))
        .when(pl.col("htf_corr_regime") == pl.lit("divergent"))
        .then(pl.lit("cluster_neg"))
        .otherwise(pl.lit("cluster_neutral"))
        .alias("corr_cluster_id_htf"),
        pl.col("htf_corr_confidence").alias("corr_cluster_confidence_htf"),
        pl.when(pl.lit(cluster_stability_cut) > 0)
        .then((1 - (pl.col("_corr_htf_std") / pl.lit(cluster_stability_cut))).clip(0.0, 1.0))
        .otherwise(pl.lit(None).cast(pl.Float64))
        .alias("corr_cluster_stability_htf"),
    ).select(
        "instrument",
        "ts",
        "corr_htf_ref1_id",
        "corr_htf_ref1",
        "corr_htf_ref2_id",
        "corr_htf_ref2",
        "corr_htf_ref3_id",
        "corr_htf_ref3",
        "htf_corr_regime",
        "htf_corr_confidence",
        "corr_cluster_id_htf",
        "corr_cluster_confidence_htf",
        "corr_cluster_stability_htf",
    )

    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "ts"],
        where="corr_htf",
        allow_extra=False,
    )

    log.info("corr_htf: built rows=%d window=%d", out.height, window)
    return out
