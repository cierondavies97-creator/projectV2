from __future__ import annotations

import logging
import math
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


def _compute_primary_peer_metrics(
    wide_returns: pl.DataFrame,
    window: int,
    min_periods: int,
    method: str,
    min_corr: float,
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

    for idx, ts in enumerate(ts_list):
        start = max(0, idx - window + 1)
        window_arr = arr[start : idx + 1]
        if window_arr.shape[0] < min_periods:
            for inst in instruments:
                rows.append(
                    {
                        "instrument": inst,
                        "ts": ts,
                        "xs_relval_primary_peer": None,
                        "xs_relval_primary_peer_corr": None,
                        "xs_relval_primary_peer_beta": None,
                        "xs_relval_coint_pvalue": None,
                        "xs_relval_half_life_bars": None,
                        "xs_relval_residual": None,
                        "xs_relval_residual_z": None,
                    }
                )
            continue

        for j, inst in enumerate(instruments):
            x = window_arr[:, j]
            if not np.isfinite(x).any():
                rows.append(
                    {
                        "instrument": inst,
                        "ts": ts,
                        "xs_relval_primary_peer": None,
                        "xs_relval_primary_peer_corr": None,
                        "xs_relval_primary_peer_beta": None,
                        "xs_relval_coint_pvalue": None,
                        "xs_relval_half_life_bars": None,
                        "xs_relval_residual": None,
                        "xs_relval_residual_z": None,
                    }
                )
                continue

            best_peer = None
            best_corr = None
            best_beta = None
            best_metric = None
            best_resid = None

            for k, peer in enumerate(instruments):
                if k == j:
                    continue
                y = window_arr[:, k]
                corr_val = _corrcoef_pair(x, y, min_periods=min_periods)
                if corr_val is None:
                    continue
                mask = np.isfinite(x) & np.isfinite(y)
                xv = x[mask]
                yv = y[mask]
                y_var = yv.var()
                beta = float(np.cov(xv, yv)[0, 1] / y_var) if y_var > 0 else None
                resid = xv - (beta * yv) if beta is not None else None
                if method == "min_resid_var":
                    metric = float(resid.var()) if resid is not None else None
                    if metric is None:
                        continue
                    if best_metric is None or metric < best_metric:
                        best_metric = metric
                        best_peer = peer
                        best_corr = corr_val
                        best_beta = beta
                        best_resid = resid
                else:
                    metric = abs(corr_val)
                    if best_metric is None or metric > best_metric:
                        best_metric = metric
                        best_peer = peer
                        best_corr = corr_val
                        best_beta = beta
                        best_resid = resid

            if best_peer is None or best_corr is None or abs(best_corr) < min_corr:
                rows.append(
                    {
                        "instrument": inst,
                        "ts": ts,
                        "xs_relval_primary_peer": None,
                        "xs_relval_primary_peer_corr": None,
                        "xs_relval_primary_peer_beta": None,
                        "xs_relval_coint_pvalue": None,
                        "xs_relval_half_life_bars": None,
                        "xs_relval_residual": None,
                        "xs_relval_residual_z": None,
                    }
                )
                continue

            residual_series = best_resid
            residual_z = None
            half_life = None
            if residual_series is not None and residual_series.size >= 3:
                resid_mean = residual_series.mean()
                resid_std = residual_series.std()
                if resid_std > 0 and np.isfinite(x[-1]):
                    current_mask = np.isfinite(x) & np.isfinite(window_arr[:, instruments.index(best_peer)])
                    if current_mask.any():
                        current_x = x[current_mask][-1]
                        current_y = window_arr[:, instruments.index(best_peer)][current_mask][-1]
                        residual_z = float(((current_x - best_beta * current_y) - resid_mean) / resid_std)

                resid_lag = residual_series[:-1]
                resid_next = residual_series[1:]
                phi = _corrcoef_pair(resid_lag, resid_next, min_periods=2)
                if phi is not None and 0 < phi < 1:
                    half_life = int(round(-math.log(2) / math.log(phi)))

            coint_pvalue = None
            if best_corr is not None:
                coint_pvalue = float((1 - min(1.0, abs(best_corr))))

            rows.append(
                {
                    "instrument": inst,
                    "ts": ts,
                    "xs_relval_primary_peer": best_peer,
                    "xs_relval_primary_peer_corr": float(best_corr),
                    "xs_relval_primary_peer_beta": float(best_beta) if best_beta is not None else None,
                    "xs_relval_coint_pvalue": coint_pvalue,
                    "xs_relval_half_life_bars": half_life,
                    "xs_relval_residual": float(residual_series[-1]) if residual_series is not None else None,
                    "xs_relval_residual_z": residual_z,
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
            "xs_relval_spread_level": pl.Series([], dtype=pl.Float64),
            "xs_relval_spread_zscore": pl.Series([], dtype=pl.Float64),
            "xs_relval_carry_rank": pl.Series([], dtype=pl.Float64),
            "xs_relval_momo_rank": pl.Series([], dtype=pl.Float64),
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
    cfg = dict(auto_cfg.get("xs_relval", {}) if isinstance(auto_cfg, Mapping) else {})
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
    Cross-sectional relative value features from per-bar returns.

    Table: data/features_corr
    Keys : instrument, ts
    """
    if candles is None or candles.is_empty():
        log.warning("xs_relval: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    required = {"instrument", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("xs_relval: missing columns=%s; returning null-filled frame", missing)
        base = _base_keys_from_candles(candles)
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        return conform_to_registry(
            base,
            registry_entry=registry_entry,
            key_cols=["instrument", "ts"],
            where="xs_relval",
            allow_extra=False,
        )

    cfg = _merge_cfg(ctx, family_cfg)
    spread_type = str(cfg.get("xs_relval_spread_type", "log_ratio"))
    z_window = max(5, int(cfg.get("xs_relval_spread_z_window", 60)))
    z_clip = float(cfg.get("xs_relval_z_clip_abs", 8.0))
    mild_cut = float(cfg.get("xs_relval_bucket_mild_cut", 1.5))
    strong_cut = float(cfg.get("xs_relval_bucket_strong_cut", 2.5))
    peer_set_used = str(cfg.get("xs_relval_peer_set_id", "cluster_peers"))
    primary_peer_method = str(cfg.get("xs_relval_primary_peer_method", "max_abs_corr"))
    primary_peer_min_corr = float(cfg.get("xs_relval_primary_peer_min_corr", 0.50))
    coint_window = max(10, int(cfg.get("xs_relval_coint_window_bars", 500)))

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
        return _empty_keyed_frame(registry_entry)

    c = c.sort(["instrument", "ts"]).with_columns(
        pl.col("close").pct_change().over("instrument").alias("_ret"),
    )

    wide = (
        c.select("ts", "instrument", "_ret")
        .sort("ts")
        .pivot(values="_ret", index="ts", columns="instrument")
    )
    peer_metrics = _compute_primary_peer_metrics(
        wide_returns=wide,
        window=coint_window,
        min_periods=max(5, coint_window // 3),
        method=primary_peer_method,
        min_corr=primary_peer_min_corr,
    )

    by_ts = ["ts"]
    mean_ret = pl.col("_ret").mean().over(by_ts)
    mean_close = pl.col("close").mean().over(by_ts)
    mean_log_close = pl.col("close").log().mean().over(by_ts)
    if spread_type == "price_ratio":
        spread_level = (safe_div(pl.col("close"), mean_close, default=None) - pl.lit(1.0)).alias(
            "xs_relval_spread_level"
        )
    elif spread_type == "log_ratio":
        spread_level = (pl.col("close").log() - mean_log_close).alias("xs_relval_spread_level")
    else:
        spread_level = (pl.col("_ret") - mean_ret).alias("xs_relval_spread_level")

    ret_rank = pl.col("_ret").rank("average").over(by_ts)
    count = pl.count().over(by_ts)
    carry_rank = safe_div(ret_rank - 1, count - 1, default=0.0).alias("xs_relval_carry_rank")

    momo = pl.col("_ret").rolling_mean(window_size=5, min_periods=2).over("instrument").alias("_momo")
    momo_rank = safe_div(pl.col("_momo").rank("average").over(by_ts) - 1, count - 1, default=0.0).alias("xs_relval_momo_rank")

    out = c.with_columns(spread_level, momo, carry_rank, momo_rank)
    out = out.join(peer_metrics, on=["instrument", "ts"], how="left")
    if spread_type == "hedged_residual":
        out = out.with_columns(
            pl.col("xs_relval_residual").alias("xs_relval_spread_level")
        )
    spread_mean = (
        pl.col("xs_relval_spread_level")
        .rolling_mean(window_size=z_window, min_periods=max(3, z_window // 3))
        .over("instrument")
    )
    spread_std = (
        pl.col("xs_relval_spread_level")
        .rolling_std(window_size=z_window, min_periods=max(3, z_window // 3))
        .over("instrument")
    )
    spread_z = safe_div(
        pl.col("xs_relval_spread_level") - spread_mean,
        spread_std,
        default=None,
    ).clip(-z_clip, z_clip).alias("xs_relval_spread_zscore")
    out = out.with_columns(spread_z)

    out = out.with_columns(
        pl.when(pl.col("xs_relval_spread_zscore") <= pl.lit(-strong_cut))
        .then(pl.lit("far_undervalued"))
        .when(pl.col("xs_relval_spread_zscore") <= pl.lit(-mild_cut))
        .then(pl.lit("undervalued"))
        .when(pl.col("xs_relval_spread_zscore") >= pl.lit(strong_cut))
        .then(pl.lit("far_overvalued"))
        .when(pl.col("xs_relval_spread_zscore") >= pl.lit(mild_cut))
        .then(pl.lit("overvalued"))
        .otherwise(pl.lit("fair"))
        .alias("xs_relval_spread_bucket"),
        pl.when(pl.col("xs_relval_spread_zscore") <= pl.lit(-mild_cut))
        .then(pl.lit("long"))
        .when(pl.col("xs_relval_spread_zscore") >= pl.lit(mild_cut))
        .then(pl.lit("short"))
        .otherwise(pl.lit("neutral"))
        .alias("xs_relval_signal"),
        pl.when(pl.lit(strong_cut) > 0)
        .then((pl.col("xs_relval_spread_zscore").abs() / pl.lit(strong_cut)).clip(0.0, 1.0))
        .otherwise(pl.lit(0.0))
        .alias("xs_relval_signal_strength"),
        pl.count().over(by_ts).cast(pl.Int64).alias("xs_relval_peer_count"),
        pl.lit(peer_set_used).alias("xs_relval_peer_set_used"),
        pl.col("xs_relval_primary_peer").cast(pl.Utf8),
        pl.col("xs_relval_primary_peer_corr").cast(pl.Float64),
        pl.col("xs_relval_primary_peer_beta").cast(pl.Float64),
        pl.when(pl.col("xs_relval_spread_zscore") <= pl.lit(-mild_cut))
        .then(pl.lit("buy_this"))
        .when(pl.col("xs_relval_spread_zscore") >= pl.lit(mild_cut))
        .then(pl.lit("sell_this"))
        .otherwise(pl.lit("neutral"))
        .alias("xs_relval_primary_peer_role"),
        pl.col("xs_relval_coint_pvalue").cast(pl.Float64),
        pl.col("xs_relval_half_life_bars").cast(pl.Int64),
        pl.when(pl.lit(spread_type) == pl.lit("hedged_residual"))
        .then(pl.col("xs_relval_residual_z"))
        .otherwise(pl.lit(None).cast(pl.Float64))
        .alias("xs_relval_residual_z"),
    ).select(
        "instrument",
        "ts",
        "xs_relval_spread_level",
        "xs_relval_spread_zscore",
        "xs_relval_carry_rank",
        "xs_relval_momo_rank",
        "xs_relval_spread_bucket",
        "xs_relval_signal",
        "xs_relval_signal_strength",
        "xs_relval_peer_count",
        "xs_relval_peer_set_used",
        "xs_relval_primary_peer",
        "xs_relval_primary_peer_corr",
        "xs_relval_primary_peer_beta",
        "xs_relval_primary_peer_role",
        "xs_relval_coint_pvalue",
        "xs_relval_half_life_bars",
        "xs_relval_residual_z",
    )

    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "ts"],
        where="xs_relval",
        allow_extra=False,
    )

    log.info("xs_relval: built rows=%d", out.height)
    return out