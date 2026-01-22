from __future__ import annotations

import logging

import polars as pl

from collections.abc import Mapping
import json

import numpy as np

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


def _compute_topk_neighbors(
    wide_returns: pl.DataFrame,
    window: int,
    min_periods: int,
    topk: int,
    min_abs: float,
) -> pl.DataFrame:
    if wide_returns is None or wide_returns.is_empty():
        return pl.DataFrame(
            {
                "instrument": pl.Series([], dtype=pl.Utf8),
                "ts": pl.Series([], dtype=pl.Datetime("us")),
                "corr_ref2_id": pl.Series([], dtype=pl.Utf8),
                "corr_ref2": pl.Series([], dtype=pl.Float64),
                "corr_ref3_id": pl.Series([], dtype=pl.Utf8),
                "corr_ref3": pl.Series([], dtype=pl.Float64),
                "corr_topk_neighbors_json": pl.Series([], dtype=pl.Utf8),
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
                        "corr_ref2_id": None,
                        "corr_ref2": None,
                        "corr_ref3_id": None,
                        "corr_ref3": None,
                        "corr_topk_neighbors_json": None,
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
            top = peers[:topk]
            ref2_id, ref2_val = (top[0][0], top[0][1]) if len(top) > 0 else (None, None)
            ref3_id, ref3_val = (top[1][0], top[1][1]) if len(top) > 1 else (None, None)
            if top:
                neighbors_json = json.dumps(
                    [{"peer": peer, "corr": corr, "lag": 0} for peer, corr in top],
                    separators=(",", ":"),
                )
            else:
                neighbors_json = None
            rows.append(
                {
                    "instrument": inst,
                    "ts": ts,
                    "corr_ref2_id": ref2_id,
                    "corr_ref2": ref2_val,
                    "corr_ref3_id": ref3_id,
                    "corr_ref3": ref3_val,
                    "corr_topk_neighbors_json": neighbors_json,
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
    cfg = dict(auto_cfg.get("corr_micro", {}) if isinstance(auto_cfg, Mapping) else {})
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
    Micro correlation features using rolling correlation vs a cluster mean return.

    Table: data/features_corr
    Keys : instrument, ts
    """
    if candles is None or candles.is_empty():
        log.warning("corr_micro: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    required = {"instrument", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("corr_micro: missing columns=%s; returning null-filled frame", missing)
        base = _base_keys_from_candles(candles)
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        out = conform_to_registry(
            base,
            registry_entry=registry_entry,
            key_cols=["instrument", "ts"],
            where="corr_micro",
            allow_extra=False,
        )
        return out

    fam_cfg = _merge_cfg(ctx, family_cfg)

    window = max(5, int(fam_cfg.get("corr_window_bars", 50)))
    min_periods = int(fam_cfg.get("corr_min_periods", max(10, window // 2)))
    strong_cut = float(fam_cfg.get("corr_strong_cut", 0.5))
    clip_abs = float(fam_cfg.get("corr_clip_abs", 0.999))
    stability_window = max(5, int(fam_cfg.get("corr_stability_window_bars", 200)))
    flip_cut = float(fam_cfg.get("corr_flip_abs_delta_cut", 0.5))
    unstable_cut = float(fam_cfg.get("corr_unstable_std_cut", 0.25))
    lag_max = max(0, int(fam_cfg.get("corr_lag_max_bars", 5)))
    vol_proxy = str(fam_cfg.get("corr_vol_proxy", "ret_std"))
    vol_window = max(3, int(fam_cfg.get("corr_vol_window_bars", 20)))
    ref_slots = max(1, int(fam_cfg.get("corr_ref_slots", 3)))
    topk_neighbors = max(0, int(fam_cfg.get("corr_topk_neighbors", 5)))
    topk_min_abs = float(fam_cfg.get("corr_topk_min_abs_corr", 0.5))

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []

    c = candles.select(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
        pl.col("close").cast(pl.Float64),
        pl.col("atr").cast(pl.Float64) if "atr" in candles.columns else pl.lit(None).cast(pl.Float64).alias("atr"),
        pl.col("tf").cast(pl.Utf8).alias("_tf") if "tf" in candles.columns else pl.lit(None).cast(pl.Utf8).alias("_tf"),
    ).drop_nulls(["instrument", "ts"])

    if anchor_tfs and "_tf" in c.columns:
        c = c.filter(pl.col("_tf").is_in([str(tf) for tf in anchor_tfs]))

    if c.is_empty():
        return _empty_keyed_frame(registry_entry)

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

    cov = (mean_prod - (mean_ret * mean_mkt)).alias("_cov")
    corr_raw = safe_div(cov, std_ret * std_mkt, default=None)
    corr = corr_raw.clip(-clip_abs, clip_abs).alias("corr_index_major")

    df = c.with_columns(cov, corr)

    if topk_neighbors > 0:
        wide = (
            df.select("ts", "instrument", "_ret")
            .sort("ts")
            .pivot(values="_ret", index="ts", columns="instrument")
        )
        neighbors = _compute_topk_neighbors(
            wide_returns=wide,
            window=window,
            min_periods=min_periods,
            topk=topk_neighbors,
            min_abs=topk_min_abs,
        )
        df = df.join(neighbors, on=["instrument", "ts"], how="left")
    else:
        df = df.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("corr_ref2_id"),
            pl.lit(None).cast(pl.Float64).alias("corr_ref2"),
            pl.lit(None).cast(pl.Utf8).alias("corr_ref3_id"),
            pl.lit(None).cast(pl.Float64).alias("corr_ref3"),
            pl.lit(None).cast(pl.Utf8).alias("corr_topk_neighbors_json"),
        )

    df = df.with_columns(
        pl.col("corr_index_major").alias("corr_dxy"),
        pl.col("corr_index_major").alias("corr_oil"),
        pl.lit("MARKET").alias("corr_ref1_id"),
        pl.col("corr_index_major").alias("corr_ref1"),
    )
    if ref_slots < 2:
        df = df.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("corr_ref2_id"),
            pl.lit(None).cast(pl.Float64).alias("corr_ref2"),
        )
    if ref_slots < 3:
        df = df.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("corr_ref3_id"),
            pl.lit(None).cast(pl.Float64).alias("corr_ref3"),
        )
    ref1_abs = pl.col("corr_ref1").abs().fill_null(0.0)
    ref2_abs = pl.col("corr_ref2").abs().fill_null(0.0)
    ref3_abs = pl.col("corr_ref3").abs().fill_null(0.0)
    max_abs = pl.max_horizontal(ref1_abs, ref2_abs, ref3_abs)
    corr_ref_max = (
        pl.when(ref1_abs == max_abs)
        .then(pl.col("corr_ref1"))
        .when(ref2_abs == max_abs)
        .then(pl.col("corr_ref2"))
        .otherwise(pl.col("corr_ref3"))
    )
    df = df.with_columns(
        pl.when(corr_ref_max.is_null())
        .then(pl.lit(None).cast(pl.Utf8))
        .when(corr_ref_max >= 0)
        .then(pl.lit("pos"))
        .otherwise(pl.lit("neg"))
        .alias("micro_corr_sign"),
        corr_ref_max.abs().clip(0.0, 1.0).alias("micro_corr_strength"),
    )
    corr_count = (
        pl.col("_ret")
        .rolling_count(window_size=window, min_periods=1)
        .over("instrument")
        .alias("_corr_count")
    )
    df = df.with_columns(corr_count)
    df = df.with_columns(
        pl.when(pl.col("_corr_count") >= pl.lit(min_periods))
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .alias("micro_corr_confidence")
    )
    corr_std = (
        pl.col("corr_ref1")
        .rolling_std(window_size=stability_window, min_periods=min_periods)
        .over("instrument")
        .alias("micro_corr_std")
    )
    df = df.with_columns(corr_std)
    df = df.with_columns(
        (pl.col("micro_corr_std") >= pl.lit(unstable_cut)).alias("micro_corr_unstable_flag"),
        (
            (pl.col("corr_ref1") * pl.col("corr_ref1").shift(1).over("instrument") < 0)
            | ((pl.col("corr_ref1") - pl.col("corr_ref1").shift(1).over("instrument")).abs() >= pl.lit(flip_cut))
        ).alias("micro_corr_flip_flag"),
    )
    df = df.with_columns(
        pl.when(pl.col("corr_ref1") >= pl.lit(strong_cut))
        .then(pl.lit("aligned"))
        .when(pl.col("corr_ref1") <= pl.lit(-strong_cut))
        .then(pl.lit("divergent"))
        .otherwise(pl.lit("unstable"))
        .alias("micro_corr_regime")
    )
    df = df.with_columns(
        pl.when(pl.col("micro_corr_regime") == pl.lit("aligned"))
        .then(pl.lit("cluster_pos"))
        .when(pl.col("micro_corr_regime") == pl.lit("divergent"))
        .then(pl.lit("cluster_neg"))
        .otherwise(pl.lit("cluster_neutral"))
        .alias("corr_cluster_id"),
    )

    lag_cols: list[str] = []
    if lag_max > 0:
        for lag in range(-lag_max, lag_max + 1):
            lag_suffix = f"_xcorr_{lag}"
            shifted = pl.col("_market_ret").shift(lag).over("instrument")
            mean_mkt_lag = shifted.rolling_mean(window_size=window, min_periods=window).over(by)
            mean_prod_lag = (pl.col("_ret") * shifted).rolling_mean(window_size=window, min_periods=window).over(by)
            std_mkt_lag = shifted.rolling_std(window_size=window, min_periods=window).over(by)
            cov_lag = mean_prod_lag - (mean_ret * mean_mkt_lag)
            corr_lag = safe_div(cov_lag, std_ret * std_mkt_lag, default=None).alias(lag_suffix)
            df = df.with_columns(corr_lag)
            lag_cols.append(lag_suffix)
    else:
        lag_cols.append("corr_ref1")

    lag_abs_cols = [pl.col(col).abs() for col in lag_cols]
    lag_max_abs = pl.max_horizontal(*lag_abs_cols).alias("_xcorr_max_abs")
    df = df.with_columns(lag_max_abs)
    lag_expr = pl.lit(0).cast(pl.Int64)
    for lag, col in zip(range(-lag_max, lag_max + 1), lag_cols):
        lag_expr = pl.when(pl.col(col).abs() == pl.col("_xcorr_max_abs")).then(pl.lit(lag)).otherwise(lag_expr)
    lag_expr = pl.when(pl.col("_xcorr_max_abs").is_null()).then(pl.lit(None)).otherwise(lag_expr)
    df = df.with_columns(
        pl.col("_xcorr_max_abs").alias("corr_ref1_xcorr_max"),
        lag_expr.cast(pl.Int64).alias("corr_ref1_xcorr_lag_bars"),
    )

    if vol_proxy == "atr" and "atr" in candles.columns:
        vol_series = pl.col("atr").cast(pl.Float64).alias("_vol_proxy")
    else:
        vol_series = (
            pl.col("_ret")
            .rolling_std(window_size=vol_window, min_periods=max(3, vol_window // 2))
            .over("instrument")
            .alias("_vol_proxy")
        )
    df = df.with_columns(vol_series)
    market_vol = (
        df.group_by("ts")
        .agg(pl.col("_vol_proxy").mean().alias("_market_vol"))
        .sort("ts")
    )
    df = df.join(market_vol, on="ts", how="left")
    vol_by = ["instrument"]
    vol_mean = pl.col("_vol_proxy").rolling_mean(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_mkt_mean = pl.col("_market_vol").rolling_mean(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_prod = (pl.col("_vol_proxy") * pl.col("_market_vol")).rolling_mean(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_std = pl.col("_vol_proxy").rolling_std(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_mkt_std = pl.col("_market_vol").rolling_std(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_cov = vol_prod - (vol_mean * vol_mkt_mean)
    df = df.with_columns(
        safe_div(vol_cov, vol_std * vol_mkt_std, default=None).alias("corr_vol_ref1")
    )

    peer_vol = df.select(
        pl.col("instrument").alias("_peer"),
        pl.col("ts"),
        pl.col("_vol_proxy").alias("_peer_vol"),
    )
    df = df.join(peer_vol, left_on=["corr_ref2_id", "ts"], right_on=["_peer", "ts"], how="left")
    vol_peer_mean = pl.col("_peer_vol").rolling_mean(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_peer_std = pl.col("_peer_vol").rolling_std(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_peer_prod = (pl.col("_vol_proxy") * pl.col("_peer_vol")).rolling_mean(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_peer_cov = vol_peer_prod - (vol_mean * vol_peer_mean)
    df = df.with_columns(
        safe_div(vol_peer_cov, vol_std * vol_peer_std, default=None).alias("corr_vol_ref2")
    ).drop(["_peer", "_peer_vol"], strict=False)

    df = df.join(peer_vol, left_on=["corr_ref3_id", "ts"], right_on=["_peer", "ts"], how="left")
    vol_peer_mean = pl.col("_peer_vol").rolling_mean(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_peer_std = pl.col("_peer_vol").rolling_std(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_peer_prod = (pl.col("_vol_proxy") * pl.col("_peer_vol")).rolling_mean(window_size=vol_window, min_periods=vol_window).over(vol_by)
    vol_peer_cov = vol_peer_prod - (vol_mean * vol_peer_mean)
    df = df.with_columns(
        safe_div(vol_peer_cov, vol_std * vol_peer_std, default=None).alias("corr_vol_ref3")
    ).drop(["_peer", "_peer_vol"], strict=False)

    out = df.select(
        "instrument",
        "ts",
        "corr_dxy",
        "corr_index_major",
        "corr_oil",
        "micro_corr_regime",
        "corr_cluster_id",
        "corr_ref1_id",
        "corr_ref1",
        "corr_ref2_id",
        "corr_ref2",
        "corr_ref3_id",
        "corr_ref3",
        "micro_corr_sign",
        "micro_corr_strength",
        "micro_corr_confidence",
        "micro_corr_std",
        "micro_corr_unstable_flag",
        "micro_corr_flip_flag",
        "corr_ref1_xcorr_max",
        "corr_ref1_xcorr_lag_bars",
        "corr_vol_ref1",
        "corr_vol_ref2",
        "corr_vol_ref3",
        "corr_topk_neighbors_json",
    )

    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "ts"],
        where="corr_micro",
        allow_extra=False,
    )

    log.info("corr_micro: built rows=%d window=%d", out.height, window)
    return out
