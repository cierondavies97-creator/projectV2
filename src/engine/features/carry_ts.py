from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry, safe_div

log = logging.getLogger(__name__)


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
            "carry_ts_carry_score": pl.Series([], dtype=pl.Float64),
            "carry_ts_ts_slope": pl.Series([], dtype=pl.Float64),
            "carry_ts_ts_regime": pl.Series([], dtype=pl.Utf8),
        }
    )


def _base_keys_from_candles(candles: pl.DataFrame, anchor_tfs: list[str]) -> pl.DataFrame:
    if candles is None or candles.is_empty():
        return pl.DataFrame()
    if not {"instrument", "tf", "ts"}.issubset(set(candles.columns)):
        return pl.DataFrame()
    df = candles.select(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("tf").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
    ).drop_nulls(["instrument", "tf", "ts"])
    if anchor_tfs:
        df = df.filter(pl.col("tf").is_in(anchor_tfs))
    return (
        df.select(
            pl.col("instrument"),
            pl.col("tf").alias("anchor_tf"),
            pl.col("ts").alias("anchor_ts"),
        )
        .unique()
        .sort(["instrument", "anchor_tf", "anchor_ts"])
    )


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg: dict[str, object] = {}
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("carry_ts", {}) or {})
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
    Carry/term-structure proxy using rolling returns.

    Table: data/windows
    Keys : instrument, anchor_tf, anchor_ts
    """
    if candles is None or candles.is_empty():
        log.warning("carry_ts: candles empty; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

    required = {"instrument", "tf", "ts", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("carry_ts: missing columns=%s; returning null-filled frame", missing)
        base = _base_keys_from_candles(candles, [])
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        out = base.with_columns(
            pl.lit(None).cast(pl.Float64).alias("carry_ts_carry_score"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_ts_slope"),
            pl.lit(None).cast(pl.Utf8).alias("carry_ts_ts_regime"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_level"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_level_z"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_slope_z"),
            pl.lit(None).cast(pl.Utf8).alias("carry_ts_sign_bucket"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_confidence"),
            pl.lit(None).cast(pl.Utf8).alias("carry_ts_source_used"),
            pl.lit(None).cast(pl.Utf8).alias("carry_ts_proxy_method_used"),
            pl.lit(None).cast(pl.Boolean).alias("carry_ts_missing_flag"),
            pl.lit(None).cast(pl.Int64).alias("carry_ts_data_age_bars"),
        )
        return conform_to_registry(
            out,
            registry_entry=registry_entry,
            key_cols=["instrument", "anchor_tf", "anchor_ts"],
            where="carry_ts",
            allow_extra=False,
        )

    fam_cfg = _merge_cfg(ctx, family_cfg)

    slope_window = max(5, int(fam_cfg.get("carry_slope_window_bars", 60)))
    score_window = max(10, int(fam_cfg.get("carry_score_window_bars", 252)))
    flat_cut = float(fam_cfg.get("carry_regime_flat_cut", 0.10))
    strong_cut = float(fam_cfg.get("carry_regime_strong_cut", 1.00))
    min_periods = max(5, int(fam_cfg.get("carry_min_periods", 10)))

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs and "tf" in candles.columns:
        anchor_tfs = (
            candles.select(pl.col("tf").cast(pl.Utf8))
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
    if not anchor_tfs:
        log.warning("carry_ts: no anchor_tfs resolved; returning empty keyed frame")
        return _empty_keyed_frame(registry_entry)

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
        carry_score = (
            pl.col("_ret")
            .rolling_mean(window_size=score_window, min_periods=min_periods)
            .over(by)
            .alias("carry_ts_carry_score")
        )
        carry_slope = (
            safe_div(
                pl.col("close") - pl.col("close").shift(slope_window).over("instrument"),
                pl.lit(float(slope_window)),
                default=0.0,
            )
            .over("instrument")
            .alias("carry_ts_ts_slope")
        )

        df = df.with_columns(carry_score, carry_slope)

        z_mean = pl.col("carry_ts_carry_score").rolling_mean(window_size=score_window, min_periods=min_periods).over(by)
        z_std = pl.col("carry_ts_carry_score").rolling_std(window_size=score_window, min_periods=min_periods).over(by)
        z = safe_div(pl.col("carry_ts_carry_score") - z_mean, z_std, default=0.0)

        df = df.with_columns(
            pl.when(z.abs() < pl.lit(flat_cut))
            .then(pl.lit("flat"))
            .when(z >= pl.lit(strong_cut))
            .then(pl.lit("contango"))
            .when(z <= pl.lit(-strong_cut))
            .then(pl.lit("backwardated"))
            .otherwise(pl.lit("unknown"))
            .alias("carry_ts_ts_regime"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_level"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_level_z"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_slope_z"),
            pl.when(pl.col("carry_ts_ts_regime") == pl.lit("contango"))
            .then(pl.lit("positive"))
            .when(pl.col("carry_ts_ts_regime") == pl.lit("backwardated"))
            .then(pl.lit("negative"))
            .when(pl.col("carry_ts_ts_regime") == pl.lit("flat"))
            .then(pl.lit("flat"))
            .otherwise(pl.lit("unknown"))
            .alias("carry_ts_sign_bucket"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_confidence"),
            pl.lit(str(fam_cfg.get("carry_source", "external"))).alias("carry_ts_source_used"),
            pl.lit(str(fam_cfg.get("carry_proxy_method", "none"))).alias("carry_ts_proxy_method_used"),
            pl.lit(False).cast(pl.Boolean).alias("carry_ts_missing_flag"),
            pl.lit(0).cast(pl.Int64).alias("carry_ts_data_age_bars"),
        )

        out_frames.append(
            df.select(
                pl.col("instrument"),
                pl.col("anchor_tf"),
                pl.col("ts").alias("anchor_ts"),
                "carry_ts_carry_score",
                "carry_ts_ts_slope",
                "carry_ts_ts_regime",
                "carry_ts_level",
                "carry_ts_level_z",
                "carry_ts_slope_z",
                "carry_ts_sign_bucket",
                "carry_ts_confidence",
                "carry_ts_source_used",
                "carry_ts_proxy_method_used",
                "carry_ts_missing_flag",
                "carry_ts_data_age_bars",
            )
        )

    if not out_frames:
        base = _base_keys_from_candles(candles, [str(tf) for tf in anchor_tfs])
        if base.is_empty():
            return _empty_keyed_frame(registry_entry)
        out = base.with_columns(
            pl.lit(None).cast(pl.Float64).alias("carry_ts_carry_score"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_ts_slope"),
            pl.lit(None).cast(pl.Utf8).alias("carry_ts_ts_regime"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_level"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_level_z"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_slope_z"),
            pl.lit(None).cast(pl.Utf8).alias("carry_ts_sign_bucket"),
            pl.lit(None).cast(pl.Float64).alias("carry_ts_confidence"),
            pl.lit(str(fam_cfg.get("carry_source", "external"))).alias("carry_ts_source_used"),
            pl.lit(str(fam_cfg.get("carry_proxy_method", "none"))).alias("carry_ts_proxy_method_used"),
            pl.lit(True).cast(pl.Boolean).alias("carry_ts_missing_flag"),
            pl.lit(None).cast(pl.Int64).alias("carry_ts_data_age_bars"),
        )
        return conform_to_registry(
            out,
            registry_entry=registry_entry,
            key_cols=["instrument", "anchor_tf", "anchor_ts"],
            where="carry_ts",
            allow_extra=False,
        )

    out = pl.concat(out_frames, how="vertical").sort(["instrument", "anchor_tf", "anchor_ts"])
    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "anchor_ts"],
        where="carry_ts",
        allow_extra=False,
    )

    log.info("carry_ts: built rows=%d", out.height)
    return out
