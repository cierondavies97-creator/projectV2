from __future__ import annotations

import logging

import polars as pl

from engine.features import FeatureBuildContext

log = logging.getLogger(__name__)


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": pl.Series([], dtype=pl.Datetime("us")),
            "macro_regime_id": pl.Series([], dtype=pl.Utf8),
            "macro_regime_label": pl.Series([], dtype=pl.Utf8),
            "macro_regime_confidence": pl.Series([], dtype=pl.Float64),
        }
    )


def build_feature_frame(
    ctx: FeatureBuildContext,
    candles: pl.DataFrame | None = None,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Macro regime features derived from macro calendar state.

    Table: data/macro
    Keys : ts
    """
    if macro is None or macro.is_empty():
        log.warning("macro_regime: macro empty; returning empty keyed frame")
        return _empty_keyed_frame()

    if "ts" not in macro.columns:
        log.warning("macro_regime: macro missing ts column; returning empty keyed frame")
        return _empty_keyed_frame()

    df = macro.select(
        pl.col("ts").cast(pl.Datetime("us"), strict=False),
        pl.col("macro_state").cast(pl.Utf8, strict=False).alias("_macro_state")
        if "macro_state" in macro.columns
        else pl.lit(None).cast(pl.Utf8).alias("_macro_state"),
        pl.col("impact_level").cast(pl.Float64, strict=False).alias("_impact")
        if "impact_level" in macro.columns
        else pl.lit(None).cast(pl.Float64).alias("_impact"),
    ).drop_nulls(["ts"])

    regime_label = (
        pl.when(pl.col("_macro_state").is_in(["red_window", "pre_event", "post_event"]))
        .then(pl.lit("event_risk"))
        .when(pl.col("_macro_state") == pl.lit("quiet"))
        .then(pl.lit("risk_on"))
        .when(pl.col("_macro_state") == pl.lit("unknown"))
        .then(pl.lit("unknown"))
        .otherwise(pl.lit("risk_on"))
        .alias("macro_regime_label")
    )

    conf = (
        pl.when(pl.col("_impact").is_null())
        .then(pl.lit(0.0))
        .otherwise((pl.col("_impact") / pl.lit(3.0)).clip(0.0, 1.0))
        .alias("macro_regime_confidence")
    )

    out = df.with_columns(
        regime_label,
        conf,
        pl.col("macro_regime_label").alias("macro_regime_id"),
    ).select(
        "ts",
        "macro_regime_id",
        "macro_regime_label",
        "macro_regime_confidence",
    )

    log.info("macro_regime: built rows=%d", out.height)
    return out
