from __future__ import annotations

import logging

import polars as pl

from engine.features import FeatureBuildContext

log = logging.getLogger(__name__)


def build_feature_frame(
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None = None,
    macro: pl.DataFrame | None = None,
    external: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    DEV_STUB: higher timeframe correlation / regime features.

    Final design will attach HTF correlation signals to the features table.

    For now: no-op, returns empty.
    """
    log.warning("DEV_STUB: corr_htf.build_feature_frame called as no-op")
    return pl.DataFrame()
