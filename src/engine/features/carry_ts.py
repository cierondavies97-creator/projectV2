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
    DEV_STUB: carry / term-structure time-series features.

    Future design will include roll yields, carry estimates, and related
    term-structure features in data/features.

    Currently: returns empty frame.
    """
    log.warning("DEV_STUB: carry_ts.build_feature_frame called as no-op")
    return pl.DataFrame()
