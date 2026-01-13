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
    DEV_STUB: PCrA tick-level microstructure features.

    Full implementation will add tick-resolution PCrA columns
    feeding into data/pcr_a.

    Currently disabled: returns empty frame.
    """
    log.warning("DEV_STUB: pcra_tick.build_feature_frame called as no-op")
    return pl.DataFrame()
