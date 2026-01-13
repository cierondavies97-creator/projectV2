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
    DEV_STUB: Volume profile core zone features.

    Will eventually emit VP nodes/zones into data/zones_state and attach
    node strength, ranges, etc.

    Currently: returns empty frame.
    """
    log.warning("DEV_STUB: vp_core.build_feature_frame called as no-op")
    return pl.DataFrame()
