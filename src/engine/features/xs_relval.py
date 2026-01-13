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
    DEV_STUB: cross-section relative value features.

    Final implementation will compute cross-sectional signals between
    instruments (spreads, ranks, etc.) into data/features.

    For now: no-op stub.
    """
    log.warning("DEV_STUB: xs_relval.build_feature_frame called as no-op")
    return pl.DataFrame()
