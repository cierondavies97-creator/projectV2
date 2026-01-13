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
    DEV_STUB: unsupervised regime features.

    Future implementation will cluster market states and emit regime IDs
    + confidence into data/features.

    Currently: returns empty frame.
    """
    log.warning("DEV_STUB: unsup_regime.build_feature_frame called as no-op")
    return pl.DataFrame()
