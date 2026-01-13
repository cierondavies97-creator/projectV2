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
    DEV_STUB: trade path classification features.

    Full implementation will attach class IDs / confidences to windows
    or path candidates in data/features.

    For now: no-op stub.
    """
    log.warning("DEV_STUB: trade_path_class.build_feature_frame called as no-op")
    return pl.DataFrame()
