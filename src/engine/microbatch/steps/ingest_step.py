from __future__ import annotations

import polars as pl

from engine.core.config_models import (
    ClusterPlan,
    build_cluster_plan,
    load_retail_config,
)
from engine.io.parquet_io import read_parquet_dir
from engine.io.paths import (
    candles_dir,
    macro_dir,
    ticks_dir,
)
from engine.microbatch.types import BatchState
from engine.research.snapshots import load_snapshot_manifest


def _cluster_plan_for_state(state: BatchState) -> ClusterPlan:
    """
    Resolve the ClusterPlan for this (RunContext, MicrobatchKey).

    Uses:
      - snapshots/<snapshot_id>.json   -> SnapshotManifest
      - conf/retail.yaml               -> RetailConfig

    Then applies build_cluster_plan(...) to get:
      - instruments
      - anchor_tfs
      - entry_tfs
    """
    snapshot = load_snapshot_manifest(state.ctx.snapshot_id)
    retail = load_retail_config()
    return build_cluster_plan(snapshot, retail, state.key.cluster_id)


def _read_cluster_candles(state: BatchState, plan: ClusterPlan) -> pl.DataFrame:
    """
    Load candles for all instruments in the cluster for this trading day.

    Expects per-instrument partitions under:

        data/candles/mode=<MODE>/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/

    For each instrument:
      - Read all *.parquet files in that directory.
      - Ensure there is an 'instrument' column (add if missing).
    """
    frames: list[pl.DataFrame] = []

    for instrument_id in plan.instruments:
        dir_path = candles_dir(
            state.ctx,
            state.key.trading_day,
            instrument_id=instrument_id,
        )
        df = read_parquet_dir(dir_path)

        # Ensure we always have an 'instrument' column for downstream joins.
        if "instrument" not in df.columns:
            df = df.with_columns(pl.lit(instrument_id).alias("instrument"))

        frames.append(df)

    if not frames:
        # This should not happen if ClusterPlan has instruments and data is present.
        # If it does, return an empty DataFrame.
        return pl.DataFrame()

    return pl.concat(frames, how="vertical")


def _read_cluster_ticks(state: BatchState, plan: ClusterPlan) -> pl.DataFrame:
    """
    Load ticks for all instruments in the cluster for this trading day.

    Expects per-instrument partitions under:

        data/ticks/mode=<MODE>/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/

    If a tick directory is missing for an instrument, it is skipped.
    """
    frames: list[pl.DataFrame] = []

    for instrument_id in plan.instruments:
        dir_path = ticks_dir(
            state.ctx,
            state.key.trading_day,
            instrument_id=instrument_id,
        )

        try:
            df = read_parquet_dir(dir_path)
        except FileNotFoundError:
            # Tick data is optional; just skip if not present.
            continue

        if "instrument" not in df.columns:
            df = df.with_columns(pl.lit(instrument_id).alias("instrument"))

        frames.append(df)

    if not frames:
        # No tick data for this cluster/day -> empty frame.
        return pl.DataFrame()

    return pl.concat(frames, how="vertical")


def _read_macro(state: BatchState) -> pl.DataFrame:
    """
    Load macro / calendar state for this trading day.

    Expects partitions under:

        data/macro/dt=<YYYY-MM-DD>/

    If the directory is missing or empty, returns an empty DataFrame.
    """
    dir_path = macro_dir(
        state.ctx,
        state.key.trading_day,
    )

    try:
        return read_parquet_dir(dir_path)
    except FileNotFoundError:
        return pl.DataFrame()


def run(state: BatchState) -> BatchState:
    """
    Ingest step.

    Responsibilities:
    - Resolve the ClusterPlan for (state.ctx, state.key).
    - Read raw data for (state.key.trading_day, cluster instruments) from:
        * data/candles/...
        * data/ticks/...  (optional, may be empty)
        * data/macro/...  (optional, may be empty)
      according to state.ctx.env and state.ctx.mode.

    - Populate the following tables in state.tables:
        * 'candles'
        * 'ticks'
        * 'macro'

    - At this stage, no features or decisions are computed; this step
      only brings raw inputs into the BatchState.
    """
    plan = _cluster_plan_for_state(state)

    candles_df = _read_cluster_candles(state, plan)
    ticks_df = _read_cluster_ticks(state, plan)
    macro_df = _read_macro(state)

    state.set("candles", candles_df)
    state.set("ticks", ticks_df)
    state.set("macro", macro_df)

    # 'external' is left for a dedicated external data ingest path and
    # is not loaded here yet.

    return state
