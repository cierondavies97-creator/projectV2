from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

# Where the engine is running
EnvId = Literal["research", "paper", "live"]

# What kind of lane this run is (used in paths like data/candles/mode=...)
Mode = Literal["backtest", "paper", "live"]


@dataclass(frozen=True)
class RunContext:
    """
    Global identity + control for a single engine or trainer run.

    Every job (microbatch, trainer, GA, etc.) must be passed a RunContext.
    Nothing in the engine is allowed to invent snapshot_id or run_id itself.
    """

    env: EnvId  # research | paper | live
    mode: Mode  # backtest | paper | live
    snapshot_id: str  # pins config world (snapshots/<snapshot_id>.json)
    run_id: str  # unique per job under (env, snapshot_id)
    experiment_id: str | None = None  # research-only
    candidate_id: str | None = None  # research-only
    base_seed: int = 0  # root seed for RNG derivation


@dataclass(frozen=True)
class MicrobatchKey:
    """
    Unit of work for the deterministic engine pipeline.

    One microbatch processes a single trading_day for a configured
    instrument cluster (cluster_id) under a given RunContext.
    """

    trading_day: date  # trading date for this microbatch
    cluster_id: str  # cluster label defined in conf/retail.yaml
