from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from engine.core.ids import MicrobatchKey, RunContext


CANONICAL_TABLE_KEYS = (
    # Raw-ish inputs
    "candles",
    "ticks",
    "external",
    "macro",
    # Feature / memory model tables
    "features",
    "zones_state",
    "pcr_a",
    "windows",
    # Trade path and principle-level tables
    "trade_paths",
    "trade_clusters",
    "principles_context",
    # Decision / execution tables
    "decisions_hypotheses",
    "decisions_critic",
    "decisions_pretrade",
    "decisions_gatekeeper",
    "decisions_portfolio",
    "brackets",
    "orders",
    "fills",
    # Reports / diagnostics
    "critic",
    "reports",
)


@dataclass
class BatchState:
    """
    Shared state passed between microbatch pipeline steps.

    - ctx, key identify the run and unit of work:
        * ctx: RunContext (env, mode, snapshot_id, run_id, etc.)
        * key: MicrobatchKey (trading_day, cluster_id)

    - tables is a mapping from canonical table name -> Polars DataFrame.

      The canonical keys are:

        Raw inputs:
          - 'candles'              : candles for this trading_day & cluster
          - 'ticks'                : ticks for this trading_day & cluster
          - 'external'             : external series for this day & cluster
          - 'macro'                : macro / calendar state for this day

        Feature / memory model tables:
          - 'features'             : feature frames per (instrument, anchor_tf)
          - 'zones_state'          : ZMF + VP state
          - 'pcr_a'                : PCrA microstructure table
          - 'windows'              : anchor-time windows for this run_id/day/cluster

        Decision / execution:
          - 'decisions_hypotheses' : raw hypothesis decisions
          - 'decisions_critic'     : critic-scored decisions
          - 'decisions_pretrade'   : pretrade-filtered decisions
          - 'decisions_gatekeeper' : gatekeeper-approved decisions
          - 'decisions_portfolio'  : portfolio-level allocations/decisions
          - 'brackets'             : final bracket plans (entries, stops, TPs)
          - 'orders'               : order events (SIM/backtest or broker adapter)
          - 'fills'                : fills (SIM/backtest or broker adapter)

        Reports / diagnostics:
          - 'critic'               : critic diagnostics
          - 'reports'              : aggregated run reports
    """

    ctx: RunContext
    key: MicrobatchKey
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)

    def get(self, key: str) -> pl.DataFrame:
        if key not in self.tables:
            raise KeyError(f"BatchState missing required table key: {key!r}")
        return self.tables[key]

    def get_optional(self, key: str) -> pl.DataFrame | None:
        return self.tables.get(key)

    def set(self, key: str, value: pl.DataFrame | None) -> None:
        if value is None:
            self.tables.pop(key, None)
            return
        self.tables[key] = value

    def has(self, key: str) -> bool:
        return key in self.tables

    def keys(self) -> list[str]:
        return list(self.tables.keys())

    def as_dict(self) -> dict[str, Any]:
        return dict(self.tables)
