"""
Canonical import surface for engine.data writers/readers.

This module should remain side-effect free.
Importing it must not trigger any registration, IO, or config mutation.
"""

from __future__ import annotations

from engine.data.brackets import read_brackets_for_instrument_day, write_brackets_for_instrument_day
from engine.data.decisions import read_decisions_for_stage, write_decisions_for_stage
from engine.data.fills import read_fills_for_instrument_day, write_fills_for_instrument_day
from engine.data.orders import read_orders_for_instrument_day, write_orders_for_instrument_day
from engine.data.run_reports import read_run_reports_for_cluster_day, write_run_reports_for_cluster_day
from engine.data.trade_paths import read_trade_paths_for_day, write_trade_paths_for_day

__all__ = [
    # decisions
    "write_decisions_for_stage",
    "read_decisions_for_stage",
    # trade_paths
    "write_trade_paths_for_day",
    "read_trade_paths_for_day",
    # brackets
    "write_brackets_for_instrument_day",
    "read_brackets_for_instrument_day",
    # orders
    "write_orders_for_instrument_day",
    "read_orders_for_instrument_day",
    # fills
    "write_fills_for_instrument_day",
    "read_fills_for_instrument_day",
    # run_reports
    "write_run_reports_for_cluster_day",
    "read_run_reports_for_cluster_day",
]
