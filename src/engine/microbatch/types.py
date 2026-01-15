from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


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

def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

        Decision / execution:
          - 'decisions_hypotheses' : raw hypothesis decisions
          - 'decisions_critic'     : critic-scored decisions
          - 'decisions_pretrade'   : pretrade-filtered decisions
          - 'decisions_gatekeeper' : gatekeeper-approved decisions
          - 'decisions_portfolio'  : portfolio-level allocations/decisions
          - 'brackets'             : final bracket plans (entries, stops, TPs)
          - 'orders'               : order events (SIM/backtest or broker adapter)
          - 'fills'                : fills (SIM/backtest or broker adapter)

_ensure_pkg("engine")
_ensure_pkg("engine.microbatch")
_ensure_pkg("engine.microbatch.steps")

io_contract = _load_module(
    "engine.microbatch.io_contract",
    SRC / "engine" / "microbatch" / "io_contract.py",
)
contract_guard = _load_module(
    "engine.microbatch.steps.contract_guard",
    SRC / "engine" / "microbatch" / "steps" / "contract_guard.py",
)

ContractWrite = contract_guard.ContractWrite
assert_contract_alignment = contract_guard.assert_contract_alignment
validate_contract = io_contract.validate_contract


def test_contract_guard_alignment_current() -> None:
    assert_contract_alignment(
        step_name="features_step",
        writes=(
            ContractWrite(table_key="features", writer_fn="write_features_for_instrument_tf_day"),
            ContractWrite(table_key="zones_state", writer_fn="write_zones_state_for_instrument_tf_day"),
            ContractWrite(table_key="pcr_a", writer_fn="write_pcra_for_instrument_tf_day"),
        ),
    )
    assert_contract_alignment(
        step_name="windows_step",
        writes=(ContractWrite(table_key="windows", writer_fn="write_windows_for_instrument_tf_day"),),
    )
    assert_contract_alignment(
        step_name="hypotheses_step",
        writes=(
            ContractWrite(
                table_key="decisions_hypotheses",
                writer_fn="write_decisions_for_stage",
                stage="hypotheses",
            ),
            ContractWrite(table_key="trade_paths", writer_fn="write_trade_paths_for_day"),
        ),
    )
    assert_contract_alignment(
        step_name="critic_step",
        writes=(
            ContractWrite(
                table_key="decisions_critic",
                writer_fn="write_decisions_for_stage",
                stage="critic",
            ),
        ),
    )
    assert_contract_alignment(
        step_name="pretrade_step",
        writes=(
            ContractWrite(
                table_key="decisions_pretrade",
                writer_fn="write_decisions_for_stage",
                stage="pretrade",
            ),
        ),
    )
    assert_contract_alignment(
        step_name="gatekeeper_step",
        writes=(
            ContractWrite(
                table_key="decisions_gatekeeper",
                writer_fn="write_decisions_for_stage",
                stage="gatekeeper",
            ),
            ContractWrite(
                table_key="principles_context",
                writer_fn="write_principles_context_for_cluster_day",
            ),
        ),
    )
    assert_contract_alignment(
        step_name="portfolio_step",
        writes=(
            ContractWrite(
                table_key="decisions_portfolio",
                writer_fn="write_decisions_for_stage",
                stage="portfolio",
            ),
            ContractWrite(
                table_key="trade_clusters",
                writer_fn="write_trade_clusters_for_cluster_day",
            ),
        ),
    )
    assert_contract_alignment(
        step_name="brackets_step",
        writes=(ContractWrite(table_key="brackets", writer_fn="write_brackets_for_instrument_day"),),
    )
    assert_contract_alignment(
        step_name="fills_step",
        writes=(
            ContractWrite(table_key="orders", writer_fn="write_orders_for_instrument_day"),
            ContractWrite(table_key="fills", writer_fn="write_fills_for_instrument_day"),
        ),
    )
    assert_contract_alignment(
        step_name="reports_step",
        writes=(ContractWrite(table_key="reports", writer_fn="write_run_reports_for_cluster_day"),),
    )


def test_contract_manifest_validate() -> None:
    issues = validate_contract(mode="current")
    assert issues == [], f"Unexpected contract validation issues: {issues}"
