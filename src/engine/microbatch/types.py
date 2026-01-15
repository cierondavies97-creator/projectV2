from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _ensure_pkg(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(SRC / name.replace(".", "/"))]
    sys.modules[name] = pkg
    return pkg


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
