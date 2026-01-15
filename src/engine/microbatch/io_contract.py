from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from engine.microbatch.types import CANONICAL_TABLE_KEYS

ModeGuard = Literal["always", "backtest_only", "never"]
ContractMode = Literal["current", "target"]


@dataclass(frozen=True)
class TableContract:
    """
    Ownership contract for one canonical BatchState table key.

    key:
      Canonical BatchState key (should appear in CANONICAL_TABLE_KEYS, except for transitional keys).

    dataset:
      On-disk dataset folder (e.g. "decisions", "trade_paths", "features"). For purely in-memory tables
      this can be None.

    persisted_current / persisted_target:
      Whether this key is expected to be persisted in the current code vs in the Phase-B target.

    owner_step_current / owner_step_target:
      Which microbatch step is responsible for the persistence boundary.

      Note: a table may be mutated by multiple steps in-memory; "owner" is the step that writes it to disk.

    writer_fn:
      Canonical writer function name invoked by the owning step.

    stage:
      For decisions stage keys, the stage string ("critic", "pretrade", ...).

    mode_guard:
      Execution mode guard for the owning write (e.g. fills/orders only exist in backtest SIM).

    mutators:
      Steps that may set/overwrite the key in BatchState (in-memory). Used for sanity checks.

    notes:
      Human notes / rationale.
    """
    key: str
    dataset: Optional[str]
    persisted_current: bool
    persisted_target: bool
    owner_step_current: Optional[str]
    owner_step_target: Optional[str]
    writer_fn: Optional[str]
    stage: Optional[str] = None
    mode_guard: ModeGuard = "always"
    mutators: tuple[str, ...] = ()
    notes: str = ""


# ---------------------------------------------------------------------------
# Canonical contract (edit this file; do not auto-generate)
# ---------------------------------------------------------------------------

CONTRACT: dict[str, TableContract] = {
    # -------------------------
    # Raw-ish inputs (loaded by ingest_step, written by separate ingest jobs)
    # -------------------------
    "candles": TableContract(
        key="candles",
        dataset="candles",
        persisted_current=False,
        persisted_target=False,
        owner_step_current=None,
        owner_step_target=None,
        writer_fn=None,
        mutators=("ingest_step",),
        notes="Loaded by ingest_step from pre-ingested Parquet; not written by microbatch.",
    ),
    "ticks": TableContract(
        key="ticks",
        dataset="ticks",
        persisted_current=False,
        persisted_target=False,
        owner_step_current=None,
        owner_step_target=None,
        writer_fn=None,
        mutators=("ingest_step",),
        notes="Optional input; loaded by ingest_step; not written by microbatch.",
    ),
    "external": TableContract(
        key="external",
        dataset="external",
        persisted_current=False,
        persisted_target=False,
        owner_step_current=None,
        owner_step_target=None,
        writer_fn=None,
        mutators=("ingest_step",),
        notes="Placeholder for external/macro joins; currently in-memory only.",
    ),
    "macro": TableContract(
        key="macro",
        dataset="macro",
        persisted_current=False,
        persisted_target=False,
        owner_step_current=None,
        owner_step_target=None,
        writer_fn=None,
        mutators=("ingest_step",),
        notes="Optional input; loaded by ingest_step; not written by microbatch.",
    ),

    # -------------------------
    # Feature / memory model tables
    # -------------------------
    "features": TableContract(
        key="features",
        dataset="features",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="features_step",
        owner_step_target="features_step",
        writer_fn="write_features_for_instrument_tf_day",
        mutators=("features_step",),
        notes="Deterministic features are persisted once per (instrument, anchor_tf, dt).",
    ),
    "zones_state": TableContract(
        key="zones_state",
        dataset="zones_state",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="features_step",
        owner_step_target="features_step",
        writer_fn="write_zones_state_for_instrument_tf_day",
        mutators=("features_step",),
        notes="Long-lived memory; currently persisted by features_step.",
    ),
    "pcr_a": TableContract(
        key="pcr_a",
        dataset="pcr_a",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="features_step",
        owner_step_target="features_step",
        writer_fn="write_pcra_for_instrument_tf_day",
        mutators=("features_step",),
        notes="PCrA microstructure memory; currently persisted by features_step.",
    ),
    "windows": TableContract(
        key="windows",
        dataset="windows",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="windows_step",
        owner_step_target="windows_step",
        writer_fn="write_windows_for_instrument_tf_day",
        mutators=("windows_step",),
        notes="Windows are derived from features/state; persisted by windows_step.",
    ),

    # -------------------------
    # Trade path and context tables
    # -------------------------
    "trade_paths": TableContract(
        key="trade_paths",
        dataset="trade_paths",
        # Current code persists early (hypotheses_step) but then the table is mutated later in-memory.
        persisted_current=True,
        persisted_target=True,
        owner_step_current="hypotheses_step",
        # Phase-B target (recommended): persist once at the end, after all in-memory mutations.
        owner_step_target="reports_step",
        writer_fn="write_trade_paths_for_day",
        mutators=("hypotheses_step", "critic_step", "pretrade_step", "gatekeeper_step", "portfolio_step"),
        notes=(
            "Current: persisted by hypotheses_step, then mutated later in-memory. "
            "Target: persist final trade_paths once at end (reports_step) for auditability."
        ),
    ),
    "principles_context": TableContract(
        key="principles_context",
        dataset="principles_context",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="gatekeeper_step",
        owner_step_target="gatekeeper_step",
        writer_fn="write_principles_context_for_cluster_day",  # target will need a writer (e.g. write_principles_context_for_day)
        mutators=("gatekeeper_step",),
        notes="Currently in-memory only; Phase-B target is to persist for training/audit.",
    ),
    "trade_clusters": TableContract(
        key="trade_clusters",
        dataset="trade_clusters",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="portfolio_step",
        owner_step_target="portfolio_step",
        writer_fn="write_trade_clusters_for_cluster_day",  # target will need a writer
        mutators=(),
        notes="Currently not produced; Phase-B target: persist clustering used by portfolio/risk.",
    ),

    # -------------------------
    # Decisions (single dataset partitioned by stage)
    # -------------------------
    "decisions_hypotheses": TableContract(
        key="decisions_hypotheses",
        dataset="decisions",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="hypotheses_step",
        owner_step_target="hypotheses_step",
        writer_fn="write_decisions_for_stage",
        stage="hypotheses",
        mutators=("hypotheses_step",),
        notes="Stage-partitioned decisions dataset: hypotheses stage.",
    ),
    "decisions_critic": TableContract(
        key="decisions_critic",
        dataset="decisions",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="critic_step",
        owner_step_target="critic_step",
        writer_fn="write_decisions_for_stage",
        stage="critic",
        mutators=("critic_step",),
        notes="Stage-partitioned decisions dataset: critic stage.",
    ),
    "decisions_pretrade": TableContract(
        key="decisions_pretrade",
        dataset="decisions",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="pretrade_step",
        owner_step_target="pretrade_step",
        writer_fn="write_decisions_for_stage",
        stage="pretrade",
        mutators=("pretrade_step",),
        notes="Stage-partitioned decisions dataset: pretrade stage.",
    ),
    "decisions_gatekeeper": TableContract(
        key="decisions_gatekeeper",
        dataset="decisions",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="gatekeeper_step",
        owner_step_target="gatekeeper_step",
        writer_fn="write_decisions_for_stage",
        stage="gatekeeper",
        mutators=("gatekeeper_step",),
        notes="Stage-partitioned decisions dataset: gatekeeper stage.",
    ),
    "decisions_portfolio": TableContract(
        key="decisions_portfolio",
        dataset="decisions",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="portfolio_step",
        owner_step_target="portfolio_step",
        writer_fn="write_decisions_for_stage",
        stage="portfolio",
        mutators=("portfolio_step",),
        notes="Stage-partitioned decisions dataset: portfolio stage.",
    ),

    # -------------------------
    # Execution artifacts
    # -------------------------
    "brackets": TableContract(
        key="brackets",
        dataset="brackets",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="brackets_step",
        owner_step_target="brackets_step",
        writer_fn="write_brackets_for_instrument_day",
        mutators=("brackets_step",),
        notes="Bracket plans persisted by brackets_step.",
    ),

    # NOTE: orders/fills are currently produced by fills_step (backtest SIM),
    # but your CANONICAL_TABLE_KEYS does not yet include these keys. This contract
    # includes them as transitional keys; the audit will flag that mismatch.
    "orders": TableContract(
        key="orders",
        dataset="orders",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="fills_step",
        owner_step_target="fills_step",
        writer_fn="write_orders_for_instrument_day",
        mode_guard="backtest_only",
        mutators=("fills_step",),
        notes="Phase-A SIM only (backtest). Add 'orders' to CANONICAL_TABLE_KEYS to remove drift.",
    ),
    "fills": TableContract(
        key="fills",
        dataset="fills",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="fills_step",
        owner_step_target="fills_step",
        writer_fn="write_fills_for_instrument_day",
        mode_guard="backtest_only",
        mutators=("fills_step",),
        notes="Phase-A SIM only (backtest). Add 'fills' to CANONICAL_TABLE_KEYS to remove drift.",
    ),

    # -------------------------
    # Diagnostics / reports
    # -------------------------
    "critic": TableContract(
        key="critic",
        dataset=None,
        persisted_current=False,
        persisted_target=False,
        owner_step_current=None,
        owner_step_target=None,
        writer_fn=None,
        mutators=("critic_step",),
        notes="Critic diagnostics currently in-memory only.",
    ),
    "reports": TableContract(
        key="reports",
        dataset="run_reports",
        persisted_current=True,
        persisted_target=True,
        owner_step_current="reports_step",
        owner_step_target="reports_step",
        writer_fn="write_run_reports_for_cluster_day",
        mutators=("reports_step",),
        notes="Run reports persisted by reports_step.",
    ),
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def expected_owner(contract: TableContract, mode: ContractMode) -> Optional[str]:
    return contract.owner_step_current if mode == "current" else contract.owner_step_target


def expected_persisted(contract: TableContract, mode: ContractMode) -> bool:
    return contract.persisted_current if mode == "current" else contract.persisted_target


def validate_contract(*, mode: ContractMode = "current") -> list[str]:
    """
    Validate the manifest structurally (no filesystem/AST scan).

    Returns list of human-readable issues. Empty list means OK.
    """
    issues: list[str] = []

    # 1) Every canonical key should be present in the contract
    canonical = set(CANONICAL_TABLE_KEYS)
    contract_keys = set(CONTRACT.keys())

    missing = sorted(canonical - contract_keys)
    if missing:
        issues.append(f"Missing CONTRACT entries for CANONICAL_TABLE_KEYS: {missing}")

    # 2) Warn about contract keys not in canonical (transitional keys)
    extra = sorted(contract_keys - canonical)
    if extra:
        issues.append(
            "CONTRACT contains keys not present in CANONICAL_TABLE_KEYS (transitional drift): "
            f"{extra}. Consider updating CANONICAL_TABLE_KEYS."
        )

    # 3) Persisted tables must have owner + writer_fn in the given mode
    for k, c in CONTRACT.items():
        if expected_persisted(c, mode):
            if expected_owner(c, mode) is None:
                issues.append(f"{k}: persisted_{mode}=True but owner_step_{mode} is None")
            if c.writer_fn is None:
                issues.append(f"{k}: persisted_{mode}=True but writer_fn is None")

    # 4) Decisions keys should have a stage if they use write_decisions_for_stage
    for k, c in CONTRACT.items():
        if c.writer_fn == "write_decisions_for_stage" and not c.stage:
            issues.append(f"{k}: writer_fn=write_decisions_for_stage but stage is missing")

    return issues
