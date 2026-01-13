from __future__ import annotations

# --- bootstrap so Python can find src/engine when running from scripts/ ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# -------------------------------------------------------------------------

import argparse
import logging
from datetime import date

from engine.api import EnvId, MicrobatchKey, Mode, RunContext, run_microbatch


def _default_mode_for_env(env: EnvId) -> Mode:
    """
    Default engine mode for a given environment.

    - research -> backtest
    - paper    -> paper
    - live     -> live
    """
    if env == "research":
        return "backtest"
    if env == "paper":
        return "paper"
    if env == "live":
        return "live"
    raise ValueError(f"Unknown env: {env}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single microbatch for (trading_day, cluster_id).")

    parser.add_argument(
        "--env",
        type=str,
        choices=["research", "paper", "live"],
        required=True,
        help="Environment: research | paper | live",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "paper", "live"],
        required=False,
        help="Engine mode: backtest | paper | live (default depends on env)",
    )

    parser.add_argument(
        "--snapshot-id",
        type=str,
        required=True,
        help="Snapshot ID (matches snapshots/<snapshot_id>.json)",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID for this job (unique under (env, snapshot_id))",
    )

    parser.add_argument(
        "--trading-day",
        type=str,
        required=True,
        help="Trading day in YYYY-MM-DD format",
    )

    parser.add_argument(
        "--cluster-id",
        type=str,
        required=True,
        help="Instrument cluster ID (defined in conf/retail.yaml)",
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base RNG seed for this run (default: 0)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level (default: INFO)",
    )

    return parser.parse_args()


# ... existing imports and parse_args ...


def main() -> None:
    args = parse_args()

    # Logging setup
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    env: EnvId = args.env  # type: ignore[assignment]
    if args.mode is not None:
        mode: Mode = args.mode  # type: ignore[assignment]
    else:
        mode = _default_mode_for_env(env)

    trading_day = date.fromisoformat(args.trading_day)

    ctx = RunContext(
        env=env,
        mode=mode,
        snapshot_id=args.snapshot_id,
        run_id=args.run_id,
        experiment_id=None,
        candidate_id=None,
        base_seed=args.base_seed,
    )

    key = MicrobatchKey(
        trading_day=trading_day,
        cluster_id=args.cluster_id,
    )

    logging.getLogger(__name__).info(
        "Starting microbatch run env=%s mode=%s snapshot_id=%s run_id=%s trading_day=%s cluster_id=%s base_seed=%d",
        ctx.env,
        ctx.mode,
        ctx.snapshot_id,
        ctx.run_id,
        key.trading_day.isoformat(),
        key.cluster_id,
        ctx.base_seed,
    )

    # Run the deterministic pipeline once
    state = run_microbatch(ctx, key)

    # Simple shape summary
    for name in (
        "features",
        "windows",
        "trade_paths",
        "decisions_hypotheses",
        "decisions_critic",
        "decisions_pretrade",
        "decisions_gatekeeper",
        "decisions_portfolio",
        "brackets",
        "reports",
    ):
        df = state.tables.get(name)
        n = 0 if df is None else df.height
        print(f"{name}: {n} rows")


if __name__ == "__main__":
    main()
