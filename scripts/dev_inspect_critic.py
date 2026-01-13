from __future__ import annotations

import argparse
import datetime as dt

from engine.core.api import MicrobatchKey, RunContext
from engine.microbatch.api import BatchState
from engine.microbatch.steps import (
    critic_step,
    features_step,
    hypotheses_step,
    ingest_step,
    windows_step,
)


def _parse_trading_day(s: str) -> dt.date:
    """
    Parse YYYY-MM-DD into a date.
    """
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def inspect(
    env: str,
    snapshot_id: str,
    run_id: str,
    trading_day: str,
    cluster_id: str,
) -> None:
    """
    Run ingest Ã¢â€ â€™ features Ã¢â€ â€™ windows Ã¢â€ â€™ hypotheses Ã¢â€ â€™ critic
    for a single (trading_day, cluster_id) and print summaries.
    """
    td = _parse_trading_day(trading_day)

    ctx = RunContext(
        env=env,
        mode="backtest",
        snapshot_id=snapshot_id,
        run_id=run_id,
        base_seed=0,
    )

    # NOTE: MicrobatchKey in your current code only accepts (trading_day, cluster_id)
    key = MicrobatchKey(
        trading_day=td,
        cluster_id=cluster_id,
    )

    state = BatchState(ctx=ctx, key=key)

    # Pipeline subset
    state = ingest_step.run(state)
    state = features_step.run(state)
    state = windows_step.run(state)
    state = hypotheses_step.run(state)
    state = critic_step.run(state)

    windows_df = state.get("windows")
    trade_paths_df = state.get("trade_paths")
    decisions_hypo_df = state.get("decisions_hypotheses")
    decisions_critic_df = state.get("decisions_critic")
    critic_df = state.get("critic")

    print(
        f"=== dev_inspect_critic for {trading_day} cluster={cluster_id} snapshot_id={snapshot_id} run_id={run_id} ==="
    )
    print(f"windows rows:            {windows_df.height}")
    print(f"trade_paths rows:        {trade_paths_df.height}")
    print(f"decisions_hypotheses:    {decisions_hypo_df.height}")
    print(f"decisions_critic:        {decisions_critic_df.height}")
    print(f"critic summary rows:     {critic_df.height}")

    if not trade_paths_df.is_empty():
        cols = [
            c
            for c in [
                "instrument",
                "trade_id",
                "side",
                "entry_ts",
                "critic_score_at_entry",
                "critic_reason_tags_at_entry",
                "critic_reason_cluster_id",
            ]
            if c in trade_paths_df.columns
        ]
        print("\ntrade_paths sample (with critic fields):")
        print(trade_paths_df.select(cols).head(10))

    if not decisions_critic_df.is_empty():
        cols = [
            c
            for c in [
                "instrument",
                "trade_id",
                "critic_score_at_entry",
                "critic_reason_tags_at_entry",
                "critic_reason_cluster_id",
            ]
            if c in decisions_critic_df.columns
        ]
        print("\ndecisions_critic sample:")
        print(decisions_critic_df.select(cols).head(10))

    if not critic_df.is_empty():
        print("\ncritic run-level summary:")
        print(critic_df.head(10))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="research")
    parser.add_argument("--snapshot-id", default="dev_retail_v1")
    parser.add_argument(
        "--run-id",
        default="dev_inspect_critic_20250102",
        help="run_id for this dev inspect harness",
    )
    parser.add_argument("--trading-day", default="2025-01-02")
    parser.add_argument("--cluster-id", default="metals")
    args = parser.parse_args()

    inspect(
        env=args.env,
        snapshot_id=args.snapshot_id,
        run_id=args.run_id,
        trading_day=args.trading_day,
        cluster_id=args.cluster_id,
    )


if __name__ == "__main__":
    main()
