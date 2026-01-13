from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from engine.core.api import RunContext
from engine.io.api import (
    decisions_dir,
    features_dir,
    read_parquet_dir,
    trade_paths_dir,
)


def _safe_read(dir_path: Path) -> pl.DataFrame:
    """
    Read all Parquet files under dir_path into a DataFrame, or return
    an empty frame if the directory does not exist.
    """
    if not dir_path.exists():
        return pl.DataFrame()
    return read_parquet_dir(dir_path)


def inspect(ctx: RunContext, trading_day_str: str, instrument: str) -> None:
    """
    Inspect the Memory Model slice for a given (run_id, trading_day, instrument).

    Shows row counts for:
      - features (anchor_tf=M5 for now)
      - trade_paths
      - decisions_hypotheses / critic / pretrade / gatekeeper / portfolio

    And prints a small sample from trade_paths and critic.
    """
    # Convert "YYYY-MM-DD" into a date
    trading_day = datetime.strptime(trading_day_str, "%Y-%m-%d").date()

    # ------------------------------------------------------------------
    # features (anchor_tf=M5 for now)
    # ------------------------------------------------------------------
    feat_dir = features_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        anchor_tf="M5",
        sandbox=False,
    )
    features_df = _safe_read(feat_dir)

    # ------------------------------------------------------------------
    # trade_paths
    # ------------------------------------------------------------------
    tp_dir = trade_paths_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        sandbox=False,
    )
    trade_paths_df = _safe_read(tp_dir)

    # ------------------------------------------------------------------
    # decisions (all stages)
    # ------------------------------------------------------------------
    decisions: dict[str, pl.DataFrame] = {}
    for stage in ["hypotheses", "critic", "pretrade", "gatekeeper", "portfolio"]:
        d_dir = decisions_dir(
            ctx=ctx,
            trading_day=trading_day,
            instrument=instrument,
            stage=stage,
            sandbox=False,
        )
        decisions[stage] = _safe_read(d_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    dt = trading_day_str
    print(f"=== Memory Model snapshot for {instrument} {dt} run_id={ctx.run_id} ===")
    print(f"features rows:           {features_df.height}")
    print(f"trade_paths rows:        {trade_paths_df.height}")
    for stage, df in decisions.items():
        print(f"decisions_{stage:11s}: {df.height} rows")

    # ------------------------------------------------------------------
    # Samples
    # ------------------------------------------------------------------
    if not trade_paths_df.is_empty():
        print("\ntrade_paths sample:")
        print(trade_paths_df.select(["instrument", "trade_id", "side", "entry_ts"]).head(5))

    critic_df = decisions.get("critic", pl.DataFrame())
    if not critic_df.is_empty():
        print("\ncritic sample:")
        print(critic_df.select(["instrument", "trade_id", "critic_score_at_entry"]).head(5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="research")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trading-day", required=True)  # YYYY-MM-DD
    parser.add_argument("--instrument", default="XAUUSD")
    args = parser.parse_args()

    ctx = RunContext(
        env=args.env,
        mode="backtest",
        snapshot_id=args.snapshot_id,
        run_id=args.run_id,
        base_seed=0,
    )

    inspect(ctx, args.trading_day, args.instrument)


if __name__ == "__main__":
    main()
