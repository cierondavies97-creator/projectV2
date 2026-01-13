from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from engine.core.api import RunContext
from engine.io.api import read_parquet_dir, windows_dir


def _safe_read(dir_path: Path) -> pl.DataFrame:
    if not dir_path.exists():
        return pl.DataFrame()
    return read_parquet_dir(dir_path)


def inspect(
    ctx: RunContext,
    trading_day_str: str,
    instrument: str,
    anchor_tf: str,
) -> None:
    trading_day = datetime.strptime(trading_day_str, "%Y-%m-%d").date()

    win_dir = windows_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        anchor_tf=anchor_tf,
        sandbox=False,
    )
    windows_df = _safe_read(win_dir)

    print(f"=== dev_inspect_windows for {instrument} {trading_day_str} run_id={ctx.run_id} anchor_tf={anchor_tf} ===")
    print(f"windows rows: {windows_df.height}")

    if windows_df.is_empty():
        return

    # Show key columns + a few context fields if present
    base_cols = [
        "instrument",
        "anchor_tf",
        "anchor_ts",
        "tf_entry",
    ]
    context_cols = [
        "tod_bucket",
        "dow_bucket",
        "vol_regime",
        "trend_regime",
        "macro_state",
        "micro_corr_regime",
        "corr_cluster_id",
        "zone_behaviour_type_bucket",
        "zone_freshness_bucket",
        "zone_stack_depth_bucket",
        "zone_htf_confluence_bucket",
        "zone_vp_type_bucket",
        "unsup_regime_id",
        "entry_profile_id",
        "management_profile_id",
    ]

    cols = [c for c in base_cols + context_cols if c in windows_df.columns]

    print("\nwindows sample:")
    print(windows_df.select(cols).head(10))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="research")
    parser.add_argument("--snapshot-id", default="dev_retail_v1")
    parser.add_argument(
        "--run-id",
        default="dev_inspect_windows_20250102",
        help="run_id for this dev inspect harness",
    )
    parser.add_argument("--trading-day", default="2025-01-02")
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--anchor-tf", default="M5")
    args = parser.parse_args()

    ctx = RunContext(
        env=args.env,
        mode="backtest",
        snapshot_id=args.snapshot_id,
        run_id=args.run_id,
        base_seed=0,
    )

    inspect(
        ctx=ctx,
        trading_day_str=args.trading_day,
        instrument=args.instrument,
        anchor_tf=args.anchor_tf,
    )


if __name__ == "__main__":
    main()
