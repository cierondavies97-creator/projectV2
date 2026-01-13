from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

# --- bootstrap so Python can find src/engine when running from scripts/ ---
ROOT = Path(__file__).resolve().parents[1]  # project root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# -------------------------------------------------------------------------


from engine.core.api import MicrobatchKey, RunContext
from engine.microbatch.api import BatchState
from engine.microbatch.steps import features_step, ingest_step


def main() -> None:
    # Match the microbatch youÃ¢â‚¬â„¢ve been running
    ctx = RunContext(
        env="research",
        mode="backtest",
        snapshot_id="dev_retail_v1",
        run_id="dev_smoke_001",
        experiment_id=None,
        candidate_id=None,
        base_seed=0,
    )

    key = MicrobatchKey(
        trading_day=date(2025, 1, 2),
        cluster_id="metals",
    )

    state = BatchState(ctx=ctx, key=key)

    # Run just ingest + features (no windows / hypotheses / etc.)
    state = ingest_step.run(state)
    state = features_step.run(state)

    features = state.get("features")

    if features is None or features.is_empty():
        print("features table is empty")
        return

    print("=== Full features schema ===")
    print(features.schema)
    print()

    print("=== Sample of combined features (first 5 rows) ===")
    print(features.head(5))
    print()

    # ICT struct slice
    ict_cols = [c for c in features.columns if c.startswith("ict_struct_")]
    print("=== ICT struct columns (head) ===")
    if ict_cols:
        print(features.select(["instrument", "anchor_tf", "ts"] + ict_cols).head(10))
    else:
        print("No ict_struct_* columns present")
    print()

    # stat_ts slice
    stat_cols = [c for c in features.columns if c.startswith("stat_ts_")]
    print("=== stat_ts columns (head) ===")
    if stat_cols:
        print(features.select(["instrument", "anchor_tf", "ts"] + stat_cols).head(10))
    else:
        print("No stat_ts_* columns present")

    # ICT swings where something actually fires
    swings = features.filter((pl.col("ict_struct_swing_high") == 1.0) | (pl.col("ict_struct_swing_low") == 1.0))
    print("=== ICT swings (first 20 rows where swing_high or swing_low == 1) ===")
    print(
        swings.select(
            "instrument",
            "anchor_tf",
            "ts",
            "ict_struct_swing_high",
            "ict_struct_swing_low",
            "ict_struct_pd_index",
            "ict_struct_liquidity_tag",
        ).head(20)
    )


if __name__ == "__main__":
    main()
