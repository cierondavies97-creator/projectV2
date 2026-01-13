# scripts/dev_inspect_hypotheses.py
from __future__ import annotations

import logging
from datetime import date

from engine.core.api import EnvId, MicrobatchKey, Mode, RunContext
from engine.microbatch.api import BatchState
from engine.microbatch.steps import (
    critic_step,
    features_step,
    hypotheses_step,
    ingest_step,
    windows_step,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    # Match the dev setup youÃ¢â‚¬â„¢ve been using elsewhere
    env: EnvId = "research"
    mode: Mode = "backtest"

    ctx = RunContext(
        env=env,
        mode=mode,
        snapshot_id="dev_retail_v1",
        run_id="dev_inspect_hypotheses",
        experiment_id=None,
        candidate_id=None,
        base_seed=0,
    )

    key = MicrobatchKey(
        trading_day=date(2025, 1, 2),
        cluster_id="metals",
    )

    state = BatchState(ctx=ctx, key=key)

    # 1) Ingest raw inputs (candles/ticks/macro/external)
    state = ingest_step.run(state)

    # 2) Build features (ict_struct, stat_ts, macro stubs, etc.)
    state = features_step.run(state)

    # 3) Build windows from features
    state = windows_step.run(state)

    # 4) Run hypotheses step (currently DEV STUB in your repo)
    state = hypotheses_step.run(state)

    # 5) Run critic step (dev scoring)
    state = critic_step.run(state)

    windows = state.get("windows")
    decisions_h = state.tables.get("decisions_hypotheses")
    trade_paths = state.tables.get("trade_paths")

    print("=== windows ===")
    print("rows:", windows.height)
    print("schema:", windows.schema, "\n")
    print(windows.head(10))

    if decisions_h is None:
        print("\n=== decisions_hypotheses ===")
        print("No 'decisions_hypotheses' table present in state.")
    else:
        print("\n=== decisions_hypotheses ===")
        print("rows:", decisions_h.height)
        print("schema:", decisions_h.schema)
        print(decisions_h.head(10))

    if trade_paths is None:
        print("\n=== trade_paths ===")
        print("No 'trade_paths' table present in state.")
    else:
        print("\n=== trade_paths ===")
        print("rows:", trade_paths.height)
        print("schema:", trade_paths.schema)
        print(trade_paths.head(10))
    decisions_critic = state.tables.get("decisions_critic")
    critic = state.tables.get("critic")

    if decisions_critic is None:
        print("\n=== decisions_critic ===")
        print("No 'decisions_critic' table present in state.")
    else:
        print("\n=== decisions_critic ===")
        print("rows:", decisions_critic.height)
        print("schema:", decisions_critic.schema)
        print(decisions_critic.head(10))

    if critic is None:
        print("\n=== critic ===")
        print("No 'critic' table present in state.")
    else:
        print("\n=== critic ===")
        print("rows:", critic.height)
        print("schema:", critic.schema)
        print(critic)


if __name__ == "__main__":
    main()
