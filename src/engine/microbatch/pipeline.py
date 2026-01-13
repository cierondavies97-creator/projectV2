from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from engine.bootstrap import bootstrap_engine
from engine.core.ids import MicrobatchKey, RunContext
from engine.microbatch.steps import (
    brackets_step,
    critic_step,
    features_step,
    fills_step,
    gatekeeper_step,
    hypotheses_step,
    ingest_step,
    portfolio_step,
    pretrade_step,
    reports_step,
    windows_step,
)
from engine.microbatch.types import BatchState


class MicrobatchStep(Protocol):
    def run(self, state: BatchState) -> BatchState: ...


def _mode_str(ctx: RunContext) -> str:
    m = ctx.mode
    return m.value if hasattr(m, "value") else str(m)


def _pipeline_steps(ctx: RunContext) -> Sequence[MicrobatchStep]:
    """
    Deterministic microbatch pipeline.

    Ordering is a contract:
      ingest -> features -> windows -> hypotheses -> critic -> pretrade
      -> gatekeeper -> portfolio -> brackets -> (fills in backtest) -> reports

    Note: fills_step is backtest SIM execution step and runs only in backtest mode.
    """
    steps: list[MicrobatchStep] = [
        ingest_step,
        features_step,
        windows_step,
        hypotheses_step,
        critic_step,
        pretrade_step,
        gatekeeper_step,
        portfolio_step,
        brackets_step,
    ]

    if _mode_str(ctx) == "backtest":
        steps.append(fills_step)

    steps.append(reports_step)
    return tuple(steps)


def run_microbatch(ctx: RunContext, key: MicrobatchKey) -> BatchState:
    bootstrap_engine()

    state = BatchState(ctx=ctx, key=key)

    # Minimal observability hook (safe for research/backtest; no config mutation).
    # If you already have structured logging elsewhere, you can remove this.
    # step_names = [type(s).__name__ if hasattr(s, "__class__") else str(s) for s in _pipeline_steps(ctx)]
    # print(f"microbatch: run_id={ctx.run_id} mode={_mode_str(ctx)} day={key.trading_day} cluster={key.cluster_id} steps={step_names}")

    for step in _pipeline_steps(ctx):
        state = step.run(state)

    return state

