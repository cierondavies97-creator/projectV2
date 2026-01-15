from __future__ import annotations

import logging
from typing import Optional

import polars as pl

from engine.data.principles_context import build_principles_context_from_trade_paths, write_principles_context_for_cluster_day
from engine.data.decisions import write_decisions_for_stage
from engine.microbatch.steps.contract_guard import ContractWrite, assert_contract_alignment
from engine.microbatch.types import BatchState

log = logging.getLogger(__name__)

_EVAL_KEYS = ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]


def _is_eval_mode(df: pl.DataFrame | None) -> bool:
    return df is not None and (not df.is_empty()) and ("paradigm_id" in df.columns)


def _normalize_eval_identity(df: pl.DataFrame | None, *, name: str) -> pl.DataFrame:
    """
    Phase B invariant:
      If paradigm_id exists, principle_id must exist.

    Normalization:
      - ensure candidate_id/experiment_id exist
      - cast eval columns to Utf8
      - fill nulls with sentinel "∅" (partition-stable, join-safe)
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = df
    if "paradigm_id" not in out.columns:
        return out

    if "principle_id" not in out.columns:
        raise ValueError(f"gatekeeper_step: {name} has paradigm_id but missing principle_id (Phase B requires both).")

    if "candidate_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("candidate_id"))
    if "experiment_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("experiment_id"))

    for c in _EVAL_KEYS:
        out = out.with_columns(pl.col(c).cast(pl.Utf8))

    out = out.with_columns(
        [
            pl.col("candidate_id").fill_null("∅"),
            pl.col("experiment_id").fill_null("∅"),
        ]
    )
    return out


def _log_eval_counts(prefix: str, df_in: pl.DataFrame, df_out: pl.DataFrame) -> None:
    """
    Log compact counts by evaluation identity when present, else log totals.
    """
    if df_in.is_empty() and df_out.is_empty():
        log.info("%s: n_in=0 n_out=0", prefix)
        return

    if "paradigm_id" not in df_in.columns and "paradigm_id" not in df_out.columns:
        log.info("%s: n_in=%d n_out=%d", prefix, df_in.height, df_out.height)
        return

    keys = _EVAL_KEYS
    in_counts = df_in.group_by(keys, maintain_order=True).agg(pl.len().alias("n_in")) if not df_in.is_empty() else pl.DataFrame()
    out_counts = df_out.group_by(keys, maintain_order=True).agg(pl.len().alias("n_out")) if not df_out.is_empty() else pl.DataFrame()

    if not in_counts.is_empty() and not out_counts.is_empty():
        merged = in_counts.join(out_counts, on=keys, how="outer")
    elif not in_counts.is_empty():
        merged = in_counts.with_columns(pl.lit(0).cast(pl.Int64).alias("n_out"))
    else:
        merged = out_counts.with_columns(pl.lit(0).cast(pl.Int64).alias("n_in"))

    merged = merged.with_columns(
        [
            pl.col("n_in").fill_null(0),
            pl.col("n_out").fill_null(0),
        ]
    )

    head = merged.head(10)
    log.info("%s: groups=%d sample=%s", prefix, merged.height, head.to_dicts())


def _run_gatekeeper(
    *,
    state: BatchState,
    decisions_pretrade: pl.DataFrame | None,
    principles_context: pl.DataFrame | None,
    trade_paths: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Gatekeeper step (Phase B compatible DEV stub).

    Behavior (unchanged intent):
      - Pass through all pretrade decisions
      - Stamp 'gatekeeper_status' = 'allowed_dev_stub'
      - Do NOT change principles_context
      - Do NOT change trade_paths

    Phase B additions:
      - Normalize evaluation identity columns when present.
      - Ensure candidate_id/experiment_id are stable ("∅") in eval-mode.
    """
    dp = _normalize_eval_identity(decisions_pretrade, name="decisions_pretrade")
    pc = _normalize_eval_identity(principles_context, name="principles_context") if _is_eval_mode(principles_context) else (principles_context or pl.DataFrame())
    tp = _normalize_eval_identity(trade_paths, name="trade_paths") if _is_eval_mode(trade_paths) else (trade_paths or pl.DataFrame())

    if dp.is_empty():
        dg = dp
    else:
        # Row-wise stamp; evaluation identity preserved automatically.
        dg = dp.with_columns(pl.lit("allowed_dev_stub").cast(pl.Utf8).alias("gatekeeper_status"))

    _log_eval_counts(
        f"gatekeeper_step: snapshot_id={state.ctx.snapshot_id} run_id={state.ctx.run_id}",
        dp,
        dg,
    )

    return dg, pc, tp


def run(state: BatchState) -> BatchState:
    """
    Gatekeeper step (Phase B compatible DEV stub).

    Inputs:
      - 'decisions_pretrade'
      - 'principles_context' (optional)
      - 'trade_paths'

    Outputs:
      - 'decisions_gatekeeper'
      - updated 'principles_context'
      - updated 'trade_paths'
    """
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
    decisions_pretrade = state.get_optional("decisions_pretrade")
    principles_context = state.get_optional("principles_context")
    trade_paths = state.get_optional("trade_paths")

    decisions_gatekeeper_df, principles_context_df, trade_paths_updated_df = _run_gatekeeper(
        state=state,
        decisions_pretrade=decisions_pretrade,
        principles_context=principles_context,
        trade_paths=trade_paths,
    )

    state.set("decisions_gatekeeper", decisions_gatekeeper_df)
    state.set("principles_context", principles_context_df)
    state.set("trade_paths", trade_paths_updated_df)

    # Persist gatekeeper decisions for this run (writer is Phase-B aware)
    if decisions_gatekeeper_df is not None and not decisions_gatekeeper_df.is_empty():
        write_decisions_for_stage(
            ctx=state.ctx,
            trading_day=state.key.trading_day,
            stage="gatekeeper",
            decisions_df=decisions_gatekeeper_df,
        )

    # Persist principles_context here (contract owner boundary).

    td = state.key.trading_day

    trade_paths = state.get("trade_paths")

    principles_context = build_principles_context_from_trade_paths(

        trade_paths,

        cluster_id=state.key.cluster_id,

        trading_day=td,

    )

    state.set("principles_context", principles_context)

    write_principles_context_for_cluster_day(

        ctx=state.ctx,

        df=principles_context,

        cluster_id=state.key.cluster_id,

        trading_day=td,

        sandbox=False,

    )

    return state
