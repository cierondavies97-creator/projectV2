from __future__ import annotations

import logging

import polars as pl

from engine.data.trade_clusters import build_trade_clusters_from_trade_paths, write_trade_clusters_for_cluster_day
from engine.data.decisions import write_decisions_for_stage
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
      - ensure candidate_id / experiment_id exist
      - cast eval columns to Utf8
      - fill nulls with sentinel "∅" (partition-stable, join-safe)
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = df
    if "paradigm_id" not in out.columns:
        return out

    if "principle_id" not in out.columns:
        raise ValueError(f"portfolio_step: {name} has paradigm_id but missing principle_id (Phase B requires both).")

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

    merged = merged.with_columns([pl.col("n_in").fill_null(0), pl.col("n_out").fill_null(0)])
    log.info("%s: groups=%d sample=%s", prefix, merged.height, merged.head(10).to_dicts())


def _run_portfolio(
    *,
    state: BatchState,
    decisions_gatekeeper: pl.DataFrame | None,
    trade_paths: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Portfolio step (Phase B compatible DEV STUB).

    Behavior (unchanged intent):
      - Pass through all gatekeeper decisions
      - Add 'portfolio_risk_bps_at_entry' = 0.0
      - Do NOT change trade_paths

    Phase B additions:
      - Preserve and normalize evaluation identity columns (candidate_id/experiment_id -> "∅")
      - Do not assume a single paradigm from ctx
    """
    dg = _normalize_eval_identity(decisions_gatekeeper, name="decisions_gatekeeper")
    tp = _normalize_eval_identity(trade_paths, name="trade_paths") if _is_eval_mode(trade_paths) else (trade_paths or pl.DataFrame())

    if dg.is_empty():
        dp = dg
    else:
        # Row-wise stamp; evaluation identity preserved automatically.
        dp = dg.with_columns(pl.lit(0.0).cast(pl.Float64).alias("portfolio_risk_bps_at_entry"))

    _log_eval_counts(
        f"portfolio_step: snapshot_id={state.ctx.snapshot_id} run_id={state.ctx.run_id}",
        dg,
        dp,
    )
    return dp, tp


def run(state: BatchState) -> BatchState:
    """
    Portfolio step (Phase B compatible DEV STUB).

    Inputs:
      - 'decisions_gatekeeper'
      - 'trade_paths'

    Outputs:
      - 'decisions_portfolio'
      - updated 'trade_paths'
    """
    decisions_gatekeeper = state.get_optional("decisions_gatekeeper")
    trade_paths = state.get_optional("trade_paths")

    decisions_portfolio_df, trade_paths_updated_df = _run_portfolio(
        state=state,
        decisions_gatekeeper=decisions_gatekeeper,
        trade_paths=trade_paths,
    )

    state.set("decisions_portfolio", decisions_portfolio_df)
    state.set("trade_paths", trade_paths_updated_df)

    if decisions_portfolio_df is not None and not decisions_portfolio_df.is_empty():
        write_decisions_for_stage(
            ctx=state.ctx,
            trading_day=state.key.trading_day,
            stage="portfolio",
            decisions_df=decisions_portfolio_df,
        )

    # Persist trade_clusters here (contract owner boundary).

    td = state.key.trading_day

    trade_paths = state.get("trade_paths")

    trade_clusters = build_trade_clusters_from_trade_paths(

        trade_paths,

        cluster_id=state.key.cluster_id,

        trading_day=td,

    )

    state.set("trade_clusters", trade_clusters)

    write_trade_clusters_for_cluster_day(

        ctx=state.ctx,

        df=trade_clusters,

        cluster_id=state.key.cluster_id,

        trading_day=td,

        sandbox=False,

    )

    return state
