from __future__ import annotations

import logging
from typing import Iterable

import polars as pl

from engine.data.run_reports import write_run_reports_for_cluster_day
from engine.microbatch.steps.contract_guard import ContractWrite, assert_contract_alignment
from engine.microbatch.types import BatchState

log = logging.getLogger(__name__)

# Phase-B evaluation identity (stable join / grouping surface)
_EVAL_KEYS = ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]
_SENTINEL = "∅"


def _n(df: pl.DataFrame | None) -> int:
    return 0 if df is None or df.is_empty() else int(df.height)


def _ensure_eval_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure all eval identity columns exist as Utf8 and are null-safe.

    Note:
      - We do NOT enforce that paradigm_id/principle_id are present; we fill missing with sentinel.
      - This keeps Phase A backward compatible while making Phase B grouping stable.
    """
    out = df

    for c in _EVAL_KEYS:
        if c not in out.columns:
            out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias(c))

    # Cast and fill
    out = out.with_columns([pl.col(c).cast(pl.Utf8, strict=False).fill_null(_SENTINEL) for c in _EVAL_KEYS])
    return out


def _build_eval_map(
    *,
    brackets: pl.DataFrame | None,
    trade_paths: pl.DataFrame | None,
    critic: pl.DataFrame | None,
) -> pl.DataFrame:
    """
    Build an (instrument, trade_id) -> eval identity mapping.

    Priority:
      1) brackets (closest to final trade intent)
      2) trade_paths
      3) critic
    """
    candidates: list[pl.DataFrame] = []
    for df in (brackets, trade_paths, critic):
        if df is None or df.is_empty():
            continue
        if not {"instrument", "trade_id"}.issubset(set(df.columns)):
            continue

        # If any eval key is missing, we still create them (filled with sentinel)
        m = _ensure_eval_cols(df).select(["instrument", "trade_id", *_EVAL_KEYS]).unique(
            subset=["instrument", "trade_id", *_EVAL_KEYS],
            maintain_order=True,
        )
        if not m.is_empty():
            candidates.append(m)

    if not candidates:
        # Empty map
        return pl.DataFrame(schema={"instrument": pl.Utf8, "trade_id": pl.Utf8, **{k: pl.Utf8 for k in _EVAL_KEYS}})

    # First non-empty is authoritative; later sources are ignored by design.
    return candidates[0]


def _attach_eval_identity(df: pl.DataFrame, eval_map: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure df has eval keys populated. If df already has them, we coalesce nulls;
    otherwise we join them from eval_map on (instrument, trade_id).

    If df lacks instrument/trade_id, we fall back to sentinel eval keys.
    """
    out = df
    if out.is_empty():
        return _ensure_eval_cols(out)

    has_keys = {"instrument", "trade_id"}.issubset(set(out.columns))
    if not has_keys or eval_map.is_empty():
        return _ensure_eval_cols(out)

    # Ensure base has eval cols to coalesce into
    out = _ensure_eval_cols(out)

    joined = out.join(
        eval_map.rename({k: f"{k}_map" for k in _EVAL_KEYS}),
        on=["instrument", "trade_id"],
        how="left",
    )

    # Coalesce: prefer df's existing value unless it's sentinel, then use map; finally sentinel.
    # (This is conservative and avoids overwriting explicit values.)
    coalesced_exprs: list[pl.Expr] = []
    for k in _EVAL_KEYS:
        km = f"{k}_map"
        if km in joined.columns:
            coalesced_exprs.append(
                pl.when(pl.col(k).is_null() | (pl.col(k) == _SENTINEL))
                .then(pl.col(km).fill_null(_SENTINEL))
                .otherwise(pl.col(k))
                .alias(k)
            )
        else:
            coalesced_exprs.append(pl.col(k).fill_null(_SENTINEL).alias(k))

    drop_cols = [c for c in joined.columns if c.endswith("_map")]
    joined = joined.with_columns(coalesced_exprs).drop(drop_cols)

    # Final hygiene
    joined = joined.with_columns([pl.col(k).cast(pl.Utf8, strict=False).fill_null(_SENTINEL) for k in _EVAL_KEYS])
    return joined


def _count_by_eval(df: pl.DataFrame | None, *, label: str, eval_map: pl.DataFrame) -> pl.DataFrame:
    """
    Produce a frame: eval_keys + {label} count, grouped by eval identity.
    If df is empty -> empty frame (caller will outer-join and fill 0).
    """
    if df is None or df.is_empty():
        return pl.DataFrame(schema={**{k: pl.Utf8 for k in _EVAL_KEYS}, label: pl.Int64})

    x = _attach_eval_identity(df, eval_map)

    if x.is_empty():
        return pl.DataFrame(schema={**{k: pl.Utf8 for k in _EVAL_KEYS}, label: pl.Int64})

    return (
        x.group_by(_EVAL_KEYS, maintain_order=True)
        .agg(pl.len().cast(pl.Int64).alias(label))
        .with_columns([pl.col(k).cast(pl.Utf8).fill_null(_SENTINEL) for k in _EVAL_KEYS])
    )


def _outer_join_counts(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    """
    Outer-join a list of eval-keyed count frames into one eval summary.
    Missing counts are left as null (caller fills with 0).
    """
    frames = [f for f in frames if f is not None and not f.is_empty()]
    if not frames:
        # Default single sentinel group
        return pl.DataFrame({k: [_SENTINEL] for k in _EVAL_KEYS})

    base = frames[0].select(_EVAL_KEYS).unique(maintain_order=True)
    for f in frames[1:]:
        base = pl.concat(
            [base, f.select(_EVAL_KEYS)],
            how="diagonal",
        ).unique(maintain_order=True)

    out = base
    for f in frames:
        out = out.join(f, on=_EVAL_KEYS, how="left")

    return out


def _run_reports(
    *,
    state: BatchState,
    windows: pl.DataFrame | None,
    trade_paths: pl.DataFrame | None,
    critic: pl.DataFrame | None,
    brackets: pl.DataFrame | None,
    decisions_hypotheses: pl.DataFrame | None,
    decisions_critic: pl.DataFrame | None,
    decisions_pretrade: pl.DataFrame | None,
    decisions_gatekeeper: pl.DataFrame | None,
    decisions_portfolio: pl.DataFrame | None,
) -> pl.DataFrame:
    """
    Reports logic (Phase B compatible):

    - Always emits:
        1) run_total row (single)
        2) evaluation rows grouped by eval identity (when resolvable)
    - Output is still counts-only (for now), but structurally ready for Phase B.
    """
    # Build eval map for decisions identity propagation
    eval_map = _build_eval_map(brackets=brackets, trade_paths=trade_paths, critic=critic)

    # Total counts (run_total)
    n_windows_total = _n(windows)
    n_trades_total = _n(trade_paths)
    n_brackets_total = _n(brackets)
    n_critic_rows_total = _n(critic)
    n_hypo_total = _n(decisions_hypotheses)
    n_critic_dec_total = _n(decisions_critic)
    n_pretrade_total = _n(decisions_pretrade)
    n_gate_total = _n(decisions_gatekeeper)
    n_port_total = _n(decisions_portfolio)

    # Per-eval counts
    c_trades = _count_by_eval(trade_paths, label="n_trades", eval_map=eval_map)
    c_brackets = _count_by_eval(brackets, label="n_brackets", eval_map=eval_map)
    c_critic_rows = _count_by_eval(critic, label="n_critic_rows", eval_map=eval_map)

    c_hypo = _count_by_eval(decisions_hypotheses, label="n_decisions_hypotheses", eval_map=eval_map)
    c_critic_dec = _count_by_eval(decisions_critic, label="n_decisions_critic", eval_map=eval_map)
    c_pretrade = _count_by_eval(decisions_pretrade, label="n_decisions_pretrade", eval_map=eval_map)
    c_gate = _count_by_eval(decisions_gatekeeper, label="n_decisions_gatekeeper", eval_map=eval_map)
    c_port = _count_by_eval(decisions_portfolio, label="n_decisions_portfolio", eval_map=eval_map)

    eval_counts = _outer_join_counts(
        [
            c_trades,
            c_brackets,
            c_critic_rows,
            c_hypo,
            c_critic_dec,
            c_pretrade,
            c_gate,
            c_port,
        ]
    )

    # Fill missing count columns with 0
    for c in [
        "n_trades",
        "n_brackets",
        "n_critic_rows",
        "n_decisions_hypotheses",
        "n_decisions_critic",
        "n_decisions_pretrade",
        "n_decisions_gatekeeper",
        "n_decisions_portfolio",
    ]:
        if c not in eval_counts.columns:
            eval_counts = eval_counts.with_columns(pl.lit(0).cast(pl.Int64).alias(c))
        else:
            eval_counts = eval_counts.with_columns(pl.col(c).fill_null(0).cast(pl.Int64).alias(c))

    n_eval_groups = int(eval_counts.height) if eval_counts is not None and not eval_counts.is_empty() else 0

    # Add run identity columns + windows total replicated (windows are market-state, not strategy-specific)
    eval_rows = (
        eval_counts.with_columns(
            [
                pl.lit(state.ctx.snapshot_id).alias("snapshot_id"),
                pl.lit(state.ctx.run_id).alias("run_id"),
                pl.lit(state.ctx.mode).alias("mode"),
                pl.lit(state.key.trading_day).alias("trading_day"),
                pl.lit(state.key.cluster_id).alias("cluster_id"),
                pl.lit("evaluation").alias("report_level"),
                pl.lit(n_windows_total).cast(pl.Int64).alias("n_windows"),
                pl.lit(n_eval_groups).cast(pl.Int64).alias("n_eval_groups"),
            ]
        )
        .select(
            [
                "snapshot_id",
                "run_id",
                "mode",
                "trading_day",
                "cluster_id",
                "report_level",
                *_EVAL_KEYS,
                "n_eval_groups",
                "n_windows",
                "n_trades",
                "n_decisions_hypotheses",
                "n_decisions_critic",
                "n_decisions_pretrade",
                "n_decisions_gatekeeper",
                "n_decisions_portfolio",
                "n_brackets",
                "n_critic_rows",
            ]
        )
        .sort(_EVAL_KEYS)
    )

    run_total_row = pl.DataFrame(
        {
            "snapshot_id": [state.ctx.snapshot_id],
            "run_id": [state.ctx.run_id],
            "mode": [state.ctx.mode],
            "trading_day": [state.key.trading_day],
            "cluster_id": [state.key.cluster_id],
            "report_level": ["run_total"],
            "paradigm_id": [_SENTINEL],
            "principle_id": [_SENTINEL],
            "candidate_id": [_SENTINEL],
            "experiment_id": [_SENTINEL],
            "n_eval_groups": [n_eval_groups],
            "n_windows": [n_windows_total],
            "n_trades": [n_trades_total],
            "n_decisions_hypotheses": [n_hypo_total],
            "n_decisions_critic": [n_critic_dec_total],
            "n_decisions_pretrade": [n_pretrade_total],
            "n_decisions_gatekeeper": [n_gate_total],
            "n_decisions_portfolio": [n_port_total],
            "n_brackets": [n_brackets_total],
            "n_critic_rows": [n_critic_rows_total],
        }
    )

    reports_df = pl.concat([run_total_row, eval_rows], how="diagonal") if not eval_rows.is_empty() else run_total_row

    log.info(
        "reports_step: snapshot_id=%s run_id=%s dt=%s cluster=%s "
        "windows=%d trades=%d hypo=%d critic_dec=%d pretrade=%d gate=%d "
        "portfolio=%d brackets=%d critic_rows=%d eval_groups=%d",
        state.ctx.snapshot_id,
        state.ctx.run_id,
        state.key.trading_day,
        state.key.cluster_id,
        n_windows_total,
        n_trades_total,
        n_hypo_total,
        n_critic_dec_total,
        n_pretrade_total,
        n_gate_total,
        n_port_total,
        n_brackets_total,
        n_critic_rows_total,
        n_eval_groups,
    )

    return reports_df


def run(state: BatchState) -> BatchState:
    """
    Reports step (Phase B compatible counts).

    Inputs (all optional):
      - 'windows'
      - 'trade_paths'
      - 'critic'
      - 'brackets'
      - 'decisions_hypotheses'
      - 'decisions_critic'
      - 'decisions_pretrade'
      - 'decisions_gatekeeper'
      - 'decisions_portfolio'

    Outputs:
      - 'reports'
    """
    assert_contract_alignment(
        step_name="reports_step",
        writes=(ContractWrite(table_key="reports", writer_fn="write_run_reports_for_cluster_day"),),
    )
    windows = state.get_optional("windows")
    trade_paths = state.get_optional("trade_paths")
    critic = state.get_optional("critic")
    brackets = state.get_optional("brackets")
    decisions_hypotheses = state.get_optional("decisions_hypotheses")
    decisions_critic = state.get_optional("decisions_critic")
    decisions_pretrade = state.get_optional("decisions_pretrade")
    decisions_gatekeeper = state.get_optional("decisions_gatekeeper")
    decisions_portfolio = state.get_optional("decisions_portfolio")

    reports_df = _run_reports(
        state=state,
        windows=windows,
        trade_paths=trade_paths,
        critic=critic,
        brackets=brackets,
        decisions_hypotheses=decisions_hypotheses,
        decisions_critic=decisions_critic,
        decisions_pretrade=decisions_pretrade,
        decisions_gatekeeper=decisions_gatekeeper,
        decisions_portfolio=decisions_portfolio,
    )

    state.set("reports", reports_df)

    # Persist run_reports (canonical table)
    td = state.key.trading_day
    write_run_reports_for_cluster_day(
        ctx=state.ctx,
        trading_day=state.key.trading_day,
        cluster_id=state.key.cluster_id,
        reports_df=reports_df,
        sandbox=False,
    )

    return state
