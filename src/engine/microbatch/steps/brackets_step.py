from __future__ import annotations

import logging

import polars as pl

from engine.data.brackets import write_brackets_for_instrument_day
from engine.microbatch.steps.contract_guard import ContractWrite, assert_contract_alignment
from engine.microbatch.types import BatchState

log = logging.getLogger(__name__)

# Phase-B evaluation identity (stable join / grouping surface)
_EVAL_KEYS = ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]
_SENTINEL = "∅"
def _maybe_col(
    df: pl.DataFrame,
    name: str,
    *,
    dtype: pl.DataType | None = None,
    default=None,
) -> pl.Expr:
    """
    Build a select() expression robust to missing columns.
    """
    if name in df.columns:
        expr = pl.col(name)
        return expr.cast(dtype) if dtype is not None else expr
    expr = pl.lit(default)
    return expr.cast(dtype) if dtype is not None else expr


def _as_ts(df: pl.DataFrame, name: str) -> pl.Expr:
    # Keep time unit consistent with the rest of the engine (us).
    return _maybe_col(df, name, dtype=pl.Datetime(time_unit="us"), default=None)


def _is_eval_mode(df: pl.DataFrame | None) -> bool:
    return df is not None and (not df.is_empty()) and ("paradigm_id" in df.columns)


def _normalize_eval_identity(df: pl.DataFrame | None, *, name: str) -> pl.DataFrame:
    """
    Normalize evaluation identity columns for Phase B.

    - If paradigm_id exists, principle_id must exist.
    - candidate_id/experiment_id always exist, Utf8.
    - null OR blank -> sentinel "" (partition-stable)
    - IMPORTANT: use pl.lit(...) for sentinels (avoid Polars treating strings as column names).
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = df
    if "paradigm_id" not in out.columns:
        return out

    if "principle_id" not in out.columns:
        raise ValueError(f"brackets_step: {name} has paradigm_id but missing principle_id")

    if "candidate_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("candidate_id"))
    if "experiment_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("experiment_id"))

    for c in _EVAL_KEYS:
        out = out.with_columns(pl.col(c).cast(pl.Utf8))

    out = out.with_columns(
        [
            pl.when(pl.col("candidate_id").is_null() | (pl.col("candidate_id").str.strip_chars() == ""))
            .then(pl.lit(_SENTINEL))
            .otherwise(pl.col("candidate_id"))
            .alias("candidate_id"),
            pl.when(pl.col("experiment_id").is_null() | (pl.col("experiment_id").str.strip_chars() == ""))
            .then(pl.lit(_SENTINEL))
            .otherwise(pl.col("experiment_id"))
            .alias("experiment_id"),
        ]
    )
    return out


def _log_eval_counts(prefix: str, brackets: pl.DataFrame) -> None:
    if brackets is None or brackets.is_empty():
        log.info("%s: n=0", prefix)
        return
    if "paradigm_id" not in brackets.columns:
        log.info("%s: n=%d", prefix, brackets.height)
        return

    counts = brackets.group_by(_EVAL_KEYS, maintain_order=True).agg(pl.len().alias("n"))
    log.info("%s: groups=%d sample=%s", prefix, counts.height, counts.head(10).to_dicts())


def _run_brackets(
    *,
    state: BatchState,
    decisions_portfolio: pl.DataFrame | None,
    trade_paths: pl.DataFrame | None,
) -> pl.DataFrame:
    """
    Brackets step (Phase B compatible DEV STUB but schema-complete for fills simulation).

    Prefer trade_paths as the source of entry/TP/SL fields because fills_step requires:
      - entry_ts, entry_px, sl_px, tp_px

    Phase B:
      - Preserve evaluation identity columns when present so fills/orders can remain comparable.
      - Do not invent paradigm_id/principle_id from ctx (multi-evaluation microbatch).
    """
    tp = trade_paths if (trade_paths is not None and not trade_paths.is_empty()) else None
    dp = decisions_portfolio if (decisions_portfolio is not None and not decisions_portfolio.is_empty()) else None

    src = tp if tp is not None else dp
    if src is None:
        return pl.DataFrame()

    # Normalize eval identity on the source if present
    src_norm = _normalize_eval_identity(src, name="trade_paths" if src is tp else "decisions_portfolio") if _is_eval_mode(src) else src

    # Build bracket frame; include eval identity if present on source (Phase B)
    select_exprs: list[pl.Expr] = [
        pl.lit(state.ctx.snapshot_id).cast(pl.Utf8).alias("snapshot_id"),
        pl.lit(state.ctx.run_id).cast(pl.Utf8).alias("run_id"),
        pl.lit(state.ctx.mode).cast(pl.Utf8).alias("mode"),
        pl.lit(state.key.trading_day).cast(pl.Date).alias("dt"),

        _maybe_col(src_norm, "trade_id", dtype=pl.Utf8, default=None).alias("trade_id"),
        _maybe_col(src_norm, "instrument", dtype=pl.Utf8, default=None).alias("instrument"),
        _maybe_col(src_norm, "side", dtype=pl.Utf8, default=None).alias("side"),

        # Required by fills_step:
        _as_ts(src_norm, "entry_ts").alias("entry_ts"),
        _maybe_col(src_norm, "entry_px", dtype=pl.Float64, default=None).alias("entry_px"),
        _maybe_col(src_norm, "sl_px", dtype=pl.Float64, default=None).alias("sl_px"),
        _maybe_col(src_norm, "tp_px", dtype=pl.Float64, default=None).alias("tp_px"),

        pl.lit("dev_stub").cast(pl.Utf8).alias("order_type"),
    ]

    # Phase B evaluation identity passthrough (stable defaults)
    if "paradigm_id" in src_norm.columns:
        select_exprs.extend(
            [
                _maybe_col(src_norm, "paradigm_id", dtype=pl.Utf8, default=None).alias("paradigm_id"),
                _maybe_col(src_norm, "principle_id", dtype=pl.Utf8, default=None).alias("principle_id"),
                _maybe_col(src_norm, "candidate_id", dtype=pl.Utf8, default=_SENTINEL).alias("candidate_id"),
                _maybe_col(src_norm, "experiment_id", dtype=pl.Utf8, default=_SENTINEL).alias("experiment_id"),
            ]
        )

    brackets = src_norm.select(select_exprs)

    # Re-normalize eval identity on the output (ensures candidate_id/experiment_id -> "")
    if "paradigm_id" in brackets.columns:
        brackets = _normalize_eval_identity(brackets, name="brackets")

    log.info(
        "brackets_step: built %d brackets snapshot_id=%s run_id=%s",
        brackets.height,
        state.ctx.snapshot_id,
        state.ctx.run_id,
    )
    _log_eval_counts("brackets_step: eval breakdown", brackets)

    return brackets


def run(state: BatchState) -> BatchState:
    """
    Inputs:
      - 'decisions_portfolio' (optional)
      - 'trade_paths' (optional; preferred)

    Outputs:
      - 'brackets' (and persists to data/brackets/...)
    """
    assert_contract_alignment(
        step_name="brackets_step",
        writes=(ContractWrite(table_key="brackets", writer_fn="write_brackets_for_instrument_day"),),
    )
    decisions_portfolio = state.get_optional("decisions_portfolio")
    trade_paths = state.get_optional("trade_paths")

    brackets_df = _run_brackets(
        state=state,
        decisions_portfolio=decisions_portfolio,
        trade_paths=trade_paths,
    )

    state.set("brackets", brackets_df)

    # Persist per-instrument; writer will further split by evaluation identity when present
    if brackets_df is not None and not brackets_df.is_empty():
        for g in brackets_df.partition_by("instrument", maintain_order=True):
            instrument = str(g["instrument"][0])
            write_brackets_for_instrument_day(
                ctx=state.ctx,
                trading_day=state.key.trading_day,
                instrument=instrument,
                brackets_df=g,
                sandbox=False,
            )

    return state
