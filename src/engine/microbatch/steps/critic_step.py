from __future__ import annotations

import logging
from typing import Any, Optional

import polars as pl

from engine.data.decisions import write_decisions_for_stage
from engine.microbatch.steps.contract_guard import ContractWrite, assert_contract_alignment
from engine.microbatch.types import BatchState

log = logging.getLogger(__name__)

# Phase B evaluation identity (long format)
_EVAL_KEYS = ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]

# Critic output fields (kept compatible with your current schemas: reason tags as string)
_CRITIC_FIELDS = [
    "critic_score_at_entry",
    "critic_reason_tags_at_entry",
    "critic_reason_cluster_id",
]


# ---------------------------------------------------------------------------
# Empty frames (stable schemas)
# ---------------------------------------------------------------------------

def _empty_decisions_critic_frame() -> pl.DataFrame:
    """
    Empty frame for decisions_critic with a stable schema.

    One row per trade_id with critic metadata.
    Includes instrument + evaluation identity so decisions_* tables can be partitioned safely.
    """
    return pl.DataFrame(
        {
            "snapshot_id": pl.Series([], dtype=pl.Utf8),
            "run_id": pl.Series([], dtype=pl.Utf8),
            "mode": pl.Series([], dtype=pl.Utf8),
            "paradigm_id": pl.Series([], dtype=pl.Utf8),
            "principle_id": pl.Series([], dtype=pl.Utf8),
            "candidate_id": pl.Series([], dtype=pl.Utf8),
            "experiment_id": pl.Series([], dtype=pl.Utf8),
            "instrument": pl.Series([], dtype=pl.Utf8),
            "trade_id": pl.Series([], dtype=pl.Utf8),
            "critic_score_at_entry": pl.Series([], dtype=pl.Float64),
            "critic_reason_tags_at_entry": pl.Series([], dtype=pl.Utf8),
            "critic_reason_cluster_id": pl.Series([], dtype=pl.Utf8),
        }
    )


def _empty_critic_frame() -> pl.DataFrame:
    """
    Empty frame for critic diagnostics at run/microbatch level.
    In Phase B we emit per-evaluation diagnostics.
    """
    return pl.DataFrame(
        {
            "snapshot_id": pl.Series([], dtype=pl.Utf8),
            "run_id": pl.Series([], dtype=pl.Utf8),
            "mode": pl.Series([], dtype=pl.Utf8),
            "paradigm_id": pl.Series([], dtype=pl.Utf8),
            "principle_id": pl.Series([], dtype=pl.Utf8),
            "candidate_id": pl.Series([], dtype=pl.Utf8),
            "experiment_id": pl.Series([], dtype=pl.Utf8),
            "n_trades": pl.Series([], dtype=pl.Int64),
            "avg_critic_score": pl.Series([], dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Normalization / invariants
# ---------------------------------------------------------------------------

def _normalize_eval_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    If a dataframe is in eval-mode (has paradigm_id), ensure the full eval identity
    columns exist and are normalized for grouping/joining.

    Policy:
      - If paradigm_id exists, principle_id must exist (Phase B invariant).
      - candidate_id/experiment_id are normalized to the stable sentinel "∅" when null/missing.
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = df

    if "paradigm_id" not in out.columns:
        return out

    if "principle_id" not in out.columns:
        raise ValueError("Phase B invariant violated: dataframe has paradigm_id but missing principle_id.")

    # Ensure candidate_id / experiment_id exist
    if "candidate_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("candidate_id"))
    if "experiment_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("experiment_id"))

    # Normalize identity dtypes and nulls
    for c in ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]:
        out = out.with_columns(pl.col(c).cast(pl.Utf8))

    out = out.with_columns(
        [
            pl.when(pl.col("candidate_id").is_null() | (pl.col("candidate_id").cast(pl.Utf8).str.strip_chars() == "")).then(pl.lit("∅")).otherwise(pl.col("candidate_id")).alias("candidate_id"),
            pl.when(pl.col("experiment_id").is_null() | (pl.col("experiment_id").cast(pl.Utf8).str.strip_chars() == "")).then(pl.lit("∅")).otherwise(pl.col("experiment_id")).alias("experiment_id"),
        ]
    )
    return out


def _ensure_decisions_critic_surface(ctx, decisions_critic: pl.DataFrame, trade_paths_group: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure decisions_critic has the minimum required columns to persist and to join back to trade_paths.

    We do NOT invent paradigm_id/principle_id if absent in eval-mode; those must come from inputs or critic.
    We DO stamp snapshot/run/mode if missing (audit identity).
    """
    if decisions_critic is None or decisions_critic.is_empty():
        return pl.DataFrame()

    out = decisions_critic

    exprs: list[pl.Expr] = []
    if "snapshot_id" not in out.columns:
        exprs.append(pl.lit(ctx.snapshot_id).cast(pl.Utf8).alias("snapshot_id"))
    if "run_id" not in out.columns:
        exprs.append(pl.lit(ctx.run_id).cast(pl.Utf8).alias("run_id"))
    if "mode" not in out.columns:
        m = ctx.mode.value if hasattr(ctx.mode, "value") else str(ctx.mode)
        exprs.append(pl.lit(m).cast(pl.Utf8).alias("mode"))

    # instrument is required for partitioning; if critic omits it, recover from trade_paths
    if "instrument" not in out.columns:
        if "instrument" in trade_paths_group.columns and not trade_paths_group.is_empty():
            # map trade_id -> instrument when possible; otherwise fallback UNKNOWN
            inst_map = trade_paths_group.select(["trade_id", "instrument"]).unique()
            out = out.join(inst_map, on="trade_id", how="left")
        else:
            exprs.append(pl.lit("UNKNOWN").cast(pl.Utf8).alias("instrument"))

    # Fill anchor context from trade_paths when critic omits it.
    anchor_cols = ["anchor_tf", "anchor_ts", "tf_entry"]
    missing_anchor = [c for c in anchor_cols if c not in out.columns]
    if missing_anchor and trade_paths_group is not None and not trade_paths_group.is_empty():
        available = [c for c in anchor_cols if c in trade_paths_group.columns]
        if available:
            anchor_map = trade_paths_group.select(["trade_id", *available]).unique()
            out = out.join(anchor_map, on="trade_id", how="left")

    if exprs:
        out = out.with_columns(exprs)

    # Eval columns: if present, normalize
    out = _normalize_eval_cols(out) if "paradigm_id" in out.columns else out

    # Ensure critic fields exist
    for c in _CRITIC_FIELDS:
        if c not in out.columns:
            if c == "critic_score_at_entry":
                out = out.with_columns(pl.lit(0.0).cast(pl.Float64).alias(c))
            else:
                out = out.with_columns(pl.lit("dev_noop").cast(pl.Utf8).alias(c))

    # Ensure trade_id exists
    if "trade_id" not in out.columns:
        raise ValueError("critic_fn must return a frame containing trade_id.")

    # Normalize common string identity fields
    for c in ["snapshot_id", "run_id", "mode", "instrument", "trade_id"]:
        if c in out.columns:
            out = out.with_columns(pl.col(c).cast(pl.Utf8))

    return out


def _noop_critic_for_group(ctx, trade_paths_group: pl.DataFrame, decisions_hypotheses_group: pl.DataFrame, critic_cfg: dict[str, Any]) -> pl.DataFrame:
    """
    Safe fallback critic for paradigms that are registered without a critic yet (Phase B 'A' mode).

    Produces one row per trade_id with neutral scoring.
    """
    if trade_paths_group is None or trade_paths_group.is_empty():
        return pl.DataFrame()

    base_cols = ["trade_id"]
    for c in ["paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument"]:
        if c in trade_paths_group.columns:
            base_cols.append(c)

    base = trade_paths_group.select(base_cols).unique()

    # If eval-mode, ensure full eval columns exist
    if "paradigm_id" in base.columns:
        base = _normalize_eval_cols(base)

    return base.with_columns(
        [
            pl.lit(0.0).cast(pl.Float64).alias("critic_score_at_entry"),
            pl.lit("dev_noop").cast(pl.Utf8).alias("critic_reason_tags_at_entry"),
            pl.lit("dev_noop").cast(pl.Utf8).alias("critic_reason_cluster_id"),
        ]
    )


def _join_keys_for_trade_paths(trade_paths: pl.DataFrame, decisions_critic: pl.DataFrame) -> list[str]:
    """
    Determine safe join keys to write critic fields back onto trade_paths.

    Phase B: include evaluation identity + instrument when available to prevent collisions.
    """
    keys: list[str] = []
    for c in ["trade_id", "instrument"]:
        if c in trade_paths.columns and c in decisions_critic.columns:
            keys.append(c)

    for c in _EVAL_KEYS:
        if c in trade_paths.columns and c in decisions_critic.columns:
            keys.append(c)

    # Fallback (Phase A): trade_id only
    if not keys and "trade_id" in trade_paths.columns and "trade_id" in decisions_critic.columns:
        keys = ["trade_id"]

    if not keys:
        raise ValueError("Unable to determine join keys for critic -> trade_paths (missing trade_id).")

    return keys


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run(state: BatchState) -> BatchState:
    """
    Critic step (Phase B).

    Inputs:
      - 'trade_paths' (long format, may include multiple evaluations)
      - 'decisions_hypotheses' (long format)

    Outputs:
      - 'decisions_critic' (long format)
      - 'critic' (diagnostics; per evaluation)
      - updated 'trade_paths' (critic fields filled)
    """
    assert_contract_alignment(
        step_name="critic_step",
        writes=(
            ContractWrite(
                table_key="decisions_critic",
                writer_fn="write_decisions_for_stage",
                stage="critic",
            ),
        ),
    )
    trade_paths = state.get_optional("trade_paths")
    if trade_paths is None:
        trade_paths = pl.DataFrame()
    decisions_hypotheses = state.get_optional("decisions_hypotheses")
    if decisions_hypotheses is None:
        decisions_hypotheses = pl.DataFrame()

    # Normalize eval identity if present
    trade_paths = _normalize_eval_cols(trade_paths) if not trade_paths.is_empty() else trade_paths
    decisions_hypotheses = _normalize_eval_cols(decisions_hypotheses) if not decisions_hypotheses.is_empty() else decisions_hypotheses

    if trade_paths.is_empty():
        # No trades -> empty outputs
        state.set("decisions_critic", _empty_decisions_critic_frame())
        state.set("critic", _empty_critic_frame())
        state.set("trade_paths", trade_paths)
        return state

    # Determine whether we're in Phase B eval-mode
    eval_mode = ("paradigm_id" in trade_paths.columns)

    # Grouping keys: Phase B uses full evaluation identity
    group_cols = _EVAL_KEYS if eval_mode else []

    from engine.paradigms.registry import get_critic

    decisions_critic_groups: list[pl.DataFrame] = []

    if group_cols:
        groups = trade_paths.partition_by(group_cols, maintain_order=True)
    else:
        groups = [trade_paths]

    for tp_g in groups:
        # Establish group identity
        if eval_mode:
            pid = str(tp_g["paradigm_id"][0])
            prid = str(tp_g["principle_id"][0])
            cand = str(tp_g["candidate_id"][0]) if "candidate_id" in tp_g.columns else "∅"
            exp = str(tp_g["experiment_id"][0]) if "experiment_id" in tp_g.columns else "∅"
        else:
            pid = getattr(state.ctx, "paradigm_id", "ict")
            prid = getattr(state.ctx, "principle_id", "ict_all_windows")
            cand = "∅"
            exp = "∅"

        # Filter decisions_hypotheses to the same evaluation group when possible
        if eval_mode and not decisions_hypotheses.is_empty():
            dh_g = decisions_hypotheses.filter(
                (pl.col("paradigm_id") == pid)
                & (pl.col("principle_id") == prid)
                & (pl.col("candidate_id") == cand)
                & (pl.col("experiment_id") == exp)
            )
        else:
            dh_g = decisions_hypotheses

        # Critic config (Phase A/B stub; later resolve from principle/paradigm config)
        critic_cfg: dict[str, Any] = {}

        # Resolve critic implementation; if missing, use noop to allow stub paradigms to run
        try:
            critic_fn = get_critic(pid)
            # NOTE: We keep your current calling convention to avoid breaking existing critics:
            #   critic_fn(ctx, trade_paths_df, decisions_hypotheses_df, cfg)
            dc_g = critic_fn(state.ctx, tp_g, dh_g, critic_cfg)
        except KeyError:
            log.warning("critic_step: no critic registered for paradigm_id=%s; using noop critic", pid)
            dc_g = _noop_critic_for_group(state.ctx, tp_g, dh_g, critic_cfg)

        dc_g = _ensure_decisions_critic_surface(state.ctx, dc_g, tp_g)

        # Stamp eval identity if critic returned without it (allowed in Phase A-like critics)
        if eval_mode:
            # Ensure these columns exist and match group identity
            if "paradigm_id" not in dc_g.columns:
                dc_g = dc_g.with_columns(pl.lit(pid).cast(pl.Utf8).alias("paradigm_id"))
            if "principle_id" not in dc_g.columns:
                dc_g = dc_g.with_columns(pl.lit(prid).cast(pl.Utf8).alias("principle_id"))
            if "candidate_id" not in dc_g.columns:
                dc_g = dc_g.with_columns(pl.lit(cand).cast(pl.Utf8).alias("candidate_id"))
            if "experiment_id" not in dc_g.columns:
                dc_g = dc_g.with_columns(pl.lit(exp).cast(pl.Utf8).alias("experiment_id"))
            dc_g = _normalize_eval_cols(dc_g)

        decisions_critic_groups.append(dc_g)

        log.info(
            "critic_step: paradigm=%s principle=%s candidate=%s experiment=%s n_trades=%d n_critic=%d",
            pid,
            prid,
            cand,
            exp,
            tp_g.height,
            0 if dc_g.is_empty() else dc_g.height,
        )

    decisions_critic_df = (
        pl.concat(decisions_critic_groups, how="diagonal")
        if decisions_critic_groups
        else _empty_decisions_critic_frame()
    )

    # Diagnostics: per evaluation group when eval identity exists, else run-level
    if decisions_critic_df.is_empty():
        critic_df = _empty_critic_frame()
    else:
        base_keys = ["snapshot_id", "run_id", "mode"]
        if "paradigm_id" in decisions_critic_df.columns:
            base_keys += _EVAL_KEYS

        critic_df = (
            decisions_critic_df.select(base_keys + ["critic_score_at_entry"])
            .group_by(base_keys, maintain_order=True)
            .agg(
                pl.len().alias("n_trades"),
                pl.col("critic_score_at_entry").mean().alias("avg_critic_score"),
            )
        )

    # Join critic fields back onto trade_paths using safe keys
    if not trade_paths.is_empty() and not decisions_critic_df.is_empty():
        join_keys = _join_keys_for_trade_paths(trade_paths, decisions_critic_df)
        trade_paths_updated_df = trade_paths.join(
            decisions_critic_df.select(join_keys + _CRITIC_FIELDS),
            on=join_keys,
            how="left",
        )
    else:
        trade_paths_updated_df = trade_paths

    # Write to BatchState
    state.set("decisions_critic", decisions_critic_df)
    state.set("critic", critic_df)
    state.set("trade_paths", trade_paths_updated_df)

    # Persist critic decisions (writer is Phase-B aware and will partition by evaluation identity)
    if decisions_critic_df is not None and not decisions_critic_df.is_empty():
        write_decisions_for_stage(
            ctx=state.ctx,
            trading_day=state.key.trading_day,
            stage="critic",
            decisions_df=decisions_critic_df,
        )

    return state
