from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import polars as pl

from engine.core.config_models import load_retail_config
from engine.core.timegrid import tf_to_truncate_rule
from engine.data.decisions import write_decisions_for_stage
from engine.microbatch.steps.contract_guard import ContractWrite, assert_contract_alignment
from engine.microbatch.types import BatchState

log = logging.getLogger(__name__)

# Phase B evaluation identity columns (long-format decisions)
_EVAL_KEYS = ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroPretradePolicy:
    """
    Simple macro blackout policy for pretrade.

    Fields:
      enabled             : if False, macro blackout is ignored.
      on_blackout         : "drop" -> drop blackout trades,
                            "tag_only" -> keep trades, add tag column.
      require_macro_column: if True and macro_is_blackout is missing,
                            we log a warning but still pass through.
    """

    enabled: bool
    on_blackout: str  # "drop" or "tag_only"
    require_macro_column: bool


def _load_macro_policy() -> MacroPretradePolicy:
    """
    Load macro pretrade policy from typed RetailConfig.

    Current retail.yaml (example):

      macro:
        enable_macro_state: true
        macro_calendar_source: "data/macro"
        blackout_default_minutes_pre: 15
        blackout_default_minutes_post: 15

    Interpret as:
      - if enable_macro_state is False or macro block missing:
          enabled=False
      - else:
          enabled=True
          on_blackout="drop"
          require_macro_column=False

    Optional future extension:
      macro.pretrade:
        enable_blackout_filter: bool
        on_blackout: "drop" | "tag_only"
        require_macro_column: bool
    """
    retail = load_retail_config()
    macro_cfg = getattr(retail, "macro", None)

    if macro_cfg is None or not getattr(macro_cfg, "enable_macro_state", False):
        return MacroPretradePolicy(enabled=False, on_blackout="drop", require_macro_column=False)

    pretrade_cfg = getattr(macro_cfg, "pretrade", None)

    enabled = True
    on_blackout = "drop"
    require_macro_column = False

    if pretrade_cfg is not None:
        enabled = bool(getattr(pretrade_cfg, "enable_blackout_filter", True))
        on_blackout = str(getattr(pretrade_cfg, "on_blackout", "drop"))
        require_macro_column = bool(getattr(pretrade_cfg, "require_macro_column", False))

    if on_blackout not in ("drop", "tag_only"):
        log.warning("pretrade_step: unsupported on_blackout='%s'; falling back to 'drop'", on_blackout)
        on_blackout = "drop"

    return MacroPretradePolicy(enabled=enabled, on_blackout=on_blackout, require_macro_column=require_macro_column)


# ---------------------------------------------------------------------------
# Phase B helpers (evaluation-safe)
# ---------------------------------------------------------------------------


def _is_eval_mode(df: pl.DataFrame) -> bool:
    return df is not None and (not df.is_empty()) and ("paradigm_id" in df.columns)


def _normalize_eval_identity(df: pl.DataFrame) -> pl.DataFrame:
    """
    Phase B invariant:
      If paradigm_id exists, principle_id must exist.

    Normalization:
      - ensure candidate_id/experiment_id columns exist
      - cast eval columns to Utf8
      - fill nulls with sentinel "∅" (partition-stable, join-safe)
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = df
    if "paradigm_id" not in out.columns:
        return out

    if "principle_id" not in out.columns:
        raise ValueError("pretrade_step: decisions has paradigm_id but missing principle_id (Phase B requires both).")

    if "candidate_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("candidate_id"))
    if "experiment_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("experiment_id"))

    for c in ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]:
        out = out.with_columns(pl.col(c).cast(pl.Utf8))

    out = out.with_columns(
        [
            pl.when(pl.col("candidate_id").is_null() | (pl.col("candidate_id").cast(pl.Utf8).str.strip_chars() == "")).then(pl.lit("∅")).otherwise(pl.col("candidate_id")).alias("candidate_id"),
            pl.when(pl.col("experiment_id").is_null() | (pl.col("experiment_id").cast(pl.Utf8).str.strip_chars() == "")).then(pl.lit("∅")).otherwise(pl.col("experiment_id")).alias("experiment_id"),
        ]
    )
    return out


def _coerce_macro_column(decisions: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure macro_is_blackout behaves as a boolean with nulls treated as False.
    """
    if decisions is None or decisions.is_empty():
        return pl.DataFrame()

    if "macro_is_blackout" not in decisions.columns:
        return decisions

    out = decisions
    # Cast to Boolean; treat null as False for filtering logic
    out = out.with_columns(pl.col("macro_is_blackout").cast(pl.Boolean).fill_null(False))
    return out


def _log_eval_counts(prefix: str, df: pl.DataFrame) -> None:
    """
    Log compact counts by evaluation identity when present, else log total.
    """
    if df is None or df.is_empty():
        log.info("%s: n=0", prefix)
        return

    if not _is_eval_mode(df):
        log.info("%s: n=%d", prefix, df.height)
        return

    keys = ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]
    counts = df.group_by(keys, maintain_order=True).agg(pl.len().alias("n"))
    # Avoid noisy logs: show first 10 groups plus total groups.
    head = counts.head(10)
    log.info("%s: groups=%d sample=%s", prefix, counts.height, head.to_dicts())


# ---------------------------------------------------------------------------
# Policy application
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Enrichment: bring macro blackout from windows into decisions (if available)
# ---------------------------------------------------------------------------

def _enrich_decisions_with_windows_macro(decisions: pl.DataFrame, windows: pl.DataFrame | None) -> pl.DataFrame:
    """
    Join macro blackout fields from windows into decisions, if decisions do not already contain them.

    Join keys:
      decisions.(instrument, anchor_tf, join_ts) == windows.(instrument, anchor_tf, anchor_ts)

    join_ts selection on decisions (first found):
      anchor_ts, window_ts, ts, entry_ts

    If only entry_ts is available and exactly one anchor_tf exists in this microbatch,
    we truncate entry_ts to the anchor grid to match windows.anchor_ts.
    """
    if decisions is None or decisions.is_empty():
        return pl.DataFrame()
    if windows is None or windows.is_empty():
        return decisions
    if "macro_is_blackout" in decisions.columns:
        return decisions

    need_w = {"instrument", "anchor_tf", "anchor_ts", "macro_is_blackout"}
    if not need_w.issubset(set(windows.columns)):
        return decisions

    if ("instrument" not in decisions.columns) or ("anchor_tf" not in decisions.columns):
        return decisions

    if "anchor_ts" in decisions.columns:
        ts_col = "anchor_ts"
    else:
        ts_candidates = ["window_ts", "ts", "entry_ts"]
        ts_col = next((c for c in ts_candidates if c in decisions.columns), None)
        if ts_col is None:
            return decisions

    d = decisions.with_columns(
        pl.col("instrument").cast(pl.Utf8, strict=False),
        pl.col("anchor_tf").cast(pl.Utf8, strict=False),
    )

    join_ts = pl.col(ts_col).cast(pl.Datetime("us"), strict=False)

    if ts_col == "entry_ts":
        # Safe only if one anchor_tf for the batch; otherwise skip truncation.
        uniq = d.select(pl.col("anchor_tf").unique()).to_series().to_list()
        if len(uniq) == 1 and uniq[0]:
            rule = tf_to_truncate_rule(str(uniq[0]))
            join_ts = join_ts.dt.truncate(rule)

    d = d.with_columns(join_ts.alias("_join_anchor_ts"))

    w = windows.select(
        [
            pl.col("instrument").cast(pl.Utf8, strict=False),
            pl.col("anchor_tf").cast(pl.Utf8, strict=False),
            pl.col("anchor_ts").cast(pl.Datetime("us"), strict=False).alias("_join_anchor_ts"),
            pl.col("macro_is_blackout").cast(pl.Boolean, strict=False),
            pl.col("macro_blackout_max_impact").cast(pl.Int64, strict=False)
            if "macro_blackout_max_impact" in windows.columns
            else pl.lit(None).cast(pl.Int64).alias("macro_blackout_max_impact"),
            pl.col("macro_state").cast(pl.Utf8, strict=False)
            if "macro_state" in windows.columns
            else pl.lit(None).cast(pl.Utf8).alias("macro_state"),
        ]
    ).unique(subset=["instrument", "anchor_tf", "_join_anchor_ts"], keep="first")

    out = d.join(w, on=["instrument", "anchor_tf", "_join_anchor_ts"], how="left").drop("_join_anchor_ts")
    return out


def _ensure_anchor_fields(decisions: pl.DataFrame) -> pl.DataFrame:
    if decisions is None or decisions.is_empty():
        return pl.DataFrame()
    if "anchor_tf" not in decisions.columns:
        return decisions

    out = decisions

    if "anchor_ts" in out.columns:
        out = out.with_columns(pl.col("anchor_ts").cast(pl.Datetime("us"), strict=False).alias("anchor_ts"))
    else:
        ts_sources = [c for c in ["entry_ts", "window_ts", "ts"] if c in out.columns]
        if not ts_sources:
            return out
        out = out.with_columns(
            pl.coalesce([pl.col(c) for c in ts_sources]).cast(pl.Datetime("us"), strict=False).alias("anchor_ts")
        )

    frames: list[pl.DataFrame] = []
    for (anchor_tf,), grp in out.group_by(["anchor_tf"], maintain_order=True):
        if anchor_tf is None:
            frames.append(grp)
            continue
        rule = tf_to_truncate_rule(str(anchor_tf))
        frames.append(
            grp.with_columns(pl.col("anchor_ts").dt.truncate(rule).alias("anchor_ts"))
        )
    return pl.concat(frames, how="vertical") if frames else out


def _apply_macro_policy(
    *,
    decisions: pl.DataFrame,
    policy: MacroPretradePolicy,
    snapshot_id: str,
    run_id: str,
) -> pl.DataFrame:
    """
    Apply macro blackout policy to hypotheses decisions.

    Expects a 'macro_is_blackout' boolean column if macro features
    were joined into the hypotheses step.

    Phase B: operates row-wise and preserves evaluation identity columns.
    """
    if decisions is None or decisions.is_empty():
        return pl.DataFrame()

    dec = decisions
    dec = _normalize_eval_identity(dec)
    dec = _coerce_macro_column(dec)

    if not policy.enabled:
        log.info(
            "pretrade_step: macro policy disabled; snapshot_id=%s run_id=%s passing through",
            snapshot_id,
            run_id,
        )
        _log_eval_counts("pretrade_step: passthrough (macro disabled)", dec)
        return dec

    has_macro = "macro_is_blackout" in dec.columns
    if not has_macro:
        msg = "pretrade_step: macro policy enabled but 'macro_is_blackout' missing; passing through unchanged."
        if policy.require_macro_column:
            log.warning("%s snapshot_id=%s run_id=%s (require_macro_column=True)", msg, snapshot_id, run_id)
        else:
            log.info("%s snapshot_id=%s run_id=%s", msg, snapshot_id, run_id)
        _log_eval_counts("pretrade_step: passthrough (macro col missing)", dec)
        return dec

    if policy.on_blackout == "drop":
        n_before = dec.height
        kept = dec.filter(~pl.col("macro_is_blackout"))
        n_after = kept.height
        dropped = n_before - n_after

        if _is_eval_mode(dec):
            # Log per-evaluation dropped counts (bounded)
            keys = ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]
            by = (
                dec.group_by(keys, maintain_order=True)
                .agg(
                    pl.len().alias("n_before"),
                    pl.col("macro_is_blackout").cast(pl.Int64).sum().alias("n_dropped"),
                )
                .with_columns((pl.col("n_before") - pl.col("n_dropped")).alias("n_after"))
            )
            log.info(
                "pretrade_step: snapshot_id=%s run_id=%s macro drop applied (total_dropped=%d total_before=%d total_after=%d) sample=%s",
                snapshot_id,
                run_id,
                dropped,
                n_before,
                n_after,
                by.head(10).to_dicts(),
            )
        else:
            log.info(
                "pretrade_step: snapshot_id=%s run_id=%s dropped %d trades due to macro blackout (n_before=%d n_after=%d)",
                snapshot_id,
                run_id,
                dropped,
                n_before,
                n_after,
            )
        return kept

    if policy.on_blackout == "tag_only":
        tagged = dec.with_columns(
            pl.when(pl.col("macro_is_blackout"))
            .then(pl.lit("macro_blackout"))
            .otherwise(pl.lit(""))
            .alias("macro_blackout_tag")
        )

        n_blackout = int(
            tagged.select(pl.col("macro_is_blackout").cast(pl.Int64).sum().alias("n"))["n"][0]
        )

        if _is_eval_mode(tagged):
            keys = ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]
            by = (
                tagged.group_by(keys, maintain_order=True)
                .agg(pl.col("macro_is_blackout").cast(pl.Int64).sum().alias("n_blackout"), pl.len().alias("n_total"))
            )
            log.info(
                "pretrade_step: snapshot_id=%s run_id=%s tag_only macro applied (total_blackout=%d total=%d) sample=%s",
                snapshot_id,
                run_id,
                n_blackout,
                tagged.height,
                by.head(10).to_dicts(),
            )
        else:
            log.info(
                "pretrade_step: snapshot_id=%s run_id=%s tag_only macro applied (blackout_trades=%d total=%d)",
                snapshot_id,
                run_id,
                n_blackout,
                tagged.height,
            )
        return tagged

    log.warning(
        "pretrade_step: unexpected policy.on_blackout=%s; returning decisions unchanged",
        policy.on_blackout,
    )
    _log_eval_counts("pretrade_step: passthrough (unexpected policy)", dec)
    return dec


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run(state: BatchState) -> BatchState:
    assert_contract_alignment(
        step_name="pretrade_step",
        writes=(
            ContractWrite(
                table_key="decisions_pretrade",
                writer_fn="write_decisions_for_stage",
                stage="pretrade",
            ),
        ),
    )
    ctx = state.ctx

    # Correct input: decisions_critic
    decisions = state.get_optional("decisions_critic")
    if decisions is None:
        decisions = pl.DataFrame()
    policy = _load_macro_policy()

    # Enrich decisions with macro blackout fields from windows (if windows carries them).
    windows = state.get_optional("windows")
    decisions = _enrich_decisions_with_windows_macro(decisions, windows)

    decisions = _apply_macro_policy(
        decisions=decisions,
        policy=policy,
        snapshot_id=ctx.snapshot_id,
        run_id=ctx.run_id,
    )
    decisions = _ensure_anchor_fields(decisions)
    # Ensure macro_is_blackout exists before referencing it (critic decisions may not carry macro columns).
    if "macro_is_blackout" not in decisions.columns:
        decisions = decisions.with_columns(pl.lit(False).cast(pl.Boolean).alias("macro_is_blackout"))



    # Add required columns (stubbed or from logic)
    decisions = decisions.with_columns([
        pl.col("macro_is_blackout").alias("blocked_by_macro"),
        pl.lit(False).alias("blocked_by_spread"),
        pl.lit(1.0).cast(pl.Float64).alias("micro_risk_scale"),
        pl.lit("").cast(pl.Utf8).alias("pretrade_notes"),
    ])

    state.set("decisions_pretrade", decisions)

    trade_paths = state.get_optional("trade_paths")
    if trade_paths is not None:
        state.set("trade_paths", trade_paths)

    if not decisions.is_empty():
        write_decisions_for_stage(
            ctx=ctx,
            trading_day=state.key.trading_day,
            stage="pretrade",
            decisions_df=decisions,
        )

    return state



