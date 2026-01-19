from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import polars as pl
import yaml

from engine.core.timegrid import tf_to_truncate_rule
from engine.core.schema import TRADE_PATHS_SCHEMA, polars_dtype
from engine.data.decisions import write_decisions_for_stage
from engine.data.trade_paths import write_trade_paths_for_day
from engine.microbatch.steps.contract_guard import ContractWrite, assert_contract_alignment
from engine.microbatch.types import BatchState
from engine.paradigms.api import get_hypotheses_builder
from engine.research.snapshots import load_snapshot_manifest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _none_if_blank(v: Any) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    s = v.strip()
    return None if s == "" else s


def _candidate_norm(v: Any) -> str:
    s = _none_if_blank(v)
    return s if s is not None else "∅"
# ---------------------------------------------------------------------------
# Evaluation model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationSpec:
    paradigm_id: str
    principle_id: str
    candidate_id: Optional[str] = None
    experiment_id: Optional[str] = None
    params: dict[str, Any] | None = None


def _resolve_principles_dir(snapshot: Any) -> Path:
    refs = _get(snapshot, "config_refs", {}) or {}
    p = refs.get("principles_dir") if isinstance(refs, dict) else None
    if not p:
        p = "conf/principles"
    return Path(p)


def _find_principle_cfg_path(snapshot: Any, principles_dir: Path, paradigm_id: str, principle_id: str) -> Path:
    principles = _get(snapshot, "principles", None)
    if isinstance(principles, list):
        for it in principles:
            if not isinstance(it, dict):
                continue
            if str(it.get("paradigm_id", "")) == paradigm_id and str(it.get("principle_id", "")) == principle_id:
                p = it.get("principle_config_path")
                if p:
                    return Path(str(p))
    return principles_dir / paradigm_id / f"{principle_id}.yaml"


def _load_principle_cfg(snapshot: Any, principles_dir: Path, paradigm_id: str, principle_id: str) -> dict[str, Any]:
    path = _find_principle_cfg_path(snapshot, principles_dir, paradigm_id, principle_id)
    if not path.exists():
        raise FileNotFoundError(f"principle config not found: {path}")

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        obj = {}
    if not isinstance(obj, dict):
        raise ValueError(f"principle config must be a mapping: {path}")
    return obj


def _merge_params(base_cfg: dict[str, Any], eval_params: Optional[dict[str, Any]]) -> dict[str, Any]:
    out = dict(base_cfg)
    base_params = out.get("params") or {}
    if not isinstance(base_params, dict):
        base_params = {}
    merged = dict(base_params)
    if isinstance(eval_params, dict) and eval_params:
        merged.update(eval_params)
    out["params"] = merged
    return out


def _extract_evaluations(state: BatchState, snapshot: Any) -> list[EvaluationSpec]:
    eval_plan = _get(snapshot, "evaluation_plan", None)
    evals = _get(eval_plan, "evaluations", None)

    # Phase B preferred: explicit evaluation plan
    if isinstance(evals, list) and len(evals) > 0:
        out: list[EvaluationSpec] = []
        for e in evals:
            if not isinstance(e, dict):
                raise ValueError("evaluation_plan.evaluations must be a list of objects")
            pid = _none_if_blank(e.get("paradigm_id"))
            prid = _none_if_blank(e.get("principle_id"))
            if not pid or not prid:
                raise ValueError(f"Invalid evaluation entry (missing paradigm_id/principle_id): {e}")
            out.append(
                EvaluationSpec(
                    paradigm_id=str(pid),
                    principle_id=str(prid),
                    candidate_id=_none_if_blank(e.get("candidate_id")),
                    experiment_id=_none_if_blank(e.get("experiment_id")),
                    params=(e.get("params") or {}) if isinstance(e.get("params") or {}, dict) else {},
                )
            )
        return out

    # Fallback: snapshot.principles list (common in your current snapshots)
    principles = _get(snapshot, "principles", None)
    if isinstance(principles, list) and len(principles) > 0:
        out2: list[EvaluationSpec] = []
        for p in principles:
            if not isinstance(p, dict):
                continue
            pid = _none_if_blank(p.get("paradigm_id"))
            prid = _none_if_blank(p.get("principle_id"))
            if not pid or not prid:
                continue
            out2.append(EvaluationSpec(paradigm_id=str(pid), principle_id=str(prid)))
        if out2:
            return out2

    # Final fallback
    return [EvaluationSpec(paradigm_id="ict", principle_id="ict_all_windows")]


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _default_expr_for_type(col_name: str, col_type: str) -> pl.Expr:
    if col_type == "string":
        return pl.lit(None).cast(pl.Utf8).alias(col_name)
    if col_type in ("int", "int64"):
        return pl.lit(None).cast(pl.Int64).alias(col_name)
    if col_type in ("double", "float", "float64"):
        return pl.lit(None).cast(pl.Float64).alias(col_name)
    if col_type == "boolean":
        return pl.lit(False).alias(col_name)
    if col_type == "timestamp":
        return pl.lit(None).cast(polars_dtype(col_type)).alias(col_name)
    return pl.lit(None).cast(pl.Utf8).alias(col_name)


def _ensure_trade_paths_schema(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame()

    missing_exprs: list[pl.Expr] = []
    existing = set(df.columns)
    for name, typ in TRADE_PATHS_SCHEMA.columns.items():
        if name not in existing:
            missing_exprs.append(_default_expr_for_type(name, typ))

    out = df if not missing_exprs else df.with_columns(missing_exprs)

    # Stable partitions (Phase B)
    if "candidate_id" in out.columns:
        out = out.with_columns(pl.col("candidate_id").fill_null("∅"))
        if "experiment_id" in out.columns:
            out = out.with_columns(pl.col("experiment_id").fill_null("∅"))
        else:
            out = out.with_columns(pl.lit("∅").cast(pl.Utf8).alias("experiment_id"))
    return out


def _safe_truncate_rule(tf: str | None) -> str | None:
    if tf is None:
        return None
    try:
        return tf_to_truncate_rule(str(tf))
    except ValueError:
        return None


def _add_alignment_flag(df: pl.DataFrame, *, tf_col: str, flag_col: str) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame()
    if "entry_ts" not in df.columns or tf_col not in df.columns:
        return df

    frames: list[pl.DataFrame] = []
    for (tf_val,), grp in df.group_by([tf_col], maintain_order=True):
        rule = _safe_truncate_rule(str(tf_val) if tf_val is not None else None)
        if rule is None:
            frames.append(grp.with_columns(pl.lit(None).cast(pl.Boolean).alias(flag_col)))
            continue
        entry_ts = pl.col("entry_ts").cast(pl.Datetime("us"), strict=False)
        aligned = entry_ts.dt.truncate(rule) == entry_ts
        frames.append(grp.with_columns(aligned.alias(flag_col)))
    return pl.concat(frames, how="vertical") if frames else df


def _finalize_trade_paths(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = _ensure_trade_paths_schema(df)

    if "anchor_ts" not in out.columns and "entry_ts" in out.columns:
        out = out.with_columns(pl.col("entry_ts").alias("anchor_ts"))

    if "entry_ts_source" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("entry_ts_source").is_null() | (pl.col("entry_ts_source").cast(pl.Utf8).str.strip_chars() == ""))
            .then(
                pl.when(
                    (pl.col("anchor_ts").is_not_null())
                    & (pl.col("entry_ts").cast(pl.Datetime("us"), strict=False) == pl.col("anchor_ts").cast(pl.Datetime("us"), strict=False))
                )
                .then(pl.lit("anchor_close"))
                .otherwise(pl.lit("signal_ts"))
            )
            .otherwise(pl.col("entry_ts_source"))
            .alias("entry_ts_source")
        )

    if "entry_ts_offset_ms" in out.columns:
        entry_us = pl.col("entry_ts").cast(pl.Datetime("us"), strict=False).cast(pl.Int64)
        anchor_us = pl.col("anchor_ts").cast(pl.Datetime("us"), strict=False).cast(pl.Int64)
        out = out.with_columns(
            pl.when(pl.col("entry_ts_offset_ms").is_null() & pl.col("anchor_ts").is_not_null() & pl.col("entry_ts").is_not_null())
            .then(((entry_us - anchor_us) / 1000).cast(pl.Int64))
            .otherwise(pl.col("entry_ts_offset_ms"))
            .alias("entry_ts_offset_ms")
        )

    out = _add_alignment_flag(out, tf_col="anchor_tf", flag_col="entry_ts_is_aligned_anchor_tf")
    out = _add_alignment_flag(out, tf_col="tf_entry", flag_col="entry_ts_is_aligned_entry_tf")
    return out


def _ensure_decisions_min_schema(df: pl.DataFrame) -> pl.DataFrame:
    if df is None:
        return pl.DataFrame()
    if df.is_empty():
        return df

    needed: dict[str, pl.Expr] = {
        "snapshot_id": pl.lit(None).cast(pl.Utf8).alias("snapshot_id"),
        "run_id": pl.lit(None).cast(pl.Utf8).alias("run_id"),
        "mode": pl.lit(None).cast(pl.Utf8).alias("mode"),
        "paradigm_id": pl.lit(None).cast(pl.Utf8).alias("paradigm_id"),
        "principle_id": pl.lit(None).cast(pl.Utf8).alias("principle_id"),
        "candidate_id": pl.lit("∅").cast(pl.Utf8).alias("candidate_id"),
        "experiment_id": pl.lit("∅").cast(pl.Utf8).alias("experiment_id"),
        "instrument": pl.lit(None).cast(pl.Utf8).alias("instrument"),
        "trade_id": pl.lit(None).cast(pl.Utf8).alias("trade_id"),
    }

    exprs: list[pl.Expr] = []
    cols = set(df.columns)
    for k, e in needed.items():
        if k not in cols:
            exprs.append(e)

    out = df if not exprs else df.with_columns(exprs)

    # normalize candidate_id if present
    if "candidate_id" in out.columns:
        out = out.with_columns(pl.col("candidate_id").fill_null("∅"))

    return out


def _anchor_join_keys(decisions: pl.DataFrame, trade_paths: pl.DataFrame) -> list[str]:
    keys: list[str] = []
    for c in ["trade_id", "instrument"]:
        if c in decisions.columns and c in trade_paths.columns:
            keys.append(c)
    for c in ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]:
        if c in decisions.columns and c in trade_paths.columns:
            keys.append(c)
    if not keys and "trade_id" in decisions.columns and "trade_id" in trade_paths.columns:
        keys = ["trade_id"]
    return keys


def _truncate_anchor_ts_by_tf(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame()
    if "anchor_tf" not in df.columns or "anchor_ts" not in df.columns:
        return df

    frames: list[pl.DataFrame] = []
    for (anchor_tf,), grp in df.group_by(["anchor_tf"], maintain_order=True):
        if anchor_tf is None:
            frames.append(grp)
            continue
        rule = tf_to_truncate_rule(str(anchor_tf))
        frames.append(
            grp.with_columns(
                pl.col("anchor_ts")
                .cast(pl.Datetime("us"), strict=False)
                .dt.truncate(rule)
                .alias("anchor_ts")
            )
        )
    return pl.concat(frames, how="vertical") if frames else df


def _enrich_decisions_anchor(decisions: pl.DataFrame, trade_paths: pl.DataFrame | None) -> pl.DataFrame:
    if decisions is None or decisions.is_empty():
        return pl.DataFrame()

    out = decisions
    tp = trade_paths if trade_paths is not None else pl.DataFrame()

    if tp is not None and not tp.is_empty():
        keys = _anchor_join_keys(out, tp)
        if keys:
            tp_cols = []
            if "anchor_tf" in tp.columns:
                tp_cols.append(pl.col("anchor_tf").alias("_tp_anchor_tf"))
            if "entry_ts" in tp.columns:
                tp_cols.append(pl.col("entry_ts").alias("_tp_entry_ts"))
            if tp_cols:
                anchor_map = tp.select([*keys, *tp_cols]).unique()
                out = out.join(anchor_map, on=keys, how="left")

    anchor_exprs: list[pl.Expr] = []
    if "_tp_anchor_tf" in out.columns:
        if "anchor_tf" in out.columns:
            anchor_exprs.append(pl.coalesce([pl.col("anchor_tf"), pl.col("_tp_anchor_tf")]).alias("anchor_tf"))
        else:
            anchor_exprs.append(pl.col("_tp_anchor_tf").alias("anchor_tf"))

    anchor_ts_sources: list[pl.Expr] = []
    if "anchor_ts" in out.columns:
        anchor_ts_sources.append(pl.col("anchor_ts"))
    if "entry_ts" in out.columns:
        anchor_ts_sources.append(pl.col("entry_ts"))
    if "_tp_entry_ts" in out.columns:
        anchor_ts_sources.append(pl.col("_tp_entry_ts"))
    if anchor_ts_sources:
        anchor_exprs.append(
            pl.coalesce(anchor_ts_sources).cast(pl.Datetime("us"), strict=False).alias("anchor_ts")
        )

    if anchor_exprs:
        out = out.with_columns(anchor_exprs)

    drop_cols = [c for c in ["_tp_anchor_tf", "_tp_entry_ts"] if c in out.columns]
    if drop_cols:
        out = out.drop(drop_cols)

    return _truncate_anchor_ts_by_tf(out)


def _stamp_eval_identity(
    df: pl.DataFrame,
    *,
    ctx_snapshot_id: str,
    ctx_run_id: str,
    ctx_mode: str,
    eval_spec: EvaluationSpec,
) -> pl.DataFrame:
    if df is None:
        return pl.DataFrame()
    if df.is_empty():
        return df

    cand = _candidate_norm(eval_spec.candidate_id)
    exp = _none_if_blank(eval_spec.experiment_id) or "∅"
    exprs: list[pl.Expr] = [
        pl.lit(ctx_snapshot_id).cast(pl.Utf8).alias("snapshot_id"),
        pl.lit(ctx_run_id).cast(pl.Utf8).alias("run_id"),
        pl.lit(ctx_mode).cast(pl.Utf8).alias("mode"),
        pl.lit(eval_spec.paradigm_id).cast(pl.Utf8).alias("paradigm_id"),
        pl.lit(eval_spec.principle_id).cast(pl.Utf8).alias("principle_id"),
        pl.lit(cand).cast(pl.Utf8).alias("candidate_id"),
    ]
    # Phase B: always stamp experiment_id (partition-stable)
    exprs.append(pl.lit(exp).cast(pl.Utf8).alias("experiment_id"))
    return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run(state: BatchState) -> BatchState:
    """
    Phase B hypotheses step: multi-evaluation dispatcher.

    Inputs:
      - 'windows'
      - 'features'

    Outputs:
      - 'decisions_hypotheses' (long format)
      - 'trade_paths' (long format)
    """
    assert_contract_alignment(
        step_name="hypotheses_step",
        writes=(
            ContractWrite(
                table_key="decisions_hypotheses",
                writer_fn="write_decisions_for_stage",
                stage="hypotheses",
            ),
            ContractWrite(table_key="trade_paths", writer_fn="write_trade_paths_for_day"),
        ),
    )
    windows = state.get("windows")
    features = state.get("features")

    if windows is None or windows.is_empty():
        state.set("decisions_hypotheses", pl.DataFrame())
        state.set("trade_paths", pl.DataFrame())
        return state

    snapshot = load_snapshot_manifest(state.ctx.snapshot_id)
    principles_dir = _resolve_principles_dir(snapshot)

    evals = _extract_evaluations(state, snapshot)

    log.info(
        "hypotheses_step: resolved %d evaluation(s) trading_day=%s cluster_id=%s",
        len(evals),
        state.key.trading_day.isoformat(),
        state.key.cluster_id,
    )

    decisions_frames: list[pl.DataFrame] = []
    trade_paths_frames: list[pl.DataFrame] = []

    for ev in evals:
        base_cfg = _load_principle_cfg(snapshot, principles_dir, ev.paradigm_id, ev.principle_id)
        cfg = _merge_params(base_cfg, ev.params)

        builder = get_hypotheses_builder(ev.paradigm_id, ev.principle_id)
        dec_df, tp_df = builder(state.ctx, windows, features, cfg)

        dec_df = _stamp_eval_identity(
            dec_df,
            ctx_snapshot_id=state.ctx.snapshot_id,
            ctx_run_id=state.ctx.run_id,
            ctx_mode=state.ctx.mode,
            eval_spec=ev,
        )
        dec_df = _ensure_decisions_min_schema(dec_df)

        tp_df = _stamp_eval_identity(
            tp_df,
            ctx_snapshot_id=state.ctx.snapshot_id,
            ctx_run_id=state.ctx.run_id,
            ctx_mode=state.ctx.mode,
            eval_spec=ev,
        )

        if not dec_df.is_empty():
            for c in ["instrument", "trade_id"]:
                if c not in dec_df.columns:
                    raise ValueError(
                        f"hypotheses builder output missing required column {c!r} for {ev.paradigm_id}/{ev.principle_id}"
                    )

        if tp_df is not None and not tp_df.is_empty():
            for c in ["instrument", "trade_id", "anchor_tf", "tf_entry", "entry_ts", "side"]:
                if c not in tp_df.columns:
                    raise ValueError(
                        f"trade_paths output missing required column {c!r} for {ev.paradigm_id}/{ev.principle_id}"
                    )

            # Keep trade_paths schema-complete early (reports_step will validate again if needed)
            tp_df = _ensure_trade_paths_schema(tp_df)

        if dec_df is not None and not dec_df.is_empty():
            dec_df = _enrich_decisions_anchor(dec_df, tp_df)
            if "decision_ts" in dec_df.columns:
                dec_df = dec_df.drop("decision_ts")
        decisions_frames.append(dec_df)
        if tp_df is not None and not tp_df.is_empty():
            trade_paths_frames.append(tp_df)

        log.info(
            "hypotheses_step: eval=%s/%s decisions=%d trade_paths=%d",
            ev.paradigm_id,
            ev.principle_id,
            0 if dec_df is None else dec_df.height,
            0 if tp_df is None else tp_df.height,
        )

    decisions_hypotheses_df = pl.concat(decisions_frames, how="diagonal") if decisions_frames else pl.DataFrame()
    trade_paths_df = pl.concat(trade_paths_frames, how="diagonal") if trade_paths_frames else pl.DataFrame()
    trade_paths_df = _finalize_trade_paths(trade_paths_df) if trade_paths_df is not None else pl.DataFrame()

    state.set("decisions_hypotheses", decisions_hypotheses_df)
    state.set("trade_paths", trade_paths_df)
    state.metrics["trade_paths_written_rows"] = int(trade_paths_df.height) if trade_paths_df is not None else 0

    trading_day: date = state.key.trading_day

    # Persist decisions (writer partitions by eval identity if those cols exist)
    if decisions_hypotheses_df is not None and not decisions_hypotheses_df.is_empty():
        write_decisions_for_stage(
            ctx=state.ctx,
            trading_day=trading_day,
            stage="hypotheses",
            decisions_df=decisions_hypotheses_df,
        )
    # Persist trade_paths (contract owner boundary for hypotheses step).
    if trade_paths_df is not None and not trade_paths_df.is_empty():
        for df_part in trade_paths_df.partition_by("instrument", as_dict=False, maintain_order=True):
            instrument = str(df_part["instrument"][0])
            write_trade_paths_for_day(
                ctx=state.ctx,
                df=df_part,
                instrument=instrument,
                trading_day=trading_day,
                sandbox=False,
            )

    return state
