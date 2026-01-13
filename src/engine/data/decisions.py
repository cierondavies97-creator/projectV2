from __future__ import annotations

import os
from datetime import date
from typing import Literal, Optional

import polars as pl

from engine.core.ids import RunContext
from engine.core.schema import enforce_table
from engine.io.parquet_io import read_parquet_dir, write_parquet
from engine.io.paths import decisions_dir

DecisionStage = Literal["hypotheses", "critic", "pretrade", "gatekeeper", "portfolio"]

_SENTINEL = "∅"


def _is_eval_mode(df: pl.DataFrame) -> bool:
    return df is not None and (not df.is_empty()) and ("paradigm_id" in df.columns)


def _ensure_core_identity_columns(ctx: RunContext, trading_day: date, df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure core run identity columns exist for downstream joins and auditability.

    Phase B policy:
      - dt (Date) is the canonical persisted day column.
      - If legacy trading_day exists but dt does not, we set dt := trading_day.
      - We do NOT stamp trading_day anymore.
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = df
    td = pl.lit(trading_day).cast(pl.Date)

    exprs: list[pl.Expr] = []

    if "snapshot_id" not in out.columns:
        exprs.append(pl.lit(ctx.snapshot_id).cast(pl.Utf8).alias("snapshot_id"))
    if "run_id" not in out.columns:
        exprs.append(pl.lit(ctx.run_id).cast(pl.Utf8).alias("run_id"))
    if "mode" not in out.columns:
        exprs.append(pl.lit(ctx.mode).cast(pl.Utf8).alias("mode"))

    if "dt" not in out.columns:
        if "trading_day" in out.columns:
            exprs.append(pl.col("trading_day").cast(pl.Date, strict=False).fill_null(td).alias("dt"))
        else:
            exprs.append(td.alias("dt"))
    else:
        exprs.append(pl.col("dt").cast(pl.Date, strict=False).fill_null(td).alias("dt"))

    if exprs:
        out = out.with_columns(exprs)

    return out


def _normalize_eval_identity(df: pl.DataFrame) -> pl.DataFrame:
    """
    Phase B policy:
      - paradigm_id implies principle_id must exist (required).
      - candidate_id is always present in eval-mode; fill null with sentinel.
      - experiment_id is optional; if present fill null with sentinel.
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = df
    if "paradigm_id" not in out.columns:
        return out

    if "principle_id" not in out.columns:
        raise ValueError("decisions frame has paradigm_id but is missing principle_id (Phase B requires both).")

    if "candidate_id" not in out.columns:
        out = out.with_columns(pl.lit(_SENTINEL).cast(pl.Utf8).alias("candidate_id"))
    else:
        out = out.with_columns(
            pl.col("candidate_id").cast(pl.Utf8, strict=False).fill_null(_SENTINEL).alias("candidate_id")
        )

    if "experiment_id" in out.columns:
        out = out.with_columns(
            pl.col("experiment_id").cast(pl.Utf8, strict=False).fill_null(_SENTINEL).alias("experiment_id")
        )

    for c in ("paradigm_id", "principle_id"):
        out = out.with_columns(pl.col(c).cast(pl.Utf8, strict=False).alias(c))

    return out


def _validate_decisions_frame(stage: DecisionStage, df: pl.DataFrame) -> None:
    if df is None or df.is_empty():
        return

    required = ["instrument", "trade_id", "dt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"decisions frame for stage={stage!r} missing required columns: {missing}")

    if "paradigm_id" in df.columns and "principle_id" not in df.columns:
        raise ValueError("decisions frame has paradigm_id but is missing principle_id (Phase B requires both).")


def write_decisions_for_stage(
    ctx: RunContext,
    trading_day: date,
    stage: DecisionStage,
    decisions_df: pl.DataFrame,
    *,
    sandbox: bool = False,
) -> None:
    if decisions_df is None or decisions_df.is_empty():
        return

    df = _ensure_core_identity_columns(ctx, trading_day, decisions_df)
    df = _normalize_eval_identity(df)

    if "stage" not in df.columns:
        df = df.with_columns(pl.lit(stage).cast(pl.Utf8).alias("stage"))

    _validate_decisions_frame(stage, df)

    eval_mode = _is_eval_mode(df)

    group_cols = ["instrument"]
    if eval_mode:
        group_cols += ["paradigm_id", "principle_id", "candidate_id"]
        if "experiment_id" in df.columns:
            group_cols.append("experiment_id")

    schema_key = f"decisions_{stage}"

    for g in df.partition_by(group_cols, maintain_order=True):
        instrument = str(g["instrument"][0])

        paradigm_id: Optional[str] = None
        principle_id: Optional[str] = None
        candidate_id: Optional[str] = None
        experiment_id: Optional[str] = None

        if eval_mode:
            paradigm_id = str(g["paradigm_id"][0])
            principle_id = str(g["principle_id"][0])
            candidate_id = str(g["candidate_id"][0]) if "candidate_id" in g.columns else _SENTINEL
            if "experiment_id" in g.columns:
                experiment_id = str(g["experiment_id"][0])

        out_dir = decisions_dir(
            ctx=ctx,
            trading_day=trading_day,
            instrument=instrument,
            stage=stage,
            paradigm_id=paradigm_id,
            principle_id=principle_id,
            candidate_id=candidate_id,
            experiment_id=experiment_id,
            sandbox=sandbox,
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        g2 = enforce_table(g, schema_key, allow_extra=True, reorder=False)
        write_parquet(g2, out_dir, file_name="0000.parquet")


def read_decisions_for_stage(
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    stage: DecisionStage,
    *,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> pl.DataFrame:
    dir_path = decisions_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        stage=stage,
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
        sandbox=sandbox,
    )

    if not os.path.isdir(dir_path):
        return pl.DataFrame()

    return read_parquet_dir(dir_path)