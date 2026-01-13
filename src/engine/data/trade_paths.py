from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl

from engine.core.ids import RunContext
from engine.core.schema import TRADE_PATHS_SCHEMA
from engine.io.parquet_io import read_parquet_dir, write_parquet
from engine.io.paths import trade_paths_dir

_SENTINEL = "∅"
_TS = pl.Datetime("us")


def _dtype_for_schema_type(t: str) -> pl.DataType:
    t = (t or "").lower()
    if t == "string":
        return pl.Utf8
    if t in ("int", "int64"):
        return pl.Int64
    if t in ("int32",):
        return pl.Int32
    if t in ("double", "float", "float64"):
        return pl.Float64
    if t == "boolean":
        return pl.Boolean
    if t == "timestamp":
        return _TS
    if t == "date":
        return pl.Date
    return pl.Utf8


def _is_eval_mode(df: pl.DataFrame) -> bool:
    return df is not None and (not df.is_empty()) and ("paradigm_id" in df.columns)


def _require_trade_identity(df: pl.DataFrame) -> None:
    """
    Trade identity must be produced by the deterministic lane (steps). IO must not invent it.
    """
    missing = [c for c in ("instrument", "trade_id") if c not in df.columns]
    if missing:
        raise ValueError(f"trade_paths missing required identity columns (must be set by step logic): {missing}")


def ensure_trade_paths_engine_cols(ctx: RunContext, df: pl.DataFrame, trading_day: date) -> pl.DataFrame:
    """
    Ensure trade_paths has all columns required by TRADE_PATHS_SCHEMA, with correct dtypes,
    and with stable run identity columns.

    Phase B policy:
      - If paradigm_id exists, principle_id must exist.
      - In eval-mode, candidate_id always exists (use sentinel if absent/null).
      - experiment_id optional; if present, null -> sentinel.

    Note:
      - We add placeholder columns as typed NULLs.
      - We do NOT create trade_id/instrument; those must already exist.
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    _require_trade_identity(df)

    out = df

    # Stamp run identity (safe)
    if "snapshot_id" not in out.columns:
        out = out.with_columns(pl.lit(ctx.snapshot_id).cast(pl.Utf8).alias("snapshot_id"))
    if "run_id" not in out.columns:
        out = out.with_columns(pl.lit(ctx.run_id).cast(pl.Utf8).alias("run_id"))
    if "mode" not in out.columns:
        out = out.with_columns(pl.lit(ctx.mode).cast(pl.Utf8).alias("mode"))

    # dt is required and should be a Date
    if "dt" not in out.columns:
        out = out.with_columns(pl.lit(trading_day).cast(pl.Date).alias("dt"))
    else:
        out = out.with_columns(pl.col("dt").cast(pl.Date, strict=False).alias("dt"))

    # Enforce Phase B eval identity constraints if present
    if "paradigm_id" in out.columns:
        if "principle_id" not in out.columns:
            raise ValueError("trade_paths has paradigm_id but is missing principle_id (Phase B requires both).")
        if "candidate_id" not in out.columns:
            out = out.with_columns(pl.lit(_SENTINEL).cast(pl.Utf8).alias("candidate_id"))

    # Normalize candidate_id / experiment_id if present
    if "candidate_id" in out.columns:
        out = out.with_columns(pl.col("candidate_id").cast(pl.Utf8, strict=False).fill_null(_SENTINEL).alias("candidate_id"))
    if "experiment_id" in out.columns:
        out = out.with_columns(pl.col("experiment_id").cast(pl.Utf8, strict=False).fill_null(_SENTINEL).alias("experiment_id"))

    # Key dtype normalization
    cast_exprs: list[pl.Expr] = []
    for k in ("run_id", "instrument", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "trade_id"):
        if k in out.columns:
            cast_exprs.append(pl.col(k).cast(pl.Utf8, strict=False).alias(k))
    if "entry_ts" in out.columns:
        cast_exprs.append(pl.col("entry_ts").cast(_TS, strict=False).alias("entry_ts"))
    if "exit_ts" in out.columns:
        cast_exprs.append(pl.col("exit_ts").cast(_TS, strict=False).alias("exit_ts"))
    if cast_exprs:
        out = out.with_columns(cast_exprs)

    # Add any missing schema columns as typed NULLs (or stable JSON default)
    cols = set(out.columns)
    add_exprs: list[pl.Expr] = []

    for name, typ in TRADE_PATHS_SCHEMA.columns.items():
        if name in cols:
            # Cast to expected type when possible (strict=False to avoid hard failures during transition)
            add_exprs.append(pl.col(name).cast(_dtype_for_schema_type(typ), strict=False).alias(name))
            continue

        dtype = _dtype_for_schema_type(typ)
        if name == "dt":
            add_exprs.append(pl.lit(trading_day).cast(pl.Date).alias(name))
        elif name in ("snapshot_id", "run_id", "mode"):
            # already added above, but keep deterministic if schema demands it
            if name == "snapshot_id":
                add_exprs.append(pl.lit(ctx.snapshot_id).cast(pl.Utf8).alias(name))
            elif name == "run_id":
                add_exprs.append(pl.lit(ctx.run_id).cast(pl.Utf8).alias(name))
            else:
                add_exprs.append(pl.lit(ctx.mode).cast(pl.Utf8).alias(name))
        else:
            add_exprs.append(pl.lit(None).cast(dtype).alias(name))

    if add_exprs:
        out = out.with_columns(add_exprs)

    # Final: strict day scoping (do not write off-day rows)
    if "entry_ts" in out.columns:
        out = out.filter(pl.col("entry_ts").dt.date() == pl.lit(trading_day))

    # Deterministic de-dupe on trade identity (and eval identity when present)
    key_cols = ["instrument", "dt", "trade_id"]
    if _is_eval_mode(out):
        key_cols += ["paradigm_id", "principle_id", "candidate_id"]
        if "experiment_id" in out.columns:
            key_cols.append("experiment_id")

    out = out.sort(key_cols).unique(subset=key_cols, keep="first")

    return out


def validate_trade_paths_frame(df: pl.DataFrame) -> None:
    """
    Validate trade_paths frames before writing.

    Enforces:
      1) All required columns exist (per TRADE_PATHS_SCHEMA).
      2) Partition keys stable dtypes: run_id Utf8, instrument Utf8, dt Date.
      3) dt non-null.
      4) Phase B sanity: paradigm_id implies principle_id.
    """
    if df is None or df.is_empty():
        return

    missing = [c for c in TRADE_PATHS_SCHEMA.columns.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Trade paths frame missing required columns: {missing}")

    schema = df.schema
    expected = {"run_id": pl.Utf8, "instrument": pl.Utf8, "dt": pl.Date}
    for col, dt in expected.items():
        got = schema.get(col)
        if got != dt:
            raise ValueError(f"trade_paths.{col} dtype mismatch: expected={dt} got={got}")

    if df.select(pl.col("dt").is_null().any()).item():
        raise ValueError("trade_paths.dt contains null values")

    if "paradigm_id" in df.columns and "principle_id" not in df.columns:
        raise ValueError("trade_paths has paradigm_id but is missing principle_id (Phase B requires both).")


def write_trade_paths_for_day(
    ctx: RunContext,
    df: pl.DataFrame,
    instrument: str,
    trading_day: date,
    *,
    sandbox: bool = False,
) -> Path:
    """
    Write trade path rows for a given instrument and trading_day.

    Backward compatible:
      - If df lacks paradigm_id -> Phase A layout.
      - If df has paradigm_id + principle_id -> evaluation-aware Phase B layout.

    Writes:
      - One parquet file per (instrument, eval identity group) at 0000.parquet.
    """
    # Return the legacy (non-eval) path as a conventional "base" even in Phase B.
    base_dir = trade_paths_dir(ctx, trading_day, instrument, sandbox=sandbox)

    if df is None or df.is_empty():
        return base_dir

    df2 = ensure_trade_paths_engine_cols(ctx, df, trading_day)
    validate_trade_paths_frame(df2)

    eval_mode = ("paradigm_id" in df2.columns) and ("principle_id" in df2.columns)

    group_cols = ["instrument"]
    if eval_mode:
        group_cols += ["paradigm_id", "principle_id", "candidate_id"]
        if "experiment_id" in df2.columns:
            group_cols.append("experiment_id")

    for g in df2.partition_by(group_cols, maintain_order=True):
        if g.is_empty():
            continue

        inst = str(g["instrument"][0])

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

        out_dir = trade_paths_dir(
            ctx=ctx,
            trading_day=trading_day,
            instrument=inst,
            paradigm_id=paradigm_id,
            principle_id=principle_id,
            candidate_id=candidate_id,
            experiment_id=experiment_id,
            sandbox=sandbox,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(g, out_dir, file_name="0000.parquet")

    return base_dir


def read_trade_paths_for_day(
    ctx: RunContext,
    instrument: str,
    trading_day: date,
    *,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> pl.DataFrame:
    """
    Read trade paths for a given instrument and trading_day, optionally scoped to evaluation identity.

    Returns empty DataFrame if no directory exists.
    """
    dir_path = trade_paths_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
        sandbox=sandbox,
    )
    if not os.path.isdir(dir_path):
        return pl.DataFrame()
    return read_parquet_dir(dir_path)


def read_trade_paths_for_day_any(
    ctx: RunContext,
    instrument: str,
    trading_day: date,
    *,
    sandbox: bool = False,
) -> pl.DataFrame:
    """
    Read trade_paths for (instrument, trading_day) across BOTH layouts:

      - Phase A: data/trade_paths/run_id=.../instrument=.../dt=...
      - Phase B: data/trade_paths/run_id=.../paradigm_id=.../principle_id=.../candidate_id=.../(experiment_id=...)/
                 instrument=.../dt=...

    Intended for research tooling / audits, not deterministic execution.
    """
    legacy_dir = trade_paths_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        paradigm_id=None,
        sandbox=sandbox,
    )

    frames: list[pl.DataFrame] = []
    if os.path.isdir(legacy_dir):
        frames.append(read_parquet_dir(legacy_dir))

    root = legacy_dir.parents[1]  # .../trade_paths/run_id=<RUN_ID>
    if os.path.isdir(root):
        tail = f"instrument={instrument}{os.sep}dt={trading_day.strftime('%Y-%m-%d')}"
        for dirpath, _dirnames, _filenames in os.walk(root):
            if dirpath.endswith(tail):
                try:
                    frames.append(read_parquet_dir(Path(dirpath)))
                except FileNotFoundError:
                    pass

    if not frames:
        return pl.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="diagonal")
