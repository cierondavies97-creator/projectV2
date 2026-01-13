from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import polars as pl

from engine.core.ids import RunContext
from engine.core.schema import enforce_table
from engine.io.parquet_io import read_parquet_dir, write_parquet
from engine.io.paths import orders_dir

log = logging.getLogger(__name__)

_SENTINEL = ""


def _is_eval_mode(df: pl.DataFrame) -> bool:
    return df is not None and (not df.is_empty()) and ("paradigm_id" in df.columns)


def _ensure_identity_columns(ctx: RunContext, trading_day: date, df: pl.DataFrame) -> pl.DataFrame:
    """
    Phase B:
      - dt is canonical persisted date column.
      - If legacy trading_day exists but dt does not, set dt := trading_day.
      - Do not stamp trading_day.
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

    if _is_eval_mode(out):
        if "principle_id" not in out.columns:
            raise ValueError("orders: eval-mode requires principle_id when paradigm_id is present")

        for c in ("paradigm_id", "principle_id"):
            if c in out.columns:
                out = out.with_columns(pl.col(c).cast(pl.Utf8, strict=False).alias(c))

        if "candidate_id" not in out.columns:
            out = out.with_columns(pl.lit(_SENTINEL).cast(pl.Utf8).alias("candidate_id"))
        else:
            out = out.with_columns(pl.col("candidate_id").cast(pl.Utf8, strict=False).fill_null(_SENTINEL).alias("candidate_id"))

        if "experiment_id" in out.columns:
            out = out.with_columns(pl.col("experiment_id").cast(pl.Utf8, strict=False).fill_null(_SENTINEL).alias("experiment_id"))

    return out


def _validate_orders_frame(df: pl.DataFrame) -> None:
    if df is None or df.is_empty():
        return
    required = ["instrument", "dt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"orders frame missing required columns: {missing}")


def write_orders_for_instrument_day(
    *,
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    orders_df: pl.DataFrame,
    file_name: str = "0000.parquet",
    sandbox: bool = False,
) -> None:
    if orders_df is None or orders_df.is_empty():
        return

    df = _ensure_identity_columns(ctx, trading_day, orders_df)

    if "instrument" not in df.columns:
        df = df.with_columns(pl.lit(instrument).cast(pl.Utf8).alias("instrument"))
    else:
        df = df.with_columns(pl.col("instrument").cast(pl.Utf8, strict=False).alias("instrument"))

    _validate_orders_frame(df)

    eval_mode = _is_eval_mode(df) and ("principle_id" in df.columns)

    group_cols = ["instrument"]
    if eval_mode:
        group_cols += ["paradigm_id", "principle_id", "candidate_id"]
        if "experiment_id" in df.columns:
            group_cols.append("experiment_id")

    parts = df.partition_by(group_cols, maintain_order=True)
    for g in parts:
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

        out_dir = orders_dir(
            ctx=ctx,
            trading_day=trading_day,
            instrument=inst,
            paradigm_id=paradigm_id,
            principle_id=principle_id,
            candidate_id=candidate_id,
            experiment_id=experiment_id,
            sandbox=sandbox,
        )

        g2 = enforce_table(g, "orders", allow_extra=True, reorder=False)
        write_parquet(g2, out_dir, file_name=file_name)

    log.info("orders.write_orders_for_instrument_day: wrote %d rows across %d partition(s) for dt=%s", df.height, len(parts), trading_day)


def read_orders_for_instrument_day(
    *,
    ctx: RunContext,
    trading_day: date,
    instrument: str,
    paradigm_id: Optional[str] = None,
    principle_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    sandbox: bool = False,
) -> pl.DataFrame:
    dir_path = orders_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        paradigm_id=paradigm_id,
        principle_id=principle_id,
        candidate_id=candidate_id,
        experiment_id=experiment_id,
        sandbox=sandbox,
    )

    if not dir_path.is_dir():
        return pl.DataFrame()

    return read_parquet_dir(dir_path)