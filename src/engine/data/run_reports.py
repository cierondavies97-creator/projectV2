from __future__ import annotations

import logging
from datetime import date

import polars as pl

from engine.core.ids import RunContext
from engine.core.schema import enforce_table
from engine.io.parquet_io import read_parquet_dir, write_parquet
from engine.io.paths import run_reports_dir

log = logging.getLogger(__name__)


def _ensure_identity_columns(ctx: RunContext, trading_day: date, cluster_id: str, df: pl.DataFrame) -> pl.DataFrame:
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

    if "cluster_id" not in out.columns:
        exprs.append(pl.lit(cluster_id).cast(pl.Utf8).alias("cluster_id"))
    else:
        exprs.append(pl.col("cluster_id").cast(pl.Utf8, strict=False).alias("cluster_id"))

    if exprs:
        out = out.with_columns(exprs)

    return out


def write_run_reports_for_cluster_day(
    *,
    ctx: RunContext,
    trading_day: date,
    cluster_id: str,
    reports_df: pl.DataFrame,
    file_name: str = "0000.parquet",
    sandbox: bool = False,
) -> None:
    if reports_df is None or reports_df.is_empty():
        return

    df = _ensure_identity_columns(ctx, trading_day, cluster_id, reports_df)

    out_dir = run_reports_dir(ctx=ctx, trading_day=trading_day, cluster_id=cluster_id, sandbox=sandbox)
    df2 = enforce_table(df, "reports", allow_extra=True, reorder=False)
    write_parquet(df2, out_dir, file_name=file_name)

    log.info("run_reports.write_run_reports_for_cluster_day: wrote %d rows to %s", df2.height, out_dir / file_name)


def read_run_reports_for_cluster_day(
    *,
    ctx: RunContext,
    trading_day: date,
    cluster_id: str,
    sandbox: bool = False,
) -> pl.DataFrame:
    dir_path = run_reports_dir(ctx=ctx, trading_day=trading_day, cluster_id=cluster_id, sandbox=sandbox)
    if not dir_path.is_dir():
        return pl.DataFrame()
    return read_parquet_dir(dir_path)