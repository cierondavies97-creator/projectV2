from __future__ import annotations

from datetime import date

import polars as pl

from engine.core.ids import RunContext
from engine.core.schema import enforce_table
from engine.io.parquet_io import write_parquet
from engine.io.paths import trade_clusters_dir

_SENTINEL = "∅"


def build_trade_clusters_from_trade_paths(
    trade_paths: pl.DataFrame,
    *,
    cluster_id: str,
    trading_day: date,
) -> pl.DataFrame:
    """
    Phase B v0 trade_clusters (dt is canonical; no trading_day stamping).
    """
    if trade_paths is None or trade_paths.is_empty():
        return pl.DataFrame()

    df = trade_paths

    cluster_key = None
    for c in ["corr_cluster_id_entry", "corr_cluster_id", "instrument_cluster_id", "cluster_id"]:
        if c in df.columns:
            cluster_key = c
            break

    group_cols: list[str] = []
    for c in ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]:
        if c in df.columns:
            group_cols.append(c)

    if cluster_key is not None:
        group_cols.append(cluster_key)

    if not group_cols:
        out = pl.DataFrame({"trade_count": [df.height]})
    else:
        out = df.group_by(group_cols, maintain_order=True).agg(pl.len().alias("trade_count"))

    out = out.with_columns(
        pl.lit(trading_day).cast(pl.Date).alias("dt"),
        pl.lit(cluster_id).cast(pl.Utf8).alias("cluster_id"),
    )
    return out


def write_trade_clusters_for_cluster_day(
    *,
    ctx: RunContext,
    df: pl.DataFrame,
    cluster_id: str,
    trading_day: date,
    sandbox: bool = False,
) -> None:
    if df is None or df.is_empty():
        return

    out = df
    td = pl.lit(trading_day).cast(pl.Date)

    if "snapshot_id" not in out.columns:
        out = out.with_columns(pl.lit(ctx.snapshot_id).cast(pl.Utf8).alias("snapshot_id"))
    if "run_id" not in out.columns:
        out = out.with_columns(pl.lit(ctx.run_id).cast(pl.Utf8).alias("run_id"))
    if "mode" not in out.columns:
        out = out.with_columns(pl.lit(ctx.mode).cast(pl.Utf8).alias("mode"))

    if "dt" not in out.columns:
        if "trading_day" in out.columns:
            out = out.with_columns(pl.col("trading_day").cast(pl.Date, strict=False).fill_null(td).alias("dt"))
        else:
            out = out.with_columns(td.alias("dt"))
    else:
        out = out.with_columns(pl.col("dt").cast(pl.Date, strict=False).fill_null(td).alias("dt"))

    if "cluster_id" not in out.columns:
        out = out.with_columns(pl.lit(cluster_id).cast(pl.Utf8).alias("cluster_id"))
    else:
        out = out.with_columns(pl.col("cluster_id").cast(pl.Utf8, strict=False).alias("cluster_id"))

    eval_mode = "paradigm_id" in out.columns
    if eval_mode:
        if "principle_id" not in out.columns:
            raise ValueError("trade_clusters: eval-mode requires principle_id when paradigm_id is present")

        out = out.with_columns(
            pl.col("paradigm_id").cast(pl.Utf8, strict=False).alias("paradigm_id"),
            pl.col("principle_id").cast(pl.Utf8, strict=False).alias("principle_id"),
        )

        if "candidate_id" not in out.columns:
            out = out.with_columns(pl.lit(_SENTINEL).cast(pl.Utf8).alias("candidate_id"))
        else:
            out = out.with_columns(pl.col("candidate_id").cast(pl.Utf8, strict=False).fill_null(_SENTINEL).alias("candidate_id"))

        if "experiment_id" in out.columns:
            out = out.with_columns(pl.col("experiment_id").cast(pl.Utf8, strict=False).fill_null(_SENTINEL).alias("experiment_id"))

        eval_cols = ["paradigm_id", "principle_id", "candidate_id"]
        if "experiment_id" in out.columns:
            eval_cols.append("experiment_id")

        for part in out.partition_by(eval_cols, maintain_order=True):
            paradigm_id = str(part["paradigm_id"][0])
            principle_id = str(part["principle_id"][0])
            candidate_id = str(part["candidate_id"][0])
            experiment_id = str(part["experiment_id"][0]) if "experiment_id" in part.columns else None

            out_dir = trade_clusters_dir(
                ctx,
                trading_day,
                cluster_id=cluster_id,
                paradigm_id=paradigm_id,
                principle_id=principle_id,
                candidate_id=candidate_id,
                experiment_id=experiment_id,
                sandbox=sandbox,
            )
            part2 = enforce_table(part, "trade_clusters", allow_extra=True, reorder=False)
            write_parquet(part2, out_dir, file_name="0000.parquet")
        return

    out_dir = trade_clusters_dir(ctx, trading_day, cluster_id=cluster_id, sandbox=sandbox)
    out2 = enforce_table(out, "trade_clusters", allow_extra=True, reorder=False)
    write_parquet(out2, out_dir, file_name="0000.parquet")