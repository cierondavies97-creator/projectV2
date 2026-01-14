from __future__ import annotations

import json
import logging
from typing import Any

import polars as pl

from engine.features import FeatureBuildContext

log = logging.getLogger(__name__)


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_id": pl.Series([], dtype=pl.Utf8),
            "path_shape": pl.Series([], dtype=pl.Utf8),
            "path_cluster_id": pl.Series([], dtype=pl.Utf8),
            "path_family_id": pl.Series([], dtype=pl.Utf8),
            "path_filter_primary": pl.Series([], dtype=pl.Utf8),
            "path_filter_tags_json": pl.Series([], dtype=pl.Utf8),
            "time_to_1R_bars": pl.Series([], dtype=pl.Int64),
            "time_to_2R_bars": pl.Series([], dtype=pl.Int64),
            "mae_R": pl.Series([], dtype=pl.Float64),
            "mae_R_bucket": pl.Series([], dtype=pl.Utf8),
            "mfe_R": pl.Series([], dtype=pl.Float64),
            "exit_reason": pl.Series([], dtype=pl.Utf8),
        }
    )


def _parse_edges(edges_json: str | None) -> list[float]:
    """
    Parse JSON array string of numeric edges for MAE bucketing.
    Deterministic fallback if missing/invalid.
    """
    if not edges_json:
        return [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    try:
        v = json.loads(edges_json)
        if isinstance(v, list) and v:
            out = sorted(float(x) for x in v)
            return out
    except Exception:
        pass
    return [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def _bucket_expr(x: pl.Expr, edges: list[float]) -> pl.Expr:
    """
    Bucket label strings like:
      "<=0.25", "(0.25,0.5]", ..., ">2"
    """
    expr = pl.when(x.is_null()).then(pl.lit(None, dtype=pl.Utf8))
    lo = None
    for e in edges:
        if lo is None:
            expr = expr.when(x <= pl.lit(e)).then(pl.lit(f"<= {e:g}"))
        else:
            expr = expr.when((x > pl.lit(lo)) & (x <= pl.lit(e))).then(pl.lit(f"({lo:g},{e:g}]"))
        lo = e
    return expr.otherwise(pl.lit(f"> {edges[-1]:g}"))


def _col_or_null(df: pl.DataFrame, name: str, dtype: pl.DataType) -> pl.Expr:
    return pl.col(name).cast(dtype, strict=False) if name in df.columns else pl.lit(None, dtype=dtype)


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    trade_paths: pl.DataFrame | None = None,
    **_,
) -> pl.DataFrame:
    """
    Table: data/trade_paths
    Keys : trade_id

    Threshold keys (ONLY these are read from ctx.features_auto_cfg["trade_path_class"]):
      - path_cluster_n_clusters
      - path_shape_time_to_1R_bars_max
      - path_mae_R_bucket_edges_json

    Engine-lane determinism:
      - No clustering is computed here. If upstream provides path_cluster_id, we pass it through.
      - Shapes/filters are deterministic heuristics from existing diagnostics.
    """
    if trade_paths is None or trade_paths.is_empty():
        log.warning("trade_path_class: trade_paths empty; returning empty keyed frame")
        return _empty_keyed_frame()

    if "trade_id" not in trade_paths.columns:
        log.warning("trade_path_class: missing trade_id; returning empty keyed frame")
        return _empty_keyed_frame()

    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg: dict[str, Any] = dict(auto_cfg.get("trade_path_class", {}) if isinstance(auto_cfg, dict) else {})

    # registry-governed knobs
    n_clusters_cfg = int(fam_cfg.get("path_cluster_n_clusters", 0))  # not used for clustering in-engine
    t1_max = int(fam_cfg.get("path_shape_time_to_1R_bars_max", 12))
    edges = _parse_edges(fam_cfg.get("path_mae_R_bucket_edges_json"))

    # Normalize + dedupe deterministically
    src = (
        trade_paths.select([pl.all()])
        .with_columns(pl.col("trade_id").cast(pl.Utf8))
        .drop_nulls(["trade_id"])
        .unique(subset=["trade_id"], keep="last")
    )

    # Optional diagnostics (pass-through if already computed upstream)
    t1 = _col_or_null(src, "time_to_1R_bars", pl.Int64).alias("time_to_1R_bars")
    t2 = _col_or_null(src, "time_to_2R_bars", pl.Int64).alias("time_to_2R_bars")
    mae = _col_or_null(src, "mae_R", pl.Float64).alias("mae_R")
    mfe = _col_or_null(src, "mfe_R", pl.Float64).alias("mfe_R")
    exit_reason = _col_or_null(src, "exit_reason", pl.Utf8).alias("exit_reason")

    # Cluster id (pass-through only; remains null until research/backfill)
    path_cluster_id = _col_or_null(src, "path_cluster_id", pl.Utf8).alias("path_cluster_id")

    # Buckets
    mae_bucket = _bucket_expr(pl.col("mae_R"), edges).alias("mae_R_bucket")

    # Deterministic shape heuristics (uses only available diagnostics)
    # Labels are constrained to the set you documented in the registry comment.
    path_shape = (
        pl.when(pl.col("time_to_1R_bars").is_null() & pl.col("mfe_R").is_null() & pl.col("mae_R").is_null())
        .then(pl.lit("unknown"))
        .when(
            (pl.col("time_to_1R_bars").is_not_null())
            & (pl.col("time_to_1R_bars") <= pl.lit(t1_max))
            & (pl.col("mfe_R").is_not_null())
            & (pl.col("mfe_R") >= pl.lit(2.0))
            & (pl.col("mae_R").is_not_null())
            & (pl.col("mae_R") <= pl.lit(0.5))
        )
        .then(pl.lit("straight_runner"))
        .when(
            (pl.col("mfe_R").is_not_null())
            & (pl.col("mfe_R") >= pl.lit(2.0))
            & (pl.col("mae_R").is_not_null())
            & (pl.col("mae_R") > pl.lit(0.5))
        )
        .then(pl.lit("dip_then_go"))
        .when(
            (pl.col("time_to_1R_bars").is_not_null())
            & (pl.col("time_to_1R_bars") > pl.lit(t1_max))
            & (pl.col("mfe_R").is_not_null())
            & (pl.col("mfe_R") >= pl.lit(1.5))
        )
        .then(pl.lit("grind_then_go"))
        .when(
            (pl.col("mfe_R").is_not_null())
            & (pl.col("mfe_R") < pl.lit(1.0))
            & (pl.col("mae_R").is_not_null())
            & (pl.col("mae_R") >= pl.lit(1.0))
        )
        .then(pl.lit("straight_fail"))
        .when(
            (pl.col("time_to_1R_bars").is_not_null())
            & (pl.col("time_to_1R_bars") > pl.lit(t1_max))
            & (pl.col("mfe_R").is_not_null())
            & (pl.col("mfe_R") < pl.lit(1.0))
        )
        .then(pl.lit("chop_and_die"))
        .otherwise(pl.lit("unknown"))
        .alias("path_shape")
    )

    # Coarse family label (stable mapping)
    path_family_id = (
        pl.when(pl.col("path_shape").is_in(["straight_runner", "dip_then_go", "grind_then_go"]))
        .then(pl.lit("winners"))
        .when(pl.col("path_shape").is_in(["straight_fail", "chop_and_die"]))
        .then(pl.lit("losers"))
        .otherwise(pl.lit("unknown"))
        .alias("path_family_id")
    )

    # Primary filter (one canonical tag)
    path_filter_primary = (
        pl.when(pl.col("path_shape") == "straight_runner").then(pl.lit("fast_winner"))
        .when(pl.col("path_shape") == "dip_then_go").then(pl.lit("drawdown_then_win"))
        .when(pl.col("path_shape") == "grind_then_go").then(pl.lit("slow_winner"))
        .when(pl.col("path_shape") == "straight_fail").then(pl.lit("fast_loser"))
        .when(pl.col("path_shape") == "chop_and_die").then(pl.lit("chop_loser"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("path_filter_primary")
    )

    # Tag list -> JSON array string (deterministic, minimal)
    tags_list = pl.concat_list(
        [
            pl.when(pl.col("path_shape") == "straight_runner").then(pl.lit("fast_winner")).otherwise(pl.lit(None)),
            pl.when(pl.col("path_shape") == "dip_then_go").then(pl.lit("drawdown_then_win")).otherwise(pl.lit(None)),
            pl.when(pl.col("path_shape") == "grind_then_go").then(pl.lit("slow_winner")).otherwise(pl.lit(None)),
            pl.when(pl.col("path_shape") == "straight_fail").then(pl.lit("fast_loser")).otherwise(pl.lit(None)),
            pl.when(pl.col("path_shape") == "chop_and_die").then(pl.lit("chop_loser")).otherwise(pl.lit(None)),
            pl.when(pl.col("mae_R").is_not_null() & (pl.col("mae_R") >= pl.lit(1.0))).then(pl.lit("deep_mae")).otherwise(pl.lit(None)),
            pl.when(pl.col("mfe_R").is_not_null() & (pl.col("mfe_R") >= pl.lit(2.0))).then(pl.lit("high_mfe")).otherwise(pl.lit(None)),
        ]
    ).list.drop_nulls()

    # Encode as a JSON array string without non-deterministic ordering
    joined = tags_list.list.join(pl.lit('","')).alias("_tags_joined")
    path_filter_tags_json = (
        pl.when(joined.is_null() | (joined == pl.lit("")))
        .then(pl.lit("[]"))
        .otherwise(pl.lit('["') + joined + pl.lit('"]'))
        .alias("path_filter_tags_json")
    )

    out = (
        src.select(pl.col("trade_id").cast(pl.Utf8))
        .with_columns(
            t1,
            t2,
            mae,
            mfe,
            exit_reason,
            path_cluster_id,
        )
        .with_columns(
            mae_bucket,
            path_shape,
            path_family_id,
            path_filter_primary,
            path_filter_tags_json,
        )
        .select(_empty_keyed_frame().columns)
    )

    log.info(
        "trade_path_class: built rows=%d shapes=%s (cluster_n_cfg=%d; clustering is not computed in-engine)",
        out.height,
        out.select("path_shape").unique().to_series().to_list(),
        n_clusters_cfg,
    )
    return out
