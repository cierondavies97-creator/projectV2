from __future__ import annotations

import logging
from collections.abc import Iterable

import polars as pl

from engine.core.config_models import ClusterPlan, build_cluster_plan, load_retail_config
from engine.core.schema import WINDOWS_SCHEMA
from engine.core.timegrid import validate_anchor_grid
from engine.data.windows import write_windows_for_instrument_tf_day
from engine.microbatch.types import BatchState
from engine.research.snapshots import load_snapshot_manifest

logger = logging.getLogger(__name__)

_ZONE_CONTEXT_COLS = [
    "zone_behaviour_type_bucket",
    "zone_freshness_bucket",
    "zone_stack_depth_bucket",
    "zone_htf_confluence_bucket",
    "zone_vp_type_bucket",
]


def _cluster_plan_for_state(state: BatchState) -> ClusterPlan:
    snapshot = load_snapshot_manifest(state.ctx.snapshot_id)
    retail = load_retail_config()
    plan = build_cluster_plan(snapshot, retail, state.key.cluster_id)

    logger.info(
        "windows_step: cluster plan resolved cluster_id=%s instruments=%s anchor_tfs=%s tf_entries=%s",
        plan.cluster_id,
        plan.instruments,
        plan.anchor_tfs,
        getattr(plan, "tf_entries", getattr(plan, "entry_tfs", [])),
    )
    return plan


def _dtype_for_schema_type(t: str) -> pl.DataType:
    t = (t or "").lower()
    if t == "string":
        return pl.Utf8
    if t in ("int", "int64"):
        return pl.Int64
    if t in ("double", "float", "float64"):
        return pl.Float64
    if t == "boolean":
        return pl.Boolean
    if t == "timestamp":
        return pl.Datetime("us")
    if t == "date":
        return pl.Date
    # Conservative fallback
    return pl.Utf8


def _empty_frame_from_schema() -> pl.DataFrame:
    cols: dict[str, pl.Series] = {}
    for name, typ in WINDOWS_SCHEMA.columns.items():
        cols[name] = pl.Series(name, [], dtype=_dtype_for_schema_type(typ))
    return pl.DataFrame(cols)


def _align_to_windows_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure the frame has all columns from WINDOWS_SCHEMA and casts them.
    Missing columns are created as NULL (never fake defaults).
    """
    if df is None:
        return _empty_frame_from_schema()

    out = df
    exprs: list[pl.Expr] = []

    for name, typ in WINDOWS_SCHEMA.columns.items():
        dtype = _dtype_for_schema_type(typ)
        if name not in out.columns:
            exprs.append(pl.lit(None).cast(dtype).alias(name))
        else:
            exprs.append(pl.col(name).cast(dtype, strict=False).alias(name))

    return out.with_columns(exprs) if exprs else out


def _tf_entries_from_plan(plan: ClusterPlan) -> list[str]:
    tfs = getattr(plan, "tf_entries", None)
    if not tfs:
        tfs = getattr(plan, "entry_tfs", None)
    if not tfs:
        # Phase A default; Phase B expects this to come from snapshot.data_slice.tf_entries
        tfs = ["M1"]
    return [str(x) for x in tfs]


def _infer_instrument_col(df: pl.DataFrame) -> str:
    if "instrument" in df.columns:
        return "instrument"
    if "instrument_id" in df.columns:
        return "instrument_id"
    raise ValueError("Cannot infer instrument column (expected 'instrument' or 'instrument_id').")


def _infer_ts_col(df: pl.DataFrame) -> str:
    if "ts" in df.columns:
        return "ts"
    if "timestamp" in df.columns:
        return "timestamp"
    raise ValueError("Cannot infer timestamp column (expected 'ts' or 'timestamp').")


def _infer_tf_col(df: pl.DataFrame) -> str | None:
    if "tf" in df.columns:
        return "tf"
    if "anchor_tf" in df.columns:
        return "anchor_tf"
    return None


def _base_windows_from_candles(
    candles: pl.DataFrame,
    *,
    anchor_tf: str,
) -> pl.DataFrame:
    """
    Build 1 row per (instrument, anchor_tf, anchor_ts) from candles.
    candles are expected to already be sliced to the microbatch day.
    """
    if candles is None or candles.is_empty():
        return pl.DataFrame()

    inst_col = _infer_instrument_col(candles)
    ts_col = _infer_ts_col(candles)
    tf_col = _infer_tf_col(candles)

    d = candles
    if tf_col is not None:
        d = d.filter(pl.col(tf_col) == pl.lit(anchor_tf))
        if d.is_empty():
            return pl.DataFrame()

    # Validate timestamps lie on the anchor grid (do not silently truncate).
    chk = validate_anchor_grid(d, anchor_tf=anchor_tf, ts_col=ts_col)
    if chk.bad_rows > 0:
        raise ValueError(
            f"windows_step: candles not on anchor grid anchor_tf={anchor_tf} bad_rows={chk.bad_rows} "
            f"(sample below)\n{chk.sample_bad}"
        )

    base = (
        d.select(
            pl.col(inst_col).cast(pl.Utf8, strict=False).alias("instrument"),
            pl.lit(anchor_tf).cast(pl.Utf8).alias("anchor_tf"),
            pl.col(ts_col).cast(pl.Datetime("us"), strict=False).alias("anchor_ts"),
        )
        .unique(subset=["instrument", "anchor_tf", "anchor_ts"])
        .sort(["instrument", "anchor_tf", "anchor_ts"])
    )
    return base


def _base_windows_from_features(features: pl.DataFrame, *, anchor_tf: str) -> pl.DataFrame:
    """
    Fallback base grid if candles are unavailable: derive windows from feature timestamps.
    """
    if features is None or features.is_empty():
        return pl.DataFrame()

    required = {"instrument", "anchor_tf", "ts"}
    if not required.issubset(set(features.columns)):
        return pl.DataFrame()

    d = features.filter(pl.col("anchor_tf") == pl.lit(anchor_tf))
    if d.is_empty():
        return pl.DataFrame()

    chk = validate_anchor_grid(d, anchor_tf=anchor_tf, ts_col="ts")
    if chk.bad_rows > 0:
        raise ValueError(
            f"windows_step: features not on anchor grid anchor_tf={anchor_tf} bad_rows={chk.bad_rows} "
            f"(sample below)\n{chk.sample_bad}"
        )

    base = (
        d.select(
            pl.col("instrument").cast(pl.Utf8, strict=False).alias("instrument"),
            pl.col("anchor_tf").cast(pl.Utf8, strict=False).alias("anchor_tf"),
            pl.col("ts").cast(pl.Datetime("us"), strict=False).alias("anchor_ts"),
        )
        .unique(subset=["instrument", "anchor_tf", "anchor_ts"])
        .sort(["instrument", "anchor_tf", "anchor_ts"])
    )
    return base


def _explode_tf_entries(base: pl.DataFrame, tf_entries: Iterable[str]) -> pl.DataFrame:
    if base is None or base.is_empty():
        return base
    tf_df = pl.DataFrame({"tf_entry": list(tf_entries)}).with_columns(pl.col("tf_entry").cast(pl.Utf8, strict=False))
    return base.join(tf_df, how="cross")


def _reduce_features_to_bar(features: pl.DataFrame) -> pl.DataFrame:
    """
    Reduce features to one row per (instrument, anchor_tf, anchor_ts).
    If duplicates exist, we take the first per column deterministically after sorting.
    """
    if features is None or features.is_empty():
        return pl.DataFrame()

    required = {"instrument", "anchor_tf", "ts"}
    if not required.issubset(set(features.columns)):
        return pl.DataFrame()

    d = (
        features.with_columns(
            pl.col("instrument").cast(pl.Utf8, strict=False),
            pl.col("anchor_tf").cast(pl.Utf8, strict=False),
            pl.col("ts").cast(pl.Datetime("us"), strict=False),
        )
        .sort(["instrument", "anchor_tf", "ts"])
        .rename({"ts": "anchor_ts"})
    )

    keys = ["instrument", "anchor_tf", "anchor_ts"]
    non_keys = [c for c in d.columns if c not in keys]
    if not non_keys:
        return d.unique(subset=keys).sort(keys)

    d2 = d.group_by(keys).agg([pl.first(c).alias(c) for c in non_keys]).sort(keys)
    return d2


def _zones_context_from_zones_state(zones_state: pl.DataFrame) -> pl.DataFrame:
    """
    Reduce zones_state to 1 row per (instrument, anchor_tf, anchor_ts).
    Prevents row explosion.
    """
    if zones_state is None or zones_state.is_empty():
        return pl.DataFrame()

    zone_cols_present = [c for c in _ZONE_CONTEXT_COLS if c in zones_state.columns]
    if not zone_cols_present:
        return pl.DataFrame()

    if not {"instrument", "anchor_tf", "ts"}.issubset(set(zones_state.columns)):
        return pl.DataFrame()

    zs = (
        zones_state.select(["instrument", "anchor_tf", "ts", *zone_cols_present])
        .with_columns(
            pl.col("instrument").cast(pl.Utf8, strict=False),
            pl.col("anchor_tf").cast(pl.Utf8, strict=False),
            pl.col("ts").cast(pl.Datetime("us"), strict=False),
            *[pl.col(c).cast(pl.Utf8, strict=False) for c in zone_cols_present],
        )
        .sort(["instrument", "anchor_tf", "ts"])
        .group_by(["instrument", "anchor_tf", "ts"])
        .agg([pl.first(c).alias(c) for c in zone_cols_present])
        .rename({"ts": "anchor_ts"})
        .sort(["instrument", "anchor_tf", "anchor_ts"])
    )
    return zs


def run(state: BatchState) -> BatchState:
    plan = _cluster_plan_for_state(state)
    td = state.key.trading_day

    candles = state.get_optional("candles")
    features = state.get_optional("features")
    zones_state = state.get_optional("zones_state")

    tf_entries = _tf_entries_from_plan(plan)

    windows_parts: list[pl.DataFrame] = []

    for anchor_tf in plan.anchor_tfs:
        base = _base_windows_from_candles(candles, anchor_tf=str(anchor_tf))
        if base.is_empty():
            base = _base_windows_from_features(features, anchor_tf=str(anchor_tf))
        if base.is_empty():
            continue

        base = _explode_tf_entries(base, tf_entries=tf_entries)
        windows_parts.append(base)

    if not windows_parts:
        windows = _empty_frame_from_schema()
        state.set("windows", windows)
        logger.warning("windows_step: no windows produced (candles/features empty); wrote empty windows.")
        return state

    windows = pl.concat(windows_parts, how="vertical", rechunk=True)

    # Join features (bar-level) if present; will duplicate across tf_entry (intended).
    feat_bar = _reduce_features_to_bar(features) if features is not None else pl.DataFrame()
    if feat_bar is not None and not feat_bar.is_empty():
        windows = windows.join(feat_bar, on=["instrument", "anchor_tf", "anchor_ts"], how="left")

    # Join zones context if present.
    zs_bar = _zones_context_from_zones_state(zones_state) if zones_state is not None else pl.DataFrame()
    if zs_bar is not None and not zs_bar.is_empty():
        windows = windows.join(zs_bar, on=["instrument", "anchor_tf", "anchor_ts"], how="left")

    # Stamp audit dt (not a partition dependency, but useful to keep in-row).
    windows = windows.with_columns(pl.lit(td).cast(pl.Date).alias("dt"))

    # Align to schema without fabricating values.
    windows = _align_to_windows_schema(windows)

    state.set("windows", windows)

    if not windows.is_empty():
        for (instrument, anchor_tf), df_grp in windows.group_by(["instrument", "anchor_tf"], maintain_order=True):
            write_windows_for_instrument_tf_day(
                ctx=state.ctx,
                df=df_grp,
                instrument=str(instrument),
                anchor_tf=str(anchor_tf),
                trading_day=td,
                sandbox=False,
            )

    logger.info(
        "windows_step: built %d rows instruments=%s anchor_tfs=%s tf_entries=%s",
        windows.height,
        windows.select("instrument").unique().to_series().to_list() if "instrument" in windows.columns else [],
        windows.select("anchor_tf").unique().to_series().to_list() if "anchor_tf" in windows.columns else [],
        tf_entries,
    )
    return state
