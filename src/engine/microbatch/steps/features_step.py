from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from engine.core.config_models import (
    ClusterPlan,
    FeatureFamilyRegistryEntry,
    build_cluster_plan,
    families_for_table,
    load_features_registry,
    load_retail_config,
)
from engine.core.timegrid import validate_anchor_grid
from engine.data.features import write_features_for_instrument_tf_day
from engine.data.pcra import write_pcra_for_instrument_tf_day
from engine.data.zones_state import write_zones_state_for_instrument_tf_day
from engine.features import FeatureBuildContext
from engine.microbatch.types import BatchState
from engine.research.snapshots import load_snapshot_manifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy / invariants
# ---------------------------------------------------------------------------

_TS_DTYPE = pl.Datetime("us")


@dataclass(frozen=True)
class _TableSpec:
    table_name: str
    key_cols: list[str]
    ts_col: str | None  # used for day filtering + dtype normalization
    anchor_tf_col: str | None  # used for anchor grid validation when applicable


FEATURES_SPEC = _TableSpec(
    table_name="data/features",
    key_cols=["instrument", "anchor_tf", "ts"],
    ts_col="ts",
    anchor_tf_col="anchor_tf",
)

ZONES_STATE_SPEC = _TableSpec(
    table_name="data/zones_state",
    key_cols=["instrument", "anchor_tf", "zone_id", "ts"],
    ts_col="ts",
    anchor_tf_col="anchor_tf",
)

PCRA_SPEC = _TableSpec(
    table_name="data/pcr_a",
    key_cols=["instrument", "anchor_tf", "pcr_window_ts"],
    ts_col="pcr_window_ts",
    anchor_tf_col="anchor_tf",
)


# ---------------------------------------------------------------------------
# Cluster plan
# ---------------------------------------------------------------------------

def _cluster_plan_for_state(state: BatchState) -> ClusterPlan:
    """
    Resolve the ClusterPlan for this (RunContext, MicrobatchKey).

    Uses:
      - snapshots/<snapshot_id>.json   -> SnapshotManifest
      - conf/retail.yaml               -> RetailConfig
    """
    snapshot = load_snapshot_manifest(state.ctx.snapshot_id)
    retail = load_retail_config()
    plan = build_cluster_plan(snapshot, retail, state.key.cluster_id)

    logger.info(
        "features_step: cluster plan resolved cluster_id=%s instruments=%s anchor_tfs=%s entry_tfs=%s",
        plan.cluster_id,
        plan.instruments,
        getattr(plan, "anchor_tfs", []),
        getattr(plan, "entry_tfs", []),
    )
    return plan


# ---------------------------------------------------------------------------
# Feature module calling (compat)
# ---------------------------------------------------------------------------

def _import_feature_module(module_name: str):
    return importlib.import_module(f"engine.features.{module_name}")


def _family_cfg_of(family: FeatureFamilyRegistryEntry) -> Mapping[str, Any] | None:
    for attr in ("family_cfg", "cfg", "config"):
        v = getattr(family, attr, None)
        if isinstance(v, Mapping):
            return v
    return None


def _registry_entry_of(family: FeatureFamilyRegistryEntry) -> Mapping[str, Any] | None:
    for attr in ("registry_entry", "raw", "raw_entry", "entry"):
        v = getattr(family, attr, None)
        if isinstance(v, Mapping):
            return v
    return None


def _call_feature_builder(mod, **kwargs) -> pl.DataFrame:
    fn = getattr(mod, "build_feature_frame", None)
    if fn is None:
        raise AttributeError(f"{mod.__name__} does not define build_feature_frame(...)")

    sig = inspect.signature(fn)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        out = fn(**kwargs)
    else:
        filtered = {k: v for k, v in kwargs.items() if k in params}
        out = fn(**filtered)

    if not isinstance(out, pl.DataFrame):
        raise TypeError(f"Feature module {mod.__name__}.build_feature_frame must return a polars.DataFrame")
    return out


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _cast_key_columns(df: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    if df.is_empty():
        return df
    out = df
    for c in key_cols:
        if c not in out.columns:
            continue
        if c in ("instrument", "anchor_tf", "zone_id"):
            out = out.with_columns(pl.col(c).cast(pl.Utf8, strict=False))
        elif c in ("ts", "pcr_window_ts"):
            out = out.with_columns(pl.col(c).cast(_TS_DTYPE, strict=False))
    return out


def _filter_to_trading_day(df: pl.DataFrame, *, ts_col: str, trading_day) -> pl.DataFrame:
    if df.is_empty():
        return df
    if ts_col not in df.columns:
        return df
    return df.filter(pl.col(ts_col).dt.date() == pl.lit(trading_day))


def _validate_anchor_grid_if_applicable(df: pl.DataFrame, *, anchor_tf_col: str, ts_col: str) -> None:
    """
    Only applies to tables keyed by (instrument, anchor_tf, ts) and friends.
    """
    if df.is_empty():
        return
    if anchor_tf_col not in df.columns or ts_col not in df.columns:
        return
    # validate_anchor_grid expects a single anchor_tf value at a time,
    # so we validate per anchor_tf.
    for (anchor_tf,), grp in df.group_by([anchor_tf_col], maintain_order=True):
        chk = validate_anchor_grid(grp, anchor_tf=str(anchor_tf), ts_col=ts_col)
        if chk.bad_rows > 0:
            raise ValueError(
                f"features_step: rows not on anchor grid anchor_tf={anchor_tf} ts_col={ts_col} "
                f"bad_rows={chk.bad_rows}\n{chk.sample_bad}"
            )


def _dedupe_on_keys(df: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    """
    Deterministic de-dupe: sort by keys, keep first.
    """
    if df.is_empty():
        return df
    out = df.sort(key_cols)
    return out.unique(subset=key_cols, keep="first")


def _assert_no_nonkey_collisions(frames: list[pl.DataFrame], key_cols: list[str]) -> None:
    """
    Enforce a clean contract: two families must not emit the same non-key column name.
    This prevents silent suffixing and ambiguous meaning.
    """
    seen: set[str] = set()
    for df in frames:
        for c in df.columns:
            if c in key_cols:
                continue
            if c in seen:
                raise ValueError(
                    f"features_step: feature column collision on '{c}'. "
                    f"Two families emitted the same column name. "
                    f"Resolve by namespacing (e.g., '{c}' -> '<family_id>__{c}') or by designating a single owner."
                )
            seen.add(c)


def _cleanup_key_duplicates(features: pl.DataFrame) -> pl.DataFrame:
    """
    Coalesce and drop *_right duplicates introduced by outer joins for canonical keys only.
    """
    if features.is_empty():
        return features

    key_pairs = [
        ("instrument", "instrument_right"),
        ("anchor_tf", "anchor_tf_right"),
        ("ts", "ts_right"),
    ]
    out = features
    for left, right in key_pairs:
        if right in out.columns and left in out.columns:
            out = out.with_columns(
                pl.when(pl.col(left).is_null()).then(pl.col(right)).otherwise(pl.col(left)).alias(left)
            ).drop(right)
        elif right in out.columns:
            out = out.drop(right)
    return out


def _reorder_columns(df: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    if df.is_empty():
        return df
    keys = [c for c in key_cols if c in df.columns]
    rest = [c for c in df.columns if c not in keys]
    rest_sorted = sorted(rest)
    return df.select(keys + rest_sorted)


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _build_table_for_cluster(
    *,
    state: BatchState,
    plan: ClusterPlan,
    families: Sequence[FeatureFamilyRegistryEntry],
    candles: pl.DataFrame,
    ticks: pl.DataFrame | None,
    macro: pl.DataFrame | None,
    external: pl.DataFrame | None,
    spec: _TableSpec,
) -> pl.DataFrame:
    """
    Build one physical table by running all feature families targeting it,
    then outer-joining on spec.key_cols.
    """
    if not families:
        logger.info("features_step: no families active for table=%s", spec.table_name)
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []

    for family in families:
        module_name = family.family_id
        mod = _import_feature_module(module_name)

        ctx = FeatureBuildContext(
            run_ctx=state.ctx,
            cluster=plan,
            trading_day=state.key.trading_day,
            family_ids=[family.family_id],
        )

        df = _call_feature_builder(
            mod,
            ctx=ctx,
            candles=candles,
            ticks=ticks,
            macro=macro,
            external=external,
            external_df=external,  # tolerated alias
            family_cfg=_family_cfg_of(family),
            registry_entry=_registry_entry_of(family),
        )

        if df.is_empty():
            logger.info("features_step: family '%s' produced empty frame; skipping", family.family_id)
            continue

        from engine.features._shared import conform_to_registry

        df = conform_to_registry(

            df,

            registry_entry=_registry_entry_of(family),

            key_cols=spec.key_cols,

            where=f"features_step:{family.family_id}",

            allow_extra=False,

        )


        missing_keys = [k for k in spec.key_cols if k not in df.columns]
        if missing_keys:
            raise ValueError(
                f"Feature family '{family.family_id}' did not produce required key "
                f"columns {missing_keys} for table {spec.table_name} keys={spec.key_cols}"
            )

        # Key dtype normalization + strict day scoping
        df = _cast_key_columns(df, spec.key_cols)
        if spec.ts_col is not None:
            df = _filter_to_trading_day(df, ts_col=spec.ts_col, trading_day=state.key.trading_day)

        # De-dupe per family (deterministic)
        df = _dedupe_on_keys(df, spec.key_cols)

        frames.append(df)

    if not frames:
        logger.info("features_step: all families for table=%s returned empty frames", spec.table_name)
        return pl.DataFrame()

    # Collision policy across families (critical for research correctness)
    _assert_no_nonkey_collisions(frames, spec.key_cols)

    result = frames[0]
    for df in frames[1:]:
        if result.is_empty():
            result = df
            continue
        if df.is_empty():
            continue

        result = result.join(df, on=spec.key_cols, how="outer")
        # Coalesce canonical keys only
        if set(["instrument", "anchor_tf", "ts"]).issubset(set(spec.key_cols)):
            result = _cleanup_key_duplicates(result)

    # Final normalization
    result = _cast_key_columns(result, spec.key_cols)
    if spec.ts_col is not None:
        result = _filter_to_trading_day(result, ts_col=spec.ts_col, trading_day=state.key.trading_day)
    result = _dedupe_on_keys(result, spec.key_cols)

    # Validate anchor grid when applicable
    if spec.anchor_tf_col and spec.ts_col and spec.anchor_tf_col in result.columns and spec.ts_col in result.columns:
        _validate_anchor_grid_if_applicable(result, anchor_tf_col=spec.anchor_tf_col, ts_col=spec.ts_col)

    result = _reorder_columns(result, spec.key_cols)

    logger.info(
        "features_step: built table=%s rows=%d cols=%d keys=%s",
        spec.table_name,
        result.height,
        len(result.columns),
        spec.key_cols,
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(state: BatchState) -> BatchState:
    """
    Features step.

    Inputs in BatchState:
      - 'candles'
      - 'ticks' (optional)
      - 'macro' (optional)
      - 'external' (optional)

    Outputs in BatchState:
      - 'features'
      - 'zones_state'
      - 'pcr_a'
    """
    plan = _cluster_plan_for_state(state)

    candles = state.get("candles")
    ticks = state.get_optional("ticks")
    macro = state.get_optional("macro")
    external = state.get_optional("external")

    # Load registry once
    registry = load_features_registry()

    features_families = families_for_table(registry, table=FEATURES_SPEC.table_name)
    zones_families = families_for_table(registry, table=ZONES_STATE_SPEC.table_name)
    pcra_families = families_for_table(registry, table=PCRA_SPEC.table_name)

    features_df = _build_table_for_cluster(
        state=state,
        plan=plan,
        families=features_families,
        candles=candles,
        ticks=ticks,
        macro=macro,
        external=external,
        spec=FEATURES_SPEC,
    )
    zones_state_df = _build_table_for_cluster(
        state=state,
        plan=plan,
        families=zones_families,
        candles=candles,
        ticks=ticks,
        macro=macro,
        external=external,
        spec=ZONES_STATE_SPEC,
    )
    pcra_df = _build_table_for_cluster(
        state=state,
        plan=plan,
        families=pcra_families,
        candles=candles,
        ticks=ticks,
        macro=macro,
        external=external,
        spec=PCRA_SPEC,
    )

    state.set("features", features_df)
    state.set("zones_state", zones_state_df)
    state.set("pcr_a", pcra_df)

    logger.info(
        "features_step.run: features_rows=%d zones_state_rows=%d pcra_rows=%d",
        features_df.height if features_df is not None else 0,
        zones_state_df.height if zones_state_df is not None else 0,
        pcra_df.height if pcra_df is not None else 0,
    )

    # Persist: data/features partitioned by instrument/anchor_tf/day
    if features_df is not None and not features_df.is_empty():
        for (instrument, anchor_tf), grp in features_df.group_by(["instrument", "anchor_tf"], maintain_order=True):
            write_features_for_instrument_tf_day(
                ctx=state.ctx,
                df=grp,
                instrument=str(instrument),
                anchor_tf=str(anchor_tf),
                trading_day=state.key.trading_day,
                sandbox=False,
            )
    else:
        # Materialize empty contract files for reproducibility (missing-data/debug visibility).
        for instrument in plan.instruments:
            for anchor_tf in plan.anchor_tfs:
                write_features_for_instrument_tf_day(
                    ctx=state.ctx,
                    df=pl.DataFrame(),
                    instrument=str(instrument),
                    anchor_tf=str(anchor_tf),
                    trading_day=state.key.trading_day,
                    sandbox=False,
                )

# Persist: data/zones_state
    if zones_state_df is not None and not zones_state_df.is_empty():
        for (instrument, anchor_tf), grp in zones_state_df.group_by(["instrument", "anchor_tf"], maintain_order=True):
            write_zones_state_for_instrument_tf_day(
                ctx=state.ctx,
                df=grp,
                instrument=str(instrument),
                anchor_tf=str(anchor_tf),
                trading_day=state.key.trading_day,
                sandbox=False,
            )
    else:
        # Materialize empty contract files (zone discovery may legitimately yield no rows).
        for instrument in plan.instruments:
            for anchor_tf in plan.anchor_tfs:
                write_zones_state_for_instrument_tf_day(
                    ctx=state.ctx,
                    df=pl.DataFrame(),
                    instrument=str(instrument),
                    anchor_tf=str(anchor_tf),
                    trading_day=state.key.trading_day,
                    sandbox=False,
                )

# Persist: data/pcr_a
    if pcra_df is not None and not pcra_df.is_empty():
        for (instrument, anchor_tf), grp in pcra_df.group_by(["instrument", "anchor_tf"], maintain_order=True):
            write_pcra_for_instrument_tf_day(
                ctx=state.ctx,
                df=grp,
                instrument=str(instrument),
                anchor_tf=str(anchor_tf),
                trading_day=state.key.trading_day,
                sandbox=False,
            )
    else:
        # Materialize empty contract files (if tick/footprint inputs are missing).
        for instrument in plan.instruments:
            for anchor_tf in plan.anchor_tfs:
                write_pcra_for_instrument_tf_day(
                    ctx=state.ctx,
                    df=pl.DataFrame(),
                    instrument=str(instrument),
                    anchor_tf=str(anchor_tf),
                    trading_day=state.key.trading_day,
                    sandbox=False,
                )

    return state
