from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import polars as pl
import yaml

# ---------------------------------------------------------------------------
# Snapshot manifest (snapshots/<SNAPSHOT_ID>.json)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotConfigRefs:
    retail_config_path: str
    features_registry_path: str
    features_auto_path: str
    rails_auto_path: str
    portfolio_auto_path: str
    path_filters_auto_path: str
    paradigms_dir: str
    principles_dir: str


@dataclass(frozen=True)
class SnapshotParadigmRef:
    paradigm_id: str
    paradigm_config_path: str


@dataclass(frozen=True)
class SnapshotPrincipleRef:
    paradigm_id: str
    principle_id: str
    principle_config_path: str


@dataclass(frozen=True)
class SnapshotDataSlice:
    start_dt: str
    end_dt: str
    instruments: list[str]
    anchor_tfs: list[str]
    tf_entries: list[str]
    contexts_filter: str | None


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    description: str
    created_ts: str
    config_refs: SnapshotConfigRefs
    paradigms: list[SnapshotParadigmRef]
    principles: list[SnapshotPrincipleRef]
    data_slice: SnapshotDataSlice


def load_snapshot_manifest(path: Path | str) -> SnapshotManifest:
    """Load snapshots/<SNAPSHOT_ID>.json (stored as YAML/JSON) into a SnapshotManifest."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Snapshot manifest not found at: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg_refs_raw = raw["config_refs"]
    cfg_refs = SnapshotConfigRefs(
        retail_config_path=cfg_refs_raw["retail_config_path"],
        features_registry_path=cfg_refs_raw["features_registry_path"],
        features_auto_path=cfg_refs_raw["features_auto_path"],
        rails_auto_path=cfg_refs_raw["rails_auto_path"],
        portfolio_auto_path=cfg_refs_raw["portfolio_auto_path"],
        path_filters_auto_path=cfg_refs_raw["path_filters_auto_path"],
        paradigms_dir=cfg_refs_raw["paradigms_dir"],
        principles_dir=cfg_refs_raw["principles_dir"],
    )

    paradigms = [
        SnapshotParadigmRef(
            paradigm_id=item["paradigm_id"],
            paradigm_config_path=item["paradigm_config_path"],
        )
        for item in raw.get("paradigms", [])
    ]

    principles = [
        SnapshotPrincipleRef(
            paradigm_id=item["paradigm_id"],
            principle_id=item["principle_id"],
            principle_config_path=item["principle_config_path"],
        )
        for item in raw.get("principles", [])
    ]

    ds_raw = raw["data_slice"]
    data_slice = SnapshotDataSlice(
        start_dt=ds_raw["start_dt"],
        end_dt=ds_raw["end_dt"],
        instruments=list(ds_raw.get("instruments", [])),
        anchor_tfs=list(ds_raw.get("anchor_tfs", [])),
        tf_entries=list(ds_raw.get("tf_entries", [])),
        contexts_filter=ds_raw.get("contexts_filter"),
    )

    return SnapshotManifest(
        snapshot_id=raw["snapshot_id"],
        description=raw.get("description", ""),
        created_ts=raw["created_ts"],
        config_refs=cfg_refs,
        paradigms=paradigms,
        principles=principles,
        data_slice=data_slice,
    )


# ---------------------------------------------------------------------------
# Table schema helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSchema:
    """Logical schema for a Parquet-backed table."""

    name: str
    partition_cols: list[str]
    columns: dict[str, str]


def polars_dtype(type_str: str) -> pl.DataType:
    t = type_str.strip().lower()
    if t in ("string", "utf8", "str"):
        return pl.Utf8
    if t in ("int", "int64", "i64"):
        return pl.Int64
    if t in ("double", "float", "float64", "f64"):
        return pl.Float64
    if t in ("boolean", "bool"):
        return pl.Boolean
    if t in ("timestamp", "datetime"):
        return pl.Datetime("us", "UTC")
    if t in ("date",):
        return pl.Date
    if t in ("array<string>", "list<string>"):
        return pl.List(pl.Utf8)
    return pl.Utf8


def empty_frame(schema: TableSchema) -> pl.DataFrame:
    cols: dict[str, pl.Series] = {}
    for name, t in schema.columns.items():
        cols[name] = pl.Series(name=name, values=[], dtype=polars_dtype(t))
    return pl.DataFrame(cols)


def enforce_schema(
    df: pl.DataFrame,
    schema: TableSchema,
    *,
    allow_extra: bool = True,
    reorder: bool = False,
) -> pl.DataFrame:
    """Ensure df contains all schema.columns with correct dtypes."""
    if df is None or df.is_empty():
        out = empty_frame(schema)
        return out if allow_extra else out.select(list(schema.columns.keys()))

    out = df

    missing: list[pl.Expr] = []
    for col, t in schema.columns.items():
        if col not in out.columns:
            missing.append(pl.lit(None).cast(polars_dtype(t)).alias(col))
    if missing:
        out = out.with_columns(missing)

    cast_exprs: list[pl.Expr] = []
    for col, t in schema.columns.items():
        if col in out.columns:
            cast_exprs.append(pl.col(col).cast(polars_dtype(t), strict=False).alias(col))
    if cast_exprs:
        out = out.with_columns(cast_exprs)

    if not allow_extra:
        return out.select(list(schema.columns.keys()))

    if reorder:
        schema_cols = [c for c in schema.columns.keys() if c in out.columns]
        extras = [c for c in out.columns if c not in schema.columns]
        out = out.select(schema_cols + extras)

    return out


def _merge_columns(*parts: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in parts:
        for k, v in p.items():
            if k in out and out[k] != v:
                raise ValueError(f"Schema dtype conflict for col={k!r}: {out[k]!r} vs {v!r}")
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Base identity columns (engine-owned; do not declare these in the registry)
# ---------------------------------------------------------------------------

_BASE_META_COLS: Dict[str, str] = {
    "snapshot_id": "string",
    "run_id": "string",
    "mode": "string",
    "dt": "date",
}

# ---------------------------------------------------------------------------
# Registry-derived columns (compiled contract)
#
# The registry must NOT own engine identity/partition columns. We treat those as
# reserved and ignore them if they appear in generated output to avoid drift bugs.
# ---------------------------------------------------------------------------

_RESERVED_ENGINE_COLS = {"snapshot_id", "run_id", "mode", "dt", "trading_day"}

try:
    # Generated by scripts/compile_features_registry_contract.py
    from engine.core._registry_table_columns import REGISTRY_TABLE_COLUMNS as _REGISTRY_TABLE_COLUMNS  # type: ignore
except Exception:  # pragma: no cover
    _REGISTRY_TABLE_COLUMNS = {}


def _registry_cols(table: str) -> Dict[str, str]:
    cols = dict(_REGISTRY_TABLE_COLUMNS.get(table, {})) if _REGISTRY_TABLE_COLUMNS else {}
    # Safety: never allow registry to define reserved cols (prevents dt dtype conflicts).
    for k in list(cols.keys()):
        if k in _RESERVED_ENGINE_COLS:
            cols.pop(k, None)
    return cols




def _strip_registry_overlap(table: str, cols: Dict[str, str]) -> Dict[str, str]:
    """
    Return only engine-owned extras from a manual column dict.

    Any key already present in the compiled registry contract for `table` is removed,
    as are base meta columns (snapshot_id/run_id/mode/dt). This prevents dtype drift
    conflicts while still allowing engine-only or legacy compatibility fields.
    """
    reg = _registry_cols(table)
    return {k: v for k, v in cols.items() if k not in reg and k not in _BASE_META_COLS}


# ---------------------------------------------------------------------------
# Concrete table schemas (dt is canonical; trading_day is legacy/optional)
# ---------------------------------------------------------------------------

# NOTE: The column dicts for registry-backed tables (features/windows/zones_state/...)
# are compiled into this module to keep runtime deterministic and avoid import-time IO.
# When the schema generator is introduced, these dicts should be written into a
# generated module (e.g., engine.core.schema_compiled) and imported here.

WINDOWS_COLUMNS: Dict[str, str] = {
    'snapshot_id': 'string',
    'run_id': 'string',
    'mode': 'string',
    'dt': 'date',
    'paradigm_id': 'string',
    'principle_id': 'string',
    'instrument': 'string',
    'anchor_tf': 'string',
    'anchor_ts': 'timestamp',
    'tf_entry': 'string',
    'tod_bucket': 'string',
    'dow_bucket': 'string',
    'vol_regime': 'string',
    'trend_regime': 'string',
    'macro_state': 'string',
    'macro_is_blackout': 'boolean',
    'macro_blackout_max_impact': 'int',
    'micro_corr_regime': 'string',
    'corr_cluster_id': 'string',
    'zone_behaviour_type_bucket': 'string',
    'zone_freshness_bucket': 'string',
    'zone_stack_depth_bucket': 'string',
    'zone_htf_confluence_bucket': 'string',
    'zone_vp_type_bucket': 'string',
    'unsup_regime_id': 'string',
    'entry_profile_id': 'string',
    'management_profile_id': 'string',
    'carry_ts_carry_score': 'double',
    'carry_ts_confidence': 'double',
    'carry_ts_data_age_bars': 'int',
    'carry_ts_level': 'double',
    'carry_ts_level_z': 'double',
    'carry_ts_missing_flag': 'boolean',
    'carry_ts_proxy_method_used': 'string',
    'carry_ts_sign_bucket': 'string',
    'carry_ts_slope_z': 'double',
    'carry_ts_source_used': 'string',
    'carry_ts_ts_regime': 'string',
    'carry_ts_ts_slope': 'double',
    'pac_anchor_body_ratio': 'double',
    'pac_anchor_clv': 'double',
    'pac_anchor_range': 'double',
    'pac_anchor_range_z': 'double',
    'pac_anchor_wick_ratio': 'double',
    'pac_context_body_ratio_mean': 'double',
    'pac_context_body_ratio_z': 'double',
    'pac_context_range_mean': 'double',
    'pac_context_range_z': 'double',
    'pac_context_wick_ratio_mean': 'double',
    'pac_context_wick_ratio_z': 'double',
    'pac_entry_immediate_followthrough_flag': 'boolean',
    'pac_entry_immediate_rejection_flag': 'boolean',
    'pac_micro_state_anchor': 'string',
    'pac_micro_state_end': 'string',
    'pac_micro_state_mid': 'string',
    'pac_micro_state_post_entry': 'string',
    'pac_micro_state_pre_entry': 'string',
    'pac_micro_state_start': 'string',
    'pac_path_breakout_flag': 'boolean',
    'pac_path_breakout_side': 'string',
    'pac_path_chop_score': 'double',
    'pac_path_impulse_score': 'double',
    'pac_path_regime': 'string',
    'pac_path_transition': 'string',
    'pac_path_trend_score': 'double',
    'stat_ts_autocorr1': 'double',
    'stat_ts_half_life_bars': 'double',
    'stat_ts_hurst': 'double',
    'stat_ts_kurtosis': 'double',
    'stat_ts_range': 'double',
    'stat_ts_range_zscore': 'double',
    'stat_ts_return': 'double',
    'stat_ts_return_zscore': 'double',
    'stat_ts_rsi': 'double',
    'stat_ts_rsi_zscore': 'double',
    'stat_ts_skew': 'double',
    'stat_ts_trend_score': 'double',
    'stat_ts_trend_zscore': 'double',
    'stat_ts_vol': 'double',
    'stat_ts_vol_zscore': 'double',
    'unsup_feature_set_id_used': 'string',
    'unsup_model_id': 'string',
    'unsup_refit_ts': 'timestamp',
    'unsup_regime_confidence': 'double',
    'unsup_regime_entropy': 'double',
    'unsup_regime_id': 'string',
    'vol_range_compress_flag': 'boolean',
    'vol_range_expand_flag': 'boolean',
    'vol_range_keltner_width': 'double',
    'vol_range_keltner_width_z': 'double',
    'vol_range_lookback_bars': 'int',
    'vol_range_realised_range': 'double',
    'vol_range_range_z': 'double',
    'vol_range_regime_confidence': 'double',
    'vol_range_regime_id': 'string',
    'vol_range_state': 'string',
}

WINDOWS_SCHEMA = TableSchema(
    name="data/windows",
    partition_cols=["run_id", "instrument", "anchor_tf", "dt"],
    columns=_merge_columns(_BASE_META_COLS, _registry_cols('data/windows'), _strip_registry_overlap('data/windows', WINDOWS_COLUMNS)),
)


TRADE_PATHS_SCHEMA = TableSchema(
    name="data/trade_paths",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "instrument": "string",
        "trade_id": "string",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "instrument_cluster_id": "string",
        "side": "string",
        "anchor_tf": "string",
        "tf_entry": "string",
        "entry_ts": "timestamp",
        "exit_ts": "timestamp",
        "entry_px": "double",
        "exit_px": "double",
        "sl_px": "double",
        "tp_px": "double",
        "minutes_since_session_open_bucket": "string",
        "minutes_since_session_open": "int",
        "day_of_week_entry": "int",
        "tod_bucket": "string",
        "dow_bucket": "string",
        "vol_regime_entry": "string",
        "trend_regime_entry": "string",
        "macro_state_entry": "string",
        "macro_state_path": "string",
        "corr_cluster_entry": "string",
        "micro_corr_regime_entry": "string",
        "micro_corr_flip_flag": "boolean",
        "unsup_regime_id_entry": "string",
        "entry_profile_id": "string",
        "management_profile_id": "string",
        "principle_status_at_entry": "string",
        "rail_version_at_entry": "string",
        "zone_behaviour_type_bucket_entry": "string",
        "zone_freshness_bucket_entry": "string",
        "zone_stack_depth_bucket_entry": "string",
        "zone_htf_confluence_bucket_entry": "string",
        "zone_vp_type_bucket_entry": "string",
        "zone_vp_skew_entry": "double",
        "zone_vp_poc_relative_position_entry": "double",
        "context_session_trade_index": "int",
        "principle_occurrence_index": "int",
        "zone_touch_rank_for_trade": "int",
        "first_touch_flag": "boolean",
        "second_touch_flag": "boolean",
        "third_plus_touch_flag": "boolean",
        "fraction_in_trade_liquidity_stressed": "double",
        "fraction_exit_in_spread_spike": "double",
        "counterfactual_best_R": "double",
        "counterfactual_best_R_delta": "double",
        "would_hit_1R_before_SL_flag": "boolean",
        "would_hit_2R_if_no_timeout_flag": "boolean",
        "critic_score_at_entry": "double",
        "critic_reason_tags_at_entry": "string",
        "critic_reason_cluster_id": "string",
        "entry_mode": "string",
        "entry_structure_anchor": "string",
        "entry_structure_fraction_bucket": "string",
        "entry_max_bars_after_setup_bucket": "string",
        "mgmt_sl_style": "string",
        "mgmt_tp_style": "string",
        "mgmt_has_runners_flag": "boolean",
        "mgmt_has_timeout_flag": "boolean",
        "mgmt_timeout_bucket": "string",
        "portfolio_risk_bps_at_entry": "double",
        "instrument_weight_at_entry": "double",
        "cluster_weight_at_entry": "double",
        "simultaneous_trades_in_cluster": "int",
        "effective_risk_multiplier": "double",
        "effective_management_profile_id": "string",
        "confluence_state_entry": "string",
        "confluence_paradigms_entry": "string",
        "confluence_rule_ids_entry": "string",
        "conflict_state_entry": "string",
        "conflict_paradigms_entry": "string",
        "conflict_rule_ids_entry": "string",
        "path_cluster_id": "string",
        "path_family_id": "string",
        "path_shape": "string",
        "path_filter_primary": "string",
        "path_filter_tags_json": "string",
        "time_to_1R_bars": "int",
        "time_to_2R_bars": "int",
        "mae_R": "double",
        "mae_R_bucket": "string",
        "mfe_R": "double",
        "exit_reason": "string",
    },
)


DECISIONS_SCHEMA = TableSchema(
    name="data/decisions",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "stage", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "stage": "string",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "instrument": "string",
        "trade_id": "string",
        "decision_ts": "timestamp",
    },
)

# ---------------------------------------------------------------------------
# Decisions (stage-specific schemas)
# ---------------------------------------------------------------------------

_DECISIONS_HYPOTHESES_EXTRA_COLS: dict[str, str] = {
    "anchor_tf": "string",
    "anchor_ts": "timestamp",
    "tf_entry": "string",
    "side": "string",
    "setup_logic_id": "string",
    "entry_profile_id": "string",
    "management_profile_id": "string",
    "entry_hint_price": "double",
    "stop_hint_price": "double",
    "tp_hint_prices": "string",
    "tod_bucket": "string",
    "dow_bucket": "string",
    "vol_regime": "string",
    "trend_regime": "string",
    "macro_state": "string",
    "macro_is_blackout": "boolean",
    "macro_blackout_max_impact": "int",
    "micro_corr_regime": "string",
    "corr_cluster_id": "string",
    "unsup_regime_id": "string",
    "context_keys_json": "string",
    "hypothesis_features_json": "string",
}

_DECISIONS_CRITIC_EXTRA_COLS: dict[str, str] = {
    "critic_score_at_entry": "double",
    "critic_reason_tags_at_entry": "string",
    "critic_reason_cluster_id": "string",
}

_DECISIONS_PRETRADE_EXTRA_COLS: dict[str, str] = {
    "macro_is_blackout": "boolean",
    "macro_blackout_max_impact": "int",
    "macro_blackout_tag": "string",
    "blocked_by_macro": "boolean",
    "blocked_by_spread": "boolean",
    "micro_risk_scale": "double",
    "pretrade_notes": "string",
}

_DECISIONS_GATEKEEPER_EXTRA_COLS: dict[str, str] = {
    "gatekeeper_status": "string",
    "gate_status": "string",
    "status": "string",
    "rails_passed_flag": "boolean",
    "rails_config_key": "string",
    "gate_rails_version": "string",
    "gate_reason": "string",
    "risk_mode": "string",
    "risk_per_trade_bps": "double",
}

_DECISIONS_PORTFOLIO_EXTRA_COLS: dict[str, str] = {
    "portfolio_risk_bps_at_entry": "double",
    "allocated_notional": "double",
    "allocated_risk_bps": "double",
    "instrument_weight_fraction": "double",
    "cluster_weight_fraction": "double",
    "portfolio_risk_bps_after": "double",
    "allocation_mode": "string",
    "drop_reason": "string",
}

_DECISIONS_BRACKET_FALLBACK_COLS: dict[str, str] = {
    "entry_ts": "timestamp",
    "entry_px": "double",
    "sl_px": "double",
    "tp_px": "double",
    "entry_mode": "string",
    "order_type": "string",
}

DECISIONS_HYPOTHESES_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={**dict(DECISIONS_SCHEMA.columns), **_DECISIONS_HYPOTHESES_EXTRA_COLS},
)

DECISIONS_CRITIC_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={**dict(DECISIONS_SCHEMA.columns), **_DECISIONS_HYPOTHESES_EXTRA_COLS, **_DECISIONS_CRITIC_EXTRA_COLS},
)

DECISIONS_PRETRADE_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={
        **dict(DECISIONS_SCHEMA.columns),
        **_DECISIONS_HYPOTHESES_EXTRA_COLS,
        **_DECISIONS_CRITIC_EXTRA_COLS,
        **_DECISIONS_PRETRADE_EXTRA_COLS,
    },
)

DECISIONS_GATEKEEPER_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={
        **dict(DECISIONS_SCHEMA.columns),
        **_DECISIONS_HYPOTHESES_EXTRA_COLS,
        **_DECISIONS_CRITIC_EXTRA_COLS,
        **_DECISIONS_PRETRADE_EXTRA_COLS,
        **_DECISIONS_GATEKEEPER_EXTRA_COLS,
    },
)

DECISIONS_PORTFOLIO_SCHEMA = TableSchema(
    name=DECISIONS_SCHEMA.name,
    partition_cols=DECISIONS_SCHEMA.partition_cols,
    columns={
        **dict(DECISIONS_SCHEMA.columns),
        **_DECISIONS_HYPOTHESES_EXTRA_COLS,
        **_DECISIONS_CRITIC_EXTRA_COLS,
        **_DECISIONS_PRETRADE_EXTRA_COLS,
        **_DECISIONS_GATEKEEPER_EXTRA_COLS,
        **_DECISIONS_PORTFOLIO_EXTRA_COLS,
        **_DECISIONS_BRACKET_FALLBACK_COLS,
    },
)

BRACKETS_SCHEMA = TableSchema(
    name="data/brackets",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "instrument": "string",
        "trade_id": "string",
    },
)

ORDERS_SCHEMA = TableSchema(
    name="data/orders",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "instrument": "string",
    },
)

FILLS_SCHEMA = TableSchema(
    name="data/fills",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "dt"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "instrument": "string",
    },
)

RUN_REPORTS_SCHEMA = TableSchema(
    name="data/run_reports",
    partition_cols=["run_id", "dt", "cluster_id"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "cluster_id": "string",
    },
)

PRINCIPLES_CONTEXT_SCHEMA = TableSchema(
    name="data/principles_context",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "dt", "cluster_id"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "cluster_id": "string",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "trade_count": "int",
    },
)

TRADE_CLUSTERS_SCHEMA = TableSchema(
    name="data/trade_clusters",
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "dt", "cluster_id"],
    columns={
        "snapshot_id": "string",
        "run_id": "string",
        "mode": "string",
        "dt": "date",
        "cluster_id": "string",
        "paradigm_id": "string",
        "principle_id": "string",
        "candidate_id": "string",
        "experiment_id": "string",
        "trade_count": "int",
    },
)

# ---------------------------------------------------------------------------
# Registry-backed tables (compiled surface)
# ---------------------------------------------------------------------------

FEATURES_COLUMNS: Dict[str, str] = {
    'snapshot_id': 'string',
    'run_id': 'string',
    'mode': 'string',
    'dt': 'date',
    'instrument': 'string',
    'anchor_tf': 'string',
    'ts': 'timestamp',
    'fvg_direction': 'string',
    'fvg_gap_ticks': 'double',
    'fvg_origin_tf': 'string',
    'fvg_origin_ts': 'timestamp',
    'fvg_fill_state': 'string',
    'fvg_location_bucket': 'string',
    'ob_type': 'string',
    'ob_high': 'double',
    'ob_low': 'double',
    'ob_origin_ts': 'timestamp',
    'ob_freshness_bucket': 'string',
    'eqh_flag': 'boolean',
    'eql_flag': 'boolean',
    'liq_grab_flag': 'boolean',
    'ict_struct_liquidity_tag': 'string',
    'ict_struct_swing_high': 'double',
    'ict_struct_swing_low': 'double',
    'ict_struct_pd_index': 'int',
    'ict_struct_swing_strength': 'int',
    'ict_struct_swing_trend_dir': 'string',
    'ict_struct_dealing_range_high': 'double',
    'ict_struct_dealing_range_low': 'double',
    'ict_struct_dealing_range_mid': 'double',
    'fvg_fill_duration_bars': 'int',
    'fvg_was_mitigated_flag': 'boolean',
    'bos_type': 'string',
    'ict_struct_context_tag': 'string',
    'atr_anchor': 'double',
    'atr_z': 'double',
    'vol_regime': 'string',
    'rsi_value': 'double',
    'rsi_bucket': 'string',
    'rsi_slope': 'double',
    'ema_fast': 'double',
    'ema_slow': 'double',
    'ema_spread': 'double',
    'ema_spread_z': 'double',
    'ema_cross_state': 'string',
    'adx_value': 'double',
    'adx_bucket': 'string',
    'ta_vol__atr': 'double',
    'atr_value': 'double',
    'ta_vol__ret1_std': 'double',
    'ta_vol__range_mean': 'double',
    'ta_vol__ret_std_z': 'double',
    'ta_vol__range_mean_z': 'double',
    'ta_vol__atr_z': 'double',
    'ta_vol__tr_norm': 'double',
    'ta_vol__gap_norm': 'double',
    'ta_vol__vol_of_vol': 'double',
    'ta_vol__parkinson': 'double',
    'ta_vol__garman_klass': 'double',
    'ta_vol__yang_zhang': 'double',
    'ta_vol__squeeze_flag': 'boolean',
    'ta_vol__squeeze_state': 'string',
    'ta_vol__squeeze_intensity': 'double',
    'ta_vol__range_expansion_flag': 'boolean',
    'ta_vol__range_expansion_z': 'double',
    'ta_vol__atr_regime': 'string',
    'ta_vol__range_regime': 'string',
    'ta_vol__vol_regime': 'string',
    'ta_vol__realised_vol': 'double',
    'ta_vol__realised_vol_z': 'double',
    'ta_vol__atr_pct': 'double',
    'ta_vol__range_pct': 'double',
    'ta_vol__bar_vol_norm': 'double',
    'ta_vol__clv': 'double',
    'ta_vol__clv_z': 'double',
    'ta_vol__wick_ratio': 'double',
    'ta_vol__wick_ratio_z': 'double',
    'ta_vol__body_ratio': 'double',
    'ta_vol__body_ratio_z': 'double',
    'ta_vol__inside_bar_flag': 'boolean',
    'ta_vol__outside_bar_flag': 'boolean',
    'ta_vol__gap_flag': 'boolean',
    'ta_vol__bar_type': 'string',
    'ta_vol__bar_quality': 'string',
    'ta_vol__ret1': 'double',
    'ta_vol__ret1_z': 'double',
    'ta_vol__ret5': 'double',
    'ta_vol__ret5_z': 'double',
    'ta_vol__ret20': 'double',
    'ta_vol__ret20_z': 'double',
    'ta_vol__momo_state': 'string',
    'ta_vol__trend_state': 'string',
    'ta_vol__meanrev_state': 'string',
    'ta_vol__micro_shock_flag': 'boolean',
    'ta_vol__micro_shock_score': 'double',
    'ta_vol__range_compress_flag': 'boolean',
    'ta_vol__range_compress_score': 'double',
    'ta_vol__range_expand_score': 'double',
    'ta_vol__range_state': 'string',
    'ta_vol__roll_atr_slope': 'double',
    'ta_vol__roll_atr_slope_z': 'double',
    'ta_vol__roll_range_slope': 'double',
    'ta_vol__roll_range_slope_z': 'double',
    'ta_vol__vol_surface_state': 'string',
    'ta_vol__vol_surface_score': 'double',
    'ta_vol__vol_surface_conf': 'double',
    'ta_vol__stdev_up': 'double',
    'ta_vol__stdev_dn': 'double',
    'ta_vol__bb_width': 'double',
    'ta_vol__bb_width_z': 'double',
    'ta_vol__kc_width': 'double',
    'ta_vol__kc_width_z': 'double',
    'ta_vol__bb_kc_squeeze_flag': 'boolean',
    'ta_vol__bb_kc_squeeze_intensity': 'double',
    'ta_vol__bb_pos': 'double',
    'ta_vol__kc_pos': 'double',
    'ta_vol__range_pos': 'double',
    'ta_vol__risk_units': 'double',
    'ta_vol__risk_bucket': 'string',
    'ta_vol__risk_z': 'double',
    'ta_vol__risk_regime': 'string',
}

FEATURES_SCHEMA = TableSchema(
    name="data/features",
    partition_cols=["run_id", "instrument", "anchor_tf", "dt"],
    columns=_merge_columns(_BASE_META_COLS, _registry_cols('data/features'), _strip_registry_overlap('data/features', FEATURES_COLUMNS)),
)

FEATURES_CORR_COLUMNS: Dict[str, str] = {
    'snapshot_id': 'string',
    'run_id': 'string',
    'mode': 'string',
    'dt': 'date',
    'instrument': 'string',
    'ts': 'timestamp',
    'corr_dxy': 'double',
    'corr_index_major': 'double',
    'corr_oil': 'double',
    'micro_corr_regime': 'string',
    'corr_cluster_id': 'string',
    'corr_ref1_id': 'string',
    'corr_ref1': 'double',
    'corr_ref1_xcorr_max': 'double',
    'corr_ref1_xcorr_lag_bars': 'int',
    'corr_ref2_id': 'string',
    'corr_ref2': 'double',
    'corr_ref3_id': 'string',
    'corr_ref3': 'double',
    'corr_topk_neighbors_json': 'string',
    'corr_vol_ref1': 'double',
    'corr_vol_ref2': 'double',
    'corr_vol_ref3': 'double',
    'corr_cluster_id_htf': 'string',
    'corr_cluster_confidence_htf': 'double',
    'corr_cluster_stability_htf': 'double',
    'corr_htf_ref1_id': 'string',
    'corr_htf_ref1': 'double',
    'corr_htf_ref2_id': 'string',
    'corr_htf_ref2': 'double',
    'corr_htf_ref3_id': 'string',
    'corr_htf_ref3': 'double',
    'xs_relval_spread_level': 'double',
    'xs_relval_spread_zscore': 'double',
    'xs_relval_carry_rank': 'double',
    'xs_relval_momo_rank': 'double',
    'xs_relval_beta_to_cluster': 'double',
    'xs_relval_idio_vol': 'double',
    'xs_relval_vol_rank': 'double',
    'xs_relval_trend_rank': 'double',
    'xs_relval_meanrev_rank': 'double',
    'xs_relval_dispersion': 'double',
    'xs_relval_dispersion_z': 'double',
    'xs_relval_spread_rank': 'double',
    'xs_relval_signal': 'string',
    'xs_relval_signal_conf': 'double',
    'xs_relval_bucket': 'string',
    'xs_relval_pair_id': 'string',
    'xs_relval_pair_spread_level': 'double',
    'xs_relval_pair_spread_z': 'double',
    'xs_relval_pair_signal': 'double',
    'xs_relval_pair_bucket': 'string',
    'xs_relval_cluster_id': 'string',
    'xs_relval_cluster_rank': 'double',
    'xs_relval_cluster_signal': 'double',
}

FEATURES_CORR_SCHEMA = TableSchema(
    name="data/features_corr",
    partition_cols=["run_id", "instrument", "dt"],
    columns=_merge_columns(_BASE_META_COLS, _registry_cols('data/features_corr'), _strip_registry_overlap('data/features_corr', FEATURES_CORR_COLUMNS)),
)

MACRO_COLUMNS: Dict[str, str] = {
    'snapshot_id': 'string',
    'run_id': 'string',
    'mode': 'string',
    'dt': 'date',
    'ts': 'timestamp',
    'release_ts': 'timestamp',
    'window_start': 'timestamp',
    'window_end': 'timestamp',
    'macro_state': 'string',
    'blackout': 'boolean',
    'event_name': 'string',
    'impact_level': 'int',
    'currency': 'string',
    'country': 'string',
    'severity': 'int',
    'asset_relevance': 'string',
    'asset_class': 'string',
    'macro_event_id': 'string',
    'macro_event_source': 'string',
    'macro_event_version': 'string',
    'macro_event_updated_ts': 'timestamp',
    'dominant_event_id': 'string',
    'event_rank': 'int',
    'is_quiet': 'boolean',
    'is_pre_event': 'boolean',
    'is_red_window': 'boolean',
    'is_post_event': 'boolean',
    'macro_regime_id': 'string',
    'macro_regime_label': 'string',
    'macro_regime_confidence': 'double',
    'macro_regime_entropy': 'double',
    'macro_regime_method_used': 'string',
    'macro_regime_model_id': 'string',
    'macro_regime_fit_ts': 'timestamp',
    'macro_regime_feature_set_used': 'string',
}

MACRO_SCHEMA = TableSchema(
    name="data/macro",
    partition_cols=["run_id", "dt"],
    columns=_merge_columns(_BASE_META_COLS, _registry_cols('data/macro'), _strip_registry_overlap('data/macro', MACRO_COLUMNS)),
)

ZONES_STATE_COLUMNS: Dict[str, str] = {
    'snapshot_id': 'string',
    'run_id': 'string',
    'mode': 'string',
    'dt': 'date',
    'instrument': 'string',
    'anchor_tf': 'string',
    'zone_id': 'string',
    'ts': 'timestamp',
    'zone_index': 'int',
    'zone_lower_price': 'double',
    'zone_upper_price': 'double',
    'zone_mid_price': 'double',
    'touch_count': 'int',
    'zone_touch_count_bucket': 'string',
    'fvg_origin_count': 'int',
    'ob_origin_count': 'int',
    'liq_grab_count': 'int',
    'zone_reaction_strength_bucket': 'string',
    'zone_reaction_consistency_bucket': 'string',
    'zone_freshness_bucket': 'string',
    'avg_dwell_bars': 'double',
    'zone_avg_dwell_bucket': 'string',
    'zone_behaviour_type': 'string',
    'zone_behaviour_type_bucket': 'string',
    'zone_rsi_prev_overbought_flag': 'boolean',
    'zone_rsi_prev_oversold_flag': 'boolean',
    'zone_rsi_bias_bucket': 'string',
    'zone_htf_confluence': 'string',
    'zone_htf_confluence_bucket': 'string',
    'zone_stack_depth': 'int',
    'zone_stack_depth_bucket': 'string',
    'zmf_zone_kind': 'string',
    'zmf_zone_lo': 'double',
    'zmf_zone_hi': 'double',
    'zmf_zone_mid': 'double',
    'zmf_zone_width': 'double',
    'zone_vp_total_volume_units': 'double',
    'zone_vp_poc_price': 'double',
    'zone_vp_poc_relative_position': 'double',
    'zone_vp_volume_lower_units': 'double',
    'zone_vp_volume_upper_units': 'double',
    'zone_vp_skew': 'double',
    'zone_vp_hvn_flag': 'boolean',
    'zone_vp_lvn_flag': 'boolean',
    'zone_vp_type': 'string',
    'zone_vp_type_bucket': 'string',
    'ta_memory_bars_since_last_touch': 'int',
    'ta_memory_touch_count_lookback': 'int',
    'ta_memory_touch_recency_score': 'double',
    'ta_memory_rsi_last_value': 'double',
    'ta_memory_rsi_cross_count_lookback': 'int',
    'ta_memory_ema_fast_touch_count_lookback': 'int',
    'ta_memory_ema_slow_touch_count_lookback': 'int',
    'zone_width_bps': 'double',
    'zone_width_atr': 'double',
    'zone_width_bucket': 'string',
    'zone_age_bars': 'int',
    'zone_age_bucket': 'string',
    'zone_last_touch_ts': 'timestamp',
    'zone_bars_since_last_touch': 'int',
    'zone_recency_score': 'double',
    'zone_touch_density': 'double',
    'zone_touch_spacing_mean_bars': 'double',
    'zone_break_count': 'int',
    'zone_hold_count': 'int',
    'zone_break_hold_ratio': 'double',
    'zone_reject_count': 'int',
    'zone_pass_through_count': 'int',
    'zone_reject_ratio': 'double',
    'zone_reaction_median_R': 'double',
    'zone_reaction_median_R_bucket': 'string',
    'zone_reaction_stdev_R': 'double',
    'zone_reaction_quality_score': 'double',
    'zone_flip_flag': 'boolean',
    'zone_directional_bias': 'string',
    'zone_bias_confidence': 'double',
    'zone_stack_id': 'string',
    'zone_stack_rank': 'int',
    'zone_stack_total_zones': 'int',
    'zone_stack_width_sum_atr': 'double',
    'zone_stack_density': 'double',
    'zone_htf_parent_id': 'string',
    'zone_htf_distance_atr': 'double',
    'zone_htf_distance_bucket': 'string',
    'zone_context_cluster_id': 'string',
    'zone_context_cluster_conf': 'double',
    'zone_quality_score': 'double',
    'zone_quality_bucket': 'string',
    'zone_is_tradeable_flag': 'boolean',
    'zone_invalid_reason': 'string',
    'zone_tag_json': 'string',
    'zone_notes': 'string',
}

ZONES_STATE_SCHEMA = TableSchema(
    name="data/zones_state",
    partition_cols=["instrument", "anchor_tf", "dt"],
    columns=_merge_columns(_BASE_META_COLS, _registry_cols('data/zones_state'), _strip_registry_overlap('data/zones_state', ZONES_STATE_COLUMNS)),
)

PCRA_COLUMNS: Dict[str, str] = {
    'snapshot_id': 'string',
    'run_id': 'string',
    'mode': 'string',
    'dt': 'date',
    'anchor_ts': 'timestamp',
    'instrument': 'string',
    'anchor_tf': 'string',
    'pcr_window_ts': 'timestamp',
    'pcr_ofi_value': 'double',
    'pcr_ofi_bucket': 'string',
    'pcr_micro_vol_value': 'double',
    'pcr_micro_vol_bucket': 'string',
    'pcr_range_value': 'double',
    'pcr_range_bucket': 'string',
    'pcr_clv_bucket': 'string',
    'pcr_ofi_z': 'double',
    'pcr_ofi_abs': 'double',
    'pcr_ofi_direction': 'string',
    'pcr_micro_vol_z': 'double',
    'pcr_range_z': 'double',
    'pcr_clv_value': 'double',
    'pcr_spread_proxy': 'double',
    'pcr_spread_proxy_z': 'double',
    'pcr_sweep_proxy': 'double',
    'pcr_sweep_proxy_flag': 'boolean',
    'pcr_absorption_proxy': 'double',
    'pcr_absorption_proxy_flag': 'boolean',
    'pcr_liquidity_stress_score': 'double',
    'pcr_liquidity_stress_bucket': 'string',
    'pcr_tick_total_volume_units': 'double',
    'pcr_tick_delta_units': 'double',
    'pcr_tick_delta_z': 'double',
    'pcr_tick_imbalance_ratio': 'double',
    'pcr_tick_imbalance_flag': 'boolean',
    'pcr_tick_concentration_ratio_at': 'double',
    'pcr_tick_concentration_flag_at': 'boolean',
    'pcr_tick_sweep_flag': 'boolean',
    'pcr_tick_absorption_flag': 'boolean',
    'pcr_footprint_pattern': 'string',
    'pcr_tick_bid_vol_units': 'double',
    'pcr_tick_ask_vol_units': 'double',
    'pcr_tick_max_imbalance_level': 'double',
    'pcr_tick_imbalance_levels_count': 'int',
    'pcr_tick_value_area_low': 'double',
    'pcr_tick_value_area_high': 'double',
    'pcr_tick_poc_price': 'double',
    'pcr_tick_poc_delta_units': 'double',
    'pcr_tick_poc_imbalance_ratio': 'double',
    'pcr_tick_poc_flag': 'boolean',
    'pcr_tick_sweep_range_ticks': 'double',
    'pcr_tick_sweep_range_flag': 'boolean',
    'pcr_tick_absorption_volume_units': 'double',
    'pcr_tick_absorption_range_ticks': 'double',
    'pcr_tick_absorption_strength': 'double',
    'pcr_tick_delta_divergence_flag': 'boolean',
    'pcr_tick_delta_divergence_score': 'double',
    'pcr_tick_ofi_delta_alignment': 'string',
    'pcr_tick_session_volume_pct': 'double',
    'pcr_tick_session_delta_pct': 'double',
    'pcr_tick_session_imbalance_pct': 'double',
    'pcr_tick_quality_score': 'double',
    'pcr_tick_quality_bucket': 'string',
    'pcr_tick_notes': 'string',
}

PCRA_SCHEMA = TableSchema(
    name="data/pcr_a",
    partition_cols=["instrument", "anchor_tf", "dt"],
    columns=_merge_columns(_BASE_META_COLS, _registry_cols('data/pcr_a'), _strip_registry_overlap('data/pcr_a', PCRA_COLUMNS)),
)

TABLE_SCHEMAS: dict[str, TableSchema] = {
    "windows": WINDOWS_SCHEMA,
    "trade_paths": TRADE_PATHS_SCHEMA,
    "decisions_hypotheses": DECISIONS_HYPOTHESES_SCHEMA,
    "decisions_critic": DECISIONS_CRITIC_SCHEMA,
    "decisions_pretrade": DECISIONS_PRETRADE_SCHEMA,
    "decisions_gatekeeper": DECISIONS_GATEKEEPER_SCHEMA,
    "decisions_portfolio": DECISIONS_PORTFOLIO_SCHEMA,
    "brackets": BRACKETS_SCHEMA,
    "orders": ORDERS_SCHEMA,
    "fills": FILLS_SCHEMA,
    "reports": RUN_REPORTS_SCHEMA,
    "principles_context": PRINCIPLES_CONTEXT_SCHEMA,
    "trade_clusters": TRADE_CLUSTERS_SCHEMA,
    "features": FEATURES_SCHEMA,
    "features_corr": FEATURES_CORR_SCHEMA,
    "macro": MACRO_SCHEMA,
    "zones_state": ZONES_STATE_SCHEMA,
    "pcr_a": PCRA_SCHEMA,
}


def get_table_schema(key: str) -> TableSchema:
    try:
        return TABLE_SCHEMAS[key]
    except KeyError as e:
        raise KeyError(f"No TableSchema registered for key={key!r}") from e


def enforce_table(
    df: pl.DataFrame,
    key: str,
    *,
    allow_extra: bool = True,
    reorder: bool = False,
) -> pl.DataFrame:
    return enforce_schema(df, get_table_schema(key), allow_extra=allow_extra, reorder=reorder)
