from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    """
    Load snapshots/<SNAPSHOT_ID>.json (stored as YAML/JSON) into a SnapshotManifest.
    """
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
    """
    Logical schema for a Parquet-backed table.

    Type strings:
      - "string", "int", "double", "boolean", "timestamp", "date"
      - "array<string>"
    """

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
        return pl.Datetime("us", time_zone="UTC")
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
    """
    Ensure df contains all schema.columns with correct dtypes.

    - Adds missing schema columns as null (typed).
    - Casts existing schema columns using strict=False.
    - If allow_extra=False, drops columns not in schema.
    - If reorder=True, orders schema cols first (extras appended).

    Note: legacy columns (e.g. "trading_day") are tolerated via allow_extra=True
    but are not required by Phase B.
    """
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
        out = out.select(list(schema.columns.keys()))
        return out

    if reorder:
        schema_cols = [c for c in schema.columns.keys() if c in out.columns]
        extras = [c for c in out.columns if c not in schema.columns]
        out = out.select(schema_cols + extras)

    return out


# ---------------------------------------------------------------------------
# Concrete table schemas (dt is canonical; trading_day is legacy/optional)
# ---------------------------------------------------------------------------

WINDOWS_SCHEMA = TableSchema(
    name="data/windows",
    partition_cols=["run_id", "instrument", "anchor_tf", "dt"],
    columns={
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
        'carry_ts_ts_regime': 'string',
        'carry_ts_ts_slope': 'double',
        'pac_entry_immediate_followthrough_flag': 'boolean',
        'pac_entry_immediate_rejection_flag': 'boolean',
        'pac_micro_state_anchor': 'string',
        'pac_micro_state_end': 'string',
        'pac_micro_state_mid': 'string',
        'pac_micro_state_post_entry': 'string',
        'pac_micro_state_pre_entry': 'string',
        'pac_micro_state_start': 'string',
        'pac_path_regime': 'string',
        'stat_ts_autocorr1': 'double',
        'stat_ts_kurtosis': 'double',
        'stat_ts_range': 'double',
        'stat_ts_range_zscore': 'double',
        'stat_ts_return': 'double',
        'stat_ts_return_zscore': 'double',
        'stat_ts_skew': 'double',
        'stat_ts_vol': 'double',
        'stat_ts_vol_zscore': 'double',
        'unsup_regime_confidence': 'double',
        'vol_range_compress_flag': 'boolean',
        'vol_range_expand_flag': 'boolean',
        'vol_range_lookback_bars': 'int',
        'vol_range_range_z': 'double',
        'vol_range_realised_range': 'double',
        'vol_range_regime_confidence': 'double',
        'vol_range_regime_id': 'string',
        'vol_range_state': 'string',
    },
)

TRADE_PATHS_SCHEMA = TableSchema(
    name="data/trade_paths",
    # Phase B long-format partitioning (evaluation-aware). Phase A callers will still write the
    # legacy layout (no eval ids) but the schema remains stable.
    partition_cols=["run_id", "paradigm_id", "principle_id", "candidate_id", "experiment_id", "instrument", "dt"],
    columns={
        # -------------------------------------------------------------------
        # Tier A — Engine-required identity / partition (contract-critical)
        # -------------------------------------------------------------------
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

        # Engine-defined cluster identity (microbatch cluster_id)
        "instrument_cluster_id": "string",

        # Minimum direction/timeframe keys (contract hardening)
        "side": "string",
        "anchor_tf": "string",
        "tf_entry": "string",

        # -------------------------------------------------------------------
        # Tier B — Engine-owned but allowed-null early (stable placeholders)
        # -------------------------------------------------------------------
        # Entry/exit timestamps and prices
        "entry_ts": "timestamp",
        "exit_ts": "timestamp",
        "entry_px": "double",
        "exit_px": "double",
        "sl_px": "double",
        "tp_px": "double",

        # Session/time context
        "minutes_since_session_open_bucket": "string",
        "minutes_since_session_open": "int",
        "day_of_week_entry": "int",
        "tod_bucket": "string",
        "dow_bucket": "string",

        # Regime/context at entry (and optional path summary)
        "vol_regime_entry": "string",
        "trend_regime_entry": "string",
        "macro_state_entry": "string",
        "macro_state_path": "string",
        "corr_cluster_entry": "string",
        "micro_corr_regime_entry": "string",
        "micro_corr_flip_flag": "boolean",
        "unsup_regime_id_entry": "string",

        # Profiles and governance-at-entry
        "entry_profile_id": "string",
        "management_profile_id": "string",
        "principle_status_at_entry": "string",
        "rail_version_at_entry": "string",

        # ZMF / VP semantics at entry
        "zone_behaviour_type_bucket_entry": "string",
        "zone_freshness_bucket_entry": "string",
        "zone_stack_depth_bucket_entry": "string",
        "zone_htf_confluence_bucket_entry": "string",
        "zone_vp_type_bucket_entry": "string",
        "zone_vp_skew_entry": "double",
        "zone_vp_poc_relative_position_entry": "double",

        # Lifecycle/opportunity and counterfactual hooks (allowed-null placeholders)
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

        # Critic fields (become features for pretrade/gatekeeper/trainers)
        "critic_score_at_entry": "double",
        "critic_reason_tags_at_entry": "string",  # JSON array string for portability
        "critic_reason_cluster_id": "string",

        # Entry/mgmt descriptors (placeholders; may be promoted from decisions later)
        "entry_mode": "string",
        "entry_structure_anchor": "string",
        "entry_structure_fraction_bucket": "string",
        "entry_max_bars_after_setup_bucket": "string",
        "mgmt_sl_style": "string",
        "mgmt_tp_style": "string",
        "mgmt_has_runners_flag": "boolean",
        "mgmt_has_timeout_flag": "boolean",
        "mgmt_timeout_bucket": "string",

        # Portfolio state (placeholders; portfolio step will populate)
        "portfolio_risk_bps_at_entry": "double",
        "instrument_weight_at_entry": "double",
        "cluster_weight_at_entry": "double",
        "simultaneous_trades_in_cluster": "int",
        "effective_risk_multiplier": "double",
        "effective_management_profile_id": "string",

        # Confluence/conflict (portable arrays stored as JSON strings)
        "confluence_state_entry": "string",
        "confluence_paradigms_entry": "string",     # JSON array
        "confluence_rule_ids_entry": "string",      # JSON array
        "conflict_state_entry": "string",
        "conflict_paradigms_entry": "string",       # JSON array
        "conflict_rule_ids_entry": "string",        # JSON array

        # -------------------------------------------------------------------
        # Tier C — Research-backfill-only (must exist; engine emits placeholders)
        # -------------------------------------------------------------------
        "path_cluster_id": "string",
        "path_family_id": "string",
        "path_shape": "string",
        "path_filter_primary": "string",
        "path_filter_tags_json": "string",  # JSON array string, recommended default "[]"
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
    },
)


# ---------------------------------------------------------------------------
# Decisions (stage-specific schemas)
#
# NOTE:
# - Missing columns are auto-added as typed nulls by enforce_schema(...),
#   so you can land the contract first and fill producers later.
# - Arrays are represented as JSON-encoded strings for portability.
# ---------------------------------------------------------------------------

_DECISIONS_HYPOTHESES_EXTRA_COLS: dict[str, str] = {
    # Timeframes / structure
    "anchor_tf": "string",
    "anchor_ts": "timestamp",
    "tf_entry": "string",

    # Trade intent
    "side": "string",
    "setup_logic_id": "string",
    "entry_profile_id": "string",
    "management_profile_id": "string",

    # Price hints
    "entry_hint_price": "double",
    "stop_hint_price": "double",
    "tp_hint_prices": "string",  # JSON array of floats

    # Context tuple + regimes
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

    # Portable payloads
    "context_keys_json": "string",
    "hypothesis_features_json": "string",
}

_DECISIONS_CRITIC_EXTRA_COLS: dict[str, str] = {
    "critic_score_at_entry": "double",
    "critic_reason_tags_at_entry": "string",  # JSON array of strings
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
    # Keep both during migration
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
    # Current dev stub
    "portfolio_risk_bps_at_entry": "double",

    # Target allocator outputs
    "allocated_notional": "double",
    "allocated_risk_bps": "double",
    "instrument_weight_fraction": "double",
    "cluster_weight_fraction": "double",
    "portfolio_risk_bps_after": "double",

    "allocation_mode": "string",
    "drop_reason": "string",
}

# Optional: make portfolio decisions sufficient as a fallback source for brackets
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
    columns={
        **dict(DECISIONS_SCHEMA.columns),
        **_DECISIONS_HYPOTHESES_EXTRA_COLS,
        **_DECISIONS_CRITIC_EXTRA_COLS,
    },
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
# Feature-registry derived schemas
#
# These schemas include the full "final" column set declared in conf/features_registry.yaml.
# Producers may populate a subset; enforce_schema(...) will insert typed NULL placeholders
# for all remaining contractual columns.
# ---------------------------------------------------------------------------

FEATURES_SCHEMA = TableSchema(
    name="data/features",
    partition_cols=["run_id", "instrument", "anchor_tf", "dt"],
    columns={
        'snapshot_id': 'string',
        'run_id': 'string',
        'mode': 'string',
        'dt': 'date',
        'instrument': 'string',
        'anchor_tf': 'string',
        'ts': 'timestamp',
        'atr_anchor': 'double',
        'atr_value': 'double',
        'atr_z': 'double',
        'ema_fast': 'double',
        'ema_slow': 'double',
        'eqh_flag': 'boolean',
        'eql_flag': 'boolean',
        'fvg_direction': 'string',
        'fvg_fill_state': 'string',
        'fvg_gap_ticks': 'double',
        'fvg_location_bucket': 'string',
        'fvg_origin_tf': 'string',
        'fvg_origin_ts': 'timestamp',
        'ict_struct_liquidity_tag': 'string',
        'ict_struct_pd_index': 'int',
        'ict_struct_swing_high': 'double',
        'ict_struct_swing_low': 'double',
        'liq_grab_flag': 'boolean',
        'ob_freshness_bucket': 'string',
        'ob_high': 'double',
        'ob_low': 'double',
        'ob_origin_ts': 'timestamp',
        'ob_type': 'string',
        'rsi_bucket': 'string',
        'rsi_value': 'double',
        'ta_vol__atr': 'double',
        'ta_vol__range_mean': 'double',
        'ta_vol__ret1_std': 'double',
        'vol_regime': 'string',
    },
)

FEATURES_CORR_SCHEMA = TableSchema(
    name="data/features_corr",
    partition_cols=["run_id", "instrument", "dt"],
    columns={
        'snapshot_id': 'string',
        'run_id': 'string',
        'mode': 'string',
        'dt': 'date',
        'instrument': 'string',
        'ts': 'timestamp',
        'corr_cluster_id': 'string',
        'corr_dxy': 'double',
        'corr_index_major': 'double',
        'corr_oil': 'double',
        'micro_corr_regime': 'string',
        'xs_relval_carry_rank': 'double',
        'xs_relval_momo_rank': 'double',
        'xs_relval_spread_level': 'double',
        'xs_relval_spread_zscore': 'double',
    },
)

MACRO_SCHEMA = TableSchema(
    name="data/macro",
    partition_cols=["run_id", "dt"],
    columns={
        'snapshot_id': 'string',
        'run_id': 'string',
        'mode': 'string',
        'dt': 'date',
        'ts': 'timestamp',
        'asset_relevance': 'string',
        'blackout': 'boolean',
        'country': 'string',
        'currency': 'string',
        'event_name': 'string',
        'impact_level': 'int',
        'macro_regime_confidence': 'double',
        'macro_regime_id': 'string',
        'macro_regime_label': 'string',
        'macro_state': 'string',
        'release_ts': 'timestamp',
        'severity': 'int',
        'window_end': 'timestamp',
        'window_start': 'timestamp',
    },
)

ZONES_STATE_SCHEMA = TableSchema(
    name="data/zones_state",
    partition_cols=["instrument", "anchor_tf", "dt"],
    columns={
        'snapshot_id': 'string',
        'run_id': 'string',
        'mode': 'string',
        'dt': 'date',
        'instrument': 'string',
        'anchor_tf': 'string',
        'zone_id': 'string',
        'ts': 'timestamp',
        'avg_dwell_bars': 'double',
        'fvg_origin_count': 'int',
        'liq_grab_count': 'int',
        'ob_origin_count': 'int',
        'touch_count': 'int',
        'zmf_zone_hi': 'double',
        'zmf_zone_kind': 'string',
        'zmf_zone_lo': 'double',
        'zmf_zone_mid': 'double',
        'zmf_zone_width': 'double',
        'zone_avg_dwell_bucket': 'string',
        'zone_behaviour_type': 'string',
        'zone_behaviour_type_bucket': 'string',
        'zone_freshness_bucket': 'string',
        'zone_htf_confluence': 'string',
        'zone_htf_confluence_bucket': 'string',
        'zone_index': 'int',
        'zone_lower_price': 'double',
        'zone_mid_price': 'double',
        'zone_reaction_consistency_bucket': 'string',
        'zone_reaction_strength_bucket': 'string',
        'zone_rsi_bias_bucket': 'string',
        'zone_rsi_prev_overbought_flag': 'boolean',
        'zone_rsi_prev_oversold_flag': 'boolean',
        'zone_stack_depth': 'int',
        'zone_stack_depth_bucket': 'string',
        'zone_touch_count_bucket': 'string',
        'zone_upper_price': 'double',
        'zone_vp_hvn_flag': 'boolean',
        'zone_vp_lvn_flag': 'boolean',
        'zone_vp_poc_price': 'double',
        'zone_vp_poc_relative_position': 'double',
        'zone_vp_skew': 'double',
        'zone_vp_total_volume_units': 'double',
        'zone_vp_type': 'string',
        'zone_vp_type_bucket': 'string',
        'zone_vp_volume_lower_units': 'double',
        'zone_vp_volume_upper_units': 'double',
    },
)

PCRA_SCHEMA = TableSchema(
    name="data/pcr_a",
    partition_cols=["instrument", "anchor_tf", "dt"],
    columns={
        'snapshot_id': 'string',
        'run_id': 'string',
        'mode': 'string',
        'dt': 'date',
        'instrument': 'string',
        'anchor_tf': 'string',
        'anchor_ts': 'timestamp',
        'pcr_window_ts': 'timestamp',
        'pcr_clv_bucket': 'string',
        'pcr_footprint_pattern': 'string',
        'pcr_micro_vol_bucket': 'string',
        'pcr_micro_vol_value': 'double',
        'pcr_ofi_bucket': 'string',
        'pcr_ofi_value': 'double',
        'pcr_range_bucket': 'string',
        'pcr_range_value': 'double',
        'pcr_tick_absorption_flag': 'boolean',
        'pcr_tick_concentration_flag_at': 'boolean',
        'pcr_tick_concentration_ratio_at': 'double',
        'pcr_tick_delta_units': 'double',
        'pcr_tick_delta_z': 'double',
        'pcr_tick_imbalance_flag': 'boolean',
        'pcr_tick_imbalance_ratio': 'double',
        'pcr_tick_sweep_flag': 'boolean',
        'pcr_tick_total_volume_units': 'double',
    },
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

