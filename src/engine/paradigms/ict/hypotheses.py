from __future__ import annotations

import logging
from typing import Any

import polars as pl

from engine.config.features_auto import get_threshold_float, load_features_auto
from engine.core.schema import TRADE_PATHS_SCHEMA
from engine.core.timegrid import tf_to_truncate_rule
from engine.paradigms.registry import register_hypotheses_builder

log = logging.getLogger(__name__)

HypoOut = tuple[pl.DataFrame, pl.DataFrame]
_SENTINEL = "∅"


# -----------------------------------------------------------------------------
# Schema helpers (trade_paths padding)
# -----------------------------------------------------------------------------

def _polars_dtype(type_str: str) -> pl.DataType:
    t = type_str.strip().lower()
    if t in ("string", "utf8", "str"):
        return pl.Utf8
    if t in ("int", "int64", "i64"):
        return pl.Int64
    if t in ("double", "float", "float64", "f64"):
        return pl.Float64
    if t in ("boolean", "bool"):
        return pl.Boolean
    # Canonical engine timestamp unit: microseconds (no TZ unless your global schema enforces it elsewhere)
    if t in ("timestamp", "datetime"):
        return pl.Datetime("us")
    if t in ("date",):
        return pl.Date
    if t in ("array<string>", "list<string>"):
        return pl.List(pl.Utf8)
    return pl.Utf8


def _default_expr_for_type(col_name: str, col_type: str) -> pl.Expr:
    t = col_type.strip().lower()

    if t in ("string", "utf8", "str"):
        return pl.lit("").cast(pl.Utf8).alias(col_name)
    if t in ("int", "int64", "i64"):
        return pl.lit(None).cast(pl.Int64).alias(col_name)
    if t in ("double", "float", "float64", "f64"):
        return pl.lit(None).cast(pl.Float64).alias(col_name)
    if t in ("boolean", "bool"):
        return pl.lit(False).cast(pl.Boolean).alias(col_name)
    if t in ("timestamp", "datetime"):
        return pl.lit(None).cast(pl.Datetime("us")).alias(col_name)
    if t in ("date",):
        return pl.lit(None).cast(pl.Date).alias(col_name)
    if t in ("array<string>", "list<string>"):
        return pl.lit(None).cast(pl.List(pl.Utf8)).alias(col_name)

    return pl.lit(None).cast(_polars_dtype(col_type)).alias(col_name)


def _ensure_trade_paths_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pad a minimally-emitted trade_paths frame to TRADE_PATHS_SCHEMA columns.

    This is intentionally permissive for Phase A/early Phase B: it only adds
    missing columns; it does not drop extras.
    """
    if df is None or df.is_empty():
        return pl.DataFrame() if df is None else df

    existing = set(df.columns)
    missing_exprs: list[pl.Expr] = []
    for name, typ in TRADE_PATHS_SCHEMA.columns.items():
        if name not in existing:
            missing_exprs.append(_default_expr_for_type(name, typ))

    out = df.with_columns(missing_exprs) if missing_exprs else df

    # Best-effort casts for schema columns that already exist (prevents drift).
    cast_exprs: list[pl.Expr] = []
    for name, typ in TRADE_PATHS_SCHEMA.columns.items():
        if name in out.columns:
            cast_exprs.append(pl.col(name).cast(_polars_dtype(typ), strict=False).alias(name))
    if cast_exprs:
        out = out.with_columns(cast_exprs)

    return out


def _truncate_anchor_ts_by_tf(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame() if df is None else df
    if "anchor_tf" not in df.columns or "anchor_ts" not in df.columns:
        return df

    frames: list[pl.DataFrame] = []
    for (anchor_tf,), grp in df.group_by(["anchor_tf"], maintain_order=True):
        if anchor_tf is None:
            frames.append(grp)
            continue
        rule = tf_to_truncate_rule(str(anchor_tf))
        frames.append(
            grp.with_columns(
                pl.col("anchor_ts")
                .cast(pl.Datetime("us"), strict=False)
                .dt.truncate(rule)
                .alias("anchor_ts")
            )
        )
    return pl.concat(frames, how="vertical") if frames else df


def _safe_truncate_rule(tf: str | None) -> str | None:
    if tf is None:
        return None
    try:
        return tf_to_truncate_rule(str(tf))
    except ValueError:
        return None


def _add_entry_alignment_flag(
    df: pl.DataFrame,
    *,
    tf_col: str,
    flag_col: str,
) -> pl.DataFrame:
    if df is None or df.is_empty():
        return pl.DataFrame() if df is None else df
    if "entry_ts" not in df.columns or tf_col not in df.columns:
        return df

    frames: list[pl.DataFrame] = []
    for (tf_val,), grp in df.group_by([tf_col], maintain_order=True):
        rule = _safe_truncate_rule(str(tf_val) if tf_val is not None else None)
        if rule is None:
            frames.append(grp.with_columns(pl.lit(None).cast(pl.Boolean).alias(flag_col)))
            continue
        entry_ts = pl.col("entry_ts").cast(pl.Datetime("us"), strict=False)
        aligned = entry_ts.dt.truncate(rule) == entry_ts
        frames.append(grp.with_columns(aligned.alias(flag_col)))
    return pl.concat(frames, how="vertical") if frames else df


def _coalesce_cols(df: pl.DataFrame, cols: list[str], *, dtype: pl.DataType) -> pl.Expr:
    existing = [pl.col(c).cast(dtype, strict=False) for c in cols if c in df.columns]
    if not existing:
        return pl.lit(None).cast(dtype)
    return pl.coalesce(existing)


def _resolve_price_sources(
    cfg: dict[str, Any],
    *,
    key: str,
    defaults_long: list[str],
    defaults_short: list[str],
) -> tuple[list[str], list[str]]:
    params = cfg.get("params") if isinstance(cfg, dict) else {}
    sources = params.get(key) if isinstance(params, dict) else None
    if isinstance(sources, dict):
        long_list = [str(x) for x in sources.get("long", []) if str(x).strip()]
        short_list = [str(x) for x in sources.get("short", []) if str(x).strip()]
        return (long_list or defaults_long), (short_list or defaults_short)
    if isinstance(sources, list):
        flat = [str(x) for x in sources if str(x).strip()]
        return (flat or defaults_long), (flat or defaults_short)
    return defaults_long, defaults_short


# -----------------------------------------------------------------------------
# Decisions helpers
# -----------------------------------------------------------------------------

def _empty_decisions(ctx: Any, paradigm_id: str, principle_id: str) -> pl.DataFrame:
    # Decisions table is stage-written later; keep it minimal and typed.
    return pl.DataFrame(
        {
            "snapshot_id": pl.Series([], dtype=pl.Utf8),
            "run_id": pl.Series([], dtype=pl.Utf8),
            "mode": pl.Series([], dtype=pl.Utf8),
            "paradigm_id": pl.Series([], dtype=pl.Utf8),
            "principle_id": pl.Series([], dtype=pl.Utf8),
            "instrument": pl.Series([], dtype=pl.Utf8),
            "trade_id": pl.Series([], dtype=pl.Utf8),
            "macro_is_blackout": pl.Series([], dtype=pl.Boolean),
            "macro_blackout_max_impact": pl.Series([], dtype=pl.Int64),
        }
    )


# -----------------------------------------------------------------------------
# Feature join (FIXED: align RHS timestamp dtype to LHS, do not force ns)
# -----------------------------------------------------------------------------

def _join_windows_with_features(windows: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """
    Join windows with features using stable join keys.

    Key requirement: Polars requires exact dtype matches on join keys. Upstream
    feature families may emit timestamps in different time units (ns/us). We align
    feature join key dtypes to the windows schema (authoritative anchor grid).
    """
    if windows is None or windows.is_empty():
        return pl.DataFrame() if windows is None else windows
    if features is None or features.is_empty():
        return windows

    feat = features

    # Back-compat: some feature tables use "ts" instead of "anchor_ts".
    if "anchor_ts" not in feat.columns and "ts" in feat.columns:
        feat = feat.rename({"ts": "anchor_ts"})

    needed = ["instrument", "anchor_tf", "anchor_ts"]
    if any(c not in windows.columns for c in needed):
        return windows
    if any(c not in feat.columns for c in needed):
        return windows

    # Normalize string keys
    windows2 = windows.with_columns(
        [
            pl.col("instrument").cast(pl.Utf8, strict=False),
            pl.col("anchor_tf").cast(pl.Utf8, strict=False),
        ]
    )
    feat2 = feat.with_columns(
        [
            pl.col("instrument").cast(pl.Utf8, strict=False),
            pl.col("anchor_tf").cast(pl.Utf8, strict=False),
        ]
    )

    # Align RHS join key dtypes to LHS join key dtypes (fixes ns/us mismatch).
    lhs_anchor_ts_dtype = windows2.schema.get("anchor_ts")
    if lhs_anchor_ts_dtype is not None:
        feat2 = feat2.with_columns(pl.col("anchor_ts").cast(lhs_anchor_ts_dtype, strict=False).alias("anchor_ts"))

    base_keep = [
        "instrument",
        "anchor_tf",
        "anchor_ts",
        "ict_struct_swing_high",
        "ict_struct_swing_low",
        "stat_ts_vol_zscore",
        "macro_is_blackout",
        "macro_blackout_max_impact",
    ]
    keep = [c for c in base_keep if c in feat2.columns]
    if not keep:
        return windows2

    feat2 = feat2.select(keep)
    return windows2.join(feat2, on=needed, how="left")


# -----------------------------------------------------------------------------
# Candidate selection
# -----------------------------------------------------------------------------

def _select_candidate_windows(df: pl.DataFrame) -> pl.DataFrame:
    """
    Return candidate windows without implicit time-based downsampling.
    """
    if df is None or df.is_empty():
        return pl.DataFrame() if df is None else df
    return df


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------

def build_ict_hypotheses(
    ctx: Any,
    windows: pl.DataFrame,
    features: pl.DataFrame,
    principle_cfg: dict,
) -> HypoOut:
    paradigm_id = str(principle_cfg.get("paradigm_id", "ict"))
    principle_id = str(principle_cfg.get("principle_id", "ict_all_windows"))
    hypotheses_cfg = principle_cfg.get("hypotheses", {}) if isinstance(principle_cfg, dict) else {}
    tf_entry_allowlist = hypotheses_cfg.get("tf_entry_allowlist", []) if isinstance(hypotheses_cfg, dict) else []

    if windows is None or windows.is_empty():
        return _empty_decisions(ctx, paradigm_id, principle_id), pl.DataFrame()

    # Ensure minimum keys exist
    for k in ("instrument", "anchor_tf", "anchor_ts", "tf_entry"):
        if k not in windows.columns:
            log.warning("ict.hypotheses: windows missing %s; returning empty", k)
            return _empty_decisions(ctx, paradigm_id, principle_id), pl.DataFrame()

    df = windows.sort(["instrument", "anchor_tf", "anchor_ts"])
    if tf_entry_allowlist:
        tf_entries = [str(tf) for tf in tf_entry_allowlist if str(tf).strip()]
        if tf_entries:
            df = df.filter(pl.col("tf_entry").cast(pl.Utf8, strict=False).is_in(tf_entries))
            if df.is_empty():
                log.info("ict.hypotheses: tf_entry_allowlist=%s filtered all windows", tf_entries)
                return _empty_decisions(ctx, paradigm_id, principle_id), pl.DataFrame()
    df = _join_windows_with_features(df, features)

    # Threshold (safe default)
    features_auto = load_features_auto()
    vol_z_abs_min = get_threshold_float(
        features_auto=features_auto,
        paradigm_id=paradigm_id,
        family="stat_ts",
        feature="stat_ts_vol_zscore",
        threshold_key="abs_min_for_signal",
        default=1.0,
    )

    cols = set(df.columns)

    swing_expr = None
    if ("ict_struct_swing_high" in cols) or ("ict_struct_swing_low" in cols):
        swing_expr = (pl.col("ict_struct_swing_high").cast(pl.Float64, strict=False).fill_null(0.0) == 1.0) | (
            pl.col("ict_struct_swing_low").cast(pl.Float64, strict=False).fill_null(0.0) == 1.0
        )

    vol_expr = None
    if "stat_ts_vol_zscore" in cols:
        vol_expr = pl.col("stat_ts_vol_zscore").cast(pl.Float64, strict=False).fill_null(0.0).abs() > float(vol_z_abs_min)

    cond = None
    for e in (swing_expr, vol_expr):
        if e is None:
            continue
        cond = e if cond is None else (cond | e)

    candidates = df.filter(cond) if cond is not None else df
    selected = _select_candidate_windows(candidates if not candidates.is_empty() else df)

    entry_sources_long, entry_sources_short = _resolve_price_sources(
        cfg,
        key="entry_px_sources",
        defaults_long=["ict_struct_swing_low", "bos_level_px", "choch_level_px", "ict_struct_dealing_range_mid"],
        defaults_short=["ict_struct_swing_high", "bos_level_px", "choch_level_px", "ict_struct_dealing_range_mid"],
    )
    exit_sources_long, exit_sources_short = _resolve_price_sources(
        cfg,
        key="exit_px_sources",
        defaults_long=["ict_struct_dealing_range_high", "eqh_level_px", "bos_level_px"],
        defaults_short=["ict_struct_dealing_range_low", "eql_level_px", "bos_level_px"],
    )

    entry_px_expr = pl.when(pl.col("anchor_ts").dt.hour() < 12).then(
        _coalesce_cols(selected, entry_sources_long, dtype=pl.Float64)
    ).otherwise(
        _coalesce_cols(selected, entry_sources_short, dtype=pl.Float64)
    )

    exit_px_expr = pl.when(pl.col("anchor_ts").dt.hour() < 12).then(
        _coalesce_cols(selected, exit_sources_long, dtype=pl.Float64)
    ).otherwise(
        _coalesce_cols(selected, exit_sources_short, dtype=pl.Float64)
    )
    if selected.is_empty():
        return _empty_decisions(ctx, paradigm_id, principle_id), pl.DataFrame()

    snapshot_id = str(getattr(ctx, "snapshot_id", "") or "")
    run_id = str(getattr(ctx, "run_id", "") or "")
    mode = str(getattr(ctx, "mode", "") or "")

    # Ensure anchor_ts is datetime and on-grid for deterministic formatting
    selected = _truncate_anchor_ts_by_tf(selected)

    # trade_id = "{run_id}-{instrument}-{anchor_tf}-{YYYYMMDDTHHMMSS}"
    selected = selected.with_columns(
        pl.format(
            "{}-{}-{}-{}",
            pl.lit(run_id),
            pl.col("instrument").cast(pl.Utf8, strict=False),
            pl.col("anchor_tf").cast(pl.Utf8, strict=False),
            pl.col("anchor_ts").dt.strftime("%Y%m%dT%H%M%S"),
        ).alias("trade_id")
    )

    # Macro fields (defensive defaults)
    macro_is_blackout_expr = (
        pl.col("macro_is_blackout").cast(pl.Boolean, strict=False).fill_null(False)
        if "macro_is_blackout" in selected.columns
        else pl.lit(False).cast(pl.Boolean)
    )
    macro_blackout_max_impact_expr = (
        pl.col("macro_blackout_max_impact").cast(pl.Int64, strict=False)
        if "macro_blackout_max_impact" in selected.columns
        else pl.lit(None).cast(pl.Int64)
    )

    side_expr = pl.when(pl.col("anchor_ts").dt.hour() < 12).then(pl.lit("long")).otherwise(pl.lit("short"))

    # Decisions frame (stage writer will add dt later; keep minimal here)
    decisions_df = selected.select(
        [
            pl.lit(snapshot_id).cast(pl.Utf8).alias("snapshot_id"),
            pl.lit(run_id).cast(pl.Utf8).alias("run_id"),
            pl.lit(mode).cast(pl.Utf8).alias("mode"),
            pl.lit(paradigm_id).cast(pl.Utf8).alias("paradigm_id"),
            pl.lit(principle_id).cast(pl.Utf8).alias("principle_id"),
            pl.col("instrument").cast(pl.Utf8, strict=False).alias("instrument"),
            pl.col("trade_id").cast(pl.Utf8, strict=False).alias("trade_id"),
            pl.col("anchor_tf").cast(pl.Utf8, strict=False).alias("anchor_tf"),
            pl.col("anchor_ts").cast(pl.Datetime("us"), strict=False).alias("anchor_ts"),
            pl.col("tf_entry").cast(pl.Utf8, strict=False).alias("tf_entry"),
            side_expr.cast(pl.Utf8).alias("side"),
            macro_is_blackout_expr.alias("macro_is_blackout"),
            macro_blackout_max_impact_expr.alias("macro_blackout_max_impact"),
        ]
    )

    # Trade paths frame (minimal, padded to TRADE_PATHS_SCHEMA)
    # Prefer dt from windows if present; otherwise derive from anchor_ts.
    if "dt" in selected.columns:
        dt_expr = pl.col("dt").cast(pl.Date, strict=False)
    else:
        dt_expr = pl.col("anchor_ts").dt.date().cast(pl.Date, strict=False)

    side_expr = pl.when(pl.col("anchor_ts").dt.hour() < 12).then(pl.lit("long")).otherwise(pl.lit("short"))

    tp_df = selected.select(
        [
            pl.lit(snapshot_id).cast(pl.Utf8).alias("snapshot_id"),
            pl.lit(run_id).cast(pl.Utf8).alias("run_id"),
            pl.lit(mode).cast(pl.Utf8).alias("mode"),
            dt_expr.alias("dt"),
            pl.lit(paradigm_id).cast(pl.Utf8).alias("paradigm_id"),
            pl.lit(principle_id).cast(pl.Utf8).alias("principle_id"),
            # candidate/experiment are Phase B concepts; keep stable sentinel
            pl.lit(_SENTINEL).cast(pl.Utf8).alias("candidate_id"),
            pl.lit(_SENTINEL).cast(pl.Utf8).alias("experiment_id"),
            pl.col("trade_id").cast(pl.Utf8, strict=False).alias("trade_id"),
            pl.col("instrument").cast(pl.Utf8, strict=False).alias("instrument"),
            side_expr.cast(pl.Utf8).alias("side"),
            pl.col("anchor_tf").cast(pl.Utf8, strict=False).alias("anchor_tf"),
            pl.col("tf_entry").cast(pl.Utf8, strict=False).alias("tf_entry"),
            pl.col("anchor_ts").cast(pl.Datetime("us"), strict=False).alias("anchor_ts"),
            pl.col("anchor_ts").alias("entry_ts"),
            pl.lit("anchor_close").cast(pl.Utf8).alias("entry_ts_source"),
            pl.lit(None).cast(pl.Int64).alias("entry_ts_offset_ms"),
            pl.lit(True).cast(pl.Boolean).alias("entry_ts_is_aligned_anchor_tf"),
            pl.lit(None).cast(pl.Boolean).alias("entry_ts_is_aligned_entry_tf"),
            entry_px_expr.alias("entry_px"),
            exit_px_expr.alias("exit_px"),
            pl.lit("candidate").cast(pl.Utf8).alias("principle_status_at_entry"),
            pl.lit("phaseA_auto").cast(pl.Utf8).alias("entry_mode"),
        ]
    )

    tp_df = _add_entry_alignment_flag(
        tp_df,
        tf_col="tf_entry",
        flag_col="entry_ts_is_aligned_entry_tf",
    )
    tp_df = _ensure_trade_paths_schema(tp_df)

    log.info("ict.hypotheses: produced decisions=%d trade_paths=%d", decisions_df.height, tp_df.height)
    return decisions_df, tp_df


def register() -> None:
    # Bind exact key used by snapshot resolution
    register_hypotheses_builder("ict", "ict_all_windows", build_ict_hypotheses)
    # Optional: allow any other ict principle to fall back here during Phase A
    register_hypotheses_builder("ict", "*", build_ict_hypotheses)


# -----------------------------------------------------------------------------
# Back-compat API expected by engine.paradigms.register_all
# -----------------------------------------------------------------------------

def build_hypotheses_for_windows(
    ctx: Any, windows: pl.DataFrame, features: pl.DataFrame, principle_cfg: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compatibility wrapper for Phase A wiring.

    register_all.py imports build_hypotheses_for_windows. Internally we delegate
    to the current implementation.
    """
    return build_ict_hypotheses(ctx, windows, features, principle_cfg)
