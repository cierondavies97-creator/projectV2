from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import polars as pl

from engine.config.features_auto import get_threshold_float, load_features_auto
from engine.paradigms.registry import RunContextLike

log = logging.getLogger(__name__)


def _safe_parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def critic_feature_hook(
    *,
    window_row: Mapping[str, Any],
    zones_state: Mapping[str, Any] | None,
    pcra: Mapping[str, Any] | None,
    macro: Mapping[str, Any] | None,
    features_corr: Mapping[str, Any] | None,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """
    Deterministic critic feature hook.

    Aligns with the doc signature: window_row / zones_state / pcra / macro / features_corr + thresholds.
    """
    zones = zones_state or {}
    pcra = pcra or {}
    macro = macro or {}
    features_corr = features_corr or {}

    feature_vector = {
        "instrument": window_row.get("instrument"),
        "anchor_tf": window_row.get("anchor_tf"),
        "anchor_ts": window_row.get("anchor_ts"),
        "tod_bucket": window_row.get("tod_bucket"),
        "dow_bucket": window_row.get("dow_bucket"),
        "vol_regime": window_row.get("vol_regime"),
        "trend_regime": window_row.get("trend_regime"),
        "macro_state": window_row.get("macro_state"),
        "micro_corr_regime": window_row.get("micro_corr_regime"),
        "corr_cluster_id": window_row.get("corr_cluster_id"),
        "unsup_regime_id": window_row.get("unsup_regime_id"),
        "macro_is_blackout": macro.get("macro_is_blackout"),
        "macro_blackout_max_impact": macro.get("macro_blackout_max_impact"),
        "zone_behaviour_type_bucket": zones.get("zone_behaviour_type_bucket"),
        "zone_freshness_bucket": zones.get("zone_freshness_bucket"),
        "zone_stack_depth_bucket": zones.get("zone_stack_depth_bucket"),
        "zone_htf_confluence_bucket": zones.get("zone_htf_confluence_bucket"),
        "zone_vp_type_bucket": zones.get("zone_vp_type_bucket"),
        "stat_ts_vol_zscore": pcra.get("stat_ts_vol_zscore"),
        "ict_struct_swing_high": pcra.get("ict_struct_swing_high"),
        "ict_struct_swing_low": pcra.get("ict_struct_swing_low"),
        "threshold_vol_z_abs_min": thresholds.get("vol_z_abs_min"),
        "threshold_vol_z_abs_max": thresholds.get("vol_z_abs_max"),
        "threshold_macro_impact": thresholds.get("macro_impact_threshold"),
    }

    feature_vector.update(
        {
            "features_corr_regime": features_corr.get("micro_corr_regime"),
            "features_corr_cluster": features_corr.get("corr_cluster_id"),
        }
    )

    return feature_vector


def _score_from_features(feature_vector: Mapping[str, Any], thresholds: Mapping[str, float]) -> tuple[float, list[str]]:
    score = 0.0
    tags: list[str] = []

    macro_is_blackout = bool(feature_vector.get("macro_is_blackout") or False)
    if macro_is_blackout:
        score -= thresholds.get("macro_blackout_penalty", 1.0)
        tags.append("macro_blackout")

    macro_impact = feature_vector.get("macro_blackout_max_impact")
    if macro_impact is not None:
        try:
            macro_impact_value = float(macro_impact)
        except (TypeError, ValueError):
            macro_impact_value = None
        if macro_impact_value is not None and macro_impact_value >= thresholds.get("macro_impact_threshold", 3.0):
            score -= thresholds.get("macro_impact_penalty", 0.4)
            tags.append("macro_blackout_impact_high")

    vol_z = feature_vector.get("stat_ts_vol_zscore")
    if vol_z is not None:
        try:
            vol_z_value = float(vol_z)
        except (TypeError, ValueError):
            vol_z_value = None
        if vol_z_value is not None:
            vol_z_abs = min(abs(vol_z_value), thresholds.get("vol_z_abs_max", 3.0))
            if vol_z_abs >= thresholds.get("vol_z_abs_min", 1.0):
                score += thresholds.get("volatility_bonus", 0.5)
                tags.append("volatility_confluence")
            else:
                score -= thresholds.get("volatility_penalty", 0.2)
                tags.append("volatility_below_threshold")
    else:
        tags.append("volatility_missing")

    swing_high = feature_vector.get("ict_struct_swing_high")
    swing_low = feature_vector.get("ict_struct_swing_low")
    if swing_high in (1, 1.0, True) or swing_low in (1, 1.0, True):
        score += thresholds.get("structure_bonus", 0.3)
        tags.append("structural_confluence")

    vol_regime = str(feature_vector.get("vol_regime") or "").lower()
    trend_regime = str(feature_vector.get("trend_regime") or "").lower()
    if vol_regime and trend_regime:
        if "low" in vol_regime and "strong" in trend_regime:
            score -= thresholds.get("regime_mismatch_penalty", 0.15)
            tags.append("regime_mismatch_vol_low_trend_strong")
        if "high" in vol_regime and "range" in trend_regime:
            score -= thresholds.get("regime_mismatch_penalty", 0.15)
            tags.append("regime_mismatch_vol_high_range")
    else:
        tags.append("regime_context_missing")

    micro_corr_regime = str(feature_vector.get("micro_corr_regime") or "").lower()
    if "high" in micro_corr_regime or "stress" in micro_corr_regime:
        score -= thresholds.get("corr_penalty", 0.1)
        tags.append("corr_regime_high")

    return score, tags


def score_trades(
    ctx: RunContextLike,
    trade_paths: pl.DataFrame,
    decisions_hypotheses: pl.DataFrame,
    critic_cfg: Mapping[str, Any],
) -> pl.DataFrame:
    """
    Phase A ICT critic: context-conditioned scorer.
    Returns decisions_critic (one row per trade_id).
    """
    if trade_paths is None or trade_paths.is_empty():
        log.info(
            "ict.critic: no trades for snapshot_id=%s run_id=%s; skipping critic.",
            ctx.snapshot_id,
            ctx.run_id,
        )
        return pl.DataFrame(
            {
                "snapshot_id": pl.Series([], dtype=pl.Utf8),
                "run_id": pl.Series([], dtype=pl.Utf8),
                "mode": pl.Series([], dtype=pl.Utf8),
                "paradigm_id": pl.Series([], dtype=pl.Utf8),
                "principle_id": pl.Series([], dtype=pl.Utf8),
                "instrument": pl.Series([], dtype=pl.Utf8),
                "trade_id": pl.Series([], dtype=pl.Utf8),
                "critic_score_at_entry": pl.Series([], dtype=pl.Float64),
                "critic_reason_tags_at_entry": pl.Series([], dtype=pl.Utf8),
                "critic_reason_cluster_id": pl.Series([], dtype=pl.Utf8),
            }
        )

    base_cols = [
        "snapshot_id",
        "run_id",
        "mode",
        "paradigm_id",
        "principle_id",
        "instrument",
        "trade_id",
    ]
    trade_extra = [c for c in ["anchor_tf", "entry_ts", "anchor_ts"] if c in trade_paths.columns]
    base = trade_paths.select(base_cols + trade_extra).unique(maintain_order=True)

    decisions_ctx = decisions_hypotheses
    if decisions_ctx is not None and not decisions_ctx.is_empty():
        join_keys = [c for c in ["trade_id", "instrument"] if c in base.columns and c in decisions_ctx.columns]

        ctx_cols = [
            "tod_bucket",
            "dow_bucket",
            "vol_regime",
            "trend_regime",
            "macro_state",
            "macro_is_blackout",
            "macro_blackout_max_impact",
            "micro_corr_regime",
            "corr_cluster_id",
            "unsup_regime_id",
            "anchor_tf",
            "anchor_ts",
            "hypothesis_features_json",
            "context_keys_json",
        ]
        ctx_cols = [c for c in ctx_cols if c in decisions_ctx.columns]

        # CRITICAL: only bring columns that do NOT already exist on base.
        # This prevents Polars from generating *_right columns (and avoids DuplicateError).
        ctx_cols_to_add = [c for c in ctx_cols if c not in base.columns]

        if join_keys and ctx_cols_to_add:
            base = base.join(
                decisions_ctx.select(join_keys + ctx_cols_to_add),
                on=join_keys,
                how="left",
            )


    features_auto = load_features_auto()
    paradigm_id = str(getattr(ctx, "paradigm_id", "dev_baseline") or "dev_baseline")
    vol_z_abs_min = get_threshold_float(
        features_auto=features_auto,
        paradigm_id=paradigm_id,
        family="stat_ts",
        feature="stat_ts_vol_zscore",
        threshold_key="abs_min_for_signal",
        default=1.0,
    )
    vol_z_abs_max = get_threshold_float(
        features_auto=features_auto,
        paradigm_id=paradigm_id,
        family="stat_ts",
        feature="stat_ts_vol_zscore",
        threshold_key="abs_max_clip",
        default=3.0,
    )

    thresholds = {
        "vol_z_abs_min": vol_z_abs_min,
        "vol_z_abs_max": vol_z_abs_max,
        "macro_impact_threshold": float(critic_cfg.get("macro_impact_threshold", 3.0)),
        "macro_blackout_penalty": float(critic_cfg.get("macro_blackout_penalty", 1.0)),
        "macro_impact_penalty": float(critic_cfg.get("macro_impact_penalty", 0.4)),
        "volatility_bonus": float(critic_cfg.get("volatility_bonus", 0.5)),
        "volatility_penalty": float(critic_cfg.get("volatility_penalty", 0.2)),
        "structure_bonus": float(critic_cfg.get("structure_bonus", 0.3)),
        "regime_mismatch_penalty": float(critic_cfg.get("regime_mismatch_penalty", 0.15)),
        "corr_penalty": float(critic_cfg.get("corr_penalty", 0.1)),
    }

    scores: list[float] = []
    tags_json: list[str] = []
    cluster_ids: list[str] = []

    for row in base.iter_rows(named=True):
        features_json = _safe_parse_json(row.get("hypothesis_features_json"))
        context_json = _safe_parse_json(row.get("context_keys_json"))

        window_row = {
            "instrument": row.get("instrument"),
            "anchor_tf": row.get("anchor_tf"),
            "anchor_ts": row.get("anchor_ts"),
            "tod_bucket": row.get("tod_bucket"),
            "dow_bucket": row.get("dow_bucket"),
            "vol_regime": row.get("vol_regime"),
            "trend_regime": row.get("trend_regime"),
            "macro_state": row.get("macro_state"),
            "micro_corr_regime": row.get("micro_corr_regime"),
            "corr_cluster_id": row.get("corr_cluster_id"),
            "unsup_regime_id": row.get("unsup_regime_id"),
        }

        zones_state = {
            "zone_behaviour_type_bucket": context_json.get("zone_behaviour_type_bucket"),
            "zone_freshness_bucket": context_json.get("zone_freshness_bucket"),
            "zone_stack_depth_bucket": context_json.get("zone_stack_depth_bucket"),
            "zone_htf_confluence_bucket": context_json.get("zone_htf_confluence_bucket"),
            "zone_vp_type_bucket": context_json.get("zone_vp_type_bucket"),
        }

        pcra = {
            "stat_ts_vol_zscore": features_json.get("stat_ts_vol_zscore"),
            "ict_struct_swing_high": features_json.get("ict_struct_swing_high"),
            "ict_struct_swing_low": features_json.get("ict_struct_swing_low"),
        }

        macro = {
            "macro_is_blackout": row.get("macro_is_blackout"),
            "macro_blackout_max_impact": row.get("macro_blackout_max_impact"),
            "macro_state": row.get("macro_state"),
        }

        features_corr = {
            "micro_corr_regime": row.get("micro_corr_regime"),
            "corr_cluster_id": row.get("corr_cluster_id"),
        }

        feature_vector = critic_feature_hook(
            window_row=window_row,
            zones_state=zones_state,
            pcra=pcra,
            macro=macro,
            features_corr=features_corr,
            thresholds=thresholds,
        )

        score, reason_tags = _score_from_features(feature_vector, thresholds)
        if feature_vector.get("anchor_tf"):
            reason_tags.append(f"anchor_tf:{feature_vector.get('anchor_tf')}")
        if feature_vector.get("instrument"):
            reason_tags.append(f"instrument:{feature_vector.get('instrument')}")

        scores.append(float(score))
        tags_json.append(json.dumps(reason_tags, separators=(",", ":")))
        cluster_ids.append(reason_tags[0] if reason_tags else "ict_context_neutral")

    decisions_critic = base.with_columns(
        pl.Series("critic_score_at_entry", scores, dtype=pl.Float64),
        pl.Series("critic_reason_tags_at_entry", tags_json, dtype=pl.Utf8),
        pl.Series("critic_reason_cluster_id", cluster_ids, dtype=pl.Utf8),
    )

    return decisions_critic