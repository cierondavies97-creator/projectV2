from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from engine.features._shared import ensure_sorted
from engine.features.market_structure.dealing_range.events import compute_event_flags
from engine.features.market_structure.dealing_range.features import compute_range_descriptors


DEFAULT_CONFIG: dict[str, Any] = {
    "lookback_bars": 50,
    "atr_window": 14,
    "width_min_atr": 1.0,
    "p_inside_min": 0.7,
    "tests_min": 3,
    "test_atr_mult": 0.2,
    "probe_atr_mult": 0.5,
    "reclaim_atr_mult": 0.2,
    "accept_atr_mult": 0.5,
    "accept_bars_min": 3,
    "reclaim_bars_max": 3,
    "trend_atr_mult": 1.5,
    "trend_width_mult": 1.0,
    "trend_bars_min": 5,
    "reentry_bars_max": 3,
    "phase_version": "dr_phase_v1",
    "threshold_bundle_id": "dr_thresholds_v1",
    "micro_policy_id": "micro_policy_v1",
    "jump_policy_id": "jump_policy_v1",
    "impact_policy_id": "impact_policy_v1",
    "options_policy_id": "options_policy_v1",
}


REQUIRED_DR_FIELDS = [
    "dr_id",
    "dr_phase",
    "dr_low",
    "dr_high",
    "dr_mid",
    "dr_width",
    "dr_width_atr",
    "dr_age_bars",
    "dr_start_ts",
    "dr_last_update_ts",
    "dr_reason_code",
]


@dataclass
class _DRState:
    phase: str | None = None
    dr_id: str | None = None
    start_ts: Any | None = None
    age_bars: int = 0
    last_update_ts: Any | None = None
    probe_side: str | None = None
    probe_age: int = 0


def _merge_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _policy_frame(base: pl.DataFrame, cfg: Mapping[str, Any]) -> pl.DataFrame:
    return base.with_columns(
        pl.lit(str(cfg.get("phase_version"))).cast(pl.Utf8).alias("phase_version"),
        pl.lit(str(cfg.get("threshold_bundle_id"))).cast(pl.Utf8).alias("threshold_bundle_id"),
        pl.lit(str(cfg.get("micro_policy_id"))).cast(pl.Utf8).alias("micro_policy_id"),
        pl.lit(str(cfg.get("jump_policy_id"))).cast(pl.Utf8).alias("jump_policy_id"),
        pl.lit(str(cfg.get("impact_policy_id"))).cast(pl.Utf8).alias("impact_policy_id"),
        pl.lit(str(cfg.get("options_policy_id"))).cast(pl.Utf8).alias("options_policy_id"),
    )


def _empty_output(base: pl.DataFrame, cfg: Mapping[str, Any]) -> pl.DataFrame:
    out = base
    for col in REQUIRED_DR_FIELDS:
        if col in ("dr_age_bars",):
            out = out.with_columns(pl.lit(None).cast(pl.Int64).alias(col))
        elif col in ("dr_start_ts", "dr_last_update_ts"):
            out = out.with_columns(pl.lit(None).cast(pl.Datetime("us")).alias(col))
        elif col in ("dr_low", "dr_high", "dr_mid", "dr_width", "dr_width_atr"):
            out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        else:
            out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
    return _policy_frame(out, cfg)


def _fold_group(df: pl.DataFrame, *, cfg: Mapping[str, Any], dr_id_prefix: str) -> pl.DataFrame:
    df = ensure_sorted(df, ["anchor_ts"], where="dealing_range.fold")
    df = compute_range_descriptors(df, cfg=cfg)
    df = compute_event_flags(df, cfg=cfg)

    width_min_atr = float(cfg["width_min_atr"])
    p_inside_min = float(cfg["p_inside_min"])
    tests_min = int(cfg["tests_min"])
    accept_bars_min = int(cfg["accept_bars_min"])
    reclaim_bars_max = int(cfg["reclaim_bars_max"])
    trend_bars_min = int(cfg["trend_bars_min"])

    state = _DRState()
    dr_counter = 0

    outside_up_run = 0
    outside_dn_run = 0
    trend_run = 0

    out_rows: list[dict[str, Any]] = []

    for row in df.iter_rows(named=True):
        ts = row["anchor_ts"]
        dr_low = row.get("dr_low")
        dr_high = row.get("dr_high")
        dr_mid = row.get("dr_mid")
        dr_width = row.get("dr_width")
        dr_width_atr = row.get("dr_width_atr")

        if dr_low is None or dr_high is None or dr_width is None:
            out_rows.append(
                {
                    "instrument": row["instrument"],
                    "anchor_tf": row["anchor_tf"],
                    "anchor_ts": ts,
                    "dr_id": None,
                    "dr_phase": None,
                    "dr_low": None,
                    "dr_high": None,
                    "dr_mid": None,
                    "dr_width": None,
                    "dr_width_atr": None,
                    "dr_age_bars": None,
                    "dr_start_ts": None,
                    "dr_last_update_ts": None,
                    "dr_reason_code": None,
                }
            )
            continue

        inside_ratio = row.get("_inside_ratio")
        tests_count = row.get("_tests_count")
        inside = bool(row.get("_inside", False))
        probe_low = bool(row.get("_probe_low", False))
        probe_high = bool(row.get("_probe_high", False))
        reclaim_from_low = bool(row.get("_reclaim_from_low", False))
        reclaim_from_high = bool(row.get("_reclaim_from_high", False))
        outside_up = bool(row.get("_outside_up", False))
        outside_dn = bool(row.get("_outside_dn", False))
        trend_far = bool(row.get("_trend_far", False))

        if outside_up:
            outside_up_run += 1
        else:
            outside_up_run = 0

        if outside_dn:
            outside_dn_run += 1
        else:
            outside_dn_run = 0

        accept_up = outside_up_run >= accept_bars_min
        accept_dn = outside_dn_run >= accept_bars_min

        if trend_far:
            trend_run += 1
        else:
            trend_run = 0

        reason = None
        if state.phase is None:
            if (
                dr_width_atr is not None
                and dr_width_atr >= width_min_atr
                and inside_ratio is not None
                and inside_ratio >= p_inside_min
                and tests_count is not None
                and tests_count >= tests_min
            ):
                dr_counter += 1
                state.phase = "B"
                state.dr_id = f"{dr_id_prefix}-{dr_counter:04d}"
                state.start_ts = ts
                state.age_bars = 0
                state.last_update_ts = ts
                reason = "ENTER_B_RANGE_VALID"
            else:
                out_rows.append(
                    {
                        "instrument": row["instrument"],
                        "anchor_tf": row["anchor_tf"],
                        "anchor_ts": ts,
                        "dr_id": None,
                        "dr_phase": None,
                        "dr_low": None,
                        "dr_high": None,
                        "dr_mid": None,
                        "dr_width": None,
                        "dr_width_atr": None,
                        "dr_age_bars": None,
                        "dr_start_ts": None,
                        "dr_last_update_ts": None,
                        "dr_reason_code": None,
                    }
                )
                continue
        elif state.phase == "B":
            if dr_width_atr is not None and dr_width_atr < width_min_atr:
                reason = "RESET_RANGE_INVALID"
                state = _DRState()
            elif accept_up or accept_dn:
                state.phase = "D"
                reason = "EXIT_B_ACCEPTED_BREAK"
            elif probe_low or probe_high:
                state.phase = "C"
                state.probe_side = "LOW" if probe_low else "HIGH"
                state.probe_age = 0
                reason = "ENTER_C_PROBE"
            else:
                reason = "STAY_B_AUCTION_IN_RANGE"
        elif state.phase == "C":
            state.probe_age += 1
            if accept_up or accept_dn:
                state.phase = "D"
                state.probe_side = None
                reason = "EXIT_C_ACCEPTED_BREAK"
            elif (state.probe_side == "LOW" and reclaim_from_low) or (
                state.probe_side == "HIGH" and reclaim_from_high
            ):
                state.phase = "B"
                state.probe_side = None
                reason = "EXIT_C_RECLAIMED"
            elif state.probe_age > reclaim_bars_max:
                state.phase = "B"
                state.probe_side = None
                reason = "EXIT_C_TIMEOUT"
            else:
                reason = "STAY_C_PROBE"
        elif state.phase == "D":
            if trend_run >= trend_bars_min:
                state.phase = "E"
                reason = "EXIT_D_TREND_ESTABLISHED"
            elif inside:
                state.phase = "B"
                reason = "REENTER_B_FROM_D"
            else:
                reason = "STAY_D_ACCEPTED"
        elif state.phase == "E":
            if inside:
                state.phase = "B"
                reason = "REENTER_B_FROM_E"
            else:
                reason = "STAY_E_TREND"

        if state.phase is None:
            out_rows.append(
                {
                    "instrument": row["instrument"],
                    "anchor_tf": row["anchor_tf"],
                    "anchor_ts": ts,
                    "dr_id": None,
                    "dr_phase": None,
                    "dr_low": None,
                    "dr_high": None,
                    "dr_mid": None,
                    "dr_width": None,
                    "dr_width_atr": None,
                    "dr_age_bars": None,
                    "dr_start_ts": None,
                    "dr_last_update_ts": None,
                    "dr_reason_code": reason,
                }
            )
            continue

        state.age_bars += 1
        state.last_update_ts = ts

        out_rows.append(
            {
                "instrument": row["instrument"],
                "anchor_tf": row["anchor_tf"],
                "anchor_ts": ts,
                "dr_id": state.dr_id,
                "dr_phase": state.phase,
                "dr_low": float(dr_low) if dr_low is not None else None,
                "dr_high": float(dr_high) if dr_high is not None else None,
                "dr_mid": float(dr_mid) if dr_mid is not None else None,
                "dr_width": float(dr_width) if dr_width is not None else None,
                "dr_width_atr": float(dr_width_atr) if dr_width_atr is not None else None,
                "dr_age_bars": state.age_bars,
                "dr_start_ts": state.start_ts,
                "dr_last_update_ts": state.last_update_ts,
                "dr_reason_code": reason,
            }
        )

    out = pl.DataFrame(out_rows)
    return out


def fold_state_machine(
    windows: pl.DataFrame,
    candles: pl.DataFrame | None,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> pl.DataFrame:
    """
    Compute per-row dealing range fields keyed by (instrument, anchor_tf, anchor_ts).
    """
    if windows is None or windows.is_empty():
        return pl.DataFrame()

    cfg = _merge_cfg(cfg)

    base = (
        windows.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("anchor_tf").cast(pl.Utf8),
            pl.col("anchor_ts").cast(pl.Datetime("us")),
        )
        .drop_nulls(["instrument", "anchor_tf", "anchor_ts"])
        .unique()
        .sort(["instrument", "anchor_tf", "anchor_ts"])
    )

    if candles is None or candles.is_empty():
        return _empty_output(base, cfg)

    required = {"instrument", "ts", "high", "low", "close"}
    if not required.issubset(set(candles.columns)):
        return _empty_output(base, cfg)

    c = (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("tf").cast(pl.Utf8, strict=False).alias("anchor_tf"),
        )
        .drop_nulls(["instrument", "anchor_tf", "ts"])
        .rename({"ts": "anchor_ts"})
        .sort(["instrument", "anchor_tf", "anchor_ts"])
    )

    if c.is_empty():
        return _empty_output(base, cfg)

    outputs: list[pl.DataFrame] = []
    for (instrument, anchor_tf), grp in c.group_by(["instrument", "anchor_tf"], maintain_order=True):
        dr_id_prefix = f"{instrument}:{anchor_tf}"
        out = _fold_group(grp, cfg=cfg, dr_id_prefix=dr_id_prefix)
        outputs.append(out)

    if not outputs:
        return _empty_output(base, cfg)

    dr_frame = pl.concat(outputs, how="vertical", rechunk=True)
    dr_frame = dr_frame.join(base, on=["instrument", "anchor_tf", "anchor_ts"], how="right")
    dr_frame = _policy_frame(dr_frame, cfg)

    return dr_frame
