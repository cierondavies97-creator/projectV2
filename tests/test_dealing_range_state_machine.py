from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.features.market_structure.dealing_range import fold_state_machine  # noqa: E402


def _make_candles(tf: str) -> pl.DataFrame:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=15 * idx) for idx in range(6)]
    highs = [110.0, 111.0, 110.0, 110.0, 110.0, 110.0]
    lows = [100.0, 99.0, 100.0, 100.0, 100.0, 100.0]
    closes = [105.0, 105.0, 106.0, 105.0, 105.0, 105.0]
    return pl.DataFrame(
        {
            "instrument": ["GC"] * len(ts),
            "tf": [tf] * len(ts),
            "ts": ts,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def _make_windows(candles: pl.DataFrame, tf: str) -> pl.DataFrame:
    return candles.select(
        pl.col("instrument"),
        pl.lit(tf).alias("anchor_tf"),
        pl.col("ts").alias("anchor_ts"),
    )


def test_dealing_range_state_machine_golden_m15() -> None:
    candles = _make_candles("M15")
    windows = _make_windows(candles, "M15")
    cfg = {
        "lookback_bars": 3,
        "atr_window": 2,
        "width_min_atr": 0.5,
        "p_inside_min": 0.6,
        "tests_min": 1,
        "test_atr_mult": 0.2,
        "probe_atr_mult": 0.5,
        "reclaim_atr_mult": 0.2,
        "accept_atr_mult": 0.5,
        "accept_bars_min": 2,
        "reclaim_bars_max": 2,
        "trend_atr_mult": 1.5,
        "trend_width_mult": 0.2,
        "trend_bars_min": 2,
        "reentry_bars_max": 2,
        "phase_version": "test_phase_v1",
        "threshold_bundle_id": "test_thresholds_v1",
        "micro_policy_id": "test_micro_v1",
        "jump_policy_id": "test_jump_v1",
        "impact_policy_id": "test_impact_v1",
        "options_policy_id": "test_options_v1",
    }

    out = fold_state_machine(windows, candles, cfg=cfg).sort(["anchor_ts"])
    phases = out.get_column("dr_phase").to_list()
    reasons = out.get_column("dr_reason_code").to_list()

    assert phases == [None, None, None, None, "B", "B"]
    assert reasons == [
        None,
        None,
        None,
        None,
        "ENTER_B_RANGE_VALID",
        "STAY_B_AUCTION_IN_RANGE",
    ]

    dr_ids = out.get_column("dr_id").to_list()
    assert dr_ids[:4] == [None, None, None, None]
    assert len({dr_id for dr_id in dr_ids[4:] if dr_id is not None}) == 1
    assert out.select(pl.col("dr_start_ts")).to_series()[4] == out.select(pl.col("anchor_ts")).to_series()[4]


def test_dealing_range_state_machine_golden_h1() -> None:
    candles = _make_candles("H1")
    windows = _make_windows(candles, "H1")
    cfg = {
        "lookback_bars": 3,
        "atr_window": 2,
        "width_min_atr": 0.5,
        "p_inside_min": 0.6,
        "tests_min": 1,
        "test_atr_mult": 0.2,
        "probe_atr_mult": 0.5,
        "reclaim_atr_mult": 0.2,
        "accept_atr_mult": 0.5,
        "accept_bars_min": 2,
        "reclaim_bars_max": 2,
        "trend_atr_mult": 1.5,
        "trend_width_mult": 0.2,
        "trend_bars_min": 2,
        "reentry_bars_max": 2,
    }

    out = fold_state_machine(windows, candles, cfg=cfg).sort(["anchor_ts"])
    phases = out.get_column("dr_phase").to_list()

    assert phases == [None, None, None, None, "B", "B"]


def test_dealing_range_state_machine_allows_nulls_for_missing_tfs() -> None:
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    anchor_ts = [start + timedelta(minutes=5 * idx) for idx in range(3)]
    windows = pl.DataFrame(
        {
            "instrument": ["GC", "GC", "GC"] * len(anchor_ts),
            "anchor_tf": ["M1", "M5", "D1"] * len(anchor_ts),
            "anchor_ts": anchor_ts * 3,
        }
    ).sort(["anchor_tf", "anchor_ts"])

    out = fold_state_machine(windows, None).sort(["anchor_tf", "anchor_ts"])
    assert set(out.get_column("anchor_tf").unique().to_list()) == {"M1", "M5", "D1"}
    assert out.get_column("dr_phase").null_count() == out.height
    assert out.get_column("dr_reason_code").null_count() == out.height
