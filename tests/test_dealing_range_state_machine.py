from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from engine.features.market_structure.dealing_range import fold_state_machine


def _windows_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": ["XAUUSD", "XAUUSD"],
            "anchor_tf": ["H1", "H1"],
            "anchor_ts": [
                datetime(2025, 1, 2, 12, tzinfo=timezone.utc),
                datetime(2025, 1, 2, 13, tzinfo=timezone.utc),
            ],
        }
    )


def test_fold_state_machine_returns_nulls_without_candles() -> None:
    windows = _windows_frame()

    out = fold_state_machine(windows, candles=None)

    assert out.height == windows.height
    assert out.select("dr_phase").to_series().null_count() == windows.height
    assert out.select("dr_reason_code").to_series().null_count() == windows.height


def test_fold_state_machine_returns_nulls_with_missing_columns() -> None:
    windows = _windows_frame()
    candles = pl.DataFrame(
        {
            "instrument": ["XAUUSD"],
            "ts": [datetime(2025, 1, 2, 12, tzinfo=timezone.utc)],
            "high": [2000.0],
        }
    )

    out = fold_state_machine(windows, candles=candles)

    assert out.height == windows.height
    assert out.select("dr_phase").to_series().null_count() == windows.height
    assert out.select("dr_reason_code").to_series().null_count() == windows.height
