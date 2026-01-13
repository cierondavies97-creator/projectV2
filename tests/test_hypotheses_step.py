from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from engine.microbatch.steps.hypotheses_step import _ensure_trade_paths_schema


def test_ensure_trade_paths_schema_adds_experiment_id_when_missing() -> None:
    df = pl.DataFrame({"candidate_id": [None, "cand-1"]})

    out = _ensure_trade_paths_schema(df)

    assert "experiment_id" in out.columns
    assert out.get_column("experiment_id").to_list() == ["∅", "∅"]
    assert out.get_column("candidate_id").to_list() == ["∅", "cand-1"]


def test_ensure_trade_paths_schema_fills_nulls_with_existing_columns() -> None:
    df = pl.DataFrame({"candidate_id": [None, "cand-2"], "experiment_id": [None, "exp-2"]})

    out = _ensure_trade_paths_schema(df)

    assert out.get_column("candidate_id").to_list() == ["∅", "cand-2"]
    assert out.get_column("experiment_id").to_list() == ["∅", "exp-2"]
