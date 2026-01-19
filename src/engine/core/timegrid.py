from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class GridCheck:
    tf: str
    rows: int
    bad_rows: int
    sample_bad: pl.DataFrame


def tf_to_truncate_rule(tf: str) -> str:
    """
    Convert engine TF strings into polars truncate rules.

    Examples:
      S1 -> "1s"
      M1 -> "1m"
      M5 -> "5m"
      H1 -> "1h"
      D1 -> "1d"
    """
    tf = (tf or "").strip().upper()
    if tf.startswith("S"):
        return f"{int(tf[1:])}s"
    if tf.startswith("M"):
        return f"{int(tf[1:])}m"
    if tf.startswith("H"):
        return f"{int(tf[1:])}h"
    if tf in ("D1", "1D", "D"):
        return "1d"
    raise ValueError(f"timegrid: unsupported tf={tf!r} (expected S#, M#, H#, D1)")


def build_anchor_grid(
    candles: pl.DataFrame,
    *,
    anchor_tf: str,
    ts_col: str = "ts",
    instrument_col: str = "instrument",
) -> pl.DataFrame:
    """
    Build a canonical anchor grid keyed by (instrument, anchor_tf, ts).

    Contract:
      - ts is truncated to the anchor_tf grid (e.g. M5 -> 00:00, 00:05, ...)
      - one row per (instrument, anchor_tf, ts)
      - sorted
    """
    if candles is None or candles.is_empty():
        return pl.DataFrame(schema={"instrument": pl.Utf8, "anchor_tf": pl.Utf8, "ts": pl.Datetime("us")})

    if instrument_col not in candles.columns or ts_col not in candles.columns:
        return pl.DataFrame(schema={"instrument": pl.Utf8, "anchor_tf": pl.Utf8, "ts": pl.Datetime("us")})

    rule = tf_to_truncate_rule(anchor_tf)

    return (
        candles.select([instrument_col, ts_col])
        .with_columns(pl.col(ts_col).cast(pl.Datetime("us"), strict=False))
        .with_columns(pl.col(ts_col).dt.truncate(rule).alias(ts_col))
        .unique()
        .with_columns(pl.lit(anchor_tf).cast(pl.Utf8).alias("anchor_tf"))
        .select([instrument_col, "anchor_tf", ts_col])
        .rename({instrument_col: "instrument"})
        .sort(["instrument", "anchor_tf", ts_col])
    )


def validate_anchor_grid(
    df: pl.DataFrame,
    *,
    anchor_tf: str,
    ts_col: str,
    max_sample: int = 50,
) -> GridCheck:
    """
    Validate that df[ts_col] is on the anchor_tf grid (ts == truncate(ts, rule)).
    """
    if df is None or df.is_empty() or ts_col not in df.columns:
        return GridCheck(tf=anchor_tf, rows=0, bad_rows=0, sample_bad=pl.DataFrame())

    rule = tf_to_truncate_rule(anchor_tf)

    d2 = df.with_columns(pl.col(ts_col).cast(pl.Datetime("us"), strict=False).alias(ts_col))
    bad = d2.filter(pl.col(ts_col) != pl.col(ts_col).dt.truncate(rule))

    sample_cols = [c for c in ["instrument", "anchor_tf", ts_col] if c in bad.columns]
    sample = bad.select(sample_cols).head(max_sample) if sample_cols else bad.head(max_sample)
    return GridCheck(tf=anchor_tf, rows=d2.height, bad_rows=bad.height, sample_bad=sample)


def validate_anchor_grid_multi(
    df: pl.DataFrame,
    *,
    tf_col: str = "anchor_tf",
    ts_col: str = "ts",
    max_sample: int = 50,
) -> tuple[int, pl.DataFrame]:
    """
    Validate grid alignment across all anchor_tf values in df.

    Returns: (bad_rows_total, sample_bad_rows)
    """
    if df is None or df.is_empty() or tf_col not in df.columns or ts_col not in df.columns:
        return 0, pl.DataFrame()

    tfs: Iterable[str] = df.select(pl.col(tf_col).unique()).to_series().to_list()

    bad_total = 0
    samples: list[pl.DataFrame] = []

    for tf in tfs:
        chk = validate_anchor_grid(
            df.filter(pl.col(tf_col) == tf),
            anchor_tf=str(tf),
            ts_col=ts_col,
            max_sample=max_sample,
        )
        bad_total += chk.bad_rows
        if chk.bad_rows > 0 and len(samples) < 3:
            samples.append(chk.sample_bad)

    sample_bad = pl.concat(samples, how="vertical") if samples else pl.DataFrame()
    return bad_total, sample_bad
