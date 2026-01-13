from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Sequence, Any

import polars as pl


def require_cols(df: pl.DataFrame, cols: Iterable[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns: {missing}")


def ensure_sorted(df: pl.DataFrame, by: Sequence[str], *, where: str = "ensure_sorted") -> pl.DataFrame:
    """
    Deterministic sort helper. Always sorts (cheap enough for dev; safe for correctness).
    """
    if df.is_empty():
        return df
    require_cols(df, by, where=where)
    return df.sort(list(by))


def to_anchor_tf(candles: pl.DataFrame, anchor_tf: str, *, where: str) -> pl.DataFrame:
    """
    Filter a mixed-tf candles frame down to a single anchor tf.
    If 'tf' is absent, returns candles unchanged (dev-tolerant).
    """
    if candles is None or candles.is_empty():
        return pl.DataFrame()
    if "tf" not in candles.columns:
        return candles
    return candles.filter(pl.col("tf") == pl.lit(anchor_tf))


def safe_div(num: pl.Expr, den: pl.Expr, *, default: float = 0.0) -> pl.Expr:
    return (
        pl.when(den.is_null() | (den == 0))
        .then(pl.lit(default))
        .otherwise(num / den)
    )


def zscore(x: pl.Expr, *, mean_expr: pl.Expr, std_expr: pl.Expr, default: float = 0.0) -> pl.Expr:
    return safe_div(x - mean_expr, std_expr, default=default)


def bucket_by_edges(x: pl.Expr, *, edges: Sequence[float], labels: Sequence[str], default: str = "unknown") -> pl.Expr:
    """
    Assign a categorical bucket using increasing 'edges'.
    labels must be len(edges)+1; all labels are literals.
    """
    if len(labels) != len(edges) + 1:
        raise ValueError("bucket_by_edges: labels must be len(edges)+1")

    out = pl.lit(default)

    out = pl.when(x <= edges[0]).then(pl.lit(labels[0])).otherwise(out)
    for i in range(1, len(edges)):
        out = pl.when((x > edges[i - 1]) & (x <= edges[i])).then(pl.lit(labels[i])).otherwise(out)
    out = pl.when(x > edges[-1]).then(pl.lit(labels[-1])).otherwise(out)
    return out


def rolling_atr(
    df: pl.DataFrame,
    *,
    group_cols: Sequence[str],
    period: int,
    out_col: str = "atr",
    tr_col: str = "_tr",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    ts_col: str = "ts",
) -> pl.DataFrame:
    """
    Deterministic ATR (Wilder-style TR rolling mean).
    Assumes df is bar-level candles; sorts by group_cols + ts_col.
    """
    if df.is_empty():
        return df

    require_cols(df, [*group_cols, ts_col, high_col, low_col, close_col], where="rolling_atr")
    df = ensure_sorted(df, [*group_cols, ts_col], where="rolling_atr")

    prev_close = pl.col(close_col).shift(1).over(list(group_cols))
    tr = pl.max_horizontal(
        (pl.col(high_col) - pl.col(low_col)),
        (pl.col(high_col) - prev_close).abs(),
        (pl.col(low_col) - prev_close).abs(),
    ).alias(tr_col)

    atr = (
        pl.col(tr_col)
        .rolling_mean(window_size=period, min_periods=period)
        .over(list(group_cols))
        .alias(out_col)
    )

    return df.with_columns(tr).with_columns(atr)


def conform_to_registry(
    df: pl.DataFrame,
    *,
    registry_entry: Mapping[str, Any] | None,
    key_cols: Sequence[str],
    where: str,
    allow_extra: bool = True,
) -> pl.DataFrame:
    """
    Align a feature frame to a registry entry (if provided).

    Behavior:
      - If registry_entry is None or lacks 'columns', return df unchanged.
      - Ensure required key columns are present; raise if missing.
      - Add any registry columns missing from df as nulls.
      - If allow_extra=False, drop columns not listed in the registry.
    """
    if df.is_empty():
        return df
    if registry_entry is None:
        return df

    columns = registry_entry.get("columns") if isinstance(registry_entry, Mapping) else None
    if not isinstance(columns, Mapping) or not columns:
        return df

    missing_keys = [c for c in key_cols if c not in df.columns]
    if missing_keys:
        raise ValueError(f"{where}: missing required key columns: {missing_keys}")

    expected_cols = list(columns.keys())
    missing_expected = [c for c in expected_cols if c not in df.columns]
    if missing_expected:
        df = df.with_columns([pl.lit(None).alias(c) for c in missing_expected])

    if not allow_extra:
        extras = [c for c in df.columns if c not in expected_cols]
        if extras:
            df = df.drop(extras)

    return df
