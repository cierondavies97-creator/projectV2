from __future__ import annotations

import logging

import polars as pl

from engine.data.fills import write_fills_for_instrument_day
from engine.data.orders import write_orders_for_instrument_day
from engine.microbatch.types import BatchState

log = logging.getLogger(__name__)

_EVAL_KEYS = ["paradigm_id", "principle_id", "candidate_id", "experiment_id"]


def _as_ts(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Datetime(time_unit="us"))


def _is_eval_mode(df: pl.DataFrame | None) -> bool:
    return df is not None and (not df.is_empty()) and ("paradigm_id" in df.columns)


def _normalize_eval_identity(df: pl.DataFrame | None, *, name: str) -> pl.DataFrame:
    """
    Phase B invariant:
      If paradigm_id exists, principle_id must exist.

    Normalization:
      - ensure candidate_id / experiment_id exist
      - cast eval columns to Utf8
      - fill nulls with sentinel "∅" (partition-stable, join-safe)
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    out = df
    if "paradigm_id" not in out.columns:
        return out

    if "principle_id" not in out.columns:
        raise ValueError(f"fills_step: {name} has paradigm_id but missing principle_id (Phase B requires both).")

    if "candidate_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("candidate_id"))
    if "experiment_id" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("experiment_id"))

    for c in _EVAL_KEYS:
        out = out.with_columns(pl.col(c).cast(pl.Utf8))

    out = out.with_columns(
        [
            pl.col("candidate_id").fill_null("∅"),
            pl.col("experiment_id").fill_null("∅"),
        ]
    )
    return out


def _simulate_fills_for_group(
    *,
    state: BatchState,
    instrument: str,
    brackets_df: pl.DataFrame,
    candles_df: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Deterministic SIM fills (Phase B compatible):

    Inputs:
      - brackets_df: rows for a single (instrument, evaluation group)
      - candles_df : M1 (preferred) or anchor TF candles containing ts, open, high, low, close

    Output:
      - orders_df: intent log derived from brackets (one row per trade)
      - fills_df : executed fills (entry+exit per trade)
    """
    if brackets_df is None or brackets_df.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    # Ensure eval identity columns are normalized if present
    brackets_df = _normalize_eval_identity(brackets_df, name="brackets")

    # Evaluation columns to carry through (if present)
    eval_select_exprs: list[pl.Expr] = []
    if "paradigm_id" in brackets_df.columns:
        eval_select_exprs = [
            pl.col("paradigm_id"),
            pl.col("principle_id"),
            pl.col("candidate_id"),
            pl.col("experiment_id"),
        ]

    # Orders = one per trade_id (minimal)
    orders_df = brackets_df.select(
        [
            pl.lit(state.ctx.snapshot_id).alias("snapshot_id"),
            pl.lit(state.ctx.run_id).alias("run_id"),
            pl.lit(state.ctx.mode).alias("mode"),
            pl.lit(state.key.trading_day).cast(pl.Date).alias("dt"),
            *eval_select_exprs,
            pl.col("trade_id"),
            pl.col("instrument").alias("instrument"),
            pl.col("side"),
            _as_ts("entry_ts").alias("submitted_ts"),
            pl.col("entry_px").alias("limit_px"),
            pl.col("sl_px"),
            pl.col("tp_px"),
            pl.lit("SIM").alias("venue"),
            pl.lit("submitted").alias("status"),
        ]
    )

    # If no candles, produce DEV fills using bracket prices (still deterministic)
    if candles_df is None or candles_df.is_empty():
        fills_df = brackets_df.select(
            [
                pl.lit(state.ctx.snapshot_id).alias("snapshot_id"),
                pl.lit(state.ctx.run_id).alias("run_id"),
                pl.lit(state.ctx.mode).alias("mode"),
                pl.lit(state.key.trading_day).cast(pl.Date).alias("dt"),
                *eval_select_exprs,
                pl.col("trade_id"),
                pl.col("instrument").alias("instrument"),
                pl.col("side"),
                _as_ts("entry_ts").alias("fill_ts"),
                pl.col("entry_px").alias("fill_px"),
                pl.lit("entry").alias("fill_type"),
            ]
        )
        return orders_df, fills_df

    # Candle schema expectations: ts, high, low, close (+ open optional)
    c = candles_df.filter(pl.col("instrument") == instrument).sort("ts")

    if c.is_empty():
        # candles present but none for this instrument -> deterministic DEV fills
        fills_df = brackets_df.select(
            [
                pl.lit(state.ctx.snapshot_id).alias("snapshot_id"),
                pl.lit(state.ctx.run_id).alias("run_id"),
                pl.lit(state.ctx.mode).alias("mode"),
                pl.lit(state.key.trading_day).cast(pl.Date).alias("dt"),
                *eval_select_exprs,
                pl.col("trade_id"),
                pl.col("instrument").alias("instrument"),
                pl.col("side"),
                _as_ts("entry_ts").alias("fill_ts"),
                pl.col("entry_px").alias("fill_px"),
                pl.lit("entry").alias("fill_type"),
            ]
        )
        return orders_df, fills_df

    # Minimal SIM: entry at first candle at/after entry_ts, exit when TP/SL touched else close of day.
    fills: list[dict] = []
    for row in brackets_df.to_dicts():
        trade_id = row.get("trade_id")
        side = row.get("side")
        entry_ts = row.get("entry_ts")
        entry_px = row.get("entry_px")
        tp = row.get("tp_px")
        sl = row.get("sl_px")

        # Evaluation identity per row (Phase B join safety)
        paradigm_id = row.get("paradigm_id")
        principle_id = row.get("principle_id")
        candidate_id = row.get("candidate_id") or "∅"
        experiment_id = row.get("experiment_id") or "∅"

        # filter candles from entry_ts onward
        sub = c.filter(pl.col("ts") >= pl.lit(entry_ts))
        if sub.is_empty():
            continue

        first = sub.head(1).to_dicts()[0]
        entry_fill_ts = first["ts"]
        entry_fill_px = entry_px if entry_px is not None else first.get("open", first.get("close"))

        last = sub.tail(1).to_dicts()[0]
        exit_fill_ts = last["ts"]
        exit_fill_px = last.get("close")

        is_long = (side == "long")
        for bar in sub.select(["ts", "high", "low", "close"]).to_dicts():
            hi = bar["high"]
            lo = bar["low"]
            ts = bar["ts"]

            if is_long:
                if tp is not None and hi is not None and hi >= tp:
                    exit_fill_ts, exit_fill_px = ts, tp
                    break
                if sl is not None and lo is not None and lo <= sl:
                    exit_fill_ts, exit_fill_px = ts, sl
                    break
            else:
                # short
                if tp is not None and lo is not None and lo <= tp:
                    exit_fill_ts, exit_fill_px = ts, tp
                    break
                if sl is not None and hi is not None and hi >= sl:
                    exit_fill_ts, exit_fill_px = ts, sl
                    break

        base = {
            "snapshot_id": state.ctx.snapshot_id,
            "run_id": state.ctx.run_id,
            "mode": state.ctx.mode,
            "dt": state.key.trading_day,
            "trade_id": trade_id,
            "instrument": instrument,
            "side": side,
        }

        # Add eval identity only when present (backward compatible)
        if paradigm_id is not None:
            base.update(
                {
                    "paradigm_id": str(paradigm_id),
                    "principle_id": str(principle_id),
                    "candidate_id": str(candidate_id),
                    "experiment_id": str(experiment_id),
                }
            )

        fills.append(
            {
                **base,
                "fill_ts": entry_fill_ts,
                "fill_px": float(entry_fill_px) if entry_fill_px is not None else None,
                "fill_type": "entry",
            }
        )
        fills.append(
            {
                **base,
                "fill_ts": exit_fill_ts,
                "fill_px": float(exit_fill_px) if exit_fill_px is not None else None,
                "fill_type": "exit",
            }
        )

    fills_df = pl.DataFrame(fills) if fills else pl.DataFrame()
    return orders_df, fills_df


def run(state: BatchState) -> BatchState:
    """
    Fills step (Phase B compatible SIM).

    Inputs:
      - 'brackets' (required)
      - 'candles' (optional; if absent produces minimal DEV fills)

    Outputs:
      - 'orders'
      - 'fills'
    """
    brackets = state.get_optional("brackets")
    if brackets is None or brackets.is_empty():
        state.set("orders", pl.DataFrame())
        state.set("fills", pl.DataFrame())
        return state

    # Normalize eval identity if present
    brackets = _normalize_eval_identity(brackets, name="brackets") if _is_eval_mode(brackets) else brackets

    candles = state.get_optional("candles")  # may not exist yet; ok

    orders_all: list[pl.DataFrame] = []
    fills_all: list[pl.DataFrame] = []

    eval_mode = _is_eval_mode(brackets)
    group_cols = ["instrument"]
    if eval_mode:
        group_cols += _EVAL_KEYS

    # Simulate per instrument (+ per evaluation group in Phase B)
    for g in brackets.partition_by(group_cols, maintain_order=True):
        instrument = str(g["instrument"][0])

        orders_df, fills_df = _simulate_fills_for_group(
            state=state,
            instrument=instrument,
            brackets_df=g,
            candles_df=candles,
        )

        if not orders_df.is_empty():
            orders_all.append(orders_df)
            write_orders_for_instrument_day(
                ctx=state.ctx,
                trading_day=state.key.trading_day,
                instrument=instrument,
                orders_df=orders_df,
                sandbox=False,
            )

        if not fills_df.is_empty():
            fills_all.append(fills_df)
            write_fills_for_instrument_day(
                ctx=state.ctx,
                trading_day=state.key.trading_day,
                instrument=instrument,
                fills_df=fills_df,
                sandbox=False,
            )

    orders_out = pl.concat(orders_all, how="diagonal") if orders_all else pl.DataFrame()
    fills_out = pl.concat(fills_all, how="diagonal") if fills_all else pl.DataFrame()

    state.set("orders", orders_out)
    state.set("fills", fills_out)

    log.info(
        "fills_step: wrote orders=%d fills=%d eval_mode=%s groups=%d",
        orders_out.height if orders_out is not None else 0,
        fills_out.height if fills_out is not None else 0,
        str(eval_mode),
        len(brackets.partition_by(group_cols, maintain_order=True)),
    )

    return state
