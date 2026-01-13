from __future__ import annotations

from datetime import date

import polars as pl

from engine.core.ids import RunContext
from engine.io.parquet_io import read_parquet_dir
from engine.io.paths import candles_dir


def read_candles_for_instrument_day(
    ctx: RunContext,
    instrument: str,
    trading_day: date,
) -> pl.DataFrame:
    """
    Read candles for a single instrument and trading day.

    Layout (matching engine.io.paths.candles_dir):

        data/candles/
          mode=<MODE>/
            instrument=<INSTRUMENT>/
              dt=<YYYY-MM-DD>/*.parquet

    - MODE comes from ctx.mode ("backtest", "paper", "live", etc.)
    - Raises FileNotFoundError if the directory does not exist,
      because read_parquet_dir does the same.
    - If the 'instrument' column is missing in the underlying files,
      it is added so downstream joins always have the key.
    """
    dir_path = candles_dir(
        ctx,
        trading_day,
        instrument_id=instrument,
    )

    df = read_parquet_dir(dir_path)

    if df.is_empty():
        # Nothing to normalise, just return as-is
        return df

    if "instrument" not in df.columns:
        df = df.with_columns(pl.lit(instrument).alias("instrument"))

    return df
