# scripts/dev_seed_candles.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl


def _intraday_drift(hour: int, day_index: int) -> float:
    """
    Deterministic drift pattern per hour, varying by day_index
    so that different days produce different stat_ts / ICT signals.

    Day 0: gentle uptrend, low vol
    Day 1: stronger uptrend, higher vol
    Day 2: downtrend
    Day 3: mean-reversion with midday spike
    Day 4: choppy, high-vol range
    """
    # Base pattern by hour (London/NY-ish sessions)
    if 0 <= hour < 6:
        base = 0.0
    elif 6 <= hour < 12:
        base = 0.02
    elif 12 <= hour < 18:
        base = -0.01
    else:
        base = 0.03

    # Day-specific adjustments
    if day_index == 0:
        # Quiet, mostly sideways/up
        return base * 0.5
    elif day_index == 1:
        # Stronger trend up
        return base * 1.5 + 0.01
    elif day_index == 2:
        # Net down day
        return -abs(base) - 0.01
    elif day_index == 3:
        # Mean-reversion with a midday spike later handled in noise
        return base * 0.3
    else:
        # Choppy high-vol range
        return base * 0.1


def _noise_magnitude(hour: int, day_index: int) -> float:
    """
    Deterministic noise magnitude per hour/day.
    """
    if 0 <= hour < 6:
        base = 0.3
    elif 6 <= hour < 12:
        base = 0.6
    elif 12 <= hour < 18:
        base = 0.8
    else:
        base = 0.5

    if day_index == 1:
        base *= 1.2
    elif day_index == 2:
        base *= 1.1
    elif day_index == 3:
        base *= 1.4
    elif day_index == 4:
        base *= 1.8

    return base


def _generate_m1_path(
    instrument: str,
    trading_day: date,
    *,
    day_index: int,
    start_price: float = 2000.0,
) -> pl.DataFrame:
    """
    Generate a fully deterministic 1-minute candle path for one day.

    - No RNG: everything is a pure function of (instrument, trading_day, minute_index).
    - Different day_index values produce different intraday regimes so
      stat_ts and ICT features have more interesting structure.
    """
    start_ts = datetime(trading_day.year, trading_day.month, trading_day.day, 0, 0)
    rows = []

    price = start_price

    # 24 * 60 minutes in a day
    for minute_index in range(24 * 60):
        ts = start_ts + timedelta(minutes=minute_index)
        hour = ts.hour

        drift = _intraday_drift(hour, day_index)
        noise_mag = _noise_magnitude(hour, day_index)

        # Deterministic "noise": function of minute_index and day_index only.
        # This gives you repeatable volatility / wiggles.
        base_noise = ((minute_index + 13 * day_index) % 19) - 9
        noise = base_noise * (noise_mag / 20.0)

        # Add a couple of deterministic "macro spike" events to create
        # obvious swings for ICT + stat_ts to latch onto.
        if day_index in (1, 3, 4):
            # e.g. spike around 13:30 and 19:00
            if ts.hour == 13 and 25 <= ts.minute <= 35:
                noise += 2.0 * (1 if day_index != 2 else -1)
            if ts.hour == 19 and 25 <= ts.minute <= 35:
                noise -= 2.0

        bar_open = price
        bar_close = price + drift + noise

        bar_high = max(bar_open, bar_close) + noise_mag * 0.2
        bar_low = min(bar_open, bar_close) - noise_mag * 0.2

        # Deterministic volume pattern, higher during "sessions".
        vol_base = 1_000
        vol_bump = 0
        if 7 <= hour < 10:
            vol_bump = 1_000
        elif 13 <= hour < 16:
            vol_bump = 1_200
        elif 19 <= hour < 21:
            vol_bump = 800

        volume = vol_base + vol_bump + ((minute_index * 17 + 31 * (day_index + 1)) % 500)

        rows.append(
            {
                "instrument": instrument,
                "tf": "M1",
                "ts": ts,
                "open": float(bar_open),
                "high": float(bar_high),
                "low": float(bar_low),
                "close": float(bar_close),
                "volume": int(volume),
            }
        )

        price = bar_close

    return pl.DataFrame(rows)


def _resample_to_m5(df_m1: pl.DataFrame) -> pl.DataFrame:
    """
    Downsample M1 candles into M5 candles.

    This is purely deterministic: uses row order and timestamp to assign
    5-minute buckets and aggregates O/H/L/C/V.

    Important: we reorder columns at the end so they exactly match the
    M1 schema, so pl.concat(..., how="vertical") works.
    """
    if df_m1.is_empty():
        return df_m1

    # Use with_row_index (new API) instead of deprecated with_row_count
    df = df_m1.sort("ts").with_row_index("row_nr")

    df = df.with_columns((pl.col("row_nr") // 5).alias("bucket"))

    agg = (
        df.group_by(["instrument", "bucket"], maintain_order=True)
        .agg(
            [
                pl.col("ts").min().alias("ts"),
                pl.first("open").alias("open"),
                pl.max("high").alias("high"),
                pl.min("low").alias("low"),
                pl.last("close").alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .drop("bucket")
        .with_columns(pl.lit("M5").alias("tf"))
    )

    # Reorder columns to exactly match the M1 schema so concat works.
    # df_m1 is the template.
    return agg.select(df_m1.columns)


def _seed_one_day(
    instrument: str,
    trading_day: date,
    *,
    day_index: int,
    start_price: float,
    out_root: Path,
) -> float:
    """
    Generate M1 + M5 candles for a single day and write them to:

        data/candles/mode=backtest/instrument=<instrument>/dt=<YYYY-MM-DD>/0000.parquet

    Returns:
        The last M1 close price, so the next day can start from it.
    """
    df_m1 = _generate_m1_path(
        instrument=instrument,
        trading_day=trading_day,
        day_index=day_index,
        start_price=start_price,
    )

    df_m5 = _resample_to_m5(df_m1)
    df_all = pl.concat([df_m1, df_m5], how="vertical").sort(["ts", "tf"])

    out_dir = out_root / "mode=backtest" / f"instrument={instrument}" / f"dt={trading_day:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "0000.parquet"

    df_all.write_parquet(out_path)

    print(f"[dev_seed_candles] Wrote {df_all.height} rows for {instrument} {trading_day} to {out_path}")
    print(df_all.head(5))

    # Last close from the M1 leg
    return float(df_m1.sort("ts")["close"][-1])


def main() -> None:
    """
    Seed a small deterministic 5-day candle dataset for dev/backtest:

    - Instrument: XAUUSD (matches your metals cluster)
    - Days: start at 2025-01-02, then next 4 calendar days
    - Mode: backtest
    - TFs: M1 (generated directly) + M5 (downsampled)
    """
    instrument = "XAUUSD"
    start_day = date(2025, 1, 2)
    num_days = 5
    start_price = 2000.0

    out_root = Path("data") / "candles"

    price = start_price
    for day_index in range(num_days):
        trading_day = start_day + timedelta(days=day_index)
        price = _seed_one_day(
            instrument=instrument,
            trading_day=trading_day,
            day_index=day_index,
            start_price=price,
            out_root=out_root,
        )


if __name__ == "__main__":
    main()
