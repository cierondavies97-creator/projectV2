from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import polars as pl

REQUIRED = ["instrument", "tf", "ts", "open", "high", "low", "close", "volume"]


def _infer_tf_from_path(p: Path) -> str | None:
    s = str(p).replace("\\", "/")
    m = re.search(r"(?:^|/)(?:timeframe|tf)=([^/]+)(?:/|$)", s, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _parse_dt_day(dt_dir_name: str) -> datetime:
    # dt=YYYY-MM-DD
    day = dt_dir_name.replace("dt=", "")
    # Use naive datetime to match your casting to Datetime("us") (naive)
    return datetime.strptime(day, "%Y-%m-%d")


def _validate_ts_in_dt_day(
    df: pl.DataFrame,
    *,
    day_start: datetime,
    on_violation: str,
    label: str,
) -> None:
    """
    Ensures all df['ts'] are within [day_start, day_start+1day).
    - on_violation: "warn" | "error" | "ignore"
    """
    if df.is_empty() or "ts" not in df.columns:
        return

    # Polars datetime is naive; compare using Python datetimes (also naive).
    day_end = day_start.replace()  # copy
    day_end = day_end + (datetime.strptime("1970-01-02", "%Y-%m-%d") - datetime.strptime("1970-01-01", "%Y-%m-%d"))
    # The above is a safe 1-day timedelta without importing timedelta (keeps script minimal).

    ts_min = df.select(pl.col("ts").min()).item()
    ts_max = df.select(pl.col("ts").max()).item()

    if ts_min is None or ts_max is None:
        return

    ok = (ts_min >= day_start) and (ts_max < day_end)
    if ok or on_violation == "ignore":
        return

    msg = (
        f"[dt-guard] {label}: ts range outside dt day.\n"
        f"  expected: [{day_start} .. {day_end})\n"
        f"  actual:   [{ts_min} .. {ts_max}]\n"
        f"  hint: check your source timezone/normalization or dt folder assignment."
    )

    if on_violation == "error":
        raise ValueError(msg)
    else:
        print(msg)


def normalize(df: pl.DataFrame, *, instrument_override: str | None, tf_override: str | None) -> pl.DataFrame:
    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename({"timestamp": "ts"})

    missing = [c for c in ["ts", "open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}. Have: {df.columns}")

    if "instrument" not in df.columns:
        if not instrument_override:
            raise ValueError("No 'instrument' column and no --instrument-override provided.")
        df = df.with_columns(pl.lit(instrument_override).alias("instrument"))

    if "tf" not in df.columns:
        if not tf_override:
            # Inference is handled in main(); normalize should not mention inference.
            raise ValueError("No 'tf' column and no --tf-override provided.")
        df = df.with_columns(pl.lit(tf_override).alias("tf"))

    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("volume"))

    if instrument_override:
        df = df.with_columns(pl.lit(instrument_override).cast(pl.Utf8).alias("instrument"))
    if tf_override:
        df = df.with_columns(pl.lit(tf_override).cast(pl.Utf8).alias("tf"))

    df = df.with_columns(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("tf").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64, strict=False),
    ).sort(["instrument", "tf", "ts"])

    return df.select(REQUIRED)


def merge_existing(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    merged = pl.concat([existing, incoming], how="vertical_relaxed")
    merged = merged.unique(subset=["instrument", "tf", "ts"], keep="last")
    return merged.sort(["instrument", "tf", "ts"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dt-root", required=True, help="Folder containing dt=YYYY-MM-DD subfolders")
    ap.add_argument("--src-glob", default="*.parquet", help="Which parquet(s) to read inside each dt folder")
    ap.add_argument("--mode", default="backtest")
    ap.add_argument("--instrument", required=True, help="Engine instrument used in target path (e.g. XAUUSD)")
    ap.add_argument("--instrument-override", default=None, help="Override instrument column inside source files")
    ap.add_argument(
        "--tf-override",
        default=None,
        help="Set tf if missing in source (optional; can be inferred)",
    )
    ap.add_argument("--out-name", default="0000.parquet", help="Target filename per day (default 0000.parquet)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--merge-into-existing",
        action="store_true",
        help="If out file exists: merge (dedupe on instrument/tf/ts) instead of error/overwrite.",
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--limit-days", type=int, default=0, help="For smoke tests; 0 = all days")

    # Upgrade: dt partition timestamp guard
    ap.add_argument(
        "--dt-guard",
        choices=["ignore", "warn", "error"],
        default="warn",
        help="Validate that candle ts fall inside each dt=YYYY-MM-DD folder's day.",
    )

    args = ap.parse_args()

    src_root = Path(args.src_dt_root)
    root = Path(args.project_root)
    out_root = root / "data" / "candles" / f"mode={args.mode}" / f"instrument={args.instrument}"

    if not args.tf_override:
        inferred_tf = _infer_tf_from_path(src_root)
        if inferred_tf:
            args.tf_override = inferred_tf

    dt_dirs = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("dt=")])
    if args.limit_days and args.limit_days > 0:
        dt_dirs = dt_dirs[: args.limit_days]

    if not dt_dirs:
        raise FileNotFoundError(f"No dt=... folders found under: {src_root}")

    for dt_dir in dt_dirs:
        day = dt_dir.name.replace("dt=", "")
        day_start = _parse_dt_day(dt_dir.name)

        src_files = sorted(dt_dir.glob(args.src_glob))
        if not src_files:
            print(f"SKIP {day}: no files matched {args.src_glob}")
            continue

        df = pl.concat([pl.read_parquet(f) for f in src_files], how="vertical_relaxed")
        df = normalize(df, instrument_override=args.instrument_override, tf_override=args.tf_override)

        # Upgrade: validate incoming df belongs to dt partition
        _validate_ts_in_dt_day(
            df,
            day_start=day_start,
            on_violation=args.dt_guard,
            label=f"incoming day={day} root={src_root}",
        )

        out_dir = out_root / f"dt={day}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.out_name

        if out_path.exists():
            if args.merge_into_existing:
                existing = pl.read_parquet(out_path)
                existing = normalize(existing, instrument_override=None, tf_override=None)

                # Validate existing too (helps detect prior bad imports)
                _validate_ts_in_dt_day(
                    existing,
                    day_start=day_start,
                    on_violation=args.dt_guard,
                    label=f"existing day={day} out={out_path}",
                )

                df = merge_existing(existing, df)

                # Validate merged result
                _validate_ts_in_dt_day(
                    df,
                    day_start=day_start,
                    on_violation=args.dt_guard,
                    label=f"merged day={day} out={out_path}",
                )
            elif not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite: {out_path} (use --overwrite or --merge-into-existing)")

        df.write_parquet(out_path)
        tfs = df.get_column("tf").unique().sort().to_list()
        print(f"WROTE {out_path} rows={df.height} tfs={tfs}")


if __name__ == "__main__":
    main()
