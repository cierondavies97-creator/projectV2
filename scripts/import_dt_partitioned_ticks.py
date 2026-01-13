from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

REQUIRED = ["instrument", "ts", "bid", "ask", "bid_size", "ask_size"]


def _infer_from_path(p: Path, key: str) -> str | None:
    # Matches .../<key>=VALUE/... (works with \ or /)
    s = str(p).replace("\\", "/")
    m = re.search(rf"(?:^|/){re.escape(key)}=([^/]+)(?:/|$)", s, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _parse_dt_day(dt_dir_name: str) -> datetime:
    # dt=YYYY-MM-DD
    day = dt_dir_name.replace("dt=", "")
    return datetime.strptime(day, "%Y-%m-%d")


def _validate_ts_in_dt_day(df: pl.DataFrame, *, day_start: datetime, on_violation: str, label: str) -> None:
    if on_violation == "ignore" or df.is_empty() or "ts" not in df.columns:
        return

    day_end = day_start + timedelta(days=1)

    ts_min = df.select(pl.col("ts").min()).item()
    ts_max = df.select(pl.col("ts").max()).item()
    if ts_min is None or ts_max is None:
        return

    ok = (ts_min >= day_start) and (ts_max < day_end)
    if ok:
        return

    msg = (
        f"[dt-guard] {label}: ts range outside dt day.\n"
        f"  expected: [{day_start} .. {day_end})\n"
        f"  actual:   [{ts_min} .. {ts_max}]\n"
        f"  hint: check timezone normalization or use --dt-from-ts/--dt-scope."
    )

    if on_violation == "error":
        raise ValueError(msg)
    else:
        print(msg)


def _rename_if_present(df: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    cols = set(df.columns)
    ren = {src: dst for src, dst in mapping.items() if src in cols and dst not in cols}
    return df.rename(ren) if ren else df


def normalize(
    df: pl.DataFrame,
    *,
    instrument: str,
    instrument_override: str | None,
    tz_policy: str,
) -> pl.DataFrame:
    # Common vendor names -> canonical names
    df = _rename_if_present(
        df,
        {
            "timestamp": "ts",
            "time": "ts",
            "bidPrice": "bid",
            "askPrice": "ask",
            "bid_volume": "bid_size",
            "ask_volume": "ask_size",
            "bidVolume": "bid_size",
            "askVolume": "ask_size",
        },
    )

    missing = [c for c in ["ts", "bid", "ask"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required tick columns: {missing}. Have: {df.columns}")

    # instrument column
    if "instrument" not in df.columns:
        df = df.with_columns(pl.lit(instrument_override or instrument).alias("instrument"))

    # size columns are optional
    if "bid_size" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("bid_size"))
    if "ask_size" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("ask_size"))

    if instrument_override:
        df = df.with_columns(pl.lit(instrument_override).cast(pl.Utf8).alias("instrument"))

    # timestamp normalization
    # - utc_naive: if tz-aware => convert to UTC and drop tz; if naive => keep as-is (assumed UTC)
    # - keep: preserve whatever polars read
    ts_expr = pl.col("ts")
    if tz_policy == "utc_naive":
        ts_expr = ts_expr.dt.convert_time_zone("UTC").dt.replace_time_zone(None)

    df = df.with_columns(
        pl.col("instrument").cast(pl.Utf8),
        ts_expr.cast(pl.Datetime("us")),
        pl.col("bid").cast(pl.Float64),
        pl.col("ask").cast(pl.Float64),
        pl.col("bid_size").cast(pl.Float64, strict=False),
        pl.col("ask_size").cast(pl.Float64, strict=False),
    ).sort(["instrument", "ts"])

    return df.select(REQUIRED)


def merge_existing(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    merged = pl.concat([existing, incoming], how="vertical_relaxed")
    merged = merged.unique(subset=["instrument", "ts"], keep="last")
    return merged.sort(["instrument", "ts"])


def _dt_str_from_ts(df: pl.DataFrame, *, dt_scope: str, market_tz: str | None) -> pl.Series:
    """
    Derive dt (YYYY-MM-DD) from ts according to dt_scope.
    Assumes ts is already normalized per tz_policy in normalize().

    - dt_scope=utc: dt from ts as-is (naive UTC).
    - dt_scope=market: interpret ts as UTC, convert to market_tz, then take date.
    """
    if dt_scope == "utc":
        return df.get_column("ts").dt.date().cast(pl.Utf8)

    # market scope
    if not market_tz:
        raise ValueError("--dt-scope market requires --market-tz, e.g. Europe/London")

    # Treat naive ts as UTC, convert to market_tz, take date.
    # (We keep the stored 'ts' column UTC-naive; only dt assignment uses market tz.)
    return df.get_column("ts").dt.replace_time_zone("UTC").dt.convert_time_zone(market_tz).dt.date().cast(pl.Utf8)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--src-dt-root", required=True, help="Folder containing dt=YYYY-MM-DD subfolders")
    ap.add_argument("--src-glob", default="*.parquet", help="Which parquet(s) to read inside each dt folder")

    ap.add_argument("--mode", default="backtest")
    ap.add_argument("--instrument", required=True, help="Engine instrument id used in target path (e.g. XAUUSD)")

    ap.add_argument("--instrument-override", default=None, help="Force instrument column value for all rows")
    ap.add_argument(
        "--instrument-from-path",
        action="store_true",
        help="Infer source instrument from .../instrument=... in src path (used with --instrument-map).",
    )
    ap.add_argument(
        "--instrument-map",
        default="",
        help='Optional mapping like "XAU_USD=XAUUSD,EUR_USD=EURUSD" (applied to inferred source instrument).',
    )

    ap.add_argument("--out-name", default="0000.parquet")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--merge-into-existing",
        action="store_true",
        help="If out file exists: merge (dedupe on instrument/ts) instead of error/overwrite.",
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--limit-days", type=int, default=0, help="For smoke tests; 0 = all days")

    ap.add_argument(
        "--dt-guard",
        choices=["ignore", "warn", "error"],
        default="warn",
        help=(
            "Validate that tick ts fall inside each dt=YYYY-MM-DD folder's day "
            "(only meaningful when not using --dt-from-ts)."
        ),
    )
    ap.add_argument(
        "--tz-policy",
        choices=["utc_naive", "keep"],
        default="utc_naive",
        help="utc_naive: convert tz-aware ts to UTC and drop tz; keep: preserve whatever polars reads.",
    )

    # NEW: repartition based on ts
    ap.add_argument(
        "--dt-from-ts",
        action="store_true",
        help="Derive engine dt partitions from ts (post-normalization) instead of trusting source dt folder naming.",
    )
    ap.add_argument(
        "--dt-scope",
        choices=["utc", "market"],
        default="utc",
        help="How to assign dt when --dt-from-ts is set. utc=calendar day in UTC. market=calendar day in --market-tz.",
    )
    ap.add_argument(
        "--market-tz",
        default=None,
        help="IANA timezone (e.g. Europe/London, America/New_York). Used when --dt-scope market.",
    )

    args = ap.parse_args()

    src_root = Path(args.src_dt_root)
    root = Path(args.project_root)
    out_root = root / "data" / "ticks" / f"mode={args.mode}" / f"instrument={args.instrument}"

    # optional instrument inference / mapping
    inferred_src_inst = _infer_from_path(src_root, "instrument") if args.instrument_from_path else None
    inst_map: dict[str, str] = {}
    if args.instrument_map.strip():
        for kv in args.instrument_map.split(","):
            k, v = kv.split("=", 1)
            inst_map[k.strip()] = v.strip()

    if inferred_src_inst and not args.instrument_override:
        mapped = inst_map.get(inferred_src_inst, inferred_src_inst)
        args.instrument_override = mapped

    dt_dirs = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("dt=")])
    if args.limit_days and args.limit_days > 0:
        dt_dirs = dt_dirs[: args.limit_days]
    if not dt_dirs:
        raise FileNotFoundError(f"No dt=... folders found under: {src_root}")

    def _write_day(df_day: pl.DataFrame, *, day: str) -> None:
        out_dir = out_root / f"dt={day}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.out_name

        if out_path.exists():
            if args.merge_into_existing:
                existing = pl.read_parquet(out_path)
                existing = normalize(
                    existing,
                    instrument=args.instrument,
                    instrument_override=None,
                    tz_policy=args.tz_policy,
                )
                df_day = merge_existing(existing, df_day)
            elif not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite: {out_path} (use --overwrite or --merge-into-existing)")

        df_day.write_parquet(out_path)
        print(f"WROTE {out_path} rows={df_day.height}")

    for dt_dir in dt_dirs:
        src_files = sorted(dt_dir.glob(args.src_glob))
        if not src_files:
            print(f"SKIP {dt_dir.name}: no files matched {args.src_glob}")
            continue

        lf = pl.scan_parquet([str(p) for p in src_files])
        df = lf.collect(streaming=True)

        df = normalize(
            df,
            instrument=args.instrument,
            instrument_override=args.instrument_override,
            tz_policy=args.tz_policy,
        )

        if args.dt_from_ts:
            # repartition into engine dt=... based on ts
            out_dt = _dt_str_from_ts(df, dt_scope=args.dt_scope, market_tz=args.market_tz)
            df = df.with_columns(out_dt.alias("_out_dt"))

            parts = df.partition_by("_out_dt", as_dict=True)
            for day, part in sorted(parts.items(), key=lambda kv: kv[0]):
                _write_day(part.drop(["_out_dt"]), day=str(day))
        else:
            # legacy behavior: trust dt folder naming; guard warns if mismatch
            day = dt_dir.name.replace("dt=", "")
            day_start = _parse_dt_day(dt_dir.name)
            _validate_ts_in_dt_day(
                df,
                day_start=day_start,
                on_violation=args.dt_guard,
                label=f"incoming day={day} root={src_root}",
            )
            _write_day(df, day=day)


if __name__ == "__main__":
    main()
