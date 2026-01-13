from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import polars as pl

REQUIRED = ["instrument", "tf", "ts", "open", "high", "low", "close", "volume"]


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename({"timestamp": "ts"})

    missing = [c for c in ["instrument", "tf", "ts", "open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {df.columns}")

    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("volume"))

    df = df.with_columns(
        pl.col("instrument").cast(pl.Utf8),
        pl.col("tf").cast(pl.Utf8),
        pl.col("ts").cast(pl.Datetime("us")),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    )

    # Canonical candle key
    df = df.unique(subset=["instrument", "tf", "ts"], keep="last")
    return df.select(REQUIRED).sort(["instrument", "tf", "ts"])


def iter_dt_dirs(candles_root: Path) -> Iterable[Path]:
    if not candles_root.exists():
        return []
    return sorted([p for p in candles_root.iterdir() if p.is_dir() and p.name.startswith("dt=")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="backtest")
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--tfs", required=True, help="Comma-separated, e.g. M1,M5,M15")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--in-name", default="0000.parquet")
    ap.add_argument("--out-name", default="0000.parquet")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root)
    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    if not tfs:
        raise ValueError("No TFs provided in --tfs")

    in_root = root / "data" / "candles" / f"mode={args.mode}" / f"instrument={args.instrument}"
    out_root = root / "data" / "research" / "candles_tf" / f"mode={args.mode}" / f"instrument={args.instrument}"

    dt_dirs = iter_dt_dirs(in_root)
    if not dt_dirs:
        raise FileNotFoundError(f"No dt=... dirs under canonical candles root: {in_root}")

    for dt_dir in dt_dirs:
        day = dt_dir.name.replace("dt=", "")
        in_path = dt_dir / args.in_name
        if not in_path.exists():
            # Allow globbed multi-part, but your canonical is typically 0000.parquet
            parts = sorted(dt_dir.glob("*.parquet"))
            if not parts:
                print(f"SKIP {day}: no parquet files found in {dt_dir}")
                continue
            df = pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")
        else:
            df = pl.read_parquet(in_path)

        df = normalize(df)

        for tf in tfs:
            df_tf = df.filter(pl.col("tf") == tf)
            if df_tf.is_empty():
                continue

            out_dir = out_root / f"tf={tf}" / f"dt={day}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / args.out_name

            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite: {out_path} (use --overwrite)")

            df_tf.write_parquet(out_path)
            print(f"WROTE {out_path} rows={df_tf.height}")


if __name__ == "__main__":
    main()
