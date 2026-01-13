from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from engine.core.api import validate_anchor_grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--inst", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--anchor-tf", default="M5")
    args = ap.parse_args()

    p = (
        Path("data")
        / "windows"
        / f"run_id={args.run_id}"
        / f"instrument={args.inst}"
        / f"anchor_tf={args.anchor_tf}"
        / f"dt={args.dt}"
        / "0000.parquet"
    )

    w = pl.read_parquet(p)

    chk = validate_anchor_grid(
        w,
        anchor_tf=str(args.anchor_tf),
        ts_col="anchor_ts",
        max_sample=20,
    )

    print(f"windows rows={chk.rows} bad_rows={chk.bad_rows}")
    if chk.bad_rows:
        if chk.sample_bad is not None and chk.sample_bad.height:
            print(chk.sample_bad)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
