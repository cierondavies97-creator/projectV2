from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def must_cols(df: pl.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{label}: missing columns {missing}")
    print(f"{label}: rows={df.height} cols={len(df.columns)} OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--inst", required=True)
    ap.add_argument("--anchor-tf", default="M5")
    args = ap.parse_args()

    base = Path("data")
    run_id = args.run_id
    dt = args.dt
    inst = args.inst
    anchor_tf = args.anchor_tf

    features = pl.read_parquet(
        base
        / "features"
        / f"run_id={run_id}"
        / f"instrument={inst}"
        / f"anchor_tf={anchor_tf}"
        / f"dt={dt}"
        / "0000.parquet"
    )
    windows = pl.read_parquet(
        base
        / "windows"
        / f"run_id={run_id}"
        / f"instrument={inst}"
        / f"anchor_tf={anchor_tf}"
        / f"dt={dt}"
        / "0000.parquet"
    )
    tpaths = pl.read_parquet(
        base / "trade_paths" / f"run_id={run_id}" / f"instrument={inst}" / f"dt={dt}" / "0000.parquet"
    )
    dhyp = pl.read_parquet(
        base
        / "decisions"
        / f"run_id={run_id}"
        / f"instrument={inst}"
        / "stage=hypotheses"
        / f"dt={dt}"
        / "0000.parquet"
    )

    must_cols(features, ["snapshot_id", "run_id", "mode", "instrument", "anchor_tf", "ts"], "features")
    must_cols(windows, ["instrument", "anchor_tf", "anchor_ts", "tf_entry"], "windows")
    must_cols(
        tpaths,
        ["snapshot_id", "run_id", "mode", "instrument", "trade_id", "paradigm_id", "principle_id"],
        "trade_paths",
    )
    must_cols(
        dhyp,
        ["snapshot_id", "run_id", "mode", "instrument", "trade_id", "paradigm_id", "principle_id"],
        "decisions_hypotheses",
    )

    print("trade_paths head:")
    print(
        tpaths.select(
            [c for c in ["run_id", "instrument", "trade_id", "paradigm_id", "principle_id"] if c in tpaths.columns]
        ).head(5)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
