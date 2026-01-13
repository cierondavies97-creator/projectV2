#!/usr/bin/env python
"""
Lightweight research sweep over stat_ts_vol_zscore.abs_min_for_signal.

Changes vs v1
-------------
- Removed dependency on dev_inspect_memory_model.py.
- Parse `trade_paths: N rows` directly from run_microbatch.py stdout.
- Treat train_features_thresholds_stub "no-op" as success (baseline threshold).

Design notes
------------
- Research lane only; does NOT change engine code.
- Temporarily overwrites conf/features_auto.yaml using train_features_thresholds_stub.py,
  runs microbatches, then restores the original YAML.
"""

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf"
FEATURES_AUTO_PATH = CONF_DIR / "features_auto.yaml"


@dataclass
class SweepConfig:
    env: str
    snapshot_id: str
    cluster_id: str
    instrument: str
    anchor_tf: str
    paradigm_id: str
    start_day: date
    end_day: date
    thresholds: list[float]
    abs_max_clip: float
    run_prefix: str
    out_csv: Path | None


def _parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(
        description="Run a small sweep over stat_ts_vol_zscore.abs_min_for_signal and count trade_paths per day."
    )

    parser.add_argument("--env", default="research")
    parser.add_argument("--snapshot-id", default="dev_retail_v1")
    parser.add_argument("--cluster-id", default="metals")
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--anchor-tf", default="M5")
    parser.add_argument("--paradigm-id", default="dev_baseline")

    parser.add_argument("--start-day", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-day", required=True, help="YYYY-MM-DD")

    parser.add_argument(
        "--thresholds",
        required=True,
        help="Comma-separated list, e.g. '0.5,1.0,1.5,2.0'",
    )
    parser.add_argument(
        "--abs-max-clip",
        type=float,
        default=3.0,
        help="abs_max_clip to keep fixed during the sweep.",
    )
    parser.add_argument(
        "--run-prefix",
        default="dev_sweep_vol_absmin",
        help="Prefix for run_id; threshold and date will be appended.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional path to write CSV summary. If empty, no file is written.",
    )

    args = parser.parse_args()

    start_day = date.fromisoformat(args.start_day)
    end_day = date.fromisoformat(args.end_day)
    if end_day < start_day:
        parser.error("--end-day must be >= --start-day")

    thresholds: list[float] = []
    for raw in args.thresholds.split(","):
        val = raw.strip()
        if not val:
            continue
        thresholds.append(float(val))

    out_csv = Path(args.out_csv) if args.out_csv else None

    return SweepConfig(
        env=args.env,
        snapshot_id=args.snapshot_id,
        cluster_id=args.cluster_id,
        instrument=args.instrument,
        anchor_tf=args.anchor_tf,
        paradigm_id=args.paradigm_id,
        start_day=start_day,
        end_day=end_day,
        thresholds=thresholds,
        abs_max_clip=args.abs_max_clip,
        run_prefix=args.run_prefix,
        out_csv=out_csv,
    )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess, capture output, and echo it to the console."""
    print(f"\n>>> {' '.join(cmd)}")
    res = subprocess.run(cmd, text=True, capture_output=True)
    if res.stdout:
        print(res.stdout, end="")
    if res.stderr:
        # Keep stderr visible for debugging
        print(res.stderr, file=sys.stderr, end="")
    return res


def _extract_trade_paths_rows_from_output(text: str) -> int | None:
    """
    Parse 'trade_paths: 12 rows' from run_microbatch.py output.
    Returns int or None if not found.
    """
    m = re.search(r"trade_paths:\s+(\d+)\s+rows", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _write_csv(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["threshold", "day", "run_id", "n_trade_paths"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote sweep summary to {out_path}")


def main() -> None:
    cfg = _parse_args()

    print("Project root:", PROJECT_ROOT)
    print("Using features_auto.yaml at:", FEATURES_AUTO_PATH)

    original_text: str | None = None
    if FEATURES_AUTO_PATH.exists():
        original_text = FEATURES_AUTO_PATH.read_text(encoding="utf-8")
        backup_path = FEATURES_AUTO_PATH.with_suffix(".yaml.bak_sweep")
        backup_path.write_text(original_text, encoding="utf-8")
        print(f"Backed up existing features_auto.yaml -> {backup_path}")
    else:
        print("No existing features_auto.yaml found; sweep will create one temporarily.")

    rows: list[dict] = []

    try:
        for thr in cfg.thresholds:
            print("\n" + "=" * 80)
            print(f"Sweeping abs_min_for_signal = {thr}")
            print("=" * 80)

            # 1) Rewrite features_auto.yaml via stub trainer for this threshold
            stub_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_features_thresholds_stub.py"),
                "--paradigm-id",
                cfg.paradigm_id,
                "--out",
                str(FEATURES_AUTO_PATH),
                "--set",
                f"abs_min_for_signal={thr}",
                "--set",
                f"abs_max_clip={cfg.abs_max_clip}",
                "--force-create-paths",
            ]
            stub_res = _run(stub_cmd)
            if stub_res.returncode != 0:
                # Special case: baseline "no-op" is OK; use existing YAML.
                if "No changes applied (no-op)" in stub_res.stdout:
                    print(
                        f"Stub trainer reported no-op for threshold={thr}. "
                        f"Treating as baseline (keeping existing features_auto.yaml)."
                    )
                else:
                    print(f"WARNING: stub trainer failed for threshold={thr}, skipping.")
                    continue

            # 2) For each day, run microbatch and parse trade_paths directly
            d = cfg.start_day
            while d <= cfg.end_day:
                day_str = d.isoformat()
                thr_str = str(thr).replace(".", "p")
                run_id = f"{cfg.run_prefix}_{d.strftime('%Y%m%d')}_min{thr_str}"

                print(f"\n--- {day_str} | run_id={run_id} ---")

                run_cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_microbatch.py"),
                    "--env",
                    cfg.env,
                    "--snapshot-id",
                    cfg.snapshot_id,
                    "--run-id",
                    run_id,
                    "--trading-day",
                    day_str,
                    "--cluster-id",
                    cfg.cluster_id,
                    "--log-level",
                    "INFO",
                ]
                run_res = _run(run_cmd)
                if run_res.returncode != 0:
                    print(f"WARNING: microbatch failed for {day_str}, threshold={thr}")
                    d += timedelta(days=1)
                    continue

                n_trades = _extract_trade_paths_rows_from_output(run_res.stdout)

                print(
                    f"Result: day={day_str}, threshold={thr}, "
                    f"trade_paths rows={n_trades if n_trades is not None else 'N/A'}"
                )

                rows.append(
                    {
                        "threshold": thr,
                        "day": day_str,
                        "run_id": run_id,
                        "n_trade_paths": n_trades,
                    }
                )

                d += timedelta(days=1)

    finally:
        # Restore original features_auto.yaml so we do not leave research tweaks live.
        if original_text is not None:
            FEATURES_AUTO_PATH.write_text(original_text, encoding="utf-8")
            print("\nRestored original conf/features_auto.yaml from backup.")
        else:
            print("\nNo original features_auto.yaml existed; leaving last sweep version on disk.")

    # Optional CSV output
    if cfg.out_csv and rows:
        _write_csv(cfg.out_csv, rows)

    # Final console summary
    if rows:
        print("\nSummary (threshold, day, n_trade_paths):")
        for row in rows:
            print(
                f"  {row['threshold']:5.2f}  {row['day']}  "
                f"{row['n_trade_paths'] if row['n_trade_paths'] is not None else 'N/A'}"
            )
    else:
        print("\nNo successful runs recorded.")


if __name__ == "__main__":
    main()
