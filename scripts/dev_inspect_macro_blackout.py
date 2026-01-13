#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger("dev_inspect_macro_blackout")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD") from exc


def _read_parquet_glob(pattern: str) -> pl.DataFrame:
    """
    Load all parquet files matching a glob pattern into a single DataFrame.

    Returns an empty frame if nothing matches.
    """
    paths = list(Path().glob(pattern))
    if not paths:
        return pl.DataFrame([])
    return pl.scan_parquet(pattern).collect()


def _load_features(
    *,
    base_dir: Path,
    run_id: str,
    instrument: str,
    anchor_tf: str,
    trading_day: date,
) -> pl.DataFrame:
    """
    Load features for (run_id, instrument, anchor_tf, trading_day).

    Expected layout:
      data/features/run_id=<RUN_ID>/instrument=<INSTRUMENT>/anchor_tf=<ANCHOR_TF>/dt=<YYYY-MM-DD>/*.parquet
    """
    pattern = (
        base_dir
        / "features"
        / f"run_id={run_id}"
        / f"instrument={instrument}"
        / f"anchor_tf={anchor_tf}"
        / f"dt={trading_day:%Y-%m-%d}"
        / "*.parquet"
    )

    logger.info("Loading features from pattern: %s", pattern)
    df = _read_parquet_glob(str(pattern))
    if df.is_empty():
        logger.warning("No features found for pattern %s", pattern)
    return df


def _load_trade_paths(
    *,
    base_dir: Path,
    run_id: str,
    instrument: str,
    trading_day: date,
) -> pl.DataFrame:
    """
    Load trade_paths for (run_id, instrument, trading_day).

    Canonical expected layout (mirrors windows/features):
      data/trade_paths/run_id=<RUN_ID>/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/*.parquet

    Fallback layout (older / alternative):
      data/trade_paths/instrument=<INSTRUMENT>/dt=<YYYY-MM-DD>/*.parquet
    """
    canonical_pattern = (
        base_dir
        / "trade_paths"
        / f"run_id={run_id}"
        / f"instrument={instrument}"
        / f"dt={trading_day:%Y-%m-%d}"
        / "*.parquet"
    )

    logger.info("Loading trade_paths (canonical) from pattern: %s", canonical_pattern)
    df = _read_parquet_glob(str(canonical_pattern))

    if df.is_empty():
        fallback_pattern = (
            base_dir / "trade_paths" / f"instrument={instrument}" / f"dt={trading_day:%Y-%m-%d}" / "*.parquet"
        )
        logger.warning(
            "No trade_paths found at canonical layout; trying fallback pattern: %s",
            fallback_pattern,
        )
        df = _read_parquet_glob(str(fallback_pattern))

    if df.is_empty():
        logger.warning(
            "No trade_paths found for run_id=%s instrument=%s dt=%s in either canonical or fallback layouts.",
            run_id,
            instrument,
            trading_day,
        )
        return df

    # If we loaded from fallback and have run_id, filter just in case
    if "run_id" in df.columns:
        df = df.filter(pl.col("run_id") == run_id)

    return df


def _join_trades_with_macro(
    *,
    features: pl.DataFrame,
    trade_paths: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join trade_paths with macro blackout flags.

    Join keys:
      - instrument
      - anchor_tf
      - entry_ts == ts (from features)

    Assumes features has at least:
      - 'instrument', 'anchor_tf', 'ts'

    Uses 'macro_is_blackout' and 'macro_blackout_max_impact' if present.
    """
    if trade_paths.is_empty():
        logger.warning("No trade_paths rows to join.")
        return trade_paths

    required_feat_cols = {"instrument", "anchor_tf", "ts"}
    missing = required_feat_cols - set(features.columns)
    if missing:
        logger.warning(
            "Features missing columns %s, cannot join trades to macro features; returning trade_paths unchanged.",
            sorted(missing),
        )
        return trade_paths

    feat = features.select(
        [
            c
            for c in features.columns
            if c
            in {
                "instrument",
                "anchor_tf",
                "ts",
                "macro_is_blackout",
                "macro_blackout_max_impact",
            }
        ]
    )

    # Align timestamp naming and type
    feat = feat.rename({"ts": "entry_ts"}).with_columns(pl.col("entry_ts").cast(pl.Datetime("ns")))

    if "entry_ts" not in trade_paths.columns:
        logger.warning("trade_paths missing 'entry_ts'; cannot join on time. Returning trade_paths unchanged.")
        return trade_paths

    tp = trade_paths.with_columns(pl.col("entry_ts").cast(pl.Datetime("ns")))

    joined = tp.join(
        feat,
        on=["instrument", "anchor_tf", "entry_ts"],
        how="left",
    )

    # Fill nulls for macro columns if present
    if "macro_is_blackout" in joined.columns:
        joined = joined.with_columns(pl.col("macro_is_blackout").fill_null(False))
    if "macro_blackout_max_impact" in joined.columns:
        joined = joined.with_columns(pl.col("macro_blackout_max_impact").fill_null(0))

    return joined


def _summarise_bars(features: pl.DataFrame) -> None:
    """
    Print bar-level summary of blackout coverage.
    """
    if features.is_empty():
        print("\n[BAR SUMMARY] No features loaded; cannot summarise blackout coverage.")
        return

    total = features.height
    has_macro = "macro_is_blackout" in features.columns

    if not has_macro:
        print(f"\n[BAR SUMMARY] features rows={total}, macro_is_blackout column not present.")
        return

    blackout_rows = features.select(pl.col("macro_is_blackout").cast(pl.Int64).sum().alias("n_blackout"))["n_blackout"][
        0
    ]
    pct = (blackout_rows / total * 100.0) if total > 0 else 0.0

    print("\n[BAR SUMMARY]")
    print(f"  total bars        : {total}")
    print(f"  blackout bars     : {blackout_rows}")
    print(f"  blackout coverage : {pct:.2f} %")


def _summarise_trades(trades_joined: pl.DataFrame) -> None:
    """
    Print trade-level summary and a preview table.
    """
    if trades_joined.is_empty():
        print("\n[TRADE SUMMARY] No trades found.")
        return

    total = trades_joined.height
    has_macro = "macro_is_blackout" in trades_joined.columns

    if has_macro:
        blackout_trades = trades_joined.select(pl.col("macro_is_blackout").cast(pl.Int64).sum().alias("n"))["n"][0]
    else:
        blackout_trades = 0

    print("\n[TRADE SUMMARY]")
    print(f"  total trades             : {total}")
    if has_macro:
        pct = (blackout_trades / total * 100.0) if total > 0 else 0.0
        print(f"  trades in blackout       : {blackout_trades}")
        print(f"  trades in blackout (pct) : {pct:.2f} %")
    else:
        print("  macro_is_blackout not present on trades; cannot classify by blackout.")

    # Preview a small table of trades
    preview_cols: list[str] = []
    for c in [
        "trade_id",
        "instrument",
        "anchor_tf",
        "side",
        "entry_ts",
        "macro_is_blackout",
        "macro_blackout_max_impact",
    ]:
        if c in trades_joined.columns:
            preview_cols.append(c)

    print("\n[TRADE PREVIEW] (up to 20 rows)")
    if preview_cols:
        preview = trades_joined.select(preview_cols).sort(["entry_ts"]).head(20)
        # Use Polars' default pretty-print
        print(preview)
    else:
        print("  No preview columns available.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect macro blackout coverage for a given run/day and show which trades fell inside blackout windows."
        )
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID (e.g. dev_full_pipeline_20250102)",
    )
    parser.add_argument(
        "--trading-day",
        type=_parse_date,
        required=True,
        help="Trading day (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--instrument",
        required=True,
        help="Instrument (e.g. XAUUSD)",
    )
    parser.add_argument(
        "--anchor-tf",
        default="M5",
        help="Anchor timeframe (default: M5)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Inspecting macro blackout for run_id=%s day=%s instrument=%s anchor_tf=%s",
        args.run_id,
        args.trading_day,
        args.instrument,
        args.anchor_tf,
    )

    features = _load_features(
        base_dir=args.base_dir,
        run_id=args.run_id,
        instrument=args.instrument,
        anchor_tf=args.anchor_tf,
        trading_day=args.trading_day,
    )

    trade_paths = _load_trade_paths(
        base_dir=args.base_dir,
        run_id=args.run_id,
        instrument=args.instrument,
        trading_day=args.trading_day,
    )

    _summarise_bars(features)

    trades_joined = _join_trades_with_macro(
        features=features,
        trade_paths=trade_paths,
    )
    _summarise_trades(trades_joined)


if __name__ == "__main__":
    main()
