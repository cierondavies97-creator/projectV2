from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG = logging.getLogger("dev_seed_macro_calendar")


@dataclass
class DevMacroEventTemplate:
    """Template for generating synthetic macro events per day."""

    label: str
    event_time: time  # time of the event (UTC)
    blackout_before_min: int  # minutes before event
    blackout_after_min: int  # minutes after event
    currency: str
    impact: int  # arbitrary numeric impact (e.g. 1=low, 3=high)


DEFAULT_TEMPLATES: list[DevMacroEventTemplate] = [
    DevMacroEventTemplate(
        label="morning_event",
        event_time=time(9, 0),
        blackout_before_min=15,
        blackout_after_min=15,
        currency="USD",
        impact=2,
    ),
    DevMacroEventTemplate(
        label="afternoon_event",
        event_time=time(13, 30),
        blackout_before_min=15,
        blackout_after_min=15,
        currency="USD",
        impact=3,
    ),
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD") from exc


def _daterange_inclusive(start: date, end: date):
    if end < start:
        raise ValueError(f"end date {end} < start date {start}")
    cur = start
    one_day = timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one_day


def _build_events_for_day(day: date, templates: list[DevMacroEventTemplate]) -> pl.DataFrame:
    rows = []

    for idx, tmpl in enumerate(templates, start=1):
        event_dt = datetime.combine(day, tmpl.event_time)

        blackout_start = event_dt - timedelta(minutes=tmpl.blackout_before_min)
        blackout_end = event_dt + timedelta(minutes=tmpl.blackout_after_min)

        rows.append(
            {
                "event_id": f"{day:%Y%m%d}_{idx}_{tmpl.label}",
                "event_ts": event_dt,
                "currency": tmpl.currency,
                "impact": tmpl.impact,
                "blackout_start_ts": blackout_start,
                "blackout_end_ts": blackout_end,
            }
        )

    if not rows:
        return pl.DataFrame([])

    return pl.DataFrame(rows)


def _target_path_for_day(root: Path, day: date) -> Path:
    # Must match engine.features.macro_calendar._load_macro_calendar_for_day
    # base = os.path.join("data", "macro", "calendar")
    # path = os.path.join(base, f"dt={trading_day:%Y-%m-%d}", "calendar.parquet")
    return root / f"dt={day:%Y-%m-%d}" / "calendar.parquet"


def seed_macro_calendar(
    *,
    start: date,
    end: date,
    out_root: Path,
    overwrite: bool,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    for day in _daterange_inclusive(start, end):
        df = _build_events_for_day(day, DEFAULT_TEMPLATES)
        out_path = _target_path_for_day(out_root, day)

        if out_path.exists() and not overwrite:
            LOG.info("Skipping %s (exists, overwrite=False)", out_path)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        LOG.info(
            "Wrote %d macro events for %s to %s",
            df.height,
            day.isoformat(),
            out_path,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Seed synthetic macro calendar Parquet files for dev backtests.\n\n"
            "Writes to data/macro/calendar/dt=YYYY-MM-DD/calendar.parquet "
            "in the schema expected by engine.features.macro_calendar."
        )
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        required=True,
        help="Start date (inclusive), format YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        required=True,
        help="End date (inclusive), format YYYY-MM-DD",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data") / "macro" / "calendar",
        help="Root directory for calendar data (default: data/macro/calendar)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Parquet files if they already exist.",
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

    LOG.info(
        "Seeding macro calendar from %s to %s under %s (overwrite=%s)",
        args.start,
        args.end,
        os.fspath(args.out_root),
        args.overwrite,
    )

    seed_macro_calendar(
        start=args.start,
        end=args.end,
        out_root=args.out_root,
        overwrite=args.overwrite,
    )

    LOG.info("Done.")


if __name__ == "__main__":
    main()
