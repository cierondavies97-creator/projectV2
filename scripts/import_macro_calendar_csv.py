from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

OUT_NAME = "calendar.parquet"

CANON_COLS = [
    "event_id",
    "event_ts",
    "blackout_start_ts",
    "blackout_end_ts",
    "currency",
    "impact",
    "name",
    "source",
    "raw_start",
]


@dataclass(frozen=True)
class BlackoutRule:
    pre_min: int
    post_min: int


def _parse_map(s: str) -> dict[str, str]:
    """
    "A=B,C=D" -> {"A":"B","C":"D"}
    """
    out: dict[str, str] = {}
    if not s:
        return out
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_blackout_map(s: str) -> dict[str, BlackoutRule]:
    """
    "LOW=10:10,MEDIUM=30:30,HIGH=60:60" -> {impact: (pre, post)}
    """
    out: dict[str, BlackoutRule] = {}
    if not s:
        return out
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=", 1)
        pre_s, post_s = v.split(":", 1)
        out[k.strip().upper()] = BlackoutRule(pre_min=int(pre_s), post_min=int(post_s))
    return out


def _event_id(ts: datetime, currency: str, impact: str, name: str, source: str) -> str:
    s = f"{ts.isoformat()}|{currency}|{impact}|{name}|{source}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _parse_ts_expr(start_tz: str) -> pl.Expr:
    """
    Robust Start->event_ts parsing.
    Produces UTC-naive Datetime[us].
    """
    s = pl.col("raw_start").cast(pl.Utf8).str.strip_chars()

    parsed = pl.coalesce(
        [
            s.str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M", strict=False),
            s.str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S", strict=False),
            s.str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M", strict=False),
            s.str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M:%S", strict=False),
            s.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M", strict=False),
            s.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
            s.str.strptime(pl.Datetime, strict=False),
        ]
    )

    return (
        parsed.dt.replace_time_zone(start_tz)
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
        .cast(pl.Datetime("us"))
        .alias("event_ts")
    )


def _compute_blackouts(
    df: pl.DataFrame,
    *,
    default_rule: BlackoutRule,
    rules_by_impact: dict[str, BlackoutRule],
) -> pl.DataFrame:
    # Polars-friendly rule selection: map impact->(pre,post) via join on small lookup table.
    if not rules_by_impact:
        pre = pl.lit(default_rule.pre_min)
        post = pl.lit(default_rule.post_min)
        return df.with_columns(
            (pl.col("event_ts") - pl.duration(minutes=pre)).alias("blackout_start_ts"),
            (pl.col("event_ts") + pl.duration(minutes=post)).alias("blackout_end_ts"),
        )

    lut = pl.DataFrame(
        {
            "impact": list(rules_by_impact.keys()),
            "pre_min": [r.pre_min for r in rules_by_impact.values()],
            "post_min": [r.post_min for r in rules_by_impact.values()],
        }
    )

    df2 = df.join(lut, on="impact", how="left").with_columns(
        pl.col("pre_min").fill_null(default_rule.pre_min),
        pl.col("post_min").fill_null(default_rule.post_min),
    )

    return df2.with_columns(
        (pl.col("event_ts") - pl.duration(minutes=pl.col("pre_min"))).alias("blackout_start_ts"),
        (pl.col("event_ts") + pl.duration(minutes=pl.col("post_min"))).alias("blackout_end_ts"),
    ).drop(["pre_min", "post_min"])


def normalize_calendar(
    df: pl.DataFrame,
    *,
    source_name: str,
    start_tz: str,
    on_parse_fail: str,
    currency_map: dict[str, str],
    impact_map: dict[str, str],
    default_blackout: BlackoutRule,
    blackout_by_impact: dict[str, BlackoutRule],
) -> pl.DataFrame:
    cols = set(df.columns)

    # Drop obvious index/junk columns
    if "Unnamed: 0" in cols:
        df = df.drop("Unnamed: 0")

    # Rename common vendor headings -> canonical raw fields
    ren: dict[str, str] = {}
    if "Start" in cols:
        ren["Start"] = "raw_start"
    if "Name" in cols:
        ren["Name"] = "name"
    if "Impact" in cols:
        ren["Impact"] = "impact"
    if "Currency" in cols:
        ren["Currency"] = "currency"
    df = df.rename(ren)

    missing = [c for c in ["raw_start", "name", "impact", "currency"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {df.columns}")

    df = df.with_columns(
        _parse_ts_expr(start_tz),
        pl.col("currency").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        pl.col("impact").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
        pl.col("name").cast(pl.Utf8).str.strip_chars(),
        pl.lit(source_name).cast(pl.Utf8).alias("source"),
    )

    # Drop / error on parse failures
    bad = df.filter(pl.col("event_ts").is_null())
    if bad.height > 0:
        sample = bad.select(["raw_start", "currency", "impact", "name"]).head(20)
        msg = (
            f"Failed to parse {bad.height} rows in Start->event_ts.\n"
            f"Check --start-tz and/or add a format. Sample bad rows:\n{sample}"
        )
        if on_parse_fail == "error":
            raise ValueError(msg)
        print("[warn_drop] " + msg)
        df = df.filter(pl.col("event_ts").is_not_null())

    # Apply optional normalization maps (for consistency across sources)
    if currency_map:
        df = df.with_columns(pl.col("currency").replace(currency_map, default=pl.col("currency")).alias("currency"))
    if impact_map:
        df = df.with_columns(pl.col("impact").replace(impact_map, default=pl.col("impact")).alias("impact"))

    # Compute blackout windows
    df = _compute_blackouts(df, default_rule=default_blackout, rules_by_impact=blackout_by_impact)

    # Deterministic event_id (sha1) – stable across machines + polars versions
    df = df.with_columns(
        pl.struct(["event_ts", "currency", "impact", "name", "source"])
        .map_elements(
            lambda r: _event_id(r["event_ts"], r["currency"], r["impact"], r["name"], r["source"]),
            return_dtype=pl.Utf8,
        )
        .alias("event_id")
    )

    return df.select(CANON_COLS).sort(["event_ts", "currency", "impact", "name"])


def _coerce_existing_to_canonical(existing: pl.DataFrame) -> pl.DataFrame:
    """
    Make merges robust across earlier importer versions:
    - rename ts->event_ts if needed
    - add blackout cols if missing (as null; caller can recompute if desired)
    - ensure all canonical cols exist and order matches
    """
    if "ts" in existing.columns and "event_ts" not in existing.columns:
        existing = existing.rename({"ts": "event_ts"})

    for c in CANON_COLS:
        if c not in existing.columns:
            # choose dtype based on column role
            if c in ("event_ts", "blackout_start_ts", "blackout_end_ts"):
                existing = existing.with_columns(pl.lit(None).cast(pl.Datetime("us")).alias(c))
            else:
                existing = existing.with_columns(pl.lit(None).cast(pl.Utf8).alias(c))

    return existing.select(CANON_COLS)


def merge_existing(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    existing = _coerce_existing_to_canonical(existing)
    incoming = incoming.select(CANON_COLS)
    merged = pl.concat([existing, incoming], how="vertical_relaxed")
    merged = merged.unique(subset=["event_id"], keep="last")
    return merged.sort(["event_ts", "currency", "impact", "name"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to macro calendar CSV")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--start-tz", default="UTC", help="Timezone of Start column, e.g. UTC or Europe/London")
    ap.add_argument("--on-parse-fail", choices=["error", "warn_drop"], default="error")

    # Optional normalization maps for consistency across vendors/sources
    ap.add_argument("--currency-map", default="", help='e.g. "USD=USD,US$=USD"')
    ap.add_argument("--impact-map", default="", help='e.g. "LOW=LOW,MED=MEDIUM,HIGH=HIGH"')

    # Blackout rules
    ap.add_argument("--blackout-default", default="30:30", help="default pre:post minutes, e.g. 30:30")
    ap.add_argument(
        "--blackout-by-impact",
        default="HIGH=60:60,MEDIUM=30:30,LOW=10:10",
        help='impact->pre:post minutes, e.g. "HIGH=60:60,MEDIUM=30:30,LOW=10:10"',
    )

    ap.add_argument(
        "--dt-guard",
        choices=["ignore", "warn", "error"],
        default="warn",
        help="Validate event_ts stays inside its dt partition (UTC day).",
    )
    ap.add_argument("--merge-into-existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    root = Path(args.project_root)
    out_root = root / "data" / "macro" / "calendar"

    currency_map = _parse_map(args.currency_map)
    impact_map = {k.upper(): v.upper() for k, v in _parse_map(args.impact_map).items()}

    dpre, dpost = args.blackout_default.split(":", 1)
    default_blackout = BlackoutRule(pre_min=int(dpre), post_min=int(dpost))
    blackout_by_impact = _parse_blackout_map(args.blackout_by_impact)

    df = pl.read_csv(src)
    df = normalize_calendar(
        df,
        source_name=src.name,
        start_tz=args.start_tz,
        on_parse_fail=args.on_parse_fail,
        currency_map=currency_map,
        impact_map=impact_map,
        default_blackout=default_blackout,
        blackout_by_impact=blackout_by_impact,
    )

    # Partition by UTC day derived from canonical event_ts
    df = df.with_columns(pl.col("event_ts").dt.date().alias("dt"))

    for dt_value, part in df.partition_by("dt", as_dict=True).items():
        # polars partition_by(as_dict=True) uses tuple keys; normalize robustly
        if isinstance(dt_value, tuple):
            dt_value = dt_value[0]
        day = str(dt_value)  # YYYY-MM-DD

        out_dir = out_root / f"dt={day}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / OUT_NAME

        payload = part.drop("dt").select(CANON_COLS)

        # dt-guard
        if args.dt_guard != "ignore" and payload.height > 0:
            day_start = datetime.strptime(day, "%Y-%m-%d")
            day_end = day_start + timedelta(days=1)
            ts_min = payload.select(pl.col("event_ts").min()).item()
            ts_max = payload.select(pl.col("event_ts").max()).item()
            ok = (ts_min >= day_start) and (ts_max < day_end)
            if not ok:
                msg = (
                    f"[dt-guard] dt={day}: event_ts range outside UTC day.\n"
                    f"  expected: [{day_start} .. {day_end})\n"
                    f"  actual:   [{ts_min} .. {ts_max}]\n"
                    f"  hint: change --start-tz (common fix) or fix source timestamps."
                )
                if args.dt_guard == "error":
                    raise ValueError(msg)
                print(msg)

        if out_path.exists():
            if args.merge_into_existing:
                existing = pl.read_parquet(out_path)
                merged = merge_existing(existing, payload)
                merged.write_parquet(out_path)
            elif args.overwrite:
                payload.write_parquet(out_path)
            else:
                raise FileExistsError(f"Refusing to overwrite: {out_path} (use --overwrite or --merge-into-existing)")
        else:
            payload.write_parquet(out_path)

        print(f"WROTE {out_path} rows={payload.height}")


if __name__ == "__main__":
    main()
