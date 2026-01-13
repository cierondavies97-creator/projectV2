from __future__ import annotations

from datetime import date

import polars as pl

from engine.core.ids import RunContext
from engine.core.timegrid import validate_anchor_grid
from engine.io.parquet_io import write_parquet
from engine.io.paths import zones_state_dir

# -----------------------------------------------------------------------------
# Canonicalization / Validation
# -----------------------------------------------------------------------------

_REQUIRED_KEYS = ["instrument", "anchor_tf", "zone_id", "ts"]
_IDENTITY_COLS = ["snapshot_id", "run_id", "mode", "trading_day"]


def _canonicalize_zones_state_frame(df: pl.DataFrame) -> pl.DataFrame:
    """
    Make zones_state deterministic and join-safe without mutating semantics.

    - Ensures required columns exist (validation happens elsewhere).
    - Casts key columns to stable dtypes.
    - Drops duplicate key rows.
    - Sorts by canonical key order for reproducibility.
    """
    if df.is_empty():
        return df

    # Cast key columns to stable dtypes.
    out = df.with_columns(
        pl.col("instrument").cast(pl.Utf8, strict=False),
        pl.col("anchor_tf").cast(pl.Utf8, strict=False),
        pl.col("zone_id").cast(pl.Utf8, strict=False),
        pl.col("ts").cast(pl.Datetime("ns"), strict=False),
    )

    # Drop duplicate keys (keep first deterministically after sort).
    out = out.unique(subset=_REQUIRED_KEYS, keep="first")

    # Deterministic ordering for audits and stable parquet diffs.
    out = out.sort(_REQUIRED_KEYS)

    return out


def validate_zones_state_frame(
    df: pl.DataFrame,
    *,
    enforce_anchor_grid: bool = True,
    max_bad_sample: int = 20,
) -> None:
    """
    Validation for zones_state frames before writing.

    Required:
      - instrument, anchor_tf, zone_id, ts

    Recommended (default):
      - ts conforms to anchor_tf grid (enforce_anchor_grid=True)

    Notes:
      - We validate *after* canonicalization (dtype casts).
      - If you later decide zones_state.ts can be non-grid event timestamps,
        flip enforce_anchor_grid to False explicitly at the callsite.
    """
    if df.is_empty():
        return

    missing = [c for c in _REQUIRED_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"zones_state frame is missing required columns: {missing}")

    # Dtype sanity.
    if df.schema.get("ts") not in (pl.Datetime("ns"), pl.Datetime("us"), pl.Datetime("ms")):
        raise ValueError(f"zones_state.ts must be a datetime dtype; got {df.schema.get('ts')}")

    if enforce_anchor_grid:
        # validate_anchor_grid expects a single anchor_tf value per check.
        # zones_state is partitioned by (instrument, anchor_tf) before write,
        # so for safety we validate on the whole df only if anchor_tf is singular.
        uniq = df.select(pl.col("anchor_tf").unique()).to_series().to_list()
        if len(uniq) != 1:
            raise ValueError(f"zones_state validation expects a single anchor_tf per partition; got {uniq}")

        anchor_tf = str(uniq[0])
        chk = validate_anchor_grid(
            df,
            anchor_tf=anchor_tf,
            ts_col="ts",
            max_sample=max_bad_sample,
        )
        if chk.bad_rows:
            # Provide a compact, actionable error.
            raise ValueError(
                f"zones_state.ts is not on anchor grid for anchor_tf={anchor_tf}: "
                f"bad_rows={chk.bad_rows} of {chk.rows}. Sample:\n{chk.sample_bad}"
            )


# -----------------------------------------------------------------------------
# Identity columns
# -----------------------------------------------------------------------------


def _ensure_identity_columns(
    ctx: RunContext,
    trading_day: date,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Ensure core identity columns exist for zones_state tables.
    """
    if df.is_empty():
        return df

    exprs: list[pl.Expr] = []
    if "snapshot_id" not in df.columns:
        exprs.append(pl.lit(ctx.snapshot_id).alias("snapshot_id"))
    if "run_id" not in df.columns:
        exprs.append(pl.lit(ctx.run_id).alias("run_id"))
    if "mode" not in df.columns:
        exprs.append(pl.lit(ctx.mode).alias("mode"))
    if "trading_day" not in df.columns:
        exprs.append(pl.lit(trading_day).alias("trading_day"))

    if exprs:
        df = df.with_columns(exprs)

    return df


# -----------------------------------------------------------------------------
# Writer
# -----------------------------------------------------------------------------


def write_zones_state_for_instrument_tf_day(
    ctx: RunContext,
    df: pl.DataFrame,
    instrument: str,
    anchor_tf: str,
    trading_day: date,
    *,
    sandbox: bool = False,
    enforce_anchor_grid: bool = True,
) -> None:
    """
    Write zones_state rows for a single (instrument, anchor_tf, trading_day).

    Contract:
      - Input df must contain at least: instrument, anchor_tf, zone_id, ts
      - Extra columns are allowed (future ZMF features).

    Behavior:
      - Canonicalize (cast, dedupe, sort)
      - Validate keys and (optionally) anchor grid
      - Ensure identity columns
      - Write to: data/zones_state/run_id=.../instrument=.../anchor_tf=.../dt=.../0000.parquet
    """
    if df.is_empty():
        return

    df2 = _canonicalize_zones_state_frame(df)
    validate_zones_state_frame(df2, enforce_anchor_grid=enforce_anchor_grid)

    df2 = _ensure_identity_columns(ctx, trading_day, df2)

    out_dir = zones_state_dir(
        ctx=ctx,
        trading_day=trading_day,
        instrument=instrument,
        anchor_tf=anchor_tf,
        sandbox=sandbox,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "0000.parquet"
    write_parquet(df2, out_path.parent, file_name=out_path.name)
