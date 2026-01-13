from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from engine.features import FeatureBuildContext

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ta_trend — Trend & momentum primitives
#
# Output schema (must match features_registry.yaml):
#   keys   : instrument, anchor_tf, ts
#   cols   : rsi_value (double), rsi_bucket (string), ema_fast (double), ema_slow (double)
#
# Runtime config:
#   ctx.features_auto_cfg["ta_trend"] keys (all optional):
#     - rsi_period: int (default 14)
#     - rsi_oversold_cut: float (default 30)
#     - rsi_overbought_cut: float (default 70)
#     - ema_fast_period: int (default 20)
#     - ema_slow_period: int (default 50)
#     - rsi_warmup_bars: int (default 2*rsi_period)  # null RSI before warmup
#     - rsi_smoothing: "wilder"|"ema" (default "wilder")
#     - doji_epsilon: float (default 1e-12)          # numerical guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Params:
    rsi_period: int
    rsi_oversold_cut: float
    rsi_overbought_cut: float
    ema_fast_period: int
    ema_slow_period: int
    rsi_warmup_bars: int
    rsi_smoothing: str
    doji_epsilon: float


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "anchor_tf": pl.Series([], dtype=pl.Utf8),
            "ts": pl.Series([], dtype=pl.Datetime("us")),
            "rsi_value": pl.Series([], dtype=pl.Float64),
            "rsi_bucket": pl.Series([], dtype=pl.Utf8),
            "ema_fast": pl.Series([], dtype=pl.Float64),
            "ema_slow": pl.Series([], dtype=pl.Float64),
        }
    )


def _read_params(ctx: FeatureBuildContext) -> _Params:
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg = dict(auto_cfg.get("ta_trend", {}) if isinstance(auto_cfg, dict) else {})

    rsi_period = max(1, int(fam_cfg.get("rsi_period", 14)))
    ema_fast_period = max(1, int(fam_cfg.get("ema_fast_period", 20)))
    ema_slow_period = max(1, int(fam_cfg.get("ema_slow_period", 50)))

    # Ensure fast <= slow conventionally; if not, swap deterministically
    if ema_fast_period > ema_slow_period:
        ema_fast_period, ema_slow_period = ema_slow_period, ema_fast_period

    smoothing = str(fam_cfg.get("rsi_smoothing", "wilder")).strip().lower()
    if smoothing not in {"wilder", "ema"}:
        smoothing = "wilder"

    warmup = int(fam_cfg.get("rsi_warmup_bars", 2 * rsi_period))
    warmup = max(0, warmup)

    return _Params(
        rsi_period=rsi_period,
        rsi_oversold_cut=float(fam_cfg.get("rsi_oversold_cut", 30.0)),
        rsi_overbought_cut=float(fam_cfg.get("rsi_overbought_cut", 70.0)),
        ema_fast_period=ema_fast_period,
        ema_slow_period=ema_slow_period,
        rsi_warmup_bars=warmup,
        rsi_smoothing=smoothing,
        doji_epsilon=float(fam_cfg.get("doji_epsilon", 1e-12)),
    )


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame,
    **_,
) -> pl.DataFrame:
    """
    Table: data/features
    Keys : instrument, anchor_tf, ts
    """
    if candles is None or candles.is_empty():
        log.warning("ta_trend: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "tf", "ts", "open", "high", "low", "close"}
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("ta_trend: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    cluster = getattr(ctx, "cluster", None)
    anchor_tfs = getattr(cluster, "anchor_tfs", None) or []
    if not anchor_tfs:
        log.warning("ta_trend: ctx.cluster.anchor_tfs empty; returning empty keyed frame")
        return _empty_keyed_frame()

    p = _read_params(ctx)

    # Normalize and de-duplicate (instrument, tf, ts). Keep last occurrence deterministically.
    c = (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("tf").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col("close").cast(pl.Float64),
        )
        .drop_nulls(["instrument", "tf", "ts"])
        .sort(["instrument", "tf", "ts"])
        .unique(subset=["instrument", "tf", "ts"], keep="last")
    )

    out_frames: list[pl.DataFrame] = []

    for anchor_tf in anchor_tfs:
        tf_str = str(anchor_tf)
        df = c.filter(pl.col("tf") == pl.lit(tf_str))
        if df.is_empty():
            continue

        df = df.with_columns(pl.lit(tf_str).alias("anchor_tf")).sort(["instrument", "ts"])

        # -------------------------------------------------------------------
        # EMA fast/slow
        # -------------------------------------------------------------------
        df = df.with_columns(
            pl.col("close").ewm_mean(span=p.ema_fast_period, adjust=False).over("instrument").alias("ema_fast"),
            pl.col("close").ewm_mean(span=p.ema_slow_period, adjust=False).over("instrument").alias("ema_slow"),
        )

        # -------------------------------------------------------------------
        # RSI (Wilder RMA or EMA variant)
        # RSI definition:
        #   diff = close[t] - close[t-1]
        #   gain = max(diff, 0)
        #   loss = max(-diff, 0)
        #   avg_gain = RMA(gain, period)
        #   avg_loss = RMA(loss, period)
        #   RS = avg_gain / avg_loss
        #   RSI = 100 - 100/(1+RS)
        #
        # Guard:
        #   avg_loss == 0 -> RSI = 100 (if avg_gain > 0), else RSI = 50 (flat series)
        # -------------------------------------------------------------------
        diff = pl.col("close").diff().over("instrument")

        df = df.with_columns(
            pl.when(diff > 0).then(diff).otherwise(pl.lit(0.0)).alias("_gain"),
            pl.when(diff < 0).then(-diff).otherwise(pl.lit(0.0)).alias("_loss"),
        )

        if p.rsi_smoothing == "wilder":
            # Wilder RMA uses alpha = 1/period
            alpha = 1.0 / float(p.rsi_period)
            df = df.with_columns(
                pl.col("_gain").ewm_mean(alpha=alpha, adjust=False).over("instrument").alias("_avg_gain"),
                pl.col("_loss").ewm_mean(alpha=alpha, adjust=False).over("instrument").alias("_avg_loss"),
            )
        else:
            # EMA smoothing using span=period (commonly used RSI variant)
            df = df.with_columns(
                pl.col("_gain").ewm_mean(span=p.rsi_period, adjust=False).over("instrument").alias("_avg_gain"),
                pl.col("_loss").ewm_mean(span=p.rsi_period, adjust=False).over("instrument").alias("_avg_loss"),
            )

        # Compute RSI with robust guards
        df = df.with_columns(
            pl.when(pl.col("_avg_loss") <= pl.lit(p.doji_epsilon))
            .then(
                pl.when(pl.col("_avg_gain") <= pl.lit(p.doji_epsilon))
                .then(pl.lit(50.0))  # flat series
                .otherwise(pl.lit(100.0))
            )
            .otherwise(
                100.0 - (100.0 / (1.0 + (pl.col("_avg_gain") / pl.col("_avg_loss"))))
            )
            .alias("rsi_value")
        )

        # Warmup mask (avoid unstable early RSI)
        if p.rsi_warmup_bars > 0:
            df = df.with_columns(
                pl.when(pl.int_range(0, pl.len()).over("instrument") >= pl.lit(p.rsi_warmup_bars))
                .then(pl.col("rsi_value"))
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("rsi_value")
            )

        # Buckets (three-way)
        df = df.with_columns(
            pl.when(pl.col("rsi_value").is_null())
            .then(pl.lit("na"))
            .when(pl.col("rsi_value") <= pl.lit(p.rsi_oversold_cut))
            .then(pl.lit("oversold"))
            .when(pl.col("rsi_value") >= pl.lit(p.rsi_overbought_cut))
            .then(pl.lit("overbought"))
            .otherwise(pl.lit("neutral"))
            .alias("rsi_bucket")
        )

        out = df.select(
            "instrument",
            "anchor_tf",
            pl.col("ts"),
            pl.col("rsi_value").cast(pl.Float64),
            pl.col("rsi_bucket").cast(pl.Utf8),
            pl.col("ema_fast").cast(pl.Float64),
            pl.col("ema_slow").cast(pl.Float64),
        )

        out_frames.append(out)

    if not out_frames:
        return _empty_keyed_frame()

    out_all = pl.concat(out_frames, how="vertical").sort(["instrument", "anchor_tf", "ts"])

    log.info(
        "ta_trend: built rows=%d instruments=%d anchor_tfs=%d smoothing=%s rsi_period=%d ema_fast=%d ema_slow=%d",
        out_all.height,
        out_all.select("instrument").n_unique(),
        out_all.select("anchor_tf").n_unique(),
        p.rsi_smoothing,
        p.rsi_period,
        p.ema_fast_period,
        p.ema_slow_period,
    )
    return out_all
