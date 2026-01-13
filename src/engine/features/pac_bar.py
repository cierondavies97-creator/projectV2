from __future__ import annotations

import logging

import polars as pl

from engine.features import FeatureBuildContext

log = logging.getLogger(__name__)


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "anchor_tf": pl.Series([], dtype=pl.Utf8),
            "anchor_ts": pl.Series([], dtype=pl.Datetime("us")),
            "pac_micro_state_anchor": pl.Series([], dtype=pl.Utf8),
            "pac_micro_state_pre_entry": pl.Series([], dtype=pl.Utf8),
            "pac_micro_state_post_entry": pl.Series([], dtype=pl.Utf8),
            "pac_micro_state_start": pl.Series([], dtype=pl.Utf8),
            "pac_micro_state_mid": pl.Series([], dtype=pl.Utf8),
            "pac_micro_state_end": pl.Series([], dtype=pl.Utf8),
            "pac_path_regime": pl.Series([], dtype=pl.Utf8),
            "pac_entry_immediate_rejection_flag": pl.Series([], dtype=pl.Boolean),
            "pac_entry_immediate_followthrough_flag": pl.Series([], dtype=pl.Boolean),
        }
    )


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    windows: pl.DataFrame | None = None,
    candles: pl.DataFrame | None = None,
    **_,
) -> pl.DataFrame:
    """
    Table: data/windows
    Keys : instrument, anchor_tf, anchor_ts

    Outputs (must match features_registry.yaml pac_bar columns):
      - pac_micro_state_*: bull|bear|doji|na
      - pac_path_regime: flat|impulsive|trend|chop|unknown
      - pac_entry_immediate_rejection_flag
      - pac_entry_immediate_followthrough_flag

    Threshold keys (read only from ctx.features_auto_cfg["pac_bar"]):
      - impulse_threshold
      - rejection_body_ratio_max
      - start_offset_bars
      - mid_offset_bars
      - end_offset_bars
      - regime_half_window_bars
      - regime_flat_body_ratio_max
      - regime_impulsive_body_ratio_min
      - regime_trend_body_ratio_min
      - regime_trend_wick_ratio_max
    """
    if windows is None or windows.is_empty():
        log.warning("pac_bar: windows empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required_win = {"instrument", "anchor_tf", "anchor_ts"}
    missing = sorted(required_win - set(windows.columns))
    if missing:
        log.warning("pac_bar: windows missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    base = (
        windows.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("anchor_tf").cast(pl.Utf8),
            pl.col("anchor_ts").cast(pl.Datetime("us")),
        )
        .drop_nulls(["instrument", "anchor_tf", "anchor_ts"])
        .unique()
    )

    # Fast path: no candles -> placeholder frame
    if candles is None or candles.is_empty():
        return base.with_columns(
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_anchor"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_pre_entry"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_post_entry"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_start"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_mid"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_end"),
            pl.lit("unknown", dtype=pl.Utf8).alias("pac_path_regime"),
            pl.lit(False, dtype=pl.Boolean).alias("pac_entry_immediate_rejection_flag"),
            pl.lit(False, dtype=pl.Boolean).alias("pac_entry_immediate_followthrough_flag"),
        ).select(_empty_keyed_frame().columns)

    required_c = {"instrument", "tf", "ts", "open", "high", "low", "close"}
    missing_c = sorted(required_c - set(candles.columns))
    if missing_c:
        log.warning("pac_bar: candles missing columns=%s; using placeholders", missing_c)
        return base.with_columns(
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_anchor"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_pre_entry"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_post_entry"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_start"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_mid"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_end"),
            pl.lit("unknown", dtype=pl.Utf8).alias("pac_path_regime"),
            pl.lit(False, dtype=pl.Boolean).alias("pac_entry_immediate_rejection_flag"),
            pl.lit(False, dtype=pl.Boolean).alias("pac_entry_immediate_followthrough_flag"),
        ).select(_empty_keyed_frame().columns)

    # -----------------------------------------------------------------------
    # Thresholds (ONLY keys declared in registry threshold_keys)
    # -----------------------------------------------------------------------
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    fam_cfg = dict(auto_cfg.get("pac_bar", {}) if isinstance(auto_cfg, dict) else {})

    impulse_threshold = float(fam_cfg.get("impulse_threshold", 0.65))
    rejection_body_ratio_max = float(fam_cfg.get("rejection_body_ratio_max", 0.25))

    start_off = int(fam_cfg.get("start_offset_bars", -2))
    mid_off = int(fam_cfg.get("mid_offset_bars", -1))
    end_off = int(fam_cfg.get("end_offset_bars", +1))

    regime_half_window = int(fam_cfg.get("regime_half_window_bars", 5))
    regime_half_window = max(1, regime_half_window)

    flat_body_max = float(fam_cfg.get("regime_flat_body_ratio_max", 0.10))
    impulsive_body_min = float(fam_cfg.get("regime_impulsive_body_ratio_min", 0.25))
    trend_body_min = float(fam_cfg.get("regime_trend_body_ratio_min", 0.35))
    trend_wick_max = float(fam_cfg.get("regime_trend_wick_ratio_max", 0.35))

    win_size = 2 * regime_half_window + 1

    # -----------------------------------------------------------------------
    # Normalize candles: dedupe (instrument, tf, ts), restrict to anchor_tfs
    # -----------------------------------------------------------------------
    anchor_tfs = base.select("anchor_tf").unique().to_series().to_list()

    c = (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("tf").cast(pl.Utf8),
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
        )
        .drop_nulls(["instrument", "tf", "ts"])
        .sort(["instrument", "tf", "ts"])
        .unique(subset=["instrument", "tf", "ts"], keep="last")
        .filter(pl.col("tf").is_in(anchor_tfs))
        .with_columns(pl.col("tf").alias("anchor_tf"))
        .sort(["instrument", "anchor_tf", "ts"])
    )

    if c.is_empty():
        return base.with_columns(
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_anchor"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_pre_entry"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_post_entry"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_start"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_mid"),
            pl.lit("na", dtype=pl.Utf8).alias("pac_micro_state_end"),
            pl.lit("unknown", dtype=pl.Utf8).alias("pac_path_regime"),
            pl.lit(False, dtype=pl.Boolean).alias("pac_entry_immediate_rejection_flag"),
            pl.lit(False, dtype=pl.Boolean).alias("pac_entry_immediate_followthrough_flag"),
        ).select(_empty_keyed_frame().columns)

    # -----------------------------------------------------------------------
    # Feature construction on candle series
    # -----------------------------------------------------------------------
    body = (pl.col("close") - pl.col("open")).abs().alias("_body")
    rng = (pl.col("high") - pl.col("low")).alias("_range")

    dir_state = (
        pl.when((pl.col("close") - pl.col("open")) > 0).then(pl.lit("bull"))
        .when((pl.col("close") - pl.col("open")) < 0).then(pl.lit("bear"))
        .otherwise(pl.lit("doji"))
        .alias("_dir")
    )

    upper_wick = (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close"))).alias("_uw")
    lower_wick = (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")).alias("_lw")

    denom = pl.when(pl.col("_range") == 0).then(pl.lit(1.0)).otherwise(pl.col("_range"))
    wick_ratio = (pl.max_horizontal(pl.col("_uw"), pl.col("_lw")) / denom).alias("_wick_ratio")
    body_ratio = (pl.col("_body") / denom).alias("_body_ratio")

    rejection_flag = (
        (pl.col("_wick_ratio") >= pl.lit(impulse_threshold)) & (pl.col("_body_ratio") <= pl.lit(rejection_body_ratio_max))
    ).alias("_rejection")

    by = ["instrument", "anchor_tf"]

    c = c.with_columns([dir_state, body, rng, upper_wick, lower_wick]).with_columns([wick_ratio, body_ratio, rejection_flag])

    # Pre/post + start/mid/end via shifts; plus window means for regime
    c = c.with_columns(
        pl.col("_dir").shift(1).over(by).alias("_dir_prev"),
        pl.col("_dir").shift(-1).over(by).alias("_dir_next"),
        pl.col("_dir").shift(start_off).over(by).alias("_dir_start"),
        pl.col("_dir").shift(mid_off).over(by).alias("_dir_mid"),
        pl.col("_dir").shift(end_off).over(by).alias("_dir_end"),
        pl.col("_body_ratio").rolling_mean(window_size=win_size, min_periods=max(3, regime_half_window)).over(by).alias("_body_ratio_win"),
        pl.col("_wick_ratio").rolling_mean(window_size=win_size, min_periods=max(3, regime_half_window)).over(by).alias("_wick_ratio_win"),
    )

    # Regime classification
    pac_path_regime = (
        pl.when(pl.col("_body_ratio_win").is_null() | pl.col("_wick_ratio_win").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("_body_ratio_win") <= pl.lit(flat_body_max))
        .then(pl.lit("flat"))
        .when((pl.col("_wick_ratio_win") >= pl.lit(impulse_threshold)) & (pl.col("_body_ratio_win") >= pl.lit(impulsive_body_min)))
        .then(pl.lit("impulsive"))
        .when((pl.col("_body_ratio_win") >= pl.lit(trend_body_min)) & (pl.col("_wick_ratio_win") <= pl.lit(trend_wick_max)))
        .then(pl.lit("trend"))
        .otherwise(pl.lit("chop"))
        .alias("pac_path_regime")
    )

    # Join onto windows keys by exact anchor_ts
    j = base.join(
        c.select(
            "instrument",
            "anchor_tf",
            pl.col("ts").alias("anchor_ts"),
            pl.col("_dir").alias("pac_micro_state_anchor"),
            pl.col("_dir_prev").alias("pac_micro_state_pre_entry"),
            pl.col("_dir_next").alias("pac_micro_state_post_entry"),
            pl.col("_dir_start").alias("pac_micro_state_start"),
            pl.col("_dir_mid").alias("pac_micro_state_mid"),
            pl.col("_dir_end").alias("pac_micro_state_end"),
            pac_path_regime,
            pl.col("_rejection").alias("pac_entry_immediate_rejection_flag"),
        ),
        on=["instrument", "anchor_tf", "anchor_ts"],
        how="left",
    )

    # Followthrough: anchor direction == post direction and not doji
    j = j.with_columns(
        (
            (pl.col("pac_micro_state_anchor").is_in(["bull", "bear"]))
            & (pl.col("pac_micro_state_anchor") == pl.col("pac_micro_state_post_entry"))
        )
        .fill_null(False)
        .alias("pac_entry_immediate_followthrough_flag")
    )

    # Fill missing micro states deterministically
    j = j.with_columns(
        pl.col("pac_micro_state_anchor").fill_null("na"),
        pl.col("pac_micro_state_pre_entry").fill_null("na"),
        pl.col("pac_micro_state_post_entry").fill_null("na"),
        pl.col("pac_micro_state_start").fill_null("na"),
        pl.col("pac_micro_state_mid").fill_null("na"),
        pl.col("pac_micro_state_end").fill_null("na"),
        pl.col("pac_path_regime").fill_null("unknown"),
        pl.col("pac_entry_immediate_rejection_flag").fill_null(False).cast(pl.Boolean),
        pl.col("pac_entry_immediate_followthrough_flag").fill_null(False).cast(pl.Boolean),
    )

    return j.select(_empty_keyed_frame().columns)
