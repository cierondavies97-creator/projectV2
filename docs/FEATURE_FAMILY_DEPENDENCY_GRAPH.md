# Feature family dependency graph + templates

This document is implementation-grade reference material for:

1. **Dependency graph** (family → required inputs → outputs → consumers).
2. **Stub templates** for adding new feature families (module + registry + tests).
3. **Phase B–E dealing-range interfaces** (events/features/state machine).

It is grounded in the current repo contracts (features registry, feature builders, and the dealing-range modules). See the cited source files for authoritative interfaces and expected columns.

---

## System concept alignment (source of truth: README)

All guidance in this document is consistent with the repository’s core system concept:

- **Multi-paradigm by construction**; no paradigm is privileged in the pipeline.
- **Hard lane separation** between deterministic engine, research/training, and control-plane.
- **Deterministic engine lane**: reproducible, audit-ready outputs; no in-run config mutation.
- **Stable identity model** and schema contracts so artifacts remain comparable across runs.

If any future update to this document conflicts with the README, the README takes precedence.

---

## 0) What this doc adds (proposed families + schema/threshold expansions)

This document is **documentation-only**. It does **not** add runtime code or enable new families by itself. It does, however, propose and catalog new families and schema/threshold expansions to make the pipeline deterministic, auditable, and multi-paradigm.

**New families introduced or expanded here (all disabled by default):**

- **`unsup_regime`** (stub, `data/windows`): Adds unsupervised regime labels with provenance and diagnostics; thresholds cover standardization, imputations, clustering knobs, and confidence calibration.
- **`carry_ts`** (stub, `data/windows`): Adds carry/term-structure proxy outputs plus provenance and data quality flags; thresholds include proxy method/normalization.
- **`ta_memory`** (stub, `data/zones_state`): Adds zone-scoped TA memory features (RSI/EMA interactions) and reversal/breakout rates; thresholds include decay and event-count guards.
- **`corr_htf`** (stub, `data/features_corr`): Adds slow correlation summaries and cluster diagnostics; thresholds include HTF windows and clustering cadence.
- **`xs_relval`** (stub, `data/features_corr`): Adds cross-sectional RV spread diagnostics and peer selection logic; thresholds include peer selection, spread type, and coint gates.
- **`macro_regime`** (stub, `data/macro`): Adds macro regime labels with provenance; thresholds include method selection and confidence policy.
- **`dealing_range_state`** (experimental, `data/windows`): Formalizes B–E dealing-range state outputs (Phase labels + evidence) with explicit threshold keys.
- **`microstructure_flow`** (experimental, `data/features`): Adds OFI/imbalance/intensity summary per anchor window.
- **`jump_variation`** (experimental, `data/features`): Adds RV/BV/JV and semivariances with configurable base TF and windows.
- **`impact_lambda`** (experimental, `data/features`): Adds Kyle’s λ impact estimates and regime labels.
- **`options_context`** (experimental, `data/windows`): Adds IV/Greeks, skew, term slope, VRP, and model-free IV summary outputs.

If you want to implement any of these, the registry entries define the **authoritative** column contracts and threshold keys; code should conform to those schemas.

---

## 1) Dependency graph (family → required inputs → outputs → consumers)

**Legend**
- **Inputs**: minimum columns read by the family builder (plus context requirements like `ctx.cluster.anchor_tfs`).
- **Outputs**: physical table + keys (the registry defines the full column list).
- **Consumers**: deterministic engine steps or downstream artifacts that typically read the table.

### 1.1 Feature families producing `data/features`

| Family | Required inputs | Outputs (table + keys) | Primary consumers |
| --- | --- | --- | --- |
| `ict_struct` | `candles` with `instrument, tf, ts, open, high, low, close`; `ctx.cluster.anchor_tfs` or fallback to `candles.tf` | `data/features` keyed by `(instrument, anchor_tf, ts)` | `hypotheses` and `critic` (paradigm logic); any gating on structure/liq features. | 
| `ta_trend` | `candles` with `instrument, tf, ts, open, high, low, close` | `data/features` keyed by `(instrument, anchor_tf, ts)` | `hypotheses`, `critic`, and regime gating. |
| `ta_vol` | `candles` with `instrument, tf, ts, open, high, low, close` | `data/features` keyed by `(instrument, anchor_tf, ts)` | `hypotheses`, `critic`, context selection. |
| `stat_ts` | `candles` with `instrument, tf, ts, open, high, low, close` | `data/features` keyed by `(instrument, anchor_tf, ts)` | `hypotheses` (stat-TS rules), `critic`. |
| `jump_variation` | `candles` with `instrument, tf, ts, close` | `data/features` keyed by `(instrument, anchor_tf, ts)` | `hypotheses` (jump-aware logic), `critic`, regime tagging. |
| `impact_lambda` | `candles` with `instrument, tf, ts, close, volume` | `data/features` keyed by `(instrument, anchor_tf, ts)` | `critic` (cost/impact gating), `hypotheses` if impact-aware. |
| `microstructure_flow` | `candles` with `instrument, tf, ts` (anchors) + `ticks` with `instrument, ts, price, size, side` if available | `data/features` keyed by `(instrument, anchor_tf, ts)` | `hypotheses` (micro confirmation), `critic` and context gating. |
| `xs_relval` | `candles` with `instrument, ts, close` | `data/features_corr` keyed by `(instrument, ts)` | `hypotheses` (cross-sectional RV), `research` (peer diagnostics). |
| `corr_micro` | `candles` with `instrument, ts, close` | `data/features_corr` keyed by `(instrument, ts)` | `hypotheses` (pairing/correlation filters), `research`. |
| `corr_htf` | `candles` with `instrument, ts, close` | `data/features_corr` keyed by `(instrument, ts)` | `hypotheses`, `research` clustering. |

### 1.2 Feature families producing `data/windows`

| Family | Required inputs | Outputs (table + keys) | Primary consumers |
| --- | --- | --- | --- |
| `dealing_range_state` | `candles` with `instrument, tf, ts, high, low, close`; `ctx.cluster.anchor_tfs` | `data/windows` keyed by `(instrument, anchor_tf, anchor_ts)` | `hypotheses` (range-context routing), `critic` (range regime checks). |
| `options_context` | `candles` with `instrument, tf, ts` (as anchor clock) + options chain inputs (dataset-specific) | `data/windows` keyed by `(instrument, anchor_tf, anchor_ts)` | `hypotheses` (vol/term/VRP gates), `critic`. |
| `carry_ts` | `candles` with `instrument, tf, ts, close`; `ctx.cluster.anchor_tfs` | `data/windows` keyed by `(instrument, anchor_tf, anchor_ts)` | `hypotheses` (carry/term regime), `critic`. |
| `consolidation_bar` | `windows` with `instrument, anchor_tf, anchor_ts` + `candles` with `instrument, tf, ts, high, low, close` | `data/windows` keyed by `(instrument, anchor_tf, anchor_ts)` | `hypotheses` (range/consolidation filters), `critic`. |
| `pac_bar` | `windows` with `instrument, anchor_tf, anchor_ts` + `candles` with `instrument, tf, ts, open, high, low, close` | `data/windows` keyed by `(instrument, anchor_tf, anchor_ts)` | `hypotheses` (pattern context), `critic`. |
| `vol_range` | `candles` with `instrument, tf, ts, high, low`; `ctx.cluster.anchor_tfs` | `data/windows` keyed by `(instrument, anchor_tf, anchor_ts)` | `hypotheses`, `critic` (range/vol context). |
| `unsup_regime` | `candles` with `instrument, tf, ts, close` | `data/windows` keyed by `(instrument, anchor_tf, anchor_ts)` | `hypotheses`, `critic`, regime selection. |

### 1.3 Feature families producing other tables

| Family | Required inputs | Outputs (table + keys) | Primary consumers |
| --- | --- | --- | --- |
| `macro_calendar` | `macro` with required calendar columns | `data/macro` keyed by `(ts)` | `windows` joins; `hypotheses` and `critic` gating on event windows. |
| `macro_regime` | `macro` with `ts` + regime inputs (rules/unsup/supervised) | `data/macro` keyed by `(ts)` | `hypotheses` and `critic` (macro regime gating). |
| `trade_path_class` | `trade_paths` with `trade_id` (+ optional path diagnostics) | `data/trade_paths` keyed by `(trade_id)` | `reports`, `research`, and trade-path filters. |
| `pcra_bar` | `candles` with `instrument, tf, ts, open, high, low, close` | `data/pcra` keyed by `(instrument, anchor_tf, pcr_window_ts)` | `hypotheses` (price/volume ratio context) and `research`. |
| `pcra_tick` | `ticks` with `instrument, ts, price` | `data/pcra` keyed by `(instrument, anchor_tf, pcr_window_ts)` | `hypotheses` and `research` (micro P/C ratio). |
| `vp_core` | `candles` with `instrument, tf, ts, high, low, close` | `data/zones_state` keyed by `(instrument, anchor_tf, zone_id, ts)` | `hypotheses` and zone-aware gating. |
| `zmf_core` | `candles` with `instrument, tf, ts, close` | `data/features` keyed by `(instrument, anchor_tf, ts)` | `hypotheses` (zero-mean flow features), `critic`. |

**Notes**
- When a family writes `data/windows`, it is **paradigm-neutral** and intended for gating and routing by any paradigm (no re-computation in hypotheses). This aligns with the deterministic-lane contract described in the README. (See `dealing_range_state` and `options_context` behaviors.)
- `data/features` is the general-purpose feature table consumed by hypotheses and critic hooks across paradigms.

---

## 2) Stub templates for new family modules

These templates are designed to be deterministic, registry-aligned, and easy to audit. They mirror the existing `build_feature_frame` pattern and enforce strict key contracts.

### 2.1 Python module stub (`src/engine/features/<family_id>.py`)

```python
from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from engine.features import FeatureBuildContext
from engine.features._shared import conform_to_registry

log = logging.getLogger(__name__)


def _empty_keyed_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument": pl.Series([], dtype=pl.Utf8),
            "anchor_tf": pl.Series([], dtype=pl.Utf8),
            "ts": pl.Series([], dtype=pl.Datetime("us")),
            # TODO: add family feature columns
        }
    )


def _merge_cfg(ctx: FeatureBuildContext, family_cfg: Mapping[str, object] | None) -> dict[str, object]:
    cfg: dict[str, object] = {}
    auto_cfg = getattr(ctx, "features_auto_cfg", None) or {}
    if isinstance(auto_cfg, Mapping):
        cfg.update(auto_cfg.get("<family_id>", {}) or {})
    if isinstance(family_cfg, Mapping):
        cfg.update(family_cfg)
    return cfg


def build_feature_frame(
    *,
    ctx: FeatureBuildContext,
    candles: pl.DataFrame | None = None,
    windows: pl.DataFrame | None = None,
    ticks: pl.DataFrame | None = None,
    family_cfg: Mapping[str, object] | None = None,
    registry_entry: Mapping[str, object] | None = None,
    **_,
) -> pl.DataFrame:
    """
    Table: data/features  # or data/windows, data/macro, data/pcra, data/trade_paths
    Keys : instrument, anchor_tf, ts  # adjust to the table
    """
    if candles is None or candles.is_empty():
        log.warning("<family_id>: candles empty; returning empty keyed frame")
        return _empty_keyed_frame()

    required = {"instrument", "tf", "ts", "close"}  # TODO: update
    missing = sorted(required - set(candles.columns))
    if missing:
        log.warning("<family_id>: missing columns=%s; returning empty keyed frame", missing)
        return _empty_keyed_frame()

    cfg = _merge_cfg(ctx, family_cfg)

    # TODO: compute features deterministically
    out = (
        candles.select(
            pl.col("instrument").cast(pl.Utf8),
            pl.col("tf").cast(pl.Utf8).alias("anchor_tf"),
            pl.col("ts").cast(pl.Datetime("us")),
            # feature expressions ...
        )
        .drop_nulls(["instrument", "anchor_tf", "ts"])
        .sort(["instrument", "anchor_tf", "ts"])
    )

    out = conform_to_registry(
        out,
        registry_entry=registry_entry,
        key_cols=["instrument", "anchor_tf", "ts"],
        where="<family_id>",
        allow_extra=False,
    )

    log.info("<family_id>: built rows=%d", out.height)
    return out
```

### 2.2 Registry stub (`conf/features_registry.yaml`)

```yaml
<family_id>:
  table: "data/features"  # or data/windows / data/macro / data/pcra / data/trade_paths
  enabled: false
  maturity: "stub"
  description: "Short, deterministic description of the family."
  primary_key: [instrument, anchor_tf, ts]
  threshold_keys:
    - <threshold_key_1>
    - <threshold_key_2>
  columns:
    instrument: { role: key, dtype: string }
    anchor_tf:  { role: key, dtype: string }
    ts:         { role: key, dtype: timestamp }
    <feature_col_1>: { role: feature, dtype: double }
    <feature_col_2>: { role: feature, dtype: string }
```

### 2.3 Tests stub (`tests/engine/features/test_<family_id>.py`)

```python
import polars as pl

from engine.features.<family_id> import build_feature_frame


def test_<family_id>_empty_returns_keyed_frame():
    out = build_feature_frame(ctx=None, candles=pl.DataFrame())
    assert out.is_empty()


def test_<family_id>_schema_alignment():
    # TODO: build minimal candles/ticks/windows fixture
    pass
```

---

## 3) Phase B–E state machine interfaces (events.py / features.py / state_machine.py)

This section maps **inputs**, **outputs**, and **contracts** for the Phase B–E dealing-range implementation in:

- `src/engine/features/market_structure/dealing_range/features.py`
- `src/engine/features/market_structure/dealing_range/events.py`
- `src/engine/features/market_structure/dealing_range/state_machine.py`

### 3.1 `features.py::compute_range_descriptors`

**Purpose**: Compute deterministic range geometry and volatility scaling used by later events/state transitions.

**Signature**
- `compute_range_descriptors(df: pl.DataFrame, *, cfg: Mapping[str, float]) -> pl.DataFrame`

**Required input columns**
- `ts`, `high`, `low`, `close`

**Outputs added**
- `atr_tr`: true-range proxy
- `atr`: rolling ATR (window = `atr_window`)
- `dr_high`: rolling max(high) over `lookback_bars`
- `dr_low`: rolling min(low) over `lookback_bars`
- `dr_width`: `dr_high - dr_low`
- `dr_mid`: `(dr_high + dr_low) / 2`
- `dr_width_atr`: `dr_width / atr`

**Config keys used**
- `lookback_bars` (default 50)
- `atr_window` (default 14)

**Determinism contract**
- Rolling calculations are deterministic; `min_periods` equals window size to avoid partial-window leakage.

### 3.2 `events.py::compute_event_flags`

**Purpose**: Derive range evidence flags from range descriptors and ATR.

**Signature**
- `compute_event_flags(df: pl.DataFrame, *, cfg: Mapping[str, float]) -> pl.DataFrame`

**Required input columns**
- `high`, `low`, `close`, `dr_low`, `dr_high`, `dr_mid`, `dr_width`, `atr`

**Outputs added** (underscored to denote internal evidence)
- `_test_band`, `_probe_min`, `_reclaim_band`, `_accept_dist` (ATR-scaled distances)
- `_inside` (close inside range)
- `_inside_ratio` (rolling mean of `_inside` over `lookback_bars`)
- `_test_high`, `_test_low`, `_tests_count` (boundary tests)
- `_pierce_low`, `_pierce_high` (distance beyond bounds)
- `_probe_low`, `_probe_high` (pierce ≥ `_probe_min`)
- `_reclaim_from_low`, `_reclaim_from_high` (reclaim thresholds)
- `_outside_up`, `_outside_dn` (acceptance distance flags)
- `_dist_mid`, `_trend_dist`, `_trend_far` (trend-distance regime)

**Config keys used**
- `lookback_bars`
- `test_atr_mult`
- `probe_atr_mult`
- `reclaim_atr_mult`
- `accept_atr_mult`
- `trend_atr_mult`
- `trend_width_mult`

### 3.3 `state_machine.py` state transitions

**Primary entrypoint**
- `fold_state_machine(windows, candles, *, cfg=None) -> pl.DataFrame`

**Input contracts**
- `windows`: must include `instrument, anchor_tf, anchor_ts` (used to seed the output grid).
- `candles`: must include `instrument, ts, high, low, close`; `tf` is mapped to `anchor_tf`.

**Outputs**
- Keys: `(instrument, anchor_tf, anchor_ts)`
- Core columns:
  - `dr_id`, `dr_phase`, `dr_low`, `dr_high`, `dr_mid`, `dr_width`, `dr_width_atr`
  - `dr_age_bars`, `dr_start_ts`, `dr_last_update_ts`, `dr_reason_code`
- Policy/version tags:
  - `phase_version`, `threshold_bundle_id`, `micro_policy_id`, `jump_policy_id`, `impact_policy_id`, `options_policy_id`

**State machine flow (B–E)**
- **B (range development)**: enter only if `dr_width_atr >= width_min_atr`, `_inside_ratio >= p_inside_min`, and `_tests_count >= tests_min`.
- **C (probe)**: from B, when `_probe_low` or `_probe_high` true. Probe side recorded.
- **D (acceptance)**: if `outside_up_run >= accept_bars_min` or `outside_dn_run >= accept_bars_min`.
- **E (trend)**: from D when `trend_run >= trend_bars_min`.
- **Re-entry**: from D or E, if `_inside` true (re-enter B).
- **Reset**: if width becomes invalid while in B (`dr_width_atr < width_min_atr`).

**Reason codes emitted**
- `ENTER_B_RANGE_VALID`
- `STAY_B_AUCTION_IN_RANGE`
- `ENTER_C_PROBE`
- `EXIT_C_ACCEPTED_BREAK`
- `EXIT_C_RECLAIMED`
- `EXIT_C_TIMEOUT`
- `EXIT_B_ACCEPTED_BREAK`
- `STAY_C_PROBE`
- `STAY_D_ACCEPTED`
- `EXIT_D_TREND_ESTABLISHED`
- `REENTER_B_FROM_D`
- `REENTER_B_FROM_E`
- `RESET_RANGE_INVALID`
- `STAY_E_TREND`

### 3.4 Contract mismatch to reconcile (important)

There is a **naming mismatch** between the dealing-range registry keys and the current state-machine config keys:

- Registry expects `dealing_range_lookback_bars`, `width_min_atr_mult`, `test_band_atr_mult`, etc.
- The state-machine config currently reads `lookback_bars`, `width_min_atr`, `test_atr_mult`, etc.

This should be resolved by either:
1. **Mapping registry keys → state machine keys** in `dealing_range_state._merge_cfg`, or
2. **Renaming state machine keys** to match the registry (preferred for clarity and future reproducibility).

Until this is aligned, the registry may show thresholds that are not actually consumed by the dealing-range state machine.
