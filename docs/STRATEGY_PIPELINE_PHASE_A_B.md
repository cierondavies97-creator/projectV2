# Strategy Pipeline (Phase A–E)

## 0) Purpose
This document defines the **strategy pipelines** for both the deterministic engine lane and the research/training lane, with explicit handling of **Phase A through Phase E**. It is designed to ensure:

- Deterministic, reproducible artifacts in the engine lane.
- Auditability of Phase A–E labeling and event provenance.
- Clear separation between deterministic production logic and stochastic research workflows.

---

## System concept alignment (source of truth: README)

This document follows the repo-wide system concept:

- **Multi-paradigm by construction**; Phase A–E outputs are paradigm-neutral inputs.
- **Hard lane separation** between deterministic engine, research/training, and control-plane.
- **Deterministic engine lane** outputs are reproducible and audit-ready; configs are immutable in-run.
- **Stable identity model** ensures comparability across runs and paradigms.

If any text here conflicts with the README, the README is authoritative.

---

## 1) Shared assumptions and identity

All pipelines use the canonical identity model:

- `env`, `mode`, `snapshot_id`, `run_id`
- `trading_day` / `dt`, `cluster_id`, `instrument`
- `paradigm_id`, `principle_id` (when applicable)

All artifacts are partitioned by identity keys to preserve reproducibility and comparability.

---

## 2) Deterministic engine lane pipeline (Phase A → Phase E)

### 2.1 Deterministic pipeline order
The deterministic lane uses the canonical step chain:

1. **ingest**
2. **features**
3. **windows**
4. **hypotheses**
5. **critic**
6. **pretrade**
7. **gatekeeper**
8. **portfolio**
9. **brackets**
10. **reports**

*(Backtest-only: **fills**.)*

Phase A–E live inside **features → windows**, and must be deterministic and repeatable.

---

### 2.2 Phase A (deterministic range-seeding)
Phase A establishes **candidate range anchors** and structural events that seed Phase B.

**Inputs** (per instrument × anchor_tf × trading_day):
- Anchor-grid OHLCV summaries
- Deterministic swing/high-low detectors
- Optional precomputed microstructure aggregates (if deterministic)

**Outputs**:
- `data/market_events` (structural events)
- Phase A precursors in `data/windows` (initial range bounds and event evidence)

**Event stream (required)**:
- event types: `SC`, `AR`, `ST`, `UT`, `SPRING`, `SOS`, `LPS`, `BREAKOUT`, `BREAKDOWN`, `RECLAIM`, `FAIL`
- evidence fields: swing IDs, wick ratio, volume z-score, etc.

**Phase A responsibilities**:
- Detect preliminary bounds (`dr_low`, `dr_high`) from Phase A events.
- Emit deterministic events for auditability.
- Seed the range state machine for Phase B.

---

### 2.3 Phase B (deterministic range development)
Phase B is a **persistent label** and context state stored in `data/windows`.

**Inputs**:
- Phase A event stream
- Anchor-grid windows
- Deterministic range descriptors (mid, width, pd_index)

**Outputs (required in data/windows)**:
- `dr_id`, `dr_phase`, `dr_low`, `dr_high`, `dr_mid`, `dr_width`
- `dr_age_bars`, `dr_start_ts`, `dr_last_update_ts`
- `pd_index`, `range_position`
- counters: `test_high_count`, `test_low_count`, `liq_eqh_count`, `liq_eql_count`
- `dr_reason_code`, optional `dr_score_*`

**Transition rules (deterministic)**:
- **A → B**: bounds confirmed + minimum width + time-in-range + boundary test.
- **Stay B**: majority inside bounds + failed breakouts.
- **B → C/D/E**: spring/upthrust or accepted breakouts (Phase C/D/E) handled by the state machine.
- **Reset**: invalid range resets Phase labels but retains historical events.

**Downstream consumption**:
- `hypotheses`, `critic`, `gatekeeper` **must consume** `dr_*` fields and **must not recompute** Phase B.

---

### 2.4 Phase C–E (deterministic transitions)
Phase C–E represent probe → acceptance → trend regimes and are written to `data/windows` as deterministic state transitions.

**Phase C (probe + reclaim)**:
- Candidate when `probe_low` or `probe_high` fires.
- Confirmed when reclaim happens within `reclaim_bars_max`.
- Failed if acceptance (Phase D) triggers before reclaim.

**Phase D (acceptance + retest)**:
- Enter when `accept_up` or `accept_dn` persists for `accept_bars_min`.
- Retest must hold; failure triggers `FAILED_BREAK` with policy-specific behavior.

**Phase E (trend regime)**:
- Enter when distance-from-mid persists (`trend_bars_min`) and no re-entry occurs.

All Phase C–E outputs are written with reason codes and evidence fields (probe side, pierce distance, reclaim margin, acceptance distance/persistence, retest pass/fail, trend distance/persistence).

---

### 2.5 Deterministic strategy hooks (Phase A–E aware)
Strategy hypotheses are conditioned on Phase context:

- **Phase A contexts**: discovery or structural setups, proto-range detection.
- **Phase B contexts**: mean-reversion, liquidity sweep reversion, midline magnet trades.
- **Phase C contexts**: sweep-reversion confirmation or trap detection.
- **Phase D contexts**: breakout continuation with acceptance/retest filters.
- **Phase E contexts**: trend continuation and regime persistence logic.

**Rule**: the engine lane only routes based on `dr_phase` and related windows fields. It does not embed paradigm-specific logic in the pipeline.

---

### 2.6 Deterministic artifact locations (entries/exits/context)
Deterministic artifacts store **entry/exit timestamps, prices, and context** in specific, normalized places:

**Hypotheses (candidate intent)**:
- **Where:** `artifacts/engine/hypotheses/`
- **Required entry context:** `entry_ts`, `entry_px` (or enough deterministic fields to derive them), `anchor_tf`, `event_ts`, `window_id`.
- **Context payload:** `params_json` (or equivalent structured column) for paradigm-specific context; this must be deterministic and serializable.

**Critic (scoring and rejection)**:
- **Where:** `artifacts/engine/critic/`
- **Required fields:** `critic_score_total`, component scores, `reject_reason_primary`, and `reject_reasons_all` tied to the hypothesis identity.

**Pretrade / orders (actionable intent)**:
- **Where:** `artifacts/engine/orders/` and `artifacts/engine/decisions/`
- **Required fields:** normalized `entry_ts`, `entry_px`, `sl_px`, `tp_px`, and order metadata.

**Gatekeeper (eligibility + context selection)**:
- **Where:** `artifacts/engine/decisions/`
- **Required fields:** eligibility flags, gating reasons, and context selection metadata linked to `hypothesis_id`.

**Trade paths (canonical lifecycle)**:
- **Where:** `artifacts/engine/trade_paths/`
- **Required fields:** `entry_ts`, `entry_px`, `exit_ts`, `exit_px`, and bracket fields `sl_px`, `tp_px`, plus context keys (`paradigm_id`, `principle_id`, `window_id`, `anchor_tf`).
- **Rule:** `trade_paths` is the canonical truth for lifecycle outcomes; do not copy these fields into ad-hoc tables.

**Fills (backtest-only)**:
- **Where:** `artifacts/engine/fills/`
- **Required fields:** realized execution timestamps and prices, slippage/cost estimates, and linkage to `trade_id`.

---

### 2.7 MarketState cube (canonical context)
Strategies consume a canonical MarketState cube keyed by `(instrument, anchor_tf, anchor_ts)`:

- Phase fields + evidence (`dr_*`, probe/reclaim/accept/retest/trend reason codes).
- Microstructure fields (AggImb/OFI/intensity).
- Jump/vol fields (RV/BV/JV + semivariance).
- Impact fields (Kyle λ + regime).
- Options context fields (ATM IV, skew, term slope, VRP, model-free implied variance if feasible).

All MarketState outputs must carry policy identifiers (`phase_version`, `threshold_bundle_id`, `micro_policy_id`, `jump_policy_id`, `impact_policy_id`, `options_policy_id`) for reproducibility.

---

## 3) Research/training lane pipeline (Phase A → Phase E)

The research lane consumes engine artifacts and trains/validates strategy logic without mutating the deterministic pipeline.

### 3.1 Research pipeline order
1. **Collect training datasets**
   - Join engine artifacts (`windows`, `market_events`, `hypotheses`, `critic`, `trade_paths`).
2. **Feature evaluation**
   - Rank Phase A–E features and thresholds for stability.
3. **Candidate generation**
   - GA/RFE generation of hypothesis templates conditioned on Phase A–E states.
4. **Parameter learning**
   - Bayes calibration of thresholds and critic weights.
5. **Portfolio evaluation**
   - Cross-sectional evaluation with risk/correlation constraints.
6. **Promotion**
   - Candidate bundle becomes new `candidate_id` and (if approved) new `snapshot_id`.

---

### 3.2 Phase A research responsibilities
- Validate Phase A event detection stability across instruments/timeframes.
- Learn threshold ranges (e.g., swing depth, volume z-score) for robust Phase A events.
- Stress test detection against regime shifts and volatility changes.

---

### 3.3 Phase B–E research responsibilities
- Evaluate Phase B–E label stability and drift risk across regimes.
- Train paradigm-specific hypothesis filters using `dr_phase`, `pd_index`, and counters.
- Optimize range acceptance/rejection thresholds for Phase C–E transitions.
- Validate microstructure/jump/impact/option primitives against realized outcomes.

---

### 3.4 Research outputs (required)
- **Posterior summaries** with dataset hashes.
- **Candidate configs** with explicit thresholds and provenance.
- **Evaluation reports** with regime splits and failure-mode clustering.

---

## 4) Strategy lifecycle: create → train → promote

This section ties together how a strategy is **created**, **trained**, and **promoted**, and how each lane uses config/data YAMLs.

### 4.1 Strategy creation (design + config)
1. **Define paradigm + principle scope**
   - Assign `paradigm_id` and optional `principle_id` for the strategy family.
2. **Create paradigm config bundle (YAML)**
   - **Where:** `conf/paradigms/<paradigm_id>.yaml`
   - **Contents:** universe, timeframes, feature toggles, hypothesis rules, critic weights, gatekeeper constraints, portfolio limits.
3. **Register data dependencies**
   - **Where:** `conf/` and `conf/features_registry.yaml`
   - **Rule:** deterministic lane reads configs only; never mutate during runs.

### 4.2 Training (research lane)
1. **Ingest deterministic artifacts**
   - Join `data/windows` + `data/market_events` + `artifacts/engine/hypotheses` + `trade_paths`.
2. **Learning workflows**
   - RFE/GA/Bayes to optimize thresholds, feature subsets, and critic weights.
3. **Write research outputs**
   - **Where:** `artifacts/research/` (posterior summaries, candidates, evaluation reports).
   - Each output is tied to `experiment_id` and `candidate_id`, with dataset hashes.

### 4.3 Promotion (research → deterministic)
1. **Candidate bundle lock-in**
   - Export a **candidate config** (YAML) with explicit thresholds, feature lists, and metadata.
2. **Snapshot config promotion**
   - Create a new `snapshot_id` that **pins** the candidate config + data snapshot.
3. **Deterministic run**
   - Engine uses the new snapshot/config bundle with no in-run mutation.

---

## 5) Deterministic vs research separation (enforced)

- **Deterministic lane**: pure, replayable, no learning.
- **Research lane**: stochastic training allowed, but must be reproducible via explicit seeds.
- **Promotion**: only explicit, versioned snapshot updates may alter deterministic configs.

---

## 6) Compliance checklist

A Phase A–E strategy pipeline is compliant only if:

- Phase A events and Phase B–E labels are deterministic and reproducible.
- Event stream can explain every Phase transition.
- `data/windows` contains complete `dr_*` fields (non-null where required).
- Downstream steps consume `dr_*` fields and do not recompute them.
- Research outputs are versioned, reproducible, and promote via explicit snapshots only.
