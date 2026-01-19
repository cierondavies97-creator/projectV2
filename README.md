# projectV2

Conceptual definition (engine-grade)

Phase B = “range development / cause-building.”

Operationally, in the engine, Phase B means:

A candidate dealing range exists (bounds and mid are defined from Phase A events).

Price is auctioning inside those bounds with repeated tests, absorbing liquidity, and producing mostly mean-reverting behavior.

No definitive evidence yet of:

Phase C (spring/upthrust and reclaim/lose behavior), or

Phase D (sign of strength / last point of support with directional follow-through), or

Phase E (trend outside the range).

In code: Phase B is a persistent label on each anchor window row (anchor_ts, anchor_tf, instrument) plus a set of quantitative range descriptors and event counters that justify the label.

Data contract: what must be written and where
1) Windows table is the source of truth for context

In data/windows/..., every anchor row must include:

Identity:

instrument, anchor_tf, anchor_ts, trading_day, cluster_id (as you already do)

Dealing range core fields:

dr_id (stable identifier)

dr_phase ∈ {A, B, C, D, E, NONE}

dr_low, dr_high, dr_mid, dr_width

dr_age_bars (bars since dr_start_ts)

dr_start_ts, dr_last_update_ts

Diagnostics / explainability:

dr_reason_code (string enum, e.g. ENTER_B_AFTER_AR, STAY_B_WITHIN_BOUNDS, EXIT_B_SPRING_CONFIRMED)

dr_score_* components (optional but recommended): dr_score_range_valid, dr_score_tests, dr_score_breakout_fail, etc.

Trading-relevant context features:

pd_index (premium/discount index relative to the current range)

range_position (normalized price position: (close - dr_low)/dr_width clipped 0..1)

liq_eqh_count, liq_eql_count (counts of equal highs/lows detections)

test_high_count, test_low_count (counts of boundary tests)

Rule: downstream steps (hypotheses/critic/gatekeeper) consume only these fields as the canonical phase context. They do not recompute Phase B.

2) Event table for auditability

Create data/market_events/... (or data/dealing_range_events/...) with one row per detected structural event:

instrument, anchor_tf, ts

event_type ∈ {SC, AR, ST, UT, SPRING, SOS, LPS, BREAKOUT, BREAKDOWN, RECLAIM, FAIL}

event_strength (float)

ref_level (price)

dr_id (the range it belongs to, if known)

evidence fields (swing ids, volume spike zscore, wick ratio, etc.)

Rule: Phase transitions must be explainable as functions of this event stream plus window-local OHLCV summaries.

This is how you keep determinism and enable future research without rewriting your pipeline.

Architecture: where this logic lives
Deterministic lane modules

Create a dedicated package, not tied to ICT:

engine/market_structure/dealing_range/

events.py (event detectors, pure functions)

state_machine.py (phase transition logic, pure functions)

features.py (range descriptors, pd_index, counts)

schemas.py (polars schema + validation rules)

config.py (threshold keys + defaults)

tests/ (golden fixtures per tf)

Pipeline integration

Phase B computation belongs in the canonical ordered steps:

features → windows: produce events + state machine output and write into data/windows

hypotheses: read dr_phase and select hypothesis families accordingly

Do not compute Phase B inside hypotheses; that will poison reproducibility and cross-paradigm comparisons.

Deterministic state machine: exact behavior you want
State representation

Maintain an explicit state struct per (instrument, anchor_tf):

dr_id

phase

low, high, mid, width

start_ts, last_update_ts

counters: test_high_count, test_low_count, eqh_count, eql_count, failed_break_count

last events: last_sc_ts, last_ar_ts, etc.

Transition rules (engine-usable, not “trader poetry”)
A → B (enter Phase B)

Enter Phase B when:

Range is “formed enough”:

you have confirmed provisional bounds: dr_low and dr_high

dr_width meets minimum viability (e.g., dr_width >= width_min_atr * atr_anchor OR >= width_min_ticks)

AND price has spent at least min_range_bars inside [dr_low, dr_high]

AND you have at least one boundary test (high or low) after the AR.

Reason code: ENTER_B_RANGE_VALID

Stay in B (Phase B persistence)

Stay in Phase B while:

Price is predominantly inside bounds:

close inside bounds for p_inside_min fraction over last lookback_bars

Breaks beyond bounds are not “accepted”:

a break is considered “not accepted” if it returns inside within break_fail_bars and/or volume/structure does not confirm continuation

Reason code: STAY_B_AUCTION_IN_RANGE

B → C (spring / upthrust candidate)

Transition to Phase C when you detect a liquidity run beyond a bound with a reversal signature:

Spring (bullish): low pierces dr_low by spring_penetration and then reclaims within reclaim_bars_max

Upthrust (bearish): high pierces dr_high similarly and reclaims downward

Reason code: EXIT_B_SPRING_CANDIDATE or EXIT_B_UT_CANDIDATE

B → D (directional confirmation from inside range)

Transition to Phase D when you detect acceptance + follow-through in the intended direction:

Break beyond bound and hold:

close outside for accept_bars_min

retest doesn’t lose the broken level (LPS/LPSY style)

Optional reinforcement: aligned swing structure (BOS) in same direction

Reason code: EXIT_B_ACCEPTED_BREAK

D → E (trend regime)

Transition to E when price is clearly trending away from range:

distance from dr_mid exceeds trend_distance_atr for trend_bars_min

and range is no longer relevant (optional: mark dr as “completed”)

Reason code: EXIT_D_TREND_ESTABLISHED

Reset / NONE

If the range becomes invalid (width collapses, bounds are overwritten repeatedly, or your detector loses confidence), reset:

dr_phase = NONE, dr_id = null, clear counters

but keep the event table—never delete history

Reason code: RESET_RANGE_INVALID

Computing Phase B deterministically (how to implement)
1) Inputs

Per microbatch, for each (instrument, anchor_tf, trading_day) you should load:

anchor-grid windows (canonical anchor_ts rows)

OHLCV summaries at that anchor_tf (or derived from M1 aggregation)

optional: volume profile / microstructure aggregates (but keep them deterministic and precomputed if expensive)

2) Event detection (pure functions)

Event detectors must be:

stateless (operate on a time-slice + lookback)

fully parameterized via threshold_keys

deterministic in ties (explicit sorting by ts, then by stable ids)

Examples of detectors you need:

swing highs/lows (for tests, BOS, EQH/EQL)

boundary tests:

test-high: high >= dr_high - test_band

test-low: low <= dr_low + test_band

liquidity sweep:

pierce + reclaim pattern with wick/close constraints

Write these events out, always.

3) State update per anchor row

For each anchor row in time order:

update range descriptors (bounds, mid, width)

update counters (tests, eqh/eql, failed breaks)

compute acceptance/rejection metrics

apply the transition function next_state = step_state(prev_state, evidence_row, config)

This should run as a deterministic fold (scan) over the anchor timeline.

4) Output join back into data/windows

Join the resulting per-anchor outputs onto your windows frame by keys:
instrument, anchor_tf, anchor_ts

Write with strict schema and explicit nullability rules.

How Phase B should affect hypotheses (without hard-wiring a paradigm)

Phase B is a context selector. The hypotheses layer should:

route to range-compatible families when dr_phase == "B"

mean reversion / fade extremes

liquidity-sweep-reversion

midline magnet / PD index reversion

suppress or downweight:

breakout continuation

trend-follow entries that assume acceptance

This is not “ICT-only.” Any paradigm can plug in, because the interface is just: dr_phase, pd_index, range_position, and a few counters.

Configuration: threshold keys (stable and future-proof)

Put these in a shared market-structure config namespace (not ICT):

dealing_range_lookback_bars

min_range_bars

width_min_ticks and/or width_min_atr_mult

test_band_ticks (or ATR-scaled band)

p_inside_min

break_fail_bars

accept_bars_min

spring_penetration_ticks (or ATR mult)

reclaim_bars_max

trend_distance_atr

trend_bars_min

eq_level_ticks_max (for eqh/eql)

Design rule: keys must be usable across instruments and timeframes; allow both tick and ATR scaling, but make the rule explicit (prefer ATR scaling; tick fallback).

Determinism, observability, and reproducibility requirements

You should enforce these as non-negotiable:

Single writer of dr_phase: only the dealing-range module writes it, only the windows table stores it.

Event provenance: every transition has dr_reason_code and can be traced back to a small set of events in data/market_events.

Strict schema & validation:

Phase B implies dr_id, dr_low/high/mid/width, pd_index are non-null.

If nulls exist, fail early in deterministic lane (do not silently pass).

Golden tests:

store fixture OHLCV segments + expected phase labels per anchor_tf

run in CI; phase labeling must not drift unintentionally.

Implementation skeleton (what you actually build)

dealing_range/events.py

detect_swings(df, cfg) -> events_df

detect_tests(df, dr_state, cfg) -> events_df

detect_springs(df, dr_state, cfg) -> events_df

dealing_range/state_machine.py

step_state(prev: DRState, row: EvidenceRow, cfg) -> DRState

dealing_range/features.py

compute_range_descriptors(df, cfg) -> descriptors_df

compute_pd_index(close, low, high) -> float

microbatch/steps/windows_step.py (or your equivalent)

load inputs

build events

fold state machine across anchors

write data/windows + data/market_events

This is the clean separation you want: deterministic state machine in one place, stored once, consumed everywhere.

If you implement Phase B exactly this way, you get:

stable, testable labels for research conditioning,

one canonical truth across paradigms,

full audit trails for every transition,

and no leakage of “strategy thinking” into the deterministic lane.

Realign Master System Spec (Engine + Research Machine)
0) One-sentence definition

A reproducible, auditable, multi-paradigm trading research and execution system that runs a deterministic microbatch engine (daily × instrument-cluster) and a separate research/training lane (Bayes/GA/RFE/portfolio evaluation) to generate, test, and promote strategies (“paradigms”) as configuration + focused logic, not hard-wired rules.

1) Non-negotiable design goals
1.1 Multi-paradigm by construction

The system must support many strategy families equally (ICT, stat-TS, mean reversion, cross-sectional RV, vol/range, meta/ensemble, microstructure, etc.).

No paradigm is “special” in code structure, data model, or orchestration.

A “paradigm” is:

Config (parameters, feature set, thresholds, constraints, assets/universe)

Focused logic (hypothesis generation + scoring + gating hooks)

Stable artifacts (hypotheses, critic, decisions, portfolio, trade paths)

1.2 Hard separation of lanes (strong boundaries)

Deterministic engine lane: pure, replayable, no learning, no mutation of live configs.

Research/training lane: Bayes/GA/RFE/Monte Carlo/Markov, hyperparam search, feature selection, evaluation.

Control-plane/tools lane: orchestration scripts, config management, run scheduling, reporting, governance.

1.3 Reproducibility and auditability

Every run must be reproducible from:

immutable inputs (data snapshot + configs)

deterministic RNG policy (seeded via RunContext)

versioned code (Git commit)

stable artifact schemas (Parquet source-of-truth)

1.4 Explicit, observable, debuggable

Every step writes artifacts and metrics.

Every decision is explainable via stored intermediate columns, scores, and gating reasons.

No “hidden state” inside long-lived objects.

2) Canonical identity model (required on all artifacts)

All runs and outputs must be keyed by a RunContext identity model:

Required identifiers

env: dev/prod/research

mode: backtest/paper/live

snapshot_id: immutable input snapshot identifier (data + configs)

run_id: unique per execution run

experiment_id: research grouping id (optional in engine, required in research)

candidate_id: strategy candidate id (required for research/training outputs)

paradigm_id: which paradigm family/config generated the hypothesis/decision

principle_id: optional sub-logic identifier inside a paradigm (rule family, setup type)

RNG policy

Central RNG policy derived from (base_seed, snapshot_id, run_id, instrument_cluster_id, dt, step_name) so that:

microbatch is deterministic

research can create controlled stochasticity with explicit seeds

3) Data & storage conventions
3.1 Data formats

Parquet is the source of truth for artifacts and datasets.

Polars is the default dataframe engine.

DuckDB is allowed as a query layer (read-only convenience), not as the canonical store.

3.2 Partitioning conventions

Partition by the keys needed for slicing, reproducibility, and incremental runs:

engine artifacts: (mode, run_id, instrument, dt) plus paradigm_id where relevant

research artifacts: (experiment_id, candidate_id, dt) plus universe partitioning if needed

3.3 Directory layout (canonical)

A minimal stable layout that supports comparison across time/paradigms:

conf/
  retail.yaml                 # global runtime config and universe metadata (read-only in runs)
  paradigms/                  # paradigm configs (YAML)
data/
  candles/                    # OHLCV / bar data (partitioned)
  ticks/                      # tick/L1 if available
  features/                   # optional cached features (if strictly deterministic + versioned)
artifacts/
  engine/
    hypotheses/
    critic/
    decisions/
    orders/
    fills/
    trade_paths/
    reports/
  research/
    feature_scores/
    bayes/
    ga/
    rfe/
    portfolio/
docs/
  MASTER_SYSTEM_SPEC.md       # this doc
src/
  engine/
  research/
  control_plane/
scripts/


Control-plane rule: engine runs must never silently change conf/ or live configs. Any config changes occur via explicit tooling and result in a new snapshot_id.

4) Engine: microbatch pipeline (deterministic lane)
4.1 Microbatch definition

A microbatch is: one trading day × one instrument cluster
Entry-point: run_microbatch(ctx, key) where key = (dt, instrument_cluster_id).

4.2 BatchState (shared state object)

A single BatchState flows through ordered steps. It must be:

immutable-by-default pattern (copy-on-write or explicit .with_*() methods)

serializable references to artifact paths

strict about required columns per stage

4.3 Canonical ordered steps

The engine runs the same ordered step chain for every paradigm, with paradigm-specific logic plugged into defined extension points:

ingest

load bars/ticks for the cluster/day

attach universe metadata (pip size, min tick, session calendar)

features

build deterministic features (HTF/LTF, microstructure where available)

windows

construct analysis windows (sessions, ranges, swing windows, volatility regimes)

hypotheses

generate candidate hypotheses/trade ideas (multi-paradigm)

critic

score and explain; attach risk, regime fit, constraint checks

pretrade

transform accepted hypotheses into actionable intents (entry/SL/TP/brackets)

gatekeeper

final eligibility filter + context selection (risk limits, correlation, schedule, liquidity)

portfolio

position sizing and cross-sectional allocation

brackets

finalize order instructions and bracket logic

reports

write summary metrics, diagnostics, traceability outputs

4.4 Step contracts (required)

Each step must have:

required_inputs: artifact references and minimum column sets

produces: artifact schema + partition keys

deterministic behavior given (ctx, key)

structured logging with run identifiers

5) Paradigm interface (how “multi-paradigm” is implemented)
5.1 Paradigm = config + plugin logic

A paradigm contributes:

Config: parameters, universe, timeframes, windows, feature toggles

Hypothesis generator: creates candidate hypotheses in normalized schema

Critic hooks: optional additional scores/filters

Pretrade mapping: entry model mapping (market/limit/stop logic)

Portfolio hooks: optional factor exposures / risk budgets (still in normalized interface)

5.2 No paradigm-specific pipelines

The pipeline order is fixed. Paradigms cannot reorder steps; they only implement hooks.

5.3 Normalized hypothesis model (engine artifact)

Every hypothesis is a row with:

identity: snapshot_id, run_id, dt, instrument, instrument_cluster_id, paradigm_id, principle_id, hypothesis_id

time/context: anchor_tf, entry_tf, window_id, event_ts, session_bucket, dow_bucket, tod_bucket

directionality: side (long/short), optionally bias_strength

proposal: entry_px, sl_px, tp_px or sufficient info to derive them deterministically

explainability: feature references, tags, and a params_json / structured column

5.4 Normalized critic model (engine artifact)

Every critic row must include:

hypothesis_id join key + all identity keys

critic_score_total and named component scores (e.g., score_regime, score_structure, score_risk, score_costs)

gating flags + reasons (string enums): reject_reason_primary, reject_reasons_all

derived risk/cost estimates: spread/slippage estimates, expected excursion stats if available

6) Research machine (training lane): Bayes + GA + RFE + portfolio evaluation
6.1 The research loop (conceptual)

Research consumes engine artifacts (hypotheses/critic/decisions/trade_paths) and produces improved candidates/configs:

Collect training sets

join outcomes to hypotheses and contexts (including microstructure datasets where available)

Feature evaluation

score features, thresholds, stability across regimes, leakage checks

Candidate generation

GA grammar generates rule structures / hypothesis templates

RFE enumerates and prunes recipes (feature subsets + rule components)

Parameter learning

Bayes: threshold/posterior learning for decision rules, gates, or scoring weights

Portfolio layer evaluation

cross-sectional construction, correlation control, risk budgets

Promotion

produce a new candidate config bundle => new candidate_id and (when promoted) new snapshot_id

6.2 Bayes training (requirements)

Bayesian components are used for:

threshold learning (feature → decision boundary)

score calibration (critic component weighting/posterior)

regime-conditional parameterization (different posteriors per context bucket)

Outputs must be:

versioned posterior summaries

explicit priors

dataset hash / snapshot references

calibration diagnostics

6.3 GA grammar training (requirements)

GA is used to evolve:

hypothesis rule structures (“grammar”)

logical compositions of features/windows/conditions

parameter sets encoded as genes with explicit bounds

GA outputs must include:

full candidate genome

decoded human-readable rule spec

fitness breakdown (return, drawdown, stability, turnover, costs, regime robustness)

seeds and reproducibility metadata

6.4 RFE (recipe/rule enumerator) (requirements)

RFE enumerates “recipes”:

feature subsets

rule components

gating variants

window definitions

RFE must produce:

ranked recipes

ablation results

sensitivity to timeframe and regime

overfit controls (walk-forward, embargo, clustering by instrument/time)

6.5 Portfolio research layer (requirements)

Cross-sectional portfolio logic is a first-class component:

allocations across instruments/cluster

correlation and exposure control

risk budgeting and leverage caps

capacity and cost constraints (spread/slippage models)

Portfolio outputs must be comparable across candidates via standardized metrics.

7) Gatekeeper & context selection (engine + research-critical)

Gatekeeper is the final enforcement point before orders:

selects the active context/regime interpretation for the day/instrument

enforces hard risk and operational constraints

filters/limits correlated bets

ensures “explainable rejection”: every reject must have a stored reason code

Context selection must be:

deterministic in engine lane

learnable in research lane (e.g., posterior over regimes), but only promoted via config snapshots

8) Trade paths and evaluation artifacts (must exist)
8.1 Trade path artifact (canonical)

A normalized table that records the full lifecycle:

ids: snapshot_id, run_id, dt, trade_id, instrument, paradigm_id, principle_id

entry/exit: entry_ts, entry_px, exit_ts, exit_px

bracket: tp_px, sl_px

context: time buckets, windows, regime labels

outcome: return, MAE/MFE, duration, costs (explicit)

8.2 Research evaluation staples

Every candidate evaluation must include:

walk-forward or out-of-sample split metadata

stability across regimes and instruments

turnover and cost-adjusted performance

failure mode clustering (why it loses, where it breaks)

9) Governance & internal controls (LLM-safe and operator-safe)
9.1 Control-plane safety rule

The LLM must never silently mutate:

live configs

production parameters

execution toggles

All changes must be:

proposed as diffs

reviewed/approved by the operator

materialized as a new snapshot/config bundle

9.2 Audit trail requirements

Every run must record:

code version (git commit)

config snapshot ids and hashes

dataset hashes (or partition lists)

step-level artifact manifests

10) Implementation defaults (technical constraints)

Python monolith, CPU-oriented.

NumPy/SciPy + Polars + Parquet as primary stack; DuckDB allowed as query convenience.

Deterministic engine microbatches with strict step ordering.

Research lane may be stochastic but must be reproducible via explicit seeds and stored metadata.

Strong schema discipline: add columns in backward-compatible ways; never break joins.

11) What “done” looks like (final target design)

A single operator can:

Run deterministic microbatches across many days/instrument clusters.

Inspect hypotheses → critic → decisions → portfolio → trade_paths with full traceability.

Run Bayes training to calibrate thresholds and critic weights.

Run GA grammar to discover rule structures across paradigms.

Run RFE enumeration to prune and validate feature/rule recipes.

Evaluate candidates through the portfolio layer with standardized metrics.

Promote a candidate to a new config snapshot safely (no silent live mutation).

Repeat across paradigms without changing the core engine pipeline.

12) Guidance to the next LLM (Codex/GitHub)

When updating this repo, prioritize:

preserving deterministic engine contracts

improving observability (schemas, logs, reports)

adding paradigms as plugins/config + focused hooks (not new pipelines)

keeping artifacts comparable across time and across paradigms

never editing live configs without explicit operator-approved diffs

If any handbook text or legacy notes conflict with the above principles:

flag the conflict explicitly

propose the better design

keep backward compatibility in artifacts whenever possible
