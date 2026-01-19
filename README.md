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

If you want, I can also specify the exact polars schema for data/windows and data/market_events, including nullability contracts and validation functions that fail-fast when Phase B invariants are violated.
