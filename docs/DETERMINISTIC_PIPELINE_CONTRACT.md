# Deterministic Pipeline Contract (Engine Lane)

## 0) Purpose
This document defines the **system-wide deterministic pipeline contract** for the engine lane. It is the enforceable standard for all engine steps, writers, and tooling that produce deterministic artifacts. The contract is designed to maximize:

- **Reproducibility** (exact replays across time)
- **Auditability** (clear provenance and explainability)
- **Cross-paradigm comparability** (uniform schemas and step order)
- **Operational clarity** (single-writer ownership, explicit inputs/outputs)

If any design or implementation conflicts with this contract, the conflict **must** be explicitly called out and resolved.

---

## System concept alignment (source of truth: README)

This contract inherits the repo-wide system concept:

- **Multi-paradigm by construction**; pipeline order and schemas are shared across paradigms.
- **Hard lane separation** between deterministic engine, research/training, and control-plane.
- **Deterministic engine lane** outputs are reproducible and audit-ready; configs are immutable in-run.
- **Stable identity model** ensures comparability across runs and paradigms.

If any text here conflicts with the README, the README is authoritative.

---

## 1) Scope
This contract applies to:

- All engine steps in the microbatch pipeline.
- Any script or tool that writes engine artifacts.
- Schema registries and IO ownership contracts that govern engine outputs.

It does **not** define strategy logic; it defines how deterministic infrastructure behaves and how artifacts are produced.

---

## 2) Deterministic market-structure scope (Phase B–E)

The deterministic pipeline **must** treat Phase B–E as a **state machine** driven by explicit formulas and evidence fields:

- All phase labels (`dr_phase`) are produced only in `features → windows`.
- All transitions emit reason codes plus measurable evidence (probe/reclaim/accept/trend).
- No step may recompute or override phase labels downstream.

**Minimum required evidence fields** (per anchor row):

- `dr_id`, `dr_phase`, `dr_low`, `dr_high`, `dr_mid`, `dr_width`, `dr_width_atr`
- `dr_start_ts`, `dr_last_update_ts`, `dr_age_bars`
- `inside_ratio_L`, `tests_L`, `test_high_count_L`, `test_low_count_L`
- transition evidence: `probe_side`, `pierce_dist`, `reclaim_margin`, `accept_dist`, `accept_bars`, `retest_pass`, `trend_dist`, `trend_bars`
- `dr_reason_code` (string enum)

These fields are required for auditability and reproducibility of Phases B–E.

---

## 3) Non-negotiable invariants

1. **Single-writer rule**
   - Each artifact/table is written by exactly one owning step.

2. **Deterministic outputs**
   - Outputs are a pure function of `(RunContext, MicrobatchKey, immutable inputs, configs, code version)`.

3. **Immutable configs**
   - Engine runs must never mutate live configs. Changes require explicit diffs and a new `snapshot_id`.

4. **No hidden state**
   - Steps cannot depend on implicit caches or mutable global state.

5. **Fail-fast validation**
   - Required columns and non-null constraints must be enforced. Silent coercion is forbidden.

6. **Stable partitioning**
   - Partition keys are part of the contract. Do not change implicitly.

7. **Full provenance**
   - Every row contains run identity + microbatch identity keys.

8. **Observability**
   - Steps must emit structured logs, row counts, and manifests (input hashes, output paths).

---

## 4) Canonical identity model (required on all artifacts)

Every deterministic artifact must include the identity model:

- `env` (dev | prod | research)
- `mode` (backtest | paper | live)
- `snapshot_id`
- `run_id`
- `trading_day` (or `dt`)
- `cluster_id`
- `instrument` (if per-instrument)
- `paradigm_id` (if applicable)
- `principle_id` (optional)

**RNG policy:** any randomness must derive from the canonical seed:

```
(base_seed, snapshot_id, run_id, cluster_id, trading_day, step_name)
```

---

## 4) Canonical step order (engine lane)

The deterministic pipeline runs the following ordered steps. **No paradigm may reorder or bypass steps.**

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

Backtest-only step:

- **fills** (only in backtest mode)

---

## 6) Step contract (minimum required interface)

Every step **must** define and enforce the following:

### 6.1 Required inputs
- Required tables and minimum column sets.
- Required config namespaces and keys.

### 6.2 Outputs
- Persisted tables, schemas, partition keys, and owning step.

### 6.3 Determinism
- No implicit randomness. All randomness derives from the canonical seed policy.
- Tie-breaking rules are explicit and deterministic.

### 6.4 Validation
- Fail fast if required inputs are missing.
- Fail fast if required output columns are null.
- Validate output schema against registry.

### 6.5 Observability
- Structured logs with identity keys.
- Row counts and manifests with input hashes/output paths.

---

## 7) Artifact ownership and IO contract

- Each artifact key maps to exactly one owning step.
- Ownership changes require updates to the IO contract and tests.
- No artifact may be written by more than one step.

**Compliance tool:** `scripts/audit_io_contract.py` must pass after any ownership changes.

---

## 8) Schema discipline

1. **Additive changes only**
   - Renames/removals require a versioned migration plan.

2. **Nullability rules**
   - Required columns must be non-null.

3. **Registry alignment**
   - Any schema change must be reflected in the registry and validation tooling.

---

## 9) Phase B–E compliance (deterministic lane requirement)

A deterministic pipeline is **not** Phase B–E compliant unless it satisfies all requirements below.

### 9.1 Windows as source of truth
`data/windows` must contain, for each anchor row:

- `dr_id`, `dr_phase`, `dr_low`, `dr_high`, `dr_mid`, `dr_width`
- `dr_age_bars`, `dr_start_ts`, `dr_last_update_ts`
- `pd_index`, `range_position`
- counters: `test_high_count`, `test_low_count`, `liq_eqh_count`, `liq_eql_count`
- `dr_reason_code` and optional `dr_score_*`
- `dr_width_atr`, `inside_ratio_L`, `tests_L`
- `probe_side`, `pierce_dist`, `reclaim_margin`, `accept_dist`, `accept_bars`, `retest_pass`, `trend_dist`, `trend_bars`

### 9.2 Event table for auditability
Write `data/market_events` (or `data/dealing_range_events`) containing:

- `instrument`, `anchor_tf`, `ts`, `event_type`, `event_strength`, `ref_level`, `dr_id`, evidence fields

### 9.3 Deterministic event detection + state machine
- Event detectors are stateless, parameterized by config, and deterministic in tie-breaking.
- Phase transitions occur via a deterministic fold over anchor rows.

### 9.4 Downstream consumption only
Hypotheses/critic/gatekeeper **consume** `dr_*` fields and **must not** recompute Phase B.

### 9.5 Fail-fast validation
If `dr_phase` is present, required `dr_*` fields must be non-null.

### 9.6 Golden fixtures
Golden fixtures must lock Phase B–E labeling stability per anchor timeframe.

---

## 10) Policy IDs and versioning (required)

Every deterministic output must carry version/policy identifiers to make runs comparable:

- `phase_version`, `threshold_bundle_id`
- `micro_policy_id`, `jump_policy_id`, `impact_policy_id`, `options_policy_id`

These IDs are part of the artifact identity and **must** be persisted in outputs.

---

## 11) MarketState cube (canonical output)

The deterministic lane must produce (or be able to derive) a canonical MarketState cube keyed by:

`(instrument, anchor_tf, anchor_ts)`

**Required components**:

- Phase B–E fields + evidence (`dr_*`, probe/reclaim/accept/retest/trend reason codes)
- Microstructure fields (AggImb/OFI/intensity)
- Jump/vol fields (RV/BV/JV + semivariance)
- Impact fields (Kyle λ + regime)
- Options context fields (ATM IV, skew, term slope, VRP, model-free implied variance if feasible)

This cube is the canonical context for strategies. Any missing component must be explicitly null with a documented policy reason.

---

## 12) Change checklist (mandatory for pipeline updates)

Any deterministic pipeline change must include:

- IO contract update (ownership + persisted keys).
- Schema registry update (columns, types, nullability).
- Tests:
  - Golden fixtures for deterministic labeling where applicable.
  - Unit tests for new step logic.
- Documentation update (this contract + any affected specs).

---

## 13) Connected scripts/tools to audit when the contract changes

If steps, schemas, or ownership rules change, audit/update:

- `scripts/run_microbatch.py`
- `scripts/run_microbatch.ps1`
- `scripts/audit_io_contract.py`
- `scripts/compile_features_registry_contract.py`
- `scripts/check_anchor_grid.py`
- `scripts/smoke_validate_parquet.py`
- `scripts/dev_inspect_*`
- `scripts/run_ingest.py` / `scripts/run_features.py` / `scripts/run_reports.py` (placeholders must be implemented or removed)

---

## 14) Compliance gates (definition of “done”)

A pipeline change is compliant only when:

- Deterministic runs reproduce identical outputs across replays.
- IO contract and schema registry match actual writers.
- Phase B outputs and event provenance are present (when enabled).
- Golden fixtures pass without drift.
- Required logs and manifests are emitted for auditability.
