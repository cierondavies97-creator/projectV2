# Deterministic Pipeline Contract

## 0) Purpose and scope
This document defines the **enforceable contract** for the deterministic engine lane. It applies to:

- Engine steps and step modules.
- Any script that writes engine artifacts (directly or indirectly).
- Any schema/registry or IO contract that governs deterministic outputs.

The goal is to maximize **reproducibility, auditability, and cross-paradigm comparability** while keeping the engine lane **pure and replayable**.

This contract **does not** define paradigm logic; it defines the *shared, non-negotiable rules* that all paradigms and steps must follow.

---

## 1) Non-negotiable invariants

1. **Single-writer rule**
   - Every artifact/table has exactly one owning step that is allowed to write it.

2. **Deterministic by default**
   - Given the same `(RunContext, MicrobatchKey, inputs, configs, code version)`, the outputs must be identical.

3. **No hidden state**
   - Step functions must not depend on global mutable state, implicit caches, or external services that are not versioned.

4. **Immutable config inputs**
   - Engine runs must never mutate `conf/` or live configs. Changes require explicit diffs and a new `snapshot_id`.

5. **Fail-fast validation**
   - Required columns and non-null constraints must be enforced. Silent coercions are forbidden.

6. **Stable partitioning**
   - Artifact partition keys are part of the contract. They must not change implicitly.

7. **Full provenance**
   - Each output row carries run identity and microbatch identity keys.

8. **Observable and debuggable**
   - Each step must produce structured logs, row counts, and manifests with input hashes and output paths.

---

## 2) Canonical identity model (required on all artifacts)

Every output produced in the deterministic lane **must** be keyed by the identity model:

- `env` (dev | prod | research)
- `mode` (backtest | paper | live)
- `snapshot_id` (immutable input snapshot)
- `run_id` (unique per run)
- `trading_day` / `dt`
- `instrument` / `cluster_id`
- `paradigm_id` (when applicable)
- `principle_id` (optional, when applicable)

**RNG policy:** any randomness must be derived from a canonical seed constructed from:

```
(base_seed, snapshot_id, run_id, cluster_id, trading_day, step_name)
```

This ensures deterministic replay while allowing controlled stochasticity in research.

---

## 3) Step contract (minimum requirements)

Each step must define and enforce:

### 3.1 Required inputs
- Explicit list of required tables and minimum column sets.
- Explicit config namespaces and keys.

### 3.2 Outputs
- Declared outputs (tables) with schemas and partition keys.
- Ownership record that identifies the single writing step.

### 3.3 Determinism
- No implicit randomness. All randomness derived from canonical seed policy.
- Tie-breaking rules must be explicit and deterministic (e.g., sort by timestamp then stable ID).

### 3.4 Validation
- Fail fast if required inputs are missing or if required outputs contain nulls.
- Output schema must conform to registry definitions (column names, dtypes, nullability).

### 3.5 Observability
- Emit structured logs with `env`, `mode`, `snapshot_id`, `run_id`, `trading_day`, `cluster_id`.
- Record row counts, input hashes, and output paths in a manifest.

---

## 4) Artifact ownership and IO contract

- Each artifact key maps to exactly one owning step.
- Any change to ownership **must** update the IO contract and corresponding tests.
- No artifact can be written by multiple steps.

**Compliance tooling:**
- `scripts/audit_io_contract.py` must pass after changes.

---

## 5) Schema discipline

### 5.1 Additive changes only
- New columns are allowed; renames/removals require a versioned migration plan.

### 5.2 Nullability rules
- Columns marked required must never be null in deterministic outputs.
- If a required column is missing or null, the step must error.

### 5.3 Registry alignment
- Any schema changes must be reflected in the schema registry and validation tooling.

---

## 6) Phase B compliance (deterministic lane)

Phase B is a required engine contract. A pipeline is **not** Phase B compliant unless it does all of the following:

1. **Windows as source of truth**
   - Every anchor row in `data/windows` includes:
     - `dr_id`, `dr_phase`, `dr_low`, `dr_high`, `dr_mid`, `dr_width`, `dr_age_bars`, `dr_start_ts`, `dr_last_update_ts`
     - `pd_index`, `range_position`
     - counters: `test_high_count`, `test_low_count`, `liq_eqh_count`, `liq_eql_count`
     - `dr_reason_code` and any `dr_score_*` components

2. **Event table for auditability**
   - Write `data/market_events` (or `data/dealing_range_events`) with:
     - `instrument`, `anchor_tf`, `ts`, `event_type`, `event_strength`, `ref_level`, `dr_id`, evidence fields

3. **Deterministic event detection + state machine**
   - Event detection must be stateless and fully parameterized via config.
   - Phase transitions must be a deterministic fold over anchor rows.

4. **No recomputation downstream**
   - Hypotheses/critic/gatekeeper must consume `dr_*` fields and must not recompute Phase B.

5. **Fail-fast validation**
   - If `dr_phase` is present, the required `dr_*` fields must be non-null.

6. **Golden tests**
   - Store fixture OHLCV segments + expected phase labels per anchor timeframe.

---

## 7) Deterministic pipeline change checklist

Any change to the deterministic pipeline **must** be accompanied by:

- IO contract update (ownership + persisted keys).
- Schema registry update (new columns, types, nullability).
- Tests:
  - golden fixtures for deterministic labels where applicable.
  - unit tests for new step logic.
- Documentation update (this contract + relevant specs).

---

## 8) Connected scripts/tools to review when changing the contract

When steps, schemas, or ownership rules change, audit and update as needed:

- `scripts/run_microbatch.py`
- `scripts/run_microbatch.ps1`
- `scripts/audit_io_contract.py`
- `scripts/compile_features_registry_contract.py`
- `scripts/check_anchor_grid.py`
- `scripts/smoke_validate_parquet.py`
- `scripts/dev_inspect_*`
- `scripts/run_ingest.py` / `scripts/run_features.py` / `scripts/run_reports.py` (placeholders must be implemented or removed)

---

## 9) Compliance gates (definition of “done”)

A pipeline change is **compliant** only when:

- Deterministic runs reproduce identical outputs across replays.
- IO contract and schema registry are consistent with actual writers.
- Phase B outputs and event provenance are present (when enabled).
- Golden fixtures pass without drift.
- All required logs and manifests are emitted for auditability.

