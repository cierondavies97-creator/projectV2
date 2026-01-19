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

## 5) Step contract (minimum required interface)

Every step **must** define and enforce the following:

### 5.1 Required inputs
- Required tables and minimum column sets.
- Required config namespaces and keys.

### 5.2 Outputs
- Persisted tables, schemas, partition keys, and owning step.

### 5.3 Determinism
- No implicit randomness. All randomness derives from the canonical seed policy.
- Tie-breaking rules are explicit and deterministic.

### 5.4 Validation
- Fail fast if required inputs are missing.
- Fail fast if required output columns are null.
- Validate output schema against registry.

### 5.5 Observability
- Structured logs with identity keys.
- Row counts and manifests with input hashes/output paths.

---

## 6) Artifact ownership and IO contract

- Each artifact key maps to exactly one owning step.
- Ownership changes require updates to the IO contract and tests.
- No artifact may be written by more than one step.

**Compliance tool:** `scripts/audit_io_contract.py` must pass after any ownership changes.

---

## 7) Schema discipline

1. **Additive changes only**
   - Renames/removals require a versioned migration plan.

2. **Nullability rules**
   - Required columns must be non-null.

3. **Registry alignment**
   - Any schema change must be reflected in the registry and validation tooling.

---

## 8) Phase B compliance (deterministic lane requirement)

A deterministic pipeline is **not** Phase B compliant unless it satisfies all requirements below.

### 8.1 Windows as source of truth
`data/windows` must contain, for each anchor row:

- `dr_id`, `dr_phase`, `dr_low`, `dr_high`, `dr_mid`, `dr_width`
- `dr_age_bars`, `dr_start_ts`, `dr_last_update_ts`
- `pd_index`, `range_position`
- counters: `test_high_count`, `test_low_count`, `liq_eqh_count`, `liq_eql_count`
- `dr_reason_code` and optional `dr_score_*`

### 8.2 Event table for auditability
Write `data/market_events` (or `data/dealing_range_events`) containing:

- `instrument`, `anchor_tf`, `ts`, `event_type`, `event_strength`, `ref_level`, `dr_id`, evidence fields

### 8.3 Deterministic event detection + state machine
- Event detectors are stateless, parameterized by config, and deterministic in tie-breaking.
- Phase transitions occur via a deterministic fold over anchor rows.

### 8.4 Downstream consumption only
Hypotheses/critic/gatekeeper **consume** `dr_*` fields and **must not** recompute Phase B.

### 8.5 Fail-fast validation
If `dr_phase` is present, required `dr_*` fields must be non-null.

### 8.6 Golden fixtures
Golden fixtures must lock Phase B labeling stability per anchor timeframe.

---

## 9) Change checklist (mandatory for pipeline updates)

Any deterministic pipeline change must include:

- IO contract update (ownership + persisted keys).
- Schema registry update (columns, types, nullability).
- Tests:
  - Golden fixtures for deterministic labeling where applicable.
  - Unit tests for new step logic.
- Documentation update (this contract + any affected specs).

---

## 10) Connected scripts/tools to audit when the contract changes

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

## 11) Compliance gates (definition of “done”)

A pipeline change is compliant only when:

- Deterministic runs reproduce identical outputs across replays.
- IO contract and schema registry match actual writers.
- Phase B outputs and event provenance are present (when enabled).
- Golden fixtures pass without drift.
- Required logs and manifests are emitted for auditability.

