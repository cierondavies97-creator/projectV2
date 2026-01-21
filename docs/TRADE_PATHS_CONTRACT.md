# Trade Paths Contract (Engine Lane)

## Purpose
Provide a deterministic, auditable contract for `trade_paths` that makes **entry timing explicit** and comparable across paradigms while preserving backtest reproducibility and engine determinism.

This contract is **paradigm‑agnostic** and supports multi‑paradigm research by separating **anchor timing**, **entry intent timing**, and **execution timing** (fills).

## Scope
- Applies to `data/trade_paths` artifacts emitted by hypotheses builders and persisted by `hypotheses_step`.
- Read by downstream steps (`brackets_step`, `fills_step`, reporting), but this contract does **not** mutate any live config.

## Core Principles
1. **Explicitness:** Every `entry_ts` must have a declared provenance.
2. **Determinism:** Entry timing must be reproducible from immutable inputs (snapshot + config + data).
3. **Separation of concerns:**
   - `anchor_ts` describes the setup’s anchor bar.
   - `entry_ts` describes the intended entry time (not necessarily aligned to anchor).
   - Actual execution is simulated by the fills step in backtest.
4. **Backward compatibility:** New fields are additive and optional, but recommended for all paradigms.

---

## System concept alignment (source of truth: README)

This contract aligns to the repo-wide system concept:

- **Multi-paradigm by construction**; trade paths are comparable across paradigms.
- **Hard lane separation** between deterministic engine, research/training, and control-plane.
- **Deterministic engine lane** outputs are reproducible and audit-ready; configs are immutable in-run.
- **Stable identity model** ensures comparability across runs and paradigms.

If any text here conflicts with the README, the README is authoritative.

---

## Required Fields (existing, baseline)
These fields are already required by engine validations and downstream steps:

- `snapshot_id`, `run_id`, `mode`, `dt`
- `instrument`, `trade_id`, `side`
- `anchor_tf`, `tf_entry`
- `entry_ts`  ✅ **Operational entry intent timestamp**

> `entry_ts` is required because `brackets_step` and `fills_step` depend on it to simulate order intent and fills.

---

## New Recommended Fields (additive)
These fields make entry timing explicit and comparable across paradigms.

| Field | Type | Description |
|---|---|---|
| `anchor_ts` | timestamp | Timestamp of the **anchor bar** used for setup logic. |
| `entry_ts_source` | string | **Provenance** of `entry_ts` (see enum below). |
| `entry_ts_offset_ms` | int | Offset in milliseconds from the source timestamp (optional). |
| `entry_ts_is_aligned_anchor_tf` | boolean | Whether `entry_ts` aligns to `anchor_tf` grid. |
| `entry_ts_is_aligned_entry_tf` | boolean | Whether `entry_ts` aligns to `tf_entry` grid. |

### Recommended enum for `entry_ts_source`
Use these canonical values (string). If a paradigm needs a custom variant, prefix with `custom:`.

- `anchor_close`
- `anchor_open`
- `entry_tf_open`
- `entry_tf_close`
- `signal_ts`
- `offset` (explicit offset from anchor or signal)
- `custom:<paradigm_id>.<name>`

> **Design note:** Use `entry_ts_offset_ms` when `entry_ts_source=offset` to preserve reproducibility.

---

## Semantics & Invariants

### 1) Anchor vs Entry
- `anchor_ts` **must** represent the anchor bar timestamp used by the paradigm’s setup logic.
- `entry_ts` **must** represent the intended entry time.
- `entry_ts` **may** equal `anchor_ts`, but if so, `entry_ts_source` must make that explicit (`anchor_close` or `anchor_open`).

### 2) Alignment Flags
- `entry_ts_is_aligned_anchor_tf` = `truncate(entry_ts, anchor_tf) == entry_ts`
- `entry_ts_is_aligned_entry_tf` = `truncate(entry_ts, tf_entry) == entry_ts`

These are diagnostic flags to reveal when entry timing is being snapped to grids.

### 3) Determinism
All entry timestamps **must be derivable** from immutable inputs (windows + features + config + snapshot). Randomness must be seeded via `RunContext` and logged externally if used.

---

## Implementation Guidance (Paradigm Authors)

### Minimal compliant emission
```
anchor_ts = <anchor bar timestamp>
entry_ts = anchor_ts
entry_ts_source = "anchor_close"
entry_ts_is_aligned_anchor_tf = true
entry_ts_is_aligned_entry_tf = true   # if tf_entry == anchor_tf
```

### Offset entry example (research‑friendly)
```
anchor_ts = <anchor bar timestamp>
entry_ts = anchor_ts + 5 minutes
entry_ts_source = "offset"
entry_ts_offset_ms = 300000
entry_ts_is_aligned_anchor_tf = false
entry_ts_is_aligned_entry_tf = true   # if tf_entry == 5m
```

### Signal timestamp example
```
anchor_ts = <anchor bar timestamp>
entry_ts = <signal_ts>
entry_ts_source = "signal_ts"
entry_ts_is_aligned_anchor_tf = false
entry_ts_is_aligned_entry_tf = false
```

---

## Engine‑Lane Consumption

### Brackets step
- Consumes `entry_ts`, `entry_px`, `sl_px`, `tp_px` from `trade_paths`.
- Does **not** require `anchor_ts` but should preserve it if present for downstream reporting.

### Pretrade macro joins
- If `anchor_ts` is present, prefer joining macro windows on `anchor_ts` rather than truncating `entry_ts`.
- This avoids accidental snapping of `entry_ts` to the anchor grid.

---

## Migration Plan (Non‑Breaking)
1. **Add fields to schema** as optional columns.
2. Update paradigms to emit `anchor_ts` and `entry_ts_source`.
3. Update pretrade macro join logic to use `anchor_ts` if present.

No changes to live configs are required. All behavior remains deterministic and replayable.

---

## Rationale
This contract makes entry timing explicit, prevents accidental coupling to anchor bars, and enables rigorous comparison across paradigms without privileging any strategy family.
