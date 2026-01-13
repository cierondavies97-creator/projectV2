from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from engine.core.api import load_features_registry

DEFAULT_OUT = Path("conf/features_auto.yaml")
TABLE_KEY = "data/features"


# ----------------------------
# YAML helpers
# ----------------------------


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _deep_get(d: dict[str, Any], path: Iterable[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _deep_ensure_dict(d: dict[str, Any], path: Iterable[str]) -> dict[str, Any]:
    cur: dict[str, Any] = d
    for k in path:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    return cur


def _backup_file(path: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return bak


# ----------------------------
# Registry helpers
# ----------------------------


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    columns: tuple[str, ...]
    threshold_keys: tuple[str, ...]


def _iter_tunable_families_for_table(table_key: str) -> dict[str, FamilySpec]:
    reg = load_features_registry()
    out: dict[str, FamilySpec] = {}

    for family_id, family in reg.families.items():
        family_tables = getattr(family, "tables", None) or {}
        if table_key not in family_tables:
            continue

        threshold_keys = tuple(str(x) for x in (getattr(family, "threshold_keys", []) or []))
        columns = tuple(str(x) for x in (getattr(family, "columns", []) or []))

        # Only families that *actually* declare tunable keys are relevant here
        if not threshold_keys or not columns:
            continue

        out[family_id] = FamilySpec(
            family_id=family_id,
            columns=columns,
            threshold_keys=threshold_keys,
        )

    return out


# ----------------------------
# Patch logic
# ----------------------------


def _apply_threshold_patches(
    doc: dict[str, Any],
    *,
    paradigm_id: str,
    set_values: dict[str, float],
    force_create_paths: bool,
) -> tuple[dict[str, Any], int]:
    """
    Apply patches in-place. Returns (doc, n_changes).
    """
    n_changes = 0

    # Ensure top-level metadata (do not destroy existing fields)
    if "schema_version" not in doc:
        doc["schema_version"] = 1
        n_changes += 1
    if "generated_by" not in doc:
        doc["generated_by"] = "train_features_thresholds_stub"
        n_changes += 1

    paradigms = _deep_ensure_dict(doc, ["paradigms"])
    pnode = _deep_ensure_dict(paradigms, [paradigm_id])
    tables = _deep_ensure_dict(pnode, ["tables"])

    # We patch only within data/features
    tnode = _deep_ensure_dict(tables, [TABLE_KEY])
    families_node = _deep_ensure_dict(tnode, ["families"])

    # Use registry to know what exists + which keys are allowed
    tunable = _iter_tunable_families_for_table(TABLE_KEY)

    # Validate requested keys are known somewhere in the registry (hard fail on typos)
    all_allowed_keys = {k for spec in tunable.values() for k in spec.threshold_keys}
    unknown = [k for k in set_values.keys() if k not in all_allowed_keys]
    if unknown:
        raise SystemExit(
            f"Unknown threshold key(s) {unknown}. Allowed keys (from features_registry.yaml): "
            f"{sorted(all_allowed_keys)}"
        )

    # Patch each (family, feature) leaf
    for family_id, spec in tunable.items():
        fam_leaf = families_node.get(family_id)
        if fam_leaf is None:
            if not force_create_paths:
                continue
            fam_leaf = {}
            families_node[family_id] = fam_leaf
            n_changes += 1

        if not isinstance(fam_leaf, dict):
            # If corrupted, we refuse to silently clobber unless forced
            if not force_create_paths:
                continue
            families_node[family_id] = {}
            fam_leaf = families_node[family_id]
            n_changes += 1

        for feature in spec.columns:
            feat_leaf = fam_leaf.get(feature)
            if feat_leaf is None:
                if not force_create_paths:
                    continue
                # Create a minimal feature leaf; we preserve your richer structure if it already exists
                feat_leaf = {"enabled": True, "thresholds": {}}
                fam_leaf[feature] = feat_leaf
                n_changes += 1

            if not isinstance(feat_leaf, dict):
                if not force_create_paths:
                    continue
                fam_leaf[feature] = {"enabled": True, "thresholds": {}}
                feat_leaf = fam_leaf[feature]
                n_changes += 1

            thresholds = feat_leaf.get("thresholds")
            if thresholds is None:
                if not force_create_paths:
                    continue
                feat_leaf["thresholds"] = {}
                thresholds = feat_leaf["thresholds"]
                n_changes += 1

            if not isinstance(thresholds, dict):
                if not force_create_paths:
                    continue
                feat_leaf["thresholds"] = {}
                thresholds = feat_leaf["thresholds"]
                n_changes += 1

            # Only apply keys that this family declares tunable
            allowed_here = set(spec.threshold_keys)
            for k, v in set_values.items():
                if k not in allowed_here:
                    continue
                prev = thresholds.get(k, None)
                if prev != v:
                    thresholds[k] = float(v)
                    n_changes += 1

    return doc, n_changes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paradigm-id", default="dev_baseline")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override threshold key values. Format: key=value (repeatable). Example: abs_min_for_signal=2.0",
    )
    p.add_argument(
        "--force-create-paths",
        action="store_true",
        help="Create missing family/feature/thresholds leaves if they do not exist in the current features_auto.yaml.",
    )
    args = p.parse_args()

    set_values: dict[str, float] = {}
    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"--set must be key=value, got: {kv!r}")
        k, v = kv.split("=", 1)
        set_values[k.strip()] = float(v.strip())

    if not set_values:
        raise SystemExit("No --set provided; refusing to run (would be a no-op).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing YAML if present; otherwise start a minimal doc.
    if out_path.exists():
        existing = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        doc = _as_dict(existing)
    else:
        doc = {
            "schema_version": 1,
            "generated_by": "train_features_thresholds_stub",
            "paradigms": {},
        }

    before = yaml.safe_dump(doc, sort_keys=False)

    doc, n_changes = _apply_threshold_patches(
        doc,
        paradigm_id=args.paradigm_id,
        set_values=set_values,
        force_create_paths=args.force_create_paths,
    )

    after = yaml.safe_dump(doc, sort_keys=False)

    if after == before:
        raise SystemExit("No changes applied (no-op). Refusing to overwrite output file.")

    # Safety: refuse to write an â€œempty tablesâ€ structure (the bug you hit)
    tables_node = _deep_get(doc, ["paradigms", args.paradigm_id, "tables"])
    if isinstance(tables_node, dict) and len(tables_node) == 0:
        raise SystemExit("Refusing to write: paradigms.<id>.tables is empty. (Would clobber your existing file.)")

    # Backup then write
    if out_path.exists():
        bak = _backup_file(out_path)
        print(f"Backup: {bak}")

    out_path.write_text(after, encoding="utf-8")
    print(f"Wrote: {out_path} (changes={n_changes})")


if __name__ == "__main__":
    main()
