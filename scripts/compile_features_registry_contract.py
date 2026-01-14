from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


# -----------------------------------------------------------------------------
# Registry → compiled contract
#
# Generates a deterministic python module containing:
#   REGISTRY_TABLE_COLUMNS: dict[str, dict[str, str]]
#
# This module is imported by engine.core.schema at runtime to expand TableSchema
# contracts without importing YAML in production code.
# -----------------------------------------------------------------------------


RESERVED_ENGINE_COLS: set[str] = {
    # Engine identity / partition owner
    "snapshot_id",
    "run_id",
    "mode",
    "dt",
    # Legacy / tolerated
    "trading_day",
}


# Common primary-key column dtypes (used only if a family forgets to declare a key in columns)
DEFAULT_KEY_DTYPES: dict[str, str] = {
    "instrument": "string",
    "anchor_tf": "string",
    "tf_entry": "string",
    "ts": "timestamp",
    "anchor_ts": "timestamp",
    "pcr_window_ts": "timestamp",
    "zone_id": "string",
    "trade_id": "string",
}


def _norm_dtype(dtype: str) -> str:
    t = (dtype or "").strip().lower()
    if t in ("str", "utf8"):
        return "string"
    if t in ("i64", "int64"):
        return "int"
    if t in ("f64", "float", "float64", "double"):
        return "double"
    if t in ("bool", "boolean"):
        return "boolean"
    if t in ("datetime", "timestamp", "datetime[us]", "datetime[μs]"):
        return "timestamp"
    if t in ("date",):
        return "date"
    if t in ("array<string>", "list<string>"):
        return "array<string>"
    return t or "string"


def _as_tables(family_spec: dict[str, Any]) -> list[str]:
    """
    Support both legacy `table: str` and newer `tables: [..]` forms.
    """
    if "tables" in family_spec and family_spec["tables"] is not None:
        v = family_spec["tables"]
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]
    if "table" in family_spec and family_spec["table"] is not None:
        return [str(family_spec["table"])]
    return []


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _merge_columns(dest: Dict[str, str], src: Dict[str, str], *, context: str) -> None:
    for k, v in src.items():
        if k in RESERVED_ENGINE_COLS:
            # registry is not allowed to own these
            continue
        v = _norm_dtype(v)
        if k in dest and dest[k] != v:
            raise ValueError(f"dtype conflict while compiling {context}: col={k!r} {dest[k]!r} vs {v!r}")
        dest[k] = v


def compile_registry(
    registry_path: Path,
    *,
    include_disabled: bool,
    strict_reserved: bool,
) -> Dict[str, Dict[str, str]]:
    spec = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("features_registry root must be a mapping of family_id -> spec")

    table_cols: Dict[str, Dict[str, str]] = {}

    for family_id, fam in spec.items():
        if not isinstance(fam, dict):
            continue

        enabled = bool(fam.get("enabled", True))
        if (not include_disabled) and (not enabled):
            continue

        tables = _as_tables(fam)
        if not tables:
            continue

        cols_spec = fam.get("columns") or {}
        if not isinstance(cols_spec, dict):
            raise ValueError(f"family {family_id!r}: columns must be a mapping")

        # Pull declared columns
        declared: Dict[str, str] = {}
        for col, meta in cols_spec.items():
            if not isinstance(meta, dict):
                continue
            dtype = _norm_dtype(str(meta.get("dtype", "string")))
            declared[str(col)] = dtype

        # Ensure primary_key columns are present (explicitly or via defaults)
        pk = fam.get("primary_key") or []
        if isinstance(pk, (list, tuple)):
            for c in pk:
                c = str(c)
                if c not in declared:
                    if c in DEFAULT_KEY_DTYPES:
                        declared[c] = DEFAULT_KEY_DTYPES[c]
                    else:
                        # Missing dtype – force explicitness (better research hygiene)
                        raise ValueError(
                            f"family {family_id!r}: primary_key col {c!r} missing from columns with dtype"
                        )

        # Reserved enforcement
        if strict_reserved:
            forbidden = sorted(set(declared).intersection(RESERVED_ENGINE_COLS))
            if forbidden:
                raise ValueError(
                    f"family {family_id!r}: registry must not declare reserved engine columns: {forbidden}"
                )

        # Merge into each target table
        for t in tables:
            table_cols.setdefault(t, {})
            _merge_columns(table_cols[t], declared, context=f"{family_id} -> {t}")

    # Ensure deterministic key order when emitting by sorting in the emitter, not here.
    return table_cols


def render_module(
    *,
    registry_path: Path,
    out_path: Path,
    table_cols: Dict[str, Dict[str, str]],
) -> str:
    gen_ts = datetime.now(timezone.utc).isoformat()

    # Deterministic ordering
    tables_sorted = sorted(table_cols.keys())
    body_lines: List[str] = []
    body_lines.append("# AUTO-GENERATED FILE. DO NOT EDIT BY HAND.")
    body_lines.append("#")
    body_lines.append("# Generated by: scripts/compile_features_registry_contract.py")
    body_lines.append(f"# Source: {registry_path.as_posix()}")
    body_lines.append(f"# Generated (UTC): {gen_ts}")
    body_lines.append("")
    body_lines.append("from __future__ import annotations")
    body_lines.append("")
    body_lines.append("from typing import Dict")
    body_lines.append("")
    body_lines.append(f"REGISTRY_SOURCE = {registry_path.as_posix()!r}")
    body_lines.append(f"GENERATED_UTC = {gen_ts!r}")
    body_lines.append("")
    body_lines.append("REGISTRY_TABLE_COLUMNS: Dict[str, Dict[str, str]] = {")
    for t in tables_sorted:
        cols = table_cols[t]
        body_lines.append(f"    {t!r}: {{")
        for c in sorted(cols.keys()):
            body_lines.append(f"        {c!r}: {cols[c]!r},")
        body_lines.append("    },")
    body_lines.append("}")
    body_lines.append("")

    text = "\n".join(body_lines)
    # add content hash as a comment to help drift-debugging
    text += f"\n# content_sha={_hash_text(text)}\n"
    return text


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=Path("conf/features_registry.yaml"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("src/engine/core/_registry_table_columns.py"),
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Fail (exit 1) if generated output differs from on-disk module.",
    )
    ap.add_argument(
        "--only-enabled",
        action="store_true",
        help="Compile only families with enabled: true.",
    )
    ap.add_argument(
        "--strict-reserved",
        action="store_true",
        help="Fail if registry declares engine-reserved columns (snapshot_id/run_id/mode/dt/trading_day).",
    )
    args = ap.parse_args(argv)

    registry_path: Path = args.registry
    out_path: Path = args.out

    if not registry_path.exists():
        print(f"[error] registry not found: {registry_path}", file=sys.stderr)
        return 2

    table_cols = compile_registry(
        registry_path,
        include_disabled=not args.only_enabled,
        strict_reserved=bool(args.strict_reserved),
    )

    expected = render_module(registry_path=registry_path, out_path=out_path, table_cols=table_cols)

    if args.check:
        if not out_path.exists():
            print(f"[drift] generated module missing: {out_path}", file=sys.stderr)
            print("        run: python scripts/compile_features_registry_contract.py", file=sys.stderr)
            return 1
        actual = out_path.read_text(encoding="utf-8")
        if actual != expected:
            print(f"[drift] generated module out of date: {out_path}", file=sys.stderr)
            print("        run: python scripts/compile_features_registry_contract.py", file=sys.stderr)
            return 1
        print(f"[ok] no drift: {out_path}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(expected, encoding="utf-8")
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
