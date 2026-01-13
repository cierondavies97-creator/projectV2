from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.microbatch.io_contract import CONTRACT, ContractMode, expected_owner, expected_persisted, validate_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = PROJECT_ROOT / "src" / "engine" / "microbatch" / "steps"


DECISIONS_STAGE_TO_KEY = {
    "hypotheses": "decisions_hypotheses",
    "critic": "decisions_critic",
    "pretrade": "decisions_pretrade",
    "gatekeeper": "decisions_gatekeeper",
    "portfolio": "decisions_portfolio",
}


WRITER_FN_TO_KEY = {
    "write_features_for_instrument_tf_day": "features",
    "write_windows_for_instrument_tf_day": "windows",
    "write_zones_state_for_instrument_tf_day": "zones_state",
    "write_pcra_for_instrument_tf_day": "pcr_a",
    "write_trade_paths_for_day": "trade_paths",
    "write_brackets_for_instrument_day": "brackets",
    "write_orders_for_instrument_day": "orders",
    "write_fills_for_instrument_day": "fills",
    "write_run_reports_for_cluster_day": "reports",
    # decisions handled specially to derive stage->key
    "write_decisions_for_stage": "__DECISIONS__",
    "write_principles_context_for_cluster_day": "principles_context",
    "write_trade_clusters_for_cluster_day": "trade_clusters",
}


@dataclass
class FoundWrite:
    file: str
    step: str
    fn: str
    key: str


def iter_step_files() -> list[Path]:
    if not STEPS_DIR.exists():
        raise FileNotFoundError(f"steps dir not found: {STEPS_DIR}")
    return sorted(STEPS_DIR.glob("*.py"))


def _call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _extract_decisions_stage(call: ast.Call) -> Optional[str]:
    # Prefer keyword argument stage="..."
    for kw in call.keywords:
        if kw.arg == "stage" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value

    # Fallback: positional stage as 3rd positional arg: write_decisions_for_stage(ctx, trading_day, stage, df, ...)
    if len(call.args) >= 3:
        a = call.args[2]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value

    return None


def scan_writes() -> list[FoundWrite]:
    found: list[FoundWrite] = []

    for path in iter_step_files():
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text, filename=str(path))

        step_name = path.stem  # e.g. "critic_step"
        rel = path.relative_to(PROJECT_ROOT).as_posix()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            fn = _call_name(node)
            if not fn:
                continue

            if fn not in WRITER_FN_TO_KEY:
                continue

            if fn == "write_decisions_for_stage":
                stage = _extract_decisions_stage(node)
                if stage and stage in DECISIONS_STAGE_TO_KEY:
                    key = DECISIONS_STAGE_TO_KEY[stage]
                else:
                    key = "decisions__UNKNOWN_STAGE__"
            else:
                key = WRITER_FN_TO_KEY[fn]

            found.append(FoundWrite(file=rel, step=step_name, fn=fn, key=key))

    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["current", "target"], default="current")
    args = ap.parse_args()
    mode: ContractMode = args.mode  # type: ignore

    issues = validate_contract(mode=mode)
    if issues:
        print("Manifest structural issues:")
        for x in issues:
            print(f"  - {x}")
        print()

    writes = scan_writes()

    actual_owners: dict[str, set[str]] = defaultdict(set)
    for w in writes:
        actual_owners[w.key].add(w.step)

    # Compare expected vs actual
    mismatches: list[str] = []
    missing_writes: list[str] = []
    extra_writes: list[str] = []

    # 1) Expected persisted keys should have exactly one actual owner step (by call-site)
    for key, c in CONTRACT.items():
        if not expected_persisted(c, mode):
            continue

        exp = expected_owner(c, mode)
        act = sorted(actual_owners.get(key, set()))

        if not act:
            missing_writes.append(f"{key}: expected owner={exp} but no writer call-sites found")
            continue

        if exp and (exp not in act or len(act) != 1):
            mismatches.append(f"{key}: expected owner={exp}, actual owners={act}")

    # 2) Any actual writes to unknown keys should be flagged
    for key in sorted(actual_owners.keys()):
        if key not in CONTRACT:
            extra_writes.append(f"{key}: writer call-sites exist but key not in CONTRACT (owners={sorted(actual_owners[key])})")

    # Print report
    print(f"IO Contract Audit (mode={mode})")
    print("-" * 80)
    print("Actual writer call-sites:")
    for k in sorted(actual_owners.keys()):
        print(f"  - {k}: {sorted(actual_owners[k])}")
    print()

    if missing_writes:
        print("Missing writes (expected persisted but no call-sites):")
        for x in missing_writes:
            print(f"  - {x}")
        print()

    if mismatches:
        print("Owner mismatches:")
        for x in mismatches:
            print(f"  - {x}")
        print()

    if extra_writes:
        print("Extra writes (call-sites not represented in CONTRACT):")
        for x in extra_writes:
            print(f"  - {x}")
        print()

    ok = not issues and not missing_writes and not mismatches and not extra_writes
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
