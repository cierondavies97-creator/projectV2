from __future__ import annotations

import ast
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"

WRITER_FUNCS = {
    "write_windows_for_instrument_tf_day": "data/windows",
    "write_features_for_instrument_tf_day": "data/features",
    "write_zones_state_for_instrument_tf_day": "data/zones_state",
    "write_pcra_for_instrument_tf_day": "data/pcr_a",
    "write_decisions_for_stage": "data/decisions",
    "write_trade_paths_for_day": "data/trade_paths",
    "write_brackets_for_instrument_day": "data/brackets",
    "write_orders_for_instrument_day": "data/orders",
    "write_fills_for_instrument_day": "data/fills",
    "write_run_reports_for_cluster_day": "data/run_reports",
    "write_parquet": "(raw write_parquet)",
}

def iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        # Skip junk if needed
        if any(part in {"__pycache__", ".venv", "venv"} for part in p.parts):
            continue
        yield p

class CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call):
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name:
            self.calls.append(name)
        self.generic_visit(node)

def main():
    owners = defaultdict(list)  # writer -> [file paths]

    for path in iter_py_files(SRC):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception:
            continue

        v = CallVisitor()
        v.visit(tree)

        for c in v.calls:
            if c in WRITER_FUNCS:
                owners[c].append(path.relative_to(PROJECT).as_posix())

    # Print grouped results
    for writer in sorted(owners.keys()):
        print(f"\n=== {writer} -> {WRITER_FUNCS[writer]} ===")
        for f in sorted(set(owners[writer])):
            print(f"  - {f}")

if __name__ == "__main__":
    main()
