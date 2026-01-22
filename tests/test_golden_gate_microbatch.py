from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from engine.core.ids import MicrobatchKey, RunContext  # noqa: E402
from engine.io.paths import data_root  # noqa: E402
from engine.microbatch.steps import features_step, ingest_step, windows_step  # noqa: E402
from engine.microbatch.types import BatchState  # noqa: E402
from tools.golden import build_manifest  # noqa: E402


def _cleanup_run_dirs(ctx: RunContext) -> None:
    root = data_root(ctx)
    for dataset in (
        "features",
        "windows",
        "decisions",
        "trade_paths",
        "brackets",
        "orders",
        "fills",
        "reports",
        "trade_clusters",
        "principles_context",
    ):
        run_dir = root / dataset / f"run_id={ctx.run_id}"
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


def test_golden_gate_microbatch_determinism() -> None:
    snapshot_id = "dev_retail_v1"
    trading_day = date(2025, 1, 2)
    cluster_id = "metals"
    policy_path = ROOT / "tests" / "golden" / "policy.json"

    ctx_a = RunContext(
        env="research",
        mode="backtest",
        snapshot_id=snapshot_id,
        run_id="golden_tmp_a",
        base_seed=1234,
    )
    ctx_b = RunContext(
        env="research",
        mode="backtest",
        snapshot_id=snapshot_id,
        run_id="golden_tmp_b",
        base_seed=1234,
    )
    key = MicrobatchKey(trading_day=trading_day, cluster_id=cluster_id)

    try:
        state_a = BatchState(ctx=ctx_a, key=key)
        state_a = ingest_step.run(state_a)
        state_a = features_step.run(state_a)
        state_a = windows_step.run(state_a)

        state_b = BatchState(ctx=ctx_b, key=key)
        state_b = ingest_step.run(state_b)
        state_b = features_step.run(state_b)
        state_b = windows_step.run(state_b)

        manifest_a = build_manifest(
            ctx=ctx_a,
            trading_day=trading_day,
            cluster_id=cluster_id,
            artifacts_root=data_root(ctx_a),
            policy_path=policy_path,
        )
        manifest_b = build_manifest(
            ctx=ctx_b,
            trading_day=trading_day,
            cluster_id=cluster_id,
            artifacts_root=data_root(ctx_b),
            policy_path=policy_path,
        )

        tables_a = manifest_a["tables"]
        tables_b = manifest_b["tables"]
        assert set(tables_a.keys()) == set(tables_b.keys())

        for table_key, entry_a in tables_a.items():
            entry_b = tables_b[table_key]
            assert entry_a["schema_fingerprint"] == entry_b["schema_fingerprint"]
            assert entry_a["row_count"] == entry_b["row_count"]
            assert entry_a["invariants"] == entry_b["invariants"]
            if "content_hash" in entry_a:
                assert entry_a["content_hash"] == entry_b["content_hash"]

        blessed_path = ROOT / "tests" / "golden" / snapshot_id / cluster_id / trading_day.isoformat() / "manifest.json"
        blessed = json.loads(blessed_path.read_text())
        blessed_tables = blessed["tables"]
        for table_key, blessed_entry in blessed_tables.items():
            assert tables_a[table_key]["content_hash"] == blessed_entry["content_hash"]
    finally:
        _cleanup_run_dirs(ctx_a)
        _cleanup_run_dirs(ctx_b)
