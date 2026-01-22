from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl

from engine.core.config_models import build_cluster_plan, load_retail_config
from engine.core.ids import RunContext
from engine.core.schema import get_table_schema
from engine.io.paths import data_root
from engine.microbatch.io_contract import CONTRACT, TableContract
from engine.research.snapshots import load_snapshot_manifest


@dataclass(frozen=True)
class GoldenPolicy:
    golden_tables: tuple[str, ...]
    pk_cols: dict[str, list[str]]
    drop_identity_cols: tuple[str, ...]


def _load_policy(path: Path) -> GoldenPolicy:
    data = json.loads(path.read_text())
    return GoldenPolicy(
        golden_tables=tuple(data.get("golden_tables", [])),
        pk_cols={k: list(v) for k, v in data.get("pk_cols", {}).items()},
        drop_identity_cols=tuple(data.get("drop_identity_cols", [])),
    )


def _cluster_plan(snapshot_id: str, cluster_id: str) -> tuple[list[str], list[str]]:
    snapshot = load_snapshot_manifest(snapshot_id)
    retail = load_retail_config()
    plan = build_cluster_plan(snapshot, retail, cluster_id)
    return list(plan.instruments), list(plan.anchor_tfs)


def _schema_fingerprint(table_key: str) -> str:
    schema = get_table_schema(table_key)
    parts = [f"{col}:{dtype}" for col, dtype in sorted(schema.columns.items())]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest


def _table_root(ctx: RunContext, contract: TableContract, artifacts_root: Path) -> Path:
    return artifacts_root / contract.dataset / f"run_id={ctx.run_id}"


def _partition_match(parts: Iterable[str], *, trading_day: date, stage: str | None, instruments: set[str]) -> bool:
    parts_set = set(parts)
    if f"dt={trading_day.isoformat()}" not in parts_set:
        return False
    if stage and f"stage={stage}" not in parts_set:
        return False
    instrument_parts = {p.split("=", 1)[1] for p in parts if p.startswith("instrument=")}
    return bool(instrument_parts & instruments)


def _collect_parquet_files(
    table_root: Path,
    *,
    trading_day: date,
    stage: str | None,
    instruments: set[str],
) -> list[Path]:
    if not table_root.exists():
        return []
    files = []
    for fp in table_root.rglob("*.parquet"):
        if _partition_match(fp.parts, trading_day=trading_day, stage=stage, instruments=instruments):
            files.append(fp)
    return sorted(files)


def _read_table(files: list[Path]) -> pl.DataFrame:
    if not files:
        return pl.DataFrame()
    frames = [pl.read_parquet(fp) for fp in files]
    if not frames:
        return pl.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical")


def _normalize_for_hash(df: pl.DataFrame, *, drop_cols: tuple[str, ...], pk_cols: list[str] | None) -> pl.DataFrame:
    if df.is_empty():
        return df
    out = df.drop([c for c in drop_cols if c in df.columns], strict=False)
    if pk_cols:
        cols = [c for c in pk_cols if c in out.columns]
        if cols:
            out = out.sort(cols)
    return out


def _content_hash(df: pl.DataFrame) -> str | None:
    if df.is_empty():
        return None
    row_hashes = df.hash_rows().to_list()
    digest = hashlib.sha256()
    for h in row_hashes:
        digest.update(int(h).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _check_pk_invariants(df: pl.DataFrame, pk_cols: list[str]) -> list[str]:
    if df.is_empty() or not pk_cols:
        return []
    missing = [c for c in pk_cols if c not in df.columns]
    if missing:
        return [f"missing_pk_columns={missing}"]
    if df.select([pl.col(c).is_null().any().alias(c) for c in pk_cols]).to_dicts()[0] != {c: False for c in pk_cols}:
        return [f"nulls_in_pk={pk_cols}"]
    dupes = df.select(pk_cols).is_duplicated().any()
    if dupes:
        return [f"duplicate_pk={pk_cols}"]
    return []


def _table_manifest(
    ctx: RunContext,
    contract: TableContract,
    *,
    trading_day: date,
    instruments: set[str],
    artifacts_root: Path,
    policy: GoldenPolicy,
) -> dict[str, Any]:
    stage = None
    if contract.key.startswith("decisions_"):
        stage = contract.key.split("_", 1)[1]

    table_root = _table_root(ctx, contract, artifacts_root)
    files = _collect_parquet_files(table_root, trading_day=trading_day, stage=stage, instruments=instruments)
    df = _read_table(files)

    schema_fp = _schema_fingerprint(contract.key)
    row_count = df.height

    pk_cols = policy.pk_cols.get(contract.key, [])
    invariants = _check_pk_invariants(df, pk_cols)

    result = {
        "table_key": contract.key,
        "dataset": contract.dataset,
        "row_count": row_count,
        "schema_fingerprint": schema_fp,
        "pk_cols": pk_cols,
        "invariants": invariants,
    }

    if contract.key in policy.golden_tables:
        normalized = _normalize_for_hash(df, drop_cols=policy.drop_identity_cols, pk_cols=pk_cols)
        result["content_hash"] = _content_hash(normalized)

    return result


def build_manifest(
    *,
    ctx: RunContext,
    trading_day: date,
    cluster_id: str,
    artifacts_root: Path,
    policy_path: Path,
) -> dict[str, Any]:
    policy = _load_policy(policy_path)
    instruments, _ = _cluster_plan(ctx.snapshot_id, cluster_id)
    instruments_set = set(instruments)

    tables: dict[str, Any] = {}
    for contract in CONTRACT.values():
        if not contract.persisted_current or not contract.dataset:
            continue
        tables[contract.key] = _table_manifest(
            ctx,
            contract,
            trading_day=trading_day,
            instruments=instruments_set,
            artifacts_root=artifacts_root,
            policy=policy,
        )

    return {
        "snapshot_id": ctx.snapshot_id,
        "run_id": ctx.run_id,
        "cluster_id": cluster_id,
        "trading_day": trading_day.isoformat(),
        "policy_path": str(policy_path),
        "tables": tables,
    }


def _write_manifest(manifest: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def bless_manifest(
    *,
    ctx: RunContext,
    trading_day: date,
    cluster_id: str,
    artifacts_root: Path,
    policy_path: Path,
    target: Path,
) -> None:
    policy = _load_policy(policy_path)
    manifest = build_manifest(
        ctx=ctx,
        trading_day=trading_day,
        cluster_id=cluster_id,
        artifacts_root=artifacts_root,
        policy_path=policy_path,
    )
    tables = {
        k: v for k, v in manifest["tables"].items() if k in policy.golden_tables
    }
    manifest["tables"] = tables
    _write_manifest(manifest, target)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden Gate tooling (bless only).")
    sub = parser.add_subparsers(dest="command", required=True)

    bless = sub.add_parser("bless", help="Bless golden manifests from existing artifacts.")
    bless.add_argument("--snapshot-id", required=True)
    bless.add_argument("--run-id", required=True)
    bless.add_argument("--cluster-id", required=True)
    bless.add_argument("--trading-day", required=True)
    bless.add_argument("--policy-path", default="tests/golden/policy.json")
    bless.add_argument("--artifacts-root", default=None)
    bless.add_argument("--target", required=True)

    args = parser.parse_args()

    ctx = RunContext(
        env="research",
        mode="backtest",
        snapshot_id=args.snapshot_id,
        run_id=args.run_id,
        base_seed=0,
    )

    trading_day = _parse_date(args.trading_day)
    policy_path = Path(args.policy_path)
    artifacts_root = Path(args.artifacts_root) if args.artifacts_root else data_root(ctx)
    target = Path(args.target)

    if args.command == "bless":
        bless_manifest(
            ctx=ctx,
            trading_day=trading_day,
            cluster_id=args.cluster_id,
            artifacts_root=artifacts_root,
            policy_path=policy_path,
            target=target,
        )


if __name__ == "__main__":
    main()
