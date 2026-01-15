from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from engine.microbatch.io_contract import (
    ContractMode,
    CONTRACT,
    expected_owner,
    expected_persisted,
)


@dataclass(frozen=True)
class ContractWrite:
    table_key: str
    writer_fn: str
    stage: str | None = None


def assert_contract_alignment(
    *,
    step_name: str,
    writes: Iterable[ContractWrite],
    mode: ContractMode = "current",
) -> None:
    issues: list[str] = []

    for write in writes:
        contract = CONTRACT.get(write.table_key)
        if contract is None:
            issues.append(f"{write.table_key}: missing CONTRACT entry")
            continue

        expected = expected_owner(contract, mode)
        if expected != step_name:
            issues.append(
                f"{write.table_key}: owner_step_{mode}={expected!r} does not match step={step_name!r}"
            )

        if not expected_persisted(contract, mode):
            issues.append(f"{write.table_key}: persisted_{mode}=False but step attempts to write")

        if contract.writer_fn != write.writer_fn:
            issues.append(
                f"{write.table_key}: writer_fn mismatch (contract={contract.writer_fn!r}, actual={write.writer_fn!r})"
            )

        if contract.writer_fn == "write_decisions_for_stage":
            if write.stage is None:
                issues.append(f"{write.table_key}: stage is required for write_decisions_for_stage")
            elif contract.stage != write.stage:
                issues.append(
                    f"{write.table_key}: stage mismatch (contract={contract.stage!r}, actual={write.stage!r})"
                )

    if issues:
        formatted = "\n  - ".join(issues)
        raise ValueError(
            "Contract drift detected for step "
            f"{step_name!r} (mode={mode}).\n  - {formatted}"
        )
