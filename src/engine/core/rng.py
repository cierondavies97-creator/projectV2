from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from engine.core.ids import MicrobatchKey, RunContext


@dataclass(frozen=True)
class RngStream:
    """
    A deterministic RNG stream derived from (RunContext, MicrobatchKey, component, scope).

    - seed: the derived integer seed (stable across processes/machines)
    - rng : numpy Generator (PCG64) seeded with that seed
    """
    seed: int
    rng: np.random.Generator


def _norm(v: Any) -> str:
    """
    Normalize values for seed material.
    Keep this conservative and stable.
    """
    if v is None:
        return "∅"
    if isinstance(v, (int, float, bool)):
        return str(v)
    return str(v)


def _materialize(parts: list[tuple[str, Any]], scope: Mapping[str, Any] | None) -> str:
    """
    Build a stable seed material string from ordered key/value pairs + an optional scope map.
    The scope map is sorted by key to avoid accidental nondeterminism.
    """
    items: list[str] = [f"{k}={_norm(v)}" for k, v in parts]
    if scope:
        for k in sorted(scope.keys()):
            items.append(f"scope.{k}={_norm(scope[k])}")
    return "|".join(items)


def derive_seed(
    ctx: RunContext,
    key: MicrobatchKey,
    *,
    component: str,
    scope: Mapping[str, Any] | None = None,
    salt: str = "core.v1",
    bits: int = 64,
) -> int:
    """
    Deterministically derive a seed integer from run identity + microbatch key + component.

    Parameters:
      - component: stable string identifier (e.g. "features_step", "windows_step").
      - scope: optional dict to further partition randomness (instrument, anchor_tf, etc.).
      - salt: versioned salt for future-proofing seed evolution.
      - bits: 32 or 64. Defaults to 64 for reduced collision risk.

    Returns:
      Integer seed in range [0, 2^bits - 1].
    """
    if bits not in (32, 64):
        raise ValueError("bits must be 32 or 64")

    material = _materialize(
        parts=[
            ("salt", salt),
            ("env", ctx.env),
            ("mode", ctx.mode),
            ("snapshot_id", ctx.snapshot_id),
            ("run_id", ctx.run_id),
            ("experiment_id", ctx.experiment_id),
            ("candidate_id", ctx.candidate_id),
            ("base_seed", ctx.base_seed),
            ("trading_day", key.trading_day.isoformat()),
            ("cluster_id", key.cluster_id),
            ("component", component),
        ],
        scope=scope,
    )

    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=16).digest()
    nbytes = 8 if bits == 64 else 4
    seed = int.from_bytes(digest[:nbytes], byteorder="big", signed=False)
    return seed


def make_rng(seed: int) -> np.random.Generator:
    """
    Construct a numpy Generator from a seed.
    Keep this centralized so you can swap PRNGs in one place if needed.
    """
    return np.random.Generator(np.random.PCG64(seed))


def rng_stream(
    ctx: RunContext,
    key: MicrobatchKey,
    *,
    component: str,
    scope: Mapping[str, Any] | None = None,
    salt: str = "core.v1",
) -> RngStream:
    """
    Convenience helper returning (seed, rng) for a component/scope.
    """
    seed = derive_seed(ctx, key, component=component, scope=scope, salt=salt, bits=64)
    return RngStream(seed=seed, rng=make_rng(seed))
