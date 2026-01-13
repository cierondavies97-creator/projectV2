from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import polars as pl


class RunContextLike(Protocol):
    """
    Minimal context surface required by paradigm callables.

    Keep intentionally small to avoid over-coupling paradigms to the full engine context.
    """
    snapshot_id: str
    run_id: str
    mode: str


HypothesesBuilder = Callable[
    # (ctx, windows_df, features_df, params) -> (hypotheses_df, trade_paths_df)
    [RunContextLike, pl.DataFrame, pl.DataFrame, Mapping[str, Any]],
    tuple[pl.DataFrame, pl.DataFrame],
]

CriticFn = Callable[
    # (ctx, hypotheses_df, trade_paths_df, params) -> scored df (or critic report df)
    [RunContextLike, pl.DataFrame, pl.DataFrame, Mapping[str, Any]],
    pl.DataFrame,
]


@dataclass(frozen=True)
class RegisteredHypothesesBuilder:
    paradigm_id: str
    principle_id: str
    fn: HypothesesBuilder


@dataclass(frozen=True)
class RegisteredCritic:
    paradigm_id: str
    fn: CriticFn


_HYP_BUILDERS: dict[tuple[str, str], HypothesesBuilder] = {}
_CRITICS: dict[str, CriticFn] = {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_hypotheses_builder(
    paradigm_id: str,
    principle_id: str,
    fn: HypothesesBuilder,
    *,
    overwrite: bool = False,
) -> None:
    """
    Register a hypotheses builder.

    overwrite=False (default) is safer for reproducibility: it prevents accidental
    replacement due to import order or duplicated registration calls.
    """
    key = (paradigm_id, principle_id)
    if not overwrite and key in _HYP_BUILDERS:
        raise KeyError(
            "Hypotheses builder already registered for "
            f"paradigm_id={paradigm_id} principle_id={principle_id}. "
            "Pass overwrite=True only if you intentionally want to replace it."
        )
    _HYP_BUILDERS[key] = fn


def register_critic(
    paradigm_id: str,
    fn: CriticFn,
    *,
    overwrite: bool = False,
) -> None:
    """
    Register a critic function.

    overwrite=False (default) prevents accidental replacement.
    """
    if not overwrite and paradigm_id in _CRITICS:
        raise KeyError(
            f"Critic already registered for paradigm_id={paradigm_id}. "
            "Pass overwrite=True only if you intentionally want to replace it."
        )
    _CRITICS[paradigm_id] = fn


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_hypotheses_builder(paradigm_id: str, principle_id: str) -> HypothesesBuilder:
    """
    Resolve a hypotheses builder by (paradigm_id, principle_id).

    Resolution order:
      1) (paradigm_id, principle_id)
      2) (paradigm_id, "*")
      3) ("*", "*")

    Raises:
      KeyError if nothing matches.
    """
    b = _HYP_BUILDERS.get((paradigm_id, principle_id))
    if b is not None:
        return b

    b = _HYP_BUILDERS.get((paradigm_id, "*"))
    if b is not None:
        return b

    b = _HYP_BUILDERS.get(("*", "*"))
    if b is not None:
        return b

    raise KeyError(
        "No hypotheses builder registered for "
        f"paradigm_id={paradigm_id} principle_id={principle_id}. "
        f"Registered keys={sorted(_HYP_BUILDERS.keys())!r}"
    )


def get_critic(paradigm_id: str) -> CriticFn:
    """
    Resolve a critic function by paradigm_id.

    Raises:
      KeyError if not registered.
    """
    fn = _CRITICS.get(paradigm_id)
    if fn is None:
        raise KeyError(
            f"No critic registered for paradigm_id={paradigm_id}. "
            f"Registered paradigms={sorted(_CRITICS.keys())!r}"
        )
    return fn


# ---------------------------------------------------------------------------
# Introspection (debuggability / tests)
# ---------------------------------------------------------------------------

def list_hypotheses_builders() -> list[RegisteredHypothesesBuilder]:
    """
    Return a stable, sorted list of all registered hypotheses builders.
    """
    items = [
        RegisteredHypothesesBuilder(k[0], k[1], fn)
        for k, fn in _HYP_BUILDERS.items()
    ]
    return sorted(items, key=lambda x: (x.paradigm_id, x.principle_id))


def list_critics() -> list[RegisteredCritic]:
    """
    Return a stable, sorted list of all registered critics.
    """
    items = [RegisteredCritic(pid, fn) for pid, fn in _CRITICS.items()]
    return sorted(items, key=lambda x: x.paradigm_id)


def clear_registry() -> None:
    """
    Clear all registrations.

    Intended for tests only. Do not call from production engine code.
    """
    _HYP_BUILDERS.clear()
    _CRITICS.clear()
