from __future__ import annotations

_BOOTSTRAPPED = False


def bootstrap_engine() -> None:
    """
    One-time process bootstrap for deterministic lane + tooling.

    Design goals:
      - Side-effect free on import (only runs when called).
      - Safe to call multiple times (idempotent).
      - Explicit: all registration is triggered by this function, not by incidental imports.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # Explicit paradigm registration entrypoint.
    from engine.paradigms.api import register_all_paradigms

    register_all_paradigms()

    _BOOTSTRAPPED = True
