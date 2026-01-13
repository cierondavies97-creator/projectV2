"""
Canonical import surface for paradigm registry access + explicit registration.

This module is side-effect free on import. Registration occurs only when
register_all_paradigms() is called (typically via engine.bootstrap.bootstrap_engine()).
"""

from __future__ import annotations

from engine.paradigms.registry import get_critic, get_hypotheses_builder


def register_all_paradigms() -> None:
    """
    Explicit registration entrypoint.

    This function triggers registration without relying on import-time side effects.
    Keeping the import inside the function makes registration explicit and easy to audit.
    """
    from engine.paradigms.register_all import register_all_paradigms as _register_all

    _register_all()


__all__ = [
    "get_hypotheses_builder",
    "get_critic",
    "register_all_paradigms",
]
