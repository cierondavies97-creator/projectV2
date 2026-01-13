from __future__ import annotations

"""
Compatibility shim.

Canonical registry lives in engine.paradigms.registry.
This module re-exports the hypotheses builder API to avoid split-brain wiring.
"""

from engine.paradigms.registry import (
    get_hypotheses_builder,
    register_hypotheses_builder,
)

__all__ = [
    "register_hypotheses_builder",
    "get_hypotheses_builder",
]
