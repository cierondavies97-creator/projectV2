"""
Canonical import surface for microbatch orchestration types/entrypoints.

Side-effect free (importing this must not register paradigms).
"""

from __future__ import annotations

from engine.microbatch.pipeline import run_microbatch
from engine.microbatch.types import BatchState

__all__ = [
    "BatchState",
    "run_microbatch",
]
