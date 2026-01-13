from __future__ import annotations

from engine.paradigms.registry import register_critic, register_hypotheses_builder

_REGISTERED = False


def register_all_paradigms() -> None:
    """
    Register all paradigm implementations into the global registries.

    Contract:
      - Idempotent: safe to call multiple times within the same process.
      - No import-time side effects: this module must NOT auto-register on import.
      - Deterministic: registration order and keys are stable.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # ICT (Phase A)
    from engine.paradigms.ict.critic import score_trades
    from engine.paradigms.ict.hypotheses import build_hypotheses_for_windows

    register_hypotheses_builder("ict", "ict_all_windows", build_hypotheses_for_windows)
    register_critic("ict", score_trades)

    _REGISTERED = True
