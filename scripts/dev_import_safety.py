from __future__ import annotations

import importlib


def main() -> None:
    # Barrels must import cleanly and be side-effect free.
    for mod in (
        "engine.api",
        "engine.core.api",
        "engine.io.api",
        "engine.microbatch.api",
        "engine.paradigms.api",
        "engine.data.api",
    ):
        importlib.import_module(mod)

    # Pipeline import should not trigger paradigm registration at import time.
    from engine.microbatch.api import run_microbatch  # noqa: F401

    # Paradigm registration must be explicit.
    from engine.paradigms.api import get_hypotheses_builder, register_all_paradigms

    register_all_paradigms()
    b = get_hypotheses_builder("ict", "ict_all_windows")
    assert callable(b), "builder must be callable"

    print("OK dev_import_safety")


if __name__ == "__main__":
    main()
