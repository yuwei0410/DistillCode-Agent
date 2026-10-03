"""Tool layer package (Phase 1 / Day 3-4) — deterministic, bounded tool schemas.

Deliberately NOT re-exporting the individual tools here: importing this package
must not pull in `docker` (the host SDK) as a side effect, so that the JSON
schemas in `registry.py` can be inspected/unit-tested on a machine without a
running Docker daemon. Import the concrete modules instead:

    from src.tools.registry import TOOL_REGISTRY, openai_tools
"""
