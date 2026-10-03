"""DistillCode-Agent source package.

Phase 1 layout:
  src/sandbox/  — Day 1-2: Docker execution bridge (DockerSandbox).
  src/tools/    — Day 3-4: deterministic, bounded tool schemas exposed to the LLM.

The source tree is a real package (not a loose script folder) because the tool
layer needs intra-package imports that must behave identically whether a module
is run as `python -m src.tools.smoke` or imported by the Day 5-7 LangGraph nodes.
"""
