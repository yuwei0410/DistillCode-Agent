"""registry.py — the single dispatch surface for the deterministic tool layer.

This is the seam the Day 5-7 LangGraph nodes bind to. It exists so the graph
never imports four modules, never touches Docker, and never hand-rolls argument
validation:

    tools = openai_tools()                      # bind to the LLM client
    result = dispatch(call.name, call.arguments, sandbox)   # always a ToolResult

Design notes that matter downstream:

* **Validation happens before execution, always.** `dispatch` validates against
  the tool's JSON Schema, then normalises via `build_payload`, and only then talks
  to the container. A model that emits `{"start_line": "5"}` gets an
  `invalid_arguments` envelope in microseconds, with a JSON-pointer-ish path to
  the offending field — that is what makes the Day 17-18 "tool syntax error rate"
  metric a property of the *model* rather than of the runtime.
* **Uniform failure, never a traceback.** `dispatch` returns
  `ToolResult(ok=False, error_kind=...)` for every runtime problem. The graph can
  therefore branch on `error_kind` instead of wrapping calls in try/except, which
  is what keeps the Reflection loop's control flow auditable.
* **`openai_tools()` is deterministic and docker-free.** Schemas are sorted by
  name and the OpenAI tool list is byte-stable across runs: a stable prompt
  prefix is required both for reproducible token accounting and for KV-cache
  reuse when benchmarking (Day 17-18). The `docker` SDK is imported lazily inside
  `dispatch`, so exporting schemas works on a machine with no Docker daemon.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from . import base as _base
from .apply_unified_patch import SPEC as APPLY_SPEC
from .find_symbol_ast import SPEC as FIND_SPEC
from .read_file_bounded import SPEC as READ_SPEC
from .run_pytest_isolated import SPEC as PYTEST_SPEC

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module docker-free
    from ..sandbox.docker_runner import DockerSandbox

#: Tool name -> spec. Insertion of a duplicate name is a programming error, and
#: the assertion below turns it into an import-time failure rather than a silently
#: shadowed tool.
TOOL_REGISTRY: dict[str, _base.ToolSpec] = {
    spec.name: spec for spec in (READ_SPEC, FIND_SPEC, APPLY_SPEC, PYTEST_SPEC)
}
assert len(TOOL_REGISTRY) == 4, f"duplicate tool names: {sorted(TOOL_REGISTRY)}"

#: Arguments may arrive as a mapping (native SDK) or as a raw JSON string (every
#: OpenAI-compatible endpoint returns the latter).
ArgumentsInput = Union[Mapping[str, Any], str, None]


def tool_names() -> list[str]:
    """Registered tool names in deterministic order."""
    return sorted(TOOL_REGISTRY)


def openai_tools() -> list[dict]:
    """OpenAI / ChatML function-calling declarations, sorted by name."""
    return [TOOL_REGISTRY[name].openai_schema() for name in tool_names()]


def parse_arguments(raw: ArgumentsInput) -> dict:
    """Accept a dict, a JSON object string, or None (meaning "no arguments").

    Raises:
        base.ToolArgumentError: on non-object JSON or unparseable text.
    """
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _base.ToolArgumentError(
                f"tool arguments are not valid JSON: {exc.msg} at position {exc.pos}"
            ) from exc
        if not isinstance(parsed, dict):
            raise _base.ToolArgumentError(
                f"tool arguments must be a JSON object, got {type(parsed).__name__}"
            )
        return parsed
    raise _base.ToolArgumentError(
        f"tool arguments must be an object or JSON string, got {type(raw).__name__}"
    )


def validate_call(name: str, raw: ArgumentsInput) -> Optional[str]:
    """Validate a call without executing it.

    Returns None when the call is well-formed, else the failure message. This is
    the hook the Day 17-18 harness uses to score "tool syntax / argument parse
    error rate" separately from repair success, so a schema violation is never
    mis-counted as a failed fix.
    """
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return f"unknown tool {name!r}; available: {tool_names()}"
    try:
        spec.validate(parse_arguments(raw))
        spec.build_payload(parse_arguments(raw))
    except _base.ToolArgumentError as exc:
        return str(exc)
    return None


def dispatch(
    name: str,
    raw_arguments: ArgumentsInput,
    sandbox: "Optional[DockerSandbox]" = None,
    *,
    repo_path: str = _base.REPO_PATH,
    ensure_runner: bool = True,
) -> _base.ToolResult:
    """Validate and execute one tool call. Never raises for runtime failures.

    Args:
        name: Registered tool name as emitted by the model.
        raw_arguments: Parsed arguments dict, a JSON string, or None.
        sandbox: A live `DockerSandbox`. When None, a default sandbox is created
            on first use (convenient for smoke scripts and manual poking); the
            Day 5-7 graph always passes its own long-lived instance so container
            startup is not paid per tool call.
        repo_path: Repo root inside the container the tools operate on.
        ensure_runner: Ship/refresh the in-container runner before calling.

    Returns:
        base.ToolResult — `ok=True` on success, otherwise `error_kind` identifies
        the failure class (see base.ERROR_KINDS).
    """
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return _base.ToolResult.failure(
            name or "<missing>",
            f"unknown tool {name!r}; available: {tool_names()}",
            "invalid_arguments",
        )

    # Step 1 — schema validation and normalisation, on the host, before any I/O.
    try:
        arguments = parse_arguments(raw_arguments)
        spec.validate(arguments)
        payload = spec.build_payload(arguments)
    except _base.ToolArgumentError as exc:
        return _base.ToolResult.failure(name, str(exc), "invalid_arguments")

    # Step 2 — resolve the sandbox. Imported lazily so that merely exporting
    # schemas (openai_tools) never requires the docker SDK.
    if sandbox is None:
        try:
            from ..sandbox.docker_runner import DockerSandbox

            sandbox = DockerSandbox()
        except Exception as exc:  # noqa: BLE001 - surfaced as an envelope
            return _base.ToolResult.failure(
                name, f"could not create a sandbox: {exc}", "sandbox_error"
            )

    # Step 3 — execute inside the container.
    from . import bridge

    result = bridge.invoke(
        sandbox,
        name,
        payload,
        timeout=spec.bridge_timeout(payload),
        repo_path=repo_path,
        ensure=ensure_runner,
    )

    # Step 4 — optional per-tool refinement of the taxonomy entry.
    if not result.ok and spec.error_kind_from is not None:
        refined = spec.error_kind_from(payload, result.to_dict())
        if refined and refined != result.error_kind and refined in _base.ERROR_KINDS:
            result.error_kind = refined
    return result
