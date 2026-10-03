"""base.py — shared contracts for the Day 3-4 deterministic tool layer.

WHY THIS MODULE EXISTS
  Every tool in this package must produce an *identical* result envelope so that
  the Day 5-7 LangGraph agent (and the Day 8-12 trajectory exporter) can treat
  observations uniformly. Three concerns are centralised here:

  1. Result envelope  — `ToolResult`: a single, always-JSON, fully-populated
     record. Tools never raise into the graph; the graph never parses prose.
     Every result carries a machine-readable `ok` flag plus an `error_kind`, so
     a Reflection loop can distinguish "the model passed a bad argument" from
     "the patch did not apply" from "pytest failed" without regexing text.

  2. Argument validation — a zero-dependency JSON-Schema subset validator. The
     project runs the whole pipeline on a free-tier budget and the host env must
     stay trivial to reproduce (`requirements.txt` = docker only), so pulling in
     the `jsonschema` package is not worth it. This validator is also the guard
     the agent calls *before* executing a model-emitted tool call, which is what
     makes the schema "deterministic" rather than advisory.

  3. Bound enforcement — the hard caps (`MAX_READ_LINES`, `MAX_OUTPUT_CHARS`,
     ...) live here as module constants, and `truncate()` is the single place
     that shortens text. Context bloat is the failure mode this whole project
     exists to fix, so every cap is explicit, global, and testable — never an
     ad-hoc slice inside a tool.

Nothing in this module imports `docker`; it is pure stdlib so both the host
(bridge.py) and the in-container runner (container_runner.py) can rely on the
same semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

# --------------------------------------------------------------------------- #
# Paths & caps (single source of truth, mirrored by container_runner.py)
# --------------------------------------------------------------------------- #

#: Target repository root *inside* the sandbox container. Mirrors
#: src.sandbox.docker_runner.REPO_PATH; duplicated (not imported) so that this
#: stdlib-only module stays importable without the docker SDK.
REPO_PATH = "/workspace/repo"

#: Directory inside the container where the tool runner is shipped to. It lives
#: under /workspace (NOT under /workspace/repo) because it must never appear as a
#: dirty file in `git status` of the target repo — DockerSandbox.reset_repo()
#: runs `git clean -fd`, which would delete it if it lived in the repo.
RUNNER_DIR = "/workspace/.distillcode"
RUNNER_FILENAME = "container_runner.py"
RUNNER_PATH = f"{RUNNER_DIR}/{RUNNER_FILENAME}"

#: Sentinel printed by the container runner immediately before its JSON payload.
#: The host greps for the *last* occurrence, so any incidental stdout produced by
#: a tool (e.g. a stray `print` inside the patched repo) can never corrupt parsing.
JSON_SENTINEL = "<<<DISTILLCODE_JSON>>>"

#: Hard cap on lines returned by read_file_bounded, regardless of the requested
#: range. A 400-line window is roughly 4-6k tokens, i.e. a tool payload that can
#: never blow up a small student model's context on its own.
MAX_READ_LINES = 400

#: Hard caps on the string/text payloads a single tool call may carry.
MAX_PATCH_BYTES = 200_000
MAX_REPLACEMENT_BYTES = 60_000

#: Per-file ceiling for find_symbol_ast. Files above this are skipped (and
#: reported) rather than parsed: `ast.parse` on a multi-MB generated file is the
#: one way a "cheap" symbol lookup turns into a memory spike.
MAX_PARSE_BYTES = 2_000_000

#: Ceiling on how many files find_symbol_ast will walk, as a runaway guard for
#: repos that vendor site-packages into the tree.
MAX_INDEX_FILES = 20_000

#: Cap on each returned stdout/stderr blob from run_pytest_isolated. Raw pytest
#: output for the full requests suite is ~200 KB; returning that verbatim is
#: exactly the "context bloat" the project is built to avoid.
MAX_OUTPUT_CHARS = 20_000

#: Default / maximum wall-clock budgets, in seconds. The sandbox enforces the
#: real kill via coreutils `timeout` inside the container.
DEFAULT_TEST_TIMEOUT = 120
MAX_TEST_TIMEOUT = 600

#: Slack added on top of a tool's internal budget before the *bridge* gives up.
#: The inner budget must always fire first so pytest's own timeout path produces
#: a clean `timeout` envelope carrying partial output, instead of the Docker
#: layer surfacing a context-free `exit_code 124`.
BRIDGE_SLACK_SECONDS = 20

#: pytest flags a `run_pytest_isolated` call may append. Value-taking flags
#: (`-k`, `-p plug`, `--rootdir`) are deliberately excluded: accepting them would
#: mean re-implementing shell argument parsing, which is precisely where command
#: injection and non-determinism creep into an agent that emits these strings.
#: Defined here (not in the runner) because the schema and the executor must
#: agree on the allowlist, and the schema is built on the host.
ALLOWED_PYTEST_EXTRA_ARGS = frozenset(
    {"-q", "-v", "-x", "-s", "--tb=short", "--tb=long", "--tb=line", "--tb=no", "-rf", "-rA"}
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ToolError(RuntimeError):
    """Base class for tool-layer failures raised on the *host* side."""


class ToolArgumentError(ToolError):
    """The caller (usually the LLM) supplied arguments that violate the schema."""


class ToolExecutionError(ToolError):
    """The tool ran but the bridge itself failed (container down, no JSON, ...)."""


# --------------------------------------------------------------------------- #
# Result envelope
# --------------------------------------------------------------------------- #

#: Stable, documented error taxonomy. Day 5-7's Reflection loop branches on
#: these values; adding a new one is a deliberate contract change.
ERROR_KINDS = (
    "invalid_arguments",  # schema violation, caught before any container call
    "path_out_of_repo",  # path escaped REPO_PATH
    "not_found",  # file / target does not exist
    "too_large",  # payload exceeded a cap
    "binary_file",  # read_file_bounded hit a non-text file
    "empty_scope",  # find_symbol_ast found no Python files to index
    "no_match",  # apply_unified_patch: old_str not found
    "ambiguous_match",  # apply_unified_patch: old_str matched more than once
    "patch_rejected",  # git apply --check failed
    "tests_failed",  # pytest exited non-zero with a real failure report
    "test_collection_error",  # pytest could not even collect the target
    "timeout",  # killed by coreutils `timeout`
    "sandbox_error",  # docker bridge failure
    "internal_error",  # anything unexpected
)


@dataclass
class ToolResult:
    """Uniform, always-JSON observation returned by every tool.

    Attributes:
        tool: Tool name, e.g. ``"read_file_bounded"``.
        ok: True only when the tool fully achieved its intent.
        data: Tool-specific success payload (never prose).
        error: Human/LLM-readable failure message; None on success.
        error_kind: One of :data:`ERROR_KINDS`, or None on success.
        meta: Non-essential diagnostics (timings, truncation flags, counts).
    """

    tool: str
    ok: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None
    error_kind: Optional[str] = None
    meta: dict = field(default_factory=dict)

    # -- construction ---------------------------------------------------- #
    @classmethod
    def success(cls, tool: str, data: Mapping[str, Any], **meta: Any) -> "ToolResult":
        return cls(tool=tool, ok=True, data=dict(data), meta=dict(meta))

    @classmethod
    def failure(
        cls, tool: str, error: str, error_kind: str, **meta: Any
    ) -> "ToolResult":
        if error_kind not in ERROR_KINDS:
            raise ValueError(f"unknown error_kind: {error_kind!r}")
        return cls(
            tool=tool,
            ok=False,
            data={},
            error=error,
            error_kind=error_kind,
            meta=dict(meta),
        )

    @classmethod
    def from_runner(cls, tool: str, raw: Mapping[str, Any], **meta: Any) -> "ToolResult":
        """Build a ToolResult from the container runner's JSON payload."""
        merged_meta = dict(raw.get("meta") or {})
        merged_meta.update(meta)
        if raw.get("ok"):
            return cls(
                tool=tool,
                ok=True,
                data=dict(raw.get("data") or {}),
                meta=merged_meta,
            )
        return cls(
            tool=tool,
            ok=False,
            data=dict(raw.get("data") or {}),
            error=raw.get("error") or "unknown container-side failure",
            error_kind=raw.get("error_kind") or "internal_error",
            meta=merged_meta,
        )

    # -- serialisation --------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "error_kind": self.error_kind,
            "meta": self.meta,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialise the envelope. `indent=None` gives the compact wire form."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_tool_message(self) -> str:
        """Compact string form fed back to the model as the `tool` role content.

        Kept as a separate method (rather than reusing `to_json`) because the
        Day 8-12 exporter records this exact string in the trajectory JSONL and
        the Day 17-18 token-efficiency metric is measured on it — so it must be
        stable, compact, and free of pretty-printing.
        """
        return self.to_json()


# --------------------------------------------------------------------------- #
# Truncation helper
# --------------------------------------------------------------------------- #


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """Return ``(text, truncated)`` with ``text`` at most ``limit`` characters.

    Uses a head-biased cut plus an explicit marker rather than a silent slice so
    the model (and the trajectory log) can tell that output was clipped and
    re-issue a narrower call instead of hallucinating the missing tail.
    """
    if text is None:
        return "", False
    if len(text) <= limit:
        return text, False
    dropped = len(text) - limit
    marker = f"\n... [truncated {dropped} chars by DistillCode-Agent] ...\n"
    keep = max(limit - len(marker), 0)
    return text[:keep] + marker, True


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema validator (stdlib only)
# --------------------------------------------------------------------------- #

_TYPE_MAP: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "integer":
        # bool is a subclass of int in Python; JSON has no such promotion.
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    py = _TYPE_MAP.get(expected)
    if py is None:
        raise ValueError(f"unsupported schema type: {expected!r}")
    return isinstance(value, py)


def validate_arguments(
    schema: Mapping[str, Any], arguments: Mapping[str, Any], *, path: str = ""
) -> None:
    """Validate ``arguments`` against a JSON-Schema subset.

    Supported keywords: ``type`` (single type or list), ``required``,
    ``properties``, ``additionalProperties`` (bool), ``enum``, ``minimum``,
    ``maximum``, ``minLength``, ``maxLength``, ``minItems``, ``maxItems``,
    ``items``.

    Raises:
        ToolArgumentError: on the first violation, with a JSON-pointer-ish path
            (``answers[2].text``) so the model can localise its own mistake.
    """
    where = path or "arguments"

    declared = schema.get("type")
    if declared is not None:
        types = declared if isinstance(declared, list) else [declared]
        if not any(_matches_type(arguments, t) for t in types):
            raise ToolArgumentError(
                f"{where}: expected type {declared!r}, got {type(arguments).__name__}"
            )

    if "enum" in schema and arguments not in schema["enum"]:
        raise ToolArgumentError(
            f"{where}: {arguments!r} is not one of {schema['enum']!r}"
        )

    if isinstance(arguments, str):
        if "minLength" in schema and len(arguments) < schema["minLength"]:
            raise ToolArgumentError(
                f"{where}: length {len(arguments)} < minLength {schema['minLength']}"
            )
        if "maxLength" in schema and len(arguments) > schema["maxLength"]:
            raise ToolArgumentError(
                f"{where}: length {len(arguments)} > maxLength {schema['maxLength']}"
            )

    if isinstance(arguments, (int, float)) and not isinstance(arguments, bool):
        if "minimum" in schema and arguments < schema["minimum"]:
            raise ToolArgumentError(
                f"{where}: {arguments} < minimum {schema['minimum']}"
            )
        if "maximum" in schema and arguments > schema["maximum"]:
            raise ToolArgumentError(
                f"{where}: {arguments} > maximum {schema['maximum']}"
            )

    if isinstance(arguments, list):
        if "minItems" in schema and len(arguments) < schema["minItems"]:
            raise ToolArgumentError(
                f"{where}: {len(arguments)} item(s) < minItems {schema['minItems']}"
            )
        if "maxItems" in schema and len(arguments) > schema["maxItems"]:
            raise ToolArgumentError(
                f"{where}: {len(arguments)} item(s) > maxItems {schema['maxItems']}"
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(arguments):
                validate_arguments(item_schema, item, path=f"{where}[{i}]")

    if isinstance(arguments, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required", []) or []:
            if key not in arguments:
                raise ToolArgumentError(f"{where}: missing required property {key!r}")
        for key, value in arguments.items():
            if key in props:
                validate_arguments(props[key], value, path=f"{where}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ToolArgumentError(
                    f"{where}: unexpected property {key!r} "
                    f"(allowed: {sorted(props)})"
                )


# --------------------------------------------------------------------------- #
# Tool specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolSpec:
    """A single tool: its OpenAI function schema plus its host-side plumbing.

    Attributes:
        name: Tool name; must match the container runner's dispatch key.
        description: Model-facing description (kept imperative and bounded-aware:
            it explicitly tells the model to make narrow calls).
        parameters: JSON Schema for the arguments object.
        build_payload: Pure function mapping validated arguments to the request
            payload sent to the container runner. Kept separate from execution so
            request construction can be unit-tested without Docker.
        timeout: Hard wall-clock budget in seconds for the *bridge* call. Tools
            that internally run pytest raise this via `timeout_from`.
        timeout_from: Optional hook deriving the bridge timeout from the
            arguments (used by run_pytest_isolated so the inner `timeout` fires
            first and yields a clean exit code 124 instead of a bridge error).
        error_kind_from: Optional hook letting a tool map a runner `ok: false`
            into a more specific taxonomy entry.
    """

    name: str
    description: str
    parameters: dict
    build_payload: Callable[[dict], dict]
    timeout: int = 30
    timeout_from: Optional[Callable[[dict], int]] = None
    error_kind_from: Optional[Callable[[dict, Mapping[str, Any]], str]] = None

    def openai_schema(self) -> dict:
        """Render the OpenAI / ChatML function-calling tool declaration."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def bridge_timeout(self, arguments: Mapping[str, Any]) -> int:
        if self.timeout_from is not None:
            return self.timeout_from(dict(arguments))
        return self.timeout

    def validate(self, arguments: Mapping[str, Any]) -> None:
        validate_arguments(self.parameters, arguments)
