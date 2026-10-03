"""run_pytest_isolated — targeted test execution inside the sandbox container.

WHY "ISOLATED" IS PART OF THE NAME
  The naive alternative — `pytest tests/` after every edit — is the single
  fastest way to destroy a small model's context budget: the requests suite is
  ~200 cases, prints tens of thousands of characters on failure, and takes well
  over the 30s sandbox cap. This tool therefore requires an explicit, narrow
  target list (node ids like `tests/test_utils.py::TestUtils::test_x`), refuses
  to run without one, and truncates what it returns.

  Concretely, the contract the agent can rely on:
  * Targets must name existing paths inside the repo; a typo fails fast with
    `not_found` rather than after a full collection pass.
  * Only a fixed allowlist of pytest flags is accepted, so no model-emitted
    string can smuggle a shell metacharacter or an env-mutating flag.
  * `-p no:cacheprovider` is always passed: `.pytest_cache` must never appear in
    the working tree, or `reset_repo()` and patch generation stop being clean.
  * stdout/stderr are each capped at `base.MAX_OUTPUT_CHARS` with an explicit
    truncation marker, and the failure list is parsed into structured
    `failed_tests` entries — regex-extractable truth instead of prose.

  What is deliberately NOT done here: traceback compaction. The container returns
  exit code + (bounded) stdout/stderr as the plan specifies; the deterministic
  traceback rewriter is `traceback_cleaner.py` in Day 5-7, where it can be
  unit-tested against captured raw output instead of being entangled with
  execution. `summary` / `failed_tests` are extracted here only because they are
  cheap, deterministic, and already line-oriented.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import base as _base
from .base import ToolArgumentError, ToolSpec

NAME = "run_pytest_isolated"

DESCRIPTION = (
    "Run specific pytest targets inside the isolated Docker sandbox. Pass narrow "
    "node ids (e.g. 'tests/test_utils.py::TestUtils::test_from_key_value_list') "
    "rather than whole directories. Returns exit_code, duration, a parsed "
    "summary/failed_tests listing, and bounded stdout/stderr. Look at "
    "failed_tests[].message for the assertion, not at the raw log."
)

PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "targets": {
            "type": ["array", "string"],
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
            "minItems": 1,
            "maxItems": 20,
            "maxLength": 400,
            "description": (
                "One or more pytest targets: a path ('tests/test_utils.py'), a node id "
                "('tests/test_utils.py::TestUtils::test_x'), or a target class. "
                "A single string is accepted as a one-element list."
            ),
        },
        "timeout": {
            "type": "integer",
            "minimum": 5,
            "maximum": _base.MAX_TEST_TIMEOUT,
            "description": (
                f"Wall-clock budget in seconds (default {_base.DEFAULT_TEST_TIMEOUT}, "
                f"max {_base.MAX_TEST_TIMEOUT}). On expiry you get error_kind='timeout'."
            ),
        },
        "extra_args": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(_base.ALLOWED_PYTEST_EXTRA_ARGS)},
            "maxItems": 6,
            "description": (
                "Optional pytest flags, allowlisted. '-q' is always applied and "
                "'-x' stops at the first failure."
            ),
        },
    },
    "required": ["targets"],
    "additionalProperties": False,
}


def build_payload(arguments: Mapping[str, Any]) -> dict:
    """Normalise targets into a list and freeze a canonical request."""
    raw_targets = arguments.get("targets")
    if isinstance(raw_targets, str):
        targets = [raw_targets]
    elif isinstance(raw_targets, list):
        targets = list(raw_targets)
    else:
        raise ToolArgumentError("'targets' must be a string or an array of strings")

    targets = [str(t).strip() for t in targets if str(t).strip()]
    if not targets:
        raise ToolArgumentError("'targets' must contain at least one non-empty target")
    if len(targets) > 20:
        raise ToolArgumentError("at most 20 targets per invocation")

    extra_args = list(dict.fromkeys(str(a) for a in (arguments.get("extra_args") or [])))
    illegal = [a for a in extra_args if a not in _base.ALLOWED_PYTEST_EXTRA_ARGS]
    if illegal:
        raise ToolArgumentError(
            f"unsupported extra_args {illegal}; allowed: {sorted(_base.ALLOWED_PYTEST_EXTRA_ARGS)}"
        )

    timeout = int(arguments.get("timeout") or _base.DEFAULT_TEST_TIMEOUT)
    if timeout < 5 or timeout > _base.MAX_TEST_TIMEOUT:
        raise ToolArgumentError(
            f"timeout must be within [5, {_base.MAX_TEST_TIMEOUT}], got {timeout}"
        )

    return {"targets": targets, "timeout": timeout, "extra_args": extra_args}


def _bridge_timeout(arguments: Mapping[str, Any]) -> int:
    """Bridge budget = inner pytest budget + slack (see base.BRIDGE_SLACK_SECONDS).

    Reads the *normalised* payload, so `timeout` is always materialised by the
    time this runs. Slack exists so the inner `subprocess.run(timeout=...)` is
    the one that expires: that path returns partial pytest output, whereas the
    outer coreutils kill returns nothing useful.
    """
    inner = int(arguments.get("timeout") or _base.DEFAULT_TEST_TIMEOUT)
    return inner + _base.BRIDGE_SLACK_SECONDS


SPEC = ToolSpec(
    name=NAME,
    description=DESCRIPTION,
    parameters=PARAMETERS,
    build_payload=build_payload,
    timeout=_base.DEFAULT_TEST_TIMEOUT,
    timeout_from=_bridge_timeout,
)
