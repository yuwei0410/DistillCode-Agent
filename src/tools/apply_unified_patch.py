"""apply_unified_patch — the only way the agent is allowed to mutate source.

WHY TWO MODES
  Small open-weight models are empirically much better at "emit the old lines and
  the new lines" than at "emit a byte-exact unified diff with correct hunk
  headers". Forcing one representation would therefore fail for reasons that have
  nothing to do with the actual bug — which would poison the Day 8-12 dataset with
  failures the student can never learn from. Both modes are exposed:

  * ``mode="replace"`` — exact `old_str` → `new_str`. This is the default choice
    and the intended one for the distillation dataset: it is diff-free, its
    failure mode (`no_match` / `ambiguous_match`) is *recoverable* and reported
    with line numbers, and it maps cleanly onto the read → patch → test loop.
  * ``mode="diff"`` — a real unified diff, validated with ``git apply --check``
    before a single byte is written.

DETERMINISM & SAFETY GUARANTEES
  * Nothing is written unless it validates. Diff mode probes ``--check`` first;
    replace mode counts matches and refuses ambiguous or missing anchors.
  * Prefix depth is auto-detected (``-p0/-p1/-p2/-p3``) and the level actually
    used is reported as ``strip_used``, so the model never has to guess whether
    the diff was generated with ``a/``/``b/`` prefixes.
  * Every mutation reports ``changed_files`` (from ``git diff --numstat``) and a
    truncated ``diffstat``, giving the loop an independent verification channel
    instead of trusting its own patch.
  * ``dry_run=true`` performs the full validation without touching disk.
  * Paths are rejected before any syscall if absolute or containing `..`, and the
    container re-checks the same invariant (defence in depth).
"""

from __future__ import annotations

from typing import Any, Mapping

from . import base as _base
from .base import ToolArgumentError, ToolSpec

NAME = "apply_unified_patch"

DESCRIPTION = (
    "Apply a code change to the target repository. Two modes: 'replace' (default) "
    "takes an exact old_str and new_str copied verbatim from read_file_bounded "
    "output; 'diff' applies a real unified diff after a git apply --check dry run. "
    "Nothing is written if validation fails. Always inspect changed_files and "
    "diffstat in the response, then re-run the affected tests."
)

PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["replace", "diff"],
            "description": (
                "'replace' for old_str/new_str edits (preferred); 'diff' for a "
                "unified diff in `patch`. Inferred from the supplied fields if omitted."
            ),
        },
        "path": {
            "type": "string",
            "maxLength": 400,
            "description": "mode=replace only. Repository-relative file to edit.",
        },
        "old_str": {
            "type": "string",
            "maxLength": _base.MAX_REPLACEMENT_BYTES,
            "description": (
                "mode=replace only. Exact existing text, copied verbatim including "
                "indentation. Must occur exactly once unless replace_all=true."
            ),
        },
        "new_str": {
            "type": "string",
            "maxLength": _base.MAX_REPLACEMENT_BYTES,
            "description": "mode=replace only. Replacement text. Use \"\" to delete old_str.",
        },
        "replace_all": {
            "type": "boolean",
            "description": "mode=replace only. Allow old_str to match more than once. Default false.",
        },
        "patch": {
            "type": "string",
            "maxLength": _base.MAX_PATCH_BYTES,
            "description": "mode=diff only. Unified diff text applied with git apply.",
        },
        "strip": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
            "description": (
                "mode=diff only. Explicit git -p<N> level. Omit it: the level is "
                "auto-detected deterministically from the patch headers "
                "(a/ b/ prefixes -> 1, otherwise 0) and reported as strip_used. "
                "Never supply this unless you know the diff uses a non-standard "
                "prefix depth."
            ),
        },
        "dry_run": {
            "type": "boolean",
            "description": "Validate without writing anything. Default false.",
        },
    },
    "required": [],
    "additionalProperties": False,
}


def _infer_mode(arguments: Mapping[str, Any]) -> str:
    mode = arguments.get("mode")
    if mode:
        return str(mode)
    if arguments.get("patch"):
        return "diff"
    if arguments.get("path") and "old_str" in arguments:
        return "replace"
    raise ToolArgumentError(
        "cannot infer mode: pass mode='replace' with path/old_str/new_str, "
        "or mode='diff' with patch"
    )


def build_payload(arguments: Mapping[str, Any]) -> dict:
    """Normalise the mode and enforce the cross-field contract.

    Size caps are re-checked here (not only in the container) so an oversized
    payload is rejected in microseconds instead of after a Docker round-trip —
    the model gets its correction while the mistake is still in context.
    """
    mode = _infer_mode(arguments)
    payload: dict[str, Any] = {"mode": mode}

    if mode == "diff":
        patch = arguments.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            raise ToolArgumentError("mode='diff' requires a non-empty 'patch'")
        if len(patch.encode("utf-8")) > _base.MAX_PATCH_BYTES:
            raise ToolArgumentError(
                f"patch exceeds MAX_PATCH_BYTES ({_base.MAX_PATCH_BYTES}); "
                f"split it into smaller hunks"
            )
        strip = arguments.get("strip")
        if strip is not None:
            strip = int(strip)
            if strip not in (0, 1, 2, 3):
                raise ToolArgumentError("strip must be one of 0, 1, 2, 3")
            payload["strip"] = strip
        payload.update(
            {"patch": patch, "dry_run": bool(arguments.get("dry_run", False))}
        )
        return payload

    if mode != "replace":
        raise ToolArgumentError(f"unsupported mode: {mode!r}")

    path = arguments.get("path")
    old_str = arguments.get("old_str")
    new_str = arguments.get("new_str")
    if not isinstance(path, str) or not path.strip():
        raise ToolArgumentError("mode='replace' requires 'path'")
    for label, value in (("old_str", old_str), ("new_str", new_str)):
        if not isinstance(value, str):
            raise ToolArgumentError(f"mode='replace' requires '{label}' to be a string")
        if len(value.encode("utf-8")) > _base.MAX_REPLACEMENT_BYTES:
            raise ToolArgumentError(
                f"{label} exceeds MAX_REPLACEMENT_BYTES ({_base.MAX_REPLACEMENT_BYTES})"
            )
    if not old_str:
        raise ToolArgumentError("'old_str' must not be empty")
    if old_str == new_str:
        raise ToolArgumentError("'old_str' and 'new_str' are identical: nothing to change")

    payload.update(
        {
            "path": path,
            "old_str": old_str,
            "new_str": new_str,
            "replace_all": bool(arguments.get("replace_all", False)),
            "dry_run": bool(arguments.get("dry_run", False)),
        }
    )
    return payload


SPEC = ToolSpec(
    name=NAME,
    description=DESCRIPTION,
    parameters=PARAMETERS,
    build_payload=build_payload,
    timeout=60,
)
