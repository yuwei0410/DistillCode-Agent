"""read_file_bounded — line-range file reads with hard, declared caps.

WHY THIS TOOL REPLACES A PLAIN `read_file`
  Context bloat is the primary failure mode of long agent runs, and it is caused
  almost entirely by full-file dumps: a single `requests/models.py` read is ~4k
  tokens, and a 4-iteration repair loop that re-reads four files per iteration
  can spend the student model's whole window on code that never needed changing.

  So this tool cannot express "read the whole file": `path` is required, and the
  returned window is the intersection of the caller's `[start_line, end_line]`
  range and a hard `max_lines` window (capped at `base.MAX_READ_LINES`, 400).
  When the cap bites, the result carries `has_more`, `next_start_line`, and an
  explicit `hint` in `meta` — the model is told exactly how to page forward,
  which is what keeps the loop from guessing or hallucinating the tail.

  Returned lines are 1-indexed with a `NNNNN | ` gutter when
  `with_line_numbers=True` (the default). That gutter is not cosmetic: it is the
  anchor the model quotes back to `apply_unified_patch` (`old_str`), and it makes
  off-by-one patch failures visible instead of mysterious.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import base as _base
from .base import ToolArgumentError, ToolSpec

NAME = "read_file_bounded"

DESCRIPTION = (
    "Read a bounded, line-numbered window from a single file inside the target "
    "repository. Always pass start_line/end_line for surgical reads; the tool "
    "never returns more than 400 lines and tells you (has_more / next_start_line) "
    "how to page forward. Use find_symbol_ast first to locate the definition, "
    "then read only the region you intend to patch."
)

PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "minLength": 1,
            "maxLength": 400,
            "description": "Repository-relative path, e.g. 'requests/models.py'.",
        },
        "start_line": {
            "type": "integer",
            "minimum": 1,
            "description": "First line to return, 1-indexed and inclusive. Default 1.",
        },
        "end_line": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Last line to return, inclusive. Defaults to start_line + "
                "max_lines - 1. Clamped to the max_lines window."
            ),
        },
        "max_lines": {
            "type": "integer",
            "minimum": 1,
            "maximum": _base.MAX_READ_LINES,
            "description": (
                f"Size of the returned window (default 200, hard ceiling "
                f"{_base.MAX_READ_LINES})."
            ),
        },
        "with_line_numbers": {
            "type": "boolean",
            "description": "Prefix each line with a 'NNNNN | ' gutter. Default true.",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


def build_payload(arguments: Mapping[str, Any]) -> dict:
    """Pass-through with the two cross-field checks the schema cannot express."""
    payload = dict(arguments)
    start = payload.get("start_line")
    end = payload.get("end_line")
    if start is not None and end is not None and int(end) < int(start):
        raise ToolArgumentError(
            f"arguments.end_line ({end}) must be >= arguments.start_line ({start})"
        )
    return payload


SPEC = ToolSpec(
    name=NAME,
    description=DESCRIPTION,
    parameters=PARAMETERS,
    build_payload=build_payload,
    timeout=30,
)
