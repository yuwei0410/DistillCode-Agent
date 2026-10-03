"""find_symbol_ast — repository-wide symbol lookup via the `ast` module.

WHY AST INSTEAD OF grep / ripgrep
  Three reasons, all of which show up as measurable failures without this tool:

  1. Signal density. `grep -rn "def get"` returns comments, strings, `mock.patch`
     targets and vendored copies. `ast` returns actual definitions — name, kind,
     qualname, exact line span — so the model can jump straight to a bounded read
     instead of burning an iteration triaging grep noise.
  2. Memory. Files are parsed one at a time and never retained: the walk yields
     paths, `ast.parse` builds a tree, symbols are extracted, and the tree is
     dropped. Only the flat `symbols` list survives, and even that aborts early
     once `max_results` is reached. A repo-wide index this cheap is what makes
     "look before you read" affordable on every iteration.
  3. Failure honesty. Files that do not parse are reported in `parse_errors`, and
     files skipped for size in `skipped_large_files` — rather than silently
     returning "not found", which would send the agent down a wrong path.

  Note the deliberate omission of `include_source`: returning definitions *and*
  their bodies would re-introduce the full-file dump this project exists to
  remove. Line numbers are returned instead; the model then calls
  `read_file_bounded` on the exact span it cares about.
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import ToolSpec

NAME = "find_symbol_ast"

DESCRIPTION = (
    "Locate class, function and method definitions across the repository using "
    "Python's AST (no false positives from comments or strings). Returns file, "
    "qualname, kind and the exact lineno/end_lineno span for each match, so you "
    "can follow up with a bounded read of just that region. Prefer this over "
    "grepping or reading files blindly."
)

PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
            "description": (
                "Symbol name to match. Substring match (case-insensitive) by "
                "default, matched against both the short name and the dotted "
                "qualname. Omit to list every symbol in scope."
            ),
        },
        "kind": {
            "type": "string",
            "enum": ["any", "function", "class", "method"],
            "description": (
                "'method' means a function defined directly in a class body; "
                "'function' covers module-level and nested functions, including "
                "`async def`. Default 'any'."
            ),
        },
        "scope": {
            "type": "string",
            "maxLength": 400,
            "description": (
                "Repository-relative subdirectory to index, or a single .py file "
                "to index just that module. Default '.' (whole repo)."
            ),
        },
        "exact": {
            "type": "boolean",
            "description": "Require an exact match on the name or qualname. Default false.",
        },
        "include_signature": {
            "type": "boolean",
            "description": "Include the reconstructed signature (and base classes for classes). Default true.",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 200,
            "description": "Stop after this many matches (default 50).",
        },
    },
    "additionalProperties": False,
}


def build_payload(arguments: Mapping[str, Any]) -> dict:
    """Normalise defaults on the host so trajectories record what actually ran."""
    return {
        "name": arguments.get("name"),
        "kind": arguments.get("kind", "any"),
        "scope": arguments.get("scope") or ".",
        "exact": bool(arguments.get("exact", False)),
        "include_signature": bool(arguments.get("include_signature", True)),
        "max_results": int(arguments.get("max_results", 50)),
    }


SPEC = ToolSpec(
    name=NAME,
    description=DESCRIPTION,
    parameters=PARAMETERS,
    build_payload=build_payload,
    timeout=60,
)
