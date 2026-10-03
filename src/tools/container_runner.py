"""container_runner.py — stdlib-only tool executor that runs INSIDE the sandbox.

WHY THIS FILE LIVES IN THE REPO BUT EXECUTES IN THE CONTAINER
  Day 3-4 tools must observe the *same* filesystem the tests will run against,
  otherwise agents "fix" the host checkout while pytest grades the baked image.
  Rather than mount the repo to the host (state drift) or shell out with
  `sed`/`grep` (no AST, no validation), the host uploads this single script plus
  `base.py` into /workspace/.distillcode/ and drives it with:

      python /workspace/.distillcode/container_runner.py \\
          --tool read_file_bounded --payload <base64-json>

  The runner prints `JSON_SENTINEL` followed by exactly one JSON envelope on
  stdout, so a stray `print()` from the patched repository can never corrupt the
  tool protocol. It always exits 0 when it produced an envelope: failure is
  carried *in* the envelope (`ok: false`, `error_kind`), which keeps the Docker
  layer dumb and the Reflection loop's branch conditions simple.

  Hard requirements enforced by construction:
    * stdlib only  — the image is `python:3.11-slim` + the target repo's deps;
      nothing may be pip-installed for tooling.
    * bounded I/O  — every read/write is capped by a constant in `base.py`, and
      file contents are never fully materialised for symbol search.
    * path safety  — every caller-supplied path is resolved and proven to be
      inside the repository root before any syscall touches it.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Iterable, Iterator, Optional

# Two import shapes, one file:
#   * host  — `python -m src.tools.container_runner` (smoke tests, linting)
#   * container — shipped standalone next to base.py, imported as `base`
try:  # pragma: no cover - depends on execution context
    from . import base as _base
except ImportError:  # pragma: no cover - container path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import base as _base  # type: ignore


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPO = _base.REPO_PATH
DEFAULT_WINDOW = 200  # lines returned when the caller gives no explicit range

#: Directories never worth index: VCS internals, caches, vendored deps, build
#: output. Skipping them is what makes find_symbol_ast cheap on a repo like
#: requests, whose tree contains no vendored site-packages but whose .git and
#: __pycache__ directories are still far larger than the source itself.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    }
)

#: Re-exported alias so the allowlist has exactly one definition (base.py) while
#: staying readable at the call site. The host schema is built from the same set.
ALLOWED_EXTRA_ARGS = _base.ALLOWED_PYTEST_EXTRA_ARGS

#: pytest reserves these exit codes; we translate them into the error taxonomy
#: so the Reflection loop knows whether to fix code or fix the test selector.
PYTEST_EXIT_KINDS = {
    1: "tests_failed",
    2: "tests_failed",  # interrupted (e.g. Ctrl-C) — tests did not all pass
    3: "internal_error",
    4: "test_collection_error",
    5: "test_collection_error",  # no tests collected: bad target/selector
}


class ToolFault(Exception):
    """A tool-level failure that maps onto an envelope error_kind.

    `data` is optional and lets a tool hand back a *partial* observation
    alongside the failure (run_pytest_isolated returns the captured pytest
    stdout/stderr for a failing or timed-out run). The Reflection loop cannot
    reason about a red test suite without that output, so dropping it on the
    error path would silently degrade every repair iteration.
    """

    def __init__(self, message: str, kind: str, data: Optional[dict] = None, **meta: Any) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind if kind in _base.ERROR_KINDS else "internal_error"
        self.data = data
        self.meta = meta


# --------------------------------------------------------------------------- #
# Envelope helpers
# --------------------------------------------------------------------------- #


def _ok(tool: str, data: dict, **meta: Any) -> dict:
    return {"tool": tool, "ok": True, "data": data, "error": None, "error_kind": None, "meta": meta}


def _fail(tool: str, message: str, kind: str, data: Optional[dict] = None, **meta: Any) -> dict:
    return {
        "tool": tool,
        "ok": False,
        "data": data or {},
        "error": message,
        "error_kind": kind,
        "meta": meta,
    }


# --------------------------------------------------------------------------- #
# Filesystem / git primitives
# --------------------------------------------------------------------------- #


def _resolve(repo_root: str, rel_path: Any) -> str:
    """Resolve `rel_path` inside `repo_root`, rejecting anything that escapes.

    Rejects absolute paths, Windows drive letters, and any `..` segment *before*
    touching the filesystem, then re-checks the realpath (symlink-safe) after.
    """
    raw = str(rel_path).replace("\\", "/").strip()
    if not raw:
        raise ToolFault("path must not be empty", "invalid_arguments")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ToolFault(f"absolute paths are not allowed: {raw!r}", "path_out_of_repo")
    if any(part == ".." for part in raw.split("/")):
        raise ToolFault(f"'..' segments are not allowed: {raw!r}", "path_out_of_repo")

    root = os.path.realpath(repo_root)
    resolved = os.path.realpath(os.path.join(root, raw))
    if resolved != root and not resolved.startswith(root + os.sep):
        raise ToolFault(f"path escapes the repository: {raw!r}", "path_out_of_repo")
    return resolved


def _rel(repo_root: str, abs_path: str) -> str:
    return os.path.relpath(abs_path, os.path.realpath(repo_root)).replace(os.sep, "/")


def _repo_snapshot(repo_root: str, limit: int = 25) -> str:
    """Short listing of the repository root, attached to `not_found` messages.

    A bare "file not found: requests/__version__.py" is unactionable for the model
    *and* for the operator: it cannot distinguish a typo from a damaged checkout.
    Echoing what the root actually contains turns the failure into a diagnosis —
    this is precisely how a container missing `requests/` while still holding
    `tests/` (so pytest silently imported the PyPI `requests` from site-packages
    and reported green) became visible instead of looking like twelve unrelated
    path bugs.
    """
    try:
        entries = sorted(os.listdir(repo_root))
    except OSError as exc:  # pragma: no cover - defensive
        return f"<unlistable: {exc}>"
    head = ", ".join(entries[:limit])
    if len(entries) > limit:
        head += f", ... (+{len(entries) - limit} more)"
    return head or "<empty directory>"


def _run(
    argv: list[str], *, cwd: str, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Run argv with captured output; never raises on non-zero exit."""
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def _git(repo_root: str, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return _run(["git", "-C", repo_root, *args], cwd=repo_root, timeout=timeout)


def _count_lines(path: str) -> int:
    """Count newline-terminated lines without loading the file into memory."""
    count = 0
    last = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":  # final line without trailing newline
        count += 1
    return count


def _looks_binary(path: str, probe: int = 8192) -> bool:
    with open(path, "rb") as fh:
        return b"\x00" in fh.read(probe)


def _detect_newline(raw: bytes) -> str:
    """The file's dominant line ending, as the exact string to write back."""
    return "\r\n" if b"\r\n" in raw else "\n"


def _restore_newlines(text: str, newline: str) -> str:
    """Convert LF-normalised text back to the target file's convention.

    Normalising first makes this idempotent for model-supplied `\\r\\n` in
    `new_str`, which would otherwise be rewritten into a `\\r\\r\\n`.
    """
    if newline == "\n":
        return text
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def _assert_in_repo(path: str) -> None:
    if os.path.isabs(path) or any(p == ".." for p in path.split("/")):
        raise ToolFault(
            f"working tree contains a path outside the repo: {path!r}", "path_out_of_repo"
        )


def _repo_state(repo_root: str) -> dict[str, dict]:
    """Snapshot the working tree as {path: {added, removed, untracked}}.

    Uncommitted changes are *accumulated* across the several edits a task needs,
    so `git diff --numstat` alone cannot answer "what did THIS call change?".
    Snapshotting before/after and diffing the two snapshots can, and that
    difference is what the agent reads back as its verification signal.

    Untracked files are folded in from `git status` because `numstat` ignores
    them entirely — a diff that *creates* a file would otherwise report no change
    at all, which is the most confusing possible answer.
    """
    state: dict[str, dict] = {}
    for line in _git(repo_root, "diff", "--numstat").stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, removed, path = parts
        _assert_in_repo(path)
        state[path] = {
            "added": None if added == "-" else int(added),
            "removed": None if removed == "-" else int(removed),
            "untracked": False,
        }
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    for line in status.stdout.splitlines():
        if line.startswith("?? "):
            path = line[3:].strip().strip('"')
            _assert_in_repo(path)
            state[path] = {"added": None, "removed": None, "untracked": True}
    return state


def _state_delta(before: dict[str, dict], after: dict[str, dict]) -> list[dict]:
    """Rows describing only the paths whose state differs between snapshots."""
    rows: list[dict] = []
    for path in sorted(set(before) | set(after)):
        prev, now = before.get(path), after.get(path)
        if prev == now:
            continue
        if now is None:
            rows.append({"path": path, "status": "reverted"})
        else:
            rows.append(
                {
                    "path": path,
                    "status": "created" if now.get("untracked") else "modified",
                    "added_lines": now.get("added"),
                    "removed_lines": now.get("removed"),
                }
            )
    return rows


def _diffstat_scoped(repo_root: str, paths: list[str], limit: int = 4000) -> dict:
    """`git diff --stat` limited to the paths this call touched (possibly none)."""
    if not paths:
        return {"text": "", "truncated": False}
    cp = _git(repo_root, "diff", "--stat", "--", *paths)
    text, clipped = _base.truncate(cp.stdout.strip(), limit)
    return {"text": text, "truncated": clipped}


def _verify_changes(repo_root: str, before: dict[str, dict]) -> tuple[list[dict], dict]:
    """Read back the effect of an applied change, scoped to this call."""
    changed = _state_delta(before, _repo_state(repo_root))
    touched = [row["path"] for row in changed if row["status"] != "reverted"]
    return changed, _diffstat_scoped(repo_root, touched)


# --------------------------------------------------------------------------- #
# Tool 1 — read_file_bounded
# --------------------------------------------------------------------------- #


def read_file_bounded(repo_root: str, payload: dict) -> tuple[dict, dict]:
    rel_path = payload["path"]
    start = int(payload.get("start_line", 1))
    end = int(payload.get("end_line", start + DEFAULT_WINDOW - 1))
    window = int(payload.get("max_lines", DEFAULT_WINDOW))
    numbered = bool(payload.get("with_line_numbers", True))

    if start < 1:
        raise ToolFault(f"start_line must be >= 1, got {start}", "invalid_arguments")
    if end < start:
        raise ToolFault(f"end_line ({end}) must be >= start_line ({start})", "invalid_arguments")
    if window < 1:
        raise ToolFault(f"max_lines must be >= 1, got {window}", "invalid_arguments")

    window = min(window, _base.MAX_READ_LINES)

    abs_path = _resolve(repo_root, rel_path)
    if not os.path.exists(abs_path):
        raise ToolFault(
            f"file not found: {rel_path} (repository root contains: "
            f"{_repo_snapshot(repo_root)})",
            "not_found",
        )
    if os.path.isdir(abs_path):
        raise ToolFault(f"{rel_path} is a directory, not a file", "invalid_arguments")
    if _looks_binary(abs_path):
        raise ToolFault(f"{rel_path} looks like a binary file", "binary_file")

    # The returned window is the intersection of what the caller asked for and
    # what the caps allow. `capped` tells the model its request was clipped, so
    # it re-issues a narrower read instead of assuming the file ends there.
    window_end = min(end, start + window - 1)
    capped = window_end < end

    collected: list[tuple[int, str]] = []
    with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            if lineno > window_end:
                break
            if lineno >= start:
                collected.append((lineno, raw.rstrip("\n").rstrip("\r")))

    total_lines = _count_lines(abs_path)
    if collected:
        first_line, last_line = collected[0][0], collected[-1][0]
    else:
        # Requested range starts past EOF: a successful read of nothing. Reported
        # explicitly (rather than as an error) so the agent can re-anchor.
        first_line = last_line = 0

    body = "\n".join(f"{n:>5} | {text}" if numbered else text for n, text in collected)
    # Anchoring past EOF is not "more to come" — reporting has_more there would
    # send the model paging in a circle at line 1.
    has_more = bool(collected) and last_line < total_lines

    data = {
        "path": _rel(repo_root, abs_path),
        "requested": {"start_line": start, "end_line": end},
        "first_line": first_line,
        "last_line": last_line,
        "total_lines": total_lines,
        "returned_lines": len(collected),
        "has_more": has_more,
        "next_start_line": last_line + 1 if has_more else None,
        "with_line_numbers": numbered,
        "content": body,
    }
    if not collected:
        data["note"] = (
            f"start_line {start} is past EOF (file has {total_lines} lines)"
        )

    meta = {"capped_by_max_lines": capped, "max_lines_applied": window}
    if capped:
        meta["hint"] = (
            f"requested end_line {end} exceeded the {window}-line window; "
            f"call again with start_line={window_end + 1} for the next chunk"
        )
    return data, meta


# --------------------------------------------------------------------------- #
# Tool 2 — find_symbol_ast
# --------------------------------------------------------------------------- #


def _iter_py_files(root: str) -> Iterator[str]:
    """Yield .py files under `root`, pruning SKIP_DIRS in-place for speed."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _node_name(node: Any) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is very robust
        return "?"


def _signature(node: Any) -> Optional[str]:
    try:
        # `ast.unparse(node.args)` yields the bare parameter list WITHOUT the
        # surrounding parentheses ("self, url, **kwargs"), so they are re-added
        # here — otherwise the reported signature reads "def prepare_urlself, ..."
        # and is worthless to the model.
        args = f"({ast.unparse(node.args)})"
        returns = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}{args}{returns}"
    except Exception:  # pragma: no cover
        return None


def _collect_symbols(tree: ast.AST, rel_path: str, include_signature: bool) -> list[dict]:
    """Walk a parsed module, emitting class/function/method records.

    `parent_kind` is threaded explicitly so a closure nested inside a method is
    classified as a `function` (not a `method`) — a distinction the model needs
    when deciding whether `self` is in scope.
    """
    out: list[dict] = []

    def visit(node: ast.AST, parent_kind: str, qual_parts: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qual = ".".join(qual_parts + [child.name])
                record = {
                    "file": rel_path,
                    "name": child.name,
                    "qualname": qual,
                    "kind": "class",
                    "lineno": child.lineno,
                    "end_lineno": getattr(child, "end_lineno", None),
                    "decorators": [_node_name(d) for d in child.decorator_list],
                    "doc": (ast.get_docstring(child) or "").splitlines()[:1] or None,
                    "bases": [_node_name(b) for b in child.bases],
                }
                # Keep the `signature` key present-but-null when disabled: every
                # symbol record has the same shape, so a model that learned the
                # schema never sees a key silently vanish.
                record["signature"] = (
                    f"class {child.name}({', '.join(record['bases'])})"
                    if include_signature
                    else None
                )
                out.append(record)
                visit(child, "class", qual_parts + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent_kind == "class" else "function"
                qual = ".".join(qual_parts + [child.name])
                record = {
                    "file": rel_path,
                    "name": child.name,
                    "qualname": qual,
                    "kind": kind,
                    "lineno": child.lineno,
                    "end_lineno": getattr(child, "end_lineno", None),
                    "decorators": [_node_name(d) for d in child.decorator_list],
                    "doc": (ast.get_docstring(child) or "").splitlines()[:1] or None,
                    "async": isinstance(child, ast.AsyncFunctionDef),
                    "signature": _signature(child) if include_signature else None,
                }
                out.append(record)
                visit(child, kind, qual_parts + [child.name])
            else:
                visit(child, parent_kind, qual_parts)

    visit(tree, "module", [])
    return out


def _symbol_matches(sym: dict, name: Optional[str], kind: str, exact: bool) -> bool:
    if kind != "any":
        if kind == "function" and sym["kind"] != "function":
            return False
        if kind == "class" and sym["kind"] != "class":
            return False
        if kind == "method" and sym["kind"] != "method":
            return False
    if name:
        haystacks = (sym["name"], sym["qualname"])
        if exact:
            return name in haystacks
        needle = name.lower()
        return any(needle in h.lower() for h in haystacks)
    return True


def find_symbol_ast(repo_root: str, payload: dict) -> tuple[dict, dict]:
    name: Optional[str] = payload.get("name") or None
    kind = payload.get("kind", "any")
    scope = payload.get("scope") or "."
    exact = bool(payload.get("exact", False))
    include_signature = bool(payload.get("include_signature", True))
    max_results = min(int(payload.get("max_results", 50)), 200)
    if max_results < 1:
        raise ToolFault("max_results must be >= 1", "invalid_arguments")
    if kind not in ("any", "function", "class", "method"):
        raise ToolFault(f"unsupported kind: {kind!r}", "invalid_arguments")

    search_root = _resolve(repo_root, scope)
    if os.path.isfile(search_root):
        # A single-file scope is a legitimate and much cheaper call ("index just
        # this module"), and it is what the agent naturally asks for after
        # read_file_bounded has already pointed it at one file.
        candidates: Iterable[str] = [search_root]
    elif os.path.isdir(search_root):
        candidates = _iter_py_files(search_root)
    else:
        raise ToolFault(
            f"scope does not exist: {scope} (repository root contains: "
            f"{_repo_snapshot(repo_root)})",
            "not_found",
        )

    symbols: list[dict] = []
    scanned = 0
    skipped_large: list[str] = []
    parse_errors: list[dict] = []
    hit_index_cap = False
    hit_result_cap = False

    for abs_file in candidates:
        if scanned >= _base.MAX_INDEX_FILES:
            hit_index_cap = True
            break
        scanned += 1
        rel_file = _rel(repo_root, abs_file)

        if os.path.getsize(abs_file) > _base.MAX_PARSE_BYTES:
            if len(skipped_large) < 20:
                skipped_large.append(rel_file)
            continue
        try:
            with open(abs_file, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
            tree = ast.parse(source, filename=rel_file)
        except (SyntaxError, ValueError) as exc:
            parse_errors.append(
                {"file": rel_file, "line": getattr(exc, "lineno", None), "error": str(exc)[:200]}
            )
            continue

        for sym in _collect_symbols(tree, rel_file, include_signature):
            if not _symbol_matches(sym, name, kind, exact):
                continue
            symbols.append(sym)
            if len(symbols) >= max_results:
                hit_result_cap = True
                break
        if hit_result_cap:
            break

    symbols.sort(key=lambda s: (s["file"], s["lineno"]))
    if not scanned:
        raise ToolFault(f"no Python files found under scope {scope!r}", "empty_scope")

    data = {
        "query": {"name": name, "kind": kind, "exact": exact, "scope": scope},
        "count": len(symbols),
        "symbols": symbols,
        "truncated": hit_result_cap or hit_index_cap,
        "scanned_files": scanned,
        "skipped_large_files": skipped_large,
        "parse_errors": parse_errors,
    }
    meta: dict[str, Any] = {"max_results_applied": max_results}
    if hit_result_cap:
        meta["hint"] = "more symbols matched; narrow `name`/`scope` or raise max_results"
    if hit_index_cap:
        meta["hint"] = f"file index stopped at MAX_INDEX_FILES={_base.MAX_INDEX_FILES}"
    return data, meta


# --------------------------------------------------------------------------- #
# Tool 3 — apply_unified_patch
# --------------------------------------------------------------------------- #


def _header_paths(patch_text: str) -> list[str]:
    """Raw target paths from `---`/`+++` header pairs, before any stripping."""
    lines = patch_text.splitlines()
    paths: list[str] = []
    for index, line in enumerate(lines[:-1]):
        if not line.startswith("--- "):
            continue
        following = lines[index + 1]
        if not following.startswith("+++ "):
            continue
        for raw in (line[4:], following[4:]):
            raw = raw.split("\t")[0].strip()  # plain diffs append a timestamp
            if not raw or raw == "/dev/null":
                continue
            paths.append(raw)
    return paths


def _detect_strip(patch_text: str) -> int:
    """Deterministic strip level from the header prefixes.

    git-style diffs prefix both sides with a/ and b/ → -p1. Anything else
    (hand-written, plain `diff -u`, or already-stripped headers) is applied
    verbatim with -p0. One rule, no probing: probing multi-level candidates is
    what let a re-applied create-file patch "succeed" at -p2 by dropping the
    leading components and landing the file at the repository root.
    """
    paths = _header_paths(patch_text)
    if paths and all("/" in p and p.split("/", 1)[0] in ("a", "b") for p in paths):
        return 1
    return 0


def _derived_paths(patch_text: str, strip: int) -> list[str]:
    """File paths a patch would touch, derived from its `---`/`+++` header pairs.

    Headers are recognised as an *adjacent* `---`/`+++` pair rather than by
    matching any line that starts with dashes: a removed source line whose content
    begins with `-- ` renders as `--- ...` inside a hunk, and mistaking that for a
    file header would mis-derive the target paths and therefore the strip level.

    An empty list means "this level is not applicable" (it would strip the entire
    path), which is a hard reject rather than a fallback trigger.
    """
    lines = patch_text.splitlines()
    paths: list[str] = []
    for index, line in enumerate(lines[:-1]):
        if not line.startswith("--- "):
            continue
        following = lines[index + 1]
        if not following.startswith("+++ "):
            continue
        for raw in (line[4:], following[4:]):
            raw = raw.split("\t")[0].strip()  # plain diffs append a timestamp
            if not raw or raw == "/dev/null":
                continue
            parts = raw.split("/")
            if strip >= len(parts):
                return []
            paths.append("/".join(parts[strip:]))
    return paths


def _paths_are_sane(repo_root: str, paths: list[str]) -> Optional[str]:
    """Reject a strip level that would land the patch somewhere nonsensical.

    This is what keeps the automatic `-p0/-p1/-p2/-p3` probing safe. Blind probing
    is genuinely dangerous: for a create-file patch (`--- /dev/null` / `+++ b/x`)
    that has *already* been applied, `--check` fails at `-p1` and then **succeeds**
    at `-p0`, which makes `git apply` create a new file at `b/x` — a stray file in
    a directory the model never named, and one that `reset_repo()` may not even
    report as the cause. Requiring every derived path's parent directory to exist
    (or the path itself to exist) rejects that level while leaving legitimate
    new-file creation inside a real directory fully supported.
    """
    if not paths:
        return "patch declares no usable file paths"
    root = os.path.realpath(repo_root)
    for path in paths:
        if path.startswith("/") or any(seg == ".." for seg in path.split("/")):
            return f"unsafe path {path!r}"
        if os.path.exists(os.path.join(root, path)):
            continue
        parent = os.path.dirname(path) or "."
        if not os.path.isdir(os.path.join(root, parent)):
            return f"{path!r} targets missing directory {parent!r} (wrong strip level?)"
    return None


def _apply_diff(repo_root: str, payload: dict) -> tuple[dict, dict]:
    patch_text = payload.get("patch") or ""
    if not patch_text.strip():
        raise ToolFault("patch must not be empty", "invalid_arguments")
    encoded = patch_text.encode("utf-8")
    if len(encoded) > _base.MAX_PATCH_BYTES:
        raise ToolFault(
            f"patch is {len(encoded)} bytes > MAX_PATCH_BYTES {_base.MAX_PATCH_BYTES}",
            "too_large",
        )

    explicit_strip = payload.get("strip")
    if explicit_strip is not None:
        explicit_strip = int(explicit_strip)
        if explicit_strip not in (0, 1, 2, 3):
            raise ToolFault("strip must be one of 0,1,2,3", "invalid_arguments")
    dry_run = bool(payload.get("dry_run", False))

    # The strip level is either an explicit override or derived DETERMINISTICALLY
    # from the header prefixes — it is never probed. Probing fell through to -p2
    # for a re-applied create-file patch and silently created the file at the
    # repository root, and earlier to -p0 creating stray b/... directories. One
    # level, one answer, or a rejection carrying the exact git error.
    chosen = explicit_strip if explicit_strip is not None else _detect_strip(patch_text)

    # `gettempdir()` resolves to /tmp inside the container (the deployed case) and
    # to the OS temp dir anywhere else, which keeps this code path exercisable in
    # host-side testing instead of failing on a hardcoded POSIX path.
    fd, patch_path = tempfile.mkstemp(
        prefix="distillcode_", suffix=".patch", dir=tempfile.gettempdir()
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(patch_text)

        # Path sanity runs before any git call: it is pure string work, and a
        # level that would write outside the intended tree is rejected without
        # touching the filesystem.
        derived = _derived_paths(patch_text, chosen)
        reason = _paths_are_sane(repo_root, derived)
        if reason:
            raise ToolFault(
                f"patch rejected: unsafe target paths at strip level {chosen}: {reason}",
                "patch_rejected",
                check_errors=[{"strip": chosen, "stderr": f"unsafe target paths: {reason}"}],
            )
        cp = _git(repo_root, "apply", "--check", f"-p{chosen}", "--whitespace=nowarn", patch_path)
        if cp.returncode != 0:
            raise ToolFault(
                f"git apply --check rejected the patch at strip level {chosen}: "
                f"{cp.stderr.strip()[:500]}",
                "patch_rejected",
                check_errors=[{"strip": chosen, "stderr": cp.stderr.strip()[:500]}],
            )
        chosen_paths = derived

        strip_mode = "explicit" if explicit_strip is not None else "auto"
        result = {
            "mode": "diff",
            "dry_run": dry_run,
            "applied": False,
            "strip_requested": explicit_strip if explicit_strip is not None else "auto",
            "strip_used": chosen,
            "target_paths": chosen_paths,
        }
        if dry_run:
            result["changed_files"] = []
            result["diffstat"] = {"text": "", "truncated": False}
            return result, {"strip_mode": strip_mode}

        before = _repo_state(repo_root)
        cp = _git(repo_root, "apply", f"-p{chosen}", "--whitespace=nowarn", patch_path)
        if cp.returncode != 0:
            raise ToolFault(
                f"git apply failed after a successful --check: {cp.stderr.strip()[:500]}",
                "patch_rejected",
            )

        result["applied"] = True
        result["changed_files"], result["diffstat"] = _verify_changes(repo_root, before)
        return result, {
            "strip_mode": strip_mode,
            "git_stderr": cp.stderr.strip()[:500],
        }
    finally:
        try:
            os.unlink(patch_path)
        except OSError:  # pragma: no cover
            pass


def _apply_replacement(repo_root: str, payload: dict) -> tuple[dict, dict]:
    old_str = payload.get("old_str")
    new_str = payload.get("new_str")
    replace_all = bool(payload.get("replace_all", False))
    dry_run = bool(payload.get("dry_run", False))

    if not isinstance(old_str, str) or not isinstance(new_str, str):
        raise ToolFault("old_str and new_str must be strings", "invalid_arguments")
    if not old_str:
        raise ToolFault("old_str must not be empty", "invalid_arguments")
    if old_str == new_str:
        raise ToolFault("old_str and new_str are identical: nothing to change", "invalid_arguments")
    for label, value in (("old_str", old_str), ("new_str", new_str)):
        if len(value.encode("utf-8")) > _base.MAX_REPLACEMENT_BYTES:
            raise ToolFault(
                f"{label} exceeds MAX_REPLACEMENT_BYTES {_base.MAX_REPLACEMENT_BYTES}",
                "too_large",
            )

    abs_path = _resolve(repo_root, payload["path"])
    if not os.path.exists(abs_path):
        raise ToolFault(
            f"file not found: {payload['path']} (repository root contains: "
            f"{_repo_snapshot(repo_root)})",
            "not_found",
        )

    with open(abs_path, "rb") as fh:
        raw = fh.read()
    if b"\x00" in raw[:8192]:
        raise ToolFault(f"{payload['path']} looks like a binary file", "binary_file")

    newline = _detect_newline(raw)
    # Matching runs on the LF-normalised view — byte-for-byte what
    # read_file_bounded showed the model, so a quoted `old_str` still matches —
    # while the write converts back to the file's own convention. Without this,
    # editing one line of a CRLF checkout (Windows host via host_repo_path, or
    # `core.autocrlf=true`) rewrote EVERY line ending in the file: git then
    # reported the file modified with an empty diff, and the agent's verification
    # signal became unusable.
    original = raw.decode("utf-8", "replace").replace("\r\n", "\n")

    # The exact count is cheap (`str.count` is a C-level linear scan). Deriving a
    # line number per match is NOT: it used to re-scan the prefix for every hit,
    # which is quadratic and turns a 1-character old_str on a large file into a
    # multi-second stall that can blow the bridge timeout. Only the first 20
    # locations are ever reported, so the scan stops there and walks the newline
    # counter forward incrementally instead of restarting it.
    occurrences = original.count(old_str)
    if occurrences == 0:
        # Whitespace drift is the single most common cause; say so explicitly so
        # the model re-reads the region instead of blindly retrying the same patch.
        raise ToolFault(
            "old_str was not found in the file (check indentation/whitespace "
            "against read_file_bounded output)",
            "no_match",
        )

    match_lines: list[int] = []
    if occurrences > 1:
        newline_cursor = 0
        for hit in re.finditer(re.escape(old_str), original):
            newline_cursor = original.count("\n", newline_cursor, hit.start())
            match_lines.append(newline_cursor + 1)
            if len(match_lines) == 20:
                break

    if occurrences > 1 and not replace_all:
        raise ToolFault(
            f"old_str matched {occurrences} times (first at lines {match_lines}); "
            f"pass replace_all=true or include more context",
            "ambiguous_match",
            match_lines=match_lines,
        )

    replacements = occurrences
    updated = original.replace(old_str, new_str)
    before = _repo_state(repo_root) if not dry_run else {}
    if not dry_run:
        # Write via a sibling temp file + os.replace so an interrupted write can
        # never leave a half-patched source file in the repo. st_mode is copied
        # because git records the executable bit and a mode change would show up
        # as a spurious diff.
        mode = os.stat(abs_path).st_mode
        fd, tmp_path = tempfile.mkstemp(prefix=".distillcode_", dir=os.path.dirname(abs_path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write(_restore_newlines(updated, newline))
            os.chmod(tmp_path, mode)
            os.replace(tmp_path, abs_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover
                pass
            raise

    data = {
        "mode": "replace",
        "dry_run": dry_run,
        "applied": not dry_run,
        "path": _rel(repo_root, abs_path),
        "replacements": replacements,
        "match_lines": match_lines,
        "changed_files": [],
        "diffstat": {"text": "", "truncated": False},
    }
    if not dry_run:
        data["changed_files"], data["diffstat"] = _verify_changes(repo_root, before)
    return data, {"file_lines": original.count("\n") + 1}


def apply_unified_patch(repo_root: str, payload: dict) -> tuple[dict, dict]:
    mode = payload.get("mode") or ("diff" if payload.get("patch") else "replace")
    if mode == "diff":
        return _apply_diff(repo_root, payload)
    if mode == "replace":
        return _apply_replacement(repo_root, payload)
    raise ToolFault(f"unsupported mode: {mode!r}", "invalid_arguments")


# --------------------------------------------------------------------------- #
# Tool 4 — run_pytest_isolated
# --------------------------------------------------------------------------- #


def _validate_target(repo_root: str, target: str) -> None:
    if not target or target.startswith("-"):
        raise ToolFault(f"invalid pytest target: {target!r}", "invalid_arguments")
    file_part = target.split("::", 1)[0]
    if not file_part:
        raise ToolFault(f"invalid pytest target: {target!r}", "invalid_arguments")
    abs_path = _resolve(repo_root, file_part)
    if not os.path.exists(abs_path):
        raise ToolFault(
            f"test target does not exist: {file_part} (repository root contains: "
            f"{_repo_snapshot(repo_root)})",
            "not_found",
        )


def _guard_checkout(repo_root: str) -> None:
    """Refuse to run tests against an incomplete checkout.

    A missing tracked file means the image/container is damaged. That is not a
    cosmetic problem: with `requests/` gone but `tests/` present, pytest happily
    imports the **PyPI `requests` that `httpbin` pulled into site-packages** and
    reports a green suite — a green run against code nobody edited. Every
    trajectory harvested from such a run is worthless and every Day 17-18 number
    is wrong, so this fails loudly instead of producing plausible output.
    """
    deleted = _git(repo_root, "ls-files", "--deleted")
    paths = deleted.stdout.split()
    if paths:
        raise ToolFault(
            f"the repository checkout is incomplete: {len(paths)} tracked file(s) are "
            f"missing from the working tree (e.g. {', '.join(paths[:5])}). Restore it "
            f"with `git -C {repo_root} checkout -- .` before running tests.",
            "sandbox_error",
            missing_files=paths[:20],
        )


def _guard_import_shadowing(repo_root: str, env: dict) -> dict:
    """Verify the repo's own top-level packages resolve to files *inside* the repo.

    Editable installs inject a meta-path finder that outranks `sys.path`, and
    site-packages frequently holds a same-named distribution (here: `requests`,
    installed as a transitive dependency of `httpbin`). Either can quietly serve
    the wrong source tree to pytest. This probe resolves each top-level package
    the repo defines and reports where it actually came from.
    """
    root = os.path.realpath(repo_root)

    # Scan the root AND the src/ convention. Flat-layout repos expose the package
    # at the root; src-layout repos (psf/requests >= 2.32, the current target)
    # one level down. Scanning only the root silently made this guard a no-op for
    # src-layout checkouts — i.e. for the actual target repository.
    candidates: list[str] = []
    root_error: Optional[str] = None
    for index, base_dir in enumerate((root, os.path.join(root, "src"))):
        try:
            for name in sorted(os.listdir(base_dir)):
                if name.isidentifier() and os.path.isfile(
                    os.path.join(base_dir, name, "__init__.py")
                ):
                    candidates.append(name)
        except OSError as exc:
            if index == 0:
                root_error = str(exc)  # pragma: no cover - defensive

    if root_error is not None:
        return {"packages": {}, "shadowed": [], "error": root_error}

    if not candidates:
        return {"packages": {}, "shadowed": []}

    probe = (
        "import importlib, json, os, sys\n"
        "sys.path.insert(0, os.getcwd())\n"
        "report = {}\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        module = importlib.import_module(name)\n"
        "        report[name] = getattr(module, '__file__', None)\n"
        "    except Exception as exc:\n"
        "        report[name] = 'IMPORT_ERROR: %s' % exc\n"
        "print(json.dumps(report))\n"
    )
    try:
        cp = subprocess.run(
            [sys.executable, "-c", probe, *candidates],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"packages": {}, "shadowed": [], "error": "import probe timed out"}

    try:
        resolved = json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {
            "packages": {},
            "shadowed": [],
            "import_errors": [],
            "error": (cp.stderr or cp.stdout or "import probe produced no output").strip()[:300],
        }

    packages: dict[str, Optional[str]] = {}
    shadowed: list[str] = []
    import_errors: list[str] = []
    for name, location in resolved.items():
        if isinstance(location, str) and location.startswith("IMPORT_ERROR"):
            # A package that cannot be imported at all is NOT shadowing — it is
            # usually the very bug under repair. Flagging it as shadowing would
            # reject the pytest run that the agent needs in order to see the
            # traceback, so it is reported informationally instead.
            packages[name] = None
            import_errors.append(f"{name}: {location[len('IMPORT_ERROR:'):].strip()}")
            continue
        packages[name] = location if isinstance(location, str) else None
        if not (location and os.path.realpath(location).startswith(root + os.sep)):
            shadowed.append(name)

    return {"packages": packages, "shadowed": shadowed, "import_errors": import_errors}


_SUMMARY_RE = re.compile(
    r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warning|warnings)\b"
)
_FAILED_RE = re.compile(r"^(FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.*))?$")
#: FAILURES/ERRORS section headers are padded with underscores, e.g.
#: ``____________________ TestBox.test_method_fails ____________________``.
#: The `=` banner lines ("==== FAILURES ====") deliberately do not match.
_SECTION_TITLE_RE = re.compile(r"^_{3,}\s+(.+?)\s+_{3,}\s*$")
#: The exception line inside a traceback section: ``E       AssertionError: boom``.
#: The first one per section is the exception header; later ones are the
#: rewritten assertion detail (``E       assert 1 == 2``).
_EXC_LINE_RE = re.compile(r"^E\s+(\S.*)$")


def _parse_summary(stdout: str) -> dict:
    summary: dict[str, int] = {}
    for line in reversed(stdout.splitlines()):
        if " in " not in line:
            continue
        found = _SUMMARY_RE.findall(line)
        if found:
            for count, label in found:
                key = {"errors": "error", "warnings": "warning"}.get(label, label)
                summary[key] = int(count)
            break
    return summary


def _failure_messages(stdout: str) -> dict[str, str]:
    """Map each FAILURES/ERRORS section title to its first exception (`E `) line.

    Why this is not just a nicety: pytest appends `` - <exception>`` to a short
    summary line **only when it fits the terminal width**, and drops the message
    entirely otherwise. Long node ids therefore lose their message in a non-tty
    ``docker exec`` — verified against pytest 8.4.2, where a 72-character node id
    produced a bare ``FAILED path::name`` while a 40-character one kept its
    ``- AssertionError: ...`` suffix. The traceback section's `E ` line has no
    such conditionality, so it is the authoritative source.
    """
    messages: dict[str, str] = {}
    title: Optional[str] = None
    for line in stdout.splitlines():
        header = _SECTION_TITLE_RE.match(line)
        if header:
            title = header.group(1)
            continue
        if title is None or title in messages:
            continue
        match = _EXC_LINE_RE.match(line)
        if match:
            messages[title] = match.group(1).strip()[:300]
    return messages


def _parse_failed(stdout: str, limit: int = 20) -> list[dict]:
    messages = _failure_messages(stdout)
    failures: list[dict] = []
    for line in stdout.splitlines():
        m = _FAILED_RE.match(line.strip())
        if not m:
            continue
        node = m.group(2)
        # Section titles are "Class.test" / "test[param]" — the node id minus its
        # file part, with `::` collapsed to `.`.
        title = node.split("::", 1)[-1].replace("::", ".")
        message = messages.get(title, "")
        if not message and "::" not in node:
            # Collection/import errors are titled "ERROR collecting <path>", so a
            # bare path node id is matched by substring instead.
            for title_text, text in messages.items():
                if node in title_text:
                    message = text
                    break
        if not message:
            # Last resort, and deliberately last: pytest clips the summary suffix
            # to the terminal width when it only just fits ("AssertionError: d..."),
            # so it is strictly less complete than the `E ` line when both exist.
            message = (m.group(3) or "").strip()
        failures.append({"outcome": m.group(1), "node": node, "message": message[:300]})
        if len(failures) >= limit:
            break
    return failures


def run_pytest_isolated(repo_root: str, payload: dict) -> tuple[dict, dict]:
    raw_targets = payload.get("targets")
    if isinstance(raw_targets, str):
        targets = [raw_targets]
    else:
        targets = list(raw_targets or [])
    targets = [t.strip() for t in targets if t and t.strip()]
    if not targets:
        raise ToolFault("at least one target is required, e.g. 'tests/test_x.py::TestY::test_z'", "invalid_arguments")
    if len(targets) > 20:
        raise ToolFault("at most 20 targets per invocation", "invalid_arguments")
    for target in targets:
        _validate_target(repo_root, target)

    extra_args = list(payload.get("extra_args") or [])
    illegal = [a for a in extra_args if a not in ALLOWED_EXTRA_ARGS]
    if illegal:
        raise ToolFault(
            f"unsupported extra_args {illegal}; allowed: {sorted(ALLOWED_EXTRA_ARGS)}",
            "invalid_arguments",
        )
    deduped_extra = [a for a in dict.fromkeys(extra_args) if a not in ("-q",)]

    timeout = int(payload.get("timeout") or _base.DEFAULT_TEST_TIMEOUT)
    if timeout < 5 or timeout > _base.MAX_TEST_TIMEOUT:
        raise ToolFault(
            f"timeout must be within [5, {_base.MAX_TEST_TIMEOUT}], got {timeout}",
            "invalid_arguments",
        )

    # `-p no:cacheprovider` keeps .pytest_cache out of the working tree: the repo
    # must stay pristine between tasks so patches and resets stay predictable.
    # `--rootdir` is pinned explicitly so rootdir inference cannot wander to a
    # parent directory that happens to hold a pytest.ini.
    argv = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
        "--color=no",
        "--rootdir",
        repo_root,
        *deduped_extra,
    ]

    # `PYTHONDONTWRITEBYTECODE=1` keeps test execution from scattering
    # __pycache__ directories through the working tree (gitignored, so `git status`
    # would stay clean while the tree is in fact no longer pristine).
    # `PYTHONHASHSEED=0` makes set/dict iteration order — and therefore any set
    # repr that ends up inside an assertion message — reproducible run to run,
    # which the plan requires of harvested trajectories.
    # `PYTHONPATH` puts the repo (and its src/ directory, when that layout is in
    # use) first, so the code under repair cannot lose an import race against a
    # same-named distribution in site-packages.
    python_path_parts = [
        part
        for part in (
            repo_root,
            os.path.join(repo_root, "src")
            if os.path.isdir(os.path.join(repo_root, "src"))
            else None,
            os.environ.get("PYTHONPATH"),
        )
        if part
    ]
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join(python_path_parts),
        # Pinned so pytest does not infer the width from the tty: otherwise the
        # same failing run wraps differently (and shows/hides short-summary
        # messages) depending on how it was launched, which breaks reproducible
        # trajectories.
        "COLUMNS": "100",
    }

    # Preflight: refuse to produce a "green" result that does not describe the
    # code under repair (see _guard_checkout / _guard_import_shadowing).
    _guard_checkout(repo_root)
    imports = _guard_import_shadowing(repo_root, env)
    if imports["shadowed"]:
        details = ", ".join(f"{n} -> {imports['packages'].get(n)}" for n in imports["shadowed"])
        raise ToolFault(
            f"tests would import {details}, which is outside {repo_root}: the suite "
            f"would grade a different copy of the package than the one being repaired",
            "sandbox_error",
            repo_imports=imports,
        )

    started = time.monotonic()
    timed_out = False
    exit_code = 0
    stdout = ""
    stderr = ""
    try:
        cp = subprocess.run(
            argv,
            cwd=repo_root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124  # mirrors coreutils `timeout`'s convention
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
    else:
        exit_code = cp.returncode
        stdout = cp.stdout or ""
        stderr = cp.stderr or ""

    duration = round(time.monotonic() - started, 3)
    out_text, out_clipped = _base.truncate(stdout, _base.MAX_OUTPUT_CHARS)
    err_text, err_clipped = _base.truncate(stderr, _base.MAX_OUTPUT_CHARS)

    data = {
        "targets": targets,
        "command": argv,
        "repo_imports": imports,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": duration,
        "stdout": out_text,
        "stderr": err_text,
        "stdout_truncated": out_clipped,
        "stderr_truncated": err_clipped,
        "summary": _parse_summary(stdout),
        "failed_tests": _parse_failed(stdout),
    }
    meta = {"timeout_applied": timeout}

    if timed_out:
        raise ToolFault(
            f"pytest exceeded the {timeout}s budget; narrow the target set",
            "timeout",
            data=data,
            **meta,
        )
    if exit_code != 0:
        raise ToolFault(
            f"pytest exited with code {exit_code}",
            PYTEST_EXIT_KINDS.get(exit_code, "tests_failed"),
            data=data,
            **meta,
        )
    return data, meta


def _as_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

HANDLERS: dict[str, Callable[[str, dict], tuple[dict, dict]]] = {
    "read_file_bounded": read_file_bounded,
    "find_symbol_ast": find_symbol_ast,
    "apply_unified_patch": apply_unified_patch,
    "run_pytest_isolated": run_pytest_isolated,
}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DistillCode-Agent in-container tool runner")
    parser.add_argument("--tool", required=True, choices=sorted(HANDLERS))
    parser.add_argument("--payload", required=True, help="base64-encoded JSON request")
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args(argv)

    envelope: dict
    tool = args.tool
    try:
        request = json.loads(base64.b64decode(args.payload).decode("utf-8"))
        payload = request.get("arguments", request) if isinstance(request, dict) else {}
        if not isinstance(payload, dict):
            raise ToolFault("payload must be a JSON object", "invalid_arguments")
    except ToolFault as fault:
        envelope = _fail(tool, fault.message, fault.kind, data=fault.data, **fault.meta)
    except Exception as exc:
        envelope = _fail(tool, f"malformed request payload: {exc}", "invalid_arguments")
    else:
        try:
            data, meta = HANDLERS[tool](args.repo, payload)
            envelope = _ok(tool, data, **meta)
        except ToolFault as fault:
            # Tools may attach the partial observation (e.g. pytest stdout for a
            # failed run) so the Reflection loop can still reason about it.
            envelope = _fail(tool, fault.message, fault.kind, data=fault.data, **fault.meta)
        except subprocess.TimeoutExpired:
            envelope = _fail(tool, "tool-level subprocess timeout", "timeout")
        except OSError as exc:
            envelope = _fail(tool, f"OS error: {exc}", "sandbox_error")
        except Exception as exc:  # pragma: no cover - defensive
            envelope = _fail(
                tool,
                f"{type(exc).__name__}: {exc}",
                "internal_error",
                traceback=traceback.format_exc()[-1500:],
            )

    print(_base.JSON_SENTINEL)
    print(json.dumps(envelope, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
