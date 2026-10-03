"""bridge.py — host-side transport between the agent and the container tools.

RESPONSIBILITY SPLIT
  This module is *transport only*: it ships the runner into the container, turns
  one tool invocation into one `docker exec`, and parses the JSON envelope back
  into a `ToolResult`. It deliberately contains **no** tool logic and no schema
  knowledge — that keeps the Day 5-7 LangGraph nodes free to call tools without
  importing anything that could not also run inside the sandbox.

WHY base64 ARGV INSTEAD OF STDIN OR `docker cp`
  * `exec_run(..., stdin=...)` is not available on the high-level Docker SDK
    path (writing to an exec stream requires a socket-based `exec_create` /
    `exec_start` dance that is easy to get subtly wrong, and it deadlocks when
    the payload exceeds the socket buffer).
  * `docker cp` / `put_archive` needs a tar stream and a host-side temp file.
  * A base64 argument is one `shlex.quote`d argv token: immune to quoting bugs,
    newline-safe (unified diffs are multi-line), size-irrelevant at the scales
    used here (ARG_MAX is ~2 MB; patches are capped at 200 KB), and trivially
    reproducible when replaying a trajectory.

  The uploaded pair (base.py + container_runner.py) is content-hashed once and
  re-uploaded only when it changes, so a 30-task harvesting run pays the upload
  cost exactly once. Uploaded files are byte-compiled inside the container,
  which turns a broken upload into an immediate, explicit error instead of a
  mysterious `SyntaxError` on the first tool call.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from docker.errors import DockerException

from ..sandbox.docker_runner import DockerSandbox, SandboxError
from . import base as _base

#: Failure classes that must degrade into an envelope instead of an exception.
#: `DockerSandbox.run_command` only converts `APIError`, but a daemon that dies
#: mid-run (very possible on the free-tier / laptop setup this project targets)
#: surfaces as a sibling `DockerException` or a raw socket `OSError`. Letting
#: those escape would violate the "dispatch never raises" contract the Day 5-7
#: graph relies on, so they are funnelled into `error_kind="sandbox_error"` too.
BRIDGE_FAILURES = (SandboxError, DockerException, OSError)

#: Files shipped into the container. `base.py` is included on purpose: it is the
#: single source of truth for the caps and the error taxonomy, so the container
#: cannot drift from the host's notion of "bounded".
UPLOAD_FILES = ("base.py", "container_runner.py")

_LOCAL_DIR = Path(__file__).resolve().parent
_HASH_MARKER = f"{_base.RUNNER_DIR}/.runner.sha256"

#: Interpreter used inside the container for the runner itself. `python` exists
#: too in python:3.11-slim, but `python3` is unambiguous.
_CONTAINER_PYTHON = "python3"

#: Single-sourced in base.py (which is uploaded into the container and therefore
#: cannot import this module); aliased here for readability at call sites.
BRIDGE_SLACK_SECONDS = _base.BRIDGE_SLACK_SECONDS


class BridgeError(RuntimeError):
    """Raised only for programming errors; runtime failures become ToolResults."""


# --------------------------------------------------------------------------- #
# Runner deployment
# --------------------------------------------------------------------------- #


def _local_digest() -> str:
    """Content hash of every uploaded file, stable across OSes and line endings."""
    hasher = hashlib.sha256()
    for name in UPLOAD_FILES:
        data = (_LOCAL_DIR / name).read_bytes().replace(b"\r\n", b"\n")
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(data)
        hasher.update(b"\0")
    hasher.update(str(len(UPLOAD_FILES)).encode())
    return hasher.hexdigest()


def ensure_runner(sandbox: DockerSandbox) -> dict:
    """Upload base.py + container_runner.py if their hash changed.

    Returns a small diagnostics dict for the trajectory log. Idempotent, so it is
    safe to call before every tool invocation (the common case is one cheap
    `cat` of the hash marker).
    """
    digest = f"distillcode-tools-v1 {_local_digest()}\n"

    marker = sandbox.run_command(f"cat {shlex.quote(_HASH_MARKER)} 2>/dev/null")
    if marker.exit_code == 0 and marker.stdout == digest:
        return {"uploaded": False, "digest": digest.split()[1][:12]}

    upload_bytes = 0
    try:
        sandbox.run_command(f"mkdir -p {shlex.quote(_base.RUNNER_DIR)}")
        for name in UPLOAD_FILES:
            raw = (_LOCAL_DIR / name).read_bytes()
            upload_bytes += len(raw)
            encoded = base64.b64encode(raw).decode("ascii")
            command = (
                f"printf %s {shlex.quote(encoded)} | base64 -d "
                f"> {shlex.quote(_base.RUNNER_DIR + '/' + name)}"
            )
            result = sandbox.run_command(command)
            if not result.ok:
                raise SandboxError(
                    f"failed to upload {name} into the container: {result.stderr.strip()}"
                )

        # Byte-compile so a truncated/corrupted upload fails loudly right here.
        check = sandbox.run_command(
            f"cd {shlex.quote(_base.RUNNER_DIR)} && "
            f"{_CONTAINER_PYTHON} -m py_compile {' '.join(shlex.quote(f) for f in UPLOAD_FILES)}"
        )
        if not check.ok:
            raise SandboxError(
                f"uploaded tool runner does not compile: {check.stderr.strip()[:800]}"
            )

        write_marker = sandbox.run_command(
            f"printf %s {shlex.quote(digest)} > {shlex.quote(_HASH_MARKER)}"
        )
        if not write_marker.ok:
            # Non-fatal: the next call simply re-uploads. Never block a task on it.
            pass
    except SandboxError:
        raise
    return {
        "uploaded": True,
        "digest": digest.split()[1][:12],
        "bytes": upload_bytes,
        "files": list(UPLOAD_FILES),
    }


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #


def _split_envelope(stdout: str) -> Optional[str]:
    """Return the JSON blob printed after the last sentinel, or None."""
    index = stdout.rfind(_base.JSON_SENTINEL)
    if index < 0:
        return None
    tail = stdout[index + len(_base.JSON_SENTINEL) :].strip()
    if not tail:
        return None
    return tail.splitlines()[-1].strip()


def invoke(
    sandbox: DockerSandbox,
    tool: str,
    arguments: Mapping[str, Any],
    *,
    timeout: int = 30,
    repo_path: str = _base.REPO_PATH,
    ensure: bool = True,
) -> _base.ToolResult:
    """Execute one tool call inside the sandbox and return its envelope.

    Never raises for runtime problems: a missing container, a crashed runner, or
    garbage on stdout all become `ToolResult(ok=False, error_kind="sandbox_error")`
    so the graph's control flow stays uniform and the trajectory log records why.
    """
    deploy: dict = {}
    if ensure:
        try:
            deploy = ensure_runner(sandbox)
        except BRIDGE_FAILURES as exc:
            return _base.ToolResult.failure(
                tool, f"sandbox unavailable: {type(exc).__name__}: {exc}", "sandbox_error"
            )

    payload = json.dumps({"arguments": dict(arguments)}, ensure_ascii=False)
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    command = " ".join(
        [
            _CONTAINER_PYTHON,
            shlex.quote(_base.RUNNER_PATH),
            "--tool",
            shlex.quote(tool),
            "--repo",
            shlex.quote(repo_path),
            "--payload",
            shlex.quote(encoded),
        ]
    )

    started = time.monotonic()
    try:
        result = sandbox.run_command(command, timeout=timeout)
    except BRIDGE_FAILURES as exc:
        return _base.ToolResult.failure(
            tool, f"docker exec failed: {type(exc).__name__}: {exc}", "sandbox_error"
        )
    duration = round(time.monotonic() - started, 3)

    meta: dict[str, Any] = {"bridge_duration_s": duration, "exit_code": result.exit_code}
    if deploy.get("uploaded"):
        meta["runner_uploaded"] = True

    raw_json = _split_envelope(result.stdout)
    if raw_json is None:
        if result.exit_code == 124:
            return _base.ToolResult.failure(
                tool,
                f"tool call exceeded the {timeout}s bridge budget (killed by coreutils timeout)",
                "timeout",
                **meta,
            )
        return _base.ToolResult.failure(
            tool,
            "tool runner produced no JSON envelope (crash or corrupted stdout)",
            "sandbox_error",
            runner_stderr=result.stderr.strip()[-800:],
            runner_stdout=result.stdout.strip()[-800:],
            **meta,
        )

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return _base.ToolResult.failure(
            tool,
            f"tool runner emitted invalid JSON: {exc}",
            "sandbox_error",
            runner_stdout=raw_json[:800],
            **meta,
        )

    if not isinstance(parsed, dict):
        return _base.ToolResult.failure(
            tool, "tool runner envelope was not a JSON object", "sandbox_error", **meta
        )

    if result.stderr.strip():
        meta["runner_stderr"] = result.stderr.strip()[-800:]
    return _base.ToolResult.from_runner(tool, parsed, **meta)


def bridge_timeout_for(timeout: int) -> int:
    """Bridge budget for a tool whose internal budget is `timeout` seconds."""
    return int(timeout) + BRIDGE_SLACK_SECONDS
