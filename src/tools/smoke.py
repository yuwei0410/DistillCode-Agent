"""smoke.py — end-to-end exercise of the Day 3-4 tool layer against the real image.

WHAT THIS PROVES (and why it is a script, not a unit test)
  Every claim the plan makes about Day 3-4 is about *behaviour inside the
  sandbox*: that reads stay bounded, that symbol lookup finds real definitions,
  that a rejected patch leaves the tree untouched, and that pytest's exit code and
  failure lines survive the round trip. Fake-repo unit tests cannot demonstrate
  any of that, and they would require running the runner on the host — precisely
  the state-drift the architecture forbids. So the smoke test drives the real
  bridge against the real image, then asserts the repo is byte-clean again.

  It also deliberately leaves the repo pristine on exit. The two mutation checks
  modify tracked files and create an (untracked) temporary test module, so the run
  finishes with `reset_repo()` and a `git status --porcelain` assertion: if the
  tool layer can leak dirty state, this script fails, which matters because a
  leaked file would silently contaminate every later trajectory in Day 8-12.

USAGE
    docker build -f Dockerfile.sandbox -t distillcode/sandbox:requests .
    python -m src.tools.smoke          # from the repository root
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from typing import Optional

from ..sandbox.docker_runner import DockerSandbox, SandboxError
from . import base as _base
from .registry import dispatch, openai_tools, tool_names, validate_call

RESULTS: list[tuple[str, bool, str]] = []

#: Artifacts the mutation checks touch. Hoisted to module scope so the cleanup
#: step can remove them by exact path instead of trusting `git clean -fd`, which
#: deliberately leaves ignored files behind.
TMP_TEST_FILE = "tests/test_distillcode_smoke_tmp.py"


def discover_package_root(sandbox: DockerSandbox, package: str = "requests") -> str:
    """Locate the target package directory instead of assuming it.

    psf/requests moved to a src/ layout (package at src/requests/) at v2.32.x.
    Hardcoding 'requests/...' therefore silently pointed every check at a path
    that does not exist, and produced an entire class of phantom failures against
    a healthy checkout. Discovery keeps this harness correct for both layouts —
    and for any future REQUESTS_REF that flips back.
    """
    for candidate in (package, f"src/{package}"):
        res = dispatch("read_file_bounded",
                       {"path": f"{candidate}/__init__.py", "max_lines": 1}, sandbox)
        if res.ok:
            print(f"    package root discovered at {candidate}/")
            return candidate
    print(f"    neither {package}/ nor src/{package}/ contains __init__.py")
    return ""


def check(name: str, condition: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(condition), detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' — ' + detail) if detail and not condition else ''}")
    return bool(condition)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_schemas() -> None:
    section("1/7 — schema export (must not require a Docker daemon to be correct)")
    names = tool_names()
    check(
        "all four Day 3-4 tools registered",
        names == ["apply_unified_patch", "find_symbol_ast", "read_file_bounded", "run_pytest_isolated"],
        f"got {names}",
    )
    tools = openai_tools()
    check("openai_tools() emits one declaration per tool", len(tools) == 4, f"got {len(tools)}")
    check(
        "declarations use the function-calling envelope",
        all(t["type"] == "function" and "parameters" in t["function"] for t in tools),
    )
    check("every schema is strict (additionalProperties=false)",
          all(t["function"]["parameters"].get("additionalProperties") is False for t in tools))
    check("schemas are stable across calls (byte-identical)",
          json.dumps(openai_tools()) == json.dumps(openai_tools()))

    # Pure-host validation: unknown tool, malformed JSON, and a type violation.
    check("validate_call rejects an unknown tool", validate_call("nope", {}) is not None)
    check("validate_call rejects malformed JSON arguments",
          validate_call("read_file_bounded", '{"path": ') is not None)
    check("validate_call rejects a wrong argument type",
          validate_call("read_file_bounded", {"path": "requests/utils.py", "start_line": "abc"}) is not None)
    check("validate_call accepts a correct call",
          validate_call("read_file_bounded", {"path": "requests/utils.py", "start_line": 1}) is None)
    check("unexpected properties are rejected",
          validate_call("read_file_bounded", {"path": "x.py", "nope": 1}) is not None)

    bad = dispatch("does_not_exist", {}, sandbox=None)
    check("dispatch of an unknown tool returns an envelope, not an exception",
          (not bad.ok) and bad.error_kind == "invalid_arguments")
    bad = dispatch("read_file_bounded", {"path": "requests/utils.py", "start_line": -3}, sandbox=None)
    check("dispatch rejects out-of-range arguments before touching Docker",
          (not bad.ok) and bad.error_kind == "invalid_arguments", bad.error or "")


def check_preflight(sandbox: DockerSandbox) -> str:
    """Bring the sandbox to a known baseline before anything is asserted.

    Every later check is a *delta* against this baseline, so establishing it here
    is not optional. A container previously left damaged (a deleted tracked file,
    a stale .pytest_cache) produced a pile of unrelated-looking path failures —
    and, far worse, let pytest grade a copy of the package imported from
    site-packages while the repo copy was missing. Failing here means the rest of
    the run cannot be misinterpreted.
    """
    section("0/7 — sandbox preflight (establish a known baseline)")
    print(f"    container {sandbox.container.name} ({sandbox.container.id[:12]})")
    print(f"    image {sandbox.image}")

    inherited = sandbox.run_command(f"git -C {_base.REPO_PATH} status --porcelain")
    if inherited.stdout.strip():
        print(f"    inherited dirty state, resetting: {inherited.stdout.strip().splitlines()[:5]}")

    reset = sandbox.reset_repo()
    check("reset_repo() succeeds", reset.ok, reset.stderr.strip()[:300])

    status = sandbox.run_command(f"git -C {_base.REPO_PATH} status --porcelain")
    check("working tree is clean before testing", not status.stdout.strip(),
          status.stdout.strip()[:300])

    deleted = sandbox.run_command(f"git -C {_base.REPO_PATH} ls-files --deleted")
    check("no tracked file is missing from the checkout", not deleted.stdout.strip(),
          deleted.stdout.strip()[:300])

    remote = sandbox.run_command(f"git -C {_base.REPO_PATH} remote get-url origin")
    history = sandbox.run_command(f"git -C {_base.REPO_PATH} log --oneline -3")
    print(f"    remote: {remote.stdout.strip() or '<none>'}")
    for line in history.stdout.strip().splitlines()[:3]:
        print(f"    {line}")

    package_root = discover_package_root(sandbox)
    if not check("the target package directory was located", bool(package_root)):
        return ""

    baseline_ok = True
    for probe_path in (f"{package_root}/__init__.py", f"{package_root}/__version__.py",
                       "tests/test_requests.py"):
        exists = sandbox.run_command(
            f"test -f {_base.REPO_PATH}/{probe_path} && echo yes || echo no"
        )
        if not check(f"expected file present: {probe_path}", "yes" in exists.stdout,
                     exists.stdout.strip()):
            baseline_ok = False

    if not baseline_ok:
        # A git tree that is clean yet lacks requests/ means the checkout was
        # tampered with at a COMMITTED level (or the container was created from a
        # different image). Continuing would produce a dozen misleading failures,
        # so the run stops here after printing the decisive facts.
        print("\n    --- baseline diagnostics ---")
        listing = sandbox.run_command(f"ls -la {_base.REPO_PATH}")
        print(listing.stdout.rstrip())
        in_head = sandbox.run_command(
            f"git -C {_base.REPO_PATH} cat-file -e HEAD:requests/__init__.py "
            "&& echo in-HEAD || echo not-in-HEAD"
        )
        print(f"    requests/__init__.py: {in_head.stdout.strip()}")
        tracked = sandbox.run_command(
            f"git -C {_base.REPO_PATH} ls-files | head -30"
        )
        print("    tracked files (first 30):")
        for line in tracked.stdout.splitlines()[:30]:
            print(f"      {line}")
        return ""
    return package_root


def check_read(sandbox: DockerSandbox, pkg: str) -> None:
    section("2/7 — read_file_bounded")
    res = dispatch("read_file_bounded", {"path": f"{pkg}/__version__.py", "start_line": 1}, sandbox)
    check("reads a real file", res.ok, res.error or "")
    if res.ok:
        check("returns line-numbered content", "|" in res.data["content"].splitlines()[0],
              res.data["content"].splitlines()[0][:80] if res.data["content"] else "<empty>")
        check("reports total_lines", res.data["total_lines"] > 0, str(res.data.get("total_lines")))

    res = dispatch(
        "read_file_bounded",
        {"path": f"{pkg}/models.py", "start_line": 10, "end_line": 5000, "max_lines": 10},
        sandbox,
    )
    check("window cap is enforced", res.ok and res.data["returned_lines"] == 10,
          str(res.error or res.data.get("returned_lines")))
    if res.ok:
        check("cap is disclosed to the model", res.meta.get("capped_by_max_lines") is True)
        check("has_more + next_start_line let the model page forward",
              res.data["has_more"] and res.data["next_start_line"] == res.data["last_line"] + 1,
              str(res.data.get("next_start_line")))

    res = dispatch("read_file_bounded", {"path": f"{pkg}/models.py", "max_lines": 1000}, sandbox)
    check("max_lines above the hard ceiling is rejected by the schema",
          (not res.ok) and res.error_kind == "invalid_arguments", res.error or "")

    res = dispatch("read_file_bounded", {"path": "../../etc/passwd"}, sandbox)
    check("path traversal is rejected", (not res.ok) and res.error_kind == "path_out_of_repo", res.error or "")

    res = dispatch("read_file_bounded", {"path": "/etc/passwd"}, sandbox)
    check("absolute paths are rejected", (not res.ok) and res.error_kind == "path_out_of_repo", res.error or "")

    res = dispatch("read_file_bounded", {"path": f"{pkg}/does_not_exist.py"}, sandbox)
    check("missing file reports not_found", (not res.ok) and res.error_kind == "not_found", res.error or "")


def check_symbols(sandbox: DockerSandbox, pkg: str) -> None:
    section("3/7 — find_symbol_ast")
    res = dispatch("find_symbol_ast", {"name": "Session", "kind": "class", "scope": pkg, "exact": True}, sandbox)
    check("finds requests.Session", res.ok and res.data["count"] >= 1, res.error or "")
    if res.ok and res.data["symbols"]:
        hit = res.data["symbols"][0]
        check("reports a file/line span for the definition",
              hit["file"].endswith(".py") and isinstance(hit["lineno"], int) and hit["end_lineno"] >= hit["lineno"],
              json.dumps(hit))
        check("reports the qualname", hit["qualname"] == "Session", hit.get("qualname", ""))

    # `prepare_url` is a stable, long-standing method of PreparedRequest — chosen
    # over a vague name like "get" so this assertion cannot succeed or fail based
    # on which requests release the image was built from.
    res = dispatch(
        "find_symbol_ast",
        {"name": "prepare_url", "kind": "method", "scope": f"{pkg}/models.py", "exact": True},
        sandbox,
    )
    check("finds methods with the signature included",
          res.ok and res.data["count"] >= 1 and all(s["kind"] == "method" for s in res.data["symbols"]),
          res.error or json.dumps(res.data.get("symbols", []))[:200])
    if res.ok and res.data["symbols"]:
        check("signature is reconstructed", bool(res.data["symbols"][0].get("signature")),
              json.dumps(res.data["symbols"][0]))

    res = dispatch("find_symbol_ast", {"name": "zzz_no_such_symbol_zzz", "scope": pkg}, sandbox)
    check("a miss is a successful empty result, not an error",
          res.ok and res.data["count"] == 0, res.error or "")

    res = dispatch("find_symbol_ast", {"kind": "class", "scope": pkg, "max_results": 3}, sandbox)
    check("max_results truncates and is disclosed",
          res.ok and res.data["count"] == 3 and res.data["truncated"] is True, res.error or "")

    res = dispatch("find_symbol_ast", {"kind": "class", "scope": pkg, "include_signature": True}, sandbox)
    check("indexes the package without parse errors",
          res.ok and not res.data["parse_errors"] and res.data["scanned_files"] > 5,
          f"scanned={res.data.get('scanned_files')} errors={res.data.get('parse_errors')}")


def check_patch(sandbox: DockerSandbox, pkg: str) -> None:
    section("4/7 — apply_unified_patch")
    target = f"{pkg}/__version__.py"
    baseline = dispatch("read_file_bounded",
                        {"path": target, "with_line_numbers": False, "max_lines": 50}, sandbox)
    if not check("baseline read for patch construction", baseline.ok, baseline.error or ""):
        return
    original = baseline.data["content"]
    lines = original.splitlines()
    old_line = lines[0]
    check("baseline file is non-empty", bool(old_line.strip()), repr(old_line))

    # Hash the file on disk rather than re-reading it through the tool: the tool
    # normalises line endings, so a tool-level round-trip comparison cannot detect
    # a silent whole-file newline rewrite. This hash is the check that catches it.
    digest_before = sandbox.run_command(
        f"sha1sum {_base.REPO_PATH}/{target}"
    ).stdout.split()[:1]
    digest_before = digest_before[0] if digest_before else ""
    check("baseline sha1 computed", bool(digest_before))

    # ---- replace mode -------------------------------------------------- #
    new_line = old_line + "  # distillcode-smoke-replace"
    res = dispatch("apply_unified_patch",
                   {"mode": "replace", "path": target, "old_str": old_line, "new_str": new_line}, sandbox)
    check("replace mode applies", res.ok and res.data["applied"], res.error or "")
    if res.ok:
        check("replace reports the modified file",
              any(c["path"] == target for c in res.data["changed_files"]),
              json.dumps(res.data.get("changed_files")))
        check("replace reports a read-back diffstat", "1 file changed" in res.data["diffstat"]["text"],
              res.data["diffstat"]["text"])

    after = dispatch("read_file_bounded", {"path": target, "with_line_numbers": False, "max_lines": 50}, sandbox)
    check("the change is visible on the next read", after.ok and new_line in after.data["content"])

    # revert with the tool itself (round-trip through the patch path)
    res = dispatch("apply_unified_patch",
                   {"mode": "replace", "path": target, "old_str": new_line, "new_str": old_line}, sandbox)
    check("replace mode reverts", res.ok and res.data["applied"], res.error or "")

    # ---- failure modes must not mutate anything ------------------------ #
    res = dispatch("apply_unified_patch",
                   {"mode": "replace", "path": target, "old_str": "zzz_not_present_zzz", "new_str": "x"}, sandbox)
    check("missing anchor -> no_match", (not res.ok) and res.error_kind == "no_match", res.error or "")

    res = dispatch("apply_unified_patch",
                   {"mode": "replace", "path": target, "old_str": "i", "new_str": "x"}, sandbox)
    check("repeated anchor -> ambiguous_match",
          (not res.ok) and res.error_kind == "ambiguous_match", res.error or "")
    if not res.ok:
        check("ambiguous_match lists candidate lines", bool(res.meta.get("match_lines")),
              json.dumps(res.meta.get("match_lines")))

    res = dispatch("apply_unified_patch",
                   {"mode": "replace", "path": "../../etc/hosts", "old_str": "a", "new_str": "b"}, sandbox)
    check("path traversal is rejected for writes too",
          (not res.ok) and res.error_kind == "path_out_of_repo", res.error or "")

    res = dispatch("apply_unified_patch",
                   {"mode": "replace", "path": target, "old_str": old_line, "new_str": old_line}, sandbox)
    check("no-op edit is rejected", (not res.ok) and res.error_kind == "invalid_arguments", res.error or "")

    # ---- diff mode ----------------------------------------------------- #
    patched = original.replace(old_line, old_line + "  # distillcode-smoke-diff", 1)
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{target}",
            tofile=f"b/{target}",
            n=1,
        )
    )
    res = dispatch("apply_unified_patch", {"mode": "diff", "patch": diff, "dry_run": True}, sandbox)
    check("diff mode dry_run validates without writing",
          res.ok and res.data["applied"] is False, res.error or "")
    if res.ok:
        check("dry_run auto-detects the a/ b/ prefix depth", res.data["strip_used"] == 1,
              str(res.data.get("strip_used")))

    untouched = dispatch("read_file_bounded", {"path": target, "with_line_numbers": False, "max_lines": 50}, sandbox)
    check("dry_run left the file untouched",
          untouched.ok and untouched.data["content"] == original,
          "dry_run mutated the working tree!")

    res = dispatch("apply_unified_patch", {"mode": "diff", "patch": diff}, sandbox)
    check("diff mode applies", res.ok and res.data["applied"], res.error or "")

    res = dispatch("apply_unified_patch", {"mode": "diff", "patch": diff}, sandbox)
    check("re-applying the same diff is rejected (not silently applied twice)",
          (not res.ok) and res.error_kind == "patch_rejected", res.error or "")

    # Restore the baseline before the pytest section runs, and prove the round
    # trip is byte-exact. Without this, every later check would execute against a
    # repo that a previous check had already edited — the kind of hidden coupling
    # that makes a smoke test pass while the tool layer is broken.
    res = dispatch("apply_unified_patch",
                   {"mode": "replace", "path": target,
                    "old_str": old_line + "  # distillcode-smoke-diff", "new_str": old_line}, sandbox)
    check("diff-mode change is reverted through the tool", res.ok and res.data["applied"], res.error or "")
    digest_after = sandbox.run_command(
        f"sha1sum {_base.REPO_PATH}/{target}"
    ).stdout.split()[:1]
    digest_after = digest_after[0] if digest_after else ""
    check("the file on disk is byte-identical to the baseline (sha1)",
          bool(digest_after) and digest_after == digest_before,
          f"{digest_before} -> {digest_after} (silent rewrite: line endings? encoding?)")
    dirty = sandbox.run_command(
        f"git -C {_base.REPO_PATH} status --porcelain -- {target}"
    ).stdout.strip()
    check("git agrees the file is unmodified after the round trip", not dirty, dirty)

    # ---- a diff that creates a failing test module --------------------- #
    tmp_test = TMP_TEST_FILE
    body = (
        "def test_distillcode_smoke_expected_failure():\n"
        '    assert False, "distillcode smoke: this failure is expected"\n'
    )
    create_patch = (
        f"diff --git a/{tmp_test} b/{tmp_test}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{tmp_test}\n"
        f"@@ -0,0 +1,{len(body.splitlines())} @@\n"
        + "".join(f"+{line}\n" for line in body.splitlines())
    )
    res = dispatch("apply_unified_patch", {"mode": "diff", "patch": create_patch}, sandbox)
    check("diff mode can create a new file", res.ok and res.data["applied"], res.error or "")
    if res.ok:
        # Regression guard: `git diff --numstat` does not list untracked files, so
        # without the repo-state snapshot this reports "nothing changed" for a
        # patch that just created a file.
        check("a created file is reported as status=created",
              any(c["path"] == tmp_test and c["status"] == "created" for c in res.data["changed_files"]),
              json.dumps(res.data.get("changed_files")))
        check("target_paths echoes the file the patch actually touches",
              res.data.get("target_paths") == [tmp_test], json.dumps(res.data.get("target_paths")))

    # Regression guard for a real defect found in review: re-applying a
    # create-file patch used to fall through to -p0 and silently create
    # `b/<name>` — a stray file in a directory the model never mentioned.
    res = dispatch("apply_unified_patch", {"mode": "diff", "patch": create_patch}, sandbox)
    check("re-applying a create-file patch is rejected (no silent a/ or b/ landing)",
          (not res.ok) and res.error_kind == "patch_rejected", res.error or "")
    stray = sandbox.run_command(
        f"ls -d {_base.REPO_PATH}/a {_base.REPO_PATH}/b 2>/dev/null || echo none"
    )
    check("no stray prefix directory was created", "none" in stray.stdout, stray.stdout.strip())


def check_pytest(sandbox: DockerSandbox) -> None:
    section("5/7 — run_pytest_isolated")
    green = [
        "tests/test_requests.py::TestRequests::test_entry_points",
        "tests/test_requests.py::TestRequests::test_basic_building",
    ]
    res = dispatch("run_pytest_isolated", {"targets": green, "timeout": 120}, sandbox)
    check("two known-green cases pass", res.ok and res.data["exit_code"] == 0, res.error or "")
    if res.ok:
        check("summary is parsed into counts", res.data["summary"].get("passed") == 2,
              json.dumps(res.data.get("summary")))
        check("no failures are reported", res.data["failed_tests"] == [])
        check("duration is recorded", res.data["duration_s"] > 0)

    # The module created by check_patch() must fail, and that failure must travel
    # back as a structured envelope — this is the exact signal the Day 5-7
    # Reflection loop consumes.
    res = dispatch("run_pytest_isolated",
                   {"targets": ["tests/test_distillcode_smoke_tmp.py"], "timeout": 120}, sandbox)
    check("a failing test surfaces as tests_failed",
          (not res.ok) and res.error_kind == "tests_failed", res.error or "")
    if not res.ok:
        check("failed test output survives the error path", res.data.get("exit_code") == 1,
              str(res.data.get("exit_code")))
        check("failed_tests lists the assertion message",
              bool(res.data.get("failed_tests"))
              and "expected" in res.data["failed_tests"][0]["message"],
              json.dumps(res.data.get("failed_tests"))
              + " | stdout tail: "
              + (res.data.get("stdout") or "")[-400:])

    res = dispatch("run_pytest_isolated", {"targets": ["tests/nope_test.py"]}, sandbox)
    check("missing target fails fast with not_found",
          (not res.ok) and res.error_kind == "not_found", res.error or "")

    res = dispatch("run_pytest_isolated", {"targets": ["-q", "tests"]}, sandbox)
    check("a flag cannot be smuggled through targets",
          (not res.ok) and res.error_kind == "invalid_arguments", res.error or "")

    res = dispatch("run_pytest_isolated",
                   {"targets": ["tests/test_utils.py"], "extra_args": ["--cov=x"]}, sandbox)
    check("non-allowlisted extra_args are rejected",
          (not res.ok) and res.error_kind == "invalid_arguments", res.error or "")

    res = dispatch("run_pytest_isolated", {"targets": ["tests/test_utils.py"], "timeout": 2}, sandbox)
    check("timeout below the floor is rejected",
          (not res.ok) and res.error_kind == "invalid_arguments", res.error or "")


def check_clean(sandbox: DockerSandbox) -> None:
    section("6/7 — the sandbox is left pristine")
    # The leak check MUST run before reset_repo(): `git clean -fd` would delete a
    # stray .pytest_cache, so asserting afterwards would pass for the wrong reason.
    junk = sandbox.run_command(
        f"test -e {_base.REPO_PATH}/.pytest_cache && echo present || echo absent"
    )
    check("no .pytest_cache leaked into the repo", "absent" in junk.stdout, junk.stdout.strip())

    sandbox.reset_repo()
    # `git clean -fd` honours .gitignore, so an *ignored* temp module would survive
    # the reset and be silently collected by every later test run. Removing the
    # artifacts by exact path closes that hole.
    sandbox.run_command(f"rm -f {_base.REPO_PATH}/{TMP_TEST_FILE}")
    status = sandbox.run_command(f"git -C {_base.REPO_PATH} status --porcelain")
    check("git status is empty after reset", status.ok and not status.stdout.strip(),
          status.stdout.strip()[:400])
    gone = sandbox.run_command(
        f"test -e {_base.REPO_PATH}/{TMP_TEST_FILE} && echo present || echo absent"
    )
    check("the temporary test module is really gone", "absent" in gone.stdout, gone.stdout.strip())
    tools_dir = sandbox.run_command(f"ls -1 {_base.RUNNER_DIR} 2>/dev/null || echo <missing>")
    if not tools_dir.ok or "<missing>" in tools_dir.stdout:
        # The runner is uploaded lazily on the first dispatch, so an aborted run
        # (broken baseline) legitimately leaves no runner behind. That is not a
        # leak and must not fail the cleanup section.
        check("the tool runner lives outside the repo (skipped: never uploaded)", True)
    else:
        check("the tool runner lives outside the repo",
              "container_runner.py" in tools_dir.stdout, tools_dir.stdout.strip())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Day 3-4 tool layer smoke test")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="only run the host-side checks that need no Docker daemon",
    )
    args = parser.parse_args(argv)

    print("DistillCode-Agent — Day 3-4 tool layer smoke test")
    check_schemas()

    if args.offline:
        return _report()

    try:
        sandbox = DockerSandbox()
    except SandboxError as exc:
        print(f"\n[SKIP] sandbox unavailable: {exc}")
        print("Build the image first:\n  docker build -f Dockerfile.sandbox -t distillcode/sandbox:requests .")
        return 2
    except Exception as exc:  # noqa: BLE001
        # The Docker SDK raises a bare DockerException (not SandboxError) when the
        # daemon is unreachable, e.g. `docker` on PATH but the service stopped, or
        # an unset DOCKER_HOST. Without this branch the smoke test dies with a raw
        # traceback that says nothing about what to do.
        print(f"\n[SKIP] cannot reach the Docker daemon: {type(exc).__name__}: {exc}")
        print(
            "Start the Docker daemon (on Windows, run this from WSL2 with Docker "
            "Desktop integration enabled), then retry."
        )
        return 2

    try:
        package_root = check_preflight(sandbox)
        if not package_root:
            print("\n[ABORT] sandbox baseline is broken; see the diagnostics above.")
            print("Destroy the possibly-tampered container and retry:")
            print(f"  docker rm -f {sandbox.container.name}")
            print("If that does not fix it, rebuild the image:")
            print("  docker rmi distillcode/sandbox:requests")
            print("  docker build -f Dockerfile.sandbox -t distillcode/sandbox:requests .")
        else:
            check_read(sandbox, package_root)
            check_symbols(sandbox, package_root)
            check_patch(sandbox, package_root)
            check_pytest(sandbox)
    finally:
        check_clean(sandbox)

    return _report()


def _report() -> int:
    failures = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failures)}/{len(RESULTS)} checks passed")
    if failures:
        print("failed checks:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("Day 3-4 tool layer: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
