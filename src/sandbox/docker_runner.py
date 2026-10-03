"""docker_runner.py — Deterministic Docker execution bridge for DistillCode-Agent.

Wraps the Docker Python SDK to run bounded shell commands inside a long-lived
sandbox container. A hard 30s command timeout is enforced via coreutils `timeout`
*inside* the container (not just the SDK socket timeout), so runaway commands are
killed reliably and reproducibly.

Design (Phase 1, Day 1-2 — "两者结合" strategy):
  * Image bakes the requests source into /workspace/repo (see Dockerfile.sandbox).
  * At runtime you may mount a host checkout over /workspace/repo for debugging
    without rebuilding the image, via `host_repo_path=...`.

This module is the execution backbone that the later LangGraph Executor Node
(Day 5-7) and Test Verifier Node will call.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Optional

import docker
from docker.errors import APIError, NotFound


DEFAULT_IMAGE = "distillcode/sandbox:requests"
DEFAULT_CONTAINER = "distillcode-sandbox"
DEFAULT_CMD_TIMEOUT = 30  # seconds, hard kill
REPO_PATH = "/workspace/repo"


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class SandboxError(RuntimeError):
    pass


class DockerSandbox:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        container_name: str = DEFAULT_CONTAINER,
        host_repo_path: Optional[str] = None,
        command_timeout: int = DEFAULT_CMD_TIMEOUT,
        auto_build: bool = False,
        dockerfile_dir: str = ".",
    ) -> None:
        self.image = image
        self.container_name = container_name
        self.command_timeout = command_timeout
        self.host_repo_path = host_repo_path
        self.dockerfile_dir = dockerfile_dir
        self.client = docker.from_env()
        self.container = self._ensure_container(auto_build)

    # ---- lifecycle -------------------------------------------------------
    def _volumes(self):
        if not self.host_repo_path:
            return None
        return {self.host_repo_path: {"bind": REPO_PATH, "mode": "rw"}}

    def _build_image(self) -> None:
        print(f"Building image {self.image} from {self.dockerfile_dir} ...")
        self.client.images.build(
            path=self.dockerfile_dir, tag=self.image, dockerfile="Dockerfile.sandbox"
        )

    def _ensure_container(self, auto_build: bool):
        # Reuse if already running — but only if it actually runs the expected
        # image. Name-based reuse without this check silently tests a wrong
        # filesystem whenever an older/experimental container shares the name
        # (every later "green" result then describes a repo nobody intended).
        try:
            container = self.client.containers.get(self.container_name)
            if container.status != "running":
                container.start()
            running_image = container.attrs.get("Config", {}).get("Image", "")
            if running_image != self.image:
                raise SandboxError(
                    f"container '{self.container_name}' runs image "
                    f"'{running_image}', but the sandbox expects '{self.image}'. "
                    f"The leftover container is being reused; remove it first:\n"
                    f"  docker rm -f {self.container_name}"
                )
            return container
        except NotFound:
            pass

        # Image present?
        try:
            self.client.images.get(self.image)
        except NotFound:
            if auto_build:
                self._build_image()
            else:
                raise SandboxError(
                    f"Image '{self.image}' not found. Build it first:\n"
                    f"  docker build -f Dockerfile.sandbox -t {self.image} .\n"
                    f"or pass auto_build=True to DockerSandbox()."
                )

        return self.client.containers.run(
            self.image,
            name=self.container_name,
            command=["sleep", "infinity"],
            detach=True,
            tty=False,
            volumes=self._volumes(),
            working_dir=REPO_PATH,
        )

    # ---- execution -------------------------------------------------------
    def run_command(self, cmd: str, timeout: Optional[int] = None) -> RunResult:
        """Run `cmd` (a shell string) inside the container with a hard timeout.

        Returns RunResult. exit_code == 124 indicates the command was killed by
        `timeout` (i.e. exceeded the limit).
        """
        t = timeout or self.command_timeout
        # Wrap with coreutils `timeout` for a guaranteed kill, then exec via sh -c.
        wrapped = ["timeout", f"{t}s", "sh", "-c", cmd]
        try:
            exit_code, (stdout_b, stderr_b) = self.container.exec_run(
                wrapped, stdout=True, stderr=True, demux=True
            )
        except APIError as e:
            raise SandboxError(f"exec_run failed: {e}") from e
        return RunResult(
            exit_code=exit_code or 0,
            stdout=(stdout_b or b"").decode("utf-8", "replace"),
            stderr=(stderr_b or b"").decode("utf-8", "replace"),
        )

    def run_pytest(self, test_path: str = "tests", timeout: Optional[int] = None) -> RunResult:
        """Convenience wrapper used by the Test Verifier Node (Day 5-7)."""
        return self.run_command(
            f"python -m pytest {shlex.quote(test_path)} -q", timeout=timeout
        )

    def reset_repo(self) -> RunResult:
        """Restore /workspace/repo to its baked/original state between tasks.

        Deliberately NOT `git clean -fdx`: that would also delete
        `requests.egg-info`, which the image's editable install (`-e .`) depends
        on, breaking `import requests` for every later task.

        The explicit sweep of .pytest_cache / __pycache__ is required, though,
        because `git checkout` and `git clean -fd` both honour .gitignore and so
        never remove ignored artifacts. Left in place they accumulate silently —
        a stray .pytest_cache from a Day 1-2 smoke run is exactly what made the
        "pristine repository" invariant look satisfied while it was not.
        """
        return self.run_command(
            f"git -C {REPO_PATH} checkout -- . && git -C {REPO_PATH} clean -fd && "
            f"rm -rf {REPO_PATH}/.pytest_cache && "
            f"find {REPO_PATH} -type d -name __pycache__ -prune -exec rm -rf {{}} +"
        )

    # ---- teardown --------------------------------------------------------
    def stop(self) -> None:
        try:
            self.container.stop()
        except APIError:
            pass

    def remove(self, force: bool = True) -> None:
        try:
            self.container.remove(force=force)
        except APIError:
            pass


if __name__ == "__main__":
    # Smoke test — assumes the image already exists (build it first):
    #   docker build -f Dockerfile.sandbox -t distillcode/sandbox:requests .
    # Only two self-contained, network-free cases are selected here so the smoke
    # test stays far below the 30s hard cap (the full ~200-case suite takes well
    # over 30s and belongs to the targeted Day 3-4 run_pytest_isolated tool).
    # run_pytest() shlex-quotes its argument as ONE node id, so each case gets
    # its own invocation rather than being passed as a single multi-arg string.
    sb = DockerSandbox()
    print(sb.run_command("python -c 'import requests; print(requests.__version__)'"))
    print(sb.run_pytest("tests/test_requests.py::TestRequests::test_entry_points"))
    print(sb.run_pytest("tests/test_requests.py::TestRequests::test_basic_building"))
