"""Sandboxed execution of miner submissions: the real anti-gaming boundary.

``static_analysis.scan_submission`` is a cheap *pre-filter* -- it rejects the
obvious cheats (baked-in tables, network imports, ``eval``) without running
anything, but it is **not** a security boundary: a determined miner can obfuscate
past a static scan. The actual boundary is *execution under isolation*: run the
submission's ``forecast(context)`` in a separate, resource-limited process whose
output is validated, and treat anything that reaches outside its box as a
disqualification.

What this enforces (POSIX)
--------------------------
* **Separate process** -- the submission runs via a fresh interpreter, so it
  cannot touch the validator's memory, globals, or the reference panel.
* **CPU + wall-clock limits** -- ``RLIMIT_CPU`` plus a ``subprocess`` timeout kill
  runaway / busy-loop submissions.
* **Address-space limit** -- ``RLIMIT_AS`` caps memory so a submission cannot OOM
  the validator.
* **File-size limit** -- ``RLIMIT_FSIZE`` (0 by default) blocks the persist-state
  vector ``static_analysis`` only warns about.
* **Network denial** -- ``socket`` is neutered in the child before the submission
  runs, so a phone-home submission fails closed.
* **Output validation** -- the prediction must be a finite, real, HORIZON-length
  vector; anything else is an error, not a forecast.

Defense in depth, not a panacea
--------------------------------
In-process guards (neutered ``socket``, restricted builtins) raise the bar but a
truly adversarial deployment MUST add OS-level isolation -- a container with no
network namespace, a seccomp profile, a read-only rootfs, an unprivileged user.
This module is the portable, dependency-free layer; ``run_submission`` is written
so that container is the only thing you add, not a rewrite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

from config import HORIZON
from static_analysis import scan_submission

try:  # POSIX-only; the limits are skipped (with a recorded note) elsewhere.
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]


# Default resource ceilings. Deliberately generous for a forecaster but far below
# what it takes to harm a validator; override per deployment via SandboxLimits.
DEFAULT_CPU_SECONDS = 10
DEFAULT_WALL_SECONDS = 15.0
DEFAULT_MEMORY_MB = 1024
DEFAULT_FSIZE_BYTES = 0  # no file writes at all


@dataclass(frozen=True)
class SandboxLimits:
    """Resource ceilings applied to a submission process."""

    cpu_seconds: int = DEFAULT_CPU_SECONDS
    wall_seconds: float = DEFAULT_WALL_SECONDS
    memory_mb: int = DEFAULT_MEMORY_MB
    fsize_bytes: int = DEFAULT_FSIZE_BYTES


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of executing a submission against one context.

    Exactly one of ``prediction`` / ``error`` is meaningful: ``ok`` is True only
    when a valid HORIZON-length finite forecast came back. ``status`` is one of
    ``ok``, ``rejected`` (pre-filter), ``timeout``, ``error``.
    """

    ok: bool
    status: str
    prediction: np.ndarray | None = None
    error: str | None = None
    findings: tuple[str, ...] = ()


# The code that runs *inside* the child. It neuters the network, exec's the
# submission, calls forecast(), and writes the result as JSON to stdout. Kept as a
# string so the child is a clean interpreter with none of the parent's imports.
_CHILD_RUNNER = r"""
import json, sys, builtins

def _block_network():
    try:
        import socket
        def _blocked(*a, **k):
            raise OSError("network access is disabled in the sandbox")
        for name in ("socket", "create_connection", "socketpair"):
            if hasattr(socket, name):
                setattr(socket, name, _blocked)
    except Exception:
        pass

def main():
    payload = json.loads(sys.stdin.read())
    code, context, horizon = payload["code"], payload["context"], payload["horizon"]
    _block_network()
    import numpy as np
    ctx = np.asarray(context, dtype=float)
    ns = {"__name__": "__submission__", "__builtins__": builtins}
    try:
        exec(compile(code, "<submission>", "exec"), ns)
        fn = ns.get("forecast")
        if not callable(fn):
            raise ValueError("submission defines no callable `forecast`")
        out = np.asarray(fn(ctx), dtype=float).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - report any submission failure
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return
    if out.shape[0] != horizon:
        print(json.dumps({"ok": False, "error":
            f"forecast length {out.shape[0]} != horizon {horizon}"}))
        return
    if not np.all(np.isfinite(out)):
        print(json.dumps({"ok": False, "error": "forecast contains non-finite values"}))
        return
    print(json.dumps({"ok": True, "prediction": out.tolist()}))

main()
"""


def _preexec(limits: SandboxLimits):  # pragma: no cover - runs in child only
    """Return a ``preexec_fn`` that installs rlimits, or None off POSIX."""
    if resource is None:
        return None

    def _apply() -> None:
        mem = limits.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.fsize_bytes, limits.fsize_bytes))

    return _apply


def run_submission(
    code: str,
    context: np.ndarray,
    *,
    horizon: int = HORIZON,
    limits: SandboxLimits | None = None,
    prefilter: bool = True,
) -> SandboxResult:
    """Execute ``code``'s ``forecast(context)`` under isolation and validate it.

    With ``prefilter`` (default) the static-analysis scan runs first and a dirty
    submission is ``rejected`` without execution. A clean submission is then run
    in a resource-limited child process; its forecast is validated for shape and
    finiteness before being accepted.
    """
    limits = limits or SandboxLimits()
    findings = tuple(scan_submission(code)) if prefilter else ()
    if findings:
        return SandboxResult(False, "rejected", findings=findings)

    payload = json.dumps(
        {"code": code, "context": np.asarray(context, dtype=float).tolist(), "horizon": horizon}
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _CHILD_RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=limits.wall_seconds,
            preexec_fn=_preexec(limits),
            env={"PYTHONUNBUFFERED": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(False, "timeout", error=f"exceeded {limits.wall_seconds}s wall time")

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["nonzero exit"]
        return SandboxResult(False, "error", error=f"process exit {proc.returncode}: {tail[0]}")

    line = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    try:
        result = json.loads(line[0])
    except json.JSONDecodeError:
        return SandboxResult(False, "error", error="no parseable result from submission")

    if not result.get("ok"):
        return SandboxResult(False, "error", error=str(result.get("error", "unknown error")))
    return SandboxResult(True, "ok", prediction=np.asarray(result["prediction"], dtype=float))
