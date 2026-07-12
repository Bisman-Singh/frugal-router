"""Hardened execution of model-written Python.

The provable tiers (program-of-thought math, execution-verified code) run code
a small model authored. That code is untrusted: it must not open a socket
(the judge disqualifies any non-Fireworks egress, so an accidental connection
is fatal), exhaust memory, spawn processes, or write large files.

This module is the single caged primitive both callers route through. It is
stdlib-only so it imports and runs under the frozen judge environment with no
new dependency and no cold-start cost.

Containment layers, defence in depth:
  * ``python -I`` isolated mode (no site, no user env hooks) and a cleared
    environment (``env={}``) so no secrets or PYTHON* switches leak in.
  * A prepended prelude that neuters ``socket.socket`` / ``_socket.socket``
    before any user statement executes, so network calls raise in-process even
    without ``--network none``.
  * POSIX ``rlimit``s (CPU seconds, address space, open files, file size,
    processes) applied in the child between fork and exec. Each limit is
    best-effort: a platform that rejects one (macOS does not honour
    ``RLIMIT_AS``) simply skips it, and the wall-clock timeout is the backstop.
  * A wall-clock timeout that kills the child.

``run_python`` never raises: every failure mode is reported in the RunResult.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

_CAP = 64 * 1024  # bytes retained per stream

# Executed before any user statement. Prints nothing, so stdout is untouched.
_PRELUDE = (
    "import socket as _s, _socket as _sl\n"
    "def _blocked(*a, **k):\n"
    "    raise OSError('network access is disabled in the sandbox')\n"
    "_s.socket = _blocked\n"
    "_sl.socket = _blocked\n"
    "_s.create_connection = _blocked\n"
    "try:\n"
    "    _s.socketpair = _blocked\n"
    "except Exception:\n"
    "    pass\n"
    "del _s, _sl, _blocked\n"
)


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


def _decode(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", "replace")
    return raw[:_CAP]


def _make_preexec(memory_mb: int, timeout_s: float):
    """Build the child-side rlimit hook, or None where unsupported."""
    if os.name != "posix":
        return None

    def _apply():
        try:
            import resource
        except Exception:
            return

        def _set(name: str, value: int) -> None:
            const = getattr(resource, name, None)
            if const is None:
                return
            try:
                resource.setrlimit(const, (value, value))
            except (ValueError, OSError):
                pass  # a limit the platform rejects; the timeout still applies

        cpu = int(timeout_s) + 1  # CPU-seconds backstop under the wall timeout
        _set("RLIMIT_CPU", cpu)
        mem = max(64, memory_mb) * 1024 * 1024
        _set("RLIMIT_AS", mem)
        _set("RLIMIT_DATA", mem)
        _set("RLIMIT_FSIZE", 1 << 20)   # 1 MB of file writes
        _set("RLIMIT_NOFILE", 64)
        _set("RLIMIT_NPROC", 256)       # contains fork bombs; per-uid, harmless to set

    return _apply


def run_python(code: str, *, timeout_s: float = 6.0, memory_mb: int = 512,
               argv: list[str] | None = None) -> RunResult:
    """Run ``code`` as an isolated, resource-limited, network-blocked process.

    Returns a RunResult; never raises. stdout/stderr are decoded and truncated
    to 64 KB each. ``timed_out`` is True when the wall-clock limit fired.
    """
    program = _PRELUDE + "\n" + (code or "")
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         dir=tempfile.gettempdir()) as fh:
            fh.write(program)
            path = fh.name
    except Exception as exc:  # pragma: no cover - tmp write failure is exotic
        return RunResult("", f"sandbox setup failed: {type(exc).__name__}: {exc}",
                         -1, False)

    cmd = [sys.executable or "python3", "-I", path, *(argv or [])]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout_s, env={},
            cwd=tempfile.gettempdir(),
            preexec_fn=_make_preexec(memory_mb, timeout_s),
        )
        return RunResult(_decode(proc.stdout), _decode(proc.stderr),
                         proc.returncode, False)
    except subprocess.TimeoutExpired as exc:
        return RunResult(_decode(exc.stdout), _decode(exc.stderr), -1, True)
    except Exception as exc:
        return RunResult("", f"sandbox error: {type(exc).__name__}: {exc}", -1, False)
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
