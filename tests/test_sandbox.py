"""Hardened sandbox: correctness plus the containment guarantees."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frugal_router.sandbox import RunResult, run_python  # noqa: E402


def test_happy_path_stdout():
    res = run_python("print(2 + 2)")
    assert res.stdout.strip() == "4"
    assert res.returncode == 0
    assert not res.timed_out


def test_argv_passthrough():
    res = run_python("import sys; print(sys.argv[1])", argv=["hello"])
    assert res.stdout.strip() == "hello"


def test_timeout_returns_flag_promptly():
    t0 = time.monotonic()
    res = run_python("while True:\n    pass", timeout_s=1.0)
    assert res.timed_out
    assert time.monotonic() - t0 < 3.0


def test_socket_creation_blocked():
    res = run_python(
        "import socket\n"
        "s = socket.socket()\n"
        "print('OPENED')\n"
    )
    assert "OPENED" not in res.stdout
    assert res.returncode != 0
    assert "disabled" in res.stderr.lower() or "oserror" in res.stderr.lower()


def test_socket_connect_blocked():
    res = run_python(
        "import socket\n"
        "socket.create_connection(('93.184.216.34', 80), timeout=2)\n"
        "print('CONNECTED')\n"
    )
    assert "CONNECTED" not in res.stdout
    assert res.returncode != 0


def test_memory_bomb_dies_not_the_parent():
    # Linux: RLIMIT_AS kills it fast with MemoryError. macOS: RLIMIT_AS is not
    # honoured, so the wall timeout is the backstop. Either way the parent must
    # survive and get a RunResult.
    res = run_python(
        "buf = []\n"
        "while True:\n"
        "    buf.append(bytes(20 * 1024 * 1024))\n",
        memory_mb=256, timeout_s=3.0,
    )
    assert isinstance(res, RunResult)
    assert res.timed_out or res.returncode != 0


def test_output_flood_truncated():
    res = run_python("print('x' * 1_000_000)")
    assert len(res.stdout) <= 64 * 1024


def test_never_raises_on_garbage():
    res = run_python("this is not valid python !!!")
    assert isinstance(res, RunResult)
    assert res.returncode != 0
