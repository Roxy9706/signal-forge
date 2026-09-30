# Tests for scanner_lock.py — single-process guard for the paper scanner.
from __future__ import annotations
import fcntl
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scanner_lock as lock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lock_path(tmp_path: Path) -> Path:
    """Create a temp directory and return a lock path inside it."""
    d = tmp_path / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d / "test.log.scanner.lock"


# ---------------------------------------------------------------------------
# acquire / release
# ---------------------------------------------------------------------------

class TestAcquireRelease:
    def test_acquire_returns_fd_on_success(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        fd = lock.acquire_scanner_lock(str(lp) + ".jsonl")
        assert fd is not None
        assert fd >= 0
        try:
            assert lock.scanner_locked() is True
        finally:
            lock.release_scanner_lock()

    def test_acquire_returns_same_fd_on_repeated_call(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        fd1 = lock.acquire_scanner_lock(str(lp) + ".jsonl")
        assert fd1 is not None
        try:
            fd2 = lock.acquire_scanner_lock(str(lp) + ".jsonl")
            assert fd2 is fd1
        finally:
            lock.release_scanner_lock()

    def test_release_clears_state(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        fd = lock.acquire_scanner_lock(str(lp) + ".jsonl")
        assert fd is not None
        lock.release_scanner_lock()
        assert lock.scanner_locked() is False

    def test_release_idempotent(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        lock.release_scanner_lock()  # no-op when not held
        lock.release_scanner_lock()  # still no-op
        assert lock.scanner_locked() is False

    def test_lock_file_created(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"
        assert lock.acquire_scanner_lock(log_path) is not None
        try:
            lock_path = lock._lock_path_for_log(log_path)
            assert lock_path.exists()
            content = lock_path.read_text(encoding="utf-8").strip()
            assert content.isdigit()
            assert int(content) == os.getpid()
        finally:
            lock.release_scanner_lock()


# ---------------------------------------------------------------------------
# Second process cannot acquire
# ---------------------------------------------------------------------------

class TestSecondProcessRefused:
    def test_second_process_cannot_acquire_same_log(self, tmp_path: Path) -> None:
        """Two processes fighting for the same log-path lock — second loses."""
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"

        # Hold the lock in this process
        fd = lock.acquire_scanner_lock(log_path)
        assert fd is not None
        try:
            # Spawn a child that tries to acquire the same lock
            script = f"""
import sys, os
sys.path.insert(0, {sys.path[0]!r})
import scanner_lock as L
log_path = {log_path!r}
result = L.acquire_scanner_lock(log_path)
if result is not None:
    L.release_scanner_lock()
    print("ACQUIRED")
    sys.exit(0)
else:
    print("REJECTED")
    sys.exit(0)
"""
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            out = proc.stdout.strip()
            assert out == "REJECTED", f"Expected REJECTED but got {out!r}; stderr={proc.stderr!r}"
            assert proc.returncode == 0
        finally:
            lock.release_scanner_lock()

    def test_cross_process_scanner_cannot_enter_loop(self, tmp_path: Path) -> None:
        """Two scanner.py-like processes cannot both enter the scanner loop.

        This is the cross-process integration test: a process that successfully
        acquires the scanner lock holds it for the duration of its work, and
        a second process targeting the same log is refused at the lock boundary.
        """
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"
        ready_file = tmp_path / "lock_holder_ready"

        # Process A: acquire the lock, signal ready via a file, hold for a while, then release.
        script_a = f"""
import sys, os, time
sys.path.insert(0, {sys.path[0]!r})
import scanner_lock as L
log_path = {log_path!r}
ready_file = {str(ready_file)!r}

if not L.acquire_scanner_lock(log_path):
    print("A_REJECTED")
    sys.exit(1)
# Signal that we hold the lock by creating a ready file
with open(ready_file, "w") as f:
    f.write(str(os.getpid()))
print(f"A_HOLDING_PID={{os.getpid()}}", flush=True)
try:
    # Hold the lock for a generous window so B has time to try
    time.sleep(10)
    print("A_DONE", flush=True)
finally:
    L.release_scanner_lock()
    try:
        os.unlink(ready_file)
    except OSError:
        pass
print("A_RELEASED", flush=True)
"""

        # Process B: wait for A's ready signal, then try ONCE to acquire.
        # Must be rejected — a single attempt, no polling loop.
        script_b = f"""
import sys, os, time
sys.path.insert(0, {sys.path[0]!r})
import scanner_lock as L
log_path = {log_path!r}
ready_file = {str(ready_file)!r}

# Wait for A to signal it holds the lock (max 5s)
deadline = time.monotonic() + 5.0
while time.monotonic() < deadline:
    if os.path.exists(ready_file):
        break
    time.sleep(0.05)
else:
    print("B_TIMEOUT_WAITING")
    sys.exit(0)

# Small extra delay to ensure A's flock is fully established
time.sleep(0.2)

# Single attempt to acquire — must fail
result = L.acquire_scanner_lock(log_path)
if result is not None:
    L.release_scanner_lock()
    print(f"B_ACQUIRED_PID={{os.getpid()}}")
    sys.exit(0)
else:
    print(f"B_REJECTED_PID={{os.getpid()}}")
    sys.exit(0)
"""

        proc_a = subprocess.Popen(
            [sys.executable, "-c", script_a],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # Wait for A to signal it is holding the lock
            deadline = time.monotonic() + 5.0
            a_holding = False
            while time.monotonic() < deadline:
                line = proc_a.stdout.readline()
                if not line:
                    break
                if "A_HOLDING_PID=" in line:
                    a_holding = True
                    break
            assert a_holding, f"Process A did not signal holding"

            # Now B should be rejected
            proc_b = subprocess.run(
                [sys.executable, "-c", script_b],
                capture_output=True,
                text=True,
                timeout=10,
            )
            b_out = proc_b.stdout.strip()
            assert "B_REJECTED" in b_out, (
                f"Cross-process test failed: B should be rejected while A holds the lock. "
                f"B output: {b_out!r}"
            )
        finally:
            # Ensure A finishes and releases
            proc_a.wait(timeout=15)
            a_out = proc_a.stdout.read()
            assert "A_RELEASED" in a_out, f"Process A did not release; stdout={a_out!r}"


    def test_different_log_paths_do_not_conflict(self, tmp_path: Path) -> None:
        """Two scanners writing to different logs should both acquire."""
        lock1_log = str(tmp_path / "a.jsonl")
        lock2_log = str(tmp_path / "b.jsonl")

        fd1 = lock.acquire_scanner_lock(lock1_log)
        assert fd1 is not None
        try:
            fd2 = lock.acquire_scanner_lock(lock2_log)
            assert fd2 is not None
            try:
                assert lock.scanner_locked() is True
            finally:
                lock.release_scanner_lock()  # releases lock2
        finally:
            lock.release_scanner_lock()  # releases lock1


# ---------------------------------------------------------------------------
# Stale lock handling
# ---------------------------------------------------------------------------

class TestStaleLock:
    def test_dead_pid_lock_is_acquirable(self, tmp_path: Path) -> None:
        """A lock held by a dead process is acquirable by a new process.

        The fcntl.flock is released by the kernel when the owning process dies,
        so the file-based PID stamp (diagnostic only) does not block re-acquisition.
        """
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"

        # Spawn a child that acquires the lock then dies without releasing.
        script = f"""
import sys, os, signal
sys.path.insert(0, {sys.path[0]!r})
import scanner_lock as L
log_path = {log_path!r}
fd = L.acquire_scanner_lock(log_path)
if fd is None:
    print("CHILD_ACQUIRE_FAILED")
    sys.exit(1)
# Write our PID for the parent to verify, then die without releasing.
lock_path = L._lock_path_for_log(log_path)
lock_path.write_text(str(os.getpid()), encoding="utf-8")
print(f"CHILD_HOLDING_PID={{os.getpid()}}", flush=True)
# Die abruptly so the lock is released by the kernel, not by our code.
os.kill(os.getpid(), signal.SIGKILL)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "CHILD_HOLDING_PID=" in proc.stdout
        child_pid = int(proc.stdout.strip().split("CHILD_HOLDING_PID=")[1])
        # Verify child is dead via os.kill(pid, 0)
        try:
            os.kill(child_pid, 0)
            assert False, f"Child PID {child_pid} should be dead"
        except ProcessLookupError:
            pass  # expected

        # Now the kernel has released the flock (process died).
        # A new process should be able to acquire.
        fd2 = lock.acquire_scanner_lock(log_path)
        assert fd2 is not None
        try:
            holder = lock.describe_holder(log_path)
            # Our process now holds the lock; the stale child PID is gone.
            assert f"live-PID={os.getpid()}" in holder
        finally:
            lock.release_scanner_lock()

    def test_holder_description_for_dead_pid(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"
        lock_path = lock._lock_path_for_log(log_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(str(999_999_999), encoding="utf-8")
        desc = lock.describe_holder(log_path)
        assert "dead-PID=999999999" in desc

    def test_holder_description_for_stale_non_pid_content(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"
        lock_path = lock._lock_path_for_log(log_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("not-a-pid", encoding="utf-8")
        desc = lock.describe_holder(log_path)
        assert "stale" in desc.lower()


# ---------------------------------------------------------------------------
# Live owner's lock is never removed by another process
# ---------------------------------------------------------------------------

class TestLiveOwnerProtection:
    def test_second_process_does_not_delete_lock_file(self, tmp_path: Path) -> None:
        """Even when refused, a second process must not remove the lock file."""
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"

        fd = lock.acquire_scanner_lock(log_path)
        assert fd is not None
        lock_path = lock._lock_path_for_log(log_path)
        try:
            # Spawn a child that tries to acquire and then checks if the lock file still exists
            script = f"""
import sys, os
sys.path.insert(0, {sys.path[0]!r})
import scanner_lock as L
log_path = {log_path!r}
result = L.acquire_scanner_lock(log_path)
if result is not None:
    L.release_scanner_lock()
print("DONE")
"""
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert proc.returncode == 0, f"Child failed: {proc.stderr!r}"
            # The lock file must still exist and still contain our PID
            assert lock_path.exists()
            content = lock_path.read_text(encoding="utf-8").strip()
            assert content == str(os.getpid()), f"Lock file content changed: {content!r}"
        finally:
            lock.release_scanner_lock()


# ---------------------------------------------------------------------------
# Lock released after clean shutdown
# ---------------------------------------------------------------------------

class TestLockReleasedOnShutdown:
    def test_release_on_normal_exit(self, tmp_path: Path) -> None:
        """After release_scanner_lock + normal exit, the lock is available."""
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"

        fd = lock.acquire_scanner_lock(log_path)
        assert fd is not None
        lock.release_scanner_lock()
        assert lock.scanner_locked() is False

        # A new process should now be able to acquire
        script = f"""
import sys
sys.path.insert(0, {sys.path[0]!r})
import scanner_lock as L
log_path = {log_path!r}
result = L.acquire_scanner_lock(log_path)
if result is None:
    print("FAIL")
    sys.exit(1)
L.release_scanner_lock()
print("OK")
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.stdout.strip() == "OK"

    def test_pid_stamp_cleared_on_release(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"
        lock_path = lock._lock_path_for_log(log_path)

        fd = lock.acquire_scanner_lock(log_path)
        assert fd is not None
        assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())
        lock.release_scanner_lock()
        # After release, the file is cleared (empty string)
        assert lock_path.read_text(encoding="utf-8").strip() == ""


# ---------------------------------------------------------------------------
# scanner_locked() predicate
# ---------------------------------------------------------------------------

class TestScannerLockedPredicate:
    def test_false_when_not_acquired(self, tmp_path: Path) -> None:
        assert lock.scanner_locked() is False

    def test_true_after_acquire(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"
        lock.acquire_scanner_lock(log_path)
        try:
            assert lock.scanner_locked() is True
        finally:
            lock.release_scanner_lock()

    def test_false_after_release(self, tmp_path: Path) -> None:
        lp = _make_lock_path(tmp_path)
        log_path = str(lp) + ".jsonl"
        lock.acquire_scanner_lock(log_path)
        lock.release_scanner_lock()
        assert lock.scanner_locked() is False
