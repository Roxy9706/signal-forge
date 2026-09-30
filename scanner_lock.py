# P5 Scanner Lock — single-process guard for the paper observation scanner.
#
# SignalForge is intentionally a single-scanner-process system.  This module
# provides an fcntl.flock advisory exclusive lock held for the lifetime of the
# scanner process so that two scanner.py processes cannot write to the same
# JSONL ledger simultaneously.
#
# The lock is independent of scanner_runner.py's short-lived operation lock:
#   - runner lock  → protects start/stop/restart lifecycle operations
#   - scanner lock → protects the scanner loop itself (including direct
#                    `python scanner.py paper ...` invocations)
# Both are desirable because the runner releases its lock after spawning the
# child scanner process, leaving a window where a second runner or a manually
# invoked scanner could write to the same ledger.
#
# The kernel releases the flock automatically when the owning process exits
# (including on SIGINT/SIGTERM), so no zombie-locked-file cleanup is required.
# The PID written into the lock file is purely diagnostic.

from __future__ import annotations

import fcntl
import os
import sys
import signal as _signal
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lock state (module-level, one scanner process = one lock)
# ---------------------------------------------------------------------------

_lock_fd: Optional[int] = None
_lock_log_path: Optional[str] = None
_registered: bool = False


# ---------------------------------------------------------------------------
# Lock path
# ---------------------------------------------------------------------------

def _lock_path_for_log(log_path: str) -> Path:
    """Sidecar lock file placed next to the JSONL ledger.

    Using a sidecar next to the ledger (instead of signalforge_runtime/) makes
    the lock scope obvious: two scanners writing to different logs do not
    conflict.  The runner's operation lock stays in signalforge_runtime/ for
    lifecycle coordination.
    """
    p = Path(log_path)
    return p.parent / (p.name + ".scanner.lock")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def acquire_scanner_lock(log_path: str) -> Optional[int]:
    """Try to acquire the exclusive scanner lock for *log_path*.

    Returns the open lock-fd on success (caller MUST pass it to
    release_scanner_lock on exit), or None if another scanner process
    already holds the lock for this ledger.

    Special log paths (e.g. /dev/null used by tests) are skipped:
    no lock file is created and the call succeeds trivially.
    """
    global _lock_fd, _lock_log_path

    if _lock_fd is not None:
        return _lock_fd  # already acquired for this process

    # Special-test path: /dev/null has no parent directory we can write to.
    p = Path(log_path)
    if p.parent == Path("/dev") or str(p.parent) == "/dev":
        _lock_log_path = log_path
        _lock_fd = -1  # sentinel: no real lock held
        _register_signal_handlers()
        return -1

    lock = _lock_path_for_log(log_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    _lock_log_path = log_path

    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        print(f"[P5] FATAL: cannot open scanner lock {lock}: {exc}", file=sys.stderr)
        return None

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        _describe_and_refuse(lock)
        return None

    # Lock acquired; stamp PID for diagnostics (overwrite stale content).
    _write_pid(lock)
    _lock_fd = fd

    _register_signal_handlers()
    return fd


def release_scanner_lock() -> None:
    """Release the scanner lock and clear the diagnostic PID stamp.

    Safe to call multiple times.  Does NOT delete the lock file — only the
    PID content is cleared, so a stale-empty file is harmless.
    If the lock was acquired with a special path (e.g. /dev/null sentinel),
    this is a no-op.
    """
    global _lock_fd, _lock_log_path

    fd = _lock_fd
    log_path = _lock_log_path
    _lock_fd = None
    _lock_log_path = None

    if fd is None or fd < 0:
        return

    lock = _lock_path_for_log(log_path) if log_path else None
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
    if lock is not None:
        try:
            lock.write_text("", encoding="utf-8")
        except OSError:
            pass


def scanner_locked() -> bool:
    """True if this process currently holds the scanner lock.

    Returns False for the special /dev/null sentinel (-1), since no actual
    lock is held in that case.
    """
    return _lock_fd is not None and _lock_fd >= 0


def describe_holder(log_path: str) -> str:
    """Best-effort diagnostic string for whoever appears to hold the lock."""
    return _describe_holder(_lock_path_for_log(log_path))


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _register_signal_handlers() -> None:
    """Release the lock on SIGINT/SIGTERM so a killed scanner does not
    leave the PID stamp stale (the kernel releases flocks on process exit,
    but we clear the PID stamp too for human diagnostics)."""
    global _registered
    if _registered:
        return
    _registered = True

    def _handler(signum, frame):
        release_scanner_lock()
        # Re-raise via default behavior so Python's default handler still fires.
        if signum == _signal.SIGINT:
            raise KeyboardInterrupt
        os.kill(os.getpid(), signum)

    _signal.signal(_signal.SIGINT, _handler)
    _signal.signal(_signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def _write_pid(lock: Path) -> None:
    try:
        lock.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass


def _describe_holder(lock: Path) -> str:
    try:
        text = lock.read_text(encoding="utf-8").strip()
    except OSError:
        return "unreadable"
    if not text.isdigit():
        return f"stale(python)={text[:32]!r}"
    pid = int(text)
    if pid <= 0:
        return f"stale-pid={pid}"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return f"dead-PID={pid}(lock-reacquirable)"
    except PermissionError:
        return f"live-PID={pid}(unreadable-cmdline)"
    except OSError:
        return f"live-PID={pid}"
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return f"live-PID={pid}"
    parts = [c for c in cmdline.split("\x00") if c]
    if not parts:
        return f"live-PID={pid}"
    return f"live-PID={pid} cmd={parts[0]!r}" + (
        f" +{len(parts)-1}more" if len(parts) > 1 else ""
    )


def _describe_and_refuse(lock: Path) -> None:
    holder = _describe_holder(lock)
    # Derive the underlying JSONL path from the sidecar lock name for a clear message.
    log_stem = lock.name.replace(".scanner.lock", "")
    log_dir = lock.parent
    print(
        f"[P5] Another scanner process holds the lock for "
        f"{log_dir / log_stem}.jsonl — not starting.  Lock file: {lock}.  "
        f"Holder: {holder}",
        file=sys.stderr,
    )
