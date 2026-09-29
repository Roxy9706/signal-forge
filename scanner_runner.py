#!/usr/bin/env python3
# SignalForge scanner runner — background lifecycle management for the
# paper observation scanner (signal-generation only, no execution).
#
# Boundary (fail closed):
#   - This runner ONLY manages the scanner process lifecycle. It does NOT
#     start the dashboard, demo_auto, or any execution component.
#   - It never places/cancels/modifies orders and never calls Binance
#     private endpoints. The scanner itself is GET-only public-market-data.
#   - Scanner stdout/stderr is written to a dedicated runtime log, NEVER to
#     data/signals/paper_trade.jsonl (the signal ledger). The scanner's own
#     --log still points at the ledger (unchanged scanner behavior).
#   - No API keys, secrets, or account info are stored, logged, or surfaced.
#
# Source of truth for process state: PID file + flock advisory lock in
# signalforge_runtime/. Data/logs are deliberately separated from the
# signal ledger under data/signals/.
#
# Survival across terminal close (Termux/Linux):
#   start_new_session=True detaches the child into its own session/process
#   group so it is NOT killed by SIGHUP when this controlling terminal exits.
#   This is the stdlib equivalent of `setsid`/`nohup`. Best-effort on a
#   mobile/Termux host — a power-cycle or app kill can still terminate the
#   process; there is NO uncontrolled respawn loop.
from __future__ import annotations
import argparse
import fcntl
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Runtime layout (deliberately outside data/signals/)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "signalforge_runtime"
PID_FILE = RUNTIME / "scanner.pid"
LOCK_FILE = RUNTIME / "scanner.lock"
LOG_DIR = RUNTIME / "logs"
SCANNER_LOG = LOG_DIR / "scanner.log"

# Default scanner invocation (reuses the existing command surface).
DEFAULT_SCAN_CMD = [
    sys.executable,
    str(HERE / "scanner.py"),
    "paper",
    "--auto-discover",
    "--interval",
    "300",
]


def _ensure_runtime() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _read_pid_file() -> Optional[int]:
    if not PID_FILE.is_file():
        return None
    try:
        text = PID_FILE.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return int(text)
    except (ValueError, OSError):
        return None


def _write_pid_file(pid: int) -> None:
    PID_FILE.write_text(str(pid), encoding="utf-8")


def _clear_pid_file() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _acquire_lock() -> Optional[int]:
    """Acquire an exclusive advisory lock. Returns the fd, or None on failure."""
    _ensure_runtime()
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    # Lock held for the lifetime of this runner invocation.
    return fd


def _release_lock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def _is_process_alive(pid: int) -> bool:
    """True if a process with this PID is alive (not a zombie) and we can signal it.

    `os.kill(pid, 0)` succeeds for zombie processes (they still occupy a process
    table slot), so we additionally check /proc/<pid>/stat: a zombie has state
    'Z' and must not be treated as alive for stop/restart purposes.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but not owned — not "ours" for stop purposes.
        return False
    except OSError:
        return False
    # Double-check the process is not a zombie.
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return True  # can't read stat, trust kill(0)
    close = raw.rfind(")")
    if close < 0:
        return True
    fields = raw[close + 2:].split()
    if not fields:
        return True
    state = fields[0]  # 1-character state code
    return state != "Z"


def _proc_start_time_desc(pid: int) -> Optional[str]:
    """Best-effort start-time hint from /proc/<pid>/stat.

    We report elapsed-since-boot as a proxy because boot time is not reliably
    readable on this host. Falls back to None when /proc is unreadable.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2:].split()
    if len(fields) < 20:
        return None
    try:
        starttime_ticks = int(fields[19])
    except (ValueError, IndexError):
        return None
    clk_tck = os.sysconf("SC_CLK_TCK")
    if clk_tck <= 0:
        return None
    elapsed_sec = starttime_ticks / clk_tck
    return f"started ~{elapsed_sec:.0f}s after boot"


def _pid_is_ours(pid: int, cmd_parts: list[str]) -> bool:
    """Best-effort check that PID looks like our scanner process.

    Checks /proc/<pid>/cmdline for the scanner executable path and the 'paper'
    subcommand, so `stop` does not kill an unrelated process that happened to
    reuse the PID.
    """
    if not _is_process_alive(pid):
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    parts = [p for p in cmdline.split("\x00") if p]
    scanner_py = str(HERE / "scanner.py")
    return any(p == scanner_py or p.endswith("scanner.py") for p in parts) and any(
        "paper" in p for p in parts
    )


def _describe_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in cmd)


# ---------------------------------------------------------------------------
# Lifecycle operations
# ---------------------------------------------------------------------------

def start(
    cmd: Optional[list[str]] = None,
    foreground: bool = False,
) -> tuple[bool, str]:
    """Start the scanner.

    In foreground mode the scanner runs in this process group (Ctrl+C propagates);
    otherwise it is detached via start_new_session=True.
    Returns (ok, message).
    """
    cmd = cmd or list(DEFAULT_SCAN_CMD)
    _ensure_runtime()

    lock_fd = _acquire_lock()
    if lock_fd is None:
        return False, "Another scanner runner instance holds the lock — not starting."

    try:
        pid = _read_pid_file()
        if pid is not None and _is_process_alive(pid) and _pid_is_ours(pid, cmd):
            return False, f"Scanner already running (PID {pid}). Use restart or stop first."

        if pid is not None and _is_process_alive(pid) and not _pid_is_ours(pid, cmd):
            return False, (
                f"PID file references PID {pid} but it is not our scanner process. "
                "Stop it manually or remove signalforge_runtime/scanner.pid if you are sure."
            )

        # Ensure the log directory exists before opening the log fd.
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_fd = os.open(str(SCANNER_LOG), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                start_new_session=not foreground,
                close_fds=True,
                cwd=str(HERE),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        finally:
            os.close(log_fd)

        _write_pid_file(proc.pid)
        mode = "foreground" if foreground else "background"
        return (
            True,
            f"Scanner started (PID {proc.pid}) in {mode} mode. "
            f"Command: {_describe_cmd(cmd)}. Log: {SCANNER_LOG}.",
        )
    finally:
        _release_lock(lock_fd)


def stop(force: bool = False, cmd: Optional[list[str]] = None) -> tuple[bool, str]:
    """Stop the scanner process managed by this runner.

    Only kills the PID recorded in the runner's PID file, and only if that PID
    still looks like our scanner process.
    """
    cmd = cmd or list(DEFAULT_SCAN_CMD)
    _ensure_runtime()
    lock_fd = _acquire_lock()
    try:
        pid = _read_pid_file()
        if pid is None or not _is_process_alive(pid):
            _clear_pid_file()
            return True, "Scanner is not running (no live PID)."

        if not _pid_is_ours(pid, cmd):
            return False, (
                f"PID {pid} in PID file is not our scanner process. "
                "Not killing it. Remove signalforge_runtime/scanner.pid manually if stale."
            )

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            if not _is_process_alive(pid):
                _clear_pid_file()
                return True, "Scanner stopped (exited before we could signal)."
            return False, f"Failed to send SIGTERM to PID {pid}: {e}"

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if not _is_process_alive(pid):
                _clear_pid_file()
                return True, "Scanner stopped (SIGTERM)."
            time.sleep(0.2)

        if force:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            time.sleep(0.3)
            if not _is_process_alive(pid):
                _clear_pid_file()
                return True, "Scanner stopped (SIGKILL after SIGTERM timeout)."
            return False, f"Scanner PID {pid} still alive after SIGKILL."
        else:
            return False, (
                f"Scanner PID {pid} did not exit within 10s of SIGTERM. "
                "Use stop --force to SIGKILL it."
            )
    finally:
        _release_lock(lock_fd)


def restart(force_stop: bool = False, cmd: Optional[list[str]] = None) -> tuple[bool, str]:
    """Stop (if running) then start — a single ordered operation, NOT an
    infinite respawn loop."""
    cmd = cmd or list(DEFAULT_SCAN_CMD)
    _, stop_msg = stop(force=force_stop, cmd=cmd)
    ok, msg = start(cmd=cmd)
    if not ok:
        return False, f"Restart failed during start: {msg}"
    return True, f"Restarted. {msg}"


def status(cmd: Optional[list[str]] = None) -> dict:
    """Return a serializable status dict for the scanner runner."""
    cmd = cmd or list(DEFAULT_SCAN_CMD)
    pid = _read_pid_file()
    alive = pid is not None and _is_process_alive(pid)
    ours = pid is not None and alive and _pid_is_ours(pid, cmd)
    start_desc = _proc_start_time_desc(pid) if (alive and ours and pid is not None) else None

    return {
        "runner": {
            "pid_file": str(PID_FILE),
            "lock_file": str(LOCK_FILE),
            "log_path": str(SCANNER_LOG),
            "runtime_dir": str(RUNTIME),
        },
        "state": "running" if (alive and ours) else ("stopped" if pid is None or not alive else "stale"),
        "pid": pid if (alive and ours) else None,
        "running": bool(alive and ours),
        "start_time": start_desc,
        "log_path": str(SCANNER_LOG),
        "command": _describe_cmd(cmd),
        "pid_file_present": PID_FILE.is_file(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SignalForge scanner runner — background lifecycle for the paper scanner",
        prog="scanner_runner.py",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_start = sub.add_parser("start", help="Start the scanner in the background")
    p_start.add_argument(
        "--cmd",
        nargs=argparse.REMAINDER,
        default=None,
        help="Override scanner command (default: python scanner.py paper --auto-discover --interval 300). "
        "Pass the full command after --cmd.",
    )
    p_start.add_argument(
        "--foreground",
        action="store_true",
        default=False,
        help="Run in foreground (do not detach). Primarily for debugging.",
    )

    sub.add_parser("stop", help="Stop the scanner process managed by this runner")
    sub.add_parser("restart", help="Stop (if running) and start")
    sub.add_parser("status", help="Show scanner runner status")

    p_stop = sub.add_parser("stop-force", help="Stop with SIGKILL if SIGTERM times out")
    p_stop.add_argument(
        "--cmd",
        nargs=argparse.REMAINDER,
        default=None,
        help="Override scanner command (for PID-ownership check).",
    )

    return parser


def _resolve_cmd_from_args(args: argparse.Namespace) -> Optional[list[str]]:
    if getattr(args, "cmd", None):
        return list(args.cmd)
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd = _resolve_cmd_from_args(args)

    if args.action == "start":
        if not getattr(args, "foreground", False):
            ok, msg = start(cmd=cmd)
            print(msg)
            if not ok:
                sys.exit(1)
            return

        _ensure_runtime()
        print(f"[runner] Running scanner in foreground (log: {SCANNER_LOG})")
        log_fd = os.open(str(SCANNER_LOG), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            proc = subprocess.Popen(
                cmd or list(DEFAULT_SCAN_CMD),
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                cwd=str(HERE),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            proc.wait()
        finally:
            os.close(log_fd)
        return

    elif args.action == "stop":
        ok, msg = stop(force=False, cmd=cmd)
        print(msg)
        if not ok:
            sys.exit(1)

    elif args.action == "stop-force":
        ok, msg = stop(force=True, cmd=cmd)
        print(msg)
        if not ok:
            sys.exit(1)

    elif args.action == "restart":
        ok, msg = restart(force_stop=False, cmd=cmd)
        print(msg)
        if not ok:
            sys.exit(1)

    elif args.action == "status":
        print(__import__("json").dumps(status(cmd=cmd), indent=2, default=str))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
