#!/usr/bin/env python3
# Tests for scanner_runner.py lifecycle helpers (no live Binance).
from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scanner_runner as runner

_FAKE_SCANNER = ROOT / "scanner_fake_for_tests.py"


def _write_fake_scanner(tmp_path: Path, *, sleep_after_pid: float = 0.0, exit_code: int = 0) -> Path:
    src = f"""
import os, sys, time

PID_FILE = r"{tmp_path / 'fake_scanner.pid'}"
with open(PID_FILE, "w") as f:
    f.write(str(os.getpid()))

print(f"[fake-scanner] started PID {{os.getpid()}}", flush=True)
sys.stdout.flush()

if {sleep_after_pid} > 0:
    time.sleep({sleep_after_pid})

sys.exit({exit_code})
"""
    path = tmp_path / "fake_scanner.py"
    path.write_text(src, encoding="utf-8")
    return path


def _configure_runner(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    runner.RUNTIME = runtime
    runner.PID_FILE = runtime / "scanner.pid"
    runner.LOCK_FILE = runtime / "scanner.lock"
    runner.LOG_DIR = runtime / "logs"
    runner.SCANNER_LOG = runtime / "logs" / "scanner.log"


# ---------------------------------------------------------------------------
# _pid helpers
# ---------------------------------------------------------------------------


class TestPidHelpers:
    def test_is_process_alive_own_pid(self) -> None:
        assert runner._is_process_alive(os.getpid()) is True

    def test_is_process_alive_dead_pid(self) -> None:
        assert runner._is_process_alive(999_999_999) is False

    def test_is_process_alive_zero(self) -> None:
        assert runner._is_process_alive(0) is False

    def test_proc_start_time_when_readable(self) -> None:
        desc = runner._proc_start_time_desc(os.getpid())
        assert desc is not None
        assert desc.startswith("started ~")

    def test_proc_start_time_dead_pid(self) -> None:
        assert runner._proc_start_time_desc(999_999_999) is None


# ---------------------------------------------------------------------------
# Duplicate protection
# ---------------------------------------------------------------------------


class TestDuplicateProtection:
    def test_start_rejects_when_lock_held(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        runtime = tmp_path / "runtime"
        lock = runtime / "scanner.lock"
        runtime.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ok, msg = runner.start(cmd=[sys.executable, "-c", "pass"])
            assert ok is False
            assert "lock" in msg.lower()
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
            os.close(fd)

    def test_start_rejects_when_already_running(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        runtime = tmp_path / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        fake = _write_fake_scanner(tmp_path, sleep_after_pid=2.0)
        proc = subprocess.Popen(
            [sys.executable, str(fake)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(tmp_path),
        )
        try:
            time.sleep(0.1)
            runner.PID_FILE.write_text(str(proc.pid), encoding="utf-8")
            ok, msg = runner.start(cmd=[sys.executable, "-c", "pass"])
            assert ok is False
            assert "not our scanner process" in msg
        finally:
            proc.kill()
            proc.wait()

    def test_start_ok_when_no_pid_file(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        ok, msg = runner.start(cmd=[sys.executable, "-c", "pass"])
        assert ok is True
        assert "started" in msg.lower()
        assert runner.PID_FILE.exists()
    def test_start_rejects_when_pid_file_has_unrelated_alive_pid(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        runtime = tmp_path / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        try:
            runner.PID_FILE.write_text(str(proc.pid), encoding="utf-8")
            ok, msg = runner.start(cmd=[sys.executable, "-c", "pass"])
            assert ok is False
            assert "not our scanner process" in msg
        finally:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


class TestStopBehaviour:
    def test_stop_when_not_running_is_ok(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        ok, msg = runner.stop()
        assert ok is True
        assert "not running" in msg.lower()

    def test_stop_terminates_owned_scanner(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        fake = _write_fake_scanner(tmp_path, sleep_after_pid=5.0)
        fake_pid: List[int] = []
        original = runner._pid_is_ours

        def fake_owns(pid: int, cmd_parts: list[str]) -> bool:
            if fake_pid and pid == fake_pid[0]:
                return True
            return original(pid, cmd_parts)

        runner._pid_is_ours = fake_owns
        try:
            proc = subprocess.Popen(
                [sys.executable, str(fake)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=str(tmp_path),
            )
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                pf = tmp_path / "fake_scanner.pid"
                if pf.exists():
                    fake_pid.append(int(pf.read_text(encoding="utf-8").strip()))
                    break
                time.sleep(0.02)
            else:
                proc.kill()
                proc.wait()
                pytest.skip("fake scanner did not start in time")

            runner.PID_FILE.write_text(str(fake_pid[0]), encoding="utf-8")
            ok, msg = runner.stop()
            assert ok is True
            assert "stopped" in msg.lower() or "sigterm" in msg.lower()
            assert not runner._is_process_alive(fake_pid[0])
        finally:
            runner._pid_is_ours = original


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_fields_present(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        s = runner.status()
        required = [
            "runner", "state", "pid", "running", "start_time",
            "log_path", "command", "pid_file_present",
        ]
        for k in required:
            assert k in s, f"Missing status key: {k}"
        assert s["running"] is False
        assert s["state"] == "stopped"
        assert s["pid"] is None
        assert s["log_path"] == str(tmp_path / "runtime" / "logs" / "scanner.log")
        assert "scanner.py" in s["command"]

    def test_status_running_when_alive(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        fake = _write_fake_scanner(tmp_path, sleep_after_pid=5.0)
        fake_pid: List[int] = []
        original = runner._pid_is_ours

        def fake_owns(pid: int, cmd_parts: list[str]) -> bool:
            if fake_pid and pid == fake_pid[0]:
                return True
            return original(pid, cmd_parts)

        runner._pid_is_ours = fake_owns
        try:
            proc = subprocess.Popen(
                [sys.executable, str(fake)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=str(tmp_path),
            )
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                pf = tmp_path / "fake_scanner.pid"
                if pf.exists():
                    fake_pid.append(int(pf.read_text(encoding="utf-8").strip()))
                    break
                time.sleep(0.02)
            else:
                proc.kill()
                proc.wait()
                pytest.skip("fake scanner did not start in time")

            runner.PID_FILE.write_text(str(fake_pid[0]), encoding="utf-8")
            s = runner.status()
            assert s["running"] is True
            assert s["state"] == "running"
            assert s["pid"] == fake_pid[0]
            assert s["start_time"] is not None
            assert s["start_time"].startswith("started ~")
        finally:
            runner._pid_is_ours = original
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# Restart is one-shot
# ---------------------------------------------------------------------------


class TestRestartIsOneShot:
    def test_restart_stops_then_starts(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        fake = _write_fake_scanner(tmp_path, sleep_after_pid=4.0)
        fake_pid: List[int] = []
        original = runner._pid_is_ours

        def fake_owns(pid: int, cmd_parts: list[str]) -> bool:
            if fake_pid and pid == fake_pid[0]:
                return True
            return original(pid, cmd_parts)

        runner._pid_is_ours = fake_owns
        try:
            proc = subprocess.Popen(
                [sys.executable, str(fake)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=str(tmp_path),
            )
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                pf = tmp_path / "fake_scanner.pid"
                if pf.exists():
                    fake_pid.append(int(pf.read_text(encoding="utf-8").strip()))
                    break
                time.sleep(0.02)
            else:
                proc.kill()
                proc.wait()
                pytest.skip("fake scanner did not start in time")

            runner.PID_FILE.write_text(str(fake_pid[0]), encoding="utf-8")
            ok, msg = runner.restart(force_stop=False)
            assert ok is True
            assert "restarted" in msg.lower()
            assert not runner._is_process_alive(fake_pid[0])
            new_pid = runner._read_pid_file()
            assert new_pid is not None
            assert runner._is_process_alive(new_pid)
            assert new_pid != fake_pid[0]
        finally:
            runner._pid_is_ours = original


# ---------------------------------------------------------------------------
# Runner log isolation
# ---------------------------------------------------------------------------


class TestRunnerLogIsolation:
    def test_scanner_log_is_outside_signals_ledger(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        log = runner.SCANNER_LOG
        assert "data/signals" not in str(log)
        assert "paper_trade.jsonl" not in str(log)

    def test_runner_log_gets_scanner_stdout(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        fake = _write_fake_scanner(tmp_path, sleep_after_pid=0.1, exit_code=0)
        ok, msg = runner.start(cmd=[sys.executable, str(fake)])
        assert ok is True
        time.sleep(0.2)
        log_text = runner.SCANNER_LOG.read_text(encoding="utf-8")
        assert "fake-scanner" in log_text
        assert "started" in log_text


# ---------------------------------------------------------------------------
# Stop only kills owned PIDs
# ---------------------------------------------------------------------------


class TestStopOnlyKillsOwned:
    def test_stop_does_not_kill_unrelated_pid_in_file(self, tmp_path: Path) -> None:
        _configure_runner(tmp_path)
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        try:
            runner.PID_FILE.write_text(str(proc.pid), encoding="utf-8")
            ok, msg = runner.stop()
            assert ok is False
            assert "not our scanner process" in msg
            assert runner._is_process_alive(proc.pid)
        finally:
            proc.kill()
            proc.wait()
            runner._clear_pid_file()
