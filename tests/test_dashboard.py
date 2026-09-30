# Dashboard tests - deterministic, no live Binance
from __future__ import annotations
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard.server import (
    get_scanner_status,
    get_open_signals,
    get_recent_signals,
    get_recent_outcomes,
    signal_age_label,
    app,
    _read_jsonl,
    _find_events,
)


def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    """Write JSONL records to a temp file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> str:
    """Create a temporary JSONL file with sample events."""
    path = str(tmp_path / "test.jsonl")
    now_ms = int(time.time() * 1000)

    records: List[Dict[str, Any]] = [
        # Scanner state - running
        {
            "event": "scanner_state",
            "state": "running",
            "poll_time_ms": now_ms,
            "log_path": path,
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "interval_sec": 300,
            "max_cycles": None,
        },
        # Cycle end
        {
            "event": "cycle_end",
            "cycle_number": 42,
            "poll_time_ms": now_ms - 60000,
            "symbols_discovered": 63,
            "symbols_processed": 63,
            "signals_generated": 2,
            "outcomes_resolved": 1,
            "data_quality_events": 0,
            "error_symbol_count": 0,
            "scan_duration_ms": 65000,
            "min_quote_volume_usdt": 50_000_000.0,
        },
        # Active signal - LONG
        {
            "event": "signal",
            "signal_id": "abc123def456789",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "signal_time": now_ms - 120000,
            "entry": 42000.0,
            "tp": 42840.0,
            "sl": 41580.0,
            "score": 73.5,
            "rr": 2.0,
            "timeframe": "5m",
            "horizon_ms": 14400000,
            "strategy_version": "c5c143cf94ed8790",
            "generated_at": now_ms,
        },
        # Active signal - SHORT
        {
            "event": "signal",
            "signal_id": "xyz789ghi012345",
            "symbol": "ETHUSDT",
            "direction": "SHORT",
            "signal_time": now_ms - 60000,
            "entry": 2200.0,
            "tp": 2156.0,
            "sl": 2244.0,
            "score": 69.1,
            "rr": 2.0,
            "timeframe": "5m",
            "horizon_ms": 14400000,
            "strategy_version": "c5c143cf94ed8790",
            "generated_at": now_ms,
        },
        # Resolved signal - no longer active
        {
            "event": "signal",
            "signal_id": "old111old222333",
            "symbol": "SOLUSDT",
            "direction": "LONG",
            "signal_time": now_ms - 3600000,
            "entry": 150.0,
            "tp": 156.0,
            "sl": 147.0,
            "score": 81.2,
            "rr": 2.0,
            "timeframe": "5m",
            "horizon_ms": 14400000,
            "strategy_version": "c5c143cf94ed8790",
            "generated_at": now_ms - 3600000,
        },
        # Outcome for resolved signal
        {
            "event": "tp_hit",
            "signal_id": "old111old222333",
            "symbol": "SOLUSDT",
            "direction": "LONG",
            "outcome": "tp_hit",
            "outcome_time": now_ms - 2400000,
            "entry": 150.0,
            "tp": 156.0,
            "sl": 147.0,
            "gross_r": 4.0,
            "net_r": 3.8,
            "holding_ms": 1200000,
            "fee": 0.15,
            "slippage": 0.10,
            "horizon_ms": 14400000,
            "strategy_version": "c5c143cf94ed8790",
            "close_time": now_ms - 2400000,
        },
        # Data quality event
        {
            "event": "data_quality",
            "symbol": "ADAUSDT",
            "interval": "5m",
            "poll_time": now_ms - 30000,
            "message": "Stale data: latest 5m close_time is old",
            "data_available": False,
            "stale": True,
        },
        # Another outcome
        {
            "event": "expired",
            "signal_id": "expired555",
            "symbol": "DOTUSDT",
            "direction": "SHORT",
            "outcome": "expired",
            "outcome_time": now_ms - 1800000,
            "entry": 7.5,
            "tp": 6.8,
            "sl": 8.1,
            "gross_r": 0.0,
            "net_r": 0.0,
            "holding_ms": 14400000,
            "strategy_version": "c5c143cf94ed8790",
        },
        # Recent signal (generated 30s ago)
        {
            "event": "signal",
            "signal_id": "rec999rec888777",
            "symbol": "LINKUSDT",
            "direction": "LONG",
            "signal_time": now_ms - 30000,
            "entry": 18.5,
            "tp": 19.0,
            "sl": 18.1,
            "score": 75.0,
            "rr": 2.0,
            "timeframe": "5m",
            "horizon_ms": 14400000,
            "strategy_version": "c5c143cf94ed8790",
            "generated_at": now_ms - 30000,
        },
    ]
    _write_jsonl(path, records)
    return path


# ---------------------------------------------------------------------------
# Scanner status tests
# ---------------------------------------------------------------------------


class TestScannerStatus:
    """Test get_scanner_status() parsing from JSONL."""

    def test_running_status(self, sample_jsonl: str) -> None:
        status = get_scanner_status(sample_jsonl)
        assert status["running"] is True
        assert status["state"] == "running"
        assert status["symbol_count"] == 3
        assert status["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        assert status["interval_sec"] == 300
        assert status["max_cycles"] is None

    def test_latest_cycle_present(self, sample_jsonl: str) -> None:
        status = get_scanner_status(sample_jsonl)
        assert status["latest_cycle"] is not None
        cycle = status["latest_cycle"]
        assert cycle["cycle_number"] == 42
        assert cycle["symbols_discovered"] == 63
        assert cycle["symbols_processed"] == 63
        assert cycle["signals_generated"] == 2
        assert cycle["outcomes_resolved"] == 1
        assert cycle["error_symbol_count"] == 0
        assert cycle["data_quality_events"] == 0
        assert cycle["scan_duration_ms"] == 65000

    def test_scanner_status_has_all_brief_fields(self, sample_jsonl: str) -> None:
        status = get_scanner_status(sample_jsonl)
        cycle = status["latest_cycle"]
        required_status = [
            "running", "state", "last_update_ms", "last_update_age_s",
            "symbols", "interval_sec", "max_cycles", "latest_cycle",
            "symbol_count",
        ]
        required_cycle = [
            "cycle_number", "symbols_discovered", "symbols_processed",
            "signals_generated", "outcomes_resolved", "data_quality_events",
            "error_symbol_count", "scan_duration_ms", "poll_time_ms",
        ]
        for key in required_status:
            assert key in status, f"Missing status field: {key}"
        for key in required_cycle:
            assert key in cycle, f"Missing cycle field: {key}"

    def test_no_events_file_empty(self, tmp_path: Path) -> None:
        path = str(tmp_path / "empty.jsonl")
        status = get_scanner_status(path)
        assert status["running"] is False
        assert status["state"] == "unknown"
        assert status["symbol_count"] == 0
        assert status["symbols"] == []
        assert status["latest_cycle"] is None
        assert status["interval_sec"] == 300

    def test_file_does_not_exist(self, tmp_path: Path) -> None:
        path = str(tmp_path / "nonexistent.jsonl")
        status = get_scanner_status(path)
        assert status["running"] is False
        assert status["state"] == "unknown"
        assert status["last_cycle_time"] is None
        assert status["age_since_last_cycle_s"] is None

    def test_stale_state_considered_stopped(self, tmp_path: Path) -> None:
        """If scanner_state is older than 2 poll intervals, considered stopped."""
        path = str(tmp_path / "stale.jsonl")
        old_ms = int(time.time() * 1000) - 2 * 300 * 1000  # 10 min ago
        records = [
            {
                "event": "scanner_state",
                "state": "running",
                "poll_time_ms": old_ms,
                "symbols": ["BTCUSDT"],
                "interval_sec": 300,
            }
        ]
        _write_jsonl(path, records)
        status = get_scanner_status(path)
        # Should be considered stopped because state is stale
        assert status["running"] is False
        assert status["state"] == "stopped"
        assert status["latest_cycle"] is None
        assert status["last_cycle_time"] is None
        assert status["age_since_last_cycle_s"] is None

    def test_fresh_running_with_recent_cycle(self, tmp_path: Path) -> None:
        """Fresh scanner_state + recent cycle_end → RUNNING."""
        path = str(tmp_path / "fresh.jsonl")
        now_ms = int(time.time() * 1000)
        records = [
            {
                "event": "scanner_state",
                "state": "running",
                "poll_time_ms": now_ms,
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "interval_sec": 300,
                "max_cycles": 100,
            },
            {
                "event": "cycle_end",
                "cycle_number": 7,
                "poll_time_ms": now_ms - 30000,
                "symbols_discovered": 62,
                "symbols_processed": 62,
                "signals_generated": 3,
                "outcomes_resolved": 1,
                "data_quality_events": 0,
                "error_symbol_count": 0,
                "scan_duration_ms": 65000,
            },
        ]
        _write_jsonl(path, records)
        status = get_scanner_status(path)
        assert status["running"] is True
        assert status["state"] == "running"
        assert status["symbol_count"] == 2
        assert status["interval_sec"] == 300
        assert status["max_cycles"] == 100
        assert status["latest_cycle"] is not None
        assert status["latest_cycle"]["cycle_number"] == 7
        assert status["last_cycle_time"] == now_ms - 30000
        assert status["age_since_last_cycle_s"] < 60  # recent

    def test_stale_scanner_state_but_recent_cycle(self, tmp_path: Path) -> None:
        """Old scanner_state (Ctrl+C leftover) + recent cycle_end → RUNNING."""
        path = str(tmp_path / "stale_state_recent_cycle.jsonl")
        now_ms = int(time.time() * 1000)
        old_state_ms = now_ms - 3 * 3600 * 1000  # 3 hours ago
        recent_cycle_ms = now_ms - 45000  # 45s ago
        records = [
            {
                "event": "scanner_state",
                "state": "running",
                "poll_time_ms": old_state_ms,
                "symbols": ["BTCUSDT"],
                "interval_sec": 300,
            },
            {
                "event": "cycle_end",
                "cycle_number": 62,
                "poll_time_ms": recent_cycle_ms,
                "symbols_discovered": 62,
                "symbols_processed": 62,
                "signals_generated": 1,
                "outcomes_resolved": 0,
                "data_quality_events": 0,
                "error_symbol_count": 0,
                "scan_duration_ms": 91437,
            },
        ]
        _write_jsonl(path, records)
        status = get_scanner_status(path)
        # The recent cycle_end wins — scanner is still considered running.
        assert status["running"] is True
        assert status["state"] == "running"
        assert status["last_update_ms"] == recent_cycle_ms
        assert status["last_cycle_time"] == recent_cycle_ms
        assert status["age_since_last_cycle_s"] < 60
        # scanner_state metadata (symbols, interval) is still used even when
        # cycle_end provides the fresher liveness timestamp.
        assert status["symbol_count"] == 1
        assert status["symbols"] == ["BTCUSDT"]
        assert status["interval_sec"] == 300

    def test_old_scanner_state_and_old_cycle(self, tmp_path: Path) -> None:
        """Both scanner_state and cycle_end old → STOPPED."""
        path = str(tmp_path / "both_old.jsonl")
        now_ms = int(time.time() * 1000)
        old_ms = now_ms - 3 * 3600 * 1000  # 3 hours ago
        records = [
            {
                "event": "scanner_state",
                "state": "running",
                "poll_time_ms": old_ms,
                "symbols": ["BTCUSDT"],
                "interval_sec": 300,
            },
            {
                "event": "cycle_end",
                "cycle_number": 1,
                "poll_time_ms": old_ms + 5000,
                "symbols_discovered": 62,
                "symbols_processed": 62,
                "signals_generated": 0,
                "outcomes_resolved": 0,
                "data_quality_events": 0,
                "error_symbol_count": 0,
                "scan_duration_ms": 60000,
            },
        ]
        _write_jsonl(path, records)
        status = get_scanner_status(path)
        assert status["running"] is False
        assert status["state"] == "stopped"
        assert status["last_update_ms"] == old_ms + 5000  # max of the two
        assert status["last_cycle_time"] == old_ms + 5000
        assert status["age_since_last_cycle_s"] > 3600 * 2  # hours old
        assert status["latest_cycle"] is not None
        assert status["latest_cycle"]["cycle_number"] == 1

    def test_cycle_end_only_no_scanner_state(self, tmp_path: Path) -> None:
        """Only a cycle_end event (no scanner_state) → running if recent."""
        path = str(tmp_path / "cycle_only.jsonl")
        now_ms = int(time.time() * 1000)
        records = [
            {
                "event": "cycle_end",
                "cycle_number": 3,
                "poll_time_ms": now_ms - 20000,
                "symbols_discovered": 10,
                "symbols_processed": 10,
                "signals_generated": 0,
                "outcomes_resolved": 0,
                "data_quality_events": 0,
                "error_symbol_count": 0,
                "scan_duration_ms": 30000,
            },
        ]
        _write_jsonl(path, records)
        status = get_scanner_status(path)
        assert status["running"] is True
        assert status["state"] == "running"
        assert status["symbol_count"] == 0
        assert status["symbols"] == []
        assert status["max_cycles"] is None
        assert status["interval_sec"] == 300
        assert status["last_cycle_time"] == now_ms - 20000

    def test_missing_state_no_events(self, tmp_path: Path) -> None:
        """No scanner_state and no cycle_end → unknown/stopped."""
        path = str(tmp_path / "empty.jsonl")
        # Create an empty file (no records)
        Path(path).touch()
        status = get_scanner_status(path)
        assert status["running"] is False
        assert status["state"] == "unknown"
        assert status["symbol_count"] == 0
        assert status["symbols"] == []
        assert status["latest_cycle"] is None
        assert status["last_cycle_time"] is None
        assert status["age_since_last_cycle_s"] is None
        assert status["interval_sec"] == 300

    def test_status_displays_age_since_last_cycle(self, sample_jsonl: str) -> None:
        """age_since_last_cycle_s is present and reasonable when a cycle exists."""
        status = get_scanner_status(sample_jsonl)
        assert "age_since_last_cycle_s" in status
        assert "last_cycle_time" in status
        assert status["age_since_last_cycle_s"] is not None
        assert status["last_cycle_time"] is not None
        # sample_jsonl has a cycle_end 60s old → age should be ~60s
        assert 50 < status["age_since_last_cycle_s"] < 70


# ---------------------------------------------------------------------------
# Open signals tests
# ---------------------------------------------------------------------------


class TestOpenSignals:
    """Test get_open_signals() — signals without outcomes."""

    def test_returns_only_unresolved(self, sample_jsonl: str) -> None:
        open_sigs = get_open_signals(sample_jsonl)
        ids = {s["signal_id"] for s in open_sigs}
        # BTCUSDT and ETHUSDT should be open (no outcomes)
        assert "abc123def456789" in ids
        assert "xyz789ghi012345" in ids
        # SOLUSDT and DOTUSDT should NOT be open (they have outcomes)
        assert "old111old222333" not in ids
        assert "expired555" not in ids
        # LINKUSDT should be open
        assert "rec999rec888777" in ids

    def test_empty_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "empty.jsonl")
        open_sigs = get_open_signals(path)
        assert open_sigs == []

    def test_signals_without_outcomes(self, tmp_path: Path) -> None:
        """Signals with no outcome events at all are open."""
        path = str(tmp_path / "signals_only.jsonl")
        now_ms = int(time.time() * 1000)
        records = [
            {
                "event": "signal",
                "signal_id": "sig111",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "signal_time": now_ms,
                "entry": 42000.0,
                "tp": 42840.0,
                "sl": 41580.0,
                "score": 73.5,
                "rr": 2.0,
            }
        ]
        _write_jsonl(path, records)
        open_sigs = get_open_signals(path)
        assert len(open_sigs) == 1
        assert open_sigs[0]["signal_id"] == "sig111"

    def test_all_signals_resolved(self, tmp_path: Path) -> None:
        """If every signal has an outcome, open list is empty."""
        path = str(tmp_path / "all_resolved.jsonl")
        records = [
            {
                "event": "signal",
                "signal_id": "sig111",
                "symbol": "BTCUSDT",
                "direction": "LONG",
            },
            {
                "event": "tp_hit",
                "signal_id": "sig111",
                "symbol": "BTCUSDT",
                "outcome": "tp_hit",
            },
        ]
        _write_jsonl(path, records)
        open_sigs = get_open_signals(path)
        assert open_sigs == []

    def test_no_active_signals_empty_file(self, tmp_path: Path) -> None:
        """No-signal banner is justified when there are zero active signals."""
        path = str(tmp_path / "only_outcomes.jsonl")
        records = [
            {"event": "tp_hit", "signal_id": "s1", "symbol": "BTCUSDT", "outcome": "tp_hit"},
        ]
        _write_jsonl(path, records)
        open_sigs = get_open_signals(path)
        assert open_sigs == []


# ---------------------------------------------------------------------------
# Recent signals tests
# ---------------------------------------------------------------------------


class TestRecentSignals:
    """Test get_recent_signals() — most recent, newest first."""

    def test_returns_newest_first(self, sample_jsonl: str) -> None:
        signals = get_recent_signals(sample_jsonl, limit=10)
        assert len(signals) > 0
        # First should be most recent by generated_at
        assert signals[0]["signal_id"] == "abc123def456789"  # latest generated_at

    def test_limit_applied(self, sample_jsonl: str) -> None:
        signals = get_recent_signals(sample_jsonl, limit=2)
        assert len(signals) <= 2

    def test_empty_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "empty.jsonl")
        signals = get_recent_signals(path)
        assert signals == []

    def test_no_signal_events(self, tmp_path: Path) -> None:
        """File with only outcomes should return empty signal list."""
        path = str(tmp_path / "outcomes_only.jsonl")
        records = [
            {
                "event": "tp_hit",
                "signal_id": "sig111",
                "symbol": "BTCUSDT",
            }
        ]
        _write_jsonl(path, records)
        signals = get_recent_signals(path)
        assert signals == []


# ---------------------------------------------------------------------------
# Recent outcomes tests
# ---------------------------------------------------------------------------


class TestRecentOutcomes:
    """Test get_recent_outcomes() — most recent, newest first."""

    def test_returns_newest_first(self, sample_jsonl: str) -> None:
        outcomes = get_recent_outcomes(sample_jsonl, limit=10)
        assert len(outcomes) > 0
        # Should include tp_hit and expired
        types = {o["event"] for o in outcomes}
        assert "tp_hit" in types or "expired" in types

    def test_limit_applied(self, sample_jsonl: str) -> None:
        outcomes = get_recent_outcomes(sample_jsonl, limit=1)
        assert len(outcomes) <= 1

    def test_empty_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "empty.jsonl")
        outcomes = get_recent_outcomes(path)
        assert outcomes == []

    def test_no_outcome_events(self, tmp_path: Path) -> None:
        """File with only signals should return empty outcome list."""
        path = str(tmp_path / "signals_only.jsonl")
        records = [
            {
                "event": "signal",
                "signal_id": "sig111",
                "symbol": "BTCUSDT",
            }
        ]
        _write_jsonl(path, records)
        outcomes = get_recent_outcomes(path)
        assert outcomes == []

    def test_all_outcome_types(self, tmp_path: Path) -> None:
        """tp_hit, sl_hit, expired all returned."""
        path = str(tmp_path / "all_outcomes.jsonl")
        now_ms = int(time.time() * 1000)
        records = [
            {"event": "tp_hit", "signal_id": "s1", "symbol": "BTCUSDT", "outcome": "tp_hit", "outcome_time": now_ms},
            {"event": "sl_hit", "signal_id": "s2", "symbol": "ETHUSDT", "outcome": "sl_hit", "outcome_time": now_ms - 1000},
            {"event": "expired", "signal_id": "s3", "symbol": "SOLUSDT", "outcome": "expired", "outcome_time": now_ms - 2000},
        ]
        _write_jsonl(path, records)
        outcomes = get_recent_outcomes(path)
        assert len(outcomes) == 3
        types = {o["event"] for o in outcomes}
        assert types == {"tp_hit", "sl_hit", "expired"}


# ---------------------------------------------------------------------------
# Mobile view signal contract tests
# ---------------------------------------------------------------------------


class TestMobileViewSignalContract:
    """Each signal rendered in the mobile view must carry these fields."""

    def test_each_signal_has_mobile_fields(self, sample_jsonl: str) -> None:
        signals = get_recent_signals(sample_jsonl, limit=20)
        assert len(signals) >= 3
        for s in signals:
            assert isinstance(s["signal_id"], str) and s["signal_id"]
            assert isinstance(s["symbol"], str) and s["symbol"]
            assert s["direction"] in ("LONG", "SHORT")
            assert isinstance(s["score"], (int, float))
            assert isinstance(s["entry"], (int, float))
            assert isinstance(s["sl"], (int, float))
            assert isinstance(s["tp"], (int, float))
            assert isinstance(s["rr"], (int, float))
            assert isinstance(s["signal_time"], (int, float))

    def test_long_vs_short_distinguishable(self, sample_jsonl: str) -> None:
        signals = get_recent_signals(sample_jsonl, limit=20)
        longs = [s for s in signals if s["direction"] == "LONG"]
        shorts = [s for s in signals if s["direction"] == "SHORT"]
        assert len(longs) >= 2
        assert len(shorts) >= 1
        for s in longs:
            assert s["direction"] == "LONG"
        for s in shorts:
            assert s["direction"] == "SHORT"

    def test_active_has_no_outcomes(self, sample_jsonl: str) -> None:
        active = get_open_signals(sample_jsonl)
        active_ids = {s["signal_id"] for s in active}
        # Each active signal still has signal event fields, never outcome fields
        for s in active:
            assert s.get("event") == "signal"


# ---------------------------------------------------------------------------
# Signal age label tests
# ---------------------------------------------------------------------------


class TestSignalAgeLabel:
    """Test signal_age_label() human-readable output."""

    def test_just_now(self) -> None:
        now = int(time.time() * 1000)
        label = signal_age_label(now)
        assert label == "just now"

    def test_seconds_ago(self) -> None:
        now = int(time.time() * 1000)
        age = now - 30000  # 30 seconds ago
        label = signal_age_label(age)
        assert "30s" in label or "s ago" in label

    def test_minutes_ago(self) -> None:
        now = int(time.time() * 1000)
        age = now - 120000  # 2 minutes ago
        label = signal_age_label(age)
        assert "m ago" in label

    def test_hours_ago(self) -> None:
        now = int(time.time() * 1000)
        age = now - 7200000  # 2 hours ago
        label = signal_age_label(age)
        assert "h ago" in label

    def test_days_ago(self) -> None:
        now = int(time.time() * 1000)
        age = now - 172800000  # 2 days ago
        label = signal_age_label(age)
        assert "d ago" in label

    def test_future_signal(self) -> None:
        """Signal time in the future should return 'just now'."""
        future = int(time.time() * 1000) + 60000
        label = signal_age_label(future)
        assert "now" in label


class TestTemplateJsSignalAge:
    """Regression tests for the browser-side signalAge() in index.html."""

    _TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "templates" / "index.html"

    def _js(self) -> str:
        """Return the embedded template HTML as a string."""
        assert self._TEMPLATE_PATH.exists(), f"Template not found: {self._TEMPLATE_PATH}"
        return self._TEMPLATE_PATH.read_text(encoding="utf-8")

    def test_template_contains_negative_age_guard(self) -> None:
        """signalAge() must have a branch for age < 0 (future timestamps)."""
        html = self._js()
        assert "if (age < 0)" in html, (
            "dashboard/templates/index.html signalAge() is missing a guard "
            "for future timestamps (age < 0). Future outcome_time values will "
            "render as negative 'Ns ago' strings."
        )

    def test_template_future_branch_formats_in_nunits(self) -> None:
        """Future-timestamp branch must use 'in N' format, not 'Ns ago'."""
        html = self._js()
        # Extract the signalAge function body for a lightweight syntax check.
        start = html.find("function signalAge(")
        assert start != -1, "signalAge() function not found in template"
        body = html[start:]
        # The future branch must return strings starting with 'in '.
        assert "'in '" in body or '"in "' in body, (
            "signalAge() future branch does not produce 'in N units' labels."
        )
        # The past branch must still use 'ago' suffixes.
        assert "'s ago'" in body or '"s ago"' in body, (
            "signalAge() past branch lost its 's ago' suffix."
        )


# ---------------------------------------------------------------------------
# Endpoint / data shape tests (lightweight, no live server)
# ---------------------------------------------------------------------------


class TestDashboardDataShapes:
    """Verify the data structures returned match what the frontend expects."""

    def test_status_has_required_fields(self, sample_jsonl: str) -> None:
        status = get_scanner_status(sample_jsonl)
        required = [
            "running", "state", "last_update_ms", "last_update_age_s",
            "symbols", "interval_sec", "max_cycles", "latest_cycle",
            "symbol_count", "last_cycle_time", "age_since_last_cycle_s",
        ]
        for key in required:
            assert key in status, f"Missing field: {key}"

    def test_signal_has_required_fields(self, sample_jsonl: str) -> None:
        signals = get_recent_signals(sample_jsonl, limit=1)
        assert len(signals) > 0
        sig = signals[0]
        required = [
            "signal_id", "symbol", "direction", "signal_time",
            "entry", "tp", "sl", "score", "rr", "timeframe",
            "horizon_ms", "strategy_version", "generated_at",
        ]
        for key in required:
            assert key in sig, f"Missing field in signal: {key}"

    def test_outcome_has_required_fields(self, sample_jsonl: str) -> None:
        """Verify outcome dicts include horizon_ms when present in source data."""
        outcomes = get_recent_outcomes(sample_jsonl, limit=1)
        assert len(outcomes) > 0
        out = outcomes[0]
        required = [
            "event", "signal_id", "symbol", "direction", "outcome",
            "outcome_time", "entry", "tp", "sl", "gross_r", "net_r",
            "holding_ms", "strategy_version",
        ]
        for key in required:
            assert key in out, f"Missing field in outcome: {key}"
        # horizon_ms is optional (may be absent in older/outcome-only records)
        assert out.get("horizon_ms", 0) >= 0

    def test_no_secrets_in_any_record(self, sample_jsonl: str) -> None:
        """Dashboard must never surface API keys / secrets / account info."""
        records = [
            get_scanner_status(sample_jsonl),
            *get_open_signals(sample_jsonl),
            *get_recent_signals(sample_jsonl, limit=20),
            *get_recent_outcomes(sample_jsonl, limit=20),
        ]
        forbidden = ["api_key", "apikey", "secret", "api_secret", "token", "password"]
        for rec in records:
            for k in rec:
                low = k.lower()
                assert low not in forbidden, f"Forbidden key surfaced: {k}"
                v = rec[k]
                if isinstance(v, str):
                    vl = v.lower()
                    # Keys/secrets are usually high-entropy hex/base64 strings;
                    # reject obvious patterns just to be safe.
                    if any(tok in vl for tok in ("sk-", "pk-", "bfx-", "demo_")):
                        raise AssertionError(f"Possible secret value in field {k}: {v[:20]!r}")


# ---------------------------------------------------------------------------
# Regression: duplicate signal / outcome events in JSONL
# ---------------------------------------------------------------------------


class TestDuplicateSignalDedup:
    """The dashboard must show each signal_id once even when JSONL has dupes.

    The scanner can re-emit the same signal_id during restart/reload, so the
    JSONL source of truth may contain duplicate ``signal`` lines.  The
    dashboard treats ``signal_id`` as the unique key and keeps only the
    newest event per id (newest = highest ``generated_at``).
    """

    def test_duplicate_signal_id_appears_once(self, tmp_path: Path) -> None:
        """One signal_id emitted twice → Recent Signals shows it once."""
        path = str(tmp_path / "dup_signal.jsonl")
        now_ms = int(time.time() * 1000)
        t1 = now_ms - 10000
        t2 = now_ms  # second emission is newer
        records = [
            {
                "event": "signal",
                "signal_id": "dup123",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "signal_time": t1,
                "entry": 42000.0,
                "tp": 42840.0,
                "sl": 41580.0,
                "score": 73.5,
                "rr": 2.0,
                "timeframe": "5m",
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "generated_at": t1,
            },
            # Duplicate: same signal_id, later generated_at
            {
                "event": "signal",
                "signal_id": "dup123",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "signal_time": t1,
                "entry": 42100.0,  # different numbers — keep newest
                "tp": 42900.0,
                "sl": 41600.0,
                "score": 74.0,
                "rr": 2.0,
                "timeframe": "5m",
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "generated_at": t2,
            },
            {
                "event": "signal",
                "signal_id": "other456",
                "symbol": "ETHUSDT",
                "direction": "SHORT",
                "signal_time": now_ms - 5000,
                "entry": 2200.0,
                "tp": 2156.0,
                "sl": 2244.0,
                "score": 69.1,
                "rr": 2.0,
                "timeframe": "5m",
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "generated_at": now_ms - 5000,
            },
        ]
        _write_jsonl(path, records)
        signals = get_recent_signals(path, limit=20)
        ids = [s["signal_id"] for s in signals]
        assert ids.count("dup123") == 1, f"dup123 appeared {ids.count('dup123')} times"
        assert "other456" in ids
        assert len(signals) == 2
        # Keep the newest (higher generated_at) dup record.
        dup_rec = next(s for s in signals if s["signal_id"] == "dup123")
        assert dup_rec["generated_at"] == t2
        assert dup_rec["entry"] == 42100.0

    def test_duplicate_keeps_newest_generated_at(self, tmp_path: Path) -> None:
        """Out-of-order dup lines: the one with the higher generated_at wins."""
        path = str(tmp_path / "unordered_dup.jsonl")
        base = int(time.time() * 1000)
        # Write newer one first, older one second — file order is NOT the
        # dedup criterion; generated_at is.
        records = [
            {
                "event": "signal",
                "signal_id": "ooo789",
                "symbol": "SOLUSDT",
                "direction": "LONG",
                "signal_time": base - 30000,
                "entry": 150.0,
                "tp": 156.0,
                "sl": 147.0,
                "score": 80.0,
                "rr": 2.0,
                "timeframe": "5m",
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "generated_at": base,  # newer
            },
            {
                "event": "signal",
                "signal_id": "ooo789",
                "symbol": "SOLUSDT",
                "direction": "LONG",
                "signal_time": base - 30000,
                "entry": 149.0,
                "tp": 155.0,
                "sl": 146.0,
                "score": 79.0,
                "rr": 2.0,
                "timeframe": "5m",
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "generated_at": base - 5000,  # older
            },
        ]
        _write_jsonl(path, records)
        signals = get_recent_signals(path, limit=20)
        assert len(signals) == 1
        assert signals[0]["entry"] == 150.0  # from the newer record

    def test_distinct_signals_same_symbol_both_survive(self, tmp_path: Path) -> None:
        """Two genuinely different signals (different ids) on the same symbol
        are NOT collapsed — only exact signal_id dupes are removed."""
        path = str(tmp_path / "same_symbol_diff_id.jsonl")
        now_ms = int(time.time() * 1000)
        records = [
            {
                "event": "signal",
                "signal_id": "sig-a-001",
                "symbol": "LINKUSDT",
                "direction": "LONG",
                "signal_time": now_ms - 120000,
                "entry": 14.646,
                "tp": 14.892659779543752,
                "sl": 14.522670110228125,
                "score": 65.6,
                "rr": 2.0,
                "timeframe": "5m",
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "generated_at": now_ms - 120000,
            },
            {
                "event": "signal",
                "signal_id": "sig-b-002",
                "symbol": "LINKUSDT",
                "direction": "LONG",
                "signal_time": now_ms - 60000,
                "entry": 14.700,
                "tp": 14.900,
                "sl": 14.550,
                "score": 66.2,
                "rr": 2.0,
                "timeframe": "5m",
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "generated_at": now_ms - 60000,
            },
        ]
        _write_jsonl(path, records)
        signals = get_recent_signals(path, limit=20)
        assert len(signals) == 2
        assert {s["signal_id"] for s in signals} == {"sig-a-001", "sig-b-002"}

    def test_duplicate_in_sample_jsonl(self, sample_jsonl: str) -> None:
        """Existing sample_jsonl (no dupes) still returns one-per-id."""
        signals = get_recent_signals(sample_jsonl, limit=20)
        ids = [s["signal_id"] for s in signals]
        assert len(ids) == len(set(ids)), f"Duplicate signal_ids in sample: {ids}"


class TestDuplicateOutcomeDedup:
    """Each unique (signal_id, outcome) appears once in Outcome History.

    The scanner can write duplicate outcome lines (e.g. tp_hit emitted twice
    for the same signal).  The dashboard dedup key is the pair
    (signal_id, outcome), keeping the newest event per pair.
    """

    def test_duplicate_outcome_appears_once(self, tmp_path: Path) -> None:
        """Same signal_id + outcome written twice → shown once."""
        path = str(tmp_path / "dup_outcome.jsonl")
        now_ms = int(time.time() * 1000)
        t1 = now_ms - 10000
        t2 = now_ms
        records = [
            {
                "event": "tp_hit",
                "signal_id": "o123",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "outcome": "tp_hit",
                "outcome_time": t1,
                "entry": 42000.0,
                "tp": 42840.0,
                "sl": 41580.0,
                "gross_r": 2.0,
                "net_r": 1.8,
                "holding_ms": 1800000,
                "fee": 0.001,
                "slippage": 0.001,
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "close_time": t1,
            },
            # Duplicate outcome line (sometimes the scanner re-stamps).
            {
                "event": "tp_hit",
                "signal_id": "o123",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "outcome": "tp_hit",
                "outcome_time": t2,
                "entry": 42000.0,
                "tp": 42840.0,
                "sl": 41580.0,
                "gross_r": 2.0,
                "net_r": 1.8,
                "holding_ms": 1800000,
                "fee": 0.001,
                "slippage": 0.001,
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "close_time": t2,
            },
            {
                "event": "sl_hit",
                "signal_id": "o456",
                "symbol": "ETHUSDT",
                "direction": "LONG",
                "outcome": "sl_hit",
                "outcome_time": now_ms - 5000,
                "entry": 2200.0,
                "tp": 2244.0,
                "sl": 2156.0,
                "gross_r": -1.0,
                "net_r": -1.2,
                "holding_ms": 600000,
                "fee": 0.001,
                "slippage": 0.001,
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "close_time": now_ms - 5000,
            },
        ]
        _write_jsonl(path, records)
        outcomes = get_recent_outcomes(path, limit=20)
        pairs = [(o["signal_id"], o["outcome"]) for o in outcomes]
        assert pairs.count(("o123", "tp_hit")) == 1
        assert ("o456", "sl_hit") in pairs
        assert len(outcomes) == 2

    def test_same_signal_different_outcomes_both_survive(self, tmp_path: Path) -> None:
        """One signal can only have one real outcome, but if the JSONL somehow
        carried two different outcome types for the same id, both survive
        (they are different pairs)."""
        path = str(tmp_path / "same_id_two_outcomes.jsonl")
        now_ms = int(time.time() * 1000)
        records = [
            {
                "event": "tp_hit",
                "signal_id": "dual999",
                "symbol": "SOLUSDT",
                "direction": "LONG",
                "outcome": "tp_hit",
                "outcome_time": now_ms,
                "entry": 150.0,
                "tp": 156.0,
                "sl": 147.0,
                "gross_r": 2.0,
                "net_r": 1.9,
                "holding_ms": 1800000,
                "fee": 0.001,
                "slippage": 0.001,
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "close_time": now_ms,
            },
            {
                "event": "expired",
                "signal_id": "dual999",
                "symbol": "SOLUSDT",
                "direction": "LONG",
                "outcome": "expired",
                "outcome_time": now_ms - 1000,
                "entry": 150.0,
                "tp": 156.0,
                "sl": 147.0,
                "gross_r": 0.0,
                "net_r": 0.0,
                "holding_ms": 14400000,
                "fee": 0.001,
                "slippage": 0.001,
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "close_time": now_ms - 1000,
            },
        ]
        _write_jsonl(path, records)
        outcomes = get_recent_outcomes(path, limit=20)
        pairs = [(o["signal_id"], o["outcome"]) for o in outcomes]
        assert pairs.count(("dual999", "tp_hit")) == 1
        assert pairs.count(("dual999", "expired")) == 1
        assert len(outcomes) == 2

    def test_outcome_chronological_order_preserved(self, tmp_path: Path) -> None:
        """After dedup, newest-first chronological order is preserved."""
        path = str(tmp_path / "outcome_order.jsonl")
        now_ms = int(time.time() * 1000)
        records = [
            {"event": "expired", "signal_id": "e1", "symbol": "A", "outcome": "expired",
             "outcome_time": now_ms - 5000, "entry": 1.0, "tp": 2.0, "sl": 0.5,
             "gross_r": 0.0, "net_r": 0.0, "holding_ms": 14400000,
             "strategy_version": "sv"},
            {"event": "expired", "signal_id": "e2", "symbol": "B", "outcome": "expired",
             "outcome_time": now_ms - 2000, "entry": 1.0, "tp": 2.0, "sl": 0.5,
             "gross_r": 0.0, "net_r": 0.0, "holding_ms": 14400000,
             "strategy_version": "sv"},
            {"event": "expired", "signal_id": "e3", "symbol": "C", "outcome": "expired",
             "outcome_time": now_ms - 8000, "entry": 1.0, "tp": 2.0, "sl": 0.5,
             "gross_r": 0.0, "net_r": 0.0, "holding_ms": 14400000,
             "strategy_version": "sv"},
        ]
        _write_jsonl(path, records)
        outcomes = get_recent_outcomes(path, limit=20)
        times = [o["outcome_time"] for o in outcomes]
        assert times == sorted(times, reverse=True), "Outcomes not newest-first"

    def test_duplicate_in_paper_trade_real_ids(self) -> None:
        """Verify dedup against the actual paper_trade.jsonl duplicate ids that
        were observed (LINKUSDT fa48bdba33bc04ac, SOONUSDT 00adcd89b5d044ba)."""
        log_path = "data/signals/paper_trade.jsonl"
        p = Path(log_path)
        if not p.exists():
            pytest.skip(f"{log_path} not present")
        signals = get_recent_signals(log_path, limit=20)
        outcomes = get_recent_outcomes(log_path, limit=20)
        sig_ids = [s["signal_id"] for s in signals]
        out_pairs = [(o["signal_id"], o["outcome"]) for o in outcomes]
        # The observed dup ids must each appear at most once.
        for dup_id in ("fa48bdba33bc04ac", "00adcd89b5d044ba"):
            assert sig_ids.count(dup_id) <= 1, (
                f"signal_id {dup_id} appears {sig_ids.count(dup_id)} times in "
                f"Recent Signals (expected <= 1)"
            )
        for dup_id in ("fa48bdba33bc04ac", "00adcd89b5d044ba"):
            assert out_pairs.count((dup_id, "expired")) <= 1, (
                f"(signal_id={dup_id}, outcome=expired) appears "
                f"{out_pairs.count((dup_id, 'expired'))} times in Outcome History "
                f"(expected <= 1)"
            )


class TestSmallPricePrecisionDisplay:
    """Small-price symbols (e.g. SOONUSDT ~0.40) must keep Entry/SL/TP
    distinguishable in the dashboard.

    The underlying numeric values are already stored at full precision in
    JSONL.  The fix is in the JS price formatter (formatPrice) in the
    dashboard template, which uses a decimal-places rule driven by price
    magnitude instead of a flat 2-decimal rounding.
    """

    def test_soonusdt_entry_sl_values_differ_in_source(self) -> None:
        """SOONUSDT 00adcd89b5d044ba in paper_trade.jsonl has entry != sl,
        even though both round to 0.40 at 2 decimals."""
        log_path = "data/signals/paper_trade.jsonl"
        p = Path(log_path)
        if not p.exists():
            pytest.skip(f"{log_path} not present")
        events = _read_jsonl(log_path)
        sigs = _find_events(events, "signal")
        matches = [s for s in sigs if s.get("signal_id") == "00adcd89b5d044ba"]
        assert matches, "SOONUSDT 00adcd89b5d044ba not found in JSONL"
        # There may be 2 signal lines (dup); check at least one pair differs.
        found_diff = False
        for s in matches:
            entry = s.get("entry")
            sl = s.get("sl")
            if entry is not None and sl is not None and entry != sl:
                found_diff = True
                assert round(entry, 2) == round(sl, 2), (
                    "entry and sl round to the same 2-decimal value but differ "
                    "in full precision — this is the exact case formatPrice must "
                    "handle"
                )
                # Verify they are distinguishable with more decimals.
                assert f"{entry:.4f}" != f"{sl:.4f}", (
                    "entry and sl must be distinguishable with >=4 decimals"
                )
                break
        assert found_diff, (
            "Expected at least one SOONUSDT 00adcd89b5d044ba record where "
            "entry != sl at full precision"
        )

    def test_formatprice_rules_distinguish_small_prices(self) -> None:
        """Python mirror of the JS formatPrice rule: verify that entry vs sl
        for SOONUSDT become different strings under the magnitude-based rule."""
        entry = 0.4038
        sl = 0.3994895173242386
        tp = 0.4124209653515228

        def format_price(p):
            if p is None:
                return "—"
            abs_p = abs(p)
            if abs_p >= 1000:
                decimals = 2
            elif abs_p >= 1:
                decimals = 4
            elif abs_p >= 0.01:
                decimals = 6
            else:
                decimals = 8
            return f"{p:.{decimals}f}"

        e_s = format_price(entry)
        sl_s = format_price(sl)
        tp_s = format_price(tp)
        assert e_s != sl_s, (
            f"formatPrice renders entry={e_s} and sl={sl_s} identically — "
            f"must differ for SOONUSDT"
        )
        assert e_s == "0.403800", f"Expected 0.403800, got {e_s}"
        assert sl_s == "0.399490", f"Expected 0.399490, got {sl_s}"
        assert tp_s == "0.412421", f"Expected 0.412421, got {tp_s}"

    def test_large_price_still_concise(self) -> None:
        """BTCUSDT-level prices keep only 2 decimals, not 8."""
        def format_price(p):
            if p is None:
                return "—"
            abs_p = abs(p)
            if abs_p >= 1000:
                decimals = 2
            elif abs_p >= 1:
                decimals = 4
            elif abs_p >= 0.01:
                decimals = 6
            else:
                decimals = 8
            return f"{p:.{decimals}f}"

        assert format_price(42000.123456) == "42000.12"
        assert format_price(14.646) == "14.6460"  # >=1, <1000 → 4 decimals
        assert format_price(0.7678) == "0.767800"  # >=0.01 → 6 decimals
        assert format_price(0.005123) == "0.00512300"  # <0.01 → 8 decimals

    def test_dashboard_api_preserves_numeric_precision(self, tmp_path: Path) -> None:
        """The /api/signals and /api/history responses must carry full-precision
        floats, not pre-rounded values.  (Rounding only happens in the JS
        display layer.)"""
        path = str(tmp_path / "precision.jsonl")
        now_ms = int(time.time() * 1000)
        records = [
            {
                "event": "signal",
                "signal_id": "prec999",
                "symbol": "SOONUSDT",
                "direction": "LONG",
                "signal_time": now_ms,
                "entry": 0.4038,
                "tp": 0.4124209653515228,
                "sl": 0.3994895173242386,
                "score": 81.7,
                "rr": 2.0,
                "timeframe": "5m",
                "horizon_ms": 14400000,
                "strategy_version": "c5c143cf94ed8790",
                "generated_at": now_ms,
            }
        ]
        _write_jsonl(path, records)
        sigs = get_recent_signals(path, limit=1)
        assert sigs
        s = sigs[0]
        assert s["entry"] == 0.4038
        assert s["tp"] == 0.4124209653515228
        assert s["sl"] == 0.3994895173242386
        # Full precision preserved (not rounded to 2 decimals by the API).
        assert s["entry"] != 0.40
        assert s["sl"] != 0.40
