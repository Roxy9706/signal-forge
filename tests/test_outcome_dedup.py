# Tests for outcome dedup in scanner.py — duplicate outcome prevention.
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.jsonl import jsonl_append, jsonl_has_outcome, jsonl_find, jsonl_read
from SignalForge.paper.models import PaperOutcomeEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outcome(signal_id: str, outcome_type: str = "tp_hit", **overrides) -> dict:
    """Build a PaperOutcomeEvent dict for jsonl_append."""
    event = PaperOutcomeEvent(
        event=outcome_type,
        signal_id=signal_id,
        symbol=overrides.pop("symbol", "BTCUSDT"),
        direction=overrides.pop("direction", "LONG"),
        outcome=outcome_type,
        outcome_time=overrides.pop("outcome_time", 1000000),
        entry=overrides.pop("entry", 100.0),
        tp=overrides.pop("tp", 104.0),
        sl=overrides.pop("sl", 98.0),
        gross_r=overrides.pop("gross_r", 2.0),
        net_r=overrides.pop("net_r", 1.8),
        holding_ms=overrides.pop("holding_ms", 600000),
        fee=overrides.pop("fee", 0.1),
        slippage=overrides.pop("slippage", 0.1),
        horizon_ms=overrides.pop("horizon_ms", 4 * 3600 * 1000),
        strategy_version=overrides.pop("strategy_version", "test"),
        close_time=overrides.pop("close_time", 1000000),
    )
    d = event.to_dict()
    d.update(overrides)
    return d


def _append_outcome_with_dedup(path: str, record: dict) -> bool:
    """Simulate the scanner's dedup logic: append only if no outcome exists."""
    if not jsonl_has_outcome(path, record["signal_id"]):
        jsonl_append(path, record)
        return True
    return False


# ---------------------------------------------------------------------------
# Duplicate outcome prevention (via the dedup wrapper, matching scanner logic)
# ---------------------------------------------------------------------------

class TestDuplicateOutcomePrevention:
    def test_first_outcome_writes(self, tmp_path: Path) -> None:
        path = str(tmp_path / "test.jsonl")
        d = _make_outcome("sig1", "tp_hit")
        assert _append_outcome_with_dedup(path, d) is True
        assert jsonl_has_outcome(path, "sig1") is True
        records = jsonl_find(path, event_type="tp_hit")
        assert len(records) == 1

    def test_duplicate_outcome_not_written(self, tmp_path: Path) -> None:
        """Appending the same outcome twice does not create a duplicate line."""
        path = str(tmp_path / "test.jsonl")
        d = _make_outcome("sig1", "tp_hit")
        assert _append_outcome_with_dedup(path, d) is True
        # Second append of identical outcome — dedup rejects it
        assert _append_outcome_with_dedup(path, d) is False
        records = jsonl_find(path, event_type="tp_hit")
        assert len(records) == 1, f"Expected 1 record, got {len(records)}"

    def test_duplicate_outcome_different_type_not_written(self, tmp_path: Path) -> None:
        """A signal that already has ANY outcome must not get another."""
        path = str(tmp_path / "test.jsonl")
        d_tp = _make_outcome("sig1", "tp_hit")
        assert _append_outcome_with_dedup(path, d_tp) is True
        d_sl = _make_outcome("sig1", "sl_hit")
        assert _append_outcome_with_dedup(path, d_sl) is False
        tp_records = jsonl_find(path, event_type="tp_hit")
        sl_records = jsonl_find(path, event_type="sl_hit")
        assert len(tp_records) == 1
        assert len(sl_records) == 0, f"sl_hit should not be written for sig1; got {len(sl_records)}"

    def test_different_signal_each_gets_outcome(self, tmp_path: Path) -> None:
        path = str(tmp_path / "test.jsonl")
        assert _append_outcome_with_dedup(path, _make_outcome("sig1", "tp_hit")) is True
        assert _append_outcome_with_dedup(path, _make_outcome("sig2", "sl_hit")) is True
        assert jsonl_has_outcome(path, "sig1") is True
        assert jsonl_has_outcome(path, "sig2") is True
        tp = jsonl_find(path, event_type="tp_hit")
        sl = jsonl_find(path, event_type="sl_hit")
        assert len(tp) == 1
        assert len(sl) == 1

    def test_outcome_record_count_stable_under_repeated_appends(self, tmp_path: Path) -> None:
        """Append the same outcome N times; count stays 1."""
        path = str(tmp_path / "test.jsonl")
        d = _make_outcome("sig1", "expired")
        for _ in range(5):
            _append_outcome_with_dedup(path, d)
        records = jsonl_find(path, event_type="expired")
        assert len(records) == 1
        all_records = jsonl_read(path)
        assert len(all_records) == 1

    def test_outcome_idempotency_across_types(self, tmp_path: Path) -> None:
        """For a single signal_id, only the FIRST outcome type written survives."""
        path = str(tmp_path / "test.jsonl")
        # Write tp_hit first
        assert _append_outcome_with_dedup(path, _make_outcome("sig1", "tp_hit")) is True
        # Then try sl_hit — should be rejected
        assert _append_outcome_with_dedup(path, _make_outcome("sig1", "sl_hit")) is False
        # Then try expired — should be rejected
        assert _append_outcome_with_dedup(path, _make_outcome("sig1", "expired")) is False
        tp = jsonl_find(path, event_type="tp_hit")
        sl = jsonl_find(path, event_type="sl_hit")
        exp = jsonl_find(path, event_type="expired")
        assert len(tp) == 1
        assert len(sl) == 0
        assert len(exp) == 0


# ---------------------------------------------------------------------------
# Integration: scanner outcome dedup uses jsonl_has_outcome correctly
# ---------------------------------------------------------------------------

class TestOutcomeDedupIntegration:
    def test_jsonl_has_outcome_matches_find(self, tmp_path: Path) -> None:
        path = str(tmp_path / "test.jsonl")
        assert jsonl_has_outcome(path, "nonexistent") is False
        jsonl_append(path, _make_outcome("sig1", "tp_hit"))
        assert jsonl_has_outcome(path, "sig1") is True
        assert jsonl_has_outcome(path, "sig2") is False

    def test_outcome_prepends_signal_equivalently(self, tmp_path: Path) -> None:
        """If a signal was written, then an outcome, has_outcome finds it."""
        path = str(tmp_path / "test.jsonl")
        jsonl_append(path, {
            "event": "signal",
            "signal_id": "sig1",
            "symbol": "BTCUSDT",
            "direction": "LONG",
        })
        jsonl_append(path, _make_outcome("sig1", "tp_hit"))
        assert jsonl_has_outcome(path, "sig1") is True

    def test_raw_append_does_not_dedup(self, tmp_path: Path) -> None:
        """jsonl_append itself does NOT dedup — only the scanner wrapper does."""
        path = str(tmp_path / "test.jsonl")
        d = _make_outcome("sig1", "tp_hit")
        jsonl_append(path, d)
        jsonl_append(path, d)  # raw append, no guard
        records = jsonl_find(path, event_type="tp_hit")
        assert len(records) == 2, "raw jsonl_append has no dedup; scanner wrapper adds it"
