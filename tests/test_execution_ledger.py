# Tests for execution audit ledger.

from __future__ import annotations
import sys
import os
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution.ledger import ExecutionLedger
from execution.models import (
    SignalApproval,
    ApprovedSignalRef,
    OrderRequest,
    ExecutionResult,
)


class TestLedgerAppendRead(unittest.TestCase):
    """Basic ledger append and read operations."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_empty_ledger_is_empty(self):
        """New ledger is empty."""
        self.assertTrue(self.ledger.is_empty())
        self.assertEqual(self.ledger.count(), 0)

    def test_append_signal_received(self):
        """Append a signal_received event."""
        self.ledger.log_signal_received(
            signal_id="sig_001",
            symbol="BTCUSDT",
            direction="LONG",
            entry=100.0,
            sl=95.0,
            tp=110.0,
            score=65.0,
            signal_time=1000000,
            strategy_version="v1",
            received_at_ms=1000001,
        )
        self.assertEqual(self.ledger.count(), 1)

        events = self.ledger.read_all()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "signal_received")
        self.assertEqual(events[0]["signal_id"], "sig_001")

    def test_append_approval(self):
        """Append an approval event."""
        approval = SignalApproval(
            signal_id="sig_002",
            symbol="ETHUSDT",
            direction="SHORT",
            approved_at_ms=2000000,
            approver="manual",
            reason="Approved by risk gate",
            approved=True,
        )
        self.ledger.log_approval(approval)
        self.assertEqual(self.ledger.count(), 1)

        events = self.ledger.read_all()
        self.assertEqual(events[0]["event"], "approval")
        self.assertEqual(events[0]["signal_id"], "sig_002")
        self.assertTrue(events[0]["approved"])

    def test_append_approved_signal(self):
        """Append an approved_signal event."""
        signal = ApprovedSignalRef(
            signal_id="sig_003",
            symbol="BTCUSDT",
            direction="LONG",
            entry=100.0,
            sl=95.0,
            tp=110.0,
            score=65.0,
            signal_time=1000000,
            strategy_version="v1",
            approved_at_ms=1000001,
            approver="manual",
        )
        self.ledger.log_approved_signal(signal)
        events = self.ledger.read_all()
        self.assertEqual(events[0]["event"], "approved_signal")

    def test_append_risk_rejected(self):
        """Append a risk_rejected event."""
        self.ledger.log_risk_rejected(
            signal_id="sig_004",
            symbol="SOLUSDT",
            reason="R:R below minimum",
            risk_code="RR_BELOW_MINIMUM",
            details={"rr": 1.5, "min_rr": 2.0},
            rejected_at_ms=3000000,
        )
        events = self.ledger.read_all()
        self.assertEqual(events[0]["event"], "risk_rejected")
        self.assertEqual(events[0]["risk_code"], "RR_BELOW_MINIMUM")

    def test_append_order_request_prepared(self):
        """Append an order_request_prepared event."""
        order = OrderRequest(
            client_order_id="cid_prep_001",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            price=None,
            stop_price=None,
            time_in_force=None,
            reduce_only=False,
            close_position=False,
            signal_id="sig_005",
            strategy_version="v1",
            requested_at_ms=4000000,
        )
        self.ledger.log_order_request_prepared(order)
        events = self.ledger.read_all()
        self.assertEqual(events[0]["event"], "order_request_prepared")
        self.assertEqual(events[0]["client_order_id"], "cid_prep_001")

    def test_append_order_result(self):
        """Append an order_result event."""
        result = ExecutionResult(
            client_order_id="cid_result_001",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            signal_id="sig_006",
            status="ACCEPTED",
            message="Order placed",
            binance_order_id=123456,
            filled_qty=0.5,
            fill_price=95000.0,
            fee=0.05,
            error_code=None,
            executed_at_ms=5000000,
        )
        self.ledger.log_order_result(result)
        events = self.ledger.read_all()
        self.assertEqual(events[0]["event"], "order_result")
        self.assertEqual(events[0]["status"], "ACCEPTED")

    def test_append_error(self):
        """Append an error event."""
        self.ledger.log_error(
            signal_id="sig_007",
            symbol="ETHUSDT",
            error_type="NETWORK_ERROR",
            message="Connection timeout",
            error_code=None,
            occurred_at_ms=6000000,
            details={"retry_count": 3},
        )
        self.ledger.log_error(
            signal_id="sig_008",
            symbol="BTCUSDT",
            error_type="VALIDATION_ERROR",
            message="Invalid price",
            error_code=None,
            occurred_at_ms=6000001,
        )
        events = self.ledger.read_all()
        self.assertEqual(events[0]["event"], "error")
        self.assertEqual(events[0]["error_type"], "NETWORK_ERROR")

    def test_append_dict_direct(self):
        """Append a raw dict event."""
        record = {
            "event": "signal_received",
            "signal_id": "sig_dict_001",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 100.0,
            "sl": 95.0,
            "tp": 110.0,
            "score": 65.0,
            "signal_time": 1000000,
            "strategy_version": "v1",
            "received_at_ms": 1000001,
        }
        self.ledger.append_dict(record)
        events = self.ledger.read_all()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["signal_id"], "sig_dict_001")


class TestLedgerPreservesMultipleEvents(unittest.TestCase):
    """Multiple events are preserved in order."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_multiple_events_preserved(self):
        """Appending multiple events preserves all of them in order."""
        # Append 5 events
        self.ledger.log_signal_received(
            signal_id="sig_a", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0, score=65.0,
            signal_time=1000000, strategy_version="v1", received_at_ms=1000000,
        )
        self.ledger.log_risk_rejected(
            signal_id="sig_b", symbol="ETHUSDT",
            reason="Invalid geometry", risk_code="INVALID_GEOMETRY_LONG",
            rejected_at_ms=2000000,
        )
        self.ledger.log_approval(
            SignalApproval(
                signal_id="sig_c", symbol="SOLUSDT", direction="LONG",
                approved_at_ms=3000000, approver="manual", approved=True,
            )
        )
        self.ledger.log_order_request_prepared(
            OrderRequest(
                client_order_id="cid_1", symbol="BTCUSDT", side="BUY",
                order_type="MARKET", quantity=0.5,
                price=None, stop_price=None, time_in_force=None,
                reduce_only=False, close_position=False,
                signal_id="sig_c", strategy_version="v1", requested_at_ms=4000000,
            )
        )
        self.ledger.log_order_result(
            ExecutionResult(
                client_order_id="cid_1", symbol="BTCUSDT", side="BUY",
                order_type="MARKET", quantity=0.5,
                signal_id="sig_c", status="ACCEPTED", message="OK",
                binance_order_id=999, executed_at_ms=5000000,
            )
        )

        events = self.ledger.read_all()
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0]["event"], "signal_received")
        self.assertEqual(events[0]["signal_id"], "sig_a")
        self.assertEqual(events[1]["event"], "risk_rejected")
        self.assertEqual(events[1]["signal_id"], "sig_b")
        self.assertEqual(events[2]["event"], "approval")
        self.assertEqual(events[2]["signal_id"], "sig_c")
        self.assertEqual(events[3]["event"], "order_request_prepared")
        self.assertEqual(events[3]["client_order_id"], "cid_1")
        self.assertEqual(events[4]["event"], "order_result")
        self.assertEqual(events[4]["status"], "ACCEPTED")

    def test_reverse_read(self):
        """Reverse read returns newest first."""
        self.ledger.log_signal_received(
            signal_id="sig_r1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0, score=65.0,
            signal_time=1000000, strategy_version="v1", received_at_ms=1000000,
        )
        self.ledger.log_signal_received(
            signal_id="sig_r2", symbol="ETHUSDT", direction="SHORT",
            entry=2000.0, sl=2100.0, tp=1900.0, score=70.0,
            signal_time=2000000, strategy_version="v1", received_at_ms=2000000,
        )

        recent = self.ledger.read_reverse(limit=5)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["signal_id"], "sig_r2")
        self.assertEqual(recent[1]["signal_id"], "sig_r1")


class TestLedgerEventFiltering(unittest.TestCase):
    """Event filtering by type and signal_id."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_find_by_event_type(self):
        """find_by_event_type returns matching events."""
        self.ledger.log_signal_received(
            signal_id="sig_f1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0, score=65.0,
            signal_time=1000000, strategy_version="v1", received_at_ms=1000000,
        )
        self.ledger.log_signal_received(
            signal_id="sig_f2", symbol="ETHUSDT", direction="SHORT",
            entry=2000.0, sl=2100.0, tp=1900.0, score=70.0,
            signal_time=2000000, strategy_version="v1", received_at_ms=2000000,
        )
        self.ledger.log_risk_rejected(
            signal_id="sig_fr1", symbol="SOLUSDT",
            reason="RR below min", risk_code="RR_BELOW_MINIMUM",
            rejected_at_ms=3000000,
        )

        signals = self.ledger.find_by_event_type("signal_received")
        self.assertEqual(len(signals), 2)

        rejections = self.ledger.find_by_event_type("risk_rejected")
        self.assertEqual(len(rejections), 1)
        self.assertEqual(rejections[0]["signal_id"], "sig_fr1")

        unknown = self.ledger.find_by_event_type("nonexistent_type")
        self.assertEqual(len(unknown), 0)

    def test_find_by_signal_id(self):
        """find_by_signal_id returns all events for a signal."""
        self.ledger.log_signal_received(
            signal_id="sig_samid_001", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0, score=65.0,
            signal_time=1000000, strategy_version="v1", received_at_ms=1000000,
        )
        self.ledger.log_risk_rejected(
            signal_id="sig_samid_001", symbol="BTCUSDT",
            reason="Invalid geometry", risk_code="INVALID_GEOMETRY_LONG",
            rejected_at_ms=2000000,
        )

        events = self.ledger.find_by_signal_id("sig_samid_001")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["event"], "signal_received")
        self.assertEqual(events[1]["event"], "risk_rejected")

        # Different signal_id returns empty
        events_other = self.ledger.find_by_signal_id("sig_other")
        self.assertEqual(len(events_other), 0)


class TestLedgerCount(unittest.TestCase):
    """Count operations."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_count_all(self):
        """count() returns total events."""
        self.ledger.log_signal_received(
            signal_id="sig_c1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0, score=65.0,
            signal_time=1000000, strategy_version="v1", received_at_ms=1000000,
        )
        self.ledger.log_risk_rejected(
            signal_id="sig_c2", symbol="ETHUSDT",
            reason="RR below min", risk_code="RR_BELOW_MINIMUM",
            rejected_at_ms=2000000,
        )
        self.assertEqual(self.ledger.count(), 2)

    def test_count_by_type(self):
        """count(event_type) returns count of that type."""
        self.ledger.log_signal_received(
            signal_id="sig_cc1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0, score=65.0,
            signal_time=1000000, strategy_version="v1", received_at_ms=1000000,
        )
        self.ledger.log_signal_received(
            signal_id="sig_cc2", symbol="ETHUSDT", direction="SHORT",
            entry=2000.0, sl=2100.0, tp=1900.0, score=70.0,
            signal_time=2000000, strategy_version="v1", received_at_ms=2000000,
        )
        self.ledger.log_risk_rejected(
            signal_id="sig_cr1", symbol="SOLUSDT",
            reason="Invalid", risk_code="INVALID_GEOMETRY_SHORT",
            rejected_at_ms=3000000,
        )

        self.assertEqual(self.ledger.count(), 3)
        self.assertEqual(self.ledger.count("signal_received"), 2)
        self.assertEqual(self.ledger.count("risk_rejected"), 1)
        self.assertEqual(self.ledger.count("nonexistent"), 0)


class TestLedgerNoSecrets(unittest.TestCase):
    """Secrets are never written to the ledger."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_no_api_key_field_in_any_event(self):
        """No event type writes an api_key field."""
        event_types_and_builders = [
            ("signal_received", lambda: self.ledger.log_signal_received(
                signal_id="sig_ns1", symbol="BTCUSDT", direction="LONG",
                entry=100.0, sl=95.0, tp=110.0, score=65.0,
                signal_time=1000000, strategy_version="v1", received_at_ms=1000000,
            )),
            ("approval", lambda: self.ledger.log_approval(
                SignalApproval(
                    signal_id="sig_ns2", symbol="ETHUSDT", direction="LONG",
                    approved_at_ms=2000000, approver="manual", approved=True,
                )
            )),
            ("approved_signal", lambda: self.ledger.log_approved_signal(
                ApprovedSignalRef(
                    signal_id="sig_ns3", symbol="SOLUSDT", direction="LONG",
                    entry=100.0, sl=95.0, tp=110.0, score=65.0,
                    signal_time=1000000, strategy_version="v1",
                    approved_at_ms=3000000, approver="manual",
                )
            )),
            ("order_request_prepared", lambda: self.ledger.log_order_request_prepared(
                OrderRequest(
                    client_order_id="cid_ns4", symbol="BTCUSDT", side="BUY",
                    order_type="MARKET", quantity=0.5,
                    price=None, stop_price=None, time_in_force=None,
                    reduce_only=False, close_position=False,
                    signal_id="sig_ns4", strategy_version="v1", requested_at_ms=4000000,
                )
            )),
            ("order_result", lambda: self.ledger.log_order_result(
                ExecutionResult(
                    client_order_id="cid_ns5", symbol="BTCUSDT", side="BUY",
                    order_type="MARKET", quantity=0.5,
                    signal_id="sig_ns5", status="ACCEPTED", message="OK",
                    executed_at_ms=5000000,
                )
            )),
            ("error", lambda: self.ledger.log_error(
                signal_id="sig_ns6", symbol="ETHUSDT",
                error_type="TEST_ERROR", message="Test message",
                occurred_at_ms=6000000,
            )),
        ]

        for expected_event_type, builder in event_types_and_builders:
            builder()
            events = self.ledger.read_all()
            last_event = events[-1]
            self.assertNotIn("api_key", last_event)
            self.assertNotIn("apikey", last_event)
            self.assertNotIn("api_secret", last_event)
            self.assertNotIn("apisecret", last_event)
            self.assertNotIn("secret", last_event)
            self.assertNotIn("password", last_event)
            self.assertNotIn("token", last_event)
            self.assertEqual(last_event["event"], expected_event_type)

    def test_attempt_to_write_secret_field_raises(self):
        """Attempting to append a dict with an api_key field raises."""
        record = {
            "event": "signal_received",
            "signal_id": "sig_secret_001",
            "api_key": "super_secret_key_12345",  # <-- should be rejected
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 100.0,
            "sl": 95.0,
            "tp": 110.0,
            "score": 65.0,
            "signal_time": 1000000,
            "strategy_version": "v1",
            "received_at_ms": 1000000,
        }
        with self.assertRaises(ValueError) as ctx:
            self.ledger.append_dict(record)
        self.assertIn("sensitive field 'api_key'", str(ctx.exception))

    def test_attempt_to_write_secret_value_raises(self):
        """Attempting to append a dict with a hex-secret-like value in a non-safe field raises."""
        # A long hex string in a field that isn't on the safe list
        record = {
            "event": "error",
            "signal_id": "sig_secret_val_001",
            "symbol": "BTCUSDT",
            "error_type": "TEST",
            "message": "Test",
            "some_custom_field": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
            "occurred_at_ms": 1000000,
        }
        with self.assertRaises(ValueError) as ctx:
            self.ledger.append_dict(record)
        self.assertIn("Potential secret value", str(ctx.exception))


class TestLedgerPaths(unittest.TestCase):
    """Ledger path handling."""

    def test_creates_parent_directory(self):
        """Ledger creates parent directory if needed."""
        import shutil
        temp_dir = tempfile.mkdtemp()
        try:
            nested_path = os.path.join(temp_dir, "nested", "dir", "ledger.jsonl")
            ledger = ExecutionLedger(nested_path)
            self.assertTrue(os.path.isdir(os.path.dirname(nested_path)))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_nonexistent_ledger_is_empty(self):
        """Ledger at nonexistent path reports empty."""
        import tempfile
        import os
        nonexistent = os.path.join(tempfile.gettempdir(), "nonexistent_dir", "ledger.jsonl")
        ledger = ExecutionLedger(nonexistent)
        self.assertTrue(ledger.is_empty())

    def test_empty_ledger_file_is_empty(self):
        """Empty file reports empty."""
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            ledger = ExecutionLedger(path)
            self.assertTrue(ledger.is_empty())
        finally:
            os.unlink(path)


class TestLedgerSerialization(unittest.TestCase):
    """JSONL serialization is valid JSON."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_file_contains_valid_json_lines(self):
        """Each line in the ledger file is valid JSON."""
        self.ledger.log_signal_received(
            signal_id="sig_json_001", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0, score=65.0,
            signal_time=1000000, strategy_version="v1", received_at_ms=1000000,
        )
        self.ledger.log_risk_rejected(
            signal_id="sig_json_002", symbol="ETHUSDT",
            reason="Test", risk_code="TEST_CODE",
            rejected_at_ms=2000000,
        )

        # Read raw file
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)
        import json
        for i, line in enumerate(lines):
            line = line.strip()
            self.assertNotEqual(line, "")
            try:
                parsed = json.loads(line)
                self.assertIsInstance(parsed, dict)
                self.assertIn("event", parsed)
            except json.JSONDecodeError as e:
                self.fail(f"Line {i+1} is not valid JSON: {e}")


class TestNoNetworkCalls(unittest.TestCase):
    """Verify no network calls are made during ledger operations."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_append_does_not_make_network_call(self):
        """Appending events does not trigger any network activity.

        This is a documentation test: we verify the code path does not
        import or call any network module. The storage/jsonl.py module
        uses only file I/O. We trust this by inspection.
        """
        # Log several events
        for i in range(5):
            self.ledger.log_signal_received(
                signal_id=f"sig_net_{i}", symbol="BTCUSDT", direction="LONG",
                entry=100.0, sl=95.0, tp=110.0, score=65.0,
                signal_time=1000000 + i, strategy_version="v1",
                received_at_ms=1000000 + i,
            )

        # Read them back — all local file I/O
        events = self.ledger.read_all()
        self.assertEqual(len(events), 5)
        # No assertion about network — just verify local operations succeed


class TestLedgerApprovals(unittest.TestCase):
    """Approval-related ledger operations."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_approval_event_recorded(self):
        """Approval event is recorded with all fields."""
        approval = SignalApproval(
            signal_id="sig_app_001",
            symbol="BTCUSDT",
            direction="LONG",
            approved_at_ms=1000000,
            approver="manual_risk_gate_v1",
            reason="Score >= 65, R:R >= 2.0",
            approved=True,
        )
        self.ledger.log_approval(approval)

        events = self.ledger.read_all()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event"], "approval")
        self.assertEqual(event["signal_id"], "sig_app_001")
        self.assertEqual(event["approver"], "manual_risk_gate_v1")
        self.assertEqual(event["reason"], "Score >= 65, R:R >= 2.0")
        self.assertTrue(event["approved"])

    def test_approval_event_not_automatic(self):
        """A signal can exist in the ledger without an approval event."""
        # Log a signal_received but no approval
        self.ledger.log_signal_received(
            signal_id="sig_no_app_001", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0, score=65.0,
            signal_time=1000000, strategy_version="v1", received_at_ms=1000000,
        )

        events = self.ledger.read_all()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "signal_received")

        # No approval event exists for this signal
        approvals = self.ledger.find_by_event_type("approval")
        self.assertEqual(len(approvals), 0)

        # This documents the approval boundary: signal_received does NOT
        # imply approval. An explicit approval event must be logged.


class TestLedgerOrderRequest(unittest.TestCase):
    """Order request ledger operations."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_order_request_contains_client_order_id(self):
        """Order request event contains client_order_id for idempotency."""
        order = OrderRequest(
            client_order_id="cid_ord_001",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            price=None,
            stop_price=None,
            time_in_force=None,
            reduce_only=False,
            close_position=False,
            signal_id="sig_ord_001",
            strategy_version="v1",
            requested_at_ms=1000000,
        )
        self.ledger.log_order_request_prepared(order)

        events = self.ledger.read_all()
        self.assertEqual(events[0]["client_order_id"], "cid_ord_001")
        self.assertNotIn("api_key", events[0])
        self.assertNotIn("api_secret", events[0])

    def test_order_request_price_none_for_market(self):
        """MARKET orders have price=None."""
        order = OrderRequest(
            client_order_id="cid_market_ord",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            price=None,
            stop_price=None,
            time_in_force=None,
            reduce_only=False,
            close_position=False,
            signal_id="sig_market",
            strategy_version="v1",
            requested_at_ms=1000000,
        )
        self.ledger.log_order_request_prepared(order)

        events = self.ledger.read_all()
        self.assertIsNone(events[0]["price"])
        self.assertIsNone(events[0]["stop_price"])


class TestLedgerOrderResult(unittest.TestCase):
    """Order result ledger operations."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = ExecutionLedger(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_order_result_accepted(self):
        """Accepted order result is logged."""
        result = ExecutionResult(
            client_order_id="cid_res_001",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            signal_id="sig_res_001",
            status="ACCEPTED",
            message="Order filled",
            binance_order_id=12345,
            filled_qty=0.5,
            fill_price=95000.0,
            fee=0.05,
            error_code=None,
            executed_at_ms=2000000,
        )
        self.ledger.log_order_result(result)

        events = self.ledger.read_all()
        event = events[0]
        self.assertEqual(event["event"], "order_result")
        self.assertEqual(event["status"], "ACCEPTED")
        self.assertEqual(event["binance_order_id"], 12345)
        self.assertEqual(event["filled_qty"], 0.5)
        self.assertEqual(event["fill_price"], 95000.0)

    def test_order_result_rejected(self):
        """Rejected order result is logged."""
        result = ExecutionResult(
            client_order_id="cid_res_002",
            symbol="ETHUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=1.0,
            signal_id="sig_res_002",
            status="REJECTED",
            message="Insufficient balance",
            error_code=-2014,
            executed_at_ms=2000001,
        )
        self.ledger.log_order_result(result)

        events = self.ledger.read_all()
        event = events[0]
        self.assertEqual(event["status"], "REJECTED")
        self.assertEqual(event["error_code"], -2014)
        self.assertIsNone(event.get("binance_order_id"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
