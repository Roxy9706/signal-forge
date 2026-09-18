# Tests for execution data models.

from __future__ import annotations
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution.models import (
    SignalApproval,
    ApprovedSignalRef,
    OrderRequest,
    ExecutionResult,
    event_to_dict,
    event_from_dict,
)


class TestSignalApproval(unittest.TestCase):
    """SignalApproval model tests."""

    def test_valid_construction_defaults(self):
        """Valid approval with defaults (approved=False by default)."""
        a = SignalApproval(
            signal_id="sig_123",
            symbol="BTCUSDT",
            direction="LONG",
            approved_at_ms=1000000,
            approver="manual",
        )
        self.assertEqual(a.signal_id, "sig_123")
        self.assertEqual(a.symbol, "BTCUSDT")
        self.assertEqual(a.direction, "LONG")
        self.assertEqual(a.approved_at_ms, 1000000)
        self.assertEqual(a.approver, "manual")
        self.assertEqual(a.reason, "")
        self.assertFalse(a.approved)  # default is False — fail closed

    def test_is_approved_true(self):
        """is_approved returns True only when approved is True."""
        a_true = SignalApproval(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            approved_at_ms=1000, approver="manual", approved=True,
        )
        self.assertTrue(a_true.is_approved())

    def test_is_approved_false(self):
        """is_approved returns False when approved is False."""
        a_false = SignalApproval(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            approved_at_ms=1000, approver="manual", approved=False,
        )
        self.assertFalse(a_false.is_approved())

    def test_is_approved_missing_false(self):
        """is_approved returns False when approved key is absent."""
        d = {
            "signal_id": "s1",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "approved_at_ms": 1000,
            "approver": "manual",
        }
        a = SignalApproval.from_dict(d)
        self.assertFalse(a.is_approved())

    def test_serialization_roundtrip(self):
        """to_dict/from_dict roundtrip preserves all fields."""
        original = SignalApproval(
            signal_id="sig_test",
            symbol="ETHUSDT",
            direction="SHORT",
            approved_at_ms=2000000,
            approver="policy_v1",
            reason="Approved by risk gate",
            approved=True,
        )
        d = original.to_dict()
        restored = SignalApproval.from_dict(d)
        self.assertEqual(restored.signal_id, original.signal_id)
        self.assertEqual(restored.symbol, original.symbol)
        self.assertEqual(restored.direction, original.direction)
        self.assertEqual(restored.approved_at_ms, original.approved_at_ms)
        self.assertEqual(restored.approver, original.approver)
        self.assertEqual(restored.reason, original.reason)
        self.assertEqual(restored.approved, original.approved)

    def test_dict_event_type_is_approval(self):
        """to_dict produces event='approval'."""
        a = SignalApproval(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            approved_at_ms=1000, approver="manual",
        )
        self.assertEqual(a.to_dict()["event"], "approval")


class TestApprovedSignalRef(unittest.TestCase):
    """ApprovedSignalRef model tests."""

    def test_valid_construction(self):
        """Valid construction with all fields."""
        sig = ApprovedSignalRef(
            signal_id="sig_456",
            symbol="SOLUSDT",
            direction="LONG",
            entry=100.0,
            sl=95.0,
            tp=110.0,
            score=65.0,
            signal_time=1000000,
            strategy_version="abc123",
            approved_at_ms=1000001,
            approver="manual",
        )
        self.assertEqual(sig.signal_id, "sig_456")
        self.assertEqual(sig.symbol, "SOLUSDT")
        self.assertEqual(sig.direction, "LONG")
        self.assertEqual(sig.entry, 100.0)
        self.assertEqual(sig.sl, 95.0)
        self.assertEqual(sig.tp, 110.0)
        self.assertEqual(sig.score, 65.0)
        self.assertEqual(sig.signal_time, 1000000)
        self.assertEqual(sig.strategy_version, "abc123")

    def test_risk_distance(self):
        """risk_distance = abs(entry - sl)."""
        sig = ApprovedSignalRef(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0,
            score=65.0, signal_time=1000, strategy_version="v1",
            approved_at_ms=1000, approver="manual",
        )
        self.assertEqual(sig.risk_distance(), 5.0)

    def test_rr_ratio_long(self):
        """R:R for LONG: (TP - entry) / (entry - SL). entry=100, sl=95, tp=125 => 25/5 = 5.0."""
        sig = ApprovedSignalRef(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=125.0,
            score=65.0, signal_time=1000, strategy_version="v1",
            approved_at_ms=1000, approver="manual",
        )
        self.assertAlmostEqual(sig.rr_ratio(), 5.0)

    def test_rr_ratio_short(self):
        """R:R for SHORT: (SL - entry) / (entry - TP). entry=100, sl=120, tp=80 => 20/20 = 1.0."""
        sig = ApprovedSignalRef(
            signal_id="s1", symbol="BTCUSDT", direction="SHORT",
            entry=100.0, sl=120.0, tp=80.0,
            score=65.0, signal_time=1000, strategy_version="v1",
            approved_at_ms=1000, approver="manual",
        )
        self.assertAlmostEqual(sig.rr_ratio(), 1.0)

    def test_rr_ratio_zero_risk(self):
        """R:R is 0.0 when risk distance is 0."""
        sig = ApprovedSignalRef(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=100.0, tp=110.0,
            score=65.0, signal_time=1000, strategy_version="v1",
            approved_at_ms=1000, approver="manual",
        )
        self.assertEqual(sig.rr_ratio(), 0.0)

    def test_roundtrip(self):
        """Serialization roundtrip."""
        original = ApprovedSignalRef(
            signal_id="sig_rt",
            symbol="BTCUSDT",
            direction="LONG",
            entry=100.0,
            sl=95.0,
            tp=110.0,
            score=65.0,
            signal_time=1000000,
            strategy_version="abc123",
            approved_at_ms=1000001,
            approver="manual",
        )
        d = original.to_dict()
        restored = ApprovedSignalRef.from_dict(d)
        self.assertEqual(restored.signal_id, original.signal_id)
        self.assertEqual(restored.entry, original.entry)
        self.assertEqual(restored.sl, original.sl)
        self.assertEqual(restored.tp, original.tp)
        self.assertEqual(restored.score, original.score)
        self.assertAlmostEqual(restored.rr_ratio(), original.rr_ratio())

    def test_no_secrets_in_model(self):
        """Models must not contain any secret fields."""
        sig = ApprovedSignalRef(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0,
            score=65.0, signal_time=1000, strategy_version="v1",
            approved_at_ms=1000, approver="manual",
        )
        d = sig.to_dict()
        secret_keywords = ["api_key", "secret", "password", "token", "apisecret"]
        for key in d:
            self.assertNotIn(key.lower(), secret_keywords)

    def test_event_type_is_approved_signal(self):
        """to_dict produces event='approved_signal'."""
        sig = ApprovedSignalRef(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0,
            score=65.0, signal_time=1000, strategy_version="v1",
            approved_at_ms=1000, approver="manual",
        )
        self.assertEqual(sig.to_dict()["event"], "approved_signal")


class TestOrderRequest(unittest.TestCase):
    """OrderRequest model tests."""

    def test_valid_market_order(self):
        """Valid MARKET order request."""
        req = OrderRequest(
            client_order_id="cid_market_001",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            price=None,
            stop_price=None,
            time_in_force=None,
            reduce_only=False,
            close_position=False,
            signal_id="sig_123",
            strategy_version="v1",
            requested_at_ms=2000000,
        )
        self.assertEqual(req.client_order_id, "cid_market_001")
        self.assertEqual(req.order_type, "MARKET")
        self.assertIsNone(req.price)
        self.assertIsNone(req.stop_price)

    def test_valid_limit_order(self):
        """Valid LIMIT order request."""
        req = OrderRequest(
            client_order_id="cid_limit_001",
            symbol="ETHUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=1.0,
            price=2500.0,
            stop_price=None,
            time_in_force="GTC",
            reduce_only=False,
            close_position=False,
            signal_id="sig_456",
            strategy_version="v1",
            requested_at_ms=2000001,
        )
        self.assertEqual(req.price, 2500.0)
        self.assertEqual(req.time_in_force, "GTC")

    def test_stop_market_order(self):
        """Valid STOP_MARKET order request."""
        req = OrderRequest(
            client_order_id="cid_stop_001",
            symbol="BTCUSDT",
            side="SELL",
            order_type="STOP_MARKET",
            quantity=0.5,
            price=None,
            stop_price=95000.0,
            time_in_force=None,
            reduce_only=True,
            close_position=False,
            signal_id="sig_789",
            strategy_version="v1",
            requested_at_ms=2000002,
        )
        self.assertEqual(req.stop_price, 95000.0)
        self.assertTrue(req.reduce_only)

    def test_roundtrip(self):
        """Serialization roundtrip."""
        original = OrderRequest(
            client_order_id="cid_rt",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            price=None,
            stop_price=None,
            time_in_force=None,
            reduce_only=False,
            close_position=False,
            signal_id="sig_rt",
            strategy_version="v1",
            requested_at_ms=1000000,
        )
        d = original.to_dict()
        restored = OrderRequest.from_dict(d)
        self.assertEqual(restored.client_order_id, original.client_order_id)
        self.assertEqual(restored.symbol, original.symbol)
        self.assertEqual(restored.side, original.side)
        self.assertEqual(restored.order_type, original.order_type)
        self.assertEqual(restored.quantity, original.quantity)
        self.assertEqual(restored.signal_id, original.signal_id)


class TestExecutionResult(unittest.TestCase):
    """ExecutionResult model tests."""

    def test_accepted_result(self):
        """Accepted execution result."""
        result = ExecutionResult(
            client_order_id="cid_001",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            signal_id="sig_123",
            status="ACCEPTED",
            message="Order placed successfully",
            binance_order_id=123456789,
            filled_qty=0.5,
            fill_price=95000.0,
            fee=0.05,
            error_code=None,
            executed_at_ms=2000000,
        )
        self.assertEqual(result.status, "ACCEPTED")
        self.assertEqual(result.binance_order_id, 123456789)
        self.assertEqual(result.filled_qty, 0.5)

    def test_rejected_result(self):
        """Rejected execution result."""
        result = ExecutionResult(
            client_order_id="cid_002",
            symbol="ETHUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=1.0,
            signal_id="sig_456",
            status="REJECTED",
            message="Insufficient balance",
            binance_order_id=None,
            filled_qty=None,
            fill_price=None,
            fee=None,
            error_code=-2014,
            executed_at_ms=2000001,
        )
        self.assertEqual(result.status, "REJECTED")
        self.assertIsNone(result.binance_order_id)
        self.assertEqual(result.error_code, -2014)

    def test_error_result(self):
        """Error execution result."""
        result = ExecutionResult(
            client_order_id="cid_003",
            symbol="SOLUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=5.0,
            signal_id="sig_789",
            status="ERROR",
            message="Network timeout",
            error_code=None,
            executed_at_ms=2000002,
        )
        self.assertEqual(result.status, "ERROR")

    def test_roundtrip(self):
        """Serialization roundtrip."""
        original = ExecutionResult(
            client_order_id="cid_rt",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.5,
            signal_id="sig_rt",
            status="ACCEPTED",
            message="OK",
            binance_order_id=999,
            executed_at_ms=1000000,
        )
        d = original.to_dict()
        restored = ExecutionResult.from_dict(d)
        self.assertEqual(restored.status, original.status)
        self.assertEqual(restored.binance_order_id, original.binance_order_id)


class TestEventToDict(unittest.TestCase):
    """event_to_dict dispatches correctly."""

    def test_signal_approval(self):
        """event_to_dict handles SignalApproval."""
        a = SignalApproval(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            approved_at_ms=1000, approver="manual",
        )
        d = event_to_dict(a)
        self.assertEqual(d["event"], "approval")

    def test_approved_signal_ref(self):
        """event_to_dict handles ApprovedSignalRef."""
        sig = ApprovedSignalRef(
            signal_id="s1", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0,
            score=65.0, signal_time=1000, strategy_version="v1",
            approved_at_ms=1000, approver="manual",
        )
        d = event_to_dict(sig)
        self.assertEqual(d["event"], "approved_signal")

    def test_order_request(self):
        """event_to_dict handles OrderRequest."""
        req = OrderRequest(
            client_order_id="cid1", symbol="BTCUSDT", side="BUY",
            order_type="MARKET", quantity=0.5,
            price=None, stop_price=None, time_in_force=None,
            reduce_only=False, close_position=False,
            signal_id="s1", strategy_version="v1",
            requested_at_ms=1000,
        )
        d = event_to_dict(req)
        self.assertEqual(d["event"], "order_request_prepared")

    def test_execution_result(self):
        """event_to_dict handles ExecutionResult."""
        res = ExecutionResult(
            client_order_id="cid1", symbol="BTCUSDT", side="BUY",
            order_type="MARKET", quantity=0.5,
            signal_id="s1", status="ACCEPTED", message="OK",
            executed_at_ms=1000,
        )
        d = event_to_dict(res)
        self.assertEqual(d["event"], "order_result")

    def test_unknown_event_type_raises(self):
        """event_to_dict raises TypeError for unknown type."""
        with self.assertRaises(TypeError):
            event_to_dict("not_a_model")  # type: ignore


class TestEventFromDict(unittest.TestCase):
    """event_from_dict dispatches correctly."""

    def test_approval_from_dict(self):
        """event_from_dict returns SignalApproval."""
        d = {
            "event": "approval",
            "signal_id": "s1",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "approved_at_ms": 1000,
            "approver": "manual",
            "approved": True,
        }
        result = event_from_dict(d)
        self.assertIsInstance(result, SignalApproval)
        self.assertTrue(result.is_approved())

    def test_approved_signal_from_dict(self):
        """event_from_dict returns ApprovedSignalRef."""
        d = {
            "event": "approved_signal",
            "signal_id": "s1",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 100.0,
            "sl": 95.0,
            "tp": 110.0,
            "score": 65.0,
            "signal_time": 1000,
            "strategy_version": "v1",
            "approved_at_ms": 1000,
            "approver": "manual",
        }
        result = event_from_dict(d)
        self.assertIsInstance(result, ApprovedSignalRef)

    def test_order_request_from_dict(self):
        """event_from_dict returns OrderRequest."""
        d = {
            "event": "order_request_prepared",
            "client_order_id": "cid1",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 0.5,
            "price": None,
            "stop_price": None,
            "time_in_force": None,
            "reduce_only": False,
            "close_position": False,
            "signal_id": "s1",
            "strategy_version": "v1",
            "requested_at_ms": 1000,
        }
        result = event_from_dict(d)
        self.assertIsInstance(result, OrderRequest)

    def test_order_result_from_dict(self):
        """event_from_dict returns ExecutionResult."""
        d = {
            "event": "order_result",
            "client_order_id": "cid1",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 0.5,
            "signal_id": "s1",
            "status": "ACCEPTED",
            "message": "OK",
            "executed_at_ms": 1000,
        }
        result = event_from_dict(d)
        self.assertIsInstance(result, ExecutionResult)

    def test_unknown_event_returns_dict(self):
        """Unknown event type returns the raw dict."""
        d = {"event": "unknown_event", "data": "value"}
        result = event_from_dict(d)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["event"], "unknown_event")


if __name__ == "__main__":
    unittest.main(verbosity=2)
