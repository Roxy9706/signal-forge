"""Tests for execution/shadow_reconcile.py — Phase 1 (ALL transport mocked).

Fetchers are injected; the default wiring is verified against demo_client
with urlopen patched (never a real socket). Covers matrix T, U, V, W.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.parse
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import execution.shadow_bridge as sb  # noqa: E402
import execution.shadow_reconcile as sr  # noqa: E402
from execution.ledger import ExecutionLedger  # noqa: E402

_ENV_KEYS = ("EXECUTION_MODE", "DEMO_AUTO_TRADE", "DEMO_AUTO_KILL_SWITCH",
             "DEMO_API_KEY", "DEMO_API_SECRET", "DEMO_BRIDGE_ENABLED")


def _clear_env():
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


def _set_env(**kv):
    env = {"EXECUTION_MODE": "demo",
           "DEMO_API_KEY": "d-ummy-key",
           "DEMO_API_SECRET": "d-ummy-secret",
           "DEMO_BRIDGE_ENABLED": "true"}
    env.update({k: v for k, v in kv.items() if v is not None})
    for k, v in env.items():
        os.environ[k] = v
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)


def _open_event(signal_id="sig-001", symbol="BTCUSDT", qty=1.0, price=100.0,
                cid="DEMO-AUTO-1-1", oid=4242, event="order_submitted"):
    return {"event": event, "signal_id": signal_id, "symbol": symbol,
            "direction": "LONG", "client_order_id": cid,
            "binance_order_id": oid, "quantity": qty, "price": price,
            "status": "NEW"}


class _TmpReconcile(unittest.TestCase):
    def setUp(self):
        _clear_env()
        # Reconciler honors the bridge env gates; tests exercise that path
        # explicitly in test_env_gates_respected_by_default.
        _set_env()
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data_dir = root / "data" / "demo_execution"
        self.data_dir.mkdir(parents=True)
        self.ledger_path = self.data_dir / "execution_ledger.jsonl"
        self.ledger = ExecutionLedger(log_path=str(self.ledger_path))
        self.addCleanup(self._done)

    def _done(self):
        _clear_env()
        self._tmp.cleanup()

    def seed(self, *events):
        for e in events:
            self.ledger.append_dict(e)

    def events(self):
        return self.ledger.read_all()

    def reconcile(self, open_orders=None, trades=None, trades_error=False,
                  open_error=False, place_protective=True, **kw):
        def fo(cfg):
            if open_error:
                raise RuntimeError("transport down")
            return list(open_orders or [])

        def ft(cfg, sym):
            if trades_error:
                raise RuntimeError("trades down")
            return list((trades or {}).get(sym, []))

        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            with patch.object(sr, "_PLACE_PROTECTIVE_CALLBACK",
                              None if place_protective else lambda *a, **k: None):
                return sr.reconcile_once(ledger_path=self.ledger_path,
                                         fetch_open_orders=fo, fetch_trades=ft,
                                         **kw)


class TestLifecycleRecognition(_TmpReconcile):
    """T: fill / cancel / expiration states."""

    def test_filled_via_trades(self):
        self.seed(_open_event())
        s = self.reconcile(open_orders=[], trades={"BTCUSDT": [
            {"orderId": 4242, "clientOrderId": "DEMO-AUTO-1-1",
             "executedQty": "1.0", "price": "99.5"}]})
        self.assertEqual(s["settled"], 1)
        settle = [e for e in self.events() if e["event"] == "order_settled"]
        self.assertEqual(len(settle), 1)
        self.assertEqual(settle[0]["terminal_state"], "FILLED")
        self.assertEqual(settle[0]["filled_qty"], 1.0)
        self.assertEqual(settle[0]["fill_price"], 99.5)

    def test_cancelled_no_trades_settles_ended_unfilled(self):
        """GET-only allowlist cannot distinguish CANCELED vs EXPIRED directly;
        'absent from openOrders + zero fills' settles as ENDED_UNFILLED."""
        self.seed(_open_event())
        s = self.reconcile(open_orders=[], trades={"BTCUSDT": []})
        self.assertEqual(s["settled"], 1)
        settle = [e for e in self.events() if e["event"] == "order_settled"]
        self.assertEqual(settle[0]["terminal_state"], "ENDED_UNFILLED")
        self.assertEqual(settle[0]["filled_qty"], 0.0)

    def test_expired_same_path_as_cancelled(self):
        self.seed(_open_event(signal_id="sig-002", cid="cid-2", oid=99))
        s = self.reconcile(open_orders=[], trades={"BTCUSDT": [
            {"orderId": 1, "clientOrderId": "other", "executedQty": "5",
             "price": "100"}]})
        self.assertEqual(s["settled"], 1)
        settle = [e for e in self.events()
                  if e["event"] == "order_settled" and e["signal_id"] == "sig-002"]
        self.assertEqual(settle[0]["terminal_state"], "ENDED_UNFILLED")

    def test_partially_filled_then_closed(self):
        self.seed(_open_event(qty=1.0))
        s = self.reconcile(open_orders=[], trades={"BTCUSDT": [
            {"orderId": 4242, "clientOrderId": "DEMO-AUTO-1-1",
             "executedQty": "0.4", "price": "100.0"}]})
        self.assertEqual(s["settled"], 1)
        settle = [e for e in self.events() if e["event"] == "order_settled"]
        self.assertEqual(settle[0]["terminal_state"],
                         "PARTIALLY_FILLED_THEN_CLOSED")
        self.assertEqual(settle[0]["filled_qty"], 0.4)

    def test_still_open_confirmed_without_settle(self):
        self.seed(_open_event())
        s = self.reconcile(open_orders=[
            {"orderId": 4242, "clientOrderId": "DEMO-AUTO-1-1",
             "symbol": "BTCUSDT", "status": "NEW", "executedQty": "0"}],
            trades={})
        self.assertEqual(s["still_open"], 1)
        self.assertEqual(s["settled"], 0)
        self.assertEqual(s["confirmations"], 1)
        self.assertEqual([e for e in self.events()
                          if e["event"] == "order_settled"], [])
        conf = [e for e in self.events()
                if e["event"] == "reconcile_open_confirmed"]
        self.assertEqual(conf[0]["signal_id"], "sig-001")

    def test_open_list_filled_status_settles_immediately(self):
        self.seed(_open_event())
        s = self.reconcile(open_orders=[
            {"orderId": 4242, "clientOrderId": "DEMO-AUTO-1-1",
             "symbol": "BTCUSDT", "status": "FILLED",
             "executedQty": "1.0", "avgPrice": "100.5"}], trades={})
        self.assertEqual(s["settled"], 1)
        settle = [e for e in self.events() if e["event"] == "order_settled"]
        self.assertEqual(settle[0]["terminal_state"], "FILLED")
        self.assertEqual(settle[0]["fill_price"], 100.5)

    def test_match_by_order_id_when_client_id_absent(self):
        self.seed(_open_event(cid="cid-x"))
        s = self.reconcile(open_orders=[
            {"orderId": 4242, "clientOrderId": "", "symbol": "BTCUSDT",
             "status": "NEW", "executedQty": "0"}], trades={})
        # matched by numeric orderId -> still open
        self.assertEqual(s["still_open"], 1)


class TestExposureRelease(_TmpReconcile):
    """U: settled events release committed exposure."""

    def _committed(self):
        import execution.demo_auto as da
        return da._committed_exposure(self.ledger, None)

    def test_committed_before_reconcile(self):
        self.seed(_open_event())
        exp, cnt = self._committed()
        self.assertEqual(cnt, 1)
        self.assertAlmostEqual(exp, 100.0)

    def test_release_after_settle(self):
        self.seed(_open_event())
        self.reconcile(open_orders=[], trades={"BTCUSDT": [
            {"orderId": 4242, "clientOrderId": "DEMO-AUTO-1-1",
             "executedQty": "1.0", "price": "99.5"}]})
        exp, cnt = self._committed()
        self.assertEqual(cnt, 0)
        self.assertEqual(exp, 0.0)

    def test_no_release_while_still_open(self):
        self.seed(_open_event())
        self.reconcile(open_orders=[{"orderId": 4242,
                                     "clientOrderId": "DEMO-AUTO-1-1",
                                     "status": "NEW"}], trades={})
        exp, cnt = self._committed()
        self.assertEqual(cnt, 1)

    def test_ambiguous_keeps_exposure_fail_closed(self):
        """Trades fetch failure => order stays open, never released."""
        self.seed(_open_event())
        s = self.reconcile(open_orders=[], trades_error=True)
        self.assertEqual(s["settled"], 0)
        self.assertEqual(s["errors"], 1)
        exp, cnt = self._committed()
        self.assertEqual(cnt, 1)
        self.assertIn("reconcile_error", [e["event"] for e in self.events()])

    def test_transport_down_is_conservative_noop(self):
        self.seed(_open_event())
        s = self.reconcile(open_error=True)
        self.assertTrue(s["ran"])
        self.assertEqual(s["settled"], 0)
        self.assertIn("openOrders fetch failed", s["reason"])
        exp, cnt = self._committed()
        self.assertEqual(cnt, 1)

    def test_idempotent_second_pass(self):
        self.seed(_open_event())
        self.reconcile(open_orders=[], trades={"BTCUSDT": [
            {"orderId": 4242, "clientOrderId": "DEMO-AUTO-1-1",
             "executedQty": "1.0", "price": "99.5"}]})
        s2 = self.reconcile(open_orders=[], trades={})
        self.assertEqual(s2["open_on_ledger"], 0)
        self.assertEqual(s2["settled"], 0)


class TestReconcileSafety(_TmpReconcile):
    """No cancellation, env gates, default demo-only wiring."""

    def test_never_cancels(self):
        """Reconcile module must not import or call demo_cancel at all."""
        import inspect
        import re as _re
        src = inspect.getsource(sr)
        self.assertIsNone(
            _re.search(r"^\s*(from|import)\s+execution\.demo_cancel", src, _re.M),
            "reconciler must never import the cancel module")
        self.assertNotIn("cancel_order(", src)
        # and a full pass produces no cancellation event either
        self.seed(_open_event())
        self.reconcile(open_orders=[], trades={})
        self.assertNotIn("order_cancelled", [e["event"] for e in self.events()])

    def test_env_gates_respected_by_default(self):
        _clear_env()
        self.seed(_open_event())
        s = sr.reconcile_once(ledger_path=self.ledger_path)
        self.assertFalse(s["ran"])
        self.assertIn("EXECUTION_MODE", s["reason"])

        _clear_env()
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = "d-ummy-key"
        os.environ["DEMO_API_SECRET"] = "d-ummy-secret"
        s = sr.reconcile_once(ledger_path=self.ledger_path)
        self.assertFalse(s["ran"])
        self.assertIn("DEMO_BRIDGE_ENABLED", s["reason"])

    def test_kill_switch_blocks(self):
        _set_env(DEMO_AUTO_KILL_SWITCH="true")
        self.seed(_open_event())
        s = sr.reconcile_once(ledger_path=self.ledger_path)
        self.assertFalse(s["ran"])
        self.assertIn("KILL_SWITCH", s["reason"])

    def test_V_default_fetchers_use_demo_client_get_allowlist(self):
        """Defaults route through execute_get -> _check_demo_url. Patch at
        the transport seam; every captured URL must be demo-fapi https GET."""
        _set_env()
        self.seed(_open_event())
        captured = []

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"[]"

        def fake_urlopen(req, timeout=None):
            captured.append((req.full_url, req.get_method()))
            return _Resp()

        with patch("execution.demo_client.urlopen", side_effect=fake_urlopen):
            s = sr.reconcile_once(ledger_path=self.ledger_path)
        self.assertEqual(s["open_on_ledger"], 1)
        hosts = {(urllib.parse.urlparse(u).hostname,
                  urllib.parse.urlparse(u).scheme, m) for u, m in captured}
        self.assertTrue(hosts)
        for host, scheme, method in hosts:
            self.assertEqual(host, "demo-fapi.binance.com")
            self.assertEqual(scheme, "https")
            self.assertEqual(method, "GET")  # reconcile is strictly read-only
        paths = {urllib.parse.urlparse(u).path for u, _ in captured}
        self.assertTrue(paths <= {"/fapi/v1/openOrders", "/fapi/v1/userTrades"},
                        f"unexpected paths: {paths}")

    def test_W_reconcile_never_posts(self):
        """Tripwire: both POST (demo_orders) and DELETE (demo_cancel) urlopen
        are patched to explode; a full reconcile must complete untouched."""
        _set_env()
        self.seed(_open_event())

        def boom(*a, **k):
            raise AssertionError("WRITE-VERB NETWORK CALL DURING RECONCILE")

        with patch("execution.demo_client.urlopen",
                   side_effect=lambda req, timeout=None: (
                       boom() if req.get_method() != "GET" else _list_resp())), \
             patch("execution.demo_orders.urlopen", side_effect=boom), \
             patch("execution.demo_cancel.urlopen", side_effect=boom):
            s = sr.reconcile_once(ledger_path=self.ledger_path)
        self.assertTrue(s["ran"])

    def test_forbidden_ledger_path_rejected(self):
        """Reconciler shares the bridge write-zone guard."""
        with self.assertRaises(RuntimeError):
            sr.reconcile_once(
                ledger_path=sb.PROJECT_ROOT / "data" / "signals" / "x.jsonl",
                require_env_gates=False)

    def test_empty_ledger_no_network(self):
        _set_env()
        with patch("execution.demo_client.urlopen",
                   side_effect=AssertionError("network!")):
            s = sr.reconcile_once(ledger_path=self.ledger_path)
        self.assertTrue(s["ran"])
        self.assertEqual(s["open_on_ledger"], 0)

    def test_unknown_status_stays_open(self):
        """An exchange status outside the known vocab (gateway oddity) must
        never settle: it stays confirmed-open (conservative)."""
        self.seed(_open_event())
        s = self.reconcile(open_orders=[{"orderId": 4242,
                                         "clientOrderId": "DEMO-AUTO-1-1",
                                         "status": "SOMETHING_NEW"}],
                          trades={})
        self.assertEqual(s["settled"], 0)
        self.assertEqual(s["still_open"], 1)


def _list_resp():
    class _R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"[]"
    return _R()


if __name__ == "__main__":
    unittest.main()
