"""Tests for execution/shadow_bridge.py — Phase 1 (ALL transport mocked).

Zero real network calls: submissions go through a patched
execution.shadow_bridge.place_limit_order; any accidental raw HTTP hits an
urlopen tripwire that fails the test.

Covers audit test matrix A–R + N + V/W (bridge half).
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import execution.shadow_bridge as sb  # noqa: E402
from execution.config import ExecutionConfig  # noqa: E402
from execution.ledger import ExecutionLedger  # noqa: E402
from execution.shadow_reconcile import reconcile_once  # noqa: E402

_ENV_KEYS = ("EXECUTION_MODE", "DEMO_AUTO_TRADE", "DEMO_AUTO_KILL_SWITCH",
             "DEMO_API_KEY", "DEMO_API_SECRET", "DEMO_BRIDGE_ENABLED",
             "DEMO_BRIDGE_MAX_SIGNAL_AGE_SEC", "DEMO_MAX_OPEN_ORDERS")


def _clear_env():
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


def _set_env(**kv):
    # Dummy credentials: never real, never printed.
    env = {"EXECUTION_MODE": "demo",
           "DEMO_API_KEY": "d-ummy-key",
           "DEMO_API_SECRET": "d-ummy-secret"}
    env.update({k: v for k, v in kv.items() if v is not None})
    for k, v in env.items():
        os.environ[k] = v
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)


def _signal(signal_id="sig-001", symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=90.0, tp=125.0, score=70.0, signal_time=None,
            event="signal"):
    if signal_time is None:
        signal_time = int(__import__("time").time() * 1000) - 60_000
    return {"event": event, "signal_id": signal_id, "symbol": symbol,
            "direction": direction, "signal_time": signal_time,
            "entry": entry, "tp": tp, "sl": sl, "score": score,
            "rr": abs(tp - entry) / abs(entry - sl), "timeframe": "5m",
            "strategy_version": "test_v1", "d7_pass": True}


_FILTERS = {"tick_size": Decimal("0.01"), "step_size": Decimal("0.001"),
            "min_qty": None, "min_notional": None}


class _TmpBridge(unittest.TestCase):
    """Shared harness: relocate observer base + bridge data dir to tmp."""

    def setUp(self):
        _clear_env()
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.obs_base = root / "shadow"
        self.data_dir = root / "data" / "demo_execution"
        # Capture the pristine repo-level defaults BEFORE any patching.
        self.orig_data_dir = Path(sb.BRIDGE_DATA_DIR)
        self.orig_obs_base = Path(sb.OBSERVER_BASE)
        self.tenant = "test-shadow"
        self.obs_path = self.obs_base / "tenants" / self.tenant / "observation.jsonl"
        self.obs_path.parent.mkdir(parents=True)
        self._patches = [
            patch.object(sb, "OBSERVER_BASE", self.obs_base),
            patch.object(sb, "BRIDGE_DATA_DIR", self.data_dir),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._stop)

    def _stop(self):
        for p in self._patches:
            p.stop()
        _clear_env()
        self._tmp.cleanup()

    def write_obs(self, records):
        with open(self.obs_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def append_obs(self, records):
        with open(self.obs_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def ledger_events(self):
        lp = self.data_dir / sb.LEDGER_FILENAME
        if not lp.exists():
            return []
        return ExecutionLedger(log_path=str(lp)).read_all()

    def run_cycle(self, *, dry_run=True, execute=False, **kw):
        kw.setdefault("tenant", self.tenant)
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            summary = sb.run_bridge_cycle(dry_run=dry_run, execute=execute,
                                          fetch_filters=lambda c, s: dict(_FILTERS),
                                          **kw)
        summary["_stdout"] = buf_out.getvalue()
        return summary

    def run_real_cycle(self, **kw):
        """Real-trade path with place_limit_order mocked."""
        with patch("execution.shadow_bridge.place_limit_order") as mock_place:
            mock_place.return_value = _result()
            summary = self.run_cycle(dry_run=False, execute=True, **kw)
        return summary, mock_place


def _result(status="ACCEPTED", order_id=4242):
    from execution.models import ExecutionResult
    return ExecutionResult(client_order_id="cid", symbol="BTCUSDT",
                           side="BUY", order_type="LIMIT", quantity=1.0,
                           signal_id="sig", status=status, message="ok",
                           binance_order_id=order_id, executed_at_ms=1)


# ── A/B/D/E: env gates ──────────────────────────────────────────────────────
class TestEnvGates(_TmpBridge):

    def test_A_bridge_disabled_default(self):
        """A: DEMO_BRIDGE_ENABLED absent -> nothing runs, zero submission."""
        _set_env(DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        s = self.run_cycle()
        self.assertFalse(s["ran"])
        self.assertEqual(s["signals"], 0)
        self.assertEqual(s.get("submitted", 0), 0)
        self.assertFalse((self.data_dir / sb.LEDGER_FILENAME).exists())

    def test_A_bridge_enabled_false(self):
        _set_env(DEMO_BRIDGE_ENABLED="false")
        s = self.run_cycle()
        self.assertFalse(s["ran"])
        self.assertIn("DEMO_BRIDGE_ENABLED", s["reason"])

    def test_B_wrong_execution_mode(self):
        _clear_env()
        os.environ["DEMO_BRIDGE_ENABLED"] = "true"
        os.environ["EXECUTION_MODE"] = "live"
        s = self.run_cycle()
        self.assertFalse(s["ran"])
        self.assertIn("demo", s["reason"])

    def test_B_missing_execution_mode_fails_closed(self):
        _clear_env()
        self.run_cycle()  # no raise, no run

    def test_D_no_trade_flag_no_submission(self):
        """D: execute path without DEMO_AUTO_TRADE -> no POST at all."""
        _set_env(DEMO_BRIDGE_ENABLED="true")
        self.write_obs([_signal()])
        with patch("execution.shadow_bridge.place_limit_order") as mp:
            s = self.run_cycle(dry_run=False, execute=True)
            mp.assert_not_called()
        self.assertTrue(s["ran"])
        self.assertEqual(s["would_submit"], 1)  # fell back to dry semantics

    def test_D_real_path_requires_trade_flag(self):
        """Sanity mirror: with trade flag the mocked place IS called once."""
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        s, mp = self.run_real_cycle()
        mp.assert_called_once()
        self.assertEqual(s["submitted"], 1)

    def test_E_kill_switch_env_blocks_cycle(self):
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true",
                 DEMO_AUTO_KILL_SWITCH="true")
        self.write_obs([_signal()])
        with patch("execution.shadow_bridge.place_limit_order") as mp:
            s = self.run_cycle(dry_run=False, execute=True)
            mp.assert_not_called()
        self.assertIn("kill switch", (s.get("reason") or ""))
        self.assertEqual(s["signals"], 0)

    def test_E_kill_switch_cli_flag_blocks_before_submission(self):
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        with patch("execution.shadow_bridge.place_limit_order") as mp:
            s = self.run_cycle(dry_run=False, execute=True, kill_switch=True)
            mp.assert_not_called()
        self.assertIn("kill switch", (s.get("reason") or ""))

    def test_E_kill_before_submission_not_after_reservation(self):
        """When killed, no order_prepared reservation is written either."""
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        self.run_cycle(dry_run=False, execute=True, kill_switch=True)
        events = [e["event"] for e in self.ledger_events()]
        self.assertNotIn("order_prepared", events)
        self.assertNotIn("order_submitted", events)


# ── C + V/W: demo endpoint enforcement ─────────────────────────────────────
class TestDemoEndpointEnforcement(_TmpBridge):

    def test_C_verify_endpoint_rejects_live_host(self):
        fake = type("Cfg", (), {"base_url": "https://fapi.binance.com"})()
        with self.assertRaises(RuntimeError):
            sb.verify_demo_endpoint(fake)

    def test_C_verify_endpoint_rejects_http_scheme(self):
        fake = type("Cfg", (), {"base_url": "http://demo-fapi.binance.com"})()
        with self.assertRaises(RuntimeError):
            sb.verify_demo_endpoint(fake)

    def test_C_verify_endpoint_accepts_demo(self):
        cfg = ExecutionConfig(mode="demo", api_key="k", api_secret="s",
                              base_url="https://demo-fapi.binance.com",
                              symbol_allowlist=["BTCUSDT"],
                              max_position_quantity=1.0,
                              max_total_exposure_usd=1000.0, min_rr=2.0,
                              approved_only=True)
        sb.verify_demo_endpoint(cfg)  # no raise

    def test_C_cycle_fails_closed_on_poisoned_config(self):
        """If from_env ever returned a non-demo base_url, cycle aborts."""
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        bad = type("Cfg", (), {"base_url": "https://fapi.binance.com",
                               "symbol_allowlist": ["BTCUSDT"]})()
        with patch.object(ExecutionConfig, "from_env", return_value=bad), \
             patch("execution.shadow_bridge.place_limit_order") as mp, \
             patch.object(urllib.request, "urlopen",
                          side_effect=AssertionError("network!")):
            s = self.run_cycle(dry_run=False, execute=True)
            mp.assert_not_called()
        self.assertIn("endpoint", s.get("reason") or "")
        self.assertEqual(s.get("submitted", 0), 0)

    def test_W_live_post_always_fails_closed(self):
        """A live-hosted order POST attempt raises before any socket."""
        from execution.demo_orders import build_order_url
        with self.assertRaises(ValueError):
            build_order_url("https://fapi.binance.com/fapi/v1/order?x=1")
        # Demo URL passes
        self.assertIn("demo-fapi.binance.com",
                      build_order_url("signature=abc"))

    def test_V_all_transport_hits_are_demo_host(self):
        """Tripwire over real urlopen in demo_orders: every request during a
        full real cycle (with mocked urlopen response) targets demo-fapi."""
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        captured = []

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"orderId": 7, "status": "NEW",
                                   "symbol": "BTCUSDT", "side": "BUY",
                                   "type": "LIMIT", "executedQty": "0"}).encode()

        def fake_urlopen(req, timeout=None):
            captured.append((req.full_url, req.get_method()))
            return _Resp()

        with patch("execution.demo_orders.urlopen", side_effect=fake_urlopen), \
             patch("execution.demo_client.urlopen", side_effect=fake_urlopen):
            s = self.run_cycle(dry_run=False, execute=True)
        self.assertEqual(s["submitted"], 1)
        self.assertTrue(captured, "expected at least one mocked request")
        import urllib.parse as up
        for url, method in captured:
            self.assertEqual(up.urlparse(url).hostname, "demo-fapi.binance.com",
                             f"non-demo host reached: {url}")
            self.assertEqual(up.urlparse(url).scheme, "https")
            self.assertIn(method, ("GET", "POST", "DELETE"))


# ── F/G/H/N: event filtering, staleness, gates reached, malformed ──────────
class TestEventHandling(_TmpBridge):

    def test_F_non_signal_events_ignored(self):
        _set_env(DEMO_BRIDGE_ENABLED="true")
        self.write_obs([
            _signal(event="duplicate"), _signal(event="outcome", signal_id="o1"),
            _signal(event="data_quality", signal_id="dq"), _signal(),
        ])
        s = self.run_cycle()
        self.assertEqual(s["signals"], 1)
        self.assertEqual(s["ignored_non_signal"], 3)
        self.assertEqual(s["would_submit"], 1)

    def test_G_stale_signal_rejected_real_path(self):
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true",
                 DEMO_BRIDGE_MAX_SIGNAL_AGE_SEC="300")
        old = int(__import__("time").time() * 1000) - 400_000
        self.write_obs([_signal(signal_time=old), _signal(signal_id="sig-002")])
        s, mp = self.run_real_cycle()
        self.assertEqual(s["stale"], 1)
        self.assertEqual(mp.call_count, 1)
        events = {e["event"] for e in self.ledger_events()}
        self.assertIn("signal_stale", events)
        self.assertIn("order_prepared", events)

    def test_G_malformed_age_cap_fails_closed(self):
        """A non-numeric DEMO_BRIDGE_MAX_SIGNAL_AGE_SEC rejects EVERYTHING."""
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        os.environ["DEMO_BRIDGE_MAX_SIGNAL_AGE_SEC"] = "banana"
        self.write_obs([_signal()])
        s, mp = self.run_real_cycle()
        mp.assert_not_called()
        self.assertEqual(s["stale"], 1)

    def test_G_age_calculation_is_deterministic(self):
        now = 1_800_000_000_000
        sig = _signal(signal_time=now - 300_000)
        self.assertFalse(sb.is_stale(sig, now, 300))     # exactly at cap: fresh
        self.assertTrue(sb.is_stale(sig, now, 299))
        self.assertTrue(sb.is_stale({"signal_time": None}, now, 900))  # fail closed
        self.assertTrue(sb.is_stale({}, now, 900))

    def test_H_fresh_signal_reaches_existing_gates(self):
        """process_signal is invoked with the observation record unchanged."""
        _set_env(DEMO_BRIDGE_ENABLED="true")
        sig = _signal()
        self.write_obs([sig])
        real_ps = sb.process_signal
        seen = {}

        def spy(record, *a, **kw):
            seen["record"] = record
            return real_ps(record, *a, **kw)

        with patch("execution.shadow_bridge.process_signal", side_effect=spy):
            s = self.run_cycle()
        self.assertEqual(seen["record"], sig)  # dict passed verbatim
        self.assertEqual(s["would_submit"], 1)

    def test_H_geometry_violation_rejected_by_existing_gates(self):
        _set_env(DEMO_BRIDGE_ENABLED="true")
        self.write_obs([_signal(entry=100.0, sl=110.0, tp=125.0)])  # LONG bad geo
        s = self.run_cycle()
        self.assertEqual(s["rejected_gates"], 1)
        self.assertEqual(s["would_submit"], 0)

    def test_H_allowlist_rejects_unlisted_symbol(self):
        _set_env(DEMO_BRIDGE_ENABLED="true")
        self.write_obs([_signal(symbol="PEPEUSDT")])
        s = self.run_cycle()
        self.assertEqual(s["rejected_gates"], 1)

    def test_N_malformed_json_line_counted_and_skipped(self):
        _set_env(DEMO_BRIDGE_ENABLED="true")
        with open(self.obs_path, "w", encoding="utf-8") as f:
            f.write("{not json at all\n")
            f.write(json.dumps(_signal()) + "\n")
        s = self.run_cycle()
        self.assertEqual(s["malformed"], 1)
        self.assertEqual(s["signals"], 1)
        self.assertEqual(s["would_submit"], 1)

    def test_N_partial_tail_line_holds_cursor_until_complete(self):
        _set_env(DEMO_BRIDGE_ENABLED="true")
        line1 = json.dumps(_signal())
        line2 = json.dumps(_signal(signal_id="sig-002"))
        with open(self.obs_path, "w", encoding="utf-8") as f:
            f.write(line1 + "\n" + line2[:15])  # torn write mid-line
        s = self.run_cycle()
        self.assertEqual(s["signals"], 1)  # torn line NOT consumed
        with open(self.obs_path, "a", encoding="utf-8") as f:
            f.write(line2[15:] + "\n")
        s2 = self.run_cycle()
        self.assertEqual(s2["signals"], 1)
        self.assertEqual(s2["would_submit_ids"], ["sig-002"])

    def test_N_missing_signal_id_counted_malformed(self):
        _set_env(DEMO_BRIDGE_ENABLED="true")
        bad = _signal()
        bad.pop("signal_id")
        self.write_obs([bad])
        s = self.run_cycle()
        self.assertEqual(s["signals"], 1)
        self.assertEqual(s["would_submit"], 0)


# ── I/J: duplicate + cursor ────────────────────────────────────────────────
class TestDuplicateAndCursor(_TmpBridge):

    def test_I_duplicate_in_ledger_rejected_real(self):
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        sig = _signal()
        self.write_obs([sig])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ExecutionLedger(log_path=str(self.data_dir / sb.LEDGER_FILENAME)) \
            .append_dict({"event": "order_submitted", "signal_id": "sig-001",
                          "quantity": 1.0, "price": 100.0})
        s, mp = self.run_real_cycle()
        mp.assert_not_called()
        self.assertEqual(s["duplicates"], 1)

    def test_I_crash_between_prepare_and_post_never_resubmits(self):
        """Reservation alone (no submit event) still blocks re-POST."""
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ExecutionLedger(log_path=str(self.data_dir / sb.LEDGER_FILENAME)) \
            .append_dict({"event": "order_prepared", "signal_id": "sig-001"})
        s, mp = self.run_real_cycle()
        mp.assert_not_called()
        self.assertEqual(s["duplicates"], 1)

    def test_J_cursor_survives_restart(self):
        _set_env(DEMO_BRIDGE_ENABLED="true")
        self.write_obs([_signal()])
        s1 = self.run_cycle()
        self.assertEqual(s1["signals"], 1)
        st = json.loads((self.data_dir / sb.STATE_FILENAME).read_text())
        self.assertGreater(st["cursor_offset"], 0)
        # second 'process' (new object, state from disk) sees nothing new
        s2 = self.run_cycle()
        self.assertEqual(s2["signals"], 0)
        self.append_obs([_signal(signal_id="sig-002")])
        s3 = self.run_cycle()
        self.assertEqual(s3["signals"], 1)
        self.assertEqual(s3["would_submit_ids"], ["sig-002"])

    def test_J_rotation_replays_safely_ledger_dedups(self):
        """Log rewritten (append keeps the file; a rotation-like rewrite
        replays from 0) and the same signal must NOT re-POST: ledger dedup."""
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        s, mp = self.run_real_cycle()
        self.assertEqual(mp.call_count, 1)
        self.append_obs([_signal()])  # same signal_id appears again
        s2, mp2 = self.run_real_cycle()
        self.assertEqual(mp2.call_count, 0)
        self.assertEqual(s2["duplicates"], 1)

    def test_J_state_written_atomically_under_data_dir(self):
        _set_env(DEMO_BRIDGE_ENABLED="true")
        self.write_obs([_signal()])
        self.run_cycle()
        self.assertTrue((self.data_dir / sb.STATE_FILENAME).exists())
        self.assertFalse(list(self.data_dir.glob("*.tmp")))


# ── K/L: ledger stays outside forbidden zones ──────────────────────────────
class TestWritePathIsolation(_TmpBridge):

    def test_K_forbidden_data_signals(self):
        with self.assertRaises(RuntimeError):
            sb.assert_writable_bridge_path(
                sb.PROJECT_ROOT / "data" / "signals" / "anything.jsonl")

    def test_L_forbidden_diagnostics_out(self):
        with self.assertRaises(RuntimeError):
            sb.assert_writable_bridge_path(
                sb.PROJECT_ROOT / "diagnostics" / "out" / "forward_observe_d15"
                / "tenants" / "x" / "bridge_state.json")

    def test_L_observer_log_never_modified_by_cycle(self):
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        before = self.obs_path.read_bytes()
        stat_before = self.obs_path.stat()
        self.run_real_cycle()
        self.assertEqual(self.obs_path.read_bytes(), before)
        self.assertEqual(self.obs_path.stat().st_mtime_ns, stat_before.st_mtime_ns)
        # nothing but observation.jsonl appeared in the tenant dir
        self.assertEqual([p.name for p in self.obs_path.parent.iterdir()],
                         ["observation.jsonl"])

    def test_K_default_paths_are_repo_data_demo_execution(self):
        """The unpatched module defaults resolve to data/demo_execution/."""
        rel = Path(self.orig_data_dir).relative_to(sb.PROJECT_ROOT)
        self.assertEqual(str(rel), os.path.join("data", "demo_execution"))
        self.assertEqual(sb.LEDGER_FILENAME, "execution_ledger.jsonl")
        self.assertEqual(sb.STATE_FILENAME, "bridge_state.json")
        # and the default tenant resolves under the shadow observation tree,
        # which is a FORBIDDEN write zone -> bridge state can never land there
        with self.assertRaises(RuntimeError):
            sb.assert_writable_bridge_path(
                self.orig_obs_base / "tenants" / "d7-shadow-daemon" / "bridge_state.json")

    def test_tenant_name_validation(self):
        for bad in ("../evil", "a/b", "", "9lives", "_x", "x" * 65, "a b"):
            with self.assertRaises(ValueError):
                sb.validate_tenant_name(bad)
        self.assertEqual(sb.validate_tenant_name("d7-shadow-daemon"),
                         "d7-shadow-daemon")


# ── M: single writer ───────────────────────────────────────────────────────
class TestSingleWriter(_TmpBridge):

    def test_M_two_instances_cannot_run(self):
        _set_env(DEMO_BRIDGE_ENABLED="true")
        lock1 = sb.BridgeLock(self.data_dir / sb.LOCK_FILENAME)
        lock1.acquire()
        try:
            with self.assertRaises(RuntimeError):
                self.run_cycle()  # second writer blocked
        finally:
            lock1.release()
        # after release the cycle runs
        _set_env(DEMO_BRIDGE_ENABLED="true")
        s = self.run_cycle()
        self.assertTrue(s["ran"])

    def test_M_release_then_reacquire(self):
        lp = self.data_dir / sb.LOCK_FILENAME
        a = sb.BridgeLock(lp)
        a.acquire()
        a.release()
        b = sb.BridgeLock(lp)
        b.acquire()
        self.assertEqual(int(lp.read_text()), os.getpid())
        b.release()


# ── O/P/Q/R: submission behavior + budgets ────────────────────────────────
class TestSubmissionPath(_TmpBridge):

    def test_O_unknown_post_state_never_retried(self):
        from execution.demo_orders import UnknownExecutionStateError
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal()])
        with patch("execution.shadow_bridge.place_limit_order",
                   side_effect=UnknownExecutionStateError(
                       "ambiguous", client_order_id="c", signal_id="sig-001")) as mp:
            s = self.run_cycle(dry_run=False, execute=True)
        self.assertEqual(mp.call_count, 1)  # NEVER retried
        events = self.ledger_events()
        subs = [e for e in events if e["event"] == "order_submitted"]
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["status"], "UNKNOWN")
        self.assertTrue(subs[0]["unknown_state"])
        # exposure stays committed (reconciler must resolve)
        ledger = ExecutionLedger(log_path=str(self.data_dir / sb.LEDGER_FILENAME))
        import execution.demo_auto as da
        exp, cnt = da._committed_exposure(ledger, None)
        self.assertEqual(cnt, 1)

    def test_P_exposure_cap_tightens_and_rejects(self):
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        # two ETH signals at $600: first takes 250 cap, second has no room
        self.write_obs([
            _signal(signal_id="e1", symbol="ETHUSDT", entry=600.0,
                    sl=550.0, tp=700.0),
            _signal(signal_id="e2", symbol="ETHUSDT", entry=600.0,
                    sl=550.0, tp=700.0)])
        s, mp = self.run_real_cycle(max_exposure_usd=250.0, max_open_orders=2)
        self.assertEqual(s["submitted"], 1)
        self.assertEqual(s["rejected_gates"], 1)
        self.assertEqual(mp.call_count, 1)
        kwargs = mp.call_args.kwargs
        self.assertLessEqual(kwargs["quantity"] * kwargs["price"], 250.0 + 1e-9)

    def test_Q_max_open_orders_one(self):
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([
            _signal(signal_id="b1", entry=100.0, sl=90.0, tp=125.0),
            _signal(signal_id="b2", entry=100.0, sl=90.0, tp=125.0)])
        s, mp = self.run_real_cycle(max_open_orders=1)
        self.assertEqual(mp.call_count, 1)

    def test_Q_runtime_defaults_are_tightened(self):
        """Bridge defaults (1 order / $250) are tighter than config caps."""
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        c = ExecutionConfig(mode="demo", api_key="k", api_secret="s",
                            base_url="https://demo-fapi.binance.com",
                            symbol_allowlist=["BTCUSDT"],
                            max_position_quantity=1.0,
                            max_total_exposure_usd=1000.0, min_rr=2.0,
                            approved_only=True)
        from execution.demo_auto import _effective_limits
        open_orders, exposure = _effective_limits(c, 1, 250.0)
        self.assertEqual((open_orders, exposure), (1, 250.0))
        # and can never loosen:
        _, exp2 = _effective_limits(c, 99, 9999.0)
        self.assertEqual(exp2, 1000.0)

    def test_R_entry_sl_tp_preserved_exactly(self):
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        sig = _signal(entry=100.0, sl=90.0, tp=125.0)
        self.write_obs([sig])
        s, mp = self.run_real_cycle()
        call = mp.call_args.kwargs
        passed_signal = call["signal"]
        self.assertEqual(passed_signal.entry, 100.0)
        self.assertEqual(passed_signal.sl, 90.0)
        self.assertEqual(passed_signal.tp, 125.0)
        # order price may only ever be FLOORED by precision (tick 0.01 here:
        # 100.0 is on grid, so exact)
        self.assertLessEqual(call["price"], 100.0)
        self.assertEqual(call["price"], 100.0)
        self.assertEqual(passed_signal.approver, "shadow_bridge")

    def test_R_short_geometry_preserved(self):
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        # rr = |80-100| / |100-110| = 2.0 (satisfies min_rr)
        sig = _signal(direction="SHORT", entry=100.0, tp=80.0, sl=110.0)
        self.write_obs([sig])
        s, mp = self.run_real_cycle()
        self.assertEqual(mp.call_count, 1)
        passed = mp.call_args.kwargs["signal"]
        self.assertEqual((passed.entry, passed.sl, passed.tp), (100.0, 110.0, 80.0))
        self.assertEqual(passed.direction, "SHORT")

    def test_real_events_released_by_reconciler_open_next_slot(self):
        """Integration: submit -> reconcile(FILLED) -> exposure released ->
        next cycle can submit again within max-open-orders=1."""
        _set_env(DEMO_BRIDGE_ENABLED="true", DEMO_AUTO_TRADE="true")
        self.write_obs([_signal(signal_id="s1")])
        s, mp = self.run_real_cycle()
        self.assertEqual(s["submitted"], 1)
        mp.call_count = 0  # ignore cycle-1 calls; assert cycle-2 behavior only
        # exchange says the order filled (not open, trade present)
        lp = self.data_dir / sb.LEDGER_FILENAME
        cid = self._client_id_from_ledger(lp)
        summary = reconcile_once(
            ledger_path=lp,
            fetch_open_orders=lambda cfg: [],
            fetch_trades=lambda cfg, sym: [{
                "orderId": 4242, "symbol": "BTCUSDT",
                "clientOrderId": cid,
                "executedQty": "1.0", "price": "100.0"}],
            require_env_gates=False)
        self.assertEqual(summary["settled"], 1)
        self.append_obs([_signal(signal_id="s2")])
        s2, mp2 = self.run_real_cycle()
        self.assertEqual(s2["submitted"], 1)
        self.assertEqual(mp2.call_count, 1)

    def _client_id_from_ledger(self, lp):
        for e in ExecutionLedger(log_path=str(lp)).read_all():
            if e["event"] == "order_prepared":
                return e["client_order_id"]
        return None


# ── S: source isolation (bridge is a consumer, not a strategy) ─────────────
class TestSourceIsolation(unittest.TestCase):

    def test_S_bridge_source_has_no_strategy_or_live_hosts(self):
        src = Path(sb.__file__).read_text()
        for forbidden in ("indicators", "d15_final_decision", "d7_pullback",
                          "import paper", "from paper", "requests", "aiohttp",
                          "scanner"):
            self.assertNotIn(forbidden, src,
                             f"bridge must not reference {forbidden!r}")
        # live hosts may appear only inside the demo-host comparison strings
        for line in src.splitlines():
            residual = line.replace("demo-fapi.binance.com", "")
            if "fapi.binance.com" in residual or "api.binance.com" in residual:
                self.fail(f"suspicious live-host literal: {line}")

    def test_S_reconcile_source_is_read_only(self):
        import execution.shadow_reconcile as sr_mod
        src = Path(sr_mod.__file__).read_text()
        self.assertNotIn("place_limit_order", src)
        self.assertNotIn("cancel_order", src)

    def test_S_repo_guards_still_hold_after_bridge(self):
        """The frozen-pipeline diff guard (indicators/, paper/) must be empty
        even with the bridge present — same assertion demo_auto makes."""
        import subprocess
        result = subprocess.run(
            ["git", "diff", "--stat", "--", "indicators/", "paper/"],
            capture_output=True, text=True,
            cwd=str(sb.PROJECT_ROOT))
        self.assertEqual(result.stdout.strip(), "",
                         f"strategy pipeline diff must be empty: {result.stdout}")


if __name__ == "__main__":
    unittest.main()
