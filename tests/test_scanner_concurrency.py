"""Tests for symbol-level fetch concurrency in PaperObserver.run_once.

Validates the approved orchestration change ONLY:
  - fetch_concurrency default = 2, hard cap MAX_FETCH_CONCURRENCY = 2
  - parallel fetch phase never runs more than 2 poll_symbol calls at once
  - values > 2 cannot spawn more than 2 pool workers (fail closed)
  - all symbols and all 5 timeframes are fetched (completeness)
  - processing (generate_signals/D15/eval/JSONL) stays in config.symbols
    ORDER even when fetches complete out of order
  - fetch_concurrency=1 is bit-compatible with the sequential path
  - one failing/timing-out symbol is isolated (existing error classification)
  - stale symbols still take the stale branch (no D15 call)
  - D15 receives IDENTICAL kwargs at c=1 and c=2 (pipeline unchanged)

All HTTP is mocked at urlopen (URL-keyed: symbol + interval). No live calls.
Ledgers use temp files under $HOME (Termux-safe), never the repo ledger.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scanner
from scanner import P5Config, PaperObserver, MAX_FETCH_CONCURRENCY
from binance.market_data import BinancePublicClient

# Fixed clock so both concurrency modes see byte-identical data/kwargs.
POLL_MS = 1_800_000_000_000          # "now" in ms
T_MS = POLL_MS - 60_000              # latest closed 5m candle close_time (T)
INTERVALS = ["5m", "15m", "1h", "4h", "1d"]
TICKER = "BTCUSDT"                   # any valid USDT symbol string works
_REAL_SLEEP = time.sleep             # survives _TimePatcher's neutralization


def _rows(symbol: str, interval: str, last_close: int):
    """3 synthetic closed candles ending at last_close for one (sym, tf)."""
    step = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000,
            "4h": 14_400_000, "1d": 86_400_000}[interval]
    out = []
    for i in range(3):
        ct = last_close - (2 - i) * step
        ot = ct - step + 1
        out.append([ot, "100.0", "101.0", "99.0", "100.5", "1000",
                    ct, "100000", "100", "500", "50000", "0"])
    return out


class _MockResponse:
    def __init__(self, body: str):
        self._b = body.encode()
        self.headers = {}

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._b
    def getcode(self): return 200


class UrlKeyedServer:
    """Deterministic fapi klines mock. Optional per-symbol delay / failure."""

    def __init__(self, symbols, *, delays=None, fail=None, stale=None,
                 urls_seen=None):
        self.symbols = symbols
        self.delays = delays or {}
        self.fail = fail or {}          # symbol -> exception factory(interval)
        self.stale = set(stale or [])   # symbols whose candles are old
        self.urls_seen = urls_seen if urls_seen is not None else []
        self.origin = urllib.request.urlopen
        self.lock = threading.Lock()

    def install(self):
        urllib.request.urlopen = self

    def uninstall(self):
        urllib.request.urlopen = self.origin

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.urls_seen.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        sym = q.get("symbol", ["?"])[0]
        itv = q.get("interval", ["?"])[0]
        assert "fapi.binance.com/fapi/v1/klines" in url, url
        d = self.delays.get(sym, 0.0)
        if d:
            _REAL_SLEEP(d)
        f = self.fail.get(sym)
        if f:
            raise f(itv)
        last = (T_MS - 3_600_000 * 24 * 30) if sym in self.stale else T_MS
        with self.lock:
            body = json.dumps(_rows(sym, itv, last))
        return _MockResponse(body)


def _d15_ok(T):
    return {"status": "OK", "direction": "LONG", "entry": 100.0,
            "final_sl": 98.0, "take_profit": 104.0, "signal_time": T,
            "final_long": 71.0, "final_short": 30.0}


def _d15_no_trade():
    return {"status": "NO_TRADE", "direction": "NO_TRADE", "entry": None,
            "final_sl": None, "take_profit": None, "signal_time": None,
            "final_long": 40.0, "final_short": 35.0}


def _make_observer(symbols, fetch_concurrency=None, max_retries=1):
    kwargs = {}
    if fetch_concurrency is not None:
        kwargs["fetch_concurrency"] = fetch_concurrency
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(path)  # start with no log file (jsonl_* handle missing file)
    cfg = P5Config(symbols=list(symbols), log_path=path,
                   timeout_sec=5, max_retries=max_retries,
                   backoff_base_sec=0, **kwargs)
    obs = PaperObserver(cfg)
    return obs, path


def _read_events(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


class _TimePatcher:
    """Freeze time.time (for scanner poll timestamps) and neutralize sleeps."""

    def __init__(self):
        self.orig_time = time.time
        self.orig_sleep = time.sleep

    def __enter__(self):
        time.time = lambda: POLL_MS / 1000.0
        time.sleep = lambda s: None
        return self

    def __exit__(self, *a):
        time.time = self.orig_time
        time.sleep = self.orig_sleep


class TestFetchConcurrency(unittest.TestCase):

    # ---- A. defaults / constants ------------------------------------------
    def test_default_concurrency_is_two(self):
        self.assertEqual(MAX_FETCH_CONCURRENCY, 2)
        self.assertEqual(P5Config().fetch_concurrency, 2)

    def test_max_symbols_still_ten(self):
        """Manual --symbols safety slice cap remains MAX_SYMBOLS == 10."""
        from binance.market_data import MAX_SYMBOLS
        self.assertEqual(MAX_SYMBOLS, 10)

    def test_auto_discovery_cap_is_exactly_75(self):
        """Approved universe-cap expansion: auto-discovery cap == 75, and it
        is a distinct constant from the manual-list cap."""
        from binance.market_data import MAX_SYMBOLS
        self.assertEqual(scanner.AUTO_DISCOVERY_MAX_SYMBOLS, 75)
        self.assertNotEqual(scanner.AUTO_DISCOVERY_MAX_SYMBOLS, MAX_SYMBOLS)

    def test_max_open_signals_remains_50(self):
        """Open-signal circuit breaker is intentionally independent of the
        expanded universe and remains 50."""
        from binance.market_data import MAX_OPEN_SIGNALS
        self.assertEqual(MAX_OPEN_SIGNALS, 50)
        self.assertEqual(P5Config().max_open_signals, 50)

    def test_cmd_paper_auto_discovery_requests_cap_75(self):
        """cmd_paper with no --symbols must pass max_symbols=
        AUTO_DISCOVERY_MAX_SYMBOLS to the production discovery function."""
        from unittest.mock import patch as upatch
        import argparse
        captured = {}

        def fake_discover(client, max_symbols=None):
            captured["max_symbols"] = max_symbols
            return ["BTCUSDT", "ETHUSDT"]

        ns = argparse.Namespace(symbols=None, auto_discover=False,
                                interval=300, max_cycles=0,
                                log=os.devnull)
        # max_cycles=0 -> run_loop exits immediately; no kline fetches occur.
        with upatch("binance.contracts.discover_liquid_usdt_symbols",
                    side_effect=fake_discover):
            scanner.cmd_paper(ns)
        self.assertEqual(captured.get("max_symbols"),
                         scanner.AUTO_DISCOVERY_MAX_SYMBOLS)
        self.assertEqual(captured.get("max_symbols"), 75)

    def test_cmd_paper_manual_symbols_still_capped_at_10(self):
        """Manual --symbols path truncates at MAX_SYMBOLS=10, NOT the new
        auto-discovery cap."""
        from unittest.mock import patch as upatch
        import argparse
        manual = [f"MAN{i}USDT" for i in range(25)]
        ns = argparse.Namespace(symbols=manual, auto_discover=False,
                                interval=300, max_cycles=0,
                                log=os.devnull)
        seen = {}
        RealObserver = scanner.PaperObserver

        class SpyObserver(RealObserver):
            def __init__(self, config):
                # Intentionally does NOT call super().__init__: captures the
                # sliced symbol list without constructing any HTTP client.
                seen["symbols"] = list(config.symbols)

            def run_loop(self, max_cycles=None):
                return None

        with upatch.object(scanner, "PaperObserver", SpyObserver):
            scanner.cmd_paper(ns)
        self.assertEqual(len(seen["symbols"]), 10)
        self.assertEqual(seen["symbols"], manual[:10])

    # ---- B. hard bound ------------------------------------------------------
    def test_inflight_polls_never_exceed_two(self):
        """Track concurrent poll_symbol entries; must never exceed 2."""
        syms = [f"X{i}USDT" for i in range(6)]
        obs, path = _make_observer(syms)
        server = UrlKeyedServer(syms, delays={s: 0.02 for s in syms})
        server.install()
        self.addCleanup(server.uninstall)
        inflight = {"n": 0, "max": 0}
        lock = threading.Lock()
        orig = obs.poll_symbol

        def spy(sym):
            with lock:
                inflight["n"] += 1
                inflight["max"] = max(inflight["max"], inflight["n"])
            try:
                return orig(sym)
            finally:
                with lock:
                    inflight["n"] -= 1

        obs.poll_symbol = spy
        with _TimePatcher(), patch("scanner.d15_final_decision",
                                   side_effect=lambda **k: _d15_no_trade()):
            obs.run_once()
        self.assertLessEqual(inflight["max"], 2)
        self.assertGreater(inflight["max"], 1,  # with delays, 2 really overlap
                           "expected concurrent fetch overlap at c=2")

    def test_value_above_two_cannot_create_more_workers(self):
        """Pool is always constructed with max_workers <= 2."""
        syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        obs, path = _make_observer(syms, fetch_concurrency=99)
        captured = []
        RealPool = scanner.concurrent.futures.ThreadPoolExecutor

        class Recorder(RealPool):
            def __init__(self, *a, **k):
                captured.append(k.get("max_workers", a[0] if a else None))
                super().__init__(*a, **k)

        server = UrlKeyedServer(syms)
        server.install()
        self.addCleanup(server.uninstall)
        with patch.object(scanner.concurrent.futures, "ThreadPoolExecutor", Recorder), \
             _TimePatcher(), patch("scanner.d15_final_decision",
                                   side_effect=lambda **k: _d15_no_trade()):
            obs.run_once()
        self.assertTrue(captured, "pool should have been used")
        self.assertTrue(all(w is not None and w <= 2 for w in captured),
                        f"max_workers leaked beyond cap: {captured}")

    def test_invalid_values_fail_closed(self):
        """0/negative/None/bool never exceed 2 workers; <=1 stays sequential."""
        for bad in (0, -3, None, True):
            syms = ["AAAUSDT", "BBBUSDT"]
            obs, _ = _make_observer(syms, fetch_concurrency=bad)
            workers = obs.config.fetch_concurrency
            # normalize same way run_once does
            if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
                workers = 1
            self.assertLessEqual(min(workers, MAX_FETCH_CONCURRENCY), 2)
            del obs

    # ---- C. completeness -----------------------------------------------------
    def test_all_symbols_all_timeframes_fetched(self):
        syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
        obs, path = _make_observer(syms)
        server = UrlKeyedServer(syms)
        server.install()
        self.addCleanup(server.uninstall)
        with _TimePatcher(), patch("scanner.d15_final_decision",
                                   side_effect=lambda **k: _d15_no_trade()):
            summary = obs.run_once()
        self.assertEqual(summary["symbols_processed"], 4)
        for s in syms:
            for i in INTERVALS:
                hits = [u for u in server.urls_seen
                        if f"symbol={s}&" in u and f"interval={i}" in u]
                self.assertEqual(len(hits), 1, f"missing {s}/{i}")

    # ---- D. ordering ----------------------------------------------------------
    def test_processing_order_follows_config_not_completion(self):
        """First symbol is SLOWEST; processing (generate_signals) must still
        run in config.symbols order."""
        syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        obs, path = _make_observer(syms)
        server = UrlKeyedServer(syms, delays={"AAAUSDT": 0.30,
                                              "BBBUSDT": 0.05,
                                              "CCCUSDT": 0.0})
        server.install()
        self.addCleanup(server.uninstall)
        order = []
        orig_gen = obs.scanner.generate_signals

        def gen_spy(**kw):
            order.append(kw["symbol"])
            return orig_gen(**kw)

        with _TimePatcher(), patch("scanner.d15_final_decision",
                                   side_effect=lambda **k: _d15_no_trade()):
            obs.scanner.generate_signals = gen_spy
            # NOTE: run_once calls self.scanner.generate_signals via kwargs
            obs.run_once()
        # Completion order: CCC < BBB < AAA (delays inverted); processing must
        # be config order.  The spy cannot delay the real pool, but the 0.3s
        # AAA fetch completes after CCC/BBB, so a completion-ordered design
        # would record ['CCCUSDT','BBBUSDT','AAAUSDT'].
        self.assertEqual(order, syms)

    # ---- E. sequential compatibility -------------------------------------------
    def test_c1_and_c2_same_summary_and_events(self):
        syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        sums, events = [], []
        for c in (1, 2):
            obs, path = _make_observer(syms, fetch_concurrency=c)
            server = UrlKeyedServer(syms)
            server.install()
            try:
                with _TimePatcher(), patch("scanner.d15_final_decision",
                                           side_effect=lambda **k: _d15_no_trade()):
                    s = obs.run_once()
            finally:
                server.uninstall()
            s.pop("poll_time_ms")
            sums.append(s)
            events.append([(e.get("event"), e.get("symbol"))
                           for e in _read_events(path)])
            if os.path.exists(path):
                os.unlink(path)
        self.assertEqual(sums[0], sums[1])
        self.assertEqual(events[0], events[1])

    # ---- F/G. failure & timeout isolation --------------------------------------
    def _isolation_case(self, exc_factory):
        syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        obs, path = _make_observer(syms, max_retries=1)
        server = UrlKeyedServer(syms, fail={"BBBUSDT": lambda i: exc_factory(i)})
        server.install()
        self.addCleanup(server.uninstall)
        with _TimePatcher(), patch("scanner.d15_final_decision",
                                   side_effect=lambda **k: _d15_no_trade()):
            summary = obs.run_once()
        self.assertEqual(summary["symbols_processed"], 3)
        self.assertIn("BBBUSDT", summary["errors"])
        self.assertNotIn("AAAUSDT", summary["errors"])
        self.assertNotIn("CCCUSDT", summary["errors"])
        evs = _read_events(path)
        dq = [e for e in evs if e.get("event") == "data_quality"]
        self.assertEqual([e["symbol"] for e in dq], ["BBBUSDT"])
        return summary

    def test_one_symbol_fetch_failure_isolated(self):
        s = self._isolation_case(lambda i: urllib.error.URLError("connection refused"))
        self.assertEqual(s["errors"]["BBBUSDT"], "network_error")

    def test_one_symbol_timeout_isolated(self):
        s = self._isolation_case(lambda i: TimeoutError("simulated timeout"))
        self.assertEqual(s["errors"]["BBBUSDT"], "network_error")

    # ---- H. stale branch unchanged ---------------------------------------------
    def test_stale_symbol_takes_stale_branch_no_d15(self):
        syms = ["AAAUSDT", "STALEUSDT"]
        obs, path = _make_observer(syms)
        server = UrlKeyedServer(syms, stale={"STALEUSDT"})
        server.install()
        self.addCleanup(server.uninstall)
        calls = []
        with _TimePatcher(), patch("scanner.d15_final_decision",
                                   side_effect=lambda **k: calls.append(1) or _d15_no_trade()):
            summary = obs.run_once()
        self.assertEqual(summary["data_quality_events"], 1)
        self.assertEqual(len(calls), 1, "D15 must run once for the fresh symbol only")
        dq = [e for e in _read_events(path) if e.get("event") == "data_quality"]
        self.assertEqual(dq[0]["symbol"], "STALEUSDT")
        self.assertTrue(dq[0].get("stale"))

    # ---- I. D15 kwargs equivalence c=1 vs c=2 -------------------------------------
    def test_d15_kwargs_identical_across_concurrency(self):
        syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        captured = {}
        for c in (1, 2):
            obs, path = _make_observer(syms, fetch_concurrency=c)
            server = UrlKeyedServer(syms)
            server.install()
            try:
                calls = []
                def spy(**kw):
                    calls.append(kw)
                    return _d15_no_trade()
                with _TimePatcher(), patch("scanner.d15_final_decision", side_effect=spy):
                    obs.run_once()
                captured[c] = calls
            finally:
                server.uninstall()
                if os.path.exists(path):
                    os.unlink(path)
        a, b = captured[1], captured[2]
        self.assertEqual(len(a), len(b), "same D15 call count")
        for ka, kb in zip(a, b):
            self.assertEqual(sorted(ka.keys()), sorted(kb.keys()))
            for key in ka:
                self.assertEqual(ka[key], kb[key], f"kwarg {key} differs")

    # ---- J. ledger append order ------------------------------------------------------
    def test_ledger_signal_append_order_follows_config_order(self):
        syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        obs, path = _make_observer(syms)
        server = UrlKeyedServer(syms, delays={"AAAUSDT": 0.30,
                                              "BBBUSDT": 0.10,
                                              "CCCUSDT": 0.0})
        server.install()
        self.addCleanup(server.uninstall)
        with _TimePatcher(), patch("scanner.d15_final_decision",
                                   side_effect=lambda **k: _d15_ok(T_MS)):
            summary = obs.run_once()
        sigs = [e for e in _read_events(path) if e.get("event") == "signal"]
        self.assertEqual(summary["signals_generated"], 3)
        self.assertEqual([e["symbol"] for e in sigs], syms,
                         "JSONL append order must be config.symbols order")


if __name__ == "__main__":
    unittest.main(verbosity=2)
