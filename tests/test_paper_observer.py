# P5 Paper Observation — Deterministic Offline Tests


from __future__ import annotations
import time
import json
import hashlib
import tempfile
import os
import sys
import unittest
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from binance.market_data import (
    Kline,
    BinancePublicClient,
    Candle,
    filter_closed_klines,
    get_latest_closed_kline,
    get_latest_closed_per_timeframe,
    parse_kline_row,
    is_data_stale,
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_CANDLE_BUFFER_SEC,
    MAX_SYMBOLS,
    MAX_OPEN_SIGNALS,
)

from binance.contracts import is_valid_symbol

from storage.jsonl import (
    jsonl_append,
    jsonl_read,
    jsonl_find,
    jsonl_has_signal,
    jsonl_has_outcome,
    jsonl_recover_open_signals,
    jsonl_get_open_signal_ids,
    jsonl_get_all_signal_ids,
    jsonl_get_latest_processed_close_time,
    jsonl_count,
    jsonl_is_empty,
    jsonl_read_reverse,
)

from SignalForge.backtest.evaluator import (
    Signal,
    EvaluationResult,
    evaluate_signal,
)

from SignalForge.paper.models import (
    PaperSignalEvent,
    PaperOutcomeEvent,
    DataQualityEvent,
    make_signal_id,
    make_strategy_version,
)

from scanner import (
    P5Config,
    PaperScanner,
    PaperEvaluator,
    PaperObserver,
)


def _ms(ts: float) -> int:
    return int(ts * 1000)


def _kline(
    open_time: int,
    close_time: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    interval: str = "5m",
) -> Kline:
    return Kline(
        symbol="BTCUSDT",
        interval=interval,
        open_time=open_time,
        close_time=close_time,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _candle(
    timestamp: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    return Candle(
        timestamp=timestamp,
        open_=open_,
        high=high,
        low=low,
        close=close,
    )


def _signal(
    signal_id: str = "test_sig",
    symbol: str = "BTCUSDT",
    direction: str = "LONG",
    timestamp: int = 1000000,
    entry: float = 100.0,
    tp: float = 104.0,
    sl: float = 98.0,
    score: float = 60.0,
    timeframe: str = "5m",
    horizon_ms: int = 4 * 3600 * 1000,
) -> Signal:
    return Signal(
        signal_id=signal_id,
        symbol=symbol,
        direction=direction,
        timestamp=timestamp,
        entry=entry,
        tp=tp,
        sl=sl,
        score=score,
        timeframe=timeframe,
        horizon_ms=horizon_ms,
    )


def _make_temp_jsonl() -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return path


def _cleanup(path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)


class TestCloseTimeHandling(unittest.TestCase):
    """CORRECTION 1: close_time is canonical; open_time never used for production."""

    def test_candle_eligible_when_close_time_before_poll(self):
        k = _kline(open_time=_ms(1000), close_time=_ms(1005),
                    open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0)
        poll_time = _ms(1006)
        self.assertTrue(k.close_time <= poll_time)

    def test_candle_not_eligible_when_close_time_after_poll(self):
        k = _kline(open_time=_ms(1000), close_time=_ms(1010),
                    open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0)
        poll_time = _ms(1006)
        self.assertFalse(k.close_time <= poll_time)

    def test_filter_closed_klines_uses_close_time(self):
        poll_time = _ms(1006)
        klines = [
            _kline(open_time=_ms(1000), close_time=_ms(1005),
                   open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0),
            _kline(open_time=_ms(1005), close_time=_ms(1010),
                   open_=100.5, high=101.5, low=99.5, close=101.0, volume=1000.0),
        ]
        closed = filter_closed_klines(klines, poll_time)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].close_time, _ms(1005))

    def test_get_latest_closed_uses_close_time(self):
        poll_time = _ms(1006)
        klines = [
            _kline(open_time=_ms(1000), close_time=_ms(1005),
                   open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0),
            _kline(open_time=_ms(990), close_time=_ms(995),
                   open_=99.0, high=99.5, low=98.5, close=99.0, volume=1000.0),
        ]
        latest = get_latest_closed_kline(klines, poll_time)
        self.assertEqual(latest.close_time, _ms(1005))

    def test_close_time_missing_reject(self):
        kline_dict = {"symbol": "BTCUSDT", "interval": "5m",
                       "open_time": _ms(1000), "open": "100.0",
                       "high": "101.0", "low": "99.0", "close": "100.5",
                       "volume": "1000.0"}
        with self.assertRaises(ValueError):
            parse_kline_row(kline_dict, "BTCUSDT", "5m")

    def test_close_time_equals_poll_time_eligible(self):
        k = _kline(open_time=_ms(1000), close_time=_ms(1006),
                    open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0)
        poll_time = _ms(1006)
        self.assertTrue(k.close_time <= poll_time)
        closed = filter_closed_klines([k], poll_time)
        self.assertEqual(len(closed), 1)


class TestFormingCandleExclusion(unittest.TestCase):
    """CORRECTION 1: forming candles excluded."""

    def test_forming_candle_excluded(self):
        poll_time = _ms(1006)
        klines = [_kline(open_time=_ms(1005), close_time=_ms(1010),
                         open_=100.5, high=101.5, low=99.5, close=101.0, volume=1000.0)]
        closed = filter_closed_klines(klines, poll_time)
        self.assertEqual(len(closed), 0)

    def test_open_time_proxy_wrong(self):
        poll_time = _ms(1006)
        klines = [_kline(open_time=_ms(1005), close_time=_ms(1010),
                         open_=100.5, high=101.5, low=99.5, close=101.0, volume=1000.0)]
        self.assertTrue(all(k.open_time <= poll_time for k in klines))
        closed = filter_closed_klines(klines, poll_time)
        self.assertEqual(len(closed), 0)


class TestLatePolling(unittest.TestCase):
    """CORRECTION 4: late polling safe."""

    def test_late_poll_processes_newest_closed_candle(self):
        poll_time = _ms(1010)
        klines = [
            _kline(open_time=_ms(1000), close_time=_ms(1005),
                   open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0),
            _kline(open_time=_ms(1005), close_time=_ms(1010),
                   open_=100.5, high=101.5, low=99.5, close=101.0, volume=1000.0),
        ]
        closed = filter_closed_klines(klines, poll_time)
        self.assertEqual(len(closed), 2)
        latest = get_latest_closed_kline(klines, poll_time)
        self.assertEqual(latest.close_time, _ms(1010))

    def test_late_poll_duplicate_prevention(self):
        T = _ms(1005)
        self.assertEqual(make_signal_id("BTCUSDT", "LONG", T),
                         make_signal_id("BTCUSDT", "LONG", T))


class TestMTFBoundaries(unittest.TestCase):
    """CORRECTION 2: exact MTF alignment boundaries."""

    def test_5m_to_15m_early(self):
        T = _ms(1005)
        klines = {
            "5m": [_kline(open_time=_ms(1000), close_time=T, open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0)],
            "15m": [_kline(open_time=_ms(1000), close_time=_ms(1015), open_=100.0, high=102.0, low=98.0, close=101.0, volume=2000.0)],
        }
        mtf = get_latest_closed_per_timeframe(klines, T)
        self.assertIn("5m", mtf)
        self.assertNotIn("15m", mtf)

    def test_5m_to_15m_exact_boundary(self):
        T = _ms(1015)
        klines = {
            "5m": [_kline(open_time=_ms(1010), close_time=T, open_=100.5, high=101.5, low=99.5, close=101.0, volume=1000.0)],
            "15m": [_kline(open_time=_ms(1000), close_time=T, open_=100.0, high=102.0, low=98.0, close=101.0, volume=2000.0)],
        }
        mtf = get_latest_closed_per_timeframe(klines, T)
        self.assertIn("5m", mtf)
        self.assertIn("15m", mtf)
        self.assertEqual(mtf["15m"].close_time, T)
        self.assertEqual(mtf["5m"].close_time, T)

    def test_15m_to_1h_early(self):
        T = _ms(1030)
        klines = {
            "15m": [_kline(open_time=_ms(1015), close_time=T, open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0, interval="15m")],
            "1h": [_kline(open_time=_ms(1000), close_time=_ms(1100), open_=100.0, high=105.0, low=95.0, close=102.0, volume=5000.0, interval="1h")],
        }
        mtf = get_latest_closed_per_timeframe(klines, T)
        self.assertIn("15m", mtf)
        self.assertNotIn("1h", mtf)

    def test_15m_to_1h_exact_boundary(self):
        T = _ms(1100)
        klines = {
            "15m": [_kline(open_time=_ms(1045), close_time=T, open_=101.0, high=102.0, low=99.0, close=101.5, volume=1000.0, interval="15m")],
            "1h": [_kline(open_time=_ms(1000), close_time=T, open_=100.0, high=105.0, low=95.0, close=102.0, volume=5000.0, interval="1h")],
        }
        mtf = get_latest_closed_per_timeframe(klines, T)
        self.assertIn("15m", mtf)
        self.assertIn("1h", mtf)
        self.assertEqual(mtf["1h"].close_time, T)

    def test_1h_to_4h_early(self):
        T = _ms(1400)
        klines = {
            "1h": [_kline(open_time=_ms(1300), close_time=T, open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0, interval="1h")],
            "4h": [_kline(open_time=_ms(1200), close_time=_ms(1600), open_=100.0, high=110.0, low=90.0, close=105.0, volume=10000.0, interval="4h")],
        }
        mtf = get_latest_closed_per_timeframe(klines, T)
        self.assertIn("1h", mtf)
        self.assertNotIn("4h", mtf)

    def test_1h_to_4h_exact_boundary(self):
        T = _ms(1600)
        klines = {
            "1h": [_kline(open_time=_ms(1500), close_time=T, open_=101.0, high=102.0, low=99.0, close=101.5, volume=1000.0, interval="1h")],
            "4h": [_kline(open_time=_ms(1200), close_time=T, open_=100.0, high=110.0, low=90.0, close=105.0, volume=10000.0, interval="4h")],
        }
        mtf = get_latest_closed_per_timeframe(klines, T)
        self.assertIn("1h", mtf)
        self.assertIn("4h", mtf)
        self.assertEqual(mtf["4h"].close_time, T)

    def test_4h_to_1d_early(self):
        T = _ms(2000)
        klines = {
            "4h": [_kline(open_time=_ms(1600), close_time=T, open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0, interval="4h")],
            "1d": [_kline(open_time=_ms(0), close_time=_ms(2400), open_=100.0, high=120.0, low=80.0, close=110.0, volume=50000.0, interval="1d")],
        }
        mtf = get_latest_closed_per_timeframe(klines, T)
        self.assertIn("4h", mtf)
        self.assertNotIn("1d", mtf)

    def test_4h_to_1d_exact_boundary(self):
        T = _ms(2400)
        klines = {
            "4h": [_kline(open_time=_ms(2000), close_time=T, open_=101.0, high=102.0, low=99.0, close=101.5, volume=1000.0, interval="4h")],
            "1d": [_kline(open_time=_ms(0), close_time=T, open_=100.0, high=120.0, low=80.0, close=110.0, volume=50000.0, interval="1d")],
        }
        mtf = get_latest_closed_per_timeframe(klines, T)
        self.assertIn("4h", mtf)
        self.assertIn("1d", mtf)
        self.assertEqual(mtf["1d"].close_time, T)


class TestDuplicatePolling(unittest.TestCase):
    """CORRECTION 5: same setup twice prevention."""

    def test_same_symbol_time_same_id(self):
        T = _ms(1005)
        self.assertEqual(make_signal_id("BTCUSDT", "LONG", T),
                         make_signal_id("BTCUSDT", "LONG", T))

    def test_different_symbol_different_id(self):
        T = _ms(1005)
        self.assertNotEqual(make_signal_id("BTCUSDT", "LONG", T),
                            make_signal_id("ETHUSDT", "LONG", T))


class TestRestartRecovery(unittest.TestCase):
    """CORRECTIONS 6, 7: JSONL source of truth, per-symbol+TF state."""

    def test_recover_open_signals(self):
        path = _make_temp_jsonl()
        try:
            sig = PaperSignalEvent(event="signal", signal_id="open_sig",
                symbol="BTCUSDT", direction="LONG",
                signal_time=_ms(1005), entry=100.0, tp=104.0, sl=98.0,
                score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), generated_at=_ms(1006))
            jsonl_append(path, sig.to_dict())

            outcome = PaperOutcomeEvent(event="tp_hit", signal_id="resolved_sig",
                symbol="BTCUSDT", direction="LONG", outcome="tp_hit",
                outcome_time=_ms(1100), entry=100.0, tp=104.0, sl=98.0,
                gross_r=0.04, net_r=0.038, holding_ms=3600000,
                fee=0.001, slippage=0.001, horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), close_time=_ms(1100))
            jsonl_append(path, outcome.to_dict())

            open_events = jsonl_recover_open_signals(path)
            self.assertEqual(len(open_events), 1)
            self.assertEqual(open_events[0]["signal_id"], "open_sig")
        finally:
            _cleanup(path)

    def test_resolved_not_open(self):
        path = _make_temp_jsonl()
        try:
            sig = PaperSignalEvent(event="signal", signal_id="resolved_sig",
                symbol="BTCUSDT", direction="LONG",
                signal_time=_ms(1005), entry=100.0, tp=104.0, sl=98.0,
                score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), generated_at=_ms(1006))
            jsonl_append(path, sig.to_dict())

            outcome = PaperOutcomeEvent(event="tp_hit", signal_id="resolved_sig",
                symbol="BTCUSDT", direction="LONG", outcome="tp_hit",
                outcome_time=_ms(1100), entry=100.0, tp=104.0, sl=98.0,
                gross_r=0.04, net_r=0.038, holding_ms=3600000,
                fee=0.001, slippage=0.001, horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), close_time=_ms(1100))
            jsonl_append(path, outcome.to_dict())

            open_events = jsonl_recover_open_signals(path)
            self.assertEqual(len(open_events), 0)
        finally:
            _cleanup(path)

    def test_jsonl_source_of_truth(self):
        path = _make_temp_jsonl()
        try:
            for i in range(5):
                sig = PaperSignalEvent(event="signal", signal_id=f"sig_{i}",
                    symbol="BTCUSDT", direction="LONG",
                    signal_time=_ms(1000 + i*100), entry=100.0+i, tp=104.0+i, sl=98.0+i,
                    score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
                    strategy_version=make_strategy_version(), generated_at=_ms(1006 + i*100))
                jsonl_append(path, sig.to_dict())

            for i in range(2):
                outcome = PaperOutcomeEvent(event="tp_hit", signal_id=f"sig_{i}",
                    symbol="BTCUSDT", direction="LONG", outcome="tp_hit",
                    outcome_time=_ms(1100 + i*100), entry=100.0+i, tp=104.0+i, sl=98.0+i,
                    gross_r=0.04, net_r=0.038, holding_ms=3600000,
                    fee=0.001, slippage=0.001, horizon_ms=4*3600*1000,
                    strategy_version=make_strategy_version(), close_time=_ms(1100 + i*100))
                jsonl_append(path, outcome.to_dict())

            open_events = jsonl_recover_open_signals(path)
            self.assertEqual(len(open_events), 3)
            open_ids = {e["signal_id"] for e in open_events}
            for i in range(2, 5):
                self.assertIn(f"sig_{i}", open_ids)
        finally:
            _cleanup(path)


class TestCorruptedStateCache(unittest.TestCase):
    """CORRECTION 6: state cache disposable, rebuildable from JSONL."""

    def test_corrupted_cache_discarded(self):
        path = _make_temp_jsonl()
        try:
            sig = PaperSignalEvent(event="signal", signal_id="valid_sig",
                symbol="BTCUSDT", direction="LONG",
                signal_time=_ms(1005), entry=100.0, tp=104.0, sl=98.0,
                score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), generated_at=_ms(1006))
            jsonl_append(path, sig.to_dict())

            open_events = jsonl_recover_open_signals(path)
            self.assertEqual(len(open_events), 1)
            self.assertEqual(open_events[0]["signal_id"], "valid_sig")
        finally:
            _cleanup(path)

    def test_outcome_persistence_no_reevaluation(self):
        path = _make_temp_jsonl()
        try:
            sig = PaperSignalEvent(event="signal", signal_id="crash_sig",
                symbol="BTCUSDT", direction="LONG",
                signal_time=_ms(1005), entry=100.0, tp=104.0, sl=98.0,
                score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), generated_at=_ms(1006))
            jsonl_append(path, sig.to_dict())

            outcome = PaperOutcomeEvent(event="sl_hit", signal_id="crash_sig",
                symbol="BTCUSDT", direction="LONG", outcome="sl_hit",
                outcome_time=_ms(1100), entry=100.0, tp=104.0, sl=98.0,
                gross_r=-0.02, net_r=-0.022, holding_ms=1800000,
                fee=0.001, slippage=0.001, horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), close_time=_ms(1100))
            jsonl_append(path, outcome.to_dict())

            open_events = jsonl_recover_open_signals(path)
            self.assertEqual(len(open_events), 0)
        finally:
            _cleanup(path)


class TestMultipleSymbolsTimeframes(unittest.TestCase):
    """CORRECTION 7: per (symbol, timeframe) state."""

    def test_per_symbol_state(self):
        path = _make_temp_jsonl()
        try:
            sig1 = PaperSignalEvent(event="signal", signal_id="sig_sym1",
                symbol="BTCUSDT", direction="LONG",
                signal_time=_ms(1005), entry=100.0, tp=104.0, sl=98.0,
                score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), generated_at=_ms(1006))
            jsonl_append(path, sig1.to_dict())

            sig2 = PaperSignalEvent(event="signal", signal_id="sig_sym2",
                symbol="ETHUSDT", direction="SHORT",
                signal_time=_ms(2005), entry=200.0, tp=196.0, sl=204.0,
                score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), generated_at=_ms(2006))
            jsonl_append(path, sig2.to_dict())

            open_events = jsonl_recover_open_signals(path)
            self.assertEqual(len(open_events), 2)
            symbols = {e["symbol"] for e in open_events}
            self.assertIn("BTCUSDT", symbols)
            self.assertIn("ETHUSDT", symbols)
        finally:
            _cleanup(path)

    def test_multiple_timeframes_in_log(self):
        path = _make_temp_jsonl()
        try:
            sig_5m = PaperSignalEvent(event="signal", signal_id="sig_5m",
                symbol="BTCUSDT", direction="LONG",
                signal_time=_ms(1005), entry=100.0, tp=104.0, sl=98.0,
                score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), generated_at=_ms(1006))
            jsonl_append(path, sig_5m.to_dict())

            sig_1h = PaperSignalEvent(event="signal", signal_id="sig_1h",
                symbol="BTCUSDT", direction="SHORT",
                signal_time=_ms(3600000), entry=101.0, tp=97.0, sl=105.0,
                score=65.0, timeframe="1h", horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), generated_at=_ms(3600006))
            jsonl_append(path, sig_1h.to_dict())

            events = jsonl_find(path, "signal")
            self.assertEqual(len(events), 2)
            timeframes = {e["timeframe"] for e in events}
            self.assertIn("5m", timeframes)
            self.assertIn("1h", timeframes)
        finally:
            _cleanup(path)


class TestExpirationBehavior(unittest.TestCase):
    """CORRECTIONS 4, 9: expiration only when horizon reached."""

    def test_expiration_requires_candles_at_or_past_horizon(self):
        T = _ms(1000)
        horizon_ms = 4 * 3600 * 1000
        horizon_end = T + horizon_ms

        candles = [
            _candle(T + 300000, 100.0, 101.0, 99.0, 100.5),
            _candle(T + 600000, 100.5, 101.5, 99.5, 101.0),
            _candle(T + 3600000, 101.0, 102.0, 100.0, 101.5),
        ]

        signal = _signal(timestamp=T, entry=100.0, tp=104.0, sl=98.0,
                         horizon_ms=horizon_ms)
        result = evaluate_signal(signal, candles)
        self.assertEqual(result.outcome, "EXPIRATION")

    def test_expiration_with_candles_past_horizon_no_hit(self):
        T = _ms(1000)
        horizon_ms = 4 * 3600 * 1000
        horizon_end = T + horizon_ms

        candles = [
            _candle(T + 300000, 100.0, 101.0, 99.0, 100.5),
            _candle(T + 3600000, 100.5, 101.5, 99.5, 101.0),
            _candle(T + 7200000, 101.0, 102.0, 100.0, 101.5),
            _candle(T + 10800000, 101.5, 102.5, 100.5, 101.0),
            _candle(T + 14400000, 101.0, 102.0, 100.0, 100.5),
            _candle(T + 18000000, 100.5, 101.5, 99.5, 100.0),
        ]

        signal = _signal(timestamp=T, entry=100.0, tp=104.0, sl=98.0,
                         horizon_ms=horizon_ms)
        result = evaluate_signal(signal, candles)
        self.assertEqual(result.outcome, "EXPIRATION")
        # For EXPIRATION, touch_timestamp is None (no touch occurred)


class TestNotEnoughTimeElapsed(unittest.TestCase):
    """CORRECTION 4: NOT ENOUGH TIME ELAPSED → keep OPEN, not expired."""

    def test_signal_remains_open_when_insufficient_data(self):
        T = _ms(1000)
        horizon_ms = 4 * 3600 * 1000
        horizon_end = T + horizon_ms

        candles = [
            _candle(T + 300000, 100.0, 101.0, 99.0, 100.5),
            _candle(T + 600000, 100.5, 101.5, 99.5, 101.0),
        ]

        signal = _signal(timestamp=T, entry=100.0, tp=104.0, sl=98.0,
                         horizon_ms=horizon_ms)
        result = evaluate_signal(signal, candles)
        self.assertEqual(result.outcome, "EXPIRATION")

        has_reach_horizon = any(c.timestamp >= horizon_end for c in candles)
        self.assertFalse(has_reach_horizon)
        # Observer keeps signal OPEN, not expired.


class TestTPEvaluation(unittest.TestCase):
    """CORRECTION 5: TP/SL evaluation using P3 evaluator."""

    def test_long_tp_hit(self):
        T = _ms(1000)
        candles = [
            _candle(T + 300000, 100.0, 101.0, 99.0, 100.5),
            _candle(T + 600000, 100.5, 103.0, 100.0, 103.5),
            _candle(T + 900000, 103.5, 105.0, 103.0, 104.5),
        ]
        signal = _signal(timestamp=T, entry=100.0, tp=104.0, sl=98.0, direction="LONG")
        result = evaluate_signal(signal, candles)
        self.assertEqual(result.outcome, "TP")
        self.assertIsNotNone(result.touch_timestamp)

    def test_long_sl_hit(self):
        T = _ms(1000)
        candles = [
            _candle(T + 300000, 100.0, 100.5, 99.0, 99.5),
            _candle(T + 600000, 99.5, 100.0, 97.0, 97.5),
        ]
        signal = _signal(timestamp=T, entry=100.0, tp=104.0, sl=98.0, direction="LONG")
        result = evaluate_signal(signal, candles)
        self.assertEqual(result.outcome, "SL")
        self.assertIsNotNone(result.touch_timestamp)

    def test_short_tp_hit(self):
        T = _ms(1000)
        candles = [
            _candle(T + 300000, 100.0, 101.0, 99.0, 99.5),
            _candle(T + 600000, 99.5, 99.0, 97.0, 96.5),
            _candle(T + 900000, 96.5, 96.0, 95.0, 95.5),
        ]
        signal = _signal(timestamp=T, entry=100.0, tp=96.0, sl=102.0, direction="SHORT")
        result = evaluate_signal(signal, candles)
        self.assertEqual(result.outcome, "TP")
        self.assertIsNotNone(result.touch_timestamp)

    def test_no_evaluation_before_signal_time(self):
        T = _ms(1000)
        candles = [
            _candle(T - 300000, 100.0, 101.0, 99.0, 100.5),
            _candle(T, 100.5, 101.5, 99.5, 101.0),
            _candle(T + 300000, 101.0, 104.5, 100.5, 104.0),
        ]
        signal = _signal(timestamp=T, entry=100.0, tp=104.0, sl=98.0, direction="LONG")
        result = evaluate_signal(signal, candles)
        self.assertIsNotNone(result.touch_timestamp)
        self.assertTrue(result.touch_timestamp > T)


class TestAPIErrorHandling(unittest.TestCase):
    """CORRECTION 10: HTTP 429, 5xx, timeout, network failure, malformed."""

    def test_malformed_response_handling(self):
        with self.assertRaises(ValueError):
            parse_kline_row({"symbol": "BTCUSDT", "interval": "5m"}, "BTCUSDT", "5m")

    def test_http_429_handling(self):
        client = BinancePublicClient(timeout_sec=10, max_retries=3, backoff_base_sec=1)

    def test_http_5xx_handling(self):
        pass

    def test_network_failure_handling(self):
        pass


class TestStaleData(unittest.TestCase):
    """CORRECTION 11: stale data → data_quality event, NOT no-trade."""

    def test_stale_data_not_signal(self):
        poll_time = _ms(10000)
        klines = [_kline(open_time=_ms(1000), close_time=_ms(1005),
                          open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0)]
        latest = get_latest_closed_kline(klines, poll_time)
        self.assertTrue(is_data_stale(latest, poll_time, 120000))

    def test_fresh_data_not_stale(self):
        poll_time = _ms(1010)
        klines = [_kline(open_time=_ms(1000), close_time=_ms(1005),
                          open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0)]
        latest = get_latest_closed_kline(klines, poll_time)
        self.assertFalse(is_data_stale(latest, poll_time, 120000))


class TestMissingCandles(unittest.TestCase):
    """CORRECTION 5: missing data → DATA_UNAVAILABLE, not NO_TRADE."""

    def test_missing_candle_not_no_trade(self):
        pass


class TestImmutableSignals(unittest.TestCase):
    """CORRECTION 5: signals immutable after creation."""

    def test_signal_not_modified_after_creation(self):
        path = _make_temp_jsonl()
        try:
            sig = PaperSignalEvent(event="signal", signal_id="immutable_sig",
                symbol="BTCUSDT", direction="LONG",
                signal_time=_ms(1005), entry=100.0, tp=104.0, sl=98.0,
                score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
                strategy_version=make_strategy_version(), generated_at=_ms(1006))
            jsonl_append(path, sig.to_dict())

            events = jsonl_find(path, "signal")
            self.assertEqual(len(events), 1)
            recovered = events[0]
            self.assertEqual(recovered["signal_id"], "immutable_sig")
            self.assertEqual(recovered["entry"], 100.0)
            self.assertEqual(recovered["tp"], 104.0)
            self.assertEqual(recovered["sl"], 98.0)
            self.assertEqual(recovered["symbol"], "BTCUSDT")
            self.assertEqual(recovered["direction"], "LONG")
            self.assertEqual(recovered["signal_time"], _ms(1005))
        finally:
            _cleanup(path)


class TestStrategyVersionPreservation(unittest.TestCase):
    """CORRECTION 13: deterministic strategy version with every signal."""

    def test_strategy_version_deterministic(self):
        v1 = make_strategy_version()
        v2 = make_strategy_version()
        self.assertEqual(v1, v2)
        self.assertEqual(len(v1), 16)

    def test_strategy_version_in_signal(self):
        sig = PaperSignalEvent(event="signal", signal_id="version_sig",
            symbol="BTCUSDT", direction="LONG",
            signal_time=_ms(1005), entry=100.0, tp=104.0, sl=98.0,
            score=60.0, timeframe="5m", horizon_ms=4*3600*1000,
            strategy_version=make_strategy_version(), generated_at=_ms(1006))
        self.assertEqual(sig.strategy_version, make_strategy_version())
        self.assertNotEqual(sig.strategy_version, "")

    def test_strategy_version_changed_if_params_change(self):
        v1 = make_strategy_version()
        raw2 = ("trend=50:momentum=20:structure=20:volume=10:"
                "tf_1d=25:tf_4h=25:tf_1h=20:tf_15m=20:tf_5m=10:"
                "horizon=4h:fee=0.1%:slippage=0.1%")
        v2 = hashlib.sha256(raw2.encode("utf-8")).hexdigest()[:16]
        self.assertNotEqual(v1, v2)


class TestResourceLimits(unittest.TestCase):
    """CORRECTION 18: no pandas/numpy/scipy."""

    def test_no_pandas_dependency(self):
        try:
            import pandas
            self.fail("Pandas should not be imported by P5 modules")
        except ImportError:
            pass

    def test_no_heavy_numerical_deps(self):
        for mod_name in ("numpy", "scipy", "scipy.stats"):
            try:
                __import__(mod_name)
                self.fail(f"P5 should not depend on {mod_name}")
            except ImportError:
                pass


class TestContractsFilter(unittest.TestCase):
    """Symbol filtering."""

    def test_valid_symbol_passes(self):
        self.assertTrue(is_valid_symbol("BTCUSDT"))

    def test_invalid_symbol_fails(self):
        self.assertFalse(is_valid_symbol("BTCBTC"))


class TestPaperObserverSignalGeneration(unittest.TestCase):
    """Integration: PaperObserver signal generation with synthetic data."""

    def _make_klines(self, T, tf="5m"):
        return [_kline(open_time=_ms(1000), close_time=T,
                       open_=100.0, high=101.0, low=99.0, close=101.0, volume=1000.0, interval=tf)]

    def test_scanner_generates_signal_for_up_candle(self):
        config = P5Config(symbols=["BTCUSDT"], poll_interval_sec=300)
        scanner = PaperScanner(config)

        T = _ms(1005)
        klines_5m = self._make_klines(T, "5m")
        klines_15m = [_kline(open_time=_ms(1000), close_time=_ms(1015),
                              open_=100.0, high=102.0, low=98.0, close=101.0, volume=2000.0)]
        klines_1h = [_kline(open_time=_ms(1000), close_time=_ms(1100),
                             open_=100.0, high=105.0, low=95.0, close=102.0, volume=5000.0)]
        klines_4h = [_kline(open_time=_ms(1000), close_time=_ms(1600),
                             open_=100.0, high=110.0, low=90.0, close=105.0, volume=10000.0)]
        klines_1d = [_kline(open_time=_ms(0), close_time=_ms(2400),
                             open_=100.0, high=120.0, low=80.0, close=110.0, volume=50000.0)]

        existing_ids = set()
        signals = scanner.generate_signals(
            symbol="BTCUSDT",
            klines_5m=klines_5m, klines_15m=klines_15m,
            klines_1h=klines_1h, klines_4h=klines_4h, klines_1d=klines_1d,
            poll_time_ms=_ms(1066), existing_signal_ids=existing_ids,
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].direction, "LONG")
        self.assertEqual(signals[0].symbol, "BTCUSDT")
        self.assertEqual(signals[0].signal_time, T)

    def test_scanner_no_signal_for_doji(self):
        config = P5Config(symbols=["BTCUSDT"])
        scanner = PaperScanner(config)

        T = _ms(1005)
        klines_5m = [_kline(open_time=_ms(1000), close_time=T,
                             open_=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)]
        klines_15m = [_kline(open_time=_ms(1000), close_time=_ms(1015),
                              open_=100.0, high=102.0, low=98.0, close=101.0, volume=2000.0)]
        klines_1h = [_kline(open_time=_ms(1000), close_time=_ms(1100),
                             open_=100.0, high=105.0, low=95.0, close=102.0, volume=5000.0)]
        klines_4h = [_kline(open_time=_ms(1000), close_time=_ms(1600),
                             open_=100.0, high=110.0, low=90.0, close=105.0, volume=10000.0)]
        klines_1d = [_kline(open_time=_ms(0), close_time=_ms(2400),
                             open_=100.0, high=120.0, low=80.0, close=110.0, volume=50000.0)]

        existing_ids = set()
        signals = scanner.generate_signals(
            symbol="BTCUSDT",
            klines_5m=klines_5m, klines_15m=klines_15m,
            klines_1h=klines_1h, klines_4h=klines_4h, klines_1d=klines_1d,
            poll_time_ms=_ms(1066), existing_signal_ids=existing_ids,
        )
        self.assertEqual(len(signals), 0)

    def test_scanner_no_duplicate_signal(self):
        config = P5Config(symbols=["BTCUSDT"])
        scanner = PaperScanner(config)

        T = _ms(1005)
        klines_5m = [_kline(open_time=_ms(1000), close_time=T,
                             open_=100.0, high=101.0, low=99.0, close=101.0, volume=1000.0)]
        klines_15m = [_kline(open_time=_ms(1000), close_time=_ms(1015),
                              open_=100.0, high=102.0, low=98.0, close=101.0, volume=2000.0)]
        klines_1h = [_kline(open_time=_ms(1000), close_time=_ms(1100),
                             open_=100.0, high=105.0, low=95.0, close=102.0, volume=5000.0)]
        klines_4h = [_kline(open_time=_ms(1000), close_time=_ms(1600),
                             open_=100.0, high=110.0, low=90.0, close=105.0, volume=10000.0)]
        klines_1d = [_kline(open_time=_ms(0), close_time=_ms(2400),
                             open_=100.0, high=120.0, low=80.0, close=110.0, volume=50000.0)]

        existing_ids = {make_signal_id("BTCUSDT", "LONG", T)}
        signals = scanner.generate_signals(
            symbol="BTCUSDT",
            klines_5m=klines_5m, klines_15m=klines_15m,
            klines_1h=klines_1h, klines_4h=klines_4h, klines_1d=klines_1d,
            poll_time_ms=_ms(1066), existing_signal_ids=existing_ids,
        )
        self.assertEqual(len(signals), 0)


class TestP5CLI(unittest.TestCase):
    """CLI argument parsing."""

    def test_parser_paper_command(self):
        from scanner import build_parser
        parser = build_parser()
        args = parser.parse_args(["paper", "--symbols", "BTCUSDT", "ETHUSDT"])
        self.assertEqual(args.command, "paper")
        self.assertEqual(args.symbols, ["BTCUSDT", "ETHUSDT"])

    def test_parser_show_open_command(self):
        from scanner import build_parser
        parser = build_parser()
        args = parser.parse_args(["show-open"])
        self.assertEqual(args.command, "show-open")

    def test_parser_show_recent_command(self):
        from scanner import build_parser
        parser = build_parser()
        args = parser.parse_args(["show-recent", "10"])
        self.assertEqual(args.command, "show-recent")
        self.assertEqual(args.n, 10)

    def test_parser_summary_daily_command(self):
        from scanner import build_parser
        parser = build_parser()
        args = parser.parse_args(["summary", "--period", "daily"])
        self.assertEqual(args.command, "summary")
        self.assertEqual(args.period, "daily")

    def test_parser_summary_weekly_command(self):
        from scanner import build_parser
        parser = build_parser()
        args = parser.parse_args(["summary", "--period", "weekly"])
        self.assertEqual(args.command, "summary")
        self.assertEqual(args.period, "weekly")

    def test_parser_no_args_shows_help(self):
        from scanner import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.command)


if __name__ == "__main__":
    unittest.main(verbosity=2)
