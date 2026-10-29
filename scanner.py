# P5 Paper Observer — forward paper-trading / live-market observation
#
# Observation-only mode. NO orders, NO API keys, NO private endpoints,
# NO leverage, NO Telegram/webhooks, NO auto-trading.
#
# Architecture:
#   Binance public klines / auto-discovery → closed-candle filter → MTF alignment → scanner
#   → paper signal → JSONL log (source of truth) → paper evaluator (P3)
#   → outcome event → JSONL log → CLI/summary
#
# Closure rule: candle.close_time <= poll_time (NEVER open_time).
# MTF alignment: latest_closed_candle[tf] = max(candle where close_time <= T).
# Signal time: close_time of setup candle (5m if 5m-triggered).
# Expiration: ONLY when eligible candles exist with close_time >= T + horizon
#             AND no TP/SL hit. NOT when data is insufficient.
# State: JSONL is authoritative. State cache is disposable/rebuildable.
# State keying: per (symbol, timeframe), NOT global timestamp.
#
# Auto-discovery (P5 USDⓈ-M liquid pair discovery):
#   Uses Binance public market data only:
#     GET fapi.binance.com/fapi/v1/exchangeInfo  → status, quoteAsset
#     GET fapi.binance.com/fapi/v1/ticker/24hr  → 24h quoteVolume
#   Filters: status==TRADING, quoteAsset==USDT, not excluded, not leveraged,
#            24h quoteVolume >= 50,000,000 USDT (existing frozen threshold).
#   No private endpoints, no API keys, no orders, no execution, paper-only.



from __future__ import annotations
import time
import sys
import json
import hashlib
import argparse
import concurrent.futures
import datetime
import os as _os
import signal as _signal
from typing import Any, Dict, List, Optional, Tuple, Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Local imports
from binance.market_data import (
    BinancePublicClient,
    Kline,
    get_latest_closed_per_timeframe,
    get_latest_closed_kline,
    is_data_stale,
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_CANDLE_BUFFER_SEC,
    MAX_SYMBOLS,
    MAX_OPEN_SIGNALS,
)
from binance.contracts import (
    build_symbol_universe,
    is_valid_symbol,
    discover_liquid_usdt_symbols,
    MIN_QUOTE_VOLUME_USDT,
)
from storage.jsonl import (
    jsonl_append,
    jsonl_read,
    jsonl_read_reverse,
    jsonl_find,
    jsonl_has_signal,
    jsonl_has_outcome,
    jsonl_recover_open_signals,
    jsonl_get_open_signal_ids,
    jsonl_get_all_signal_ids,
    jsonl_get_latest_processed_close_time,
    jsonl_count,
)
from scanner_lock import acquire_scanner_lock, release_scanner_lock
from SignalForge.backtest.evaluator import (
    Candle,
    Signal,
    EvaluationResult,
    evaluate_signal,
)
from SignalForge.paper.models import (
    PaperSignalEvent,
    PaperOutcomeEvent,
    DataQualityEvent,
    CycleEndEvent,
    ScannerStateEvent,
    make_signal_id,
    make_strategy_version,
)
from indicators.d15_decision import d15_final_decision


# ---------------------------------------------------------------------------
# P5 Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Signal-only boundary (fail closed)
# ---------------------------------------------------------------------------
# SignalForge must remain SIGNAL-GENERATION ONLY.
# This module reads Binance public market data and generates LONG/SHORT/NO_TRADE
# signals with exact Entry, SL, TP, R:R, score, symbol, and timestamp. It does
# NOT place, cancel, or modify orders; does NOT access the Demo account; does
# NOT require Binance API credentials.
#
# The ONLY bridge to execution/ is file-based: execution/demo_auto.py can
# consume the output file below (data/signals/paper_trade.jsonl) when
# DEMO_AUTO_TRADE=true. For signal-only operation, do NOT run demo_auto.py
# against this file and do NOT set DEMO_AUTO_TRADE=true.
#
# Hard cap on symbol-level fetch concurrency (validated benchmark: c=2).
# Values above this are clamped/fail-closed — never spawn more workers.
_MAX_EXECUTION_MODULE_NAMES = frozenset({
    "execution", "execution.demo_auto", "execution.demo_orders",
    "execution.ledger", "execution.demo_client", "execution.risk",
    "execution.config", "execution.models", "execution.precision",
})

def _assert_no_execution_import() -> None:
    """Fail closed if any execution module is already loaded in the process."""
    loaded = set(sys.modules.keys())
    breached = sorted(
        n for n in loaded
        if n in _MAX_EXECUTION_MODULE_NAMES or n.startswith("execution.")
    )
    if breached:
        raise SystemExit(
            "SignalForge scanner refuses to run with execution modules loaded "
            f"(loaded: {breached}). Restart in a clean process."
        )


MAX_FETCH_CONCURRENCY = 2

# Cap for the AUTO-DISCOVERY universe only (approved expansion; validated
# with the real production run_once path at 74 symbols: 140.7s/cycle,
# 0×429, peak weight 932/2400, 53% headroom before the 300s boundary).
# The manual --symbols safety slice stays governed by MAX_SYMBOLS = 10
# (imported from binance.market_data) and is NOT affected by this value.
AUTO_DISCOVERY_MAX_SYMBOLS: Optional[int] = None
"""Auto-discovery universe cap.

None = no arbitrary truncation; every USDT-M perpetual that passes the
existing eligibility rules (TRADING, quoteAsset=USDT, PERPETUAL,
underlyingType=COIN guard, exclusions, 24h quoteVolume >= 50M USDT) is
returned. The manual --symbols safety slice is still governed by
MAX_SYMBOLS = 10 (imported from binance.market_data) and is NOT affected
by this value.
"""


@dataclass
class P5Config:
    """P5 paper observation configuration."""
    # Symbol universe
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    max_symbols: int = MAX_SYMBOLS

    # Fetch orchestration (symbol-level only; timeframe fetches inside
    # poll_symbol stay sequential; D15/ledger/evaluation stay single-threaded)
    fetch_concurrency: int = MAX_FETCH_CONCURRENCY

    # Polling
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC  # 5 minutes default
    candle_buffer_sec: int = DEFAULT_CANDLE_BUFFER_SEC  # 1 minute

    # Binance client
    timeout_sec: int = 10
    max_retries: int = 5
    backoff_base_sec: int = 2

    # Strategy
    strategy_version: str = ""
    horizon_ms: int = 12 * 3600 * 1000  # 4 hours

    # Fee/slippage (for P3 evaluator, net R calculation only)
    fee_pct: float = 0.001
    slippage_pct: float = 0.001

    # Logging
    log_path: str = "data/signals/paper_trade.jsonl"

    # Discovery volume threshold (USDT 24h quote volume minimum).
    min_quote_volume_usdt: float = MIN_QUOTE_VOLUME_USDT

    # Safety
    max_open_signals: int = MAX_OPEN_SIGNALS

    # Stale data
    stale_max_age_ms: int = 2 * 300 * 1000  # 2 x 5m candles


# ---------------------------------------------------------------------------
# Paper scanner — generates signals from closed candles
# ---------------------------------------------------------------------------

class PaperScanner:
    """
    Scans closed Binance klines and generates paper signals.

    Uses the existing P1 strategy (frozen) to score symbols.
    Generates a signal when the score crosses the threshold (deterministic).

    Closure rule: candle.close_time <= poll_time (NEVER open_time).
    MTF alignment: latest_closed_candle[tf] = max(candle where close_time <= T).
    Signal time: close_time of setup candle (5m if 5m-triggered).
    """

    def __init__(self, config: P5Config):
        self.config = config
        # We don't implement a real scoring engine here — that's P1 territory.
        # For P5 observation, we demonstrate the pipeline with a simplified
        # scoring placeholder that can be replaced with the real P1 scanner.
        # The key P5 requirement is the closed-candle / MTF / signal-lifecycle
        # plumbing, NOT a new scoring algorithm.

    def generate_signals(
        self,
        symbol: str,
        klines_5m: List[Kline],
        klines_15m: List[Kline],
        klines_1h: List[Kline],
        klines_4h: List[Kline],
        klines_1d: List[Kline],
        poll_time_ms: int,
        existing_signal_ids: set,
    ) -> List[PaperSignalEvent]:
        """
        Scan the latest closed candles for a symbol and generate signals.

        Args:
            symbol: trading symbol (e.g. "BTCUSDT")
            klines_*: lists of Kline objects for each timeframe (raw from API)
            poll_time_ms: current poll time in ms
            existing_signal_ids: set of signal_ids already in the JSONL log

        Returns:
            List of PaperSignalEvent to be written to JSONL.
        """
        # Filter to closed candles only (close_time <= poll_time - buffer)
        buffer_ms = self.config.candle_buffer_sec * 1000
        cutoff = poll_time_ms - buffer_ms

        closed_5m = [k for k in klines_5m if k.close_time <= cutoff]
        closed_15m = [k for k in klines_15m if k.close_time <= cutoff]
        closed_1h = [k for k in klines_1h if k.close_time <= cutoff]
        closed_4h = [k for k in klines_4h if k.close_time <= cutoff]
        closed_1d = [k for k in klines_1d if k.close_time <= cutoff]

        if not closed_5m:
            return []

        # Latest closed 5m candle is the potential setup candle
        latest_5m = max(closed_5m, key=lambda k: k.close_time)

        # MTF alignment: for each higher TF, find latest closed candle
        # with close_time <= latest_5m.close_time (the signal time T)
        T = latest_5m.close_time

        # ---------------------------------------------------------------------------
        # Convert Kline objects → dict format for D8/D15 indicator functions.
        # Each dict carries: open_time, close_time, open, high, low, close, volume.
        # ---------------------------------------------------------------------------
        def _kline_to_dict(k: Kline) -> dict:
            return {
                "open_time": k.open_time,
                "close_time": k.close_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }

        # ---------------------------------------------------------------------------
        # Build per-TF historical candle sequences filtered to close_time <= T.
        # This is mandatory anti-lookahead protection: NO candle with
        # close_time > T may enter D8 or any downstream stage.
        # ---------------------------------------------------------------------------
        candles_by_tf = {
            "5m":  [_kline_to_dict(k) for k in closed_5m if k.close_time <= T],
            "15m": [_kline_to_dict(k) for k in closed_15m if k.close_time <= T],
            "1h":  [_kline_to_dict(k) for k in closed_1h  if k.close_time <= T],
            "4h":  [_kline_to_dict(k) for k in closed_4h  if k.close_time <= T],
            "1d":  [_kline_to_dict(k) for k in closed_1d  if k.close_time <= T],
        }

        # candles_5m: the FULL closed 5m sequence through setup, each <= T
        candles_5m = candles_by_tf["5m"]

        # setup_index: index of the setup candle in candles_5m.
        # The setup candle is the latest closed 5m candle, so it is the
        # last element of candles_5m (which is sorted ascending by close_time).
        setup_index = len(candles_5m) - 1
        if setup_index < 0:
            return []

        setup_5m = candles_5m[setup_index]

        # candles_5m_before_setup: excludes the setup candle (indices [0, setup_index))
        candles_5m_before_setup = candles_5m[:setup_index]

        # candles_5m_through_setup: includes the setup candle (indices [0, setup_index])
        candles_5m_through_setup = candles_5m[:setup_index + 1]

        # ---------------------------------------------------------------------------
        # Call D15 exactly once for the setup candle.
        # D15 orchestrates D8 → D7 → D10 → D11/D12 → D13 → D14 and returns:
        #   direction, entry, final_sl, take_profit, signal_time,
        #   final_long, final_short, status
        # ---------------------------------------------------------------------------
        d15_result = d15_final_decision(
            candles_by_tf=candles_by_tf,
            candles_5m=candles_5m,
            setup_index=setup_index,
            setup_5m=setup_5m,
            candles_5m_before_setup=candles_5m_before_setup,
            candles_5m_through_setup=candles_5m_through_setup,
        )

        # ---------------------------------------------------------------------------
        # Non-OK D15 statuses produce NO signal.
        # status == "OK" is the only path that generates a PaperSignal.
        # ---------------------------------------------------------------------------
        if d15_result["status"] != "OK":
            return []

        direction = d15_result["direction"]
        entry = d15_result["entry"]
        final_sl = d15_result["final_sl"]
        take_profit = d15_result["take_profit"]
        signal_time = d15_result["signal_time"]
        final_long = d15_result["final_long"]
        final_short = d15_result["final_short"]

        # ---------------------------------------------------------------------------
        # Duplicate check — uses the ACTUAL calculated direction from D15.
        # Same (symbol, direction, setup close_time) → same signal_id → rejected.
        # existing_signal_ids is NOT mutated here (caller manages the set).
        # ---------------------------------------------------------------------------
        candidate_id = make_signal_id(symbol, direction, T)
        if candidate_id in existing_signal_ids:
            return []

        # ---------------------------------------------------------------------------
        # Extract rr exactly once, after duplicate checking.
        # ---------------------------------------------------------------------------
        rr = d15_result.get("rr")
        if rr is None:
            rr = 0.0

        # ---------------------------------------------------------------------------
        # Score: LONG uses final_long, SHORT uses final_short.
        # ---------------------------------------------------------------------------
        if direction == "LONG":
            score = final_long
        elif direction == "SHORT":
            score = final_short
        else:
            # D15 OK path should only return LONG or SHORT, but guard anyway.
            return []

        # ---------------------------------------------------------------------------
        # Build PaperSignalEvent with fields from D15 result.
        # PaperSignalEvent.event = "signal" (fixed); signal_time = T (setup close_time).
        # strategy_version from config; generated_at = poll_time_ms.
        # ---------------------------------------------------------------------------
        signal_version = self.config.strategy_version or make_strategy_version()
        sig = PaperSignalEvent(
            event="signal",
            signal_id=candidate_id,
            symbol=symbol,
            direction=direction,
            signal_time=T,
            entry=entry,
            tp=take_profit,
            sl=final_sl,
            score=score,
            rr=rr,
            timeframe="5m",
            horizon_ms=self.config.horizon_ms,
            strategy_version=signal_version,
            generated_at=poll_time_ms,
        )
        return [sig]


# ---------------------------------------------------------------------------
# Paper evaluator — evaluates open signals against new closed candles
# ---------------------------------------------------------------------------

class PaperEvaluator:
    """
    Evaluates open paper signals against newly available closed candles.

    Uses the P3 evaluator (real, tested) for TP/SL/expiration logic.
    Wraps the result into paper outcome events for JSONL logging.

    Expiration rule: ONLY when eligible candles exist with close_time >= T + horizon
                     AND no TP/SL hit. NOT when data is insufficient.
    """

    def __init__(self, config: P5Config):
        self.config = config

    def evaluate_open_signals(
        self,
        open_signals: List[PaperSignalEvent],
        new_candles_5m: List[Candle],
        current_poll_time_ms: int,
    ) -> List[PaperOutcomeEvent]:
        """
        Evaluate all open signals against new 5m candles.

        Args:
            open_signals: list of unresolved paper signals from JSONL
            new_candles_5m: list of NEW closed 5m candles since last eval
            current_poll_time_ms: current poll time (for expiration check)

        Returns:
            List of PaperOutcomeEvent for signals that resolved.
        """
        outcomes: List[PaperOutcomeEvent] = []

        for sig_event in open_signals:
            # Reconstruct a P3 Signal from the paper event
            signal = Signal(
                signal_id=sig_event.signal_id,
                symbol=sig_event.symbol,
                direction=sig_event.direction,
                timestamp=sig_event.signal_time,
                entry=sig_event.entry,
                tp=sig_event.tp,
                sl=sig_event.sl,
                score=sig_event.score,
                timeframe=sig_event.timeframe,
                horizon_ms=sig_event.horizon_ms,
            )

            # Determine the horizon end
            T = sig_event.signal_time
            horizon_end = T + sig_event.horizon_ms

            # Filter new_candles to those strictly after T and within horizon
            future_candles = [
                c for c in new_candles_5m
                if c.timestamp > T and c.timestamp <= horizon_end
            ]

            # Check if we have enough data to evaluate expiration:
            # Expiration ONLY if we have at least one candle with close_time >= horizon_end
            # OR if new_candles extend past the horizon.
            has_reach_horizon = any(
                c.timestamp >= horizon_end for c in new_candles_5m
            )

            if not future_candles:
                # No new candles in the evaluation window.
                # If has_reach_horizon is False, we cannot conclude expiration.
                # Keep signal OPEN — do NOT mark as expired.
                continue

            # Evaluate using P3 evaluator
            result = evaluate_signal(
                signal,
                future_candles,
                fee_pct=self.config.fee_pct,
                slippage_pct=self.config.slippage_pct,
            )

            if result.outcome == "EXPIRATION" and not has_reach_horizon:
                continue

            # Determine close_time for the outcome event
            if result.touch_timestamp is not None:
                outcome_close_time = result.touch_timestamp
            elif result.outcome == "EXPIRATION":
                # For expiration, use the horizon_end as the close_time
                # (this is when the signal definitively expired)
                outcome_close_time = horizon_end
            else:
                outcome_close_time = current_poll_time_ms

            # Build outcome event
            outcome_event = PaperOutcomeEvent(
                event={
                    "TP": "tp_hit",
                    "SL": "sl_hit",
                    "EXPIRATION": "expired",
                }.get(result.outcome, "expired"),
                signal_id=result.signal_id,
                symbol=result.symbol,
                direction=result.direction,
                outcome={
                    "TP": "tp_hit",
                    "SL": "sl_hit",
                    "EXPIRATION": "expired",
                }.get(result.outcome, "expired"),
                outcome_time=result.touch_timestamp if result.touch_timestamp is not None else horizon_end,
                entry=signal.entry,
                tp=signal.tp,
                sl=signal.sl,
                gross_r=result.gross_r if result.gross_r is not None else 0.0,
                net_r=result.net_r if result.net_r is not None else 0.0,
                holding_ms=result.holding_ms if result.holding_ms is not None else 0,
                fee=result.fee,
                slippage=result.slippage,
                horizon_ms=signal.horizon_ms,
                strategy_version=self.config.strategy_version,
                close_time=outcome_close_time,
            )

            outcomes.append(outcome_event)

        return outcomes


# ---------------------------------------------------------------------------
# Paper observer — main loop
# ---------------------------------------------------------------------------

class PaperObserver:
    """
    Main paper observation loop.

    Coordinates: kline fetching, signal generation, evaluation, JSONL logging.
    JSONL is the source of truth. State cache is optional/disposable.
    """

    def __init__(self, config: P5Config):
        self.config = config
        self.client = BinancePublicClient(
            timeout_sec=config.timeout_sec,
            max_retries=config.max_retries,
            backoff_base_sec=config.backoff_base_sec,
        )
        self.scanner = PaperScanner(config)
        self.evaluator = PaperEvaluator(config)
        self.log_path = config.log_path

        # Ensure log directory exists
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

        # Persisted cycle counter so restart doesn't reset cycle numbers.
        self._cycle_number = 0

    def get_existing_signal_ids(self) -> set:
        """Get set of all signal_ids already in the JSONL log."""
        return set(jsonl_get_all_signal_ids(self.log_path))

    def get_open_signal_events(self) -> List[PaperSignalEvent]:
        """Recover open (unresolved) signal events from JSONL."""
        open_sigs = jsonl_recover_open_signals(self.log_path)
        events: List[PaperSignalEvent] = []
        for d in open_sigs:
            events.append(PaperSignalEvent(
                event=d.get("event", "signal"),
                signal_id=d.get("signal_id", ""),
                symbol=d.get("symbol", ""),
                direction=d.get("direction", ""),
                signal_time=d.get("signal_time", 0),
                entry=d.get("entry", 0.0),
                tp=d.get("tp", 0.0),
                sl=d.get("sl", 0.0),
                score=d.get("score", 0.0),
                timeframe=d.get("timeframe", "5m"),
                horizon_ms=d.get("horizon_ms", 12 * 3600 * 1000),
                strategy_version=d.get("strategy_version", ""),
                generated_at=d.get("generated_at", 0),
            ))
        return events

    def poll_symbol(
        self,
        symbol: str,
    ) -> Tuple[
        List[Kline],  # 5m
        List[Kline],  # 15m
        List[Kline],  # 1h
        List[Kline],  # 4h
        List[Kline],  # 1d
        int,  # poll_time_ms
        Optional[str],  # error_type
    ]:
        """
        Fetch klines for a single symbol across all 5 timeframes.

        Returns:
            Tuple of (klines_5m, klines_15m, klines_1h, klines_4h, klines_1d,
                     poll_time_ms, error_type or None)
        """
        poll_time_ms = int(time.time() * 1000)

        intervals = ["5m", "15m", "1h", "4h", "1d"]
        all_klines: Dict[str, List[Kline]] = {}

        for interval in intervals:
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1000,
                )
                all_klines[interval] = klines
            except Exception as e:
                error_type = self._classify_error(e)
                # Return empty lists for this symbol on error
                return (
                    [],
                    [],
                    [],
                    [],
                    [],
                    poll_time_ms,
                    error_type,
                )

        return (
            all_klines["5m"],
            all_klines["15m"],
            all_klines["1h"],
            all_klines["4h"],
            all_klines["1d"],
            poll_time_ms,
            None,
        )

    def _classify_error(self, exc: Exception) -> str:
        """Classify an exception into an error type for data_quality events."""
        msg = str(exc).lower()
        if "429" in msg:
            return "rate_limit"
        if "5" in msg and "http" in msg:
            return "server_error"
        if "timeout" in msg or "timed out" in msg:
            return "network_error"
        if "url" in msg or "connection" in msg:
            return "network_error"
        if "malformed" in msg or "json" in msg:
            return "malformed"
        return "unknown"

    def run_once(self) -> Dict[str, Any]:
        """
        Run one poll cycle for all configured symbols.

        Returns a summary dict for logging/monitoring.
        """
        poll_time_ms = int(time.time() * 1000)
        summary = {
            "poll_time_ms": poll_time_ms,
            "symbols_processed": 0,
            "signals_generated": 0,
            "outcomes_resolved": 0,
            "errors": {},
            "data_quality_events": 0,
        }

        all_signal_ids = jsonl_get_all_signal_ids(self.log_path)
        existing_ids = set(all_signal_ids)
        open_events = self.get_open_signal_events()

        # Circuit breaker: intentionally independent of the auto-discovery
        # universe.  A wider universe raises the odds of reaching it;
        # exceeding 50 open signals
        # fails the cycle CLOSED (no processing), by design.
        if len(open_events) > self.config.max_open_signals:
            summary["errors"]["too_many_open"] = f"open signals ({len(open_events)}) exceed max ({self.config.max_open_signals})"
            return summary

        # ------------------------------------------------------------------
        # Phase 1 — FETCH (symbol-level bounded concurrency, GET-only).
        # poll_symbol itself is UNCHANGED: inside each symbol the 5 timeframe
        # requests remain strictly sequential.  Only independent symbols run
        # in parallel, hard-capped at MAX_FETCH_CONCURRENCY (=2, benchmark
        # validated).  fetch_concurrency <= 1 preserves exactly sequential
        # behavior.  Invalid (>2) values fail closed to the cap — never more.
        # ------------------------------------------------------------------
        workers = self.config.fetch_concurrency
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            workers = 1
        workers = min(workers, MAX_FETCH_CONCURRENCY)

        fetch_results: List[Tuple] = []
        if workers <= 1:
            for symbol in self.config.symbols:
                fetch_results.append(self.poll_symbol(symbol))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                # list(pool.map(...)) preserves config.symbols order regardless
                # of completion order; poll_symbol already converts per-TF
                # fetch exceptions into (empty klines, error_type) tuples.
                fetch_results = list(pool.map(self.poll_symbol, self.config.symbols))

        # ------------------------------------------------------------------
        # Phase 2 — PROCESS strictly sequentially in config.symbols order.
        # D15, staleness checks, open-signal evaluation, duplicate suppression
        # and JSONL writes all remain single-threaded with identical semantics
        # to the historical sequential loop.
        # ------------------------------------------------------------------
        for symbol, (klines_5m, klines_15m, klines_1h, klines_4h, klines_1d,
                     poll_ts, error) in zip(self.config.symbols, fetch_results):
            summary["symbols_processed"] += 1

            if error:
                summary["errors"][symbol] = error
                # Write data_quality event
                dq_event = DataQualityEvent(
                    event="data_quality",
                    symbol=symbol,
                    interval="5m",
                    poll_time=poll_ts,
                    message=f"Fetch error: {error}",
                    data_available=False,
                    error_type=error,
                )
                jsonl_append(self.log_path, dq_event.to_dict())
                summary["data_quality_events"] += 1
                continue

            # Check for stale data
            latest_5m = get_latest_closed_kline(klines_5m, poll_ts) if klines_5m else None
            stale_max_age = self.config.stale_max_age_ms
            if is_data_stale(latest_5m, poll_ts, stale_max_age):
                dq_event = DataQualityEvent(
                    event="data_quality",
                    symbol=symbol,
                    interval="5m",
                    poll_time=poll_ts,
                    message=f"Stale data: latest 5m close_time={latest_5m.close_time if latest_5m else 'N/A'}",
                    data_available=False,
                    stale=True,
                )
                jsonl_append(self.log_path, dq_event.to_dict())
                summary["data_quality_events"] += 1
                continue

            # Generate signals
            new_signals = self.scanner.generate_signals(
                symbol=symbol,
                klines_5m=klines_5m,
                klines_15m=klines_15m,
                klines_1h=klines_1h,
                klines_4h=klines_4h,
                klines_1d=klines_1d,
                poll_time_ms=poll_ts,
                existing_signal_ids=existing_ids,
            )

            for sig_event in new_signals:
                jsonl_append(self.log_path, sig_event.to_dict())
                existing_ids.add(sig_event.signal_id)
                summary["signals_generated"] += 1

            # Evaluate open signals for this symbol
            symbol_open = [e for e in open_events if e.symbol == symbol]
            if symbol_open and klines_5m:
                # Convert klines to candles for P3 evaluator
                new_candles = [k.to_candle() for k in klines_5m]

                outcomes = self.evaluator.evaluate_open_signals(
                    symbol_open,
                    new_candles,
                    poll_ts,
                )

                for outcome in outcomes:
                    if not jsonl_has_outcome(self.log_path, outcome.signal_id):
                        jsonl_append(self.log_path, outcome.to_dict())
                        summary["outcomes_resolved"] += 1

        return summary

    def run_loop(self, max_cycles: Optional[int] = None) -> None:
        """
        Run the paper observation loop.

        Args:
            max_cycles: optional limit on number of poll cycles (None = infinite)
        """
        cycle = 0
        try:
            # Write running marker so 'status' shows active scanner.
            state_event = ScannerStateEvent(
                state="running",
                poll_time_ms=int(time.time() * 1000),
                log_path=self.log_path,
                symbols=list(self.config.symbols),
                interval_sec=self.config.poll_interval_sec,
                max_cycles=max_cycles,
            )
            jsonl_append(self.log_path, state_event.to_dict())
            while max_cycles is None or cycle < max_cycles:
                cycle += 1
                t0 = int(time.time() * 1000)
                summary = self.run_once()
                cycle_event = CycleEndEvent(
                    cycle_number=cycle,
                    poll_time_ms=t0,
                    symbols_discovered=len(self.config.symbols),
                    symbols_processed=summary["symbols_processed"],
                    signals_generated=summary["signals_generated"],
                    outcomes_resolved=summary["outcomes_resolved"],
                    data_quality_events=summary["data_quality_events"],
                    error_symbol_count=len(summary.get("errors", {})),
                    scan_duration_ms=int(time.time() * 1000) - t0,
                    min_quote_volume_usdt=self.config.min_quote_volume_usdt,
                )
                jsonl_append(self.log_path, cycle_event.to_dict())
                print(f"[P5] Cycle {cycle}/{max_cycles or '?'}: "
                      f"processed={summary['symbols_processed']}, "
                      f"signals={summary['signals_generated']}, "
                      f"outcomes={summary['outcomes_resolved']}, "
                      f"dq={summary['data_quality_events']}, "
                      f"errors={len(summary.get('errors', {}))}, "
                      f"duration={cycle_event.scan_duration_ms}ms")
                if max_cycles is None or cycle < max_cycles:
                    time.sleep(self.config.poll_interval_sec)
        except KeyboardInterrupt:
            print("[P5] Interrupted by user.")
        except Exception as e:
            print(f"[P5] Fatal error: {e}", file=sys.stderr)
            raise


# ---------------------------------------------------------------------------
# P5 CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the P5 CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="SignalForge P5 — Paper Trading Observation (read-only)",
        prog="scanner.py",
    )
    subparsers = parser.add_subparsers(dest="command", help="P5 commands")

    # --mode paper (default observation mode)
    paper_parser = subparsers.add_parser("paper", help="Run paper observation loop")
    paper_parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Manual symbol list (e.g. BTCUSDT ETHUSDT). When supplied, "
             "auto-discovery is skipped and these symbols are used directly.",
    )
    paper_parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help=f"Poll interval in seconds (default: {DEFAULT_POLL_INTERVAL_SEC})",
    )
    paper_parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Max poll cycles (default: infinite, Ctrl+C to stop)",
    )
    paper_parser.add_argument(
        "--log",
        type=str,
        default="data/signals/paper_trade.jsonl",
        help="JSONL log path (default: data/signals/paper_trade.jsonl)",
    )
    paper_parser.add_argument(
        "--min-volume",
        type=float,
        default=MIN_QUOTE_VOLUME_USDT,
        help=f"Minimum 24h quote volume in USDT for auto-discovery "
             f"(default: {MIN_QUOTE_VOLUME_USDT:,})",
    )
    paper_parser.add_argument(
        "--auto-discover",
        action="store_true",
        default=False,
        help="Explicitly request auto-discovery (same as the default when "
             "--symbols is not supplied). Provided for backward compatibility.",
    )

    # --show-open
    open_parser = subparsers.add_parser("show-open", help="Show open (unresolved) paper signals")
    open_parser.add_argument(
        "--log",
        type=str,
        default="data/signals/paper_trade.jsonl",
        help="JSONL log path",
    )

    # --show-recent N
    recent_parser = subparsers.add_parser("show-recent", help="Show recent paper events")
    recent_parser.add_argument(
        "n",
        type=int,
        default=20,
        nargs="?",
        help="Number of recent events to show (default: 20)",
    )
    recent_parser.add_argument(
        "--log",
        type=str,
        default="data/signals/paper_trade.jsonl",
        help="JSONL log path",
    )

    # --summary --period daily|weekly [--date YYYY-MM-DD]
    summary_parser = subparsers.add_parser("summary", help="On-demand daily/weekly summary")
    summary_parser.add_argument(
        "--period",
        choices=["daily", "weekly"],
        required=True,
        help="Summary period",
    )
    summary_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Reference date YYYY-MM-DD (UTC). Default: today UTC.",
    )
    summary_parser.add_argument(
        "--log",
        type=str,
        default="data/signals/paper_trade.jsonl",
        help="JSONL log path",
    )

    return parser


def cmd_paper(args) -> None:
    """Run the paper observation loop."""
    from binance.contracts import discover_liquid_usdt_symbols
    from scanner_lock import acquire_scanner_lock, release_scanner_lock

    if args.symbols:
        # Explicit manual symbol list — skip auto-discovery.
        # Cap at MAX_SYMBOLS as a safety measure.
        final_symbols = args.symbols[:MAX_SYMBOLS]
        print(f"[P5] Using manual symbol list ({len(final_symbols)} symbols): "
              f"{', '.join(final_symbols)}")
    else:
        # Default: auto-discover liquid USDT-M pairs from Binance public data.
        # Failure here is fatal: we do NOT silently fall back to a hardcoded
        # list.  If discovery cannot be done we stop rather than silently
        # paper-trading a stale manual list.
        print("[P5] Auto-discovering active/liquid USDT-M pairs …")
        client = BinancePublicClient(
            timeout_sec=10,
            max_retries=5,
            backoff_base_sec=2,
        )
        try:
            symbols = discover_liquid_usdt_symbols(
                client,
                max_symbols=AUTO_DISCOVERY_MAX_SYMBOLS,
                min_quote_volume_usdt=args.min_volume,
            )
        except Exception as e:
            print(f"[P5] FATAL: symbol auto-discovery failed: {e}", file=sys.stderr)
            print("[P5] No fallback to manual symbol list — aborting.", file=sys.stderr)
            sys.exit(1)

        if not symbols:
            print("[P5] Auto-discovery succeeded but returned 0 liquid symbols. "
                  "Nothing to observe — aborting.", file=sys.stderr)
            sys.exit(1)

        print(f"[P5] Auto-discovered {len(symbols)} liquid USDT-M symbols: "
              f"{', '.join(symbols)}")
        # discover_liquid_usdt_symbols already caps at max_symbols internally;
        # no additional slice needed here.
        final_symbols = symbols

    if not final_symbols:
        print("[P5] No symbols to observe.  Use --symbols to supply a manual "
              "list, or ensure auto-discovery can reach Binance.", file=sys.stderr)
        sys.exit(1)

    config = P5Config(
        symbols=final_symbols,
        poll_interval_sec=args.interval,
        log_path=args.log,
        strategy_version=make_strategy_version(),
    )
    config.max_open_signals = 500
    observer = PaperObserver(config)
    print(f"[P5] Starting paper observation: {len(config.symbols)} symbols, "
          f"interval={args.interval}s, log={args.log}")
    print("[P5] Mode: PAPER ONLY — NO orders, NO keys, NO private endpoints.")
    print("[P5] Press Ctrl+C to stop.")

    # Single-process guard: acquire the scanner lock before entering the loop.
    # This protects direct `python scanner.py paper ...` usage and any runner-
    # spawned scanner against concurrent execution.  The lock is held for the
    # lifetime of the process and released on normal exit or SIGINT/SIGTERM.
    # Guard here (not at module scope) so non-loop commands (show-open, etc.)
    # are not blocked by the lock.
    if not acquire_scanner_lock(args.log):
        sys.exit(1)

    try:
        observer.run_loop(max_cycles=args.max_cycles)
    finally:
        release_scanner_lock()


def cmd_show_open(args) -> None:
    """Show open (unresolved) paper signals."""
    open_events = jsonl_recover_open_signals(args.log)
    if not open_events:
        print("[P5] No open signals.")
        return

    print(f"[P5] Open signals: {len(open_events)}")
    print("-" * 60)
    for evt in open_events:
        sid = evt.get("signal_id", "?")
        sym = evt.get("symbol", "?")
        direction = evt.get("direction", "?")
        stime = evt.get("signal_time", 0)
        score = evt.get("score", 0.0)
        tp = evt.get("tp", 0.0)
        sl = evt.get("sl", 0.0)
        print(f"  {sid}  {sym}  {direction}  "
              f"time={stime}  score={score:.1f}  "
              f"TP={tp:.2f}  SL={sl:.2f}")


def cmd_show_recent(args) -> None:
    """Show recent paper events."""
    events = jsonl_read_reverse(args.log, limit=args.n)
    if not events:
        print("[P5] No events in log.")
        return

    print(f"[P5] Recent {len(events)} events:")
    print("-" * 60)
    for evt in events:
        etype = evt.get("event", "?")
        sid = evt.get("signal_id", "?")
        sym = evt.get("symbol", "?")
        if etype == "signal":
            direction = evt.get("direction", "?")
            stime = evt.get("signal_time", 0)
            score = evt.get("score", 0.0)
            print(f"  [{etype}] {sid}  {sym}  {direction}  "
                  f"time={stime}  score={score:.1f}")
        elif etype in ("tp_hit", "sl_hit", "expired"):
            outcome = evt.get("outcome", etype)
            gross_r = evt.get("gross_r", 0.0)
            net_r = evt.get("net_r", 0.0)
            hold = evt.get("holding_ms", 0)
            print(f"  [{etype}] {sid}  {sym}  {outcome}  "
                  f"R={gross_r:.2f}  netR={net_r:.2f}  hold={hold}ms")
        elif etype == "data_quality":
            msg = evt.get("message", "?")
            print(f"  [data_quality] {sym}: {msg}")
        else:
            print(f"  [{etype}] {sid}  {sym}")


def _utc_date_from_ms(ms: int) -> str:
    """Return YYYY-MM-DD UTC string for a millisecond timestamp."""
    return datetime.datetime.fromtimestamp(ms / 1000.0, tz=datetime.timezone.utc).strftime("%Y-%m-%d")


def _parse_date_utc(date_str: str) -> datetime.datetime:
    """Parse YYYY-MM-DD as a UTC midnight datetime."""
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)


def _utc_date_to_day_range(date_str: str) -> Tuple[int, int]:
    """Return (start_ms, end_ms) for a UTC day (00:00:00 inclusive to next day 00:00:00 exclusive)."""
    dt = _parse_date_utc(date_str)
    start = int(dt.timestamp() * 1000)
    end = int((dt + datetime.timedelta(days=1)).timestamp() * 1000)
    return start, end


def _utc_week_range_from_date(date_str: str) -> Tuple[int, int]:
    """Return (start_ms, end_ms) for the ISO Mon-Sun week (UTC) containing date_str."""
    dt = _parse_date_utc(date_str)
    # ISO weekday: Monday=1 .. Sunday=7
    weekday = dt.isoweekday()
    monday = dt - datetime.timedelta(days=weekday - 1)
    sunday = monday + datetime.timedelta(days=7)
    start = int(monday.timestamp() * 1000)
    end = int(sunday.timestamp() * 1000)
    return start, end


def _resolve_period_bounds(args) -> Tuple[str, int, int]:
    """Return (ref_date_str, period_start_ms, period_end_ms) from CLI args.

    If --date is omitted, use today UTC. This is the ONLY place "now" is read,
    so the filtering logic itself is deterministic and testable.
    """
    if args.date is None:
        ref_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    else:
        ref_date = args.date

    if args.period == "daily":
        period_start, period_end = _utc_date_to_day_range(ref_date)
    else:
        period_start, period_end = _utc_week_range_from_date(ref_date)
    return ref_date, period_start, period_end


def cmd_summary(args) -> None:
    """On-demand daily/weekly summary from JSONL log, filtered by UTC period."""
    events = jsonl_read(args.log)
    if not events:
        print("[P5] No events in log. No summary available.")
        return

    # Resolve the requested UTC period. If --date omitted, use today UTC.
    ref_date, period_start, period_end = _resolve_period_bounds(args)

    # Build a human-readable period label
    if args.period == "daily":
        period_label = f"Daily ({ref_date} UTC)"
    else:
        monday_dt = _parse_date_utc(ref_date) - datetime.timedelta(days=_parse_date_utc(ref_date).isoweekday() - 1)
        sunday_dt = monday_dt + datetime.timedelta(days=6)
        period_label = (
            f"Weekly ({monday_dt.strftime('%Y-%m-%d')}–"
            f"{sunday_dt.strftime('%Y-%m-%d')} UTC)"
        )

    # Deterministic period predicate (UTC millisecond boundaries, half-open).
    # Signals use signal_time; outcomes use outcome_time; data_quality uses poll_time.
    def _in_period(ev: Dict[str, Any]) -> bool:
        ts = None
        etype = ev.get("event")
        if etype == "signal":
            ts = ev.get("signal_time", 0)
        elif etype in ("tp_hit", "sl_hit", "expired"):
            ts = ev.get("outcome_time", 0)
        elif etype == "data_quality":
            ts = ev.get("poll_time", 0) or ev.get("generated_at", 0)
        else:
            ts = ev.get("signal_time", 0) or ev.get("outcome_time", 0) or ev.get("generated_at", 0)
        if ts is None or not isinstance(ts, (int, float)):
            return False
        return period_start <= ts < period_end

    # Filter signal/outcome events by the period timestamp.
    # Signals: filtered by signal_time.
    # Outcomes: filtered by outcome_time.
    signals = [e for e in events if e.get("event") == "signal" and _in_period(e)]
    outcomes = [e for e in events if e.get("event") in ("tp_hit", "sl_hit", "expired") and _in_period(e)]

    # Open signals: unresolved signals anywhere in the log.
    # A signal created BEFORE this period but still open DURING this period
    # is counted in "Open (unresolved)" for the period, but NOT in
    # "Signals (total)" (which counts only signals whose signal_time falls
    # inside the period). Document this in the output.
    open_ids = jsonl_get_open_signal_ids(args.log)
    open_count = len(open_ids)

    # Data quality events in the period
    dq_events = [e for e in events if e.get("event") == "data_quality" and _in_period(e)]
    dq_count = len(dq_events)

    # Count by outcome type
    tp_count = sum(1 for o in outcomes if o.get("outcome") == "tp_hit")
    sl_count = sum(1 for o in outcomes if o.get("outcome") == "sl_hit")
    expired_count = sum(1 for o in outcomes if o.get("outcome") == "expired")

    # Compute gross R from outcomes
    gross_r_total = sum(o.get("gross_r", 0.0) for o in outcomes)
    net_r_total = sum(o.get("net_r", 0.0) for o in outcomes)

    # Symbols from period signals
    symbols = sorted(set(e.get("symbol", "?") for e in signals))

    # Time range from period events (signal_time / outcome_time)
    times = [e.get("signal_time", 0) for e in signals] + \
            [e.get("outcome_time", 0) for e in outcomes]
    times = [t for t in times if t > 0]
    if times:
        first_time = min(times)
        last_time = max(times)
    else:
        first_time = last_time = 0

    print(f"[P5] {period_label}")
    print("=" * 60)
    print(f"Period: {args.period} — {ref_date}")
    print(f"Symbols: {', '.join(symbols) if symbols else 'none'}")
    print(f"Time range: {first_time} — {last_time}")
    print()
    print(f"Signals (total):    {len(signals)}")
    print(f"  Completed:        {len(outcomes)}")
    print(f"  Open (unresolved): {open_count}")
    print(f"    (includes signals from prior periods still open in this period)")
    print(f"  Data quality events: {dq_count}")
    print()
    print(f"--- Outcomes ---")
    print(f"TP hits:            {tp_count}")
    print(f"SL hits:            {sl_count}")
    print(f"Expired:            {expired_count}")
    print()
    print(f"--- R Metrics ---")
    if len(outcomes) > 0:
        win_rate = tp_count / (tp_count + sl_count) if (tp_count + sl_count) > 0 else 0.0
        print(f"Win rate:           {win_rate * 100:.1f}%")
        print(f"Expectancy (R):     {gross_r_total / len(outcomes):.3f}R")
        print(f"Cumulative R (gross): {gross_r_total:.3f}R")
        print(f"Cumulative R (net):   {net_r_total:.3f}R")
    else:
        print(f"Win rate:           n/a (no completed signals)")
        print(f"Expectancy (R):     n/a")
        print(f"Cumulative R (gross): 0.000R")
        print(f"Cumulative R (net):   0.000R")
    print()
    print(f"--- Sample Size ---")
    print(f"Completed signals:  {len(outcomes)}")
    if len(outcomes) < 30:
        print(f"⚠ INSUFFICIENT SAMPLE: only {len(outcomes)} completed signals (< 30).")
        print(f"  Results are exploratory, not statistically reliable.")
    print()
    print(f"--- Data Quality ---")
    print(f"Data quality events: {dq_count}")
    if dq_events:
        for dq in dq_events[-5:]:
            print(f"  • {dq.get('message', '?')}")
    print()
    print("=" * 60)
    print("NOTE: P5 forward paper results are SEPARATE from P4 historical backtest.")
    print("P4 and P5 are NEVER merged. Side-by-side comparison only.")
    print()
    print("Forward paper-trading results do not guarantee future real-trading performance.")
    print("=" * 60)


def main() -> None:
    _assert_no_execution_import()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "paper":
        cmd_paper(args)
    elif args.command == "show-open":
        cmd_show_open(args)
    elif args.command == "show-recent":
        cmd_show_recent(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
