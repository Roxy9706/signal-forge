# P5 Paper Trade Models
#
# Immutable dataclasses for paper observation events.
# P5 is observation-only — NO orders, NO API keys, NO private endpoints,
# NO leverage, NO Telegram/webhooks, NO auto-trading.
#
# All events are appended to a JSONL log which is the source of truth.
# State cache (if any) is disposable/rebuildable from JSONL.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import hashlib


# ---------------------------------------------------------------------------
# Paper Signal Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperSignalEvent:
    """
    Represents a paper signal generated from closed candles.

    signal_time = close_time of the setup candle (5m if 5m-triggered).
    All data used must be available at signal_time.
    """
    event: str = "signal"
    signal_id: str = ""
    symbol: str = ""
    direction: str = ""  # "LONG" or "SHORT"
    signal_time: int = 0  # close_time of setup candle (ms)
    entry: float = 0.0    # close price of setup candle
    tp: float = 0.0
    sl: float = 0.0
    score: float = 0.0    # 0-100
    rr: float = 0.0       # R:R ratio (TP distance / SL distance)
    timeframe: str = ""   # timeframe of setup (e.g. "5m")
    horizon_ms: int = 0   # 4h = 4*3600*1000
    strategy_version: str = ""  # deterministic config fingerprint
    generated_at: int = 0  # poll time when signal was generated (ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "signal",
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "signal_time": self.signal_time,
            "entry": self.entry,
            "tp": self.tp,
            "sl": self.sl,
            "score": self.score,
            "rr": self.rr,
            "timeframe": self.timeframe,
            "horizon_ms": self.horizon_ms,
            "strategy_version": self.strategy_version,
            "generated_at": self.generated_at,
        }

    @staticmethod
    def from_signal(signal, strategy_version: str, generated_at_ms: int) -> "PaperSignalEvent":
        """Create a PaperSignalEvent from a P3 Signal."""
        return PaperSignalEvent(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            signal_time=signal.timestamp,
            entry=signal.entry,
            tp=signal.tp,
            sl=signal.sl,
            score=signal.score,
            timeframe=signal.timeframe,
            horizon_ms=signal.horizon_ms,
            strategy_version=strategy_version,
            generated_at=generated_at_ms,
        )


# ---------------------------------------------------------------------------
# Paper Outcome Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperOutcomeEvent:
    """
    Represents a resolution of a paper signal (TP, SL, or expiration).
    """
    event: str = ""
    signal_id: str = ""
    symbol: str = ""
    direction: str = ""
    outcome: str = ""  # "tp_hit", "sl_hit", "expired"
    outcome_time: int = 0  # close_time of the touching/expiring candle (ms)
    entry: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    gross_r: float = 0.0
    net_r: float = 0.0
    holding_ms: int = 0
    fee: float = 0.0
    slippage: float = 0.0
    horizon_ms: int = 0
    strategy_version: str = ""
    close_time: int = 0  # close_time of the relevant candle (for recovery)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "outcome": self.outcome,
            "outcome_time": self.outcome_time,
            "entry": self.entry,
            "tp": self.tp,
            "sl": self.sl,
            "gross_r": self.gross_r,
            "net_r": self.net_r,
            "holding_ms": self.holding_ms,
            "fee": self.fee,
            "slippage": self.slippage,
            "horizon_ms": self.horizon_ms,
            "strategy_version": self.strategy_version,
            "close_time": self.close_time,
        }

    @staticmethod
    def from_evaluation_result(
        result,
        signal_time: int,
        close_time: int,
        strategy_version: str,
    ) -> "PaperOutcomeEvent":
        """Create from a P3 EvaluationResult."""
        outcome_map = {
            "TP": "tp_hit",
            "SL": "sl_hit",
            "EXPIRATION": "expired",
        }
        return PaperOutcomeEvent(
            event=outcome_map.get(result.outcome, "expired"),
            signal_id=result.signal_id,
            symbol=result.symbol,
            direction=result.direction,
            outcome=outcome_map.get(result.outcome, "expired"),
            outcome_time=result.touch_timestamp if result.touch_timestamp is not None else signal_time + result.holding_ms,
            entry=0.0,
            tp=0.0,
            sl=0.0,
            gross_r=result.gross_r if result.gross_r is not None else 0.0,
            net_r=result.net_r if result.net_r is not None else 0.0,
            holding_ms=result.holding_ms if result.holding_ms is not None else 0,
            fee=result.fee,
            slippage=result.slippage,
            horizon_ms=4 * 3600 * 1000,
            strategy_version=strategy_version,
            close_time=close_time,
        )


# ---------------------------------------------------------------------------
# Data Quality Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataQualityEvent:
    """
    Represents a data quality or evaluation status note.
    """
    event: str = "data_quality"
    symbol: str = ""
    interval: str = ""
    poll_time: int = 0
    message: str = ""
    data_available: bool = True
    stale: bool = False
    missing: bool = False
    error_type: str = ""  # "rate_limit", "server_error", "network_error", "malformed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "data_quality",
            "symbol": self.symbol,
            "interval": self.interval,
            "poll_time": self.poll_time,
            "message": self.message,
            "data_available": self.data_available,
            "stale": self.stale,
            "missing": self.missing,
            "error_type": self.error_type,
        }


# ---------------------------------------------------------------------------
# Cycle End Event — persisted per-cycle metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CycleEndEvent:
    """
    One event written at the end of each run_once() cycle.

    JSONL source of truth for runtime status: last completed cycle,
    symbols discovered, symbols processed, signals generated, errors,
    data-quality events, and scan duration.
    """
    event: str = "cycle_end"
    cycle_number: int = 0
    poll_time_ms: int = 0
    symbols_discovered: int = 0
    symbols_processed: int = 0
    signals_generated: int = 0
    outcomes_resolved: int = 0
    data_quality_events: int = 0
    error_symbol_count: int = 0
    scan_duration_ms: int = 0
    min_quote_volume_usdt: float = 50_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "cycle_end",
            "cycle_number": self.cycle_number,
            "poll_time_ms": self.poll_time_ms,
            "symbols_discovered": self.symbols_discovered,
            "symbols_processed": self.symbols_processed,
            "signals_generated": self.signals_generated,
            "outcomes_resolved": self.outcomes_resolved,
            "data_quality_events": self.data_quality_events,
            "error_symbol_count": self.error_symbol_count,
            "scan_duration_ms": self.scan_duration_ms,
            "min_quote_volume_usdt": self.min_quote_volume_usdt,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CycleEndEvent":
        return cls(
            event=d.get("event", "cycle_end"),
            cycle_number=int(d.get("cycle_number", 0)),
            poll_time_ms=int(d.get("poll_time_ms", 0)),
            symbols_discovered=int(d.get("symbols_discovered", 0)),
            symbols_processed=int(d.get("symbols_processed", 0)),
            signals_generated=int(d.get("signals_generated", 0)),
            outcomes_resolved=int(d.get("outcomes_resolved", 0)),
            data_quality_events=int(d.get("data_quality_events", 0)),
            error_symbol_count=int(d.get("error_symbol_count", 0)),
            scan_duration_ms=int(d.get("scan_duration_ms", 0)),
            min_quote_volume_usdt=float(d.get("min_quote_volume_usdt", 50_000_000.0)),
        )


# ---------------------------------------------------------------------------
# Scanner State Event — running/stopped marker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScannerStateEvent:
    """
    Marker event written when the scanner starts and stops.
    Enables 'status' to show running/stopped without external state.
    """
    event: str = "scanner_state"
    state: str = ""          # "running" or "stopped"
    poll_time_ms: int = 0
    log_path: str = ""
    symbols: List[str] = field(default_factory=list)
    interval_sec: int = 0
    max_cycles: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "scanner_state",
            "state": self.state,
            "poll_time_ms": self.poll_time_ms,
            "log_path": self.log_path,
            "symbols": list(self.symbols),
            "interval_sec": self.interval_sec,
            "max_cycles": self.max_cycles,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScannerStateEvent":
        mc = d.get("max_cycles")
        return cls(
            event=d.get("event", "scanner_state"),
            state=d.get("state", ""),
            poll_time_ms=int(d.get("poll_time_ms", 0)),
            log_path=d.get("log_path", ""),
            symbols=[str(s) for s in d.get("symbols", [])],
            interval_sec=int(d.get("interval_sec", 0)),
            max_cycles=int(mc) if mc is not None else None,
        )


# ---------------------------------------------------------------------------
# Deterministic signal_id generation (preserves P1/P2 semantics)
# ---------------------------------------------------------------------------

def make_signal_id(symbol: str, direction: str, signal_time: int) -> str:
    """
    Generate a deterministic signal_id from symbol + direction + signal_time.

    Preserves the existing P1/P2 signal_id mechanism:
        signal_id = hash(symbol + direction + timestamp)

    Uses SHA-256 truncated to 16 hex chars for readability.
    """
    raw = f"{symbol}:{direction}:{signal_time}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return h[:16]


def make_strategy_version() -> str:
    """
    Generate a deterministic strategy version fingerprint.

    Encodes the current strategy parameters:
    - Score weights: Trend 40, Momentum 20, Structure 20, Volume 20
    - TF weights: 1d 25, 4h 25, 1h 20, 15m 20, 5m 10
    - Horizon: 4h
    - Fee/slippage: 0.1% each

    If these parameters ever change, the version changes.
    """
    raw = (
        "trend=40:momentum=20:structure=20:volume=20:"
        "tf_1d=25:tf_4h=25:tf_1h=20:tf_15m=20:tf_5m=10:"
        "horizon=4h:fee=0.1%:slippage=0.1%"
    )
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return h[:16]
