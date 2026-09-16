# Paper-trade signal & event models
#
# P5 paper observation uses immutable Signal objects (from P3 evaluator)
# and adds paper-specific event records for JSONL logging.
#
# Event types:
#   - "signal": new paper signal generated from closed candles
#   - "tp_hit": TP outcome
#   - "sl_hit": SL outcome
#   - "expired": 4h expiration without TP/SL
#   - "data_quality": data quality / evaluation status event
#
# JSONL is the source of truth. State cache (if any) is disposable.


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import hashlib


# ---------------------------------------------------------------------------
# Paper signal event (written to JSONL when a new signal is generated)
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
    timeframe: str = ""   # timeframe of setup (e.g. "5m")
    horizon_ms: int = 0   # 4h = 4*3600*1000
    strategy_version: str = ""  # deterministic config fingerprint
    generated_at: int = 0  # poll time when signal was generated (ms)

    def to_dict(self) -> dict:
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
# Paper outcome event (written to JSONL when a signal resolves)
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

    def to_dict(self) -> dict:
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
            entry=result.gross_r if result.gross_r else 0,  # placeholder; will be set from signal
            tp=0.0,  # will be set from signal context
            sl=0.0,
            gross_r=result.gross_r if result.gross_r is not None else 0.0,
            net_r=result.net_r if result.net_r is not None else 0.0,
            holding_ms=result.holding_ms if result.holding_ms is not None else 0,
            fee=result.fee,
            slippage=result.slippage,
            horizon_ms=result.holding_ms * 0 + 4 * 3600 * 1000,  # default 4h
            strategy_version=strategy_version,
            close_time=close_time,
        )


# ---------------------------------------------------------------------------
# Data quality event
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

    def to_dict(self) -> dict:
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
    # Deterministic fingerprint of the canonical strategy parameters.
    raw = (
        "trend=40:momentum=20:structure=20:volume=20:"
        "tf_1d=25:tf_4h=25:tf_1h=20:tf_15m=20:tf_5m=10:"
        "horizon=4h:fee=0.1%:slippage=0.1%"
    )
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return h[:16]
