# P5 Paper Trade Models
#
# Immutable dataclasses for paper observation events.
# P5 is observation-only — NO orders, NO API keys, NO private endpoints,
# NO leverage, NO Telegram/webhooks, NO auto-trading.
#
# All events are appended to a JSONL log which is the source of truth.
# State cache (if any) is disposable/rebuildable from JSONL.


from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
import hashlib


# ---------------------------------------------------------------------------
# Canonical signal ID generation (P1/P2 deterministic mechanism preserved)
# ---------------------------------------------------------------------------

def make_signal_id(symbol: str, direction: str, signal_time_ms: int) -> str:
    """
    Generate deterministic signal_id from (symbol, direction, signal_time).

    Preserves existing P1/P2 signal_id mechanism — same inputs always
    produce the same ID. This prevents duplicates across restarts.
    """
    raw = f"{symbol}:{direction}:{signal_time_ms}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return h[:16]


def make_strategy_version() -> str:
    """
    Deterministic strategy version fingerprint.

    Encodes the frozen strategy parameters:
    - Weights: Trend 40, Momentum 20, Structure 20, Volume 20
    - TF weights: 1d 25, 4h 25, 1h 20, 15m 20, 5m 10
    - Horizon: 4h
    - No threshold optimization, no ML, no parameter changes.

    If parameters ever change, this version changes — making old/new
    signals distinguishable.
    """
    raw = (
        "trend=40:momentum=20:structure=20:volume=20:"
        "tf_1d=25:tf_4h=25:tf_1h=20:tf_15m=20:tf_5m=10:"
        "horizon=4h:fee_default=0.1%:slippage_default=0.1%"
    )
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return h[:16]


# ---------------------------------------------------------------------------
# Paper Signal Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperSignal:
    """
    A paper signal generated from closed candles.

    signal_time = close_time of the setup candle (e.g. 5m if 5m-triggered).
    Only data available at signal_time is used (closed-candle rule).
    """
    event_type: str = "signal"
    signal_id: str = ""
    symbol: str = ""
    direction: str = ""  # "LONG" or "SHORT"
    signal_time: int = 0  # close_time of setup candle (ms)
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    score: float = 0.0  # 0-100
    timeframe: str = ""  # e.g. "5m", "15m", "1h", "4h", "1d"
    horizon_ms: int = 4 * 3600 * 1000  # 4h default
    strategy_version: str = ""
    generated_at: int = 0  # poll time when signal was generated (ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "signal",
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "signal_time": self.signal_time,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "score": self.score,
            "timeframe": self.timeframe,
            "horizon_ms": self.horizon_ms,
            "strategy_version": self.strategy_version,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperSignal":
        return cls(
            event_type=d.get("event", "signal"),
            signal_id=d.get("signal_id", ""),
            symbol=d.get("symbol", ""),
            direction=d.get("direction", ""),
            signal_time=int(d.get("signal_time", 0)),
            entry=float(d.get("entry", 0.0)),
            sl=float(d.get("sl", 0.0)),
            tp=float(d.get("tp", 0.0)),
            score=float(d.get("score", 0.0)),
            timeframe=d.get("timeframe", ""),
            horizon_ms=int(d.get("horizon_ms", 4 * 3600 * 1000)),
            strategy_version=d.get("strategy_version", ""),
            generated_at=int(d.get("generated_at", 0)),
        )


# ---------------------------------------------------------------------------
# Paper Outcome Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperOutcome:
    """
    Outcome of a paper signal — TP hit, SL hit, or expired.

    outcome_time = close_time of the candle that triggered the outcome
    (or horizon end for expiration).
    """
    event_type: str = "outcome"
    signal_id: str = ""
    symbol: str = ""
    direction: str = ""
    outcome: str = ""  # "tp_hit", "sl_hit", "expired"
    outcome_time: int = 0  # close_time of outcome candle (ms)
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    gross_r: float = 0.0
    net_r: float = 0.0
    holding_ms: int = 0
    fee_pct: float = 0.001
    slippage_pct: float = 0.001
    horizon_ms: int = 4 * 3600 * 1000
    strategy_version: str = ""
    close_time: int = 0  # latest closed candle time at evaluation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.outcome,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "outcome": self.outcome,
            "outcome_time": self.outcome_time,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "gross_r": self.gross_r,
            "net_r": self.net_r,
            "holding_ms": self.holding_ms,
            "fee_pct": self.fee_pct,
            "slippage_pct": self.slippage_pct,
            "horizon_ms": self.horizon_ms,
            "strategy_version": self.strategy_version,
            "close_time": self.close_time,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperOutcome":
        return cls(
            event_type=d.get("event", "outcome"),
            signal_id=d.get("signal_id", ""),
            symbol=d.get("symbol", ""),
            direction=d.get("direction", ""),
            outcome=d.get("outcome", ""),
            outcome_time=int(d.get("outcome_time", 0)),
            entry=float(d.get("entry", 0.0)),
            sl=float(d.get("sl", 0.0)),
            tp=float(d.get("tp", 0.0)),
            gross_r=float(d.get("gross_r", 0.0)),
            net_r=float(d.get("net_r", 0.0)),
            holding_ms=int(d.get("holding_ms", 0)),
            fee_pct=float(d.get("fee_pct", 0.001)),
            slippage_pct=float(d.get("slippage_pct", 0.001)),
            horizon_ms=int(d.get("horizon_ms", 4 * 3600 * 1000)),
            strategy_version=d.get("strategy_version", ""),
            close_time=int(d.get("close_time", 0)),
        )


# ---------------------------------------------------------------------------
# Data Quality / Evaluation Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataQualityEvent:
    """
    Non-signal event for logging data quality issues, evaluation status, etc.

    Used for: stale data, API errors, missing candles, evaluation notes.
    """
    event_type: str = "data_quality"
    symbol: str = ""
    message: str = ""
    poll_time: int = 0
    data_available: bool = True
    error_type: str = ""  # "rate_limit", "server_error", "network_error", "malformed"
    stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": "data_quality",
            "symbol": self.symbol,
            "message": self.message,
            "poll_time": self.poll_time,
            "data_available": self.data_available,
            "error_type": self.error_type,
            "stale": self.stale,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataQualityEvent":
        return cls(
            event_type=d.get("event", "data_quality"),
            symbol=d.get("symbol", ""),
            message=d.get("message", ""),
            poll_time=int(d.get("poll_time", 0)),
            data_available=bool(d.get("data_available", True)),
            error_type=d.get("error_type", ""),
            stale=bool(d.get("stale", False)),
        )


# ---------------------------------------------------------------------------
# All paper events union for JSONL handling
# ---------------------------------------------------------------------------

PaperEvent = PaperSignal | PaperOutcome | DataQualityEvent


def event_to_dict(event: PaperEvent) -> Dict[str, Any]:
    """Serialize any paper event to a dict for JSONL."""
    if isinstance(event, PaperSignal):
        return event.to_dict()
    elif isinstance(event, PaperOutcome):
        return event.to_dict()
    else:
        return event.to_dict()


def event_from_dict(d: Dict[str, Any]) -> PaperEvent:
    """Deserialize a JSONL dict back to the appropriate event type."""
    event_type = d.get("event", "")
    if event_type in ("signal",):
        return PaperSignal.from_dict(d)
    elif event_type in ("tp_hit", "sl_hit", "expired"):
        return PaperOutcome.from_dict(d)
    elif event_type == "data_quality":
        return DataQualityEvent.from_dict(d)
    else:
        # Unknown event type — return as generic dict-backed object
        return PaperSignal.from_dict(d)  # fallback
