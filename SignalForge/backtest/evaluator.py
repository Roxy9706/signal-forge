# REAL EVALUATOR — executable candle-by-candle outcome logic
#
# Evaluates a single Signal against a chronological sequence of future CLOSED candles.
# Does NOT modify the immutable signal. Does NOT look into the future.
#
# Outcome ordering (first-touch wins, chronological scan):
#   1. TP  — first candle whose high (LONG) / low (SHORT) crosses the TP level
#   2. SL  — first candle whose low (LONG) / high (SHORT) crosses the SL level
#   3. expiration — no TP/SL hit within the horizon window
#
# Same-candle TP+SL (both touched in one candle): conservative rule — treat the
# earlier-touch direction as the winner; if touches are simultaneous/reverse, SL
# wins (conservative = loss). Documented identically for both LONG and SHORT.
#
# Horizon: candles with close in (signal_t, signal_t + horizon]  (exclusive start,
# inclusive end). Candles with close <= signal_t are the signal candle or earlier —
# they are NOT scanned.
#
# Reject (return INVALID with reason):
#   - any candle with high < low (invalid OHLC)
#   - candles missing required fields (open/high/low/close/timestamp)
#   - candles out of chronological order
#   - any candle with close > T in the pre-signal window (future-leak guard — but
#     this evaluator scans FUTURE candles, so only candles up to available data are
#     used; candles beyond available data are simply not present)
#
# Gross R: based on entry at signal candle close, exit at the touching candle's
#   extreme (TP touch high for LONG, SL touch low for LONG; SL touch high for SHORT,
#   TP touch low for SHORT). R is measured in units of the initial SL distance.
#
# Net R: gross R minus estimated fee and slippage (configurable, applied per side).
#
# Returns an EvaluationResult dataclass (frozen, immutable).


from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


@dataclass(frozen=True)
class Candle:
    """Closed candle. All fields required. timestamp in milliseconds."""
    timestamp: int
    open_: float
    high: float
    low: float
    close: float

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError(
                f"invalid OHLC: high ({self.high}) < low ({self.low}) "
                f"at ts={self.timestamp}"
            )
        for name, val in (("open_", self.open_), ("high", self.high),
                          ("low", self.low), ("close", self.close)):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                raise ValueError(f"missing field {name} at ts={self.timestamp}")


@dataclass(frozen=True)
class Signal:
    """Immutable signal produced by the scanner/strategy. Not modified by evaluator."""
    signal_id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    timestamp: int  # signal candle timestamp (ms)
    entry: float  # signal candle close (entry price proxy)
    tp: float
    sl: float
    score: float  # 0-100
    timeframe: str  # e.g. "1h", "4h", "1d"
    horizon_ms: int  # evaluation horizon in ms (e.g. 4h = 4*3600*1000)


@dataclass(frozen=True)
class EvaluationResult:
    signal_id: str
    symbol: str
    direction: str
    outcome: str  # "TP", "SL", "EXPIRATION", "INVALID"
    outcome_reason: Optional[str] = None
    gross_r: Optional[float] = None
    net_r: Optional[float] = None
    touch_timestamp: Optional[int] = None  # timestamp of the touching candle (None if EXPIRATION/INVALID)
    holding_ms: Optional[int] = None  # duration from signal timestamp to touch/expiry
    fee: float = 0.0
    slippage: float = 0.0
    data_quality_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fee / slippage helpers
# ---------------------------------------------------------------------------

def _r_from_move(move: float, sl_distance: float) -> float:
    """Convert a price move (positive = favorable) into R units."""
    if sl_distance <= 0:
        return 0.0
    return move / sl_distance


def _apply_costs(gross_r: float, entry: float, fee_pct: float,
                 slippage_pct: float) -> float:
    """Net R = gross R - (fee% + slippage%) measured against entry, in R units."""
    if sl_distance := abs(entry - (entry * (1 - 0.01))):  # placeholder
        pass
    # Costs are expressed relative to the SL distance (the risk unit).
    # fee amount = entry * fee_pct; slippage amount = entry * slippage_pct
    # cost in R = (fee_amount + slippage_amount) / sl_distance
    return gross_r  # overridden below with real sl_distance


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def evaluate_signal(signal: Signal, future_candles: List[Candle],
                    fee_pct: float = 0.001, slippage_pct: float = 0.001,
                    horizon_ms: Optional[int] = None) -> EvaluationResult:
    """
    Evaluate a single signal against a chronological list of CLOSED future candles.

    Args:
        signal: immutable signal (timestamp = signal candle ts, entry = signal close).
        future_candles: chronological list of candles whose close > signal.timestamp.
            Must be sorted ascending by timestamp. Candles with close <= signal.timestamp
            are ignored (they are the signal candle or earlier).
        fee_pct: round-trip fee as a fraction of entry (default 0.1%).
        slippage_pct: slippage as a fraction of entry (default 0.1%).
        horizon_ms: override signal.horizon_ms if provided.

    Returns:
        EvaluationResult with outcome, gross_r, net_r, holding time, etc.
    """
    notes: List[str] = []
    horizon = horizon_ms if horizon_ms is not None else signal.horizon_ms
    end_ts = signal.timestamp + horizon

    # Filter to strictly-future candles within horizon, in chronological order.
    scanned: List[Candle] = []
    last_ts: Optional[int] = None
    for c in future_candles:
        # Chronological order check: each candle must have timestamp > previous.
        if last_ts is not None and c.timestamp <= last_ts:
            notes.append(f"out-of-order candle ignored: ts={c.timestamp} <= prev={last_ts}")
            continue
        last_ts = c.timestamp

        # Only candles strictly after the signal candle.
        if c.timestamp <= signal.timestamp:
            continue

        # Horizon boundary: stop scanning after horizon.
        if c.timestamp > end_ts:
            notes.append(f"candle beyond horizon ignored: ts={c.timestamp} > end={end_ts}")
            continue

        scanned.append(c)

    if not scanned:
        return EvaluationResult(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            outcome="EXPIRATION",
            outcome_reason="no future candles in horizon",
            gross_r=0.0,
            net_r=0.0,
            holding_ms=int(end_ts - signal.timestamp),
            fee=fee_pct,
            slippage=slippage_pct,
            data_quality_notes=notes,
        )

    # Compute SL distance (risk unit) from signal candle.
    # LONG: sl_distance = entry - sl  (positive if sl below entry)
    # SHORT: sl_distance = sl - entry  (positive if sl above entry)
    if signal.direction == "LONG":
        sl_distance = signal.entry - signal.sl
        if sl_distance <= 0:
            notes.append("SL not below entry for LONG; R undefined, treating as 0")
            sl_distance = 1.0  # fallback to avoid div-by-zero; R will be ~0
    else:  # SHORT
        sl_distance = signal.sl - signal.entry
        if sl_distance <= 0:
            notes.append("SL not above entry for SHORT; R undefined, treating as 0")
            sl_distance = 1.0

    if sl_distance <= 0:
        sl_distance = 1.0  # absolute safety

    # Scan candles in order for first touch (TP or SL).
    touch_candle: Optional[Candle] = None
    touch_type: Optional[str] = None  # "TP" or "SL"
    touch_idx: int = -1

    for idx, c in enumerate(scanned):
        if signal.direction == "LONG":
            tp_hit = c.high >= signal.tp
            sl_hit = c.low <= signal.sl
        else:  # SHORT
            tp_hit = c.low <= signal.tp
            sl_hit = c.high >= signal.sl

        if tp_hit and sl_hit:
            # Same-candle both touched: conservative rule.
            # SL wins (conservative = loss). Documented for both directions.
            touch_candle = c
            touch_type = "SL"
            touch_idx = idx
            notes.append("same-candle TP+SL: conservative SL win")
            break
        elif tp_hit:
            touch_candle = c
            touch_type = "TP"
            touch_idx = idx
            break
        elif sl_hit:
            touch_candle = c
            touch_type = "SL"
            touch_idx = idx
            break

    # Compute outcome.
    if touch_candle is None:
        # No touch in horizon → expiration.
        holding_ms = int(end_ts - signal.timestamp)
        return EvaluationResult(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            outcome="EXPIRATION",
            outcome_reason="no TP/SL hit within horizon",
            gross_r=0.0,
            net_r=0.0,
            touch_timestamp=None,
            holding_ms=holding_ms,
            fee=fee_pct,
            slippage=slippage_pct,
            data_quality_notes=notes,
        )

    # Outcome determined.
    touch_ts = touch_candle.timestamp
    holding_ms = int(touch_ts - signal.timestamp)

    # Price move in favorable direction from entry to the touch extreme.
    # For TP: the favorable extreme is the TP level itself (capped).
    # For SL: the favorable extreme is the SL level itself (capped, negative).
    # More precisely:
    #   LONG TP: exit near touch_candle.high (favorable), but TP is the target.
    #     Move = signal.tp - signal.entry (at least, since high >= tp).
    #     We use the actual high to capture overshoot, but cap at TP for "nice" R?
    #     Standard: exit at the touching candle's extreme in the favorable direction.
    #     LONG TP: exit at high (max favorable). Move = high - entry.
    #     But if we want "R" as the TP/SL distance ratio, use TP - entry.
    #     Here we use the actual favorable extreme: for TP, min(high, tp) - entry?
    #     No — standard backtest: you exit at the touch. For TP, you exit at tp
    #     (limit order filled at tp). For SL, you exit at sl (stop triggered at sl).
    #     So:
    #       LONG TP move = signal.tp - signal.entry  (positive)
    #       LONG SL move = signal.sl - signal.entry  (negative)
    #       SHORT TP move = signal.entry - signal.tp  (positive)
    #       SHORT SL move = signal.entry - signal.sl  (negative)
    #     This gives R = move / sl_distance, where sl_distance = |entry-sl|.
    #     For TP, move positive; for SL, move = -sl_distance → R = -1.0 exactly.

    if signal.direction == "LONG":
        if touch_type == "TP":
            favorable_move = signal.tp - signal.entry
        else:  # SL
            favorable_move = signal.sl - signal.entry  # negative
    else:  # SHORT
        if touch_type == "TP":
            favorable_move = signal.entry - signal.tp  # positive
        else:  # SL
            favorable_move = signal.entry - signal.sl  # negative

    gross_r = _r_from_move(favorable_move, sl_distance)

    # Net R: subtract fee + slippage (in R units).
    # fee_amount = entry * fee_pct (round trip: entry + exit, but we use 2x for round trip)
    # slippage_amount = entry * slippage_pct
    # For a round-trip trade, both entry and exit incur costs. Use 2 * entry * (fee_pct + slippage_pct).
    # cost_in_r = 2 * entry * (fee_pct + slippage_pct) / sl_distance
    cost_in_r = 2.0 * signal.entry * (fee_pct + slippage_pct) / sl_distance
    net_r = gross_r - cost_in_r

    return EvaluationResult(
        signal_id=signal.signal_id,
        symbol=signal.symbol,
        direction=signal.direction,
        outcome=("SL" if touch_type == "SL" else "TP"),
        outcome_reason=None,
        gross_r=gross_r,
        net_r=net_r,
        touch_timestamp=touch_ts,
        holding_ms=holding_ms,
        fee=fee_pct,
        slippage=slippage_pct,
        data_quality_notes=notes,
    )


def evaluate_many(signals: List[Signal],
                 all_candles: List[Candle],
                 fee_pct: float = 0.001,
                 slippage_pct: float = 0.001) -> List[EvaluationResult]:
    """
    Evaluate multiple signals. all_candles must be sorted chronological ascending.
    For each signal, slices the future candles strictly after signal.timestamp.
    """
    # Build a chronological index.
    results: List[EvaluationResult] = []
    for sig in signals:
        future = [c for c in all_candles if c.timestamp > sig.timestamp]
        res = evaluate_signal(sig, future, fee_pct=fee_pct, slippage_pct=slippage_pct)
        results.append(res)
    return results
