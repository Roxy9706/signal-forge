"""Pure Python RSI (Wilder's original formulation).

RSI period = 14. Wilder's RMA smoothing. No new strategy parameters.
Uses only closed candles. Deterministic.
"""

from __future__ import annotations
from typing import List, Optional


def compute_rsi(candles: List[dict], period: int = 14) -> Optional[float]:
    """Compute Wilder's RSI through the last candle in the list.

    Args:
        candles: list of dicts with 'close' key, sorted oldest to newest.
        period: RSI period (default 14).

    Returns:
        RSI value as float in [0, 100], or None if not enough data.
    """
    if not candles or period < 2:
        return None
    closes = [c['close'] for c in candles]
    if len(closes) < period + 1:
        return None

    # Compute gains and losses
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        elif diff < 0:
            gains.append(0.0)
            losses.append(-diff)
        else:
            gains.append(0.0)
            losses.append(0.0)

    # Need at least 'period' gains/losses (first gain/loss at index 1)
    if len(gains) < period:
        return None

    # Seed: SMA of first 'period' gains and losses
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's recurrence for the remaining periods
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Compute RSI
    if avg_loss == 0.0 and avg_gain > 0.0:
        return 100.0
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        # avg_gain == 0.0 already handled; this branch unreachable but safe
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def momentum_score(rsi: Optional[float], direction: str) -> int:
    """Map RSI to the approved 0-100 Momentum score.

    LONG scoring (RSI above 50 supports momentum):
        RSI >= 70       -> 100
        60 <= RSI < 70  -> 80
        50 <= RSI < 60  -> 65
        40 <= RSI < 50  -> 35
        30 <= RSI < 40  -> 20
        RSI < 30        -> 0

    SHORT scoring (RSI below 50 supports momentum, mirrored):
        RSI <= 30       -> 100
        30 < RSI <= 40  -> 80
        40 < RSI <= 50  -> 65
        50 < RSI <= 60  -> 35
        60 < RSI <= 70  -> 20
        RSI > 70        -> 0

    Args:
        rsi: RSI value as float, or None.
        direction: 'LONG' or 'SHORT'.

    Returns:
        Integer score in 0-100.
    """
    if rsi is None:
        return 0
    if direction == 'LONG':
        if rsi >= 70.0:
            return 100
        if rsi >= 60.0:
            return 80
        if rsi >= 50.0:
            return 65
        if rsi >= 40.0:
            return 35
        if rsi >= 30.0:
            return 20
        return 0
    elif direction == 'SHORT':
        if rsi <= 30.0:
            return 100
        if rsi <= 40.0:
            return 80
        if rsi <= 50.0:
            return 65
        if rsi <= 60.0:
            return 35
        if rsi <= 70.0:
            return 20
        return 0
    else:
        return 0
