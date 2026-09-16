"""Pure Python EMA — exponential moving average."""

from __future__ import annotations
from typing import List, Optional


def compute_ema(candles: List[dict], period: int) -> Optional[float]:
    """Compute EMA(period) through the last candle in the list.

    Args:
        candles: list of dicts with 'close' key, sorted oldest to newest.
        period: EMA period (e.g. 20).

    Returns:
        EMA value as float, or None if not enough data.
    """
    if not candles or period < 2:
        return None
    closes = [c['close'] for c in candles]
    if len(closes) < period:
        return None
    # Seed with SMA of first 'period' closes
    sma = sum(closes[:period]) / period
    k = 2.0 / (period + 1.0)
    ema = sma
    for price in closes[period:]:
        ema = price * k + ema * (1.0 - k)
    return ema
