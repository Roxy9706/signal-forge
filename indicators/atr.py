"""Pure Python ATR (Wilder's True Range / Wilder smoothing).

ATR period = 14. Uses True Range per Wilder's original formulation.
No new strategy parameters. Uses only closed candles. Deterministic.
"""

from __future__ import annotations
from typing import List, Optional


def compute_atr(candles: List[dict], period: int = 14) -> Optional[float]:
    """Compute Wilder's ATR through the last candle in the list.

    True Range for candle i (i >= 1):
        TR_i = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    ATR calculation:
        - Seed: simple average of the first `period` TR values.
        - Recurrence: ATR_t = (ATR_previous * (period - 1) + TR_t) / period

    Minimum data required:
        - At least `period + 1` candles to compute `period` TR values for the seed.
        - Previous close is required for each TR (candle i needs close[i-1]).
          The first candle (index 0) cannot produce a TR value.

    Args:
        candles: list of dicts with 'high', 'low', 'close' keys, sorted oldest to newest.
        period: ATR period (default 14).

    Returns:
        ATR value as float, or None if insufficient data.
    """
    if not candles or period < 2:
        return None
    if len(candles) < period + 1:
        return None

    # Compute True Range for each candle starting from index 1
    tr_values: List[float] = []
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i - 1]['close']

        tr_range = high - low
        tr_gap_up = abs(high - prev_close)
        tr_gap_down = abs(low - prev_close)

        tr = max(tr_range, tr_gap_up, tr_gap_down)
        tr_values.append(tr)

    # Need at least 'period' TR values for the seed
    if len(tr_values) < period:
        return None

    # Seed: simple average of first 'period' TR values
    atr = sum(tr_values[:period]) / period

    # Wilder's recurrence for remaining TR values
    for i in range(period, len(tr_values)):
        atr = (atr * (period - 1) + tr_values[i]) / period

    return atr
