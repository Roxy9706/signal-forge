"""Pure Python market structure detection — strict 3-candle swing fractal.

Uses only closed candles. No new strategy parameters.
"""

from __future__ import annotations
from typing import List, Optional


def detect_swings(candles: List[dict]) -> tuple:
    """Detect swing highs and swing lows using strict 3-candle fractal.

    A swing high at index i: high[i-1] < high[i] > high[i+1]
    A swing low  at index i: low[i-1] > low[i] < low[i+1]

    Candles must be sorted oldest-to-newest by close_time.
    Index 0 and index -1 can NEVER be swings (insufficient neighbors).

    Args:
        candles: list of dicts with 'high' and 'low' keys, sorted ascending by close_time.

    Returns:
        (swing_high_indices, swing_low_indices) — both lists of int indices.
    """
    if len(candles) < 3:
        return [], []

    swing_highs: List[int] = []
    swing_lows: List[int] = []

    for i in range(1, len(candles) - 1):
        prev_high = candles[i - 1]['high']
        curr_high = candles[i]['high']
        next_high = candles[i + 1]['high']

        prev_low = candles[i - 1]['low']
        curr_low = candles[i]['low']
        next_low = candles[i + 1]['low']

        if prev_high < curr_high > next_high:
            swing_highs.append(i)
        if prev_low > curr_low < next_low:
            swing_lows.append(i)

    return swing_highs, swing_lows


def most_recent_swing_high(candles: List[dict], exclude_index: Optional[int] = None) -> Optional[dict]:
    """Return the most recent swing high candle, excluding an index if given.

    Args:
        candles: list of dicts with 'high'/'low', sorted ascending by close_time.
        exclude_index: if set, skip this index when searching.

    Returns:
        The dict representing the candle with the most recent swing high,
        or None if no swing high exists.
    """
    if not candles:
        return None
    highs, _ = detect_swings(candles)
    candidates = [idx for idx in highs if idx != exclude_index]
    if not candidates:
        return None
    best_idx = candidates[-1]  # most recent
    return candles[best_idx]


def most_recent_swing_low(candles: List[dict], exclude_index: Optional[int] = None) -> Optional[dict]:
    """Return the most recent swing low candle, excluding an index if given.

    Args:
        candles: list of dicts with 'high'/'low', sorted ascending by close_time.
        exclude_index: if set, skip this index when searching.

    Returns:
        The dict representing the candle with the most recent swing low,
        or None if no swing low exists.
    """
    if not candles:
        return None
    _, lows = detect_swings(candles)
    candidates = [idx for idx in lows if idx != exclude_index]
    if not candidates:
        return None
    best_idx = candidates[-1]  # most recent
    return candles[best_idx]
