"""Pure Python market structure detection — strict 3-candle swing fractal.

Uses only closed candles. No new strategy parameters.
D5 structure scoring added: swing classification (HH/LH/HL/LL) and
deterministic 100/75/50/25/0 per-timeframe direction-relative score.
"""

from __future__ import annotations
from typing import List, Optional, Tuple


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


def _latest_swing_relation(
    swing_indices: List[int],
    candles: List[dict],
    key: str,  # 'high' for swing highs, 'low' for swing lows
    direction_up_is_bullish: bool,  # True for highs (higher = bullish), False for lows (higher = bearish)
) -> Optional[str]:
    """Classify the latest swing relation as HH/LH (for highs) or HL/LL (for lows).

    Returns 'bullish', 'bearish', or None if fewer than 2 swings or equality.
    For swing highs: latest > previous = bullish (HH), latest < previous = bearish (LH).
    For swing lows:  latest < previous = bullish (LL), latest > previous = bearish (HL).
    Equality returns None (no relation created).
    """
    if len(swing_indices) < 2:
        return None
    latest_idx = swing_indices[-1]
    prev_idx = swing_indices[-2]
    latest_val = candles[latest_idx][key]
    prev_val = candles[prev_idx][key]
    if latest_val == prev_val:
        return None
    if direction_up_is_bullish:
        return 'bullish' if latest_val > prev_val else 'bearish'
    else:
        return 'bullish' if latest_val < prev_val else 'bearish'


def structure_score(
    candles: List[dict],
    *,
    lookback: int = 20,
    direction: str = 'LONG',
) -> int:
    """D5 Structure score (0–100) per timeframe, direction-relative.

    Uses the latest `lookback` closed candles, excluding the last candle
    (C_tf) from swing detection.  Classifies the latest swing-high and
    swing-low relations, then maps to the approved 100/75/50/25/0 table.

    Score mapping (LONG; SHORT is the exact directional mirror):
        HH + HL        -> 100 (strong bullish)   / 0 (strong bearish for SHORT)
        one bullish, other unavailable -> 75 (clear bullish) / 25
        HH+LL or LH+HL -> 50 (mixed/neutral)    / 50
        one bearish, other unavailable -> 25 (clear bearish) / 75
        LH + LL        -> 0  (strong bearish)   / 100 (strong bullish for SHORT)
        neither relation available      -> 50 (mixed/neutral) / 50

    Args:
        candles: list of dicts with 'high'/'low', sorted ascending by close_time.
        lookback: number of most recent closed candles to consider (default 20).
        direction: 'LONG' or 'SHORT'.

    Returns:
        Integer score in 0–100.
    """
    if not candles or lookback < 2:
        return 50

    window = candles[-lookback:]
    # Exclude the last candle (C_tf) from swing detection per spec.
    swings_window = window[:-1]
    if len(swings_window) < 3:
        return 50

    highs, lows = detect_swings(swings_window)
    # swings_window is candles[start : start+len(swings_window)] where
    # start = max(0, len(candles) - lookback).  Map swing indices back.
    start = max(0, len(candles) - lookback)

    high_relation = _latest_swing_relation(
        [i + start for i in highs], candles, 'high', True)
    low_relation = _latest_swing_relation(
        [i + start for i in lows], candles, 'low', True)

    def _score(high_rel: Optional[str], low_rel: Optional[str],
               dir_: str) -> int:
        if dir_ == 'LONG':
            if high_rel == 'bullish' and low_rel == 'bullish':
                return 100
            if (high_rel == 'bullish' and low_rel is None) or \
               (low_rel == 'bullish' and high_rel is None):
                return 75
            if high_rel == 'bearish' and low_rel == 'bearish':
                return 0
            if (high_rel == 'bearish' and low_rel is None) or \
               (low_rel == 'bearish' and high_rel is None):
                return 25
            return 50
        else:  # SHORT — exact mirror
            if high_rel == 'bearish' and low_rel == 'bearish':
                return 100
            if (high_rel == 'bearish' and low_rel is None) or \
               (low_rel == 'bearish' and high_rel is None):
                return 75
            if high_rel == 'bullish' and low_rel == 'bullish':
                return 0
            if (high_rel == 'bullish' and low_rel is None) or \
               (low_rel == 'bullish' and high_rel is None):
                return 25
            return 50

    return _score(high_relation, low_relation, direction)


def latest_structure_sr(
    candles: List[dict],
    *,
    lookback: int = 20,
) -> dict:
    """Return the 2 most recent swing highs and 2 most recent swing lows
    from the latest `lookback` closed candles, excluding C_tf from swing
    detection.  Pure S/R levels for later SL/TP use.

    Returns dict with:
        'swing_highs': list of dicts (most recent first), max length 2
        'swing_lows':  list of dicts (most recent first), max length 2
        'high_relation': Optional[str] — latest HH/LH classification or None
        'low_relation':  Optional[str] — latest HL/LL classification or None
    """
    if not candles or lookback < 2:
        return {'swing_highs': [], 'swing_lows': [],
                'high_relation': None, 'low_relation': None}

    window = candles[-lookback:]
    swings_window = window[:-1]
    if len(swings_window) < 3:
        return {'swing_highs': [], 'swing_lows': [],
                'high_relation': None, 'low_relation': None}

    highs, lows = detect_swings(swings_window)
    start = max(0, len(candles) - lookback)

    raw_highs = [candles[i + start] for i in highs]
    raw_lows = [candles[i + start] for i in lows]

    high_relation = _latest_swing_relation(
        [i + start for i in highs], candles, 'high', True)
    low_relation = _latest_swing_relation(
        [i + start for i in lows], candles, 'low', True)

    return {
        'swing_highs': raw_highs[-2:][::-1] if raw_highs else [],
        'swing_lows': raw_lows[-2:][::-1] if raw_lows else [],
        'high_relation': high_relation,
        'low_relation': low_relation,
    }
