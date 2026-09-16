"""Pullback / retest confirmation — D7 strategy gate.

Deterministic, pure functions. No new strategy parameters.
Uses only closed candles at or before setup candle T.
5m is the trigger timeframe.

Imports indicators.ema and indicators.structure.
"""

from __future__ import annotations
from typing import List, Optional

from indicators.ema import compute_ema
from indicators.structure import most_recent_swing_high, most_recent_swing_low


def _select_retest_level_by_direction(
    candles_5m: List[dict],
    setup_index: int,
    direction: str,
) -> Optional[float]:
    """Select retest level L per approved precedence.

    Precedence:
      1. EMA20 on 5m computed through setup candle C.
      2. Otherwise most recent swing low (LONG) / swing high (SHORT)
         from the 20-candle structure window ending at C, excluding C.
      3. Otherwise None.

    Args:
        candles_5m: closed 5m candles sorted ascending by close_time.
        setup_index: index of setup candle C.
        direction: 'LONG' or 'SHORT'.

    Returns:
        Retest level L as float, or None.
    """
    if setup_index < 0 or setup_index >= len(candles_5m):
        return None

    series_through_c = candles_5m[:setup_index + 1]
    ema20 = compute_ema(series_through_c, 20)
    if ema20 is not None:
        return ema20

    start_idx = max(0, setup_index - 20)
    window = candles_5m[start_idx:setup_index]
    if not window:
        return None

    if direction == 'LONG':
        swing = most_recent_swing_low(window, exclude_index=None)
        return swing['low'] if swing is not None else None
    elif direction == 'SHORT':
        swing = most_recent_swing_high(window, exclude_index=None)
        return swing['high'] if swing is not None else None
    else:
        return None


def find_prior_trend_side_candle(
    candles_5m: List[dict],
    setup_index: int,
    L: float,
    direction: str,
) -> Optional[dict]:
    """Find most recent candle in W (20 candles before C) with close on trend side of L.

    W = 20 immediately preceding closed 5m candles before C.
    If fewer exist, use available. If none, return None.
    C itself (at setup_index) is explicitly excluded.

    For LONG: close > L
    For SHORT: close < L

    Args:
        candles_5m: closed 5m candles sorted ascending by close_time.
        setup_index: index of setup candle C.
        L: retest level.
        direction: 'LONG' or 'SHORT'.

    Returns:
        The dict of the most recent prior trend-side candle, or None.
    """
    if setup_index <= 0 or L is None:
        return None

    start_idx = max(0, setup_index - 20)
    end_idx = setup_index
    window = candles_5m[start_idx:end_idx]

    if direction == 'LONG':
        for i in range(len(window) - 1, -1, -1):
            if window[i]['close'] > L:
                return window[i]
    elif direction == 'SHORT':
        for i in range(len(window) - 1, -1, -1):
            if window[i]['close'] < L:
                return window[i]

    return None


def validate_pullback_long(
    candles_5m: List[dict],
    setup_index: int,
    L: Optional[float],
) -> bool:
    """Validate D7 pullback/retest for LONG.

    Requires:
      - L exists
      - pullback exists: most recent candle in W with close > L
      - rejection confirmed:
          tested:       low(C) <= L AND high(C) >= L
          closed_back:  close(C) > L   (STRICT — equality invalid)
          body_bullish: close(C) > open(C)
          no_break_below: low(C) >= L   (opposing wick may TOUCH L, not break)

    Args:
        candles_5m: closed 5m candles sorted ascending by close_time.
        setup_index: index of setup candle C.
        L: retest level, or None.

    Returns:
        True if D7 passes for LONG, False otherwise.
    """
    if L is None:
        return False
    if setup_index < 0 or setup_index >= len(candles_5m):
        return False

    C = candles_5m[setup_index]

    P = find_prior_trend_side_candle(candles_5m, setup_index, L, 'LONG')
    pullback = P is not None
    if not pullback:
        return False

    tested = (C['low'] <= L) and (C['high'] >= L)
    closed_back = C['close'] > L
    body_bullish = C['close'] > C['open']
    no_break_below = C['low'] >= L

    rejection = tested and closed_back and body_bullish and no_break_below
    return rejection


def validate_pullback_short(
    candles_5m: List[dict],
    setup_index: int,
    L: Optional[float],
) -> bool:
    """Validate D7 pullback/retest for SHORT.

    Requires:
      - L exists
      - pullback exists: most recent candle in W with close < L
      - rejection confirmed:
          tested:       low(C) <= L AND high(C) >= L
          closed_back:  close(C) < L   (STRICT — equality invalid)
          body_bearish: close(C) < open(C)
          no_break_above: high(C) <= L  (opposing wick may TOUCH L, not break)

    Args:
        candles_5m: closed 5m candles sorted ascending by close_time.
        setup_index: index of setup candle C.
        L: retest level, or None.

    Returns:
        True if D7 passes for SHORT, False otherwise.
    """
    if L is None:
        return False
    if setup_index < 0 or setup_index >= len(candles_5m):
        return False

    C = candles_5m[setup_index]

    P = find_prior_trend_side_candle(candles_5m, setup_index, L, 'SHORT')
    pullback = P is not None
    if not pullback:
        return False

    tested = (C['low'] <= L) and (C['high'] >= L)
    closed_back = C['close'] < L
    body_bearish = C['close'] < C['open']
    no_break_above = C['high'] <= L

    rejection = tested and closed_back and body_bearish and no_break_above
    return rejection


def d7_pullback_pass(
    candles_5m: List[dict],
    setup_index: int,
    direction: str,
) -> bool:
    """Top-level D7 pullback/retest gate for the approved baseline.

    Selects L via EMA20-first precedence (direction-aware for structure fallback),
    then validates pullback + rejection.

    Args:
        candles_5m: closed 5m candles sorted ascending by close_time.
        setup_index: index of setup candle C.
        direction: 'LONG' or 'SHORT'.

    Returns:
        True if D7 passes, False if NO_TRADE via failed pullback confirmation.
    """
    if direction not in ('LONG', 'SHORT'):
        return False

    L = _select_retest_level_by_direction(candles_5m, setup_index, direction)
    if L is None:
        return False

    if direction == 'LONG':
        return validate_pullback_long(candles_5m, setup_index, L)
    else:
        return validate_pullback_short(candles_5m, setup_index, L)
