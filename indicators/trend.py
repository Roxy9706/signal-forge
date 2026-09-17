"""D8 Trend component — EMA20/50/200 hierarchy + close confirmation.

Direction-relative 0-100 score per timeframe.

EMA hierarchy (independent of price):
  Bullish hierarchy: EMA20 > EMA50 > EMA200
  Bearish hierarchy: EMA20 < EMA50 < EMA200

LONG Trend score:
  bullish hierarchy AND close > EMA20   -> 100
  bullish hierarchy AND close <= EMA20  -> 75
  mixed/neither hierarchy               -> 50
  bearish hierarchy AND close <= EMA20  -> 25
  bearish hierarchy AND close > EMA20   -> 0

SHORT Trend score:
  bearish hierarchy AND close < EMA20   -> 100
  bearish hierarchy AND close >= EMA20  -> 75
  mixed/neither hierarchy               -> 50
  bullish hierarchy AND close >= EMA20  -> 25
  bullish hierarchy AND close < EMA20   -> 0

Insufficient EMA data (any of EMA20/50/200 unavailable) -> (50, 50).
No fabrication of EMA values.

Reuses compute_ema from indicators.ema. Does NOT modify compute_ema behavior.
"""

from __future__ import annotations
from typing import List, Tuple
from indicators.ema import compute_ema


def trend_score(candles: List[dict]) -> Tuple[int, int]:
    """Compute LONG and SHORT Trend scores from the latest closed candle.

    Args:
        candles: list of dicts with 'close' key, sorted oldest-to-newest,
                 all closed (close_time <= T). The last candle is C_tf.

    Returns:
        (long_score, short_score) — each int in 0-100.
    """
    ema20 = compute_ema(candles, 20)
    ema50 = compute_ema(candles, 50)
    ema200 = compute_ema(candles, 200)

    if ema20 is None or ema50 is None or ema200 is None:
        return (50, 50)

    close = candles[-1]['close']

    bullish_hier = ema20 > ema50 and ema50 > ema200
    bearish_hier = ema20 < ema50 and ema50 < ema200

    if bullish_hier:
        long_s = 100 if close > ema20 else 75
        short_s = 25 if close >= ema20 else 0
    elif bearish_hier:
        long_s = 25 if close <= ema20 else 0
        short_s = 100 if close < ema20 else 75
    else:
        long_s = 50
        short_s = 50

    return (long_s, short_s)
