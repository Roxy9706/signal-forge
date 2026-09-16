"""Pure Python ATR (Wilder's True Range / Wilder smoothing).

ATR period = 14. Uses True Range per Wilder's original formulation.
No new strategy parameters. Uses only closed candles. Deterministic.
"""

from __future__ import annotations
from typing import List, Optional, Tuple


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


def atr_spike_metadata(
    candles: List[dict],
    *,
    atr_period: int = 14,
    spike_sma_window: int = 20,
    spike_multiplier: float = 3.0,
) -> Optional[Tuple[bool, Optional[float]]]:
    """D4 ATR-spike flag — pure metadata, zero effect on scoring/direction/NO_TRADE.

    For the latest closed candle C in `candles` (must be sorted oldest->newest,
    all closed with close_time <= T):

        atr_now  = Wilder ATR(14) through C
        window    = the `spike_sma_window` most recent ATR(14) values ending at C
        sma20     = arithmetic mean of `window`
        spike     = atr_now >= spike_multiplier * sma20   (inclusive at threshold)
        ratio     = atr_now / sma20                       (None if sma20 == 0)

    Returns None when fewer than `spike_sma_window` ATR values exist (i.e. the
    full trailing window of ATR14 values ending at C is unavailable) — this is
    the approved "spike = FALSE" path (metadata unavailable, not a spike event).

    sma20 == 0 is handled safely: ratio = None, spike = False (no division by zero).

    IMPORTANT: this function is metadata-only. It MUST NOT influence trend/momentum/
    structure/volume scores, final score, direction, NO_TRADE, entry, SL, TP, or
    R:R. Consumers that use this metadata are responsible for respecting that contract.
    """
    if not candles or atr_period < 2:
        return None

    atr_now = compute_atr(candles, atr_period)
    if atr_now is None:
        return None

    n = len(candles)
    # Minimum canvas: we need (atr_period + 1) candles for the first ATR value,
    # plus (spike_sma_window - 1) more candles so the trailing `spike_sma_window`
    # ATR values ending at the last candle are available.
    min_candles = atr_period + 1 + (spike_sma_window - 1)
    if n < min_candles:
        return None  # fewer than 20 ATR values → spike = FALSE path (metadata unavailable)

    window_atrs: List[float] = []
    for k in range(spike_sma_window):
        # Building block for ATR window position k (0 = most recent, window_atrs[-1] = oldest)
        offset = spike_sma_window - 1 - k  # how many candles back from the end
        # The ATR at position (n-1-offset) needs (atr_period+1) candles ending there
        end_idx = n - 1 - offset
        prefix = candles[: end_idx + 1]
        a = compute_atr(prefix, atr_period)
        if a is None:
            return None
        window_atrs.append(a)

    # At this point window_atrs has exactly spike_sma_window values (most recent first).
    sma20 = sum(window_atrs) / spike_sma_window
    if sma20 == 0.0:
        return (False, None)
    ratio = atr_now / sma20
    spike = atr_now >= spike_multiplier * sma20 - 1e-9  # inclusive at threshold; float-tolerant
    return (spike, ratio)
