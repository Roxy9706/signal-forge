"""D8 Multi-Timeframe Aggregation — final LONG/SHORT/NO_TRADE direction gate.

Reuses existing per-timeframe component scores:
  - Trend:    indicators.trend.trend_score(candles) -> (long, short)
  - Momentum: indicators.rsi.compute_rsi(candles) + indicators.rsi.momentum_score(rsi, dir)
  - Structure: indicators.structure.structure_score(candles, direction=dir)
  - Volume:   indicators.d6_volume.compute_rvol(candles) + indicators.d6_volume.volume_score(rvol)

Category weights:  Trend 0.40, Momentum 0.20, Structure 0.20, Volume 0.20.
Timeframe weights: 1d 0.25, 4h 0.25, 1h 0.20, 15m 0.20, 5m 0.10.

Direction gate:
  LONG  if FINAL_LONG >= 65 AND FINAL_LONG > FINAL_SHORT
  SHORT if FINAL_SHORT >= 65 AND FINAL_SHORT > FINAL_LONG
  NO_TRADE otherwise (including ties).

Insufficient per-timeframe component data -> that TF score = 50 (neutral) for both
directions, matching the per-component conventions (trend/structure unavailable = 50,
momentum unavailable returns 0 from momentum_score, volume unavailable returns None).
The aggregation layer treats momentum None as 0 and volume None as 50 to reflect the
approved "no hidden invention" principle while keeping the gate deterministic.

D8 does NOT implement entry/SL/TP/R:R/pullback/D9 or trading.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP

from indicators.trend import trend_score
from indicators.rsi import compute_rsi, momentum_score
from indicators.structure import structure_score
from indicators.d6_volume import compute_rvol, volume_score

# ---------------------------------------------------------------------------
# Weights (frozen — match approved spec exactly)
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS = {
    "trend": 0.40,
    "momentum": 0.20,
    "structure": 0.20,
    "volume": 0.20,
}
TF_WEIGHTS = {
    "1d": 0.25,
    "4h": 0.25,
    "1h": 0.20,
    "15m": 0.20,
    "5m": 0.10,
}
DIRECTION_THRESHOLD = 65
TF_ORDER = ["1d", "4h", "1h", "15m", "5m"]  # canonical order, weights sum to 1.0


# ---------------------------------------------------------------------------
# Per-timeframe aggregation
# ---------------------------------------------------------------------------

def _tf_scores(candles: List[dict]) -> Tuple[float, float]:
    """Compute per-timeframe LONG and SHORT composite scores.

    TF_LONG  = Trend_LONG*0.40 + Momentum_LONG*0.20 + Structure_LONG*0.20 + Volume*0.20
    TF_SHORT = Trend_SHORT*0.40 + Momentum_SHORT*0.20 + Structure_SHORT*0.20 + Volume*0.20

    Volume is direction-independent and reused for both.

    Returns (tf_long, tf_short) as floats 0-100.
    """
    trend_long, trend_short = trend_score(candles)

    rsi = compute_rsi(candles)
    mom_long = momentum_score(rsi, "LONG")
    mom_short = momentum_score(rsi, "SHORT")

    struct_long = structure_score(candles, direction="LONG")
    struct_short = structure_score(candles, direction="SHORT")

    rvol = compute_rvol(candles)
    vol = volume_score(rvol)
    if vol is None:
        # RVOL unavailable -> treat as neutral 50 per D6-available-convention
        # where None score means no data. Aggregator uses 50 for neutrality.
        vol = 50

    tf_long = (trend_long * CATEGORY_WEIGHTS["trend"]
               + mom_long * CATEGORY_WEIGHTS["momentum"]
               + struct_long * CATEGORY_WEIGHTS["structure"]
               + vol * CATEGORY_WEIGHTS["volume"])

    tf_short = (trend_short * CATEGORY_WEIGHTS["trend"]
                + mom_short * CATEGORY_WEIGHTS["momentum"]
                + struct_short * CATEGORY_WEIGHTS["structure"]
                + vol * CATEGORY_WEIGHTS["volume"])

    return tf_long, tf_short


# ---------------------------------------------------------------------------
# Multi-timeframe aggregation + direction gate
# ---------------------------------------------------------------------------

def aggregate(
    candles_by_tf: dict,
) -> Tuple[float, float, str]:
    """Aggregate per-timeframe scores into FINAL_LONG, FINAL_SHORT, and direction.

    Args:
        candles_by_tf: dict mapping timeframe string to list of closed candles
            (sorted oldest-to-newest, all closed with close_time <= T).
            Keys must be a subset of {"1d", "4h", "1h", "15m", "5m"}.
            Missing keys are treated as zero-weight (score 50 neutral, contributes 0).

    Returns:
        (final_long, final_short, direction)
          final_long/final_short: rounded to 1 decimal place.
          direction: "LONG", "SHORT", or "NO_TRADE".
    """
    weighted_long = 0.0
    weighted_short = 0.0

    for tf in TF_ORDER:
        w = TF_WEIGHTS[tf]
        candles = candles_by_tf.get(tf)
        if candles is None or len(candles) == 0:
            # Missing TF -> neutral 50 for both, weighted contribution zero
            # (50 * w for each, but we just skip since it contributes equally
            # to both sides — actually we must add 50*w to both to keep total
            # in 0-100 range. Missing TF means we have NO data, so per the
            # approved spec we still weight it at 50 neutral.)
            weighted_long += 50.0 * w
            weighted_short += 50.0 * w
            continue

        tf_long, tf_short = _tf_scores(candles)
        weighted_long += tf_long * w
        weighted_short += tf_short * w

    final_long = _round1(weighted_long)
    final_short = _round1(weighted_short)

    if final_long >= DIRECTION_THRESHOLD and final_long > final_short:
        direction = "LONG"
    elif final_short >= DIRECTION_THRESHOLD and final_short > final_long:
        direction = "SHORT"
    else:
        direction = "NO_TRADE"

    return final_long, final_short, direction


def _round1(x: float) -> float:
    """Round to 1 decimal place using ROUND_HALF_UP for consistency."""
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Convenience: per-TF breakdown (for inspection, not scoring)
# ---------------------------------------------------------------------------

def breakdown(candles_by_tf: dict) -> dict:
    """Return per-timeframe component and composite scores.

    Returns dict:
        {tf: {"trend_long": ..., "trend_short": ...,
              "mom_long": ..., "mom_short": ...,
              "struct_long": ..., "struct_short": ...,
              "vol": ...,
              "tf_long": ..., "tf_short": ...}}
    Missing TFs yield None per field.
    """
    result = {}
    for tf in TF_ORDER:
        candles = candles_by_tf.get(tf)
        if candles is None or len(candles) == 0:
            result[tf] = {
                "trend_long": None, "trend_short": None,
                "mom_long": None, "mom_short": None,
                "struct_long": None, "struct_short": None,
                "vol": None,
                "tf_long": None, "tf_short": None,
            }
            continue

        trend_long, trend_short = trend_score(candles)
        rsi = compute_rsi(candles)
        mom_long = momentum_score(rsi, "LONG")
        mom_short = momentum_score(rsi, "SHORT")
        struct_long = structure_score(candles, direction="LONG")
        struct_short = structure_score(candles, direction="SHORT")
        rvol = compute_rvol(candles)
        vol = volume_score(rvol)
        if vol is None:
            vol = 50

        tf_long = (trend_long * CATEGORY_WEIGHTS["trend"]
                   + mom_long * CATEGORY_WEIGHTS["momentum"]
                   + struct_long * CATEGORY_WEIGHTS["structure"]
                   + vol * CATEGORY_WEIGHTS["volume"])
        tf_short = (trend_short * CATEGORY_WEIGHTS["trend"]
                    + mom_short * CATEGORY_WEIGHTS["momentum"]
                    + struct_short * CATEGORY_WEIGHTS["structure"]
                    + vol * CATEGORY_WEIGHTS["volume"])

        result[tf] = {
            "trend_long": trend_long, "trend_short": trend_short,
            "mom_long": mom_long, "mom_short": mom_short,
            "struct_long": struct_long, "struct_short": struct_short,
            "vol": vol,
            "tf_long": _round1(tf_long), "tf_short": _round1(tf_short),
        }
    return result
