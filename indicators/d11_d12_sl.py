"""D11 Structure Swing + D12 ATR Buffer — SL calculation.

Uses D7's 20-candle pre-setup window (excludes setup candle) for swing selection.
Uses existing strict 3-candle fractal logic from indicators.structure.
Uses existing ATR14 from indicators.atr.

D11 and D12 kept as separate internal functions; combined result exposed.
No D13 TP, D14 R:R, D15 NO_TRADE, D9. D8/D10 unchanged.
"""
from __future__ import annotations
from typing import Optional, Tuple, List

from indicators.structure import (
    most_recent_swing_low,
    most_recent_swing_high,
)
from indicators.atr import compute_atr


def d11_swing_selection(
    candles_5m_before_setup: List[dict],
    direction: str,
) -> Optional[dict]:
    """D11 — select the most recent valid structure swing.

    Args:
        candles_5m_before_setup: 20 (or fewer) immediately preceding
            CLOSED 5m candles before the setup candle. The setup candle
            is NOT included.
        direction: 'LONG' or 'SHORT'.

    Returns:
        The swing candle dict (with 'low' for LONG, 'high' for SHORT)
        selected by the existing 3-candle strict fractal logic,
        or None if unavailable.
    """
    if direction == "LONG":
        return most_recent_swing_low(candles_5m_before_setup)
    elif direction == "SHORT":
        return most_recent_swing_high(candles_5m_before_setup)
    else:
        return None


def d12_atr_buffer(
    candles_5m_through_setup: List[dict],
) -> Optional[float]:
    """D12 — compute ATR14 buffer (0.5 * ATR14 through setup candle).

    Args:
        candles_5m_through_setup: closed 5m candles including setup
            (used to compute ATR14 through the setup candle).

    Returns:
        buffer = 0.5 * ATR14 as float, or None if unavailable.
    """
    atr = compute_atr(candles_5m_through_setup, period=14)
    if atr is None:
        return None
    return 0.5 * float(atr)


def d11_d12_sl_result(
    d8_dir: str,
    d10_setup_5m: dict,
    candles_5m_before_setup: List[dict],
    candles_5m_through_setup: List[dict],
) -> dict:
    """Combined D11 (swing) + D12 (ATR buffer) SL result.

    Args:
        d8_dir: D8 aggregate direction ('LONG' or 'SHORT').
        d10_setup_5m: the confirmed setup 5m candle (must have 'close').
        candles_5m_before_setup: 20 (or fewer) preceding closed candles.
        candles_5m_through_setup: candles through and including setup.

    Returns:
        dict: direction, selected_swing_price, atr14, buffer,
        final_sl, status.
        status: 'OK' | 'MISSING_DATA' | 'BAD_GEOMETRY' | 'BAD_DIR'
    """
    if d8_dir not in ("LONG", "SHORT"):
        return {
            "direction": d8_dir,
            "selected_swing_price": None,
            "atr14": None,
            "buffer": None,
            "final_sl": None,
            "status": "BAD_DIR",
        }

    # D11 — swing
    swing = d11_swing_selection(candles_5m_before_setup, d8_dir)
    if swing is None:
        return {
            "direction": d8_dir,
            "selected_swing_price": None,
            "atr14": None,
            "buffer": None,
            "final_sl": None,
            "status": "MISSING_DATA",
        }

    # D12 — ATR14 buffer
    buffer = d12_atr_buffer(candles_5m_through_setup)
    if buffer is None:
        return {
            "direction": d8_dir,
            "selected_swing_price": swing.get("low")
                if d8_dir == "LONG" else swing.get("high"),
            "atr14": None,
            "buffer": None,
            "final_sl": None,
            "status": "MISSING_DATA",
        }

    entry_price = float(d10_setup_5m.get("close"))
    # SL formulas
    if d8_dir == "LONG":
        swing_price = float(swing.get("low"))
        final_sl = swing_price - buffer
    else:  # SHORT
        swing_price = float(swing.get("high"))
        final_sl = swing_price + buffer

    # Geometry checks: never artificially adjust
    if d8_dir == "LONG" and final_sl >= entry_price:
        return {
            "direction": d8_dir,
            "selected_swing_price": swing_price,
            "atr14": float(compute_atr(candles_5m_through_setup, period=14) or 0),
            "buffer": buffer,
            "final_sl": final_sl,
            "status": "BAD_GEOMETRY",
        }
    if d8_dir == "SHORT" and final_sl <= entry_price:
        return {
            "direction": d8_dir,
            "selected_swing_price": swing_price,
            "atr14": float(compute_atr(candles_5m_through_setup, period=14) or 0),
            "buffer": buffer,
            "final_sl": final_sl,
            "status": "BAD_GEOMETRY",
        }

    # Re-calculate ATR14 for output when present
    atr14_raw = compute_atr(candles_5m_through_setup, period=14)
    return {
        "direction": d8_dir,
        "selected_swing_price": swing_price,
        "atr14": float(atr14_raw) if atr14_raw is not None else None,
        "buffer": buffer,
        "final_sl": final_sl,
        "status": "OK",
    }
