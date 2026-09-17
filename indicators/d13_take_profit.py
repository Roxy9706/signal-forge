"""D13 Take Profit — pure calculation, no external dependencies.

Consumes direction + D10 entry price + D11/D12 final SL.
Computes risk R = abs(entry_price - final_sl), then TP = entry +/- 2R.
No D14 R:R, D15, NO_TRADE changes, trading/order functionality, API calls.

Reuses existing status vocabulary (OK / MISSING_DATA / BAD_DIR) without
duplicating D11/D12 internal calculation logic.
"""

from __future__ import annotations
from typing import Optional


def d13_take_profit(
    direction: str,
    entry_price: Optional[float],
    final_sl: Optional[float],
) -> dict:
    """Compute D13 take-profit from approved inputs.

    Args:
        direction: 'LONG' or 'SHORT' (mirrors D8/D10 direction).
        entry_price: D10 entry price (setup 5m close), may be None.
        final_sl: D11/D12 final stop-loss price, may be None.

    Returns:
        dict with keys:
          direction, entry_price, final_sl, risk_R, take_profit, status.
        status: 'OK' | 'MISSING_DATA' | 'BAD_DIR'.
        take_profit and risk_R are None on non-OK status.
    """
    # Normalise/validate direction
    if direction not in ("LONG", "SHORT"):
        return {
            "direction": direction,
            "entry_price": None,
            "final_sl": None,
            "risk_R": None,
            "take_profit": None,
            "status": "BAD_DIR",
        }

    # Validate required numeric inputs
    if entry_price is None or final_sl is None:
        return {
            "direction": direction,
            "entry_price": entry_price,
            "final_sl": final_sl,
            "risk_R": None,
            "take_profit": None,
            "status": "MISSING_DATA",
        }

    entry = float(entry_price)
    sl = float(final_sl)

    # Risk distance (entry to SL, absolute)
    R = abs(entry - sl)

    # Take-profit: 2R beyond entry in the trade direction
    if direction == "LONG":
        tp = entry + 2.0 * R
    else:  # SHORT
        tp = entry - 2.0 * R

    return {
        "direction": direction,
        "entry_price": entry,
        "final_sl": sl,
        "risk_R": R,
        "take_profit": tp,
        "status": "OK",
    }
