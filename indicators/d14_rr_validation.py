"""D14 R:R Validation — pure calculation/validation, no external dependencies.

Consumes direction + D10 entry price + D11/D12 final SL + D13 take profit.
Independently calculates R:R and validates against the 2.0 threshold.

No D15, no trading/order functionality, no API calls, no keys/credentials.

Reuses existing status vocabulary (OK / MISSING_DATA / BAD_DIR) plus
INVALID_RISK for zero/negative risk, and FAIL for sub-2.0 R:R.
"""

from __future__ import annotations
from typing import Optional


def d14_rr_validation(
    direction: str,
    entry_price: Optional[float],
    final_sl: Optional[float],
    take_profit: Optional[float],
) -> dict:
    """Validate risk:reward ratio independently.

    Args:
        direction: 'LONG' or 'SHORT' (mirrors D8/D10/D13 direction).
        entry_price: D10 entry price, may be None.
        final_sl: D11/D12 final stop-loss price, may be None.
        take_profit: D13 take-profit price, may be None.

    Returns:
        dict with keys:
          direction, entry_price, final_sl, take_profit,
          risk_R, reward, rr, status.
        status: 'OK' | 'FAIL' | 'INVALID_RISK' | 'MISSING_DATA' | 'BAD_DIR'.
        Numeric fields are None on non-OK/FAIL status unless the value
        itself is the problem (e.g. risk_R on INVALID_RISK).
    """
    # --- Direction validation ---
    if direction not in ("LONG", "SHORT"):
        return {
            "direction": direction,
            "entry_price": None,
            "final_sl": None,
            "take_profit": None,
            "risk_R": None,
            "reward": None,
            "rr": None,
            "status": "BAD_DIR",
        }

    # --- Missing data check: all three numeric inputs required ---
    if entry_price is None or final_sl is None or take_profit is None:
        return {
            "direction": direction,
            "entry_price": entry_price,
            "final_sl": final_sl,
            "take_profit": take_profit,
            "risk_R": None,
            "reward": None,
            "rr": None,
            "status": "MISSING_DATA",
        }

    entry = float(entry_price)
    sl = float(final_sl)
    tp = float(take_profit)

    # --- Risk: absolute distance from entry to SL ---
    R = abs(entry - sl)

    # --- INVALID_RISK when R <= 0 ---
    if R <= 0.0:
        return {
            "direction": direction,
            "entry_price": entry,
            "final_sl": sl,
            "take_profit": tp,
            "risk_R": R,
            "reward": None,
            "rr": None,
            "status": "INVALID_RISK",
        }

    # --- Reward: absolute distance from entry to TP ---
    reward = abs(tp - entry)

    # --- R:R ratio ---
    rr = reward / R

    # --- Threshold: 2.0 ---
    if rr >= 2.0:
        status = "OK"
    else:
        status = "FAIL"

    return {
        "direction": direction,
        "entry_price": entry,
        "final_sl": sl,
        "take_profit": tp,
        "risk_R": R,
        "reward": reward,
        "rr": rr,
        "status": status,
    }
