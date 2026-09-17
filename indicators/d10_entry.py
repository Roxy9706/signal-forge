"""D10 Entry — approved spec only.

D10 combines D8 direction with D7 same-direction confirmation
and emits entry price + signal_time from the confirmed 5m setup candle.

No D11/D12 SL, D13 TP, D14 R:R, D15 NO_TRADE, D9. D8 unchanged.
"""
from __future__ import annotations
from typing import Tuple, Optional
from indicators.pullback import d7_pullback_pass


def d10_entry(d8_dir: str, d7_pass: bool, setup_5m: dict,
              d7_dir: Optional[str] = None) -> dict:
    """Compute D10 entry from approved inputs.

    Args:
        d8_dir: D8 aggregate direction ("LONG", "SHORT", "NO_TRADE").
        d7_pass: Whether D7 pullback/rejection passes.
        setup_5m: dict with 'close' (float) and 'close_time' (int)
                  of the confirmed setup 5m candle (already closed).
        d7_dir: direction D7 was validated against. Must equal d8_dir.

    Returns:
        dict with keys: direction, entry, signal_time, status.
        direction: "LONG" or "SHORT" if entry confirmed, else "NO_TRADE".
        entry: close price of setup candle, or None if rejected.
        signal_time: close_time of setup candle, or None if rejected.
        status: "OK" | "MISMATCH" | "BAD_DIR" | "D7_FAIL".
    """
    # Direction match: D7/D8 mismatch or D8 NO_TRADE -> reject
    if d8_dir == "NO_TRADE" or d7_dir is None or d7_dir != d8_dir:
        return {"direction": "NO_TRADE", "entry": None,
                "signal_time": None, "status": "MISMATCH"}

    # Validate d8_dir is a known direction
    if d8_dir not in ("LONG", "SHORT"):
        return {"direction": "NO_TRADE", "entry": None,
                "signal_time": None, "status": "BAD_DIR"}

    # D7 must pass in the same direction
    if not d7_pass:
        return {"direction": "NO_TRADE", "entry": None,
                "signal_time": None, "status": "D7_FAIL"}

    # Closed-candle only: use setup_5m close as entry
    close_val = setup_5m.get("close")
    if close_val is None:
        return {"direction": "NO_TRADE", "entry": None,
                "signal_time": None, "status": "MISSING_DATA"}
    entry_price = float(close_val)
    close_time_val = setup_5m.get("close_time")
    if close_time_val is None:
        return {"direction": d8_dir, "entry": entry_price,
                "signal_time": None, "status": "MISSING_DATA"}
    signal_time = int(close_time_val)
    return {"direction": d8_dir, "entry": entry_price,
            "signal_time": signal_time, "status": "OK"}
