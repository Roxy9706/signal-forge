"""D15 — Final Strategy Decision Orchestrator.

Reads-only orchestration of D7–D14 stages.
No recalculation of formulas. No trading/order/API functionality.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from indicators.d8_aggregate import aggregate as d8_aggregate
from indicators.d10_entry import d10_entry as d10_entry_fn
from indicators.d11_d12_sl import d11_d12_sl_result as d11_d12_sl_result_fn
from indicators.d13_take_profit import d13_take_profit as d13_take_profit_fn
from indicators.d14_rr_validation import d14_rr_validation as d14_rr_validation_fn
from indicators.pullback import (
    _select_retest_level_by_direction,
    d7_pullback_pass,
)


def d15_final_decision(
    candles_by_tf: Dict[str, List[dict]],
    candles_5m: List[dict],
    setup_index: int,
    setup_5m: dict,
    candles_5m_before_setup: List[dict],
    candles_5m_through_setup: List[dict],
) -> Dict[str, Any]:
    """Orchestrate D7–D14 stages and produce a final trade signal or NO_TRADE.

    Args:
        candles_by_tf: dict mapping timeframe string to list of closed candles
            (for D8 aggregation).
        candles_5m: list of closed 5m candles sorted ascending by close_time.
        setup_index: index of the setup candle in candles_5m.
        setup_5m: the setup candle dict (must have 'close' and 'close_time').
        candles_5m_before_setup: list of candles immediately preceding the
            setup candle (for D11 swing selection).
        candles_5m_through_setup: list of candles through and including the
            setup candle (for D12 ATR calculation).

    Returns:
        dict with final decision, stage statuses, and optional reason.
    """
    # Initialize result with default INVALID_INPUT state
    result: Dict[str, Any] = {
        "direction": "NO_TRADE",
        "entry": None,
        "signal_time": None,
        "final_sl": None,
        "take_profit": None,
        "final_long": None,
        "final_short": None,
        "rr": None,
        "status": "INVALID_INPUT",
        "stage_status": {
            "d8": "SKIPPED",
            "d7": "SKIPPED",
            "d10": "SKIPPED",
            "d11_d12": "SKIPPED",
            "d13": "SKIPPED",
            "d14": "SKIPPED",
        },
        "reason": None,
    }

    # ── Input validation ──────────────────────────────────────────────
    if not isinstance(candles_by_tf, dict):
        result["reason"] = "candles_by_tf must be a dict"
        return result
    if not isinstance(candles_5m, list) or len(candles_5m) == 0:
        result["reason"] = "candles_5m must be a non-empty list"
        return result
    if not isinstance(setup_index, int) or setup_index < 0 or setup_index >= len(candles_5m):
        result["reason"] = f"setup_index must be int in [0, {len(candles_5m)})"
        return result
    if not isinstance(setup_5m, dict):
        result["reason"] = "setup_5m must be a dict"
        return result
    if "close" not in setup_5m or "close_time" not in setup_5m:
        result["reason"] = "setup_5m must have 'close' and 'close_time' keys"
        return result
    if not isinstance(candles_5m_before_setup, list):
        result["reason"] = "candles_5m_before_setup must be a list"
        return result
    if not isinstance(candles_5m_through_setup, list):
        result["reason"] = "candles_5m_through_setup must be a list"
        return result

    # Correspondence check — setup_5m must match candles_5m[setup_index]
    setup_candle = candles_5m[setup_index]
    if not isinstance(setup_candle, dict):
        result["reason"] = "candles_5m[setup_index] must be a dict"
        return result

    sc_close = setup_candle.get("close")
    s_close = setup_5m.get("close")
    if sc_close is not None and s_close is not None and sc_close != s_close:
        result["reason"] = "setup_5m['close'] must match candles_5m[setup_index]['close']"
        return result

    sc_ct = setup_candle.get("close_time")
    s_ct = setup_5m.get("close_time")
    if sc_ct is not None and s_ct is not None and sc_ct != s_ct:
        result["reason"] = "setup_5m['close_time'] must match candles_5m[setup_index]['close_time']"
        return result

    # ── D8: Direction & threshold gate ───────────────────────────────
    final_long, final_short, d8_dir = d8_aggregate(candles_by_tf)
    result["final_long"] = final_long
    result["final_short"] = final_short
    result["stage_status"]["d8"] = {
        "direction": d8_dir,
        "final_long": final_long,
        "final_short": final_short,
    }

    if d8_dir not in ("LONG", "SHORT"):
        result["status"] = "NO_TRADE"
        result["direction"] = "NO_TRADE"
        result["reason"] = (
            f"D8 threshold/directional failure: "
            f"final_long={final_long}, final_short={final_short}, direction={d8_dir}"
        )
        return result

    # ── D7: Pullback/rejection gate (Option A — no D7 modification) ──
    retest_level = _select_retest_level_by_direction(
        candles_5m, setup_index, d8_dir
    )
    if retest_level is None:
        result["status"] = "DATA_UNAVAILABLE"
        result["direction"] = "NO_TRADE"
        result["reason"] = (
            "D7: retest level unavailable — insufficient data for pullback confirmation"
        )
        result["stage_status"]["d7"] = {
            "pass": False,
            "retest_available": False,
        }
        return result

    d7_pass = d7_pullback_pass(candles_5m, setup_index, d8_dir)
    result["stage_status"]["d7"] = {
        "pass": d7_pass,
        "retest_available": True,
    }

    if not d7_pass:
        result["status"] = "NO_TRADE"
        result["direction"] = "NO_TRADE"
        result["reason"] = "D7: pullback/rejection failed"
        return result

    # ── D10: Entry ───────────────────────────────────────────────────
    d10_res = d10_entry_fn(d8_dir, d7_pass, setup_5m, d8_dir)
    result["stage_status"]["d10"] = d10_res["status"]

    if d10_res["status"] == "MISSING_DATA":
        result["status"] = "DATA_UNAVAILABLE"
        result["reason"] = "D10: missing data for entry calculation"
        return result
    elif d10_res["status"] == "BAD_DIR":
        result["status"] = "INVALID_INPUT"
        result["reason"] = "D10: invalid direction"
        return result
    elif d10_res["status"] != "OK":
        # MISMATCH or D7_FAIL — shouldn't normally occur given prior gates,
        # but propagate as NO_TRADE if it does
        result["status"] = "NO_TRADE"
        result["reason"] = f"D10: {d10_res['status']}"
        return result

    entry_price = d10_res["entry"]
    signal_time = d10_res["signal_time"]
    result["entry"] = entry_price
    result["signal_time"] = signal_time

    # ── D11/D12: Stop Loss ───────────────────────────────────────────
    sl_res = d11_d12_sl_result_fn(
        d8_dir,
        setup_5m,
        candles_5m_before_setup,
        candles_5m_through_setup,
    )
    result["stage_status"]["d11_d12"] = sl_res["status"]

    if sl_res["status"] == "MISSING_DATA":
        result["status"] = "DATA_UNAVAILABLE"
        result["reason"] = "D11/D12: missing data for SL calculation"
        return result
    elif sl_res["status"] == "BAD_GEOMETRY":
        result["status"] = "NO_TRADE"
        result["reason"] = "D11/D12: invalid SL geometry"
        return result
    elif sl_res["status"] == "BAD_DIR":
        result["status"] = "INVALID_INPUT"
        result["reason"] = "D11/D12: invalid direction"
        return result

    final_sl = sl_res["final_sl"]
    result["final_sl"] = final_sl

    # ── D13: Take Profit ─────────────────────────────────────────────
    tp_res = d13_take_profit_fn(d8_dir, entry_price, final_sl)
    result["stage_status"]["d13"] = tp_res["status"]

    if tp_res["status"] == "MISSING_DATA":
        result["status"] = "DATA_UNAVAILABLE"
        result["reason"] = "D13: missing data for TP calculation"
        return result
    elif tp_res["status"] == "BAD_DIR":
        result["status"] = "INVALID_INPUT"
        result["reason"] = "D13: invalid direction"
        return result

    take_profit = tp_res["take_profit"]
    result["take_profit"] = take_profit

    # ── D14: R:R Validation ──────────────────────────────────────────
    rr_res = d14_rr_validation_fn(d8_dir, entry_price, final_sl, take_profit)
    result["stage_status"]["d14"] = rr_res["status"]

    if rr_res["status"] == "FAIL":
        result["status"] = "NO_TRADE"
        result["reason"] = (
            f"D14: R:R {rr_res['rr']:.2f} below 2.0 threshold"
        )
        result["rr"] = rr_res["rr"]
        return result
    elif rr_res["status"] == "INVALID_RISK":
        result["status"] = "NO_TRADE"
        result["reason"] = "D14: invalid risk (zero or negative risk distance)"
        result["rr"] = rr_res["risk_R"]
        return result
    elif rr_res["status"] == "MISSING_DATA":
        result["status"] = "DATA_UNAVAILABLE"
        result["reason"] = "D14: missing data for R:R validation"
        return result
    elif rr_res["status"] == "BAD_DIR":
        result["status"] = "INVALID_INPUT"
        result["reason"] = "D14: invalid direction"
        return result

    # ── ALL GATES PASS ───────────────────────────────────────────────
    result["status"] = "OK"
    result["direction"] = d8_dir
    result["rr"] = rr_res["rr"]
    result["reason"] = None
    return result
