"""Tests for D15 Final Decision Orchestrator — indicators.d15_decision.

D15 is a pure orchestrator: it calls D7–D14 stage functions in sequence and
propagates their results. Tests verify orchestration logic, not stage internals.

Stage mocks return canned results so D15's decision flow is tested in isolation.
"""

from __future__ import annotations
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.d15_decision import d15_final_decision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _k(ct, o, h, l, c, v=1.0):
    """Minimal candle dict."""
    return {"close_time": ct, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _mock_d8_long():
    """Mock D8 to return LONG direction with scores 80/20."""
    return patch("indicators.d15_decision.d8_aggregate",
                 return_value=(80.0, 20.0, "LONG"))


def _mock_d8_short():
    """Mock D8 to return SHORT direction with scores 20/80."""
    return patch("indicators.d15_decision.d8_aggregate",
                 return_value=(20.0, 80.0, "SHORT"))


def _mock_d8_no_trade():
    """Mock D8 to return NO_TRADE (50/50 neutral)."""
    return patch("indicators.d15_decision.d8_aggregate",
                 return_value=(50.0, 50.0, "NO_TRADE"))


def _d7_patches(p=True, retest_available=True):
    """Return two patches for D7 functions as a context-manager-compatible pair.

    Usage:
        with _d7_patches(p=True):
            ...  # both _select_retest_level_by_direction and d7_pullback_pass mocked
    """
    import indicators.d15_decision as d15
    p1 = patch.object(d15, "_select_retest_level_by_direction",
                      return_value=102.0 if retest_available else None)
    p2 = patch.object(d15, "d7_pullback_pass", return_value=p)
    p1.__enter__()
    return _D7PatchesContext(p1, p2)


class _D7PatchesContext:
    """Context manager that enters two patches and exits both on __exit__."""

    def __init__(self, p1, p2):
        self._p1 = p1
        self._p2 = p2

    def __enter__(self):
        self._p2.__enter__()
        return self

    def __exit__(self, *args):
        self._p2.__exit__(*args)
        self._p1.__exit__(*args)


def _mock_d10_ok(direction="LONG", entry=103.5, signal_time=100):
    """Mock D10 to return OK with given entry."""
    return patch("indicators.d15_decision.d10_entry_fn",
                 return_value={
                     "direction": direction,
                     "entry": entry,
                     "signal_time": signal_time,
                     "status": "OK",
                 })


def _mock_d10_missing_data():
    """Mock D10 to return MISSING_DATA."""
    return patch("indicators.d15_decision.d10_entry_fn",
                 return_value={
                     "direction": "LONG",
                     "entry": None,
                     "signal_time": None,
                     "status": "MISSING_DATA",
                 })


def _mock_d10_bad_dir():
    """Mock D10 to return BAD_DIR."""
    return patch("indicators.d15_decision.d10_entry_fn",
                 return_value={
                     "direction": "INVALID",
                     "entry": None,
                     "signal_time": None,
                     "status": "BAD_DIR",
                 })


def _mock_d10_mismatch():
    """Mock D10 to return MISMATCH (shouldn't happen in normal flow)."""
    return patch("indicators.d15_decision.d10_entry_fn",
                 return_value={
                     "direction": "NO_TRADE",
                     "entry": None,
                     "signal_time": None,
                     "status": "MISMATCH",
                 })


def _mock_d11d12_ok(direction="LONG", swing_price=98.0, atr14=1.0,
                     buffer_val=0.5, final_sl=97.5):
    """Mock D11/D12 to return OK with given SL values."""
    return patch("indicators.d15_decision.d11_d12_sl_result_fn",
                 return_value={
                     "direction": direction,
                     "selected_swing_price": swing_price,
                     "atr14": atr14,
                     "buffer": buffer_val,
                     "final_sl": final_sl,
                     "status": "OK",
                 })


def _mock_d11d12_missing_data():
    """Mock D11/D12 to return MISSING_DATA."""
    return patch("indicators.d15_decision.d11_d12_sl_result_fn",
                 return_value={
                     "direction": "LONG",
                     "selected_swing_price": None,
                     "atr14": None,
                     "buffer": None,
                     "final_sl": None,
                     "status": "MISSING_DATA",
                 })


def _mock_d11d12_bad_geometry():
    """Mock D11/D12 to return BAD_GEOMETRY."""
    return patch("indicators.d15_decision.d11_d12_sl_result_fn",
                 return_value={
                     "direction": "LONG",
                     "selected_swing_price": 101.0,
                     "atr14": 0.5,
                     "buffer": 0.5,
                     "final_sl": 100.5,
                     "status": "BAD_GEOMETRY",
                 })


def _mock_d11d12_bad_dir():
    """Mock D11/D12 to return BAD_DIR."""
    return patch("indicators.d15_decision.d11_d12_sl_result_fn",
                 return_value={
                     "direction": "INVALID",
                     "selected_swing_price": None,
                     "atr14": None,
                     "buffer": None,
                     "final_sl": None,
                     "status": "BAD_DIR",
                 })


def _mock_d13_ok(direction="LONG", entry_price=103.5, final_sl=97.5,
                 risk_R=6.0, take_profit=115.5):
    """Mock D13 to return OK with given TP values."""
    return patch("indicators.d15_decision.d13_take_profit_fn",
                 return_value={
                     "direction": direction,
                     "entry_price": entry_price,
                     "final_sl": final_sl,
                     "risk_R": risk_R,
                     "take_profit": take_profit,
                     "status": "OK",
                 })


def _mock_d13_missing_data():
    """Mock D13 to return MISSING_DATA."""
    return patch("indicators.d15_decision.d13_take_profit_fn",
                 return_value={
                     "direction": "LONG",
                     "entry_price": None,
                     "final_sl": None,
                     "risk_R": None,
                     "take_profit": None,
                     "status": "MISSING_DATA",
                 })


def _mock_d13_bad_dir():
    """Mock D13 to return BAD_DIR."""
    return patch("indicators.d15_decision.d13_take_profit_fn",
                 return_value={
                     "direction": "INVALID",
                     "entry_price": None,
                     "final_sl": None,
                     "risk_R": None,
                     "take_profit": None,
                     "status": "BAD_DIR",
                 })


def _mock_d14_ok(direction="LONG", entry_price=103.5, final_sl=97.5,
                 take_profit=115.5, risk_R=6.0, reward=12.0, rr=2.0):
    """Mock D14 to return OK with given R:R values."""
    return patch("indicators.d15_decision.d14_rr_validation_fn",
                 return_value={
                     "direction": direction,
                     "entry_price": entry_price,
                     "final_sl": final_sl,
                     "take_profit": take_profit,
                     "risk_R": risk_R,
                     "reward": reward,
                     "rr": rr,
                     "status": "OK",
                 })


def _mock_d14_fail(rr_val=1.6):
    """Mock D14 to return FAIL (R:R below 2.0)."""
    return patch("indicators.d15_decision.d14_rr_validation_fn",
                 return_value={
                     "direction": "LONG",
                     "entry_price": 103.5,
                     "final_sl": 97.5,
                     "take_profit": 109.1,
                     "risk_R": 6.0,
                     "reward": 3.6,
                     "rr": rr_val,
                     "status": "FAIL",
                 })


def _mock_d14_invalid_risk(risk_R_val=0.0):
    """Mock D14 to return INVALID_RISK (R=0)."""
    return patch("indicators.d15_decision.d14_rr_validation_fn",
                 return_value={
                     "direction": "LONG",
                     "entry_price": 100.0,
                     "final_sl": 100.0,
                     "take_profit": 100.0,
                     "risk_R": risk_R_val,
                     "reward": None,
                     "rr": None,
                     "status": "INVALID_RISK",
                 })


def _mock_d14_missing_data():
    """Mock D14 to return MISSING_DATA."""
    return patch("indicators.d15_decision.d14_rr_validation_fn",
                 return_value={
                     "direction": "LONG",
                     "entry_price": 103.5,
                     "final_sl": 97.5,
                     "take_profit": None,
                     "risk_R": None,
                     "reward": None,
                     "rr": None,
                     "status": "MISSING_DATA",
                 })


def _mock_d14_bad_dir():
    """Mock D14 to return BAD_DIR."""
    return patch("indicators.d15_decision.d14_rr_validation_fn",
                 return_value={
                     "direction": "INVALID",
                     "entry_price": None,
                     "final_sl": None,
                     "take_profit": None,
                     "risk_R": None,
                     "reward": None,
                     "rr": None,
                     "status": "BAD_DIR",
                 })


def _all_stages_ok_mock(direction="LONG", entry=103.5, signal_time=100,
                         final_sl=97.5, take_profit=115.5, rr=2.0,
                         final_long=80.0, final_short=20.0):
    """Context manager that mocks ALL stages to return OK results."""
    return patch.multiple(
        "indicators.d15_decision",
        d8_aggregate=return_value_value((final_long, final_short, direction)),
        _select_retest_level_by_direction=return_value_value(102.0),
        d7_pullback_pass=return_value_value(True),
        d10_entry_fn=return_value_value({
            "direction": direction,
            "entry": entry,
            "signal_time": signal_time,
            "status": "OK",
        }),
        d11_d12_sl_result_fn=return_value_value({
            "direction": direction,
            "selected_swing_price": 98.0,
            "atr14": 1.0,
            "buffer": 0.5,
            "final_sl": final_sl,
            "status": "OK",
        }),
        d13_take_profit_fn=return_value_value({
            "direction": direction,
            "entry_price": entry,
            "final_sl": final_sl,
            "risk_R": abs(entry - final_sl),
            "take_profit": take_profit,
            "status": "OK",
        }),
        d14_rr_validation_fn=return_value_value({
            "direction": direction,
            "entry_price": entry,
            "final_sl": final_sl,
            "take_profit": take_profit,
            "risk_R": abs(entry - final_sl),
            "reward": abs(take_profit - entry),
            "rr": rr,
            "status": "OK",
        }),
    )


def return_value_value(val):
    """Helper to create a patch that returns a fixed value."""
    return patch(__import__("unittest.mock").Mock(return_value=val))


class TestD15InputValidation(unittest.TestCase):
    """Test 13 — input validation: INVALID_INPUT for malformed caller input."""

    def test_candles_by_tf_not_dict(self):
        res = d15_final_decision(
            candles_by_tf=None,  # type: ignore
            candles_5m=[_k(0, 100, 101, 99, 100)],
            setup_index=0,
            setup_5m={"close": 100, "close_time": 0},
            candles_5m_before_setup=[],
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("candles_by_tf", res["reason"])

    def test_candles_5m_empty_list(self):
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=[],
            setup_index=0,
            setup_5m={"close": 100, "close_time": 0},
            candles_5m_before_setup=[],
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("candles_5m", res["reason"])

    def test_setup_index_negative(self):
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=[_k(0, 100, 101, 99, 100)],
            setup_index=-1,
            setup_5m={"close": 100, "close_time": 0},
            candles_5m_before_setup=[],
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("setup_index", res["reason"])

    def test_setup_index_out_of_range(self):
        candles = [_k(0, 100, 101, 99, 100)]
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=candles,
            setup_index=5,
            setup_5m={"close": 100, "close_time": 0},
            candles_5m_before_setup=[],
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("setup_index", res["reason"])

    def test_setup_5m_not_dict(self):
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=[_k(0, 100, 101, 99, 100)],
            setup_index=0,
            setup_5m="not_a_dict",  # type: ignore
            candles_5m_before_setup=[],
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("setup_5m", res["reason"])

    def test_setup_5m_missing_close(self):
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=[_k(0, 100, 101, 99, 100)],
            setup_index=0,
            setup_5m={"close_time": 0},
            candles_5m_before_setup=[],
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("setup_5m", res["reason"])

    def test_setup_5m_missing_close_time(self):
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=[_k(0, 100, 101, 99, 100)],
            setup_index=0,
            setup_5m={"close": 100},
            candles_5m_before_setup=[],
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("setup_5m", res["reason"])

    def test_candles_5m_before_setup_not_list(self):
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=[_k(0, 100, 101, 99, 100)],
            setup_index=0,
            setup_5m={"close": 100, "close_time": 0},
            candles_5m_before_setup="not_a_list",  # type: ignore
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("candles_5m_before_setup", res["reason"])

    def test_candles_5m_through_setup_not_list(self):
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=[_k(0, 100, 101, 99, 100)],
            setup_index=0,
            setup_5m={"close": 100, "close_time": 0},
            candles_5m_before_setup=[],
            candles_5m_through_setup="not_a_list",  # type: ignore
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("candles_5m_through_setup", res["reason"])

    def test_mismatched_close_setup_vs_series(self):
        candles = [_k(0, 100, 101, 99, 100.0)]
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=candles,
            setup_index=0,
            setup_5m={"close": 99.0, "close_time": 0},
            candles_5m_before_setup=[],
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("close", res["reason"])

    def test_mismatched_close_time_setup_vs_series(self):
        candles = [_k(0, 100, 101, 99, 100.0)]
        res = d15_final_decision(
            candles_by_tf={},
            candles_5m=candles,
            setup_index=0,
            setup_5m={"close": 100.0, "close_time": 999},
            candles_5m_before_setup=[],
            candles_5m_through_setup=[],
        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("close_time", res["reason"])


class TestD15NoExternals(unittest.TestCase):
    """Test 20 — D15 has no API/network/trading functionality."""

    def test_no_network_or_trading_imports(self):
        import indicators.d15_decision as d15
        source = Path(d15.__file__).read_text()
        import_lines = [
            l for l in source.split("\n")
            if l.strip().startswith("import ") or l.strip().startswith("from ")
        ]
        import_text = "\n".join(import_lines).lower()
        banned = ("requests", "urllib", "socket", "http.client", "binance", "telegram")
        for term in banned:
            self.assertNotIn(term, import_text,
                             f"D15 must not import '{term}'")

    def test_no_order_or_trading_logic_in_code(self):
        import indicators.d15_decision as d15
        source = Path(d15.__file__).read_text()
        code_start = source.find("def d15_final_decision")
        code = source[code_start:] if code_start >= 0 else source
        banned_in_code = (
            "place_order", "submit_order", "open_position", "execute_trade",
            "send_signal", "api.call", "requests.post", "http.post",
        )
        for term in banned_in_code:
            self.assertNotIn(term, code,
                             f"D15 code must not contain '{term}'")


class TestD15ValidLong(unittest.TestCase):
    """Test 1 — complete valid LONG → OK."""

    def test_long_all_gates_pass(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok(direction="LONG", entry=103.5, signal_time=100):
                    with _mock_d11d12_ok(direction="LONG", final_sl=97.5):
                        with _mock_d13_ok(direction="LONG", entry_price=103.5,
                                           final_sl=97.5, take_profit=115.5):
                            with _mock_d14_ok(direction="LONG",
                                               entry_price=103.5,
                                               final_sl=97.5,
                                               take_profit=115.5):
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["direction"], "LONG")
        self.assertEqual(res["entry"], 103.5)
        self.assertEqual(res["signal_time"], 100)
        self.assertEqual(res["final_sl"], 97.5)
        self.assertEqual(res["take_profit"], 115.5)
        self.assertEqual(res["rr"], 2.0)
        self.assertIsNone(res["reason"])
        for key in ("d10", "d11_d12", "d13", "d14"):
            self.assertEqual(res["stage_status"][key], "OK")
        self.assertEqual(res["stage_status"]["d8"]["direction"], "LONG")
        self.assertTrue(res["stage_status"]["d7"]["pass"])
        self.assertTrue(res["stage_status"]["d7"]["retest_available"])

    def test_long_final_scores_populated(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_ok():
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["final_long"], 80.0)
        self.assertEqual(res["final_short"], 20.0)


class TestD15ValidShort(unittest.TestCase):
    """Test 2 — complete valid SHORT → OK."""

    def test_short_all_gates_pass(self):
        with _mock_d8_short():
            with _d7_patches(p=True):
                with _mock_d10_ok(direction="SHORT", entry=96.5, signal_time=200):
                    with _mock_d11d12_ok(direction="SHORT", final_sl=102.5):
                        with _mock_d13_ok(direction="SHORT", entry_price=96.5,
                                           final_sl=102.5, take_profit=84.5):
                            with _mock_d14_ok(direction="SHORT",
                                               entry_price=96.5,
                                               final_sl=102.5,
                                               take_profit=84.5):
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["direction"], "SHORT")
        self.assertEqual(res["entry"], 96.5)
        self.assertEqual(res["signal_time"], 200)
        self.assertEqual(res["final_sl"], 102.5)
        self.assertEqual(res["take_profit"], 84.5)
        self.assertEqual(res["rr"], 2.0)
        self.assertIsNone(res["reason"])
        for key in ("d10", "d11_d12", "d13", "d14"):
            self.assertEqual(res["stage_status"][key], "OK")
        self.assertEqual(res["stage_status"]["d8"]["direction"], "SHORT")
        self.assertTrue(res["stage_status"]["d7"]["pass"])
        self.assertTrue(res["stage_status"]["d7"]["retest_available"])


class TestD15D8NoTrade(unittest.TestCase):
    """Test 3 — D8 returns NO_TRADE → D15 NO_TRADE (strategy failure)."""

    def test_d8_no_trade_propagates(self):
        with _mock_d8_no_trade():
            res = d15_final_decision(
                candles_by_tf={},
                candles_5m=[_k(0, 100, 101, 99, 100)],
                setup_index=0,
                setup_5m={"close": 100, "close_time": 0},
                candles_5m_before_setup=[],
                candles_5m_through_setup=[],
            )
        self.assertEqual(res["status"], "NO_TRADE")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D8", res["reason"])
        self.assertIn("threshold", res["reason"].lower())
        # Later stages not reached
        for key in ("d7", "d10", "d11_d12", "d13", "d14"):
            self.assertEqual(res["stage_status"][key], "SKIPPED")


class TestD15D7DataUnavailable(unittest.TestCase):
    """Test 4 — D7 retest level unavailable → DATA_UNAVAILABLE.

    Uses Option A: inspect _select_retest_level_by_direction before calling
    d7_pullback_pass. No D7 modification.
    """

    def test_d7_no_retest_level_data_unavailable(self):
        with _mock_d8_long():
            with _d7_patches(p=False, retest_available=False):
                res = d15_final_decision(
                    candles_by_tf={},
                    candles_5m=[_k(0, 100, 101, 99, 100)],
                    setup_index=0,
                    setup_5m={"close": 100, "close_time": 0},
                    candles_5m_before_setup=[],
                    candles_5m_through_setup=[],
                )
        self.assertEqual(res["status"], "DATA_UNAVAILABLE")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D7", res["reason"])
        self.assertIn("retest", res["reason"].lower())
        self.assertIn("data", res["reason"].lower())
        self.assertEqual(res["stage_status"]["d7"]["retest_available"], False)
        self.assertEqual(res["stage_status"]["d7"]["pass"], False)
        for key in ("d10", "d11_d12", "d13", "d14"):
            self.assertEqual(res["stage_status"][key], "SKIPPED")


class TestD15D7Fail(unittest.TestCase):
    """Test 5 — D7 pullback/rejection fails → NO_TRADE (strategy failure)."""

    def test_d7_pullback_fail_no_trade(self):
        with _mock_d8_long():
            with _d7_patches(p=False, retest_available=True):
                res = d15_final_decision(
                    candles_by_tf={},
                    candles_5m=[_k(0, 100, 101, 99, 100)],
                    setup_index=0,
                    setup_5m={"close": 100, "close_time": 0},
                    candles_5m_before_setup=[],
                    candles_5m_through_setup=[],
                )
        self.assertEqual(res["status"], "NO_TRADE")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D7", res["reason"])
        self.assertIn("pullback", res["reason"].lower())
        self.assertEqual(res["stage_status"]["d7"]["retest_available"], True)
        self.assertEqual(res["stage_status"]["d7"]["pass"], False)
        for key in ("d10", "d11_d12", "d13", "d14"):
            self.assertEqual(res["stage_status"][key], "SKIPPED")


class TestD15D10MissingData(unittest.TestCase):
    """Test 6 — D10 MISSING_DATA → DATA_UNAVAILABLE."""

    def test_d10_missing_data_propagates(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_missing_data():
                    res = d15_final_decision(
                        candles_by_tf={},
                        candles_5m=[_k(0, 100, 101, 99, 100)],
                        setup_index=0,
                        setup_5m={"close": 100, "close_time": 0},
                        candles_5m_before_setup=[],
                        candles_5m_through_setup=[],
                    )
        self.assertEqual(res["status"], "DATA_UNAVAILABLE")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D10", res["reason"])
        for key in ("d11_d12", "d13", "d14"):
            self.assertEqual(res["stage_status"][key], "SKIPPED")


class TestD15D10BadDir(unittest.TestCase):
    """Test 7a — D10 BAD_DIR → INVALID_INPUT."""

    def test_d10_bad_dir_propagates(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_bad_dir():
                    res = d15_final_decision(
                        candles_by_tf={},
                        candles_5m=[_k(0, 100, 101, 99, 100)],
                        setup_index=0,
                        setup_5m={"close": 100, "close_time": 0},
                        candles_5m_before_setup=[],
                        candles_5m_through_setup=[],
                    )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D10", res["reason"])
        self.assertIn("invalid direction", res["reason"].lower())


class TestD15D10Mismatch(unittest.TestCase):
    """Test 7b — D10 MISMATCH → NO_TRADE (shouldn't occur normally)."""

    def test_d10_mismatch_propagates(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_mismatch():
                    res = d15_final_decision(
                        candles_by_tf={},
                        candles_5m=[_k(0, 100, 101, 99, 100)],
                        setup_index=0,
                        setup_5m={"close": 100, "close_time": 0},
                        candles_5m_before_setup=[],
                        candles_5m_through_setup=[],
                    )
        self.assertEqual(res["status"], "NO_TRADE")
        self.assertIn("D10", res["reason"])


class TestD15D11D12MissingData(unittest.TestCase):
    """Test 8 — D11/D12 MISSING_DATA → DATA_UNAVAILABLE."""

    def test_d11_d12_missing_data_propagates(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_missing_data():
                        res = d15_final_decision(
                            candles_by_tf={},
                            candles_5m=[_k(0, 100, 101, 99, 100)],
                            setup_index=0,
                            setup_5m={"close": 100, "close_time": 0},
                            candles_5m_before_setup=[],
                            candles_5m_through_setup=[],
                        )
        self.assertEqual(res["status"], "DATA_UNAVAILABLE")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D11/D12", res["reason"])
        for key in ("d13", "d14"):
            self.assertEqual(res["stage_status"][key], "SKIPPED")


class TestD15D11D12BadGeometry(unittest.TestCase):
    """Test 9 — D11/D12 BAD_GEOMETRY → NO_TRADE (strategy failure)."""

    def test_d11_d12_bad_geometry_no_trade(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_bad_geometry():
                        res = d15_final_decision(
                            candles_by_tf={},
                            candles_5m=[_k(0, 100, 101, 99, 100)],
                            setup_index=0,
                            setup_5m={"close": 100, "close_time": 0},
                            candles_5m_before_setup=[],
                            candles_5m_through_setup=[],
                        )
        self.assertEqual(res["status"], "NO_TRADE")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D11/D12", res["reason"])
        self.assertIn("geometry", res["reason"].lower())
        for key in ("d13", "d14"):
            self.assertEqual(res["stage_status"][key], "SKIPPED")


class TestD15D11D12BadDir(unittest.TestCase):
    """Test 9b — D11/D12 BAD_DIR → INVALID_INPUT."""

    def test_d11_d12_bad_dir_propagates(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_bad_dir():
                        res = d15_final_decision(
                            candles_by_tf={},
                            candles_5m=[_k(0, 100, 101, 99, 100)],
                            setup_index=0,
                            setup_5m={"close": 100, "close_time": 0},
                            candles_5m_before_setup=[],
                            candles_5m_through_setup=[],
                        )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("D11/D12", res["reason"])


class TestD15D13MissingData(unittest.TestCase):
    """Test 10 — D13 MISSING_DATA → DATA_UNAVAILABLE."""

    def test_d13_missing_data_propagates(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_missing_data():
                            res = d15_final_decision(
                                candles_by_tf={},
                                candles_5m=[_k(0, 100, 101, 99, 100)],
                                setup_index=0,
                                setup_5m={"close": 100, "close_time": 0},
                                candles_5m_before_setup=[],
                                candles_5m_through_setup=[],
                            )
        self.assertEqual(res["status"], "DATA_UNAVAILABLE")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D13", res["reason"])
        self.assertEqual(res["stage_status"]["d14"], "SKIPPED")


class TestD15D14Fail(unittest.TestCase):
    """Test 11 — D14 FAIL (R:R < 2.0) → NO_TRADE (strategy failure)."""

    def test_d14_fail_no_trade(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_fail(rr_val=1.6):
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["status"], "NO_TRADE")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D14", res["reason"])
        self.assertIn("R:R", res["reason"])
        self.assertEqual(res["stage_status"]["d14"], "FAIL")


class TestD15D14InvalidRisk(unittest.TestCase):
    """Test 12 — D14 INVALID_RISK (R=0) → NO_TRADE (strategy failure)."""

    def test_d14_invalid_risk_no_trade(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok(entry=100.0):
                    with _mock_d11d12_ok(final_sl=100.0):
                        with _mock_d13_ok(entry_price=100.0, final_sl=100.0,
                                           take_profit=100.0, risk_R=0.0):
                            with _mock_d14_invalid_risk(risk_R_val=0.0):
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["status"], "NO_TRADE")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertIn("D14", res["reason"])
        self.assertIn("risk", res["reason"].lower())
        self.assertEqual(res["stage_status"]["d14"], "INVALID_RISK")


class TestD15D14MissingData(unittest.TestCase):
    """Test 10b — D14 MISSING_DATA → DATA_UNAVAILABLE."""

    def test_d14_missing_data_propagates(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_missing_data():
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["status"], "DATA_UNAVAILABLE")
        self.assertIn("D14", res["reason"])


class TestD15D14BadDir(unittest.TestCase):
    """Test 12b — D14 BAD_DIR → INVALID_INPUT."""

    def test_d14_bad_dir_propagates(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_bad_dir():
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["status"], "INVALID_INPUT")
        self.assertIn("D14", res["reason"])


class TestD15OutputFields(unittest.TestCase):
    """Tests 15, 16 — final output fields and stage_status present."""

    def test_all_output_fields_present_on_ok(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_ok():
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        required = {
            "direction", "entry", "signal_time", "final_sl", "take_profit",
            "final_long", "final_short", "rr", "status", "stage_status", "reason",
        }
        missing = required - res.keys()
        self.assertEqual(missing, set(), f"Missing fields: {missing}")

    def test_stage_status_all_six_stages(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_ok():
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        expected = {"d8", "d7", "d10", "d11_d12", "d13", "d14"}
        self.assertEqual(set(res["stage_status"].keys()), expected)

    def test_stage_status_d8_is_dict(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_ok():
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        d8_status = res["stage_status"]["d8"]
        self.assertIsInstance(d8_status, dict)
        self.assertEqual(d8_status["direction"], "LONG")
        self.assertEqual(d8_status["final_long"], 80.0)
        self.assertEqual(d8_status["final_short"], 20.0)

    def test_stage_status_d7_is_dict(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_ok():
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        d7_status = res["stage_status"]["d7"]
        self.assertIsInstance(d7_status, dict)
        self.assertTrue(d7_status["pass"])
        self.assertTrue(d7_status["retest_available"])

    def test_stage_status_other_stages_are_strings(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_ok():
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        for key in ("d10", "d11_d12", "d13", "d14"):
            self.assertIsInstance(res["stage_status"][key], str)
            self.assertEqual(res["stage_status"][key], "OK")


class TestD15EntrySlTpUnchanged(unittest.TestCase):
    """Test 17 — D15 does not modify Entry, SL, or TP.

    Verifies that D15 passes through the exact values from D10/D11/D12/D13
    without alteration.
    """

    def test_entry_equals_d10_entry(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok(entry=103.5, signal_time=100):
                    with _mock_d11d12_ok(final_sl=97.5):
                        with _mock_d13_ok(entry_price=103.5, final_sl=97.5,
                                           take_profit=115.5):
                            with _mock_d14_ok(entry_price=103.5, final_sl=97.5,
                                               take_profit=115.5):
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["entry"], 103.5)
        self.assertIsInstance(res["entry"], float)

    def test_sl_passed_through_unchanged(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok(entry=103.5):
                    with _mock_d11d12_ok(final_sl=97.5):
                        with _mock_d13_ok(entry_price=103.5, final_sl=97.5,
                                           take_profit=115.5):
                            with _mock_d14_ok(entry_price=103.5, final_sl=97.5,
                                               take_profit=115.5):
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["final_sl"], 97.5)
        self.assertLess(res["final_sl"], res["entry"])

    def test_tp_passed_through_unchanged(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok(entry=103.5):
                    with _mock_d11d12_ok(final_sl=97.5):
                        with _mock_d13_ok(entry_price=103.5, final_sl=97.5,
                                           take_profit=115.5):
                            with _mock_d14_ok(entry_price=103.5, final_sl=97.5,
                                               take_profit=115.5):
                                res = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res["take_profit"], 115.5)
        self.assertEqual(res["rr"], 2.0)


class TestD15NoDuplication(unittest.TestCase):
    """Test 18 — D15 does not duplicate D8–D14 formulas."""

    def test_d15_imports_stage_functions(self):
        import indicators.d15_decision as d15
        source = Path(d15.__file__).read_text()
        self.assertIn("from indicators.d8_aggregate import", source)
        self.assertIn("from indicators.d10_entry import", source)
        self.assertIn("from indicators.d11_d12_sl import", source)
        self.assertIn("from indicators.d13_take_profit import", source)
        self.assertIn("from indicators.d14_rr_validation import", source)
        self.assertIn("from indicators.pullback import", source)

    def test_d15_does_not_contain_scoring_formulas(self):
        import indicators.d15_decision as d15
        source = Path(d15.__file__).read_text()
        self.assertNotIn("CATEGORY_WEIGHTS", source)
        self.assertNotIn("TF_WEIGHTS", source)
        self.assertNotIn("DIRECTION_THRESHOLD", source)
        # Should not call component functions directly
        self.assertNotIn("trend_score(candles)", source)
        self.assertNotIn("compute_rsi", source)

    def test_d15_does_not_contain_tp_rr_formulas(self):
        import indicators.d15_decision as d15
        source = Path(d15.__file__).read_text()
        code_start = source.find("def d15_final_decision")
        code = source[code_start:] if code_start >= 0 else source
        self.assertNotIn("2.0 * R", code)
        self.assertNotIn("risk_R = abs", code)
        self.assertNotIn("reward = abs", code)
        self.assertNotIn("rr = reward / R", code)
        # Should only call stage functions, not compute these
        self.assertNotIn("abs(entry", code)
        self.assertNotIn("abs(tp", code)


class TestD15Deterministic(unittest.TestCase):
    """Test 19 — D15 deterministic behavior."""

    def test_same_inputs_same_output(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_ok():
                                res1 = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
                                res2 = d15_final_decision(
                                    candles_by_tf={},
                                    candles_5m=[_k(0, 100, 101, 99, 100)],
                                    setup_index=0,
                                    setup_5m={"close": 100, "close_time": 0},
                                    candles_5m_before_setup=[],
                                    candles_5m_through_setup=[],
                                )
        self.assertEqual(res1, res2)

    def test_ten_calls_identical(self):
        results = []
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_ok():
                        with _mock_d13_ok():
                            with _mock_d14_ok():
                                for _ in range(10):
                                    res = d15_final_decision(
                                        candles_by_tf={},
                                        candles_5m=[_k(0, 100, 101, 99, 100)],
                                        setup_index=0,
                                        setup_5m={"close": 100, "close_time": 0},
                                        candles_5m_before_setup=[],
                                        candles_5m_through_setup=[],
                                    )
                                    results.append(res)
        first = results[0]
        for r in results[1:]:
            self.assertEqual(r, first)

    def test_deterministic_on_no_trade(self):
        results = []
        with _mock_d8_no_trade():
            for _ in range(5):
                res = d15_final_decision(
                    candles_by_tf={},
                    candles_5m=[_k(0, 100, 101, 99, 100)],
                    setup_index=0,
                    setup_5m={"close": 100, "close_time": 0},
                    candles_5m_before_setup=[],
                    candles_5m_through_setup=[],
                )
                results.append(res)
        first = results[0]
        for r in results[1:]:
            self.assertEqual(r, first)


class TestD15NoLookAhead(unittest.TestCase):
    """Test 21 — D15 does not use future data or external state."""

    def test_d15_code_no_time_random_datetime(self):
        import indicators.d15_decision as d15
        source = Path(d15.__file__).read_text()
        code_start = source.find("def d15_final_decision")
        code = source[code_start:] if code_start >= 0 else source
        self.assertNotIn("import time", code)
        self.assertNotIn("import random", code)
        self.assertNotIn("import datetime", code)
        self.assertNotIn("time.time()", code)
        self.assertNotIn("random.", code)

    def test_d15_uses_only_provided_args(self):
        import indicators.d15_decision as d15
        source = Path(d15.__file__).read_text()
        code_start = source.find("def d15_final_decision")
        code = source[code_start:] if code_start >= 0 else source
        suspicious = ["open(", "eval(", "exec(", "__import__", "globals()", "locals()"]
        for term in suspicious:
            self.assertNotIn(term, code, f"D15 should not contain '{term}'")


class TestD15StageStatusSkips(unittest.TestCase):
    """Verify that stages not reached get SKIPPED status."""

    def test_d7_skipped_when_d8_no_trade(self):
        with _mock_d8_no_trade():
            res = d15_final_decision(
                candles_by_tf={},
                candles_5m=[_k(0, 100, 101, 99, 100)],
                setup_index=0,
                setup_5m={"close": 100, "close_time": 0},
                candles_5m_before_setup=[],
                candles_5m_through_setup=[],
            )
        self.assertEqual(res["stage_status"]["d7"], "SKIPPED")
        self.assertEqual(res["stage_status"]["d10"], "SKIPPED")
        self.assertEqual(res["stage_status"]["d11_d12"], "SKIPPED")
        self.assertEqual(res["stage_status"]["d13"], "SKIPPED")
        self.assertEqual(res["stage_status"]["d14"], "SKIPPED")

    def test_d10_skipped_when_d7_data_unavailable(self):
        with _mock_d8_long():
            with _d7_patches(p=False, retest_available=False):
                res = d15_final_decision(
                    candles_by_tf={},
                    candles_5m=[_k(0, 100, 101, 99, 100)],
                    setup_index=0,
                    setup_5m={"close": 100, "close_time": 0},
                    candles_5m_before_setup=[],
                    candles_5m_through_setup=[],
                )
        self.assertEqual(res["stage_status"]["d10"], "SKIPPED")
        self.assertEqual(res["stage_status"]["d11_d12"], "SKIPPED")
        self.assertEqual(res["stage_status"]["d13"], "SKIPPED")
        self.assertEqual(res["stage_status"]["d14"], "SKIPPED")

    def test_d13_d14_skipped_when_d11_d12_fails(self):
        with _mock_d8_long():
            with _d7_patches(p=True):
                with _mock_d10_ok():
                    with _mock_d11d12_bad_geometry():
                        res = d15_final_decision(
                            candles_by_tf={},
                            candles_5m=[_k(0, 100, 101, 99, 100)],
                            setup_index=0,
                            setup_5m={"close": 100, "close_time": 0},
                            candles_5m_before_setup=[],
                            candles_5m_through_setup=[],
                        )
        self.assertEqual(res["stage_status"]["d13"], "SKIPPED")
        self.assertEqual(res["stage_status"]["d14"], "SKIPPED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
