"""Tests for D10 Entry — indicators.d10_entry."""

from __future__ import annotations
import unittest
from indicators.d10_entry import d10_entry


class TestD10Entry(unittest.TestCase):
    """Verify D10 entry logic per approved rules."""

    def setUp(self):
        self.setup_5m = {
            "close": 100.5,
            "close_time": 1005000,  # ms
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "volume": 1.0,
        }

    # LONG
    def test_long_entry_ok(self):
        """LONG with D7 pass -> entry = setup close, time = setup close_time."""
        res = d10_entry("LONG", True, self.setup_5m, "LONG")
        self.assertEqual(res["direction"], "LONG")
        self.assertEqual(res["entry"], 100.5)
        self.assertEqual(res["signal_time"], 1005000)
        self.assertEqual(res["status"], "OK")

    # SHORT
    def test_short_entry_ok(self):
        """SHORT with D7 pass -> entry = setup close, time = setup close_time."""
        res = d10_entry("SHORT", True, self.setup_5m, "SHORT")
        self.assertEqual(res["direction"], "SHORT")
        self.assertEqual(res["entry"], 100.5)
        self.assertEqual(res["signal_time"], 1005000)
        self.assertEqual(res["status"], "OK")

    # D7/D8 matching
    def test_d7_d8_match_long(self):
        """D7=LONG, D8=LONG -> OK."""
        res = d10_entry("LONG", True, self.setup_5m, "LONG")
        self.assertEqual(res["status"], "OK")

    def test_d7_d8_match_short(self):
        """D7=SHORT, D8=SHORT -> OK."""
        res = d10_entry("SHORT", True, self.setup_5m, "SHORT")
        self.assertEqual(res["status"], "OK")

    # D7/D8 mismatch
    def test_d7_d8_mismatch_long_short(self):
        """D7=LONG, D8=SHORT -> NO_TRADE MISMATCH."""
        res = d10_entry("SHORT", True, self.setup_5m, "LONG")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertEqual(res["status"], "MISMATCH")

    def test_d7_d8_mismatch_short_long(self):
        """D7=SHORT, D8=LONG -> NO_TRADE MISMATCH."""
        res = d10_entry("LONG", True, self.setup_5m, "SHORT")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertEqual(res["status"], "MISMATCH")

    def test_d7_none_mismatch(self):
        """D7 direction None -> NO_TRADE MISMATCH."""
        res = d10_entry("LONG", True, self.setup_5m, None)
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertEqual(res["status"], "MISMATCH")

    # Exact setup-candle close entry
    def test_entry_equals_setup_close(self):
        """Entry price exactly equals setup_5m['close']."""
        self.setup_5m["close"] = 99.12345
        res = d10_entry("LONG", True, self.setup_5m, "LONG")
        self.assertEqual(res["entry"], 99.12345)

    # Setup timestamp
    def test_signal_time_equals_setup_close_time(self):
        """signal_time exactly equals setup_5m['close_time']."""
        self.setup_5m["close_time"] = 1234567890
        res = d10_entry("LONG", True, self.setup_5m, "LONG")
        self.assertEqual(res["signal_time"], 1234567890)

    # Forming-candle exclusion (caller responsibility, but D10 uses only passed data)
    def test_forming_candle_not_used(self):
        """D10 uses passed setup_5m as-is; caller must ensure closed."""
        # D10 does not validate close_time; it trusts the caller's closed-candle filter.
        res = d10_entry("LONG", True, self.setup_5m, "LONG")
        self.assertEqual(res["status"], "OK")

    # No look-ahead: only setup candle data used
    def test_no_lookahead(self):
        """D10 output depends only on inputs; no future data referenced."""
        res1 = d10_entry("LONG", True, self.setup_5m, "LONG")
        res2 = d10_entry("LONG", True, self.setup_5m, "LONG")
        self.assertEqual(res1, res2)

    # Missing/stale required data -> handled by caller; D10 rejects malformed
    def test_missing_close_fails_rejected(self):
        """If setup_5m missing 'close', D10 returns MISSING_DATA."""
        bad = {**self.setup_5m}
        del bad["close"]
        res = d10_entry("LONG", True, bad, "LONG")
        self.assertIsNone(res["entry"])
        self.assertEqual(res["status"], "MISSING_DATA")

    # Deterministic output
    def test_deterministic(self):
        """Same inputs -> same outputs."""
        for _ in range(5):
            res = d10_entry("LONG", True, self.setup_5m, "LONG")
            self.assertEqual(res["direction"], "LONG")
            self.assertEqual(res["entry"], 100.5)
            self.assertEqual(res["signal_time"], 1005000)

    # Boundary/equality
    def test_d8_no_trade_rejected(self):
        """D8 = NO_TRADE -> NO_TRADE MISMATCH."""
        res = d10_entry("NO_TRADE", True, self.setup_5m, "LONG")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertEqual(res["status"], "MISMATCH")

    def test_d7_fail_rejected(self):
        """D7 pass=False -> NO_TRADE D7_FAIL."""
        res = d10_entry("LONG", False, self.setup_5m, "LONG")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertEqual(res["status"], "D7_FAIL")

    def test_bad_d8_dir_rejected(self):
        """D8 dir not LONG/SHORT/NO_TRADE -> NO_TRADE BAD_DIR."""
        res = d10_entry("INVALID", True, self.setup_5m, "INVALID")
        self.assertEqual(res["direction"], "NO_TRADE")
        self.assertEqual(res["status"], "BAD_DIR")


if __name__ == "__main__":
    unittest.main()