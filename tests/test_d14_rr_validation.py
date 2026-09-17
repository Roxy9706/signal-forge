"""Tests for D14 R:R Validation - indicators.d14_rr_validation."""
from __future__ import annotations
import unittest
from indicators.d14_rr_validation import d14_rr_validation


class TestD14RrValidation(unittest.TestCase):
    """Verify D14 R:R validation logic per approved rules."""

    # --- LONG with exactly 2.0 R:R -> PASS ---
    def test_long_exact_2r_pass(self):
        res = d14_rr_validation("LONG", 100.0, 95.0, 110.0)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["direction"], "LONG")
        self.assertEqual(res["risk_R"], 5.0)
        self.assertEqual(res["reward"], 10.0)
        self.assertEqual(res["rr"], 2.0)

    # --- SHORT with exactly 2.0 R:R -> PASS ---
    def test_short_exact_2r_pass(self):
        res = d14_rr_validation("SHORT", 100.0, 105.0, 90.0)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["direction"], "SHORT")
        self.assertEqual(res["risk_R"], 5.0)
        self.assertEqual(res["reward"], 10.0)
        self.assertEqual(res["rr"], 2.0)

    # --- R:R greater than 2.0 -> PASS ---
    def test_long_above_2r_pass(self):
        res = d14_rr_validation("LONG", 100.0, 95.0, 115.0)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["rr"], 3.0)

    def test_short_above_2r_pass(self):
        res = d14_rr_validation("SHORT", 100.0, 108.0, 80.0)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["rr"], 2.5)

    # --- R:R below 2.0 -> FAIL ---
    def test_long_below_2r_fail(self):
        res = d14_rr_validation("LONG", 100.0, 95.0, 108.0)
        self.assertEqual(res["status"], "FAIL")
        self.assertEqual(res["rr"], 1.6)

    def test_short_below_2r_fail(self):
        res = d14_rr_validation("SHORT", 100.0, 105.0, 94.0)
        self.assertEqual(res["status"], "FAIL")
        self.assertAlmostEqual(res["rr"], 1.2)

    # --- Independent R:R calculation ---
    def test_independent_rr_not_based_on_2r(self):
        entry, sl, tp = 100.0, 90.0, 130.0
        res = d14_rr_validation("LONG", entry, sl, tp)
        self.assertEqual(res["rr"], 3.0)
        self.assertEqual(res["reward"], 30.0)
        self.assertEqual(res["status"], "OK")

    def test_rr_with_non_2r_tp_short(self):
        res = d14_rr_validation("SHORT", 100.0, 110.0, 75.0)
        self.assertEqual(res["rr"], 2.5)
        self.assertEqual(res["reward"], 25.0)
        self.assertEqual(res["status"], "OK")

    # --- Zero-risk / entry == SL -> INVALID_RISK ---
    def test_zero_risk_invalid_risk_long(self):
        res = d14_rr_validation("LONG", 100.0, 100.0, 110.0)
        self.assertEqual(res["status"], "INVALID_RISK")
        self.assertEqual(res["risk_R"], 0.0)
        self.assertIsNone(res["reward"])
        self.assertIsNone(res["rr"])

    def test_zero_risk_invalid_risk_short(self):
        res = d14_rr_validation("SHORT", 100.0, 100.0, 90.0)
        self.assertEqual(res["status"], "INVALID_RISK")
        self.assertEqual(res["risk_R"], 0.0)

    # --- Missing inputs -> MISSING_DATA ---
    def test_missing_entry(self):
        res = d14_rr_validation("LONG", None, 95.0, 110.0)
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertIsNone(res["risk_R"])

    def test_missing_sl(self):
        res = d14_rr_validation("LONG", 100.0, None, 110.0)
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertIsNone(res["risk_R"])

    def test_missing_tp(self):
        res = d14_rr_validation("LONG", 100.0, 95.0, None)
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertIsNone(res["risk_R"])

    # --- Invalid direction -> BAD_DIR ---
    def test_invalid_direction(self):
        res = d14_rr_validation("INVALID", 100.0, 95.0, 110.0)
        self.assertEqual(res["status"], "BAD_DIR")
        self.assertIsNone(res["risk_R"])

    def test_no_trade_direction(self):
        res = d14_rr_validation("NO_TRADE", 100.0, 95.0, 110.0)
        self.assertEqual(res["status"], "BAD_DIR")

    def test_empty_direction(self):
        res = d14_rr_validation("", 100.0, 95.0, 110.0)
        self.assertEqual(res["status"], "BAD_DIR")

    def test_lowercase_direction(self):
        res = d14_rr_validation("long", 100.0, 95.0, 110.0)
        self.assertEqual(res["status"], "BAD_DIR")

    # --- Output fields ---
    def test_output_has_all_fields(self):
        res = d14_rr_validation("LONG", 100.0, 95.0, 110.0)
        required = {"direction", "entry_price", "final_sl", "take_profit",
                    "risk_R", "reward", "rr", "status"}
        self.assertTrue(required.issubset(res.keys()))

    def test_output_types_on_ok(self):
        res = d14_rr_validation("LONG", 100.0, 95.0, 110.0)
        for key in ("direction", "status"):
            self.assertIsInstance(res[key], str)
        for key in ("entry_price", "final_sl", "take_profit", "risk_R", "reward", "rr"):
            self.assertIsInstance(res[key], float)

    def test_output_types_on_fail(self):
        res = d14_rr_validation("LONG", 100.0, 95.0, 108.0)
        self.assertEqual(res["status"], "FAIL")
        self.assertIsInstance(res["rr"], float)
        self.assertIsInstance(res["risk_R"], float)

    def test_output_types_on_invalid_risk(self):
        res = d14_rr_validation("LONG", 100.0, 100.0, 110.0)
        self.assertEqual(res["status"], "INVALID_RISK")
        self.assertIsInstance(res["risk_R"], float)
        self.assertIsNone(res["reward"])
        self.assertIsNone(res["rr"])

    # --- Read-only: entry/SL/TP unchanged ---
    def test_inputs_unchanged_on_ok(self):
        entry, sl, tp = 100.0, 95.0, 110.0
        res = d14_rr_validation("LONG", entry, sl, tp)
        self.assertEqual(res["entry_price"], entry)
        self.assertEqual(res["final_sl"], sl)
        self.assertEqual(res["take_profit"], tp)

    def test_inputs_unchanged_on_fail(self):
        entry, sl, tp = 100.0, 95.0, 108.0
        res = d14_rr_validation("LONG", entry, sl, tp)
        self.assertEqual(res["entry_price"], entry)
        self.assertEqual(res["final_sl"], sl)
        self.assertEqual(res["take_profit"], tp)

    def test_inputs_unchanged_on_invalid_risk(self):
        entry, sl, tp = 100.0, 100.0, 110.0
        res = d14_rr_validation("LONG", entry, sl, tp)
        self.assertEqual(res["entry_price"], entry)
        self.assertEqual(res["final_sl"], sl)
        self.assertEqual(res["take_profit"], tp)

    # --- Deterministic ---
    def test_deterministic_same_inputs(self):
        inputs = ("LONG", 100.0, 95.0, 110.0)
        self.assertEqual(d14_rr_validation(*inputs),
                         d14_rr_validation(*inputs))

    def test_deterministic_repeated_calls(self):
        res = [d14_rr_validation("LONG", 100.0, 95.0, 110.0) for _ in range(10)]
        first = res[0]
        for r in res[1:]:
            self.assertEqual(r, first)

    # --- Pure calculation, no externals ---
    def test_pure_calculation(self):
        entry, sl, tp = 100.0, 95.0, 110.0
        res = d14_rr_validation("LONG", entry, sl, tp)
        self.assertEqual(res["risk_R"], abs(entry - sl))
        self.assertEqual(res["reward"], abs(tp - entry))
        self.assertAlmostEqual(res["rr"], res["reward"] / res["risk_R"])


if __name__ == "__main__":
    unittest.main()
