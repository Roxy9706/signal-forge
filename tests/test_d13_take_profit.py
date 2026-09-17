"""Tests for D13 Take Profit — indicators.d13_take_profit.

Focused tests covering valid LONG/SHORT TP, R calculation, exact 2R distance,
missing/invalid entry, missing/invalid SL, invalid direction, invalid geometry,
output fields, and purity (no look-ahead / no external functionality).
"""

from __future__ import annotations
import unittest
from indicators.d13_take_profit import d13_take_profit


class TestD13TakeProfit(unittest.TestCase):
    """Verify D13 take-profit logic per approved rules."""

    # --- Valid LONG TP ---
    def test_long_tp_valid(self):
        res = d13_take_profit("LONG", 100.0, 95.0)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["direction"], "LONG")
        self.assertEqual(res["entry_price"], 100.0)
        self.assertEqual(res["final_sl"], 95.0)
        # R = abs(100 - 95) = 5.0; TP = 100 + 2*5 = 110.0
        self.assertEqual(res["risk_R"], 5.0)
        self.assertEqual(res["take_profit"], 110.0)

    def test_long_tp_entry_above_sl(self):
        # Entry above SL is the normal LONG setup
        res = d13_take_profit("LONG", 250.75, 240.25)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["entry_price"], 250.75)
        self.assertEqual(res["final_sl"], 240.25)
        # R = 10.5; TP = 250.75 + 21.0
        self.assertEqual(res["risk_R"], 10.5)
        self.assertEqual(res["take_profit"], 271.75)

    # --- Valid SHORT TP ---
    def test_short_tp_valid(self):
        res = d13_take_profit("SHORT", 100.0, 105.0)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["direction"], "SHORT")
        self.assertEqual(res["entry_price"], 100.0)
        self.assertEqual(res["final_sl"], 105.0)
        # R = abs(100 - 105) = 5.0; TP = 100 - 2*5 = 90.0
        self.assertEqual(res["risk_R"], 5.0)
        self.assertEqual(res["take_profit"], 90.0)

    def test_short_tp_entry_below_sl(self):
        # Entry below SL is the normal SHORT setup
        res = d13_take_profit("SHORT", 300.0, 310.0)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["risk_R"], 10.0)
        self.assertEqual(res["take_profit"], 280.0)

    # --- Correct R calculation ---
    def test_risk_r_is_abs_entry_minus_sl(self):
        # LONG: entry above sl
        r1 = d13_take_profit("LONG", 100.0, 90.0)["risk_R"]
        self.assertEqual(r1, 10.0)
        # SHORT: entry below sl
        r2 = d13_take_profit("SHORT", 100.0, 110.0)["risk_R"]
        self.assertEqual(r2, 10.0)
        # Equal entry and sl -> R == 0 (edge case, allowed by spec)
        r3 = d13_take_profit("LONG", 100.0, 100.0)["risk_R"]
        self.assertEqual(r3, 0.0)

    def test_risk_r_is_positive_or_zero(self):
        res = d13_take_profit("LONG", 50.0, 40.0)
        self.assertGreaterEqual(res["risk_R"], 0.0)

    # --- Exact 2R distance ---
    def test_tp_is_exactly_2r_from_entry_long(self):
        res = d13_take_profit("LONG", 100.0, 95.0)
        # TP - entry should equal 2 * R
        self.assertEqual(res["take_profit"] - res["entry_price"],
                         2.0 * res["risk_R"])

    def test_tp_is_exactly_2r_from_entry_short(self):
        res = d13_take_profit("SHORT", 100.0, 105.0)
        # entry - TP should equal 2 * R (TP is below entry for SHORT)
        self.assertEqual(res["entry_price"] - res["take_profit"],
                         2.0 * res["risk_R"])

    def test_tp_distance_independent_of_sign(self):
        # Different entry/SL combos with same R produce same 2R TP distance
        a = d13_take_profit("LONG", 200.0, 180.0)
        b = d13_take_profit("SHORT", 200.0, 220.0)
        self.assertEqual(a["risk_R"], b["risk_R"])
        self.assertEqual(a["take_profit"] - a["entry_price"],
                         2.0 * a["risk_R"])
        self.assertEqual(b["entry_price"] - b["take_profit"],
                         2.0 * b["risk_R"])

    # --- Missing / invalid entry ---
    def test_missing_entry_returns_missing_data(self):
        res = d13_take_profit("LONG", None, 95.0)
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertIsNone(res["risk_R"])
        self.assertIsNone(res["take_profit"])

    def test_missing_entry_price_field(self):
        # Explicitly None entry
        res = d13_take_profit("SHORT", None, 105.0)
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertEqual(res["direction"], "SHORT")
        self.assertIsNone(res["entry_price"])

    # --- Missing / invalid SL ---
    def test_missing_sl_returns_missing_data(self):
        res = d13_take_profit("LONG", 100.0, None)
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertIsNone(res["risk_R"])
        self.assertIsNone(res["take_profit"])

    def test_missing_final_sl_field(self):
        res = d13_take_profit("SHORT", 100.0, None)
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertEqual(res["direction"], "SHORT")
        self.assertIsNone(res["final_sl"])

    def test_both_missing_returns_missing_data(self):
        res = d13_take_profit("LONG", None, None)
        self.assertEqual(res["status"], "MISSING_DATA")
        self.assertIsNone(res["risk_R"])
        self.assertIsNone(res["take_profit"])

    # --- Invalid direction ---
    def test_invalid_direction_returns_bad_dir(self):
        res = d13_take_profit("INVALID", 100.0, 95.0)
        self.assertEqual(res["status"], "BAD_DIR")
        self.assertIsNone(res["risk_R"])
        self.assertIsNone(res["take_profit"])

    def test_no_trade_direction_returns_bad_dir(self):
        # D13 does not special-case NO_TRADE; only LONG/SHORT accepted
        res = d13_take_profit("NO_TRADE", 100.0, 95.0)
        self.assertEqual(res["status"], "BAD_DIR")
        self.assertIsNone(res["risk_R"])

    def test_empty_string_direction_returns_bad_dir(self):
        res = d13_take_profit("", 100.0, 95.0)
        self.assertEqual(res["status"], "BAD_DIR")

    def test_case_sensitive_direction_rejected(self):
        # Only uppercase LONG/SHORT accepted per spec
        res = d13_take_profit("long", 100.0, 95.0)
        self.assertEqual(res["status"], "BAD_DIR")

    # --- Invalid geometry ---
    def test_zero_risk_produces_zero_tp_distance(self):
        # Entry == SL -> R == 0 -> TP == entry (still "OK", not an error)
        res = d13_take_profit("LONG", 100.0, 100.0)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["risk_R"], 0.0)
        self.assertEqual(res["take_profit"], 100.0)

    def test_very_small_risk_is_handled(self):
        res = d13_take_profit("LONG", 100.0, 99.999)
        self.assertEqual(res["status"], "OK")
        self.assertAlmostEqual(res["risk_R"], 0.001, places=6)
        self.assertAlmostEqual(res["take_profit"], 100.002, places=6)

    # --- Output fields ---
    def test_output_has_all_required_fields(self):
        res = d13_take_profit("LONG", 100.0, 95.0)
        required = {"direction", "entry_price", "final_sl",
                    "risk_R", "take_profit", "status"}
        self.assertTrue(required.issubset(res.keys()))

    def test_output_types_on_ok(self):
        res = d13_take_profit("LONG", 100.0, 95.0)
        self.assertIsInstance(res["entry_price"], float)
        self.assertIsInstance(res["final_sl"], float)
        self.assertIsInstance(res["risk_R"], float)
        self.assertIsInstance(res["take_profit"], float)
        self.assertIsInstance(res["status"], str)
        self.assertIsInstance(res["direction"], str)

    def test_output_types_on_error(self):
        for status in ("BAD_DIR", "MISSING_DATA"):
            with self.subTest(status=status):
                kwargs = {"direction": "LONG", "entry_price": 100.0,
                          "final_sl": 95.0}
                if status == "BAD_DIR":
                    kwargs["direction"] = "INVALID"
                else:
                    kwargs["entry_price"] = None
                res = d13_take_profit(**kwargs)
                self.assertEqual(res["status"], status)
                self.assertIsNone(res["risk_R"])
                self.assertIsNone(res["take_profit"])

    # --- Purity: no look-ahead, no external functionality ---
    def test_pure_determinism_same_inputs_same_output(self):
        inputs = ("LONG", 100.0, 95.0)
        r1 = d13_take_profit(*inputs)
        r2 = d13_take_profit(*inputs)
        self.assertEqual(r1, r2)

    def test_does_not_depend_on_system_state(self):
        # Repeated calls with same args are identical (no time/rand/network)
        res = [d13_take_profit("LONG", 100.0, 95.0) for _ in range(10)]
        first = res[0]
        for r in res[1:]:
            self.assertEqual(r, first)

    def test_no_lookahead_only_inputs_used(self):
        # Output depends only on the three supplied arguments.
        entry = 100.0
        sl = 95.0
        res = d13_take_profit("LONG", entry, sl)
        # TP is a pure function of entry and sl; no future data involved.
        self.assertEqual(res["take_profit"], entry + 2.0 * abs(entry - sl))

    def test_short_is_mirror_of_long(self):
        # For same |entry-sl|, LONG TP and SHORT TP are symmetric around entry.
        entry = 100.0
        sl_low = 95.0   # LONG: sl below entry
        sl_high = 105.0  # SHORT: sl above entry
        long_res = d13_take_profit("LONG", entry, sl_low)
        short_res = d13_take_profit("SHORT", entry, sl_high)
        self.assertEqual(long_res["risk_R"], short_res["risk_R"])
        # LONG TP above entry; SHORT TP below entry by same amount
        self.assertEqual(long_res["take_profit"] - entry,
                         entry - short_res["take_profit"])


if __name__ == "__main__":
    unittest.main()
