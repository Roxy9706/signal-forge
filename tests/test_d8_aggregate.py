"""Tests for D8 multi-timeframe aggregation — aggregate() direction gate.

These tests target the aggregation logic specifically:
  - category weights (40/20/20/20)
  - timeframe weights (25/25/20/20/10)
  - final LONG/SHORT weighted sums across 5 TFs
  - direction gate (>=65 AND strictly greater)
  - 1-decimal rounding
  - missing/empty TF handling (50 neutral per existing implementation)
  - deterministic output

The per-timeframe component functions in indicators.d8_aggregate are mocked
so each TF can be assigned deterministic composite scores without re-testing
Trend/RSI/Structure/Volume internals.
"""

from __future__ import annotations
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import indicators.d8_aggregate as d8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candles(n, close=100.0):
    """Return a list of n closed-candle dicts (minimal, enough for any component)."""
    return [{"close": close, "high": close + 1, "low": close - 1,
             "open": close, "volume": 1.0, "close_time": i * 300000}
            for i in range(n)]


def _all_tfs():
    """Return a candles_by_tf dict with all 5 TFs present."""
    return {tf: _candles(250) for tf in d8.TF_ORDER}


class _CycleMock:
    """A callable that returns canned (tf_long, tf_short) pairs in sequence,
    cycling when exhausted so it never raises StopIteration."""
    def __init__(self, values):
        self.values = list(values) or [(50.0, 50.0)]
        self.idx = 0

    def __call__(self, candles):
        val = self.values[self.idx % len(self.values)]
        self.idx += 1
        return val


def _mock_components(trend_long, trend_short,
                     mom_long, mom_short,
                     struct_long, struct_short,
                     vol):
    """Return an ExitStack that makes d8._tf_scores use the supplied
    component values.  trend_score, momentum_score, structure_score, and
    volume_score are replaced; compute_rsi/compute_rvol are stubbed so
    momentum_score still receives *something*."""
    stack = ExitStack()
    stack.enter_context(patch.object(d8, "trend_score",
                                      return_value=(trend_long, trend_short)))
    stack.enter_context(patch.object(d8, "momentum_score",
                                    side_effect=lambda rsi, direction:
                                    mom_long if direction == "LONG" else mom_short))
    stack.enter_context(patch.object(d8, "structure_score",
                                      side_effect=lambda c, direction:
                                      struct_long if direction == "LONG" else struct_short))
    stack.enter_context(patch.object(d8, "compute_rsi", return_value=50.0))
    stack.enter_context(patch.object(d8, "compute_rvol", return_value=None))
    stack.enter_context(patch.object(d8, "volume_score", return_value=vol))
    return stack


# ---------------------------------------------------------------------------
# 1. Category weights 40/20/20/20
# ---------------------------------------------------------------------------

class TestD8CategoryWeights(unittest.TestCase):
    def test_category_weights_match_spec(self):
        self.assertAlmostEqual(d8.CATEGORY_WEIGHTS["trend"], 0.40)
        self.assertAlmostEqual(d8.CATEGORY_WEIGHTS["momentum"], 0.20)
        self.assertAlmostEqual(d8.CATEGORY_WEIGHTS["structure"], 0.20)
        self.assertAlmostEqual(d8.CATEGORY_WEIGHTS["volume"], 0.20)

    def test_category_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(d8.CATEGORY_WEIGHTS.values()), 1.0)


# ---------------------------------------------------------------------------
# 2. Timeframe weights 25/25/20/20/10
# ---------------------------------------------------------------------------

class TestD8TimeframeWeights(unittest.TestCase):
    def test_timeframe_weights_match_spec(self):
        self.assertAlmostEqual(d8.TF_WEIGHTS["1d"], 0.25)
        self.assertAlmostEqual(d8.TF_WEIGHTS["4h"], 0.25)
        self.assertAlmostEqual(d8.TF_WEIGHTS["1h"], 0.20)
        self.assertAlmostEqual(d8.TF_WEIGHTS["15m"], 0.20)
        self.assertAlmostEqual(d8.TF_WEIGHTS["5m"], 0.10)

    def test_timeframe_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(d8.TF_WEIGHTS.values()), 1.0)

    def test_tf_order(self):
        self.assertEqual(d8.TF_ORDER, ["1d", "4h", "1h", "15m", "5m"])


# ---------------------------------------------------------------------------
# 3-5. Single-TF composite: LONG, SHORT, volume shared by both directions
# ---------------------------------------------------------------------------

class TestD8SingleTimeframeComposite(unittest.TestCase):
    """Verify _tf_scores applies category weights correctly.

    Mock the individual component functions so we control each component
    independently; _tf_scores then applies 40/20/20/20.
    """

    def _composite(self, trend_long, trend_short,
                   mom_long, mom_short,
                   struct_long, struct_short,
                   vol):
        """Return (tf_long, tf_short) using the approved formula."""
        tf_long = (trend_long * 0.40 + mom_long * 0.20
                   + struct_long * 0.20 + vol * 0.20)
        tf_short = (trend_short * 0.40 + mom_short * 0.20
                    + struct_short * 0.20 + vol * 0.20)
        return tf_long, tf_short

    def test_tf_long_formula_max_bullish(self):
        """All components 100 -> tf_long = 100."""
        with _mock_components(100, 0, 100, 0, 100, 0, 100):
            tf_long, tf_short = d8._tf_scores(_candles(250))
        self.assertAlmostEqual(tf_long, 100.0)
        self.assertAlmostEqual(tf_short, 20.0)

    def test_tf_formula_mixed_components(self):
        """80/60/70/50 long, 20/40/30/50 short."""
        with _mock_components(80, 20, 60, 40, 70, 30, 50):
            tf_long, tf_short = d8._tf_scores(_candles(250))
        exp_long, exp_short = self._composite(80, 20, 60, 40, 70, 30, 50)
        self.assertAlmostEqual(tf_long, exp_long)
        self.assertAlmostEqual(tf_short, exp_short)

    def test_volume_reused_identically_for_long_and_short(self):
        """Volume contributes exactly the same amount to tf_long and tf_short."""
        with _mock_components(100, 0, 100, 0, 100, 0, 80):
            long_a, short_a = d8._tf_scores(_candles(250))
        with _mock_components(0, 100, 0, 100, 0, 100, 80):
            long_b, short_b = d8._tf_scores(_candles(250))
        # long_a (trend/mom/struct=100) vs long_b (trend/mom/struct=0):
        # difference = 80, purely from trend/mom/struct, vol identical.
        self.assertAlmostEqual(long_a - long_b, 80.0)
        # short_b (trend/mom/struct=100) vs short_a (trend/mom/struct=0):
        # difference = 80, purely from trend/mom/struct, vol identical.
        self.assertAlmostEqual(short_b - short_a, 80.0)
        # Volume part in long_a: 80*0.20 = 16.
        self.assertAlmostEqual(long_a - (100*0.4 + 100*0.2 + 100*0.2), 80*0.2)
        # Volume part in short_b: 80*0.20 = 16.
        self.assertAlmostEqual(short_b - (100*0.4 + 100*0.2 + 100*0.2), 80*0.2)


# ---------------------------------------------------------------------------
# 6. Final MTF weighted aggregation
# ---------------------------------------------------------------------------

class TestD8FinalAggregation(unittest.TestCase):
    """Verify aggregate() computes FINAL_LONG/FINAL_SHORT as the TF-weighted
    sum of per-TF composites across all 5 timeframes."""

    def test_all_tfs_perfect_bullish(self):
        """All 5 TFs give (100, 0) -> final_long=100, final_short=0, LONG."""
        with patch.object(d8, "_tf_scores", return_value=(100.0, 0.0)):
            fl, fs, direction = d8.aggregate(_all_tfs())
        self.assertAlmostEqual(fl, 100.0)
        self.assertAlmostEqual(fs, 0.0)
        self.assertEqual(direction, "LONG")

    def test_all_tfs_perfect_bearish(self):
        """All 5 TFs give (0, 100) -> final_short=100, SHORT."""
        with patch.object(d8, "_tf_scores", return_value=(0.0, 100.0)):
            fl, fs, direction = d8.aggregate(_all_tfs())
        self.assertAlmostEqual(fl, 0.0)
        self.assertAlmostEqual(fs, 100.0)
        self.assertEqual(direction, "SHORT")

    def test_all_tfs_neutral(self):
        """All 5 TFs give (50, 50) -> final=50, NO_TRADE."""
        with patch.object(d8, "_tf_scores", return_value=(50.0, 50.0)):
            fl, fs, direction = d8.aggregate(_all_tfs())
        self.assertAlmostEqual(fl, 50.0)
        self.assertAlmostEqual(fs, 50.0)
        self.assertEqual(direction, "NO_TRADE")

    def test_weighted_sum_correctness(self):
        """Each TF contributes tf_score * tf_weight to the final."""
        # Per-TF (tf_long, tf_short) in TF_ORDER:
        # 1d=90/10, 4h=90/10, 1h=50/50, 15m=10/90, 5m=90/10
        pairs = [(90.0, 10.0), (90.0, 10.0), (50.0, 50.0),
                 (10.0, 90.0), (90.0, 10.0)]
        mock = _CycleMock(pairs)
        with patch.object(d8, "_tf_scores", mock):
            fl, fs, direction = d8.aggregate(_all_tfs())
        exp_fl = (90*0.25 + 90*0.25 + 50*0.20 + 10*0.20 + 90*0.10)
        exp_fs = (10*0.25 + 10*0.25 + 50*0.20 + 90*0.20 + 10*0.10)
        self.assertAlmostEqual(fl, exp_fl)   # 66.0
        self.assertAlmostEqual(fs, exp_fs)   # 34.0
        self.assertEqual(direction, "LONG")

    def test_missing_tf_counts_as_neutral_50(self):
        """Missing timeframe adds 50*weight to both final_long and final_short."""
        # Only 5m present, rest missing.
        pairs = [(80.0, 20.0)] + [(50.0, 50.0)] * 4  # 5m first
        mock = _CycleMock(pairs)
        with patch.object(d8, "_tf_scores", mock):
            fl, fs, direction = d8.aggregate({"5m": _candles(250)})
        exp_fl = 80*0.10 + 50*(0.25+0.25+0.20+0.20)  # 8 + 45 = 53.0
        exp_fs = 20*0.10 + 50*(0.25+0.25+0.20+0.20)  # 2 + 45 = 47.0
        self.assertAlmostEqual(fl, exp_fl)
        self.assertAlmostEqual(fs, exp_fs)
        self.assertEqual(direction, "NO_TRADE")


# ---------------------------------------------------------------------------
# 7-13. Direction gate: LONG/SHORT/NO_TRADE, 65 boundary, ties, rounding
# ---------------------------------------------------------------------------

class TestD8DirectionGate(unittest.TestCase):
    """Verify aggregate() returns the correct direction for given final scores.

    Because aggregate() applies TF weights, we use _tf_scores mocks that
    return per-TF composites whose weighted sum equals the desired final.
    Simplest: single-TF (5m, weight 0.10) with composite = final/0.10.
    But that requires composite > 100 for final > 10.  Instead, use all 5 TFs
    with a _CycleMock returning the same pair each time so final = pair_value.
    """

    def _agg_with_final(self, final_long, final_short):
        """Return aggregate() result where final_long/final_short are fixed.

        Achieved by returning (final_long, final_short) from _tf_scores for
        every TF, since sum(tf_weight) = 1.0.
        """
        mock = _CycleMock([(final_long, final_short)] * 5)
        with patch.object(d8, "_tf_scores", mock):
            return d8.aggregate(_all_tfs())

    def test_long_gate(self):
        """LONG when final_long >= 65 AND final_long > final_short."""
        fl, fs, direction = self._agg_with_final(70.0, 60.0)
        self.assertEqual(direction, "LONG")
        self.assertGreaterEqual(fl, 65)
        self.assertGreater(fl, fs)

    def test_short_gate(self):
        """SHORT when final_short >= 65 AND final_short > final_long."""
        fl, fs, direction = self._agg_with_final(60.0, 70.0)
        self.assertEqual(direction, "SHORT")
        self.assertGreaterEqual(fs, 65)
        self.assertGreater(fs, fl)

    def test_no_trade_both_below_65(self):
        fl, fs, direction = self._agg_with_final(60.0, 55.0)
        self.assertEqual(direction, "NO_TRADE")

    def test_no_trade_tie_both_above_65(self):
        """Equal scores with both >=65 -> NO_TRADE."""
        fl, fs, direction = self._agg_with_final(80.0, 80.0)
        self.assertEqual(direction, "NO_TRADE")
        self.assertGreaterEqual(fl, 65)
        self.assertGreaterEqual(fs, 65)

    def test_no_trade_tie_both_at_65(self):
        fl, fs, direction = self._agg_with_final(65.0, 65.0)
        self.assertEqual(direction, "NO_TRADE")

    def test_no_trade_one_above_65_but_other_higher(self):
        """LONG=70, SHORT=75 -> SHORT qualifies, LONG does not."""
        fl, fs, direction = self._agg_with_final(70.0, 75.0)
        self.assertEqual(direction, "SHORT")
        self.assertGreaterEqual(fs, 65)
        self.assertGreater(fs, fl)

    def test_no_trade_long_below_65_short_above(self):
        fl, fs, direction = self._agg_with_final(60.0, 70.0)
        self.assertEqual(direction, "SHORT")

    def test_exactly_65_long_higher(self):
        """final_long == 65.0, final_short == 64.9 -> LONG qualifies."""
        fl, fs, direction = self._agg_with_final(65.0, 64.9)
        self.assertEqual(direction, "LONG")
        self.assertEqual(fl, 65.0)
        self.assertEqual(fs, 64.9)

    def test_exactly_65_short_higher(self):
        """final_short == 65.0, final_long == 64.9 -> SHORT qualifies."""
        fl, fs, direction = self._agg_with_final(64.9, 65.0)
        self.assertEqual(direction, "SHORT")
        self.assertEqual(fl, 64.9)
        self.assertEqual(fs, 65.0)

    def test_just_below_65_is_no_trade(self):
        """final_long = 64.9 -> NO_TRADE even if higher than short."""
        fl, fs, direction = self._agg_with_final(64.9, 60.0)
        self.assertEqual(direction, "NO_TRADE")

    def test_64_99_rounds_to_65_no_trade_if_tie(self):
        """64.99 -> rounds to 65.0. If both round to 65.0 -> tie -> NO_TRADE."""
        fl, fs, direction = self._agg_with_final(64.99, 64.99)
        self.assertEqual(fl, 65.0)
        self.assertEqual(fs, 65.0)
        self.assertEqual(direction, "NO_TRADE")

    def test_64_95_rounds_to_65_0(self):
        """64.95 rounds to 65.0 (ROUND_HALF_UP)."""
        self.assertEqual(d8._round1(64.95), 65.0)


# ---------------------------------------------------------------------------
# 13. Rounding behavior
# ---------------------------------------------------------------------------

class TestD8Rounding(unittest.TestCase):
    def test_round1_half_up(self):
        self.assertEqual(d8._round1(0.05), 0.1)
        self.assertEqual(d8._round1(0.15), 0.2)
        self.assertEqual(d8._round1(0.25), 0.3)
        self.assertEqual(d8._round1(0.35), 0.4)
        self.assertEqual(d8._round1(64.95), 65.0)
        self.assertEqual(d8._round1(65.05), 65.1)
        self.assertEqual(d8._round1(64.85), 64.9)
        self.assertEqual(d8._round1(50.0), 50.0)
        self.assertEqual(d8._round1(100.0), 100.0)
        self.assertEqual(d8._round1(0.0), 0.0)

    def test_rounded_output_from_aggregate(self):
        """aggregate() returns rounded final scores."""
        # 5m only: composite (65.25, 34.75), weight 0.10.
        # final_long  = 50*0.90 + 65.25*0.10 = 45 + 6.525 = 51.525 -> 51.5
        # final_short = 50*0.90 + 34.75*0.10 = 45 + 3.475 = 48.475 -> 48.5
        pairs = [(65.25, 34.75)] + [(50.0, 50.0)] * 4
        mock = _CycleMock(pairs)
        with patch.object(d8, "_tf_scores", mock):
            fl, fs, direction = d8.aggregate({"5m": _candles(250)})
        self.assertEqual(fl, 51.5)
        self.assertEqual(fs, 48.5)
        self.assertEqual(direction, "NO_TRADE")


# ---------------------------------------------------------------------------
# 14. Missing/empty/insufficient TF handling
# ---------------------------------------------------------------------------

class TestD8InsufficientData(unittest.TestCase):
    def test_completely_empty_input(self):
        """No TFs at all -> all neutral 50 -> NO_TRADE."""
        fl, fs, direction = d8.aggregate({})
        self.assertAlmostEqual(fl, 50.0)
        self.assertAlmostEqual(fs, 50.0)
        self.assertEqual(direction, "NO_TRADE")

    def test_empty_list_is_missing(self):
        """Empty list for a TF -> treated as missing (50 neutral)."""
        pairs = [(80.0, 20.0)] + [(50.0, 50.0)] * 4
        mock = _CycleMock(pairs)
        with patch.object(d8, "_tf_scores", mock):
            fl, fs, direction = d8.aggregate({"5m": []})
        self.assertAlmostEqual(fl, 50.0)
        self.assertAlmostEqual(fs, 50.0)
        self.assertEqual(direction, "NO_TRADE")

    def test_none_tf_is_missing(self):
        """None value for a TF -> treated as missing."""
        with patch.object(d8, "_tf_scores", return_value=(80.0, 20.0)):
            fl, fs, direction = d8.aggregate({"5m": None})
        self.assertAlmostEqual(fl, 50.0)
        self.assertAlmostEqual(fs, 50.0)
        self.assertEqual(direction, "NO_TRADE")

    def test_partial_data_mixed_with_missing(self):
        """1d + 5m present, rest missing.  Verify weighted sums."""
        # 1d: 90/10 (w=0.25), 5m: 80/20 (w=0.10), missing: 50/50
        # Only 1d and 5m are present, so _tf_scores is called exactly twice
        # (once per present TF, in TF_ORDER: 1d then 5m). Missing TFs consume
        # no mock entries.
        pairs = [
            (90.0, 10.0),
            (80.0, 20.0),
        ]
        mock = _CycleMock(pairs)
        with patch.object(d8, "_tf_scores", mock):
            fl, fs, direction = d8.aggregate({
                "1d": _candles(250),
                "5m": _candles(250),
            })
        exp_fl = 90*0.25 + 50*0.25 + 50*0.20 + 50*0.20 + 80*0.10  # 63.0
        exp_fs = 10*0.25 + 50*0.25 + 50*0.20 + 50*0.20 + 20*0.10  # 37.0
        self.assertAlmostEqual(fl, exp_fl)
        self.assertAlmostEqual(fs, exp_fs)
        self.assertEqual(direction, "NO_TRADE")

    def test_insufficient_candles_for_components(self):
        """Very short candle list -> components return None/0/50 per their own
        conventions; aggregate handles it without crashing, deterministically."""
        short_candles = _candles(10)  # too few for RSI(14)/RVOL(21)/EMA200
        fl_a, fs_a, dir_a = d8.aggregate({"5m": short_candles})
        fl_b, fs_b, dir_b = d8.aggregate({"5m": short_candles})
        self.assertEqual((fl_a, fs_a, dir_a), (fl_b, fs_b, dir_b))
        self.assertIsInstance(fl_a, float)
        self.assertIsInstance(fs_a, float)


# ---------------------------------------------------------------------------
# 15. Deterministic output
# ---------------------------------------------------------------------------

class TestD8Determinism(unittest.TestCase):
    def test_same_input_same_output(self):
        mock = _CycleMock([(72.3, 61.7)] * 5)
        with patch.object(d8, "_tf_scores", mock):
            a = d8.aggregate(_all_tfs())
            b = d8.aggregate(_all_tfs())
        self.assertEqual(a, b)

    def test_multiple_tfs_deterministic(self):
        pairs = [(90.0, 10.0), (90.0, 10.0), (50.0, 50.0),
                 (10.0, 90.0), (90.0, 10.0)]
        mock_a = _CycleMock(pairs)
        mock_b = _CycleMock(pairs)
        with patch.object(d8, "_tf_scores", mock_a):
            a = d8.aggregate(_all_tfs())
        with patch.object(d8, "_tf_scores", mock_b):
            b = d8.aggregate(_all_tfs())
        self.assertEqual(a, b)

    def test_breakdown_deterministic(self):
        with patch.object(d8, "trend_score", return_value=(60, 40)), \
             patch.object(d8, "compute_rsi", return_value=50.0), \
             patch.object(d8, "momentum_score",
                           side_effect=lambda rsi, direction: 50), \
             patch.object(d8, "structure_score",
                           side_effect=lambda c, direction: 50), \
             patch.object(d8, "compute_rvol", return_value=None), \
             patch.object(d8, "volume_score", return_value=50):
            a = d8.breakdown({"5m": _candles(250)})
            b = d8.breakdown({"5m": _candles(250)})
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Bonus: breakdown() helper spot-check (inspection only, not scoring)
# ---------------------------------------------------------------------------

class TestD8BreakdownHelper(unittest.TestCase):
    def test_breakdown_returns_all_timeframes(self):
        with patch.object(d8, "trend_score", return_value=(60, 40)), \
             patch.object(d8, "compute_rsi", return_value=50.0), \
             patch.object(d8, "momentum_score",
                           side_effect=lambda rsi, direction: 50), \
             patch.object(d8, "structure_score",
                           side_effect=lambda c, direction: 50), \
             patch.object(d8, "compute_rvol", return_value=None), \
             patch.object(d8, "volume_score", return_value=50):
            bd = d8.breakdown({"5m": _candles(250)})
        for tf in d8.TF_ORDER:
            self.assertIn(tf, bd)

    def test_breakdown_missing_tf_is_none(self):
        with patch.object(d8, "trend_score", return_value=(60, 40)), \
             patch.object(d8, "compute_rsi", return_value=50.0), \
             patch.object(d8, "momentum_score",
                           side_effect=lambda rsi, direction: 50), \
             patch.object(d8, "structure_score",
                           side_effect=lambda c, direction: 50), \
             patch.object(d8, "compute_rvol", return_value=None), \
             patch.object(d8, "volume_score", return_value=50):
            bd = d8.breakdown({"5m": _candles(250)})
        for tf in ("1d", "4h", "1h", "15m"):
            self.assertIsNone(bd[tf]["trend_long"])
            self.assertIsNone(bd[tf]["tf_long"])

    def test_breakdown_present_tf_has_values(self):
        with patch.object(d8, "trend_score", return_value=(60, 40)), \
             patch.object(d8, "compute_rsi", return_value=50.0), \
             patch.object(d8, "momentum_score",
                           side_effect=lambda rsi, direction: 50), \
             patch.object(d8, "structure_score",
                           side_effect=lambda c, direction: 50), \
             patch.object(d8, "compute_rvol", return_value=None), \
             patch.object(d8, "volume_score", return_value=50):
            bd = d8.breakdown({"5m": _candles(250)})
        self.assertEqual(bd["5m"]["trend_long"], 60)
        self.assertEqual(bd["5m"]["trend_short"], 40)
        self.assertEqual(bd["5m"]["vol"], 50)
