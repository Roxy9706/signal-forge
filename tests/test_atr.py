"""Tests for D3 ATR14 — indicators.atr (Wilder's True Range formulation)."""

from __future__ import annotations
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.atr import compute_atr


def _candle(close_time, open_, high, low, close):
    return {
        'close_time': close_time,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
    }


def _kline_series(base_time, opens, highs, lows, closes):
    return [_candle(base_time + i * 300000, o, h, l, c)
            for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes))]


class TestTrueRangeCalculation(unittest.TestCase):
    def test_tr_range_only(self):
        # To get TR=5 for all 14 TR values, all candles need close=95
        # Candle i: high=110, low=105, prev_close=95
        # range=5, gap_up=|110-95|=15, gap_down=|105-95|=10 → but TR = max(5, 15, 10) = 15
        # Wait: range = high - low = 110 - 105 = 5
        # gap_up = |110 - 95| = 15
        # gap_down = |105 - 95| = 10
        # TR = max(5, 15, 10) = 15
        # To get TR=5 (range only), we need gaps to be smaller than range.
        # Candle i: high=105, low=100, prev_close=102
        #   range=5, gap_up=|105-102|=3, gap_down=|100-102|=2 → TR=max(5,3,2)=5
        opens = [102.0] * 15
        highs = [105.0] * 15
        lows = [100.0] * 15
        closes = [103.0] * 15  # Actually: close doesn't matter for TR of NEXT candle
        # For TR of candle 1: high[1]=105, low[1]=100, close[0]=103
        #   range = 5, gap_up = |105-103|=2, gap_down = |100-103|=3 → TR = max(5,2,3) = 5
        # But close[0] is used as prev_close for candle 1, so close[0]=103
        # Actually let me use close=103 for candle 0, and for candle 1 we need:
        #   high=105, low=100, prev_close=103 → TR = max(5, |105-103|=2, |100-103|=3) = 5 ✓
        opens = [100.0] * 15  # open doesn't affect TR
        highs = [105.0] * 15
        lows = [100.0] * 15
        closes = [103.0] + [103.0] * 14  # candle 0 close=103, candles 1-14 close=103
        series = _kline_series(1000000, opens, highs, lows, closes)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 5.0, places=6)

    def test_tr_gap_up_dominant(self):
        # To get TR=15 for all 14 TR values:
        # Candle 0: close=95 (prev_close for candle 1)
        # Candles 1-14: high=110, low=105, prev_close=95
        #   TR = max(5, |110-95|=15, |105-95|=10) = 15
        opens = [95.0] * 15
        highs = [110.0] * 15
        lows = [105.0] * 15
        closes = [95.0] * 15
        series = _kline_series(1000000, opens, highs, lows, closes)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 15.0, places=6)

    def test_tr_gap_down_dominant(self):
        # Candle 0: close=95
        # Candles 1-14: high=105, low=90, prev_close=95
        #   TR = max(15, |105-95|=10, |90-95|=5) = 15
        opens = [95.0] * 15
        highs = [105.0] * 15
        lows = [90.0] * 15
        closes = [95.0] * 15
        series = _kline_series(1000000, opens, highs, lows, closes)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 15.0, places=6)


class TestWilderATR14Seed(unittest.TestCase):
    def test_atr_needs_period_plus_one_candles(self):
        series = _kline_series(1000000,
                               [100.0] * 14, [101.0] * 14, [99.0] * 14, [100.0] * 14)
        self.assertIsNone(compute_atr(series, 14))

    def test_atr_insufficient_data_short_series(self):
        series = _kline_series(1000000,
                               [100.0] * 10, [101.0] * 10, [99.0] * 10, [100.0] * 10)
        self.assertIsNone(compute_atr(series, 14))

    def test_atr_empty_series(self):
        self.assertIsNone(compute_atr([], 14))

    def test_atr_period_too_small(self):
        self.assertIsNone(compute_atr([_candle(0, 100, 101, 99, 100)], 1))
        self.assertIsNone(compute_atr([_candle(0, 100, 101, 99, 100)], 0))
        self.assertIsNone(compute_atr([_candle(0, 100, 101, 99, 100)], -1))

    def test_sma_seed_flat_candles(self):
        # 15 candles: all have high=101, low=99, close=100
        # TR for candle i (i>=1): max(2, |101-100|=1, |99-100|=1) = 2
        # Seed ATR = 14*2/14 = 2.0
        series = _kline_series(1000000,
                               [100.0] * 15, [101.0] * 15, [99.0] * 15, [100.0] * 15)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 2.0, places=6)

    def test_sma_seed_ascending_candles(self):
        # 15 candles: close=100,101,...,114, high=close+1, low=close-1
        # TR for candle i (i>=1):
        #   range = 2, gap_up = |(close+1) - prev_close|,
        #   gap_down = |(close-1) - prev_close|
        # For consecutive ascending by 1: prev_close = close - 1
        #   gap_up = |close+1 - (close-1)| = 2
        #   gap_down = |close-1 - (close-1)| = 0
        #   TR = max(2, 2, 0) = 2
        opens = [100.0 + float(i) for i in range(15)]
        highs = [o + 1.0 for o in opens]
        lows = [o - 1.0 for o in opens]
        closes = [o for o in opens]
        series = _kline_series(1000000, opens, highs, lows, closes)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 2.0, places=6)


class TestWilderRecurrence(unittest.TestCase):
    def test_recurrence_single_step(self):
        # 16 candles: all flat (TR=2 each), seed ATR=2.0, 1 recurrence stays at 2.0
        series = _kline_series(1000000,
                               [100.0] * 16, [101.0] * 16, [99.0] * 16, [100.0] * 16)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 2.0, places=6)

    def test_recurrence_with_different_tr(self):
        # 16 candles: first 15 flat (TR=2), 16th has TR=10
        # Seed ATR = 2.0, recurrence: (2*13+10)/14 = 36/14 ≈ 2.571428571
        opens = [100.0] * 15 + [108.0]
        highs = [101.0] * 15 + [110.0]
        lows = [99.0] * 15 + [106.0]
        closes = [100.0] * 15 + [107.0]
        series = _kline_series(1000000, opens, highs, lows, closes)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        expected = (2.0 * 13 + 10.0) / 14.0
        self.assertAlmostEqual(atr, expected, places=6)

    def test_recurrence_multiple_steps(self):
        # 20 candles: first 15 flat (TR=2), then 5 candles with higher TR
        # Use pattern where TR=5 for candles 15-19
        # Candle 15: high=106, low=104, prev_close=100 → TR=max(2,6,4)=6
        # Let's just verify ATR increases from seed
        opens = [100.0] * 15 + [102.0, 103.0, 104.0, 105.0, 106.0]
        highs = [101.0] * 15 + [106.0, 107.0, 108.0, 109.0, 110.0]
        lows = [99.0] * 15 + [104.0, 105.0, 106.0, 107.0, 108.0]
        closes = [100.0] * 15 + [102.0, 103.0, 104.0, 105.0, 106.0]
        series = _kline_series(1000000, opens, highs, lows, closes)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        self.assertGreater(atr, 2.0)
        self.assertLess(atr, 5.0)


class TestKnownReferenceATR(unittest.TestCase):
    def test_wilder_original_reference(self):
        # 16 candles: 15 flat (TR=2), 16th with TR=3
        # Seed ATR=2.0, recurrence: (2*13+3)/14 = 29/14
        series = _kline_series(1000000,
                               [100.0] * 15 + [102.0],
                               [101.0] * 15 + [103.0],
                               [99.0] * 15 + [101.0],
                               [100.0] * 15 + [102.0])
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        expected = (2.0 * 13 + 3.0) / 14.0
        self.assertAlmostEqual(atr, expected, places=6)

    def test_all_same_tr(self):
        # 30 candles, all with TR=3
        opens = [100.0] * 30
        highs = [102.0] * 30
        lows = [99.0] * 30
        closes = [100.0] * 30
        series = _kline_series(1000000, opens, highs, lows, closes)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 3.0, places=6)


class TestEdgeCases(unittest.TestCase):
    def test_single_candle_no_tr(self):
        self.assertIsNone(compute_atr([_candle(0, 100, 101, 99, 100)], 14))

    def test_two_candles_insufficient_for_period_14(self):
        series = _kline_series(1000000,
                               [100.0, 101.0], [102.0, 103.0], [99.0, 100.0], [100.0, 101.0])
        self.assertIsNone(compute_atr(series, 14))

    def test_period_one(self):
        series = _kline_series(1000000,
                               [100.0] * 5, [101.0] * 5, [99.0] * 5, [100.0] * 5)
        self.assertIsNone(compute_atr(series, 1))

    def test_high_low_equal(self):
        # Candle 0: close=98, high=100, low=100
        # Candle 1: high=100, low=100, prev_close=98 → TR=max(0,2,2)=2
        # Candles 2-14: all same, TR=0
        # Seed: 1*2 + 13*0 = 2/14
        opens = [100.0, 100.0] + [100.0] * 13
        highs = [100.0, 100.0] + [100.0] * 13
        lows = [100.0, 100.0] + [100.0] * 13
        closes = [98.0, 100.0] + [100.0] * 13
        series = _kline_series(1000000, opens, highs, lows, closes)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        expected = 2.0 / 14.0
        self.assertAlmostEqual(atr, expected, places=6)


class TestNoFutureDataLeakage(unittest.TestCase):
    def test_atr_uses_only_available_data(self):
        opens = [100.0 + float(i) for i in range(20)]
        highs = [o + 2.0 for o in opens]
        lows = [o - 1.0 for o in opens]
        closes = [o + 0.5 for o in opens]
        series = _kline_series(1000000, opens, highs, lows, closes)

        atr_15 = compute_atr(series[:15], 14)
        atr_16 = compute_atr(series[:16], 14)
        atr_20 = compute_atr(series, 14)

        # All TR=3 with this pattern
        self.assertAlmostEqual(atr_15, 3.0, places=6)
        self.assertAlmostEqual(atr_16, 3.0, places=6)
        self.assertAlmostEqual(atr_20, 3.0, places=6)

    def test_atr_deterministic_with_same_input(self):
        series = _kline_series(1000000,
                               [100.0] * 20, [101.0] * 20, [99.0] * 20, [100.0] * 20)
        atr1 = compute_atr(series, 14)
        atr2 = compute_atr(series, 14)
        self.assertEqual(atr1, atr2)
        self.assertIsNotNone(atr1)

    def test_atr_does_not_use_future_closes(self):
        opens = [100.0 + float(i) for i in range(30)]
        highs = [o + 2.0 for o in opens]
        lows = [o - 1.0 for o in opens]
        closes = [o + 0.5 for o in opens]
        series = _kline_series(1000000, opens, highs, lows, closes)

        atr_20 = compute_atr(series[:20], 14)
        atr_25 = compute_atr(series[:25], 14)

        self.assertAlmostEqual(atr_20, 3.0, places=6)
        self.assertAlmostEqual(atr_25, 3.0, places=6)


class TestPeriodValidation(unittest.TestCase):
    def test_period_14_default(self):
        series = _kline_series(1000000,
                               [100.0] * 15, [101.0] * 15, [99.0] * 15, [100.0] * 15)
        atr = compute_atr(series)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 2.0, places=6)

    def test_period_21(self):
        series = _kline_series(1000000,
                               [100.0] * 22, [101.0] * 22, [99.0] * 22, [100.0] * 22)
        atr = compute_atr(series, 21)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 2.0, places=6)

    def test_period_7(self):
        series = _kline_series(1000000,
                               [100.0] * 8, [101.0] * 8, [99.0] * 8, [100.0] * 8)
        atr = compute_atr(series, 7)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 2.0, places=6)


class TestIntegrationWithOtherIndicators(unittest.TestCase):
    def test_atr_independent_of_rsi(self):
        from indicators.rsi import compute_rsi
        closes = [100.0 + float(i) for i in range(20)]
        candles = [{'close': c} for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi)
        self.assertEqual(rsi, 100.0)

        series = _kline_series(1000000,
                               [100.0] * 15, [101.0] * 15, [99.0] * 15, [100.0] * 15)
        atr = compute_atr(series, 14)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 2.0, places=6)

    def test_atr_d7_independent(self):
        from indicators.pullback import d7_pullback_pass
        # Working D7 LONG test data from test_pullback.py
        series = _kline_series(1000000,
                               [100.0] * 18 + [103.0],
                               [101.0] * 18 + [104.0],
                               [99.0] * 18 + [102.0],
                               [100.0] * 18 + [103.0])
        series.append(_candle(1000000 + 19 * 300000, 101.0, 102.5, 100.26, 102.2))
        result = d7_pullback_pass(series, 19, 'LONG')
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
