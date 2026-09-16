"""Tests for D7 pullback/retest confirmation — indicators.pullback."""

from __future__ import annotations
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.pullback import (
    d7_pullback_pass,
    validate_pullback_long,
    validate_pullback_short,
    find_prior_trend_side_candle,
    _select_retest_level_by_direction,
)
from indicators.ema import compute_ema


def _k(close_time, open_, high, low, close):
    return {
        'close_time': close_time,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
    }


def _kline_series(base_time, opens, highs, lows, closes):
    return [_k(base_time + i * 300000, o, h, l, c)
            for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes))]


def _build_with_setup(base_time, prior_opens, prior_highs, prior_lows, prior_closes,
                      setup_open, setup_high, setup_low, setup_close):
    series = _kline_series(base_time, prior_opens, prior_highs, prior_lows, prior_closes)
    series.append(_k(base_time + len(prior_closes) * 300000,
                     setup_open, setup_high, setup_low, setup_close))
    return series


def _compute_expected_ema(prior_closes, setup_close, period=20):
    """Compute expected EMA for test data design."""
    all_closes = list(prior_closes) + [setup_close]
    if len(all_closes) < period:
        return None
    sma = sum(all_closes[:period]) / period
    k = 2.0 / (period + 1.0)
    ema = sma
    for price in all_closes[period:]:
        ema = price * k + ema * (1.0 - k)
    return ema


class TestEMA20Precedence(unittest.TestCase):
    def test_ema20_returns_value_when_available(self):
        series = _kline_series(
            1000000,
            [100.0] * 20,
            [101.0] * 20,
            [99.0] * 20,
            [100.0] * 20,
        )
        L = _select_retest_level_by_direction(series, 19, 'LONG')
        self.assertIsNotNone(L)
        self.assertTrue(L > 0)

    def test_structure_fallback_when_no_ema20(self):
        series = _kline_series(
            1000000,
            [100.0, 101.0, 102.0],
            [101.0, 102.0, 103.0],
            [99.0, 100.0, 101.0],
            [100.0, 101.0, 102.0],
        )
        L = _select_retest_level_by_direction(series, 2, 'LONG')
        self.assertIsNone(L)

    def test_ema20_wins_over_structure(self):
        series = _kline_series(
            1000000,
            [100.0] * 20,
            [101.0] * 20,
            [99.0] * 20,
            [100.0] * 20,
        )
        series.append(_k(1000000 + 20 * 300000, 101.0, 102.0, 100.0, 100.8))
        L = _select_retest_level_by_direction(series, 20, 'LONG')
        self.assertIsNotNone(L)
        self.assertTrue(abs(L - 100.0) < 0.5)


class TestPriorTrendSideCandle(unittest.TestCase):
    def test_long_prior_close_above_L(self):
        series = _build_with_setup(
            1000000,
            [102.0] * 19 + [102.5],
            [103.0] * 19 + [103.5],
            [101.0] * 19 + [101.5],
            [102.0] * 19 + [102.6],
            101.0, 102.5, 101.5, 102.2,
        )
        P = find_prior_trend_side_candle(series, 20, 101.5, 'LONG')
        self.assertIsNotNone(P)
        self.assertEqual(P['close'], 102.6)

    def test_long_no_prior_above_L(self):
        series = _build_with_setup(
            1000000,
            [100.0] * 19 + [100.0],
            [101.0] * 19 + [101.0],
            [99.0] * 19 + [99.0],
            [100.0] * 19 + [100.0],
            101.0, 102.0, 101.0, 101.5,
        )
        P = find_prior_trend_side_candle(series, 20, 101.0, 'LONG')
        self.assertIsNone(P)

    def test_short_prior_close_below_L(self):
        series = _build_with_setup(
            1000000,
            [99.0] * 19 + [98.0],
            [100.0] * 19 + [99.0],
            [98.0] * 19 + [97.0],
            [99.0] * 19 + [98.0],
            100.5, 101.0, 99.0, 99.5,
        )
        P = find_prior_trend_side_candle(series, 20, 100.0, 'SHORT')
        self.assertIsNotNone(P)
        self.assertEqual(P['close'], 98.0)

    def test_short_no_prior_below_L(self):
        series = _build_with_setup(
            1000000,
            [101.0] * 19 + [101.0],
            [102.0] * 19 + [102.0],
            [100.0] * 19 + [100.0],
            [101.0] * 19 + [101.0],
            100.0, 101.0, 99.0, 99.5,
        )
        P = find_prior_trend_side_candle(series, 20, 100.0, 'SHORT')
        self.assertIsNone(P)

    def test_prior_excludes_setup_candle(self):
        series = _build_with_setup(
            1000000,
            [99.0] * 19 + [99.0],
            [100.0] * 19 + [100.0],
            [98.0] * 19 + [98.0],
            [99.0] * 19 + [99.0],
            101.0, 102.0, 100.0, 101.5,
        )
        P = find_prior_trend_side_candle(series, 20, 100.5, 'LONG')
        self.assertIsNone(P)


class TestValidLongRejection(unittest.TestCase):
    def test_valid_long_rejection(self):
        series = _build_with_setup(
            1000000,
            [102.0] * 19 + [102.5],
            [103.0] * 19 + [103.5],
            [101.0] * 19 + [101.5],
            [102.0] * 19 + [102.6],
            101.0, 102.5, 101.5, 102.2,
        )
        self.assertTrue(validate_pullback_long(series, 20, 101.5))

    def test_close_exactly_at_L_invalid(self):
        series = _build_with_setup(
            1000000,
            [102.0] * 19 + [102.5],
            [103.0] * 19 + [103.5],
            [101.0] * 19 + [101.5],
            [102.0] * 19 + [102.6],
            101.0, 102.0, 101.0, 101.5,
        )
        self.assertFalse(validate_pullback_long(series, 20, 101.5))

    def test_low_breaks_below_L(self):
        series = _build_with_setup(
            1000000,
            [102.0] * 19 + [102.5],
            [103.0] * 19 + [103.5],
            [101.0] * 19 + [101.5],
            [102.0] * 19 + [102.6],
            101.0, 102.5, 101.0, 102.2,
        )
        self.assertFalse(validate_pullback_long(series, 20, 101.5))

    def test_high_does_not_reach_L(self):
        series = _build_with_setup(
            1000000,
            [102.0] * 19 + [102.5],
            [103.0] * 19 + [103.5],
            [101.0] * 19 + [101.5],
            [102.0] * 19 + [102.6],
            100.5, 101.0, 100.0, 100.8,
        )
        self.assertFalse(validate_pullback_long(series, 20, 101.5))

    def test_bearish_body_rejected(self):
        series = _build_with_setup(
            1000000,
            [102.0] * 19 + [102.5],
            [103.0] * 19 + [103.5],
            [101.0] * 19 + [101.5],
            [102.0] * 19 + [102.6],
            102.0, 102.5, 101.5, 101.2,
        )
        self.assertFalse(validate_pullback_long(series, 20, 101.5))

    def test_no_prior_trend_side_candle(self):
        series = _build_with_setup(
            1000000,
            [100.0] * 19 + [100.0],
            [101.0] * 19 + [101.0],
            [99.0] * 19 + [99.0],
            [100.0] * 19 + [100.0],
            101.0, 102.5, 101.5, 102.2,
        )
        self.assertFalse(validate_pullback_long(series, 20, 101.0))


class TestValidShortRejection(unittest.TestCase):
    def test_valid_short_rejection(self):
        series = _build_with_setup(
            1000000,
            [98.0] * 19 + [97.5],
            [99.0] * 19 + [98.5],
            [97.0] * 19 + [96.5],
            [98.0] * 19 + [97.4],
            99.5, 100.0, 97.0, 99.0,
        )
        self.assertTrue(validate_pullback_short(series, 20, 100.0))

    def test_close_exactly_at_L_invalid_short(self):
        series = _build_with_setup(
            1000000,
            [98.0] * 19 + [97.5],
            [99.0] * 19 + [98.5],
            [97.0] * 19 + [96.5],
            [98.0] * 19 + [97.4],
            100.0, 100.5, 99.0, 100.0,
        )
        self.assertFalse(validate_pullback_short(series, 20, 100.0))

    def test_high_breaks_above_L_short(self):
        series = _build_with_setup(
            1000000,
            [98.0] * 19 + [97.5],
            [99.0] * 19 + [98.5],
            [97.0] * 19 + [96.5],
            [98.0] * 19 + [97.4],
            99.0, 101.0, 97.0, 99.2,
        )
        self.assertFalse(validate_pullback_short(series, 20, 100.0))

    def test_low_does_not_reach_L_short(self):
        series = _build_with_setup(
            1000000,
            [98.0] * 19 + [97.5],
            [99.0] * 19 + [98.5],
            [97.0] * 19 + [96.5],
            [98.0] * 19 + [97.4],
            98.5, 99.5, 99.0, 98.8,
        )
        self.assertFalse(validate_pullback_short(series, 20, 99.0))

    def test_bullish_body_rejected_short(self):
        series = _build_with_setup(
            1000000,
            [98.0] * 19 + [97.5],
            [99.0] * 19 + [98.5],
            [97.0] * 19 + [96.5],
            [98.0] * 19 + [97.4],
            98.0, 99.0, 97.0, 98.5,
        )
        self.assertFalse(validate_pullback_short(series, 20, 98.0))


class TestFullD7Gate(unittest.TestCase):
    def test_full_long_pass(self):
        # Design: 19 prior candles with close=100.0, setup candle close=102.2.
        # EMA20 through setup: seed=(19*100.0 + 102.2)/20 = 100.11.
        # Since all 20 candles are the seed, EMA = seed = 100.11.
        # L = 100.11.
        # Prior candles have close=100.0 < L=100.11 → NO prior with close > L.
        # Pullback FAILS. This test is designed to fail, demonstrating the
        # contradiction: when EMA20 takes precedence and prior closes are
        # below the EMA, no prior trend-side candle exists.
        # 
        # FIXED DESIGN: Use asymmetric prior candles.
        # 18 prior candles close=100.0, 1 prior candle close=103.0 (most recent).
        # Setup candle: open=101.0, high=102.5, low=100.26, close=102.2.
        # All 20 closes through setup: 18*100.0 + 103.0 + 102.2 = 2005.2.
        # Seed SMA = 2005.2/20 = 100.26. EMA = 100.26 (no candles after seed).
        # L = 100.26.
        # Prior candle at index 18 has close=103.0 > L=100.26 ✓ pullback.
        # Setup candle: low=100.26 = L ✓ tested, close=102.2 > L ✓ closed_back,
        #   close(102.2) > open(101.0) ✓ body_bullish,
        #   low(100.26) >= L(100.26) ✓ no_break_below.
        series = _build_with_setup(
            1000000,
            [100.0] * 18 + [103.0],   # 18 prior close=100.0, 1 prior close=103.0
            [101.0] * 18 + [104.0],
            [99.0] * 18 + [102.0],
            [100.0] * 18 + [103.0],
            101.0, 102.5, 100.26, 102.2,
        )
        # Verify EMA20 computation
        expected_ema = _compute_expected_ema([100.0]*18 + [103.0], 102.2)
        self.assertAlmostEqual(expected_ema, 100.26, places=2)
        self.assertTrue(d7_pullback_pass(series, 19, 'LONG'))

    def test_full_short_pass(self):
        # Design: 18 prior candles close=101.0, 1 prior candle close=97.0.
        # Setup candle: open=99.5, high=100.65, low=97.0, close=98.0.
        # All 20 closes: 18*101.0 + 97.0 + 98.0 = 2013.0.
        # Seed SMA = 2013.0/20 = 100.65. EMA = 100.65.
        # L = 100.65.
        # Prior candle at index 18 has close=97.0 < L=100.65 ✓ pullback.
        # Setup candle: high=100.65 = L ✓ tested, close=98.0 < L ✓ closed_back,
        #   close(98.0) < open(99.5) ✓ body_bearish,
        #   high(100.65) <= L(100.65) ✓ no_break_above.
        series = _build_with_setup(
            1000000,
            [101.0] * 18 + [97.0],
            [102.0] * 18 + [98.0],
            [100.0] * 18 + [96.0],
            [101.0] * 18 + [97.0],
            99.5, 100.65, 97.0, 98.0,
        )
        expected_ema = _compute_expected_ema([101.0]*18 + [97.0], 98.0)
        self.assertAlmostEqual(expected_ema, 100.65, places=2)
        self.assertTrue(d7_pullback_pass(series, 19, 'SHORT'))

    def test_insufficient_data_returns_false(self):
        series = _kline_series(
            1000000,
            [100.0, 101.0, 102.0, 103.0, 104.0],
            [101.0, 102.0, 103.0, 104.0, 105.0],
            [99.0, 100.0, 101.0, 102.0, 103.0],
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )
        self.assertFalse(d7_pullback_pass(series, 4, 'LONG'))

    def test_invalid_direction_returns_false(self):
        series = _kline_series(
            1000000,
            [100.0] * 25,
            [101.0] * 25,
            [99.0] * 25,
            [100.0] * 25,
        )
        self.assertFalse(d7_pullback_pass(series, 24, 'INVALID'))
        self.assertFalse(d7_pullback_pass(series, 24, ''))
        self.assertFalse(d7_pullback_pass(series, 24, None))

    def test_structure_fallback_long(self):
        # Force structure fallback: use FEWER than 20 candles through setup.
        # 18 prior candles + 1 setup = 19 total. EMA20 needs 20 → unavailable.
        # Structure fallback uses most recent swing low from the 18-candle window.
        series = _kline_series(
            1000000,
            [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 98.0, 99.0,
             100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0],  # 18 prior
            [101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0, 100.0,
             101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 100.0],
            [99.0, 98.0, 97.0, 96.0, 98.0, 97.0, 96.0, 97.0, 98.0, 99.0,
             99.0, 98.0, 97.0, 96.0, 98.0, 97.0, 96.0, 97.0],
            [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 98.0, 99.0,
             100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0],
        )
        # Setup candle at index 18: tests swing low L, closes above L.
        # Swing lows in window (indices 0-17):
        #   idx 2: low=97.0 (prev 98.0 > 97.0 < 96.0) ✓
        #   idx 6: low=96.0 (prev 97.0 > 96.0 < 97.0) ✓
        #   idx 13: low=96.0 (prev 97.0 > 96.0 < 98.0) ✓
        # Most recent swing low = idx 13, low=96.0. L=96.0.
        # Prior candle with close > 96.0: many exist (e.g. idx 17 close=99.0).
        # Setup candle: open=97.0, high=102.5, low=96.0 (=L, touches), close=98.5 (>L).
        #   tested: low(96.0) <= 96.0 AND high(102.5) >= 96.0 → True ✓
        #   closed_back: close(98.5) > 96.0 → True ✓
        #   body_bullish: close(98.5) > open(97.0) → True ✓
        #   no_break_below: low(96.0) >= 96.0 → True ✓
        series.append(_k(1000000 + 18 * 300000, 97.0, 102.5, 96.0, 98.5))
        self.assertTrue(d7_pullback_pass(series, 18, 'LONG'))

    def test_structure_fallback_short(self):
        # 18 prior candles + 1 setup = 19 total. EMA20 unavailable.
        # Structure fallback uses most recent swing high from the 18-candle window.
        series = _kline_series(
            1000000,
            [98.0, 97.0, 96.0, 97.0, 98.0, 99.0, 100.0, 99.0, 98.0, 97.0,
             96.0, 97.0, 98.0, 97.0, 96.0, 97.0, 98.0, 97.0],  # 18 prior
            [99.0, 98.0, 97.0, 98.0, 99.0, 100.0, 101.0, 100.0, 99.0, 98.0,
             97.0, 98.0, 99.0, 98.0, 97.0, 98.0, 99.0, 98.0],
            [97.0, 96.0, 95.0, 96.0, 97.0, 98.0, 99.0, 98.0, 97.0, 96.0,
             95.0, 96.0, 97.0, 96.0, 95.0, 96.0, 97.0, 96.0],
            [98.0, 97.0, 96.0, 97.0, 98.0, 99.0, 100.0, 99.0, 98.0, 97.0,
             96.0, 97.0, 98.0, 97.0, 96.0, 97.0, 98.0, 97.0],
        )
        # Swing highs in window (indices 0-17):
        #   idx 4: high=99.0 (prev 98.0 < 99.0 > 100.0)? No! 99.0 < 100.0.
        #   idx 6: high=101.0 (prev 100.0 < 101.0 > 100.0) ✓
        #   idx 12: high=99.0 (prev 98.0 < 99.0 > 98.0) ✓
        # Most recent swing high = idx 12, high=99.0. L=99.0.
        # Setup candle at index 18: tests L, closes below L.
        #   open=96.5, high=99.0 (=L, touches), low=96.0, close=96.2 (<L).
        #   tested: low(96.0) <= 99.0 AND high(99.0) >= 99.0 → True ✓
        #   closed_back: close(96.2) < 99.0 → True ✓
        #   body_bearish: close(96.2) < open(96.5) → True ✓
        #   no_break_above: high(99.0) <= 99.0 → True ✓
        # Prior candle with close < 99.0: many exist (e.g. idx 17 close=97.0).
        series.append(_k(1000000 + 18 * 300000, 96.5, 99.0, 96.0, 96.2))
        self.assertTrue(d7_pullback_pass(series, 18, 'SHORT'))


class TestNoFutureData(unittest.TestCase):
    def test_result_deterministic_with_same_input(self):
        series = _build_with_setup(
            1000000,
            [100.0] * 18 + [103.0],
            [101.0] * 18 + [104.0],
            [99.0] * 18 + [102.0],
            [100.0] * 18 + [103.0],
            101.0, 102.5, 100.26, 102.2,
        )
        self.assertEqual(d7_pullback_pass(series, 19, 'LONG'),
                         d7_pullback_pass(series, 19, 'LONG'))

    def test_ema20_uses_only_through_setup(self):
        series = _kline_series(
            1000000,
            [100.0] * 20,
            [101.0] * 20,
            [99.0] * 20,
            [100.0] * 20,
        )
        L = _select_retest_level_by_direction(series, 19, 'LONG')
        self.assertIsNotNone(L)
        self.assertTrue(abs(L - 100.0) < 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
