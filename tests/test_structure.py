"""Tests for indicators.structure — strict 3-candle swing fractal."""

from __future__ import annotations
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.structure import (
    detect_swings,
    most_recent_swing_high,
    most_recent_swing_low,
)


def _k(close_time, open_, high, low, close):
    return {
        'close_time': close_time,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
    }


def _series(base_time, opens, highs, lows, closes):
    return [_k(base_time + i * 300000, o, h, l, c)
            for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes))]


class TestDetectSwings(unittest.TestCase):
    def test_no_swing_flat_middle(self):
        s = _series(0, [100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
        highs, lows = detect_swings(s)
        self.assertEqual(len(highs), 0)
        self.assertEqual(len(lows), 0)

    def test_no_swing_two_candles(self):
        s = _series(0, [100, 101], [101, 102], [99, 100], [100, 101])
        highs, lows = detect_swings(s)
        self.assertEqual(len(highs), 0)
        self.assertEqual(len(lows), 0)

    def test_ordered_output(self):
        s = _series(0,
                    [100, 101, 100, 101, 100],
                    [101, 102, 101, 103, 102],
                    [99, 100, 99, 98, 99],
                    [100, 101, 100, 101, 100])
        highs, lows = detect_swings(s)
        self.assertEqual(highs, [1, 3])
        # idx 1: low[0]=99 > low[1]=100? False → no swing low at 1.
        # idx 2: low[1]=100 > low[2]=99 < low[3]=98? 99 < 98 = False → no swing low at 2.
        # idx 3: low[2]=99 > low[3]=98 < low[4]=99 → True → swing low at idx 3.
        self.assertEqual(lows, [3])

    def test_scan_1(self):
        s = _series(0,
                    [100, 102, 101.5, 102.5, 101, 103, 102],
                    [101, 103, 102, 103.5, 102, 104.5, 103.5],
                    [99, 101, 99.5, 101.5, 99, 102, 101],
                    [100, 102, 100, 102.5, 100, 103, 101])
        highs, lows = detect_swings(s)
        self.assertEqual(highs, [1, 3, 5])
        # idx 2: low[1]=101 > low[2]=99.5 < low[3]=101.5 → True → swing low at idx 2.
        # idx 4: low[3]=101.5 > low[4]=99 < low[5]=102 → True → swing low at idx 4.
        self.assertEqual(lows, [2, 4])


class TestMostRecentSwingHigh(unittest.TestCase):
    def test_returns_none_empty(self):
        self.assertIsNone(most_recent_swing_high([]))

    def test_returns_none_no_swing(self):
        s = _series(0, [100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
        self.assertIsNone(most_recent_swing_high(s))

    def test_returns_most_recent(self):
        # 6 candles. Last candle (idx 5) CANNOT be a swing — no i+1 neighbor.
        # idx 1: high[0]=101 < high[1]=102 > high[2]=101 → True → swing high at 1.
        # idx 3: high[2]=101 < high[3]=103 > high[4]=102 → True → swing high at 3.
        # most recent = idx 3, high=103
        s = _series(0,
                    [100, 101, 100, 101, 100, 101],
                    [101, 102, 101, 103, 102, 104],
                    [99, 100, 99, 98, 99, 97],
                    [100, 101, 100, 101, 100, 101])
        sw = most_recent_swing_high(s)
        self.assertIsNotNone(sw)
        self.assertEqual(sw['high'], 103)


class TestMostRecentSwingLow(unittest.TestCase):
    def test_returns_none_empty(self):
        self.assertIsNone(most_recent_swing_low([]))

    def test_returns_none_no_swing(self):
        s = _series(0, [100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
        self.assertIsNone(most_recent_swing_low(s))

    def test_returns_most_recent(self):
        # 6 candles. Last candle (idx 5) CANNOT be a swing.
        # idx 1: low[0]=99 > low[1]=98 < low[2]=99 → True → swing low at 1.
        # idx 3: low[2]=99 > low[3]=97 < low[4]=99 → True → swing low at 3.
        # most recent = idx 3, low=97
        s = _series(0,
                    [100, 99, 100, 99, 100, 99],
                    [101, 100, 101, 100, 101, 100],
                    [99, 98, 99, 97, 99, 96],
                    [100, 99, 100, 99, 100, 99])
        sw = most_recent_swing_low(s)
        self.assertIsNotNone(sw)
        self.assertEqual(sw['low'], 97)


if __name__ == '__main__':
    unittest.main(verbosity=2)
