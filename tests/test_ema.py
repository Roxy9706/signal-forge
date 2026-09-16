"""Tests for indicators.ema — pure Python EMA."""

from __future__ import annotations
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.ema import compute_ema


def _c(close: float) -> dict:
    return {'close': close}


class TestEMA(unittest.TestCase):
    def test_not_enough_data(self):
        self.assertIsNone(compute_ema([], 20))
        self.assertIsNone(compute_ema([_c(1.0)], 20))
        self.assertIsNone(compute_ema([_c(1.0)] * 19, 20))

    def test_sma_seed_then_ema(self):
        closes = [float(i) for i in range(20)]
        candles = [_c(c) for c in closes]
        ema = compute_ema(candles, 20)
        self.assertIsNotNone(ema)
        # SMA of 0..19 = 9.5
        self.assertEqual(ema, 9.5)

    def test_one_update_after_seed(self):
        candles = [_c(float(i)) for i in range(21)]
        ema20 = compute_ema(candles, 20)
        self.assertIsNotNone(ema20)
        k = 2.0 / 21.0
        expected = 20.0 * k + 9.5 * (1.0 - k)
        self.assertEqual(ema20, expected)

    def test_period_too_small(self):
        self.assertIsNone(compute_ema([_c(1.0)], 1))
        self.assertIsNone(compute_ema([_c(1.0)], 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
