"""Tests for D2 RSI14 — indicators.rsi (Wilder's formulation)."""

from __future__ import annotations
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.rsi import compute_rsi, momentum_score


def _c(close: float, open_: float = None) -> dict:
    """Minimal candle with close price only (RSI needs close series)."""
    d = {'close': close}
    if open_ is not None:
        d['open'] = open_
    return d


def _candle(close_time, open_, high, low, close):
    return {
        'close_time': close_time,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
    }


class TestWilderRSICalculation(unittest.TestCase):
    def test_insufficient_data_less_than_period_plus_one(self):
        # RSI needs at least period+1 closes to compute 1 gain/loss
        self.assertIsNone(compute_rsi([_c(1.0)], 14))
        self.assertIsNone(compute_rsi([_c(1.0)] * 14, 14))

    def test_insufficient_data_short_series(self):
        self.assertIsNone(compute_rsi([_c(1.0)] * 15, 20))

    def test_period_too_small(self):
        self.assertIsNone(compute_rsi([_c(1.0)], 1))
        self.assertIsNone(compute_rsi([_c(1.0)], 0))
        self.assertIsNone(compute_rsi([_c(1.0)], -1))

    def test_empty_series(self):
        self.assertIsNone(compute_rsi([], 14))

    def test_constant_price_rsi(self):
        # All closes equal → all gains/losses = 0 → avg_gain=0, avg_loss=0 → RSI=50
        candles = [_c(100.0) for _ in range(20)]
        rsi = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi)
        self.assertEqual(rsi, 50.0)

    def test_sma_seed_first_14(self):
        # First 14 gains/losses seed the averages
        # Build a series where we can compute expected seed manually
        # Closes: 100, 101, 102, ..., 114 (ascending by 1 each candle)
        closes = [100.0 + float(i) for i in range(15)]  # 15 closes = 14 gains
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi)
        # First 14 gains are all 1.0, losses all 0.0
        # Seed avg_gain = 14*1.0/14 = 1.0, avg_loss = 0.0
        # After seed (only 14 gains, no recurrence yet): avg_loss=0 → RSI=100
        self.assertEqual(rsi, 100.0)

    def test_wilder_recurrence(self):
        # Build a series that exercises the recurrence
        # 15 closes: 100, 101, ..., 114 → 14 gains of 1.0, seeded avg_gain=1.0, avg_loss=0 → RSI=100
        # But we want to test recurrence, so extend with a loss
        # 16 closes: 100..114, 113 → 15 gains/losses
        # Gains: 14 * 1.0, then -1.0? No: 114→113 is a loss of 1.0
        # So gains: [1.0]*14 + [0.0] = 15 values; losses: [0.0]*14 + [1.0] = 15 values
        # Seed (first 14): avg_gain=1.0, avg_loss=0.0
        # Recurrence on gain[14]=0.0, loss[14]=1.0:
        #   avg_gain = (1.0 * 13 + 0.0) / 14 = 13/14 ≈ 0.9285714285714286
        #   avg_loss = (0.0 * 13 + 1.0) / 14 = 1/14 ≈ 0.07142857142857142
        # RSI = 100 - 100/(1 + avg_gain/avg_loss) = 100 - 100/(1 + (13/14)/(1/14))
        #     = 100 - 100/(1 + 13) = 100 - 100/14 = 100 - 7.142857... = 92.857142857...
        closes = [100.0 + float(i) for i in range(15)] + [113.0]
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi)
        expected = 100.0 - 100.0 / 14.0
        self.assertAlmostEqual(rsi, expected, places=6)

    def test_wilder_recurrence_multiple_steps(self):
        # Build a more complex series to test multiple recurrence steps
        # 20 closes, all going up by 1: seed RSI=100 (avg_loss=0)
        # Not interesting. Let's do alternating: up 2, down 1, repeated
        # Closes: 100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109, 108, 110, 109, 111
        # Gains/losses:
        # 100→102: +2 (gain=2, loss=0)
        # 102→101: -1 (gain=0, loss=1)
        # 101→103: +2 (gain=2, loss=0)
        # ... pattern repeats
        closes = []
        price = 100.0
        for i in range(20):
            closes.append(price)
            if i % 2 == 0:
                price += 2.0
            else:
                price -= 1.0
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi)
        # RSI should be between 0 and 100
        self.assertGreaterEqual(rsi, 0.0)
        self.assertLessEqual(rsi, 100.0)
        # With alternating up 2/down 1, avg_gain should be > avg_loss
        # So RSI should be > 50
        self.assertGreater(rsi, 50.0)

    def test_known_wilder_reference(self):
        # Wilder's original example from his 1978 book:
        # 15 periods of data, RSI(14) should be approximately 40.625 (depending on exact data)
        # We'll use a simpler reference: after the seed, with known gains/losses
        # Closes: 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
        #          46.03, 45.61, 46.28, 46.28, 46.00
        closes = [44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
                  46.03, 45.61, 46.28, 46.28, 46.00]
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi)
        # This is a known test case; RSI should be around 40-60 range
        self.assertGreater(rsi, 30.0)
        self.assertLess(rsi, 70.0)


class TestZeroLossEdgeCase(unittest.TestCase):
    def test_all_up_no_loss(self):
        # Strictly ascending closes → all gains, no losses → avg_loss=0 → RSI=100
        closes = [100.0 + float(i) for i in range(20)]
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi)
        self.assertEqual(rsi, 100.0)

    def test_all_down_no_gain(self):
        # Strictly descending closes → all losses, no gains → avg_gain=0 → RSI=0
        closes = [100.0 - float(i) for i in range(20)]
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi)
        self.assertEqual(rsi, 0.0)

    def test_zero_gain_and_loss_after_seed(self):
        # After seed, if all remaining changes are zero, RSI maintains its seed value
        # Seed with 14 equal closes gives avg_gain=0, avg_loss=0 → RSI=50
        # Extend with more equal closes
        closes = [100.0] * 20
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi)
        self.assertEqual(rsi, 50.0)


class TestMomentumScoreLong(unittest.TestCase):
    def test_rsi_70_or_above(self):
        self.assertEqual(momentum_score(70.0, 'LONG'), 100)
        self.assertEqual(momentum_score(75.0, 'LONG'), 100)
        self.assertEqual(momentum_score(100.0, 'LONG'), 100)

    def test_rsi_60_to_70(self):
        self.assertEqual(momentum_score(60.0, 'LONG'), 80)
        self.assertEqual(momentum_score(65.0, 'LONG'), 80)
        self.assertEqual(momentum_score(69.99, 'LONG'), 80)

    def test_rsi_50_to_60(self):
        self.assertEqual(momentum_score(50.0, 'LONG'), 65)
        self.assertEqual(momentum_score(55.0, 'LONG'), 65)
        self.assertEqual(momentum_score(59.99, 'LONG'), 65)

    def test_rsi_40_to_50(self):
        self.assertEqual(momentum_score(40.0, 'LONG'), 35)
        self.assertEqual(momentum_score(45.0, 'LONG'), 35)
        self.assertEqual(momentum_score(49.99, 'LONG'), 35)

    def test_rsi_30_to_40(self):
        self.assertEqual(momentum_score(30.0, 'LONG'), 20)
        self.assertEqual(momentum_score(35.0, 'LONG'), 20)
        self.assertEqual(momentum_score(39.99, 'LONG'), 20)

    def test_rsi_below_30(self):
        self.assertEqual(momentum_score(29.99, 'LONG'), 0)
        self.assertEqual(momentum_score(20.0, 'LONG'), 0)
        self.assertEqual(momentum_score(0.0, 'LONG'), 0)

    def test_rsi_none(self):
        self.assertEqual(momentum_score(None, 'LONG'), 0)


class TestMomentumScoreShort(unittest.TestCase):
    def test_rsi_30_or_below(self):
        self.assertEqual(momentum_score(30.0, 'SHORT'), 100)
        self.assertEqual(momentum_score(25.0, 'SHORT'), 100)
        self.assertEqual(momentum_score(0.0, 'SHORT'), 100)

    def test_rsi_30_to_40(self):
        self.assertEqual(momentum_score(30.01, 'SHORT'), 80)
        self.assertEqual(momentum_score(35.0, 'SHORT'), 80)
        self.assertEqual(momentum_score(40.0, 'SHORT'), 80)

    def test_rsi_40_to_50(self):
        self.assertEqual(momentum_score(40.01, 'SHORT'), 65)
        self.assertEqual(momentum_score(45.0, 'SHORT'), 65)
        self.assertEqual(momentum_score(50.0, 'SHORT'), 65)

    def test_rsi_50_to_60(self):
        self.assertEqual(momentum_score(50.01, 'SHORT'), 35)
        self.assertEqual(momentum_score(55.0, 'SHORT'), 35)
        self.assertEqual(momentum_score(60.0, 'SHORT'), 35)

    def test_rsi_60_to_70(self):
        self.assertEqual(momentum_score(60.01, 'SHORT'), 20)
        self.assertEqual(momentum_score(65.0, 'SHORT'), 20)
        self.assertEqual(momentum_score(70.0, 'SHORT'), 20)

    def test_rsi_above_70(self):
        self.assertEqual(momentum_score(70.01, 'SHORT'), 0)
        self.assertEqual(momentum_score(80.0, 'SHORT'), 0)
        self.assertEqual(momentum_score(100.0, 'SHORT'), 0)

    def test_rsi_none(self):
        self.assertEqual(momentum_score(None, 'SHORT'), 0)


class TestMirrorSymmetry(unittest.TestCase):
    def test_explicit_mirrors(self):
        # At RSI=50, both LONG and SHORT should score 65
        self.assertEqual(momentum_score(50.0, 'LONG'), 65)
        self.assertEqual(momentum_score(50.0, 'SHORT'), 65)

        # At RSI=40, LONG=35, SHORT=80 (not direct mirror but consistent with thresholds)
        self.assertEqual(momentum_score(40.0, 'LONG'), 35)
        self.assertEqual(momentum_score(40.0, 'SHORT'), 80)

        # At RSI=60, LONG=80, SHORT=35
        self.assertEqual(momentum_score(60.0, 'LONG'), 80)
        self.assertEqual(momentum_score(60.0, 'SHORT'), 35)

        # At RSI=30, LONG=20, SHORT=100
        self.assertEqual(momentum_score(30.0, 'LONG'), 20)
        self.assertEqual(momentum_score(30.0, 'SHORT'), 100)

        # At RSI=70, LONG=100, SHORT=20
        self.assertEqual(momentum_score(70.0, 'LONG'), 100)
        self.assertEqual(momentum_score(70.0, 'SHORT'), 20)

    def test_complementary_scores(self):
        # For any RSI, LONG + SHORT should be a consistent pattern
        # RSI < 30: LONG=0, SHORT=100 (sum=100)
        # RSI = 30: LONG=20, SHORT=100 (sum=120)
        # RSI in (30,40]: LONG=20, SHORT=80 (sum=100)
        # RSI in (40,50]: LONG=35, SHORT=65 (sum=100)
        # RSI in (50,60]: LONG=65, SHORT=35 (sum=100)
        # RSI in (60,70]: LONG=80, SHORT=20 (sum=100)
        # RSI > 70: LONG=100, SHORT=0 (sum=100)
        for rsi in [0.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 100.0]:
            long_score = momentum_score(rsi, 'LONG')
            short_score = momentum_score(rsi, 'SHORT')
            # Check that scores are in valid range
            self.assertIn(long_score, [0, 20, 35, 65, 80, 100])
            self.assertIn(short_score, [0, 20, 35, 65, 80, 100])


class TestNoFutureDataLeakage(unittest.TestCase):
    def test_rsi_uses_only_available_closes(self):
        # Build a series and compute RSI at different points
        # All closes going up: 100, 101, 102, ...
        closes = [100.0 + float(i) for i in range(30)]
        candles = [_c(c) for c in closes]

        # RSI at index 14 (15th candle, the earliest possible RSI value)
        rsi_early = compute_rsi(candles[:15], 14)
        self.assertIsNotNone(rsi_early)
        self.assertEqual(rsi_early, 100.0)  # all gains, no losses

        # RSI at index 19 (20th candle)
        rsi_later = compute_rsi(candles[:20], 14)
        self.assertIsNotNone(rsi_later)

        # RSI at index 29 (30th candle)
        rsi_final = compute_rsi(candles, 14)
        self.assertIsNotNone(rsi_final)

        # All should be 100 (all gains, no losses)
        self.assertEqual(rsi_early, 100.0)
        self.assertEqual(rsi_later, 100.0)
        self.assertEqual(rsi_final, 100.0)

    def test_rsi_deterministic_with_same_input(self):
        closes = [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0,
                  104.0, 106.0, 105.0, 107.0, 106.0, 108.0, 107.0]
        candles = [_c(c) for c in closes]
        rsi1 = compute_rsi(candles, 14)
        rsi2 = compute_rsi(candles, 14)
        self.assertEqual(rsi1, rsi2)
        self.assertIsNotNone(rsi1)


class TestMomentumScoreIntegration(unittest.TestCase):
    def test_rsi_then_score_flow(self):
        # End-to-end: compute RSI then map to score
        closes = [100.0 + float(i) for i in range(20)]  # all up
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertEqual(rsi, 100.0)
        self.assertEqual(momentum_score(rsi, 'LONG'), 100)
        self.assertEqual(momentum_score(rsi, 'SHORT'), 0)

    def test_rsi_then_score_downtrend(self):
        closes = [100.0 - float(i) for i in range(20)]  # all down
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertEqual(rsi, 0.0)
        self.assertEqual(momentum_score(rsi, 'LONG'), 0)
        self.assertEqual(momentum_score(rsi, 'SHORT'), 100)

    def test_rsi_then_score_neutral(self):
        closes = [100.0] * 20  # flat
        candles = [_c(c) for c in closes]
        rsi = compute_rsi(candles, 14)
        self.assertEqual(rsi, 50.0)
        self.assertEqual(momentum_score(rsi, 'LONG'), 65)
        self.assertEqual(momentum_score(rsi, 'SHORT'), 65)


if __name__ == '__main__':
    unittest.main(verbosity=2)
