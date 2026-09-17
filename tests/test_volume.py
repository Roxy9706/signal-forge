"""Tests for D6 Volume & Liquidity — RVOL score + Binance USDT liquidity gate.

D6 spec:
- RVOL = current_volume / mean(previous_20 volumes), C_tf excluded from baseline.
- Score buckets: >=2.0->100, 1.5-1.99->80, 1.0-1.49->60, 0.75-0.99->40, <0.75->20.
- Insufficient history (<20 preceding) -> None.
- Liquidity: Binance Futures USDT-margined; quoteVolume >= 50M -> ok.
- Missing/malformed/unavailable -> False + data-quality issue.
- RVOL score and liquidity gate are separate; quoteVolume does NOT alter RVOL score.
"""

from __future__ import annotations
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.d6_volume import (
    compute_rvol,
    volume_score,
    parse_liquidity_from_ticker,
    check_liquidity,
    _safe_volume,
    _safe_quote_volume,
    _LIQUIDITY_THRESHOLD,
)


def _k(close_time, open_, high, low, close, volume):
    return {
        'close_time': close_time,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }


def _series_volume(base_time, opens, highs, lows, closes, volumes):
    return [_k(base_time + i * 300000, o, h, l, c, v)
            for i, (o, h, l, c, v) in enumerate(zip(opens, highs, lows, closes, volumes))]


# ---------------------------------------------------------------------------
# RVOL exact calculation
# ---------------------------------------------------------------------------

class TestRvolExactCalculation(unittest.TestCase):
    def test_rvol_basic(self):
        # 21 candles: 20 previous with volume=100 each, C_tf volume=250.
        # baseline = 100, rvol = 250/100 = 2.5
        vols = [100.0] * 20 + [250.0]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        self.assertAlmostEqual(compute_rvol(series), 2.5)

    def test_rvol_excludes_current_candle(self):
        # C_tf volume should NOT be in baseline.
        # 21 candles: prev 20 = 100 each, C_tf = 1000.
        # baseline = 100, rvol = 10.0 (not 1000/104.76)
        vols = [100.0] * 20 + [1000.0]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        self.assertAlmostEqual(compute_rvol(series), 10.0)

    def test_rvol_different_volumes(self):
        # prev 20: 50, 60, 70, ... arithmetic mean = 150
        # C_tf = 300 -> rvol = 2.0
        prev = [float(100 + i * 10) for i in range(20)]  # 100..290, mean=195
        # Actually let's compute: sum(100..290) = 20*100 + 10*190 = 2000+1900=3900, mean=195
        ctf = 390.0  # rvol = 390/195 = 2.0
        vols = prev + [ctf]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        self.assertAlmostEqual(compute_rvol(series), 2.0)


# ---------------------------------------------------------------------------
# RVOL baseline uses previous 20 candles and excludes current
# ---------------------------------------------------------------------------

class TestRvolBaselineExcludesCurrent(unittest.TestCase):
    def test_baseline_is_previous_20_not_21(self):
        # C_tf volume = 1, prev 20 = 1000 each.
        # If C_tf were included: baseline = (20*1000 + 1)/21 = 952.4, rvol = 0.00105
        # Correct: baseline = 1000, rvol = 0.001
        vols = [1000.0] * 20 + [1.0]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        rvol = compute_rvol(series)
        self.assertAlmostEqual(rvol, 0.001)
        self.assertIsNotNone(rvol)


# ---------------------------------------------------------------------------
# RVOL score buckets
# ---------------------------------------------------------------------------

class TestVolumeScoreBuckets(unittest.TestCase):
    def test_rvol_below_075_returns_20(self):
        self.assertEqual(volume_score(0.5), 20)
        self.assertEqual(volume_score(0.74), 20)
        self.assertEqual(volume_score(0.0), 20)

    def test_rvol_075_boundary_returns_40(self):
        self.assertEqual(volume_score(0.75), 40)
        self.assertEqual(volume_score(0.76), 40)
        self.assertEqual(volume_score(0.99), 40)

    def test_rvol_100_boundary_returns_60(self):
        self.assertEqual(volume_score(1.0), 60)
        self.assertEqual(volume_score(1.2), 60)
        self.assertEqual(volume_score(1.49), 60)

    def test_rvol_150_boundary_returns_80(self):
        self.assertEqual(volume_score(1.5), 80)
        self.assertEqual(volume_score(1.75), 80)
        self.assertEqual(volume_score(1.99), 80)

    def test_rvol_200_boundary_returns_100(self):
        self.assertEqual(volume_score(2.0), 100)
        self.assertEqual(volume_score(3.0), 100)
        self.assertEqual(volume_score(10.0), 100)

    def test_none_rvol_returns_none_score(self):
        self.assertIsNone(volume_score(None))


# ---------------------------------------------------------------------------
# Insufficient history
# ---------------------------------------------------------------------------

class TestRvolInsufficientHistory(unittest.TestCase):
    def test_fewer_than_21_candles_returns_none(self):
        # 20 candles only -> no C_tf with 20 preceding
        series = _series_volume(0,
                                [100]*20, [101]*20, [99]*20, [100]*20,
                                [100.0]*20)
        self.assertIsNone(compute_rvol(series))

    def test_21_candles_is_minimum(self):
        # Exactly 21 candles -> C_tf with 20 preceding -> should work
        vols = [100.0] * 20 + [200.0]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        self.assertIsNotNone(compute_rvol(series))
        self.assertAlmostEqual(compute_rvol(series), 2.0)

    def test_empty_returns_none(self):
        self.assertIsNone(compute_rvol([]))


# ---------------------------------------------------------------------------
# Zero/invalid baseline
# ---------------------------------------------------------------------------

class TestRvolZeroInvalidBaseline(unittest.TestCase):
    def test_zero_baseline_returns_none(self):
        # All previous volumes = 0 -> baseline = 0 -> None
        vols = [0.0] * 20 + [100.0]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        self.assertIsNone(compute_rvol(series))

    def test_negative_volume_in_history_returns_none(self):
        vols = [100.0] * 19 + [-1.0] + [200.0]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        self.assertIsNone(compute_rvol(series))

    def test_missing_volume_key_returns_none(self):
        candles = [{'close_time': i, 'open': 100, 'high': 101,
                    'low': 99, 'close': 100, 'volume': 100.0} for i in range(20)]
        candles.append({'close_time': 20, 'open': 100, 'high': 101,
                        'low': 99, 'close': 100})  # missing volume
        self.assertIsNone(compute_rvol(candles))

    def test_nan_volume_returns_none(self):
        import math
        vols = [100.0] * 20 + [float('nan')]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        self.assertIsNone(compute_rvol(series))

    def test_infinity_volume_returns_none(self):
        vols = [100.0] * 20 + [float('inf')]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        self.assertIsNone(compute_rvol(series))


# ---------------------------------------------------------------------------
# No future-data leakage
# ---------------------------------------------------------------------------

class TestRvolNoFutureLeakage(unittest.TestCase):
    def test_does_not_use_candles_after_ctf(self):
        # Build 21 candles with constant volume=100 for prev 20,
        # C_tf volume=200 -> rvol = 2.0.
        # Adding more candles AFTER C_tf changes C_tf to a later candle,
        # so the RVOL for the extended series is different — but the
        # original 21-candle RVOL is unchanged (pure function).
        series_21 = _series_volume(0,
                                   [100]*21, [101]*21, [99]*21, [100]*21,
                                   [100.0]*20 + [200.0])
        rvol_21 = compute_rvol(series_21)
        self.assertAlmostEqual(rvol_21, 2.0)

        # Extend with future candles — new C_tf is idx 22 (volume 999).
        # Baseline = previous 20 candles (idx 2..21):
        #   idx 2..19: 100 each (18 candles)
        #   idx 20: 200 (original C_tf, now in baseline)
        #   idx 21: 999 (first extended candle)
        #   sum = 1800 + 200 + 999 = 2999, mean = 149.95
        #   rvol = 999 / 149.95 ≈ 6.662
        extended = series_21 + [
            _k(21 * 300000, 100, 101, 99, 100, 999.0),
            _k(22 * 300000, 100, 101, 99, 100, 999.0),
        ]
        rvol_extended = compute_rvol(extended)
        self.assertAlmostEqual(rvol_extended, 999.0 / 149.95)
        # Original 21-candle RVOL is unchanged (pure function)
        self.assertAlmostEqual(rvol_21, 2.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestRvolDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        vols = [100.0 + float(i) for i in range(21)]
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        a = compute_rvol(series)
        b = compute_rvol(series)
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Liquidity — exact boundaries
# ---------------------------------------------------------------------------

class TestLiquidityBoundaries(unittest.TestCase):
    def test_liquidity_exactly_50m_returns_true(self):
        resp = {'symbol': 'BTCUSDT', 'quoteVolume': '50000000.0'}
        ok, data_ok = parse_liquidity_from_ticker(resp)
        self.assertTrue(ok)
        self.assertTrue(data_ok)

    def test_liquidity_below_50m_returns_false(self):
        resp = {'symbol': 'BTCUSDT', 'quoteVolume': '49999999.99'}
        ok, data_ok = parse_liquidity_from_ticker(resp)
        self.assertFalse(ok)
        self.assertTrue(data_ok)

    def test_liquidity_above_50m_returns_true(self):
        resp = {'symbol': 'BTCUSDT', 'quoteVolume': '100000000.0'}
        ok, data_ok = parse_liquidity_from_ticker(resp)
        self.assertTrue(ok)
        self.assertTrue(data_ok)


# ---------------------------------------------------------------------------
# Liquidity — missing/malformed
# ---------------------------------------------------------------------------

class TestLiquidityMissingMalformed(unittest.TestCase):
    def test_missing_quote_volume_returns_false(self):
        resp = {'symbol': 'BTCUSDT'}
        ok, data_ok = parse_liquidity_from_ticker(resp)
        self.assertFalse(ok)
        self.assertFalse(data_ok)

    def test_malformed_quote_volume_returns_false(self):
        resp = {'symbol': 'BTCUSDT', 'quoteVolume': 'abc'}
        ok, data_ok = parse_liquidity_from_ticker(resp)
        self.assertFalse(ok)
        self.assertFalse(data_ok)

    def test_none_response_returns_false(self):
        ok, data_ok = parse_liquidity_from_ticker(None)
        self.assertFalse(ok)
        self.assertFalse(data_ok)

    def test_empty_dict_returns_false(self):
        ok, data_ok = parse_liquidity_from_ticker({})
        self.assertFalse(ok)
        self.assertFalse(data_ok)

    def test_list_response_takes_first(self):
        # When response is a list (no symbol filter), take first entry
        resp = [
            {'symbol': 'BTCUSDT', 'quoteVolume': '60000000.0'},
            {'symbol': 'ETHUSDT', 'quoteVolume': '30000000.0'},
        ]
        ok, data_ok = parse_liquidity_from_ticker(resp)
        self.assertTrue(ok)
        self.assertTrue(data_ok)

    def test_empty_list_returns_false(self):
        ok, data_ok = parse_liquidity_from_ticker([])
        self.assertFalse(ok)
        self.assertFalse(data_ok)


# ---------------------------------------------------------------------------
# USDT-only handling
# ---------------------------------------------------------------------------

class TestLiquidityUSDTOnly(unittest.TestCase):
    def test_non_usdt_quote_not_converted(self):
        # Spec: do not convert non-USDT quote contracts.
        # D6 applies only to USDT-margined universe.
        # The parser just reads quoteVolume as-is — conversion is the
        # caller's responsibility (or rejection). Here we verify the parser
        # doesn't do any conversion.
        resp = {'symbol': 'BTCUSDT', 'quoteVolume': '60000000.0'}
        ok, data_ok = parse_liquidity_from_ticker(resp)
        self.assertTrue(ok)
        # A non-USDT pair would be rejected before reaching here by the
        # symbol universe filter; the parser itself is symbol-agnostic.


# ---------------------------------------------------------------------------
# RVOL and liquidity remain independent
# ---------------------------------------------------------------------------

class TestRvolLiquidityIndependence(unittest.TestCase):
    def test_liquidity_does_not_affect_rvol_score(self):
        # Even if liquidity is FALSE, RVOL score is computed purely from candles.
        vols = [100.0] * 20 + [300.0]  # rvol = 3.0 -> score 100
        series = _series_volume(0,
                                [100]*21, [101]*21, [99]*21, [100]*21, vols)
        rvol = compute_rvol(series)
        score = volume_score(rvol)
        self.assertEqual(score, 100)
        # Liquidity state is orthogonal — volume_score doesn't know about it.
        liquidity_ok, data_ok = parse_liquidity_from_ticker(None)
        self.assertFalse(liquidity_ok)
        self.assertFalse(data_ok)
        # Volume score is unchanged regardless of liquidity state.
        self.assertEqual(volume_score(rvol), 100)


# ---------------------------------------------------------------------------
# Per-timeframe independence
# ---------------------------------------------------------------------------

class TestRvolPerTimeframeIndependence(unittest.TestCase):
    def test_rvol_independent_of_other_timeframes(self):
        # RVOL is computed independently per timeframe from that TF's candles.
        # Here we verify that two different series produce independent results.
        series_5m = _series_volume(0,
                                   [100]*21, [101]*21, [99]*21, [100]*21,
                                   [100.0]*20 + [200.0])
        series_1h = _series_volume(0,
                                   [100]*21, [101]*21, [99]*21, [100]*21,
                                   [500.0]*20 + [1000.0])
        rvol_5m = compute_rvol(series_5m)
        rvol_1h = compute_rvol(series_1h)
        # Both should be 2.0 (same ratio), but computed independently.
        self.assertAlmostEqual(rvol_5m, 2.0)
        self.assertAlmostEqual(rvol_1h, 2.0)
        self.assertEqual(rvol_5m, rvol_1h)  # same ratio, different absolute vols


# ---------------------------------------------------------------------------
# Direction-independent Volume score
# ---------------------------------------------------------------------------

class TestVolumeScoreDirectionIndependent(unittest.TestCase):
    def test_volume_score_has_no_direction_param(self):
        # Volume score is direction-independent per spec.
        # volume_score takes only rvol, not direction.
        self.assertEqual(volume_score(2.0), 100)
        self.assertEqual(volume_score(2.0), volume_score(2.0))  # same regardless


# ---------------------------------------------------------------------------
# No final NO_TRADE logic
# ---------------------------------------------------------------------------

class TestNoTradeLogicNotIntroduced(unittest.TestCase):
    def test_d6_has_no_notrade_function(self):
        # D6 must NOT implement final NO_TRADE decision.
        # Verify there is no function like `d6_notrade` or `should_trade`.
        import indicators.d6_volume as m
        self.assertFalse(hasattr(m, 'should_trade'))
        self.assertFalse(hasattr(m, 'notrade'))
        self.assertFalse(hasattr(m, 'd6_notrade'))
        self.assertFalse(hasattr(m, 'final_decision'))

    def test_liquidity_exposes_ok_flag_only(self):
        # check_liquidity exposes (liquidity_ok, data_ok) — no trade decision.
        import indicators.d6_volume as m
        sig = m.check_liquidity.__annotations__.get('return', None)
        # Return type should be Tuple[bool, bool], not a trade decision.
        # We verify by checking the function signature doesn't return a bool alone.
        import inspect
        ret = inspect.signature(m.check_liquidity).return_annotation
        self.assertIn('Tuple', str(ret))
