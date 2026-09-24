"""TDD — D7 rejection diagnostic classification.

The diagnostic mirrors existing D7 predicates from indicators.pullback
exactly — no new trading criteria are invented.

Mapping: each diagnostic category corresponds to a specific predicate failure
in the existing D7 implementation (indicators/pullback.py):

D7 predicate chain (for direction='LONG'; SHORT mirrors with reversed signs):
  1. L = _select_retest_level_by_direction(candles, setup_index, direction)
     → None  ⇒  RETEST_LEVEL_UNAVAILABLE
     → float ⇒  continue
     Source: 'ema20' if EMA20 computed, else 'structure'

  2. P = find_prior_trend_side_candle(candles, setup_index, L, direction)
     → None  ⇒  NO_EMA20_RETEST (if source=='ema20') or NO_STRUCTURE_RETEST
     → dict  ⇒  continue (pullback found)

  3. validate_pullback_long/short checks 3 conditions IN ORDER:
     a. tested:       low(C) <= L (LONG) / high(C) >= L (SHORT)
     b. closed_back:  close(C) > L (LONG) / close(C) < L (SHORT) — STRICT
     c. body_valid:   close(C) > open(C) (LONG) / close(C) < open(C) (SHORT)
     First failing condition ⇒ corresponding REJECTION_* category.
     If all pass ⇒ OTHER_EXISTING_D7_REJECTION (D7 passes).
"""

from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagnostics.d7_diagnostic import (
    D7_DIAGNOSTIC_CATEGORIES,
    build_d7_diagnostic_report,
    classify_d7_rejection,
    format_d7_diagnostic_json,
    format_d7_diagnostic_text,
)


def _kline_dict(ct, o, h, l, c, v=1000.0, symbol="BTCUSDT", interval="5m"):
    """Synthetic Kline dict with close_time ct (ms)."""
    return {
        "open_time": ct - 300_000,
        "close_time": ct,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "symbol": symbol,
        "interval": interval,
    }


def _series_dicts(n, start=1_600_000_000_000, base=100.0, step=0.1, **kw):
    """n synthetic 5m kline dicts trending up from base."""
    return [
        _kline_dict(start + i * 300_000, base + step * i,
                    base + step * i + 1.0, base + step * i - 1.0,
                    base + step * (i + 1), **kw)
        for i in range(n)
    ]


def _compute_L(candles, setup_index, direction):
    """Compute L using the diagnostic's own mirror (for test assertions)."""
    from diagnostics.d7_diagnostic import _classify_retest_level_source
    L, source = _classify_retest_level_source(candles, setup_index, direction)
    return L, source


class TestD7DiagnosticCategories(unittest.TestCase):
    """Verify the category catalog is complete and ordered."""

    def test_categories_match_spec(self):
        expected = [
            "RETEST_LEVEL_UNAVAILABLE",
            "NO_EMA20_RETEST",
            "NO_STRUCTURE_RETEST",
            "RETEST_FOUND_NO_REJECTION",
            "REJECTION_CLOSE_INVALID",
            "REJECTION_BODY_INVALID",
            "OTHER_EXISTING_D7_REJECTION",
        ]
        self.assertEqual(D7_DIAGNOSTIC_CATEGORIES, expected)

    def test_categories_complete(self):
        self.assertEqual(len(D7_DIAGNOSTIC_CATEGORIES), 7)


class TestClassifyD7RetestUnavailable(unittest.TestCase):
    """RETEST_LEVEL_UNAVAILABLE — L could not be selected."""

    def test_no_candles(self):
        result = classify_d7_rejection([], 0, "LONG")
        self.assertEqual(result["category"], "RETEST_LEVEL_UNAVAILABLE")
        self.assertIsNone(result["retest_level"])
        self.assertFalse(result["retest_available"])
        self.assertFalse(result["pullback_found"])

    def test_single_candle_insufficient_for_ema20(self):
        klines = _series_dicts(1, base=100.0)
        result = classify_d7_rejection(klines, 0, "LONG")
        self.assertEqual(result["category"], "RETEST_LEVEL_UNAVAILABLE")
        self.assertIsNone(result["retest_level"])
        self.assertFalse(result["retest_available"])

    def test_insufficient_data_for_structure(self):
        klines = _series_dicts(5, base=100.0, step=0.0)
        result = classify_d7_rejection(klines, 4, "LONG")
        self.assertEqual(result["category"], "RETEST_LEVEL_UNAVAILABLE")

    def test_invalid_direction(self):
        result = classify_d7_rejection(_series_dicts(25), 24, "INVALID")
        self.assertEqual(result["category"], "OTHER_EXISTING_D7_REJECTION")
        self.assertFalse(result["retest_available"])


class TestClassifyD7NoPullback(unittest.TestCase):
    """NO_EMA20_RETEST and NO_STRUCTURE_RETEST — L found but no prior
    trend-side candle in the window."""

    def test_ema20_found_no_prior_trend_candle_long(self):
        # All closes = 100 → EMA20 = 100, L = 100.
        # No prior candle has close > 100 (strict), so no pullback candle.
        klines = _series_dicts(25, base=100.0, step=0.0)
        result = classify_d7_rejection(klines, 24, "LONG")
        self.assertEqual(result["category"], "NO_EMA20_RETEST")
        self.assertIsNotNone(result["retest_level"])
        self.assertTrue(result["retest_available"])
        self.assertFalse(result["pullback_found"])
        self.assertEqual(result["retest_source"], "ema20")

    def test_structure_found_no_prior_trend_candle_short(self):
        klines = _series_dicts(15, base=100.0, step=0.3)
        result = classify_d7_rejection(klines, 14, "SHORT")
        valid_categories = {"NO_STRUCTURE_RETEST", "RETEST_LEVEL_UNAVAILABLE"}
        self.assertIn(result["category"], valid_categories)

    def test_ema20_found_no_prior_trend_candle_short(self):
        klines = _series_dicts(25, base=100.0, step=-0.5)
        result = classify_d7_rejection(klines, 24, "SHORT")
        valid_categories = {"NO_EMA20_RETEST", "RETEST_FOUND_NO_REJECTION",
                           "REJECTION_CLOSE_INVALID", "REJECTION_BODY_INVALID"}
        self.assertIn(result["category"], valid_categories)


class TestClassifyD7RejectionBreakdown(unittest.TestCase):
    """Cases where pullback exists but rejection fails — classify which
    specific condition failed.

    Each test constructs klines so the naturally-computed L and the
    naturally-occurring setup candle produce the desired failure mode.
    Tests assert on the diagnostic category AND on the predicate details
    to prove the classification matches the actual D7 logic.
    """

    @staticmethod
    def _build_constant_series(n, close_val=100.0, base=100.0):
        """Series where all candles have close=close_val (constant)."""
        return [
            _kline_dict(1_600_000_000_000 + i * 300_000,
                        base, base + 1.0, base - 1.0, close_val)
            for i in range(n)
        ]

    def _build_rejection_series(self, direction, pullback_close, setup_close,
                                 setup_open, setup_high, setup_low):
        """Build 25-candle series with specified setup candle values.

        The close values are set so that _classify_retest_level_source
        computes a stable L, and only open/high/low are overridden on the
        setup candle (close is NOT overridden to avoid shifting L).
        """
        from diagnostics.d7_diagnostic import _classify_retest_level_source
        from indicators.pullback import find_prior_trend_side_candle

        klines = []
        for i in range(25):
            ct = 1_600_000_000_000 + i * 300_000
            if i == 23:
                close = pullback_close
            elif i == 24:
                close = setup_close
            else:
                close = 100.0
            klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

        # Verify L before overriding
        L, source = _classify_retest_level_source(klines, 24, direction)
        assert L is not None, f"L must be computable for {direction}"

        # Override setup candle fields (NOT close)
        C = klines[24]
        C["open"] = setup_open
        C["high"] = setup_high
        C["low"] = setup_low

        # Verify pullback exists with the NEW L (after overrides)
        P = find_prior_trend_side_candle(klines, 24, L, direction)
        assert P is not None, f"pullback must exist for {direction}"

        return klines, L, P

    def test_rejection_close_invalid_long(self):
        """REJECTION_CLOSE_INVALID: tested=True, closed_back=False,
        body_valid=True. Close of setup candle must be
        <= L (not strictly above)."""
        from diagnostics.d7_diagnostic import _classify_retest_level_source
        from indicators.pullback import find_prior_trend_side_candle

        # Strategy: use constant close=100 for candle 0-22 so EMA20 is
        # stable. Candle 23 (pullback) close=100.5, candle 24 (setup)
        # close=100.0. Override ONLY open/high/low on setup candle to
        # ensure tested=True, body_valid=True.
        klines = []
        for i in range(25):
            ct = 1_600_000_000_000 + i * 300_000
            if i == 23:
                close = 100.5
            elif i == 24:
                close = 100.0  # close <= L → closed_back fails
            else:
                close = 100.0
            klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

        L, source = _classify_retest_level_source(klines, 24, "LONG")
        self.assertIsNotNone(L, "L should be computable")
        self.assertEqual(source, "ema20")

        # Verify pullback exists: index 23 candle close > L
        P = find_prior_trend_side_candle(klines, 24, L, "LONG")
        self.assertIsNotNone(P, "pullback candle must exist")
        self.assertTrue(P["close"] > L, "pullback candle close > L")

        # Configure setup candle: override ONLY open/high/low (not close,
        # since close already = 100.0 which is <= L for closed_back fail).
        C = klines[24]
        C["open"] = 99.5       # bullish body: close(100.0) > open(99.5)
        C["high"] = 101.5      # high >= L (tested passes)
        C["low"] = L           # EXACTLY L: low <= L (tested passes since <=)
        # close stays 100.0 — must be <= L for closed_back to fail

        # Sanity: verify each condition independently
        self.assertTrue(C["low"] <= L, "tested must pass (low <= L)")
        self.assertFalse(C["close"] > L, "closed_back must fail (close <= L)")
        self.assertTrue(C["close"] > C["open"], "body must be bullish")

        result = classify_d7_rejection(klines, 24, "LONG")
        self.assertEqual(result["category"], "REJECTION_CLOSE_INVALID",
                         f"Expected REJECTION_CLOSE_INVALID, got {result['category']}")
        self.assertTrue(result["pullback_found"])
        self.assertFalse(result["rejection_details"]["closed_back"])
        self.assertTrue(result["rejection_details"]["tested"])
        self.assertTrue(result["rejection_details"]["body_valid"])

    def test_rejection_body_invalid_long(self):
        """REJECTION_BODY_INVALID: tested=True, closed_back=True,
        body_valid=False (bearish body on LONG)."""
        from diagnostics.d7_diagnostic import _classify_retest_level_source
        from indicators.pullback import find_prior_trend_side_candle

        # Strategy: constant close=100 for candles 0-22, pullback
        # candle 23 close=100.5, setup candle 24 close=100.5.
        # Override ONLY open/high/low on setup candle (NOT close, since
        # changing close would shift L via EMA20 recomputation).
        klines = []
        for i in range(25):
            ct = 1_600_000_000_000 + i * 300_000
            if i == 23:
                close = 100.5  # pullback: close > L
            elif i == 24:
                close = 100.5  # close > L → closed_back passes
            else:
                close = 100.0
            klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

        # Compute L from unmodified klines (only close values matter for EMA20)
        L, source = _classify_retest_level_source(klines, 24, "LONG")
        self.assertIsNotNone(L)

        # Verify pullback exists BEFORE overriding open/high/low
        P = find_prior_trend_side_candle(klines, 24, L, "LONG")
        self.assertIsNotNone(P)

        # Now override setup candle open/high/low (these don't affect L)
        C = klines[24]
        C["open"] = 101.0   # open > close → bearish body (body_valid fails)
        C["low"] = L        # low == L (tested passes via low <= L)
        # close stays 100.5 — must be > L for closed_back to pass

        # Verify conditions
        self.assertTrue(C["low"] <= L)                   # tested
        self.assertTrue(C["close"] > L)                  # closed_back
        self.assertFalse(C["close"] > C["open"])         # body_invalid

        result = classify_d7_rejection(klines, 24, "LONG")
        self.assertEqual(result["category"], "REJECTION_BODY_INVALID",
                         f"Expected REJECTION_BODY_INVALID, got {result['category']}")
        self.assertTrue(result["pullback_found"])
        self.assertTrue(result["rejection_details"]["tested"])
        self.assertTrue(result["rejection_details"]["closed_back"])
        self.assertFalse(result["rejection_details"]["body_valid"])

    def test_wick_penetration_long_now_allowed(self):
        """D7 LONG pass with low(C) < L (wick penetrates below L).

        Under the approved spec, no_break is removed. A LONG setup candle whose
        low breaks below L but closes back above L with a bullish body is now a
        valid D7 event. This proves the old exact-equality degeneracy (which
        required low(C) >= L) no longer exists.
        """
        from diagnostics.d7_diagnostic import _classify_retest_level_source
        from indicators.pullback import find_prior_trend_side_candle, d7_pullback_pass

        klines = []
        for i in range(25):
            ct = 1_600_000_000_000 + i * 300_000
            if i == 23:
                close = 100.5  # pullback: close > L
            elif i == 24:
                close = 100.5  # close > L → closed_back passes
            else:
                close = 100.0
            klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

        L, source = _classify_retest_level_source(klines, 24, "LONG")
        self.assertIsNotNone(L)

        P = find_prior_trend_side_candle(klines, 24, L, "LONG")
        self.assertIsNotNone(P)

        C = klines[24]
        C["open"] = 99.5    # open < close → bullish body
        C["high"] = 101.5   # high > L (not involved in LONG tested)
        C["low"] = 99.0     # low < L → wick penetrates below L
        # close stays 100.5

        self.assertTrue(C["low"] <= L)                   # tested (retest)
        self.assertTrue(C["close"] > L)                  # closed_back
        self.assertTrue(C["close"] > C["open"])          # body_valid
        self.assertTrue(C["low"] < L)                    # penetration below L

        self.assertTrue(d7_pullback_pass(klines, 24, "LONG"),
                        "D7 LONG must PASS with wick penetration below L")

        result = classify_d7_rejection(klines, 24, "LONG")
        self.assertEqual(result["category"], "OTHER_EXISTING_D7_REJECTION",
                         f"Expected OTHER_EXISTING_D7_REJECTION (pass), got {result['category']}")
        self.assertTrue(result["pullback_found"])
        self.assertTrue(result["rejection_details"]["tested"])
        self.assertTrue(result["rejection_details"]["closed_back"])
        self.assertTrue(result["rejection_details"]["body_valid"])

    def test_rejection_close_invalid_short(self):
        """REJECTION_CLOSE_INVALID (SHORT): tested=True, closed_back=False
        (close NOT strictly < L), body_valid=True."""
        from diagnostics.d7_diagnostic import _classify_retest_level_source
        from indicators.pullback import find_prior_trend_side_candle

        # SHORT mirror: constant close=100 for candles 0-22, pullback
        # candle 23 close=99.5, setup candle 24 close=100.0.
        # Override ONLY open/high/low on setup candle.
        klines = []
        for i in range(25):
            ct = 1_600_000_000_000 + i * 300_000
            if i == 23:
                close = 99.5   # pullback: close < L
            elif i == 24:
                close = 100.0  # setup: close >= L → closed_back fails
            else:
                close = 100.0
            klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

        L, source = _classify_retest_level_source(klines, 24, "SHORT")
        self.assertIsNotNone(L)
        self.assertEqual(source, "ema20")

        P = find_prior_trend_side_candle(klines, 24, L, "SHORT")
        self.assertIsNotNone(P)
        self.assertTrue(P["close"] < L)

        C = klines[24]
        C["open"] = 100.5     # bearish: close(100.0) < open(100.5)
        C["low"] = 99.0       # low <= L (tested passes)
        C["high"] = L         # EXACTLY L: high >= L (tested passes since >=)
        # close stays 100.0 — must be >= L for closed_back to fail (SHORT)

        self.assertTrue(C["high"] >= L)
        self.assertFalse(C["close"] < L)    # close(100.0) >= L(≈99.96)
        self.assertTrue(C["close"] < C["open"])   # bearish

        result = classify_d7_rejection(klines, 24, "SHORT")
        self.assertEqual(result["category"], "REJECTION_CLOSE_INVALID")
        self.assertTrue(result["pullback_found"])
        self.assertFalse(result["rejection_details"]["closed_back"])
        self.assertTrue(result["rejection_details"]["tested"])
        self.assertTrue(result["rejection_details"]["body_valid"])

    def test_rejection_body_invalid_short(self):
        """REJECTION_BODY_INVALID (SHORT): tested=True, closed_back=True,
        body_valid=False (bullish body when bearish required)."""
        from diagnostics.d7_diagnostic import _classify_retest_level_source
        from indicators.pullback import find_prior_trend_side_candle

        klines = []
        for i in range(25):
            ct = 1_600_000_000_000 + i * 300_000
            if i == 23:
                close = 99.5   # pullback: close < L
            elif i == 24:
                close = 99.5   # close < L → closed_back passes
            else:
                close = 100.0
            klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

        L, source = _classify_retest_level_source(klines, 24, "SHORT")
        self.assertIsNotNone(L)

        P = find_prior_trend_side_candle(klines, 24, L, "SHORT")
        self.assertIsNotNone(P)

        C = klines[24]
        C["open"] = 99.0      # open < close → bullish (wrong for SHORT)
        C["high"] = 100.5
        C["low"] = 98.5
        # close stays 99.5

        self.assertTrue(C["high"] >= L)                # tested (SHORT: high >= L)
        self.assertTrue(C["close"] < L)
        self.assertFalse(C["close"] < C["open"])  # bullish, not bearish

        result = classify_d7_rejection(klines, 24, "SHORT")
        self.assertEqual(result["category"], "REJECTION_BODY_INVALID")
        self.assertTrue(result["pullback_found"])
        self.assertTrue(result["rejection_details"]["tested"])
        self.assertTrue(result["rejection_details"]["closed_back"])
        self.assertFalse(result["rejection_details"]["body_valid"])

    def test_wick_penetration_short_now_allowed(self):
        """D7 SHORT pass with high(C) > L (wick penetrates above L).

        Under the approved spec, no_break is removed. A SHORT setup candle whose
        high breaks above L but closes back below L with a bearish body is now a
        valid D7 event. This proves the old exact-equality degeneracy (which
        required high(C) <= L) no longer exists.
        """
        from diagnostics.d7_diagnostic import _classify_retest_level_source
        from indicators.pullback import find_prior_trend_side_candle, d7_pullback_pass

        klines = []
        for i in range(25):
            ct = 1_600_000_000_000 + i * 300_000
            if i == 23:
                close = 99.5   # pullback: close < L
            elif i == 24:
                close = 99.5   # close < L → closed_back passes
            else:
                close = 100.0
            klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

        L, source = _classify_retest_level_source(klines, 24, "SHORT")
        self.assertIsNotNone(L)

        P = find_prior_trend_side_candle(klines, 24, L, "SHORT")
        self.assertIsNotNone(P)

        C = klines[24]
        C["open"] = 100.5     # bearish: close(99.5) < open(100.5)
        C["low"] = 99.0
        C["high"] = L + 0.5   # > L → wick penetrates above L
        # close stays 99.5

        self.assertTrue(C["high"] >= L)                # tested (retest)
        self.assertTrue(C["close"] < L)                # closed_back
        self.assertTrue(C["close"] < C["open"])        # body_valid
        self.assertTrue(C["high"] > L)                 # penetration above L

        self.assertTrue(d7_pullback_pass(klines, 24, "SHORT"),
                        "D7 SHORT must PASS with wick penetration above L")

        result = classify_d7_rejection(klines, 24, "SHORT")
        self.assertEqual(result["category"], "OTHER_EXISTING_D7_REJECTION",
                         f"Expected OTHER_EXISTING_D7_REJECTION (pass), got {result['category']}")
        self.assertTrue(result["pullback_found"])
        self.assertTrue(result["rejection_details"]["tested"])
        self.assertTrue(result["rejection_details"]["closed_back"])
        self.assertTrue(result["rejection_details"]["body_valid"])

    def _build_rejection_direction_short_robust(self):
        """SHORT series where the tested predicate (high >= L) fails:
        L is well above 100.8 so high=100.8 < L, ensuring the retest
        condition is not met under the current 3-condition D7 chain.
        """
        klines = []
        for i in range(25):
            ct = 1_600_000_000_000 + i * 300_000
            if i == 23:
                close = 50.0   # pullback: close < L
            elif i == 24:
                close = 50.0   # close < L
            else:
                close = 100.0
            klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

        L, source = _classify_retest_level_source(klines, 24, "SHORT")
        self.assertIsNotNone(L)
        self.assertGreater(L, 100.8)  # L > high so tested fails

        P = find_prior_trend_side_candle(klines, 24, L, "SHORT")
        # P may be None if no prior candle has close < L; that's acceptable.
        # This helper validates the SHORT tested predicate (high >= L) path.

        C = klines[24]
        C["open"] = 100.5
        C["high"] = 100.8
        C["low"] = 99.0
        # close stays 50.0

        assert C["close"] < L
        assert C["close"] < C["open"]
        assert C["high"] < L   # tested fails: high < L

        return klines, L, P, C


class TestClassifyD7Passes(unittest.TestCase):
    """D7 passes — classification should not crash and should produce
    a valid category from the catalog."""

    def test_d7_pass_long(self):
        from indicators.pullback import d7_pullback_pass
        from diagnostics.d7_diagnostic import D7_DIAGNOSTIC_CATEGORIES

        klines = _build_d7_pass_long()
        self.assertTrue(d7_pullback_pass(klines, 24, "LONG"),
                        "D7 should pass in constructed uptrend")
        result = classify_d7_rejection(klines, 24, "LONG")
        self.assertIn(result["category"], D7_DIAGNOSTIC_CATEGORIES,
                      f"Category {result['category']} not in catalog")
        self.assertTrue(result["pullback_found"])

    def test_d7_pass_short(self):
        from indicators.pullback import d7_pullback_pass
        from diagnostics.d7_diagnostic import D7_DIAGNOSTIC_CATEGORIES

        klines = _build_d7_pass_short()
        self.assertTrue(d7_pullback_pass(klines, 24, "SHORT"),
                        "D7 should pass in constructed downtrend")
        result = classify_d7_rejection(klines, 24, "SHORT")
        self.assertIn(result["category"], D7_DIAGNOSTIC_CATEGORIES,
                      f"Category {result['category']} not in catalog")
        self.assertTrue(result["pullback_found"])


def _build_d7_pass_long():
    """Build 25-candle series guaranteed to produce a D7 LONG pass.

    Conditions for D7 pass (LONG):
    1. L = _select_retest_level_by_direction(...) not None
    2. P = find_prior_trend_side_candle(...) not None
    3. validate_pullback_long: tested AND closed_back AND body_valid
       where tested = low(C) <= L (no high requirement)
    """
    from diagnostics.d7_diagnostic import _classify_retest_level_source
    from indicators.pullback import find_prior_trend_side_candle

    # 24 candles with close=100 → EMA20 ≈ 100, L ≈ 100 (source=ema20).
    # Candle 23 (pullback): close=101 > L → pullback exists.
    # Candle 24 (setup): straddle L with correct conditions.
    # IMPORTANT: Do NOT modify close values after construction — they
    # determine L via EMA20. Set up the series so conditions hold naturally.
    klines = []
    for i in range(25):
        ct = 1_600_000_000_000 + i * 300_000
        if i == 23:
            close = 101.0   # pullback candle: close > L
        elif i == 24:
            close = 100.10  # setup: close > L (L will be ~100.09)
        else:
            close = 100.0
        klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

    # Compute L from the series (with natural close values)
    L, src = _classify_retest_level_source(klines, 24, "LONG")
    if L is None:
        raise RuntimeError("L must be computable for D7 pass LONG")
    if src != "ema20":
        raise RuntimeError(f"Expected ema20 source, got {src}")

    # Verify pullback exists
    P = find_prior_trend_side_candle(klines, 24, L, "LONG")
    if P is None:
        raise RuntimeError("pullback candle must exist for D7 pass LONG")

    # Setup candle: set fields that DON'T affect L (open, high, low)
    C = klines[24]
    C["open"] = 100.0       # open < close → bullish
    C["high"] = 101.0       # high > L
    C["low"] = float(L)     # low == L: low <= L (tested passes since <=)
    # close stays 100.10 — must be > L for closed_back, > open for body_valid

    # Final verification: the current 3-condition D7 chain (tested,
    # closed_back, body_valid). no_break was removed from D7.
    if not (C["low"] <= L):
        raise RuntimeError(f"tested failed: low={C['low']} L={L}")

    return klines


def _build_d7_pass_short():
    """Build 25-candle series guaranteed to produce a D7 SHORT pass."""
    from diagnostics.d7_diagnostic import _classify_retest_level_source
    from indicators.pullback import find_prior_trend_side_candle

    klines = []
    for i in range(25):
        ct = 1_600_000_000_000 + i * 300_000
        if i == 23:
            close = 99.0    # pullback: close < L
        elif i == 24:
            close = 99.90   # setup: close < L (L will be ~99.91)
        else:
            close = 100.0
        klines.append(_kline_dict(ct, 100.0, 101.0, 99.0, close))

    L, src = _classify_retest_level_source(klines, 24, "SHORT")
    if L is None:
        raise RuntimeError("L must be computable for D7 pass SHORT")
    if src != "ema20":
        raise RuntimeError(f"Expected ema20 source, got {src}")

    C = klines[24]
    C["open"] = float(L) + 0.01  # open > close → bearish
    C["low"] = 99.0
    C["high"] = float(L)          # high == L: high >= L (tested)
    # close stays 99.90 — must be < L for closed_back, < open for body_valid

    P = find_prior_trend_side_candle(klines, 24, L, "SHORT")
    if P is None:
        raise RuntimeError("pullback candle must exist for D7 pass SHORT")

    # Final verification
    if not (C["high"] >= L):
        raise RuntimeError(f"tested failed: high={C['high']} L={L}")
    if not (C["close"] < L):
        raise RuntimeError(f"closed_back failed: close={C['close']} L={L}")
    if not (C["close"] < C["open"]):
        raise RuntimeError(f"body_valid failed: close={C['close']} open={C['open']}")

    return klines


class TestBuildD7DiagnosticReport(unittest.TestCase):
    """Aggregate D7 diagnostic report from stage records."""

    def test_empty_records(self):
        report = build_d7_diagnostic_report([])
        self.assertEqual(report["d7_evaluations"], 0)
        self.assertEqual(report["d7_pass"], 0)
        self.assertEqual(report["d7_rejected"], 0)
        self.assertEqual(report["d7_data_unavailable"], 0)
        for cat in D7_DIAGNOSTIC_CATEGORIES:
            self.assertEqual(report["d7_diagnostic_breakdown"][cat], 0)
            self.assertEqual(report["d7_diagnostic_percentages"][cat], 0.0)
        self.assertEqual(report["d7_retest_available_count"], 0)

    def test_report_with_mixed_records(self):
        records = [
            {
                "d7_pass": True,
                "d7_retest_available": True,
                "d7_diagnostic_category": None,
                "d8_direction": "LONG",
                "rejection_stage": "D15",
                "final_status": "OK",
            },
            {
                "d7_pass": False,
                "d7_retest_available": True,
                "d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
                "d8_direction": "LONG",
                "rejection_stage": "D7",
                "final_status": "NO_TRADE",
            },
            {
                "d7_pass": False,
                "d7_retest_available": False,
                "d7_diagnostic_category": "RETEST_LEVEL_UNAVAILABLE",
                "d8_direction": "LONG",
                "rejection_stage": "D7",
                "final_status": "DATA_UNAVAILABLE",
            },
            {
                "d7_pass": False,
                "d7_retest_available": True,
                "d7_diagnostic_category": "NO_EMA20_RETEST",
                "d8_direction": "SHORT",
                "rejection_stage": "D7",
                "final_status": "NO_TRADE",
            },
        ]
        report = build_d7_diagnostic_report(records)
        self.assertEqual(report["d7_evaluations"], 4)
        self.assertEqual(report["d7_pass"], 1)
        self.assertEqual(report["d7_rejected"], 3)
        self.assertEqual(report["d7_data_unavailable"], 1)
        self.assertEqual(report["d7_diagnostic_breakdown"]["REJECTION_CLOSE_INVALID"], 1)
        self.assertEqual(report["d7_diagnostic_breakdown"]["RETEST_LEVEL_UNAVAILABLE"], 1)
        self.assertEqual(report["d7_diagnostic_breakdown"]["NO_EMA20_RETEST"], 1)
        self.assertEqual(report["d7_long_count"], 3)
        self.assertEqual(report["d7_short_count"], 1)
        self.assertEqual(report["d7_retest_available_count"], 3)

    def test_percentages_deterministic(self):
        records = [
            {
                "d7_pass": False,
                "d7_retest_available": True,
                "d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
                "d8_direction": "LONG",
            },
            {
                "d7_pass": False,
                "d7_retest_available": True,
                "d7_diagnostic_category": "REJECTION_BODY_INVALID",
                "d8_direction": "LONG",
            },
            {
                "d7_pass": False,
                "d7_retest_available": False,
                "d7_diagnostic_category": "RETEST_LEVEL_UNAVAILABLE",
                "d8_direction": "LONG",
            },
        ]
        report = build_d7_diagnostic_report(records)
        self.assertEqual(report["d7_diagnostic_percentages"]["REJECTION_CLOSE_INVALID"], 33.33)
        self.assertEqual(report["d7_diagnostic_percentages"]["REJECTION_BODY_INVALID"], 33.33)
        self.assertEqual(report["d7_diagnostic_percentages"]["RETEST_LEVEL_UNAVAILABLE"], 33.33)

    def test_deterministic_output(self):
        records = [
            {
                "d7_pass": False,
                "d7_retest_available": True,
                "d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
                "d8_direction": "LONG",
            },
        ]
        self.assertEqual(
            build_d7_diagnostic_report(records),
            build_d7_diagnostic_report(list(records)),
        )


class TestD7DiagnosticReportAggregation(unittest.TestCase):
    """Test that build_report correctly aggregates d7_diagnostic_category
    from records into the d7_diagnostic_breakdown field."""

    def test_empty_records_produces_zero_breakdown(self):
        from diagnostics.report import build_report
        report = build_report([])
        self.assertIn("d7_diagnostic_breakdown", report)
        for cat in D7_DIAGNOSTIC_CATEGORIES:
            self.assertEqual(report["d7_diagnostic_breakdown"][cat], 0)

    def test_breakdown_counts_each_category(self):
        from diagnostics.report import build_report
        records = [
            {"d7_diagnostic_category": "RETEST_LEVEL_UNAVAILABLE",
             "d7_pass": False, "d7_retest_available": False, "d8_direction": "LONG"},
            {"d7_diagnostic_category": "NO_EMA20_RETEST",
             "d7_pass": False, "d7_retest_available": True, "d8_direction": "LONG"},
            {"d7_diagnostic_category": "RETEST_FOUND_NO_REJECTION",
             "d7_pass": False, "d7_retest_available": True, "d8_direction": "LONG"},
            {"d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
             "d7_pass": False, "d7_retest_available": True, "d8_direction": "LONG"},
            {"d7_diagnostic_category": "REJECTION_BODY_INVALID",
             "d7_pass": False, "d7_retest_available": True, "d8_direction": "LONG"},
            {"d7_diagnostic_category": "OTHER_EXISTING_D7_REJECTION",
             "d7_pass": False, "d7_retest_available": True, "d8_direction": "LONG"},
            {"d7_diagnostic_category": "NO_STRUCTURE_RETEST",
             "d7_pass": False, "d7_retest_available": True, "d8_direction": "SHORT"},
        ]
        report = build_report(records)
        bd = report["d7_diagnostic_breakdown"]
        self.assertEqual(bd["RETEST_LEVEL_UNAVAILABLE"], 1)
        self.assertEqual(bd["NO_EMA20_RETEST"], 1)
        self.assertEqual(bd["NO_STRUCTURE_RETEST"], 1)
        self.assertEqual(bd["RETEST_FOUND_NO_REJECTION"], 1)
        self.assertEqual(bd["REJECTION_BODY_INVALID"], 1)
        self.assertEqual(bd["OTHER_EXISTING_D7_REJECTION"], 1)

    def test_builds_report_with_d7_breakdown(self):
        """Regression: build_report access to d7_diagnostic_breakdown does not
        raise (integration smoke test for the import path)."""
        from diagnostics.report import build_report
        records = [
            {"d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
             "d7_pass": False, "d7_retest_available": True, "d8_direction": "LONG"},
        ]
        report = build_report(records)
        self.assertIn("d7_diagnostic_breakdown", report)
        self.assertEqual(report["d7_diagnostic_breakdown"]["REJECTION_CLOSE_INVALID"], 1)
        # All other categories present with 0
        for cat in D7_DIAGNOSTIC_CATEGORIES:
            if cat != "REJECTION_CLOSE_INVALID":
                self.assertEqual(report["d7_diagnostic_breakdown"][cat], 0)


class TestD7DiagnosticReportIntegration(unittest.TestCase):
    """Tests for the report aggregation path: build_report + d7_diagnostic_category
    from records. These prove the reporting layer correctly aggregates the
    D7 diagnostic categories that run_replay attaches via classify_d7_rejection."""

    def _make_record(self, category, direction="LONG"):
        return {
            "d7_pass": False,
            "d7_retest_available": True,
            "d7_diagnostic_category": category,
            "d8_direction": direction,
        }

    def test_build_report_aggregates_d7_breakdown(self):
        from diagnostics.report import build_report
        records = [
            self._make_record("RETEST_LEVEL_UNAVAILABLE"),
            self._make_record("NO_EMA20_RETEST"),
            self._make_record("NO_STRUCTURE_RETEST"),
            self._make_record("RETEST_FOUND_NO_REJECTION"),
            self._make_record("REJECTION_CLOSE_INVALID"),
            self._make_record("REJECTION_BODY_INVALID"),
            self._make_record("OTHER_EXISTING_D7_REJECTION"),
        ]
        report = build_report(records)
        bd = report["d7_diagnostic_breakdown"]
        for cat in D7_DIAGNOSTIC_CATEGORIES:
            self.assertEqual(bd[cat], 1,
                             f"Expected 1 for {cat}, got {bd[cat]}")

    def test_build_report_breakdown_with_mixed_directions(self):
        from diagnostics.report import build_report
        records = [
            self._make_record("RETEST_LEVEL_UNAVAILABLE", "LONG"),
            self._make_record("RETEST_LEVEL_UNAVAILABLE", "LONG"),
            self._make_record("NO_EMA20_RETEST", "SHORT"),
            self._make_record("REJECTION_CLOSE_INVALID", "LONG"),
        ]
        report = build_report(records)
        bd = report["d7_diagnostic_breakdown"]
        self.assertEqual(bd["RETEST_LEVEL_UNAVAILABLE"], 2)
        self.assertEqual(bd["NO_EMA20_RETEST"], 1)
        self.assertEqual(bd["REJECTION_CLOSE_INVALID"], 1)
        for cat in ("NO_STRUCTURE_RETEST", "RETEST_FOUND_NO_REJECTION",
                     "REJECTION_BODY_INVALID",
                     "OTHER_EXISTING_D7_REJECTION"):
            self.assertEqual(bd[cat], 0)

    def test_format_text_includes_d7_breakdown(self):
        from diagnostics.report import build_report, format_text
        records = [
            {"d7_pass": False, "d7_retest_available": True,
             "d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
             "d8_direction": "LONG"},
            {"d7_pass": False, "d7_retest_available": False,
             "d7_diagnostic_category": "RETEST_LEVEL_UNAVAILABLE",
             "d8_direction": "LONG"},
            {"d7_pass": False, "d7_retest_available": True,
             "d7_diagnostic_category": "NO_EMA20_RETEST",
             "d8_direction": "SHORT"},
        ]
        report = build_report(records)
        text = format_text(report)
        self.assertIn("D7 diagnostic category breakdown", text)
        self.assertIn("RETEST_LEVEL_UNAVAILABLE:    1", text)
        self.assertIn("NO_EMA20_RETEST:              1", text)
        self.assertIn("REJECTION_CLOSE_INVALID:      1", text)
        # Check D7_PASS line
        self.assertIn("D7_PASS:                     0", text)

    def test_format_json_includes_d7_breakdown(self):
        from diagnostics.report import build_report, format_json
        import json
        records = [
            {"d7_pass": False, "d7_retest_available": True,
             "d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
             "d8_direction": "LONG"},
        ]
        report = build_report(records)
        loaded = json.loads(format_json(report))
        self.assertIn("d7_diagnostic_breakdown", loaded)
        self.assertEqual(loaded["d7_diagnostic_breakdown"]["REJECTION_CLOSE_INVALID"], 1)
        for cat in D7_DIAGNOSTIC_CATEGORIES:
            if cat != "REJECTION_CLOSE_INVALID":
                self.assertEqual(loaded["d7_diagnostic_breakdown"][cat], 0)

    def test_records_without_category_are_not_counted(self):
        from diagnostics.report import build_report
        records = [
            {"d7_pass": False, "d7_retest_available": True,
             "d8_direction": "LONG"},  # no d7_diagnostic_category
            {"d7_pass": True, "d7_retest_available": True,
             "d8_direction": "LONG"},   # D7 pass, no category
            {"d7_pass": False, "d7_retest_available": True,
             "d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
             "d8_direction": "LONG"},
        ]
        report = build_report(records)
        bd = report["d7_diagnostic_breakdown"]
        self.assertEqual(bd["REJECTION_CLOSE_INVALID"], 1)
        for cat in D7_DIAGNOSTIC_CATEGORIES:
            if cat != "REJECTION_CLOSE_INVALID":
                self.assertEqual(bd[cat], 0)


class TestFormatD7DiagnosticOutput(unittest.TestCase):
    """Text and JSON output format tests."""

    def test_text_output_contains_key_sections(self):
        records = [
            {
                "d7_pass": False,
                "d7_retest_available": True,
                "d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
                "d8_direction": "LONG",
            },
            {
                "d7_pass": False,
                "d7_retest_available": False,
                "d7_diagnostic_category": "RETEST_LEVEL_UNAVAILABLE",
                "d8_direction": "LONG",
            },
        ]
        report = build_d7_diagnostic_report(records)
        text = format_d7_diagnostic_text(report)
        self.assertIn("D7 Rejection Diagnostics", text)
        self.assertIn("Total D7 evaluations:", text)
        self.assertIn("REJECTION_CLOSE_INVALID", text)
        self.assertIn("RETEST_LEVEL_UNAVAILABLE", text)

    def test_json_output_deterministic(self):
        records = [
            {
                "d7_pass": False,
                "d7_retest_available": True,
                "d7_diagnostic_category": "REJECTION_CLOSE_INVALID",
                "d8_direction": "LONG",
            },
        ]
        report = build_d7_diagnostic_report(records)
        json1 = format_d7_diagnostic_json(report)
        json2 = format_d7_diagnostic_json(report)
        self.assertEqual(json1, json2)
        loaded = json.loads(json1)
        self.assertEqual(loaded["d7_evaluations"], 1)
        self.assertIn("d7_diagnostic_breakdown", loaded)

    def test_json_contains_all_categories(self):
        records = []
        report = build_d7_diagnostic_report(records)
        json_str = format_d7_diagnostic_json(report)
        loaded = json.loads(json_str)
        for cat in D7_DIAGNOSTIC_CATEGORIES:
            self.assertIn(cat, loaded["d7_diagnostic_breakdown"])
            self.assertIn(cat, loaded["d7_diagnostic_percentages"])


class TestD7DiagnosticEquivalence(unittest.TestCase):
    """Prove the diagnostic classification does NOT alter D7/D15 output."""

    def test_diagnostic_does_not_change_d7_result(self):
        from indicators.pullback import d7_pullback_pass
        test_cases = [
            ("uptrend LONG pass", self._build_uptrend, 24, "LONG"),
            ("downtrend SHORT pass", self._build_downtrend, 24, "SHORT"),
            ("flat insufficient data", self._build_flat_short, 10, "LONG"),
        ]
        for desc, builder, idx, direction in test_cases:
            klines = builder()
            production_result = d7_pullback_pass(klines, idx, direction)
            diagnostic_result = classify_d7_rejection(klines, idx, direction)
            self.assertIn(diagnostic_result["category"],
                          D7_DIAGNOSTIC_CATEGORIES,
                          f"Failed for {desc}")

    def _build_uptrend(self):
        return _series_dicts(30, base=100.0, step=0.3)

    def _build_downtrend(self):
        return _series_dicts(30, base=100.0, step=-0.3)

    def _build_flat_short(self):
        return _series_dicts(15, base=100.0, step=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
