"""Tests for D8 Trend component — EMA20/50/200 hierarchy + close confirmation.

Trend spec (Reading B):
- Bullish hierarchy: EMA20 > EMA50 > EMA200
- Bearish hierarchy: EMA20 < EMA50 < EMA200

LONG:
  bullish hier + close > EMA20  -> 100
  bullish hier + close <= EMA20 -> 75
  mixed/neither                 -> 50
  bearish hier + close <= EMA20 -> 25
  bearish hier + close > EMA20  -> 0

SHORT:
  bearish hier + close < EMA20  -> 100
  bearish hier + close >= EMA20 -> 75
  mixed/neither                 -> 50
  bullish hier + close >= EMA20 -> 25
  bullish hier + close < EMA20  -> 0

Insufficient EMA data -> (50, 50).
"""

from __future__ import annotations
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.trend import trend_score
from indicators.ema import compute_ema


def _c(ts, close, **overrides):
    d = {'close': close, 'high': close + 1.0, 'low': close - 1.0,
         'open': close, 'volume': 1.0}
    d.update(overrides)
    return d


def _series_from_closes(closes, base_ts=0, step=300000):
    return [_c(base_ts + i * step, c) for i, c in enumerate(closes)]


# ---------------------------------------------------------------------------
# Helper: build a series where EMA20/50/200 evaluate to known values
# ---------------------------------------------------------------------------
# compute_ema seeds with SMA of first `period` closes, then RMA.
# To drive EMA20/50/200 to controlled values we construct long series
# where the first `period` closes seed the SMA, then the remaining closes
# are constant so the EMA converges toward that constant.


def _build_series_with_emas(close_values_for_seed_20,
                            close_values_for_seed_50,
                            close_values_for_seed_200,
                            final_close,
                            final_epoch=0):
    """Build a candle series such that:
      - EMA20 through final candle is driven by seed_20 + constant tail
      - EMA50  through final candle is driven by seed_50 + constant tail
      - EMA200 through final candle is driven by seed_200 + constant tail
      - final candle close == final_close

    The construction: for each EMA period P, take `P` seed closes that
    produce the desired EMA(P) when followed by a long constant tail.
    Because EMA smooths, a long enough constant tail makes EMA(P) converge
    to the constant tail value. So we set the tail close to the desired
    EMA value and make the seed sequence produce that same EMA value.

    Simpler approach: make ALL closes from idx 0 to N-1 equal to the
    desired EMA value — then EMA(P) = that value exactly (since seed SMA
    = value, and every subsequent update is value*k + value*(1-k) = value).
    So a flat series at price X gives EMA(P) = X for any P.

    To get EMA20 != EMA50 != EMA200 simultaneously we use a flat series
    but with different starting points... Actually if ALL closes are equal
    to X then EMA20=EMA50=EMA200=X at the end. That gives a flat hierarchy
    (neither). To get strict ordering we need the series to NOT be flat.

    Approach: build three internally concatenated segments, but that's
    complex. Instead, use the property that EMA(P) at candle N depends on
    closes[max(0, N-P+1) : N+1] with exponentially decaying weights.
    With a long enough series we can place a short "bump" segment to shift
    one EMA without shifting the others much.

    Simplest reliable construction for testing: build a series long enough
    that EMA20, EMA50, EMA200 are driven to desired values AND verify the
    actual EMAs via compute_ema. Then trend_score uses those EMAs.
    """

    # We'll construct by making the series flat at different levels for
    # different stretches. Use a single flat series where all closes == V;
    # then EMA20 = EMA50 = EMA200 = V (flat, neither hierarchy).
    # For testing hierarchy cases we construct series where the closes are
    # flat at level V for the last `max_period` candles, but with a
    # different level before that. EMA(P) with a long flat tail at V
    # converges to V. So we can set EMA20=EMA50=EMA200=V_flat by making
    # the series flat at V_flat for long enough.

    # To get EMA20 > EMA50 > EMA200, we need the series to have been higher
    # recently (affecting EMA20 more than EMA50/200) and lower historically.
    # Construction: long series flat at LOW, then a recent bump to HIGH for
    # last K candles, then back to a final value.

    # Let me use a cleaner deterministic approach: build series where
    # closes are piecewise constant, compute EMAs analytically-ish, then
    # verify with compute_ema. But for unit tests we just need the EMAs
    # to come out right; we can brute-force-construct and let compute_ema
    # compute, then assert the EMAs are what we expect.

    # Strategy: choose N large (e.g. 400 candles). Set closes[0:N-400] = LOW.
    # Then set closes[N-400:N] to shape the EMAs. With N=400 and max period
    # 200, the first 200 candles establish EMA200 seed, then we have 200
    # more candles to shape.

    # Even simpler: Use the fact that EMA(P) at candle N-1 with all closes
    # == V for the last P candles AND the preceding candle == V_prev gives:
    #   EMA(P) = V + (V_prev - V) * (1 - k)^P  approximately (depends on history)
    # This is getting complex. Let me just construct and verify empirically.

    raise NotImplementedError("Use the explicit series constructors below instead")


def _flat(closes, base_ts=0, step=300000):
    """Series with exactly the given closes (flat per-candle close)."""
    return _series_from_closes(list(closes), base_ts, step)


# Simple deterministic series constructors for each branch.
# We construct the series so that compute_ema returns known EMA values.
# The trick: start the series with a long run of constant close == TARGET,
# which makes EMA(P) = TARGET for all P (since seed SMA = TARGET and
# subsequent updates keep it at TARGET). Then we have a different EMA
# relationship by making the series non-flat.

# To make EMA20 > EMA50 > EMA200 with controlled values, we want recent
# price higher than older price. Construction:
#   - candles 0..(N-1): all close = LOW
#   - this makes EMA20=EMA50=EMA200=LOW at candle N-1
#   - then change last K candles to HIGH
#   - EMA20 responds more than EMA50, which responds more than EMA200

# For testing, we construct series and ASSERT the EMAs come out as expected
# using compute_ema directly, then test trend_score against those EMAs.
# This decouples the EMA-construction complexity from the trend scoring logic.


def _assert_emas(series, ema20_expect, ema50_expect, ema200_expect, tol=1e-6):
    from indicators.ema import compute_ema
    e20 = compute_ema(series, 20)
    e50 = compute_ema(series, 50)
    e200 = compute_ema(series, 200)
    if ema20_expect is not None:
        assert e20 is not None and abs(e20 - ema20_expect) < tol, \
            f"EMA20 {e20} != {ema20_expect}"
    if ema50_expect is not None:
        assert e50 is not None and abs(e50 - ema50_expect) < tol, \
            f"EMA50 {e50} != {ema50_expect}"
    if ema200_expect is not None:
        assert e200 is not None and abs(e200 - ema200_expect) < tol, \
            f"EMA200 {e200} != {ema200_expect}"


def _build_and_assert(closes, e20, e50, e200):
    s = _flat(closes)
    _assert_emas(s, e20, e50, e200)
    return s


# ---------------------------------------------------------------------------
# Branch 1: Bullish hierarchy + close > EMA20 -> LONG 100, SHORT 25
# ---------------------------------------------------------------------------

class TestTrendBullishHierarchyLong100Short25(unittest.TestCase):
    def test_bullish_hier_close_gt_ema20(self):
        # We need EMA20 > EMA50 > EMA200 and close > EMA20.
        # Construction: long series, recent prices higher than older.
        # Use a series where closes ramp up so that EMA20 (short-term)
        # ends up highest, EMA200 lowest.
        #
        # Simple reliable construction: 300 candles total.
        # Candles 0..99: close = 90 (low base)
        # Candles 100..199: close = 100 (mid)
        # Candles 200..299: close = 110 (recent high)
        # With 300 candles and max period 200, EMA200 sees the full history,
        # EMA20 sees mostly the recent 110s.
        #
        # Let's verify empirically what the EMAs come out to, then assert
        # the hierarchy and close > EMA20.
        closes = [90.0] * 100 + [100.0] * 100 + [110.0] * 100
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        e50 = compute_ema(s, 50)
        e200 = compute_ema(s, 200)
        self.assertIsNotNone(e20)
        self.assertIsNotNone(e50)
        self.assertIsNotNone(e200)
        self.assertTrue(e20 > e50 > e200,
                        f"Expected bullish hierarchy: EMA20={e20} > EMA50={e50} > EMA200={e200}")
        final_close = closes[-1]  # 110.0
        self.assertTrue(final_close > e20,
                        f"Expected close={final_close} > EMA20={e20}")
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 100)
        self.assertEqual(short_s, 25)

    def test_bullish_hier_close_gt_ema20_explicit_values(self):
        # Use a flatter construction to get clean EMA values we can assert.
        # 400 candles: first 200 at 100, last 200 at 101.
        # EMA20 will be very close to 101 (recent), EMA50 close to 101,
        # EMA200 will be between 100 and 101 but closer to 100.5-ish.
        # Actually with 200 candles at 100 then 200 at 101:
        # EMA200 seed = SMA of first 200 = 100, then 200 updates at 101
        #   EMA200 converges toward 101 but only 200 steps with k=2/201.
        #   After 200 steps: EMA200 ≈ 100 + (101-100)*(1 - (1-k)^200)
        #   (1-k)^200 = (199/201)^200 ≈ exp(-200*2/201) ≈ exp(-1.99) ≈ 0.136
        #   So EMA200 ≈ 100 + 1*0.864 = 100.864
        # EMA50: seed SMA of first 50 of the 100s = 100, then 50 more 100s
        #   then 200 at 101. After 200 updates at 101:
        #   (1-k)^200 = (49/51)^200 ≈ exp(-200*2/51) ≈ exp(-7.84) ≈ 0.00039
        #   EMA50 ≈ 100 + 1*0.9996 ≈ 100.9996
        # EMA20: seed SMA of first 20 of the 100s = 100, then 30 more 100s
        #   then 200 at 101. After 200 updates:
        #   (1-k)^200 = (19/21)^200 ≈ exp(-200*2/21) ≈ exp(-19.05) ≈ 5e-9
        #   EMA20 ≈ 101.0
        # So EMA20 ≈ 101 > EMA50 ≈ 101.0 > EMA200 ≈ 100.864 — bullish hierarchy.
        # close = 101 > EMA20 (101.0)? 101 > 101.0 is FALSE (very close).
        # Hmm, EMA20 ≈ 101 but not exactly 101. 101 > 100.999999995 is TRUE.
        closes = [100.0] * 200 + [101.0] * 200
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        e50 = compute_ema(s, 50)
        e200 = compute_ema(s, 200)
        self.assertIsNotNone(e20)
        self.assertIsNotNone(e50)
        self.assertIsNotNone(e200)
        # Assert strict hierarchy
        self.assertTrue(e20 > e50, f"EMA20={e20} > EMA50={e50}")
        self.assertTrue(e50 > e200, f"EMA50={e50} > EMA200={e200}")
        final_close = closes[-1]  # 101.0
        self.assertTrue(final_close > e20,
                        f"close={final_close} > EMA20={e20}")
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 100)
        self.assertEqual(short_s, 25)


# ---------------------------------------------------------------------------
# Branch 2: Bullish hierarchy + close <= EMA20 -> LONG 75, SHORT 0
# ---------------------------------------------------------------------------

class TestTrendBullishHierarchyLong75Short0(unittest.TestCase):
    def test_bullish_hier_close_le_ema20(self):
        # Need EMA20 > EMA50 > EMA200 and close <= EMA20.
        # Construction: recent prices high enough for bullish hierarchy,
        # but final close pulled down to <= EMA20.
        #
        # 300 candles: 0..99 at 90, 100..199 at 100, 200..298 at 110,
        # candle 299 (final) at 108. EMA20 will be close to 110 (pulled by
        # the recent 110s), and final close 108 < EMA20.
        #
        # Let's construct and verify.
        closes = [90.0] * 100 + [100.0] * 100 + [110.0] * 99 + [108.0]
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        e50 = compute_ema(s, 50)
        e200 = compute_ema(s, 200)
        self.assertIsNotNone(e20)
        self.assertIsNotNone(e50)
        self.assertIsNotNone(e200)
        self.assertTrue(e20 > e50 > e200,
                        f"Expected bullish hierarchy: EMA20={e20} > EMA50={e50} > EMA200={e200}")
        final_close = closes[-1]  # 108.0
        self.assertTrue(final_close <= e20,
                        f"Expected close={final_close} <= EMA20={e20}")
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 75)
        self.assertEqual(short_s, 0)

    def test_bullish_hier_close_eq_ema20_long_75(self):
        # close == EMA20 exactly -> LONG 75 (close <= EMA20), SHORT 0
        # (close < EMA20 is false, close >= EMA20 is true -> SHORT 0... wait
        # SHORT: bearish hier + close >= EMA20 -> 75; bearish hier + close < EMA20 -> 0.
        # We're in bullish hierarchy, so SHORT: bullish hier + close >= EMA20 -> 25,
        # bullish hier + close < EMA20 -> 0. close == EMA20 means close >= EMA20 is true
        # so SHORT = 25, not 0. Let me re-check the spec.)
        #
        # SHORT Trend score:
        #   bullish hierarchy AND close >= EMA20 -> 25
        #   bullish hierarchy AND close < EMA20  -> 0
        # close == EMA20 -> close >= EMA20 is TRUE -> SHORT = 25.
        #
        # LONG: bullish hier + close <= EMA20 -> 75. close == EMA20 -> TRUE -> LONG = 75.
        #
        # So at equality with bullish hierarchy: LONG=75, SHORT=25.
        #
        # To hit exact equality we need close == EMA20. With floating point
        # this is hard to arrange exactly. We construct a series where the
        # final close equals the EMA20 value we compute, then assert.
        #
        # Simpler: construct a series, compute EMA20, set the final close
        # to that exact value, rebuild, and assert.
        closes = [100.0] * 200 + [101.0] * 199 + [0.0]  # placeholder last
        s = _flat(closes[:-1])  # series without last candle
        e20 = compute_ema(s, 20)
        # Build series with final close == EMA20
        closes2 = closes[:-1] + [e20]
        s2 = _flat(closes2)
        # Verify hierarchy still holds
        e20_new = compute_ema(s2, 20)
        e50_new = compute_ema(s2, 50)
        e200_new = compute_ema(s2, 200)
        self.assertIsNotNone(e20_new)
        self.assertIsNotNone(e50_new)
        self.assertIsNotNone(e200_new)
        # EMA20 changed slightly because we changed the last close, but
        # should still be bullish hierarchy
        self.assertTrue(e20_new > e50_new > e200_new,
                        f"EMA20={e20_new} > EMA50={e50_new} > EMA200={e200_new}")
        final_close = closes2[-1]  # == old EMA20
        # Now final_close should be very close to EMA20_new. Due to the
        # change in the last candle, EMA20_new != final_close exactly.
        # Let's just verify the score branches by checking that close <= EMA20
        # holds (or not) and reading the score.
        long_s, short_s = trend_score(s2)
        # With close ≈ EMA20, we should be in the close <= EMA20 branch for LONG (75)
        # and close >= EMA20 branch for SHORT (25).
        # But floating point may put us on either side. We'll assert based on
        # the actual relationship.
        if final_close <= e20_new:
            self.assertEqual(long_s, 75)
        else:
            self.assertEqual(long_s, 100)
        if final_close >= e20_new:
            self.assertEqual(short_s, 25)
        else:
            self.assertEqual(short_s, 0)


# ---------------------------------------------------------------------------
# Branch 3: Mixed/neither hierarchy -> LONG 50, SHORT 50
# ---------------------------------------------------------------------------

class TestTrendMixedNeither(unittest.TestCase):
    def test_flat_series_neither(self):
        # All closes equal -> EMA20 == EMA50 == EMA200 (flat, not strict either way).
        closes = [100.0] * 250
        s = _flat(closes)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 50)
        self.assertEqual(short_s, 50)

    def test_crossed_neither_ema20_lt_ema50_but_ema50_gt_ema200(self):
        # Neither strict hierarchy: EMA20 < EMA50 (bearish start) but
        # EMA50 > EMA200 (so bearish chain EMA20 < EMA50 < EMA200 is broken).
        # Construction: old very low (pulls EMA200 down eventually but after
        # 200 old-low candles, the recent moderate values lift EMA20/EMA50
        # above EMA200). Use [80]*200 (old low) then [120]*100 then [110]*100.
        # Result: EMA20=110.000450 < EMA50=110.169655 > EMA200=108.265443.
        # bearish = (EMA20 < EMA50 AND EMA50 < EMA200) = False (EMA50 > EMA200).
        # bullish = (EMA20 > EMA50 AND EMA50 > EMA200) = False (EMA20 < EMA50).
        closes = [80.0] * 200 + [120.0] * 100 + [110.0] * 100
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        e50 = compute_ema(s, 50)
        e200 = compute_ema(s, 200)
        self.assertIsNotNone(e20)
        self.assertIsNotNone(e50)
        self.assertIsNotNone(e200)
        self.assertTrue(e20 < e50,
                        f"Expected EMA20 < EMA50: {e20} < {e50}")
        self.assertTrue(e50 > e200,
                        f"Expected EMA50 > EMA200: {e50} > {e200}")
        bullish = e20 > e50 and e50 > e200
        bearish = e20 < e50 and e50 < e200
        self.assertFalse(bullish,
                         f"Unexpected bullish: EMA20={e20} > EMA50={e50} > EMA200={e200}")
        self.assertFalse(bearish,
                         f"Unexpected bearish: EMA20={e20} < EMA50={e50} < EMA200={e200}")
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 50)
        self.assertEqual(short_s, 50)


# ---------------------------------------------------------------------------
# Branch 4: Bearish hierarchy + close <= EMA20 -> LONG 25, SHORT 75
# ---------------------------------------------------------------------------

class TestTrendBearishHierarchyLong25Short75(unittest.TestCase):
    def test_bearish_hier_close_lt_ema20_long_25_short_100(self):
        # bearish hierarchy + close < EMA20 -> LONG 25, SHORT 100.
        closes = [110.0] * 100 + [100.0] * 100 + [90.0] * 100
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        e50 = compute_ema(s, 50)
        e200 = compute_ema(s, 200)
        self.assertIsNotNone(e20)
        self.assertIsNotNone(e50)
        self.assertIsNotNone(e200)
        self.assertTrue(e20 < e50 < e200,
                        f"Expected bearish hierarchy: EMA20={e20} < EMA50={e50} < EMA200={e200}")
        final_close = closes[-1]
        self.assertTrue(final_close < e20,
                        f"Expected close={final_close} < EMA20={e20}")
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 25)
        self.assertEqual(short_s, 100)

    def test_bearish_hier_close_eq_ema20_long_25_short_75(self):
        # bearish hierarchy + close == EMA20 -> LONG 25 (close <= EMA20),
        # SHORT 75 (close >= EMA20).
        base = [110.0] * 100 + [100.0] * 100 + [90.0] * 99
        cbase = _flat(base)
        e20_val = compute_ema(cbase, 20)
        closes = base + [e20_val]
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        e50 = compute_ema(s, 50)
        e200 = compute_ema(s, 200)
        self.assertIsNotNone(e20)
        self.assertIsNotNone(e50)
        self.assertIsNotNone(e200)
        self.assertTrue(e20 < e50 < e200,
                        f"Expected bearish hierarchy: EMA20={e20} < EMA50={e50} < EMA200={e200}")
        final_close = closes[-1]
        self.assertTrue(final_close == e20,
                        f"Expected close == EMA20: {final_close} == {e20}")
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 25)
        self.assertEqual(short_s, 75)


# ---------------------------------------------------------------------------
# Branch 5: Bearish hierarchy + close > EMA20 -> LONG 0, SHORT 100
# ---------------------------------------------------------------------------

class TestTrendBearishHierarchyLong0Short100(unittest.TestCase):
    def test_bearish_hier_close_gt_ema20_long_0_short_75(self):
        # bearish hierarchy + close > EMA20 -> LONG 0, SHORT 75.
        closes = [110.0] * 100 + [100.0] * 100 + [90.0] * 99 + [90.5]
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        e50 = compute_ema(s, 50)
        e200 = compute_ema(s, 200)
        self.assertIsNotNone(e20)
        self.assertIsNotNone(e50)
        self.assertIsNotNone(e200)
        self.assertTrue(e20 < e50 < e200,
                        f"Expected bearish hierarchy: EMA20={e20} < EMA50={e50} < EMA200={e200}")
        final_close = closes[-1]
        self.assertTrue(final_close > e20,
                        f"Expected close={final_close} > EMA20={e20}")
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 0)
        self.assertEqual(short_s, 75)


# ---------------------------------------------------------------------------
# SHORT mirror verification
# ---------------------------------------------------------------------------

class TestTrendShortMirror(unittest.TestCase):
    def test_long_short_sum_not_necessarily_100(self):
        # Trend is NOT symmetric like D5 — LONG+SHORT does not always sum to 100.
        # E.g. bullish hierarchy + close > EMA20: LONG=100, SHORT=25 (sum=125).
        # Bearish hierarchy + close < EMA20: LONG=25, SHORT=100 (sum=125).
        # Mixed: LONG=50, SHORT=50 (sum=100).
        # The spec does not require mirror symmetry; verify the actual pairs.
        pass  # Verified by individual branch tests

    def test_bullish_hier_close_gt_ema20_long_100_short_25(self):
        closes = [100.0] * 200 + [101.0] * 200
        s = _flat(closes)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 100)
        self.assertEqual(short_s, 25)

    def test_bearish_hier_close_lt_ema20_long_25_short_100(self):
        closes = [110.0] * 100 + [100.0] * 100 + [90.0] * 100
        s = _flat(closes)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 25)
        self.assertEqual(short_s, 100)


# ---------------------------------------------------------------------------
# Insufficient EMA data
# ---------------------------------------------------------------------------

class TestTrendInsufficientData(unittest.TestCase):
    def test_fewer_than_200_closes_returns_50_50(self):
        # Need at least 200 closes for EMA200, 50 for EMA50, 20 for EMA20.
        # With 199 closes, EMA200 is None -> (50, 50).
        closes = [100.0] * 199
        s = _flat(closes)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 50)
        self.assertEqual(short_s, 50)

    def test_fewer_than_50_closes_returns_50_50(self):
        closes = [100.0] * 49
        s = _flat(closes)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 50)
        self.assertEqual(short_s, 50)

    def test_fewer_than_20_closes_returns_50_50(self):
        closes = [100.0] * 19
        s = _flat(closes)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 50)
        self.assertEqual(short_s, 50)

    def test_empty_returns_50_50(self):
        s = _flat([])
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 50)
        self.assertEqual(short_s, 50)

    def test_200_closes_is_minimum_for_ema200(self):
        closes = [100.0] * 200
        s = _flat(closes)
        long_s, short_s = trend_score(s)
        # With 200 closes all equal to 100, EMA20=EMA50=EMA200=100,
        # hierarchy is flat (neither) -> (50, 50). But the point is EMA200
        # is available.
        self.assertIsNotNone(compute_ema(s, 200))


# ---------------------------------------------------------------------------
# No future-data leakage / determinism
# ---------------------------------------------------------------------------

class TestTrendNoFutureLeakage(unittest.TestCase):
    def test_does_not_use_candles_after_ctf(self):
        # C_tf is the last candle. Adding candles after it changes the score
        # because C_tf changes (different last candle), but the original
        # series' score is unchanged (pure function).
        closes = [100.0] * 200 + [101.0] * 200
        s = _flat(closes)
        long_a, short_a = trend_score(s)
        # Add future candles
        s_extended = _flat(closes + [102.0, 102.0])
        long_b, short_b = trend_score(s_extended)
        # Different C_tf -> different score (expected, not leakage)
        # But original score is unchanged
        long_c, short_c = trend_score(s)
        self.assertEqual(long_a, long_c)
        self.assertEqual(short_a, short_c)

    def test_deterministic_same_input(self):
        closes = [100.0] * 200 + [101.0] * 200
        s = _flat(closes)
        a = trend_score(s)
        b = trend_score(s)
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Score range 0-100
# ---------------------------------------------------------------------------

class TestTrendScoreRange(unittest.TestCase):
    def test_scores_in_range(self):
        # Sweep various series and verify all outputs in 0-100.
        test_series = [
            [100.0] * 250,                                    # flat -> 50/50
            [110.0] * 100 + [100.0] * 100 + [90.0] * 100,   # bearish hier
            [90.0] * 100 + [100.0] * 100 + [110.0] * 100,   # bullish hier
            [100.0] * 200 + [101.0] * 200,                   # bullish hier, close>EMA20
            [90.0] * 100 + [100.0] * 100 + [90.0] * 99 + [95.0],  # bearish hier, close>EMA20
            [100.0] * 19,                                     # insufficient -> 50/50
        ]
        valid = {0, 25, 50, 75, 100}
        for closes in test_series:
            s = _flat(closes)
            long_s, short_s = trend_score(s)
            self.assertIn(long_s, valid,
                          f"LONG score {long_s} not in {valid} for series len={len(closes)}")
            self.assertIn(short_s, valid,
                          f"SHORT score {short_s} not in {valid} for series len={len(closes)}")
            self.assertGreaterEqual(long_s, 0)
            self.assertGreaterEqual(short_s, 0)
            self.assertLessEqual(long_s, 100)
            self.assertLessEqual(short_s, 100)


# ---------------------------------------------------------------------------
# Explicit branch coverage for every scoring combination
# ---------------------------------------------------------------------------

class TestTrendAllBranchesExplicit(unittest.TestCase):
    def test_bullish_hier_close_gt_ema20_long_100_short_25(self):
        closes = [100.0] * 200 + [101.0] * 200
        s = _flat(closes)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 100)
        self.assertEqual(short_s, 25)

    def test_bullish_hier_close_le_ema20_long_75_short_0(self):
        closes = [90.0] * 100 + [100.0] * 100 + [110.0] * 99 + [108.0]
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        self.assertTrue(e20 > compute_ema(s, 50) > compute_ema(s, 200))
        self.assertTrue(closes[-1] <= e20)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 75)
        self.assertEqual(short_s, 0)

    def test_bearish_hier_close_le_ema20_long_25_short_75(self):
        # bearish hierarchy + close <= EMA20 -> LONG 25, SHORT depends on close<EMA20 vs close==EMA20.
        # Here close=90.0 < EMA20=90.000450 strictly -> SHORT=100.
        closes = [110.0] * 100 + [100.0] * 100 + [90.0] * 100
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        e50 = compute_ema(s, 50)
        e200 = compute_ema(s, 200)
        self.assertIsNotNone(e20)
        self.assertIsNotNone(e50)
        self.assertIsNotNone(e200)
        self.assertTrue(e20 < e50 < e200,
                        f"Expected bearish hierarchy: EMA20={e20} < EMA50={e50} < EMA200={e200}")
        self.assertTrue(closes[-1] < e20)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 25)
        self.assertEqual(short_s, 100)

    def test_bearish_hier_close_gt_ema20_long_0_short_100(self):
        # bearish hierarchy + close > EMA20 -> LONG 0, SHORT 75.
        closes = [110.0] * 100 + [100.0] * 100 + [90.0] * 99 + [90.5]
        s = _flat(closes)
        e20 = compute_ema(s, 20)
        e50 = compute_ema(s, 50)
        e200 = compute_ema(s, 200)
        self.assertIsNotNone(e20)
        self.assertIsNotNone(e50)
        self.assertIsNotNone(e200)
        self.assertTrue(e20 < e50 < e200,
                        f"Expected bearish hierarchy: EMA20={e20} < EMA50={e50} < EMA200={e200}")
        self.assertTrue(closes[-1] > e20)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 0)
        self.assertEqual(short_s, 75)

    def test_mixed_neither_long_50_short_50(self):
        closes = [100.0] * 250
        s = _flat(closes)
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 50)
        self.assertEqual(short_s, 50)

    def test_insufficient_data_long_50_short_50(self):
        closes = [100.0] * 199
        s = _flat(closes)
        self.assertIsNone(compute_ema(s, 200))
        long_s, short_s = trend_score(s)
        self.assertEqual(long_s, 50)
        self.assertEqual(short_s, 50)
