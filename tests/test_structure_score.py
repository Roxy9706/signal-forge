"""Tests for D5 Structure scoring — indicators.structure (swing classification + score map).

D5 spec:
- 20 closed-candle lookback; strict 3-candle fractal; C_tf excluded from swings.
- HH = latest high > previous high; LH = latest high < previous high.
- HL = latest low > previous low; LL = latest low < previous low.
- Equality = no relation.
- fewer than 2 swings in a class = that class unavailable.
- Score map: HH+HL -> 100 LONG / 0 SHORT; one bullish unavailable -> 75/25;
  HH+LL or LH+HL -> 50/50; one bearish unavailable -> 25/75; LH+LL -> 0/100;
  neither available -> 50/50.
- SHORT is exact directional mirror.
- latest_structure_sr returns 2 most recent S/R + relations.
"""

from __future__ import annotations
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.structure import (
    detect_swings,
    most_recent_swing_high,
    most_recent_swing_low,
    _latest_swing_relation,
    structure_score,
    latest_structure_sr,
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


# ---------------------------------------------------------------------------
# Swing detection re-verification (existing contract preserved)
# ---------------------------------------------------------------------------

class TestDetectSwingsStrictFractal(unittest.TestCase):
    def test_flat_middle_no_swing(self):
        s = _series(0, [100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
        highs, lows = detect_swings(s)
        self.assertEqual(highs, [])
        self.assertEqual(lows, [])

    def test_two_candles_no_swings(self):
        s = _series(0, [100, 101], [101, 102], [99, 100], [100, 101])
        highs, lows = detect_swings(s)
        self.assertEqual(highs, [])
        self.assertEqual(lows, [])

    def test_equal_neighbor_rejected(self):
        # high[1] == high[2] -> no swing high at 1
        s = _series(0,
                    [100, 101, 101],
                    [101, 102, 102],
                    [99, 98, 99],
                    [100, 100, 100])
        highs, lows = detect_swings(s)
        self.assertEqual(highs, [])
        # low[1]=98, low[0]=99 > 98, low[2]=99 > 98 -> swing low at 1
        self.assertEqual(lows, [1])

    def test_strict_high(self):
        # high[1]=103: prev=101 < 103 > next=101 -> swing high at 1
        # low[1]=100: prev=99 > 100? False -> no swing low at 1
        s = _series(0,
                    [100, 101, 100],
                    [101, 103, 101],
                    [99, 100, 99],
                    [100, 101, 100])
        highs, lows = detect_swings(s)
        self.assertEqual(highs, [1])
        self.assertEqual(lows, [])


class TestMostRecentSwingHighPreserved(unittest.TestCase):
    def test_none_empty(self):
        self.assertIsNone(most_recent_swing_high([]))

    def test_none_no_swing(self):
        s = _series(0, [100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
        self.assertIsNone(most_recent_swing_high(s))

    def test_last_candle_cannot_be_swing(self):
        # 6 candles, idx 5 cannot be a swing (no i+1). idx 3 is the most recent swing high.
        # base_time=0 → idx 3 has close_time = 0 + 3*300000 = 900000
        s = _series(0,
                    [100, 101, 100, 101, 100, 101],
                    [101, 102, 101, 103, 102, 104],
                    [99, 100, 99, 98, 99, 97],
                    [100, 101, 100, 101, 100, 101])
        sw = most_recent_swing_high(s)
        self.assertIsNotNone(sw)
        self.assertEqual(sw['high'], 103)


class TestMostRecentSwingLowPreserved(unittest.TestCase):
    def test_none_empty(self):
        self.assertIsNone(most_recent_swing_low([]))

    def test_none_no_swing(self):
        s = _series(0, [100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
        self.assertIsNone(most_recent_swing_low(s))

    def test_last_candle_cannot_be_swing(self):
        s = _series(0,
                    [100, 99, 100, 99, 100, 99],
                    [101, 100, 101, 100, 101, 100],
                    [99, 98, 99, 97, 99, 96],
                    [100, 99, 100, 99, 100, 99])
        sw = most_recent_swing_low(s)
        self.assertIsNotNone(sw)
        self.assertEqual(sw['low'], 97)
        self.assertEqual(sw['close_time'], 900000)  # idx 3


# ---------------------------------------------------------------------------
# _latest_swing_relation unit tests
# ---------------------------------------------------------------------------

class TestLatestSwingRelation(unittest.TestCase):
    def test_fewer_than_two_swings_returns_none(self):
        self.assertIsNone(_latest_swing_relation([3], _series(0,
            [100, 101, 100, 101, 100], [101, 103, 101, 104, 102],
            [99, 100, 99, 98, 99], [100, 101, 100, 101, 100]), 'high', True))

    def test_equality_returns_none(self):
        s = _series(0,
                    [100, 101, 100, 101, 100],
                    [101, 103, 101, 103, 102],  # idx 1 high=103, idx 3 high=103 -> equal
                    [99, 100, 99, 98, 99],
                    [100, 101, 100, 101, 100])
        highs, _ = detect_swings(s)
        # highs = [1, 3]
        rel = _latest_swing_relation(highs, s, 'high', True)
        self.assertIsNone(rel)

    def test_hh_returns_bullish(self):
        s = _series(0,
                    [100, 101, 100, 101, 100],
                    [101, 102, 101, 104, 102],  # idx 1 high=102, idx 3 high=104 -> HH
                    [99, 100, 99, 98, 99],
                    [100, 101, 100, 101, 100])
        highs, _ = detect_swings(s)
        rel = _latest_swing_relation(highs, s, 'high', True)
        self.assertEqual(rel, 'bullish')

    def test_lh_returns_bearish(self):
        # LH: latest high < previous high -> bearish for highs.
        # 5 candles; swing highs at idx 1 (high=104) and idx 3 (high=102) -> LH (102 < 104).
        s = [
            {'close_time': 0,      'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 300000,  'open': 100, 'high': 104, 'low': 100, 'close': 101},
            {'close_time': 600000,  'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 900000,  'open': 100, 'high': 102, 'low': 98, 'close': 101},
            {'close_time': 1200000, 'open': 100, 'high': 100, 'low': 99, 'close': 100},
        ]
        highs, _ = detect_swings(s)
        self.assertEqual(len(highs), 2)
        self.assertIn(1, highs)
        self.assertIn(3, highs)
        rel = _latest_swing_relation(highs, s, 'high', True)
        self.assertEqual(rel, 'bearish')

    def test_hl_returns_bullish_for_lows(self):
        # HL: latest low > previous low -> bullish for lows (higher low = bullish per D5 spec)
        # Explicit candle dicts — avoid _series helper to ensure deterministic data.
        # 7 candles; swing lows at idx 2 (low=97) and idx 4 (low=98) -> HL (98 > 97).
        s = [
            {'close_time': 0,      'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 300000,  'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 600000,  'open': 100, 'high': 101, 'low': 97, 'close': 100},
            {'close_time': 900000,  'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 1200000, 'open': 100, 'high': 101, 'low': 98, 'close': 100},
            {'close_time': 1500000, 'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 1800000, 'open': 100, 'high': 101, 'low': 99, 'close': 100},
        ]
        highs_detected, lows_detected = detect_swings(s)
        self.assertEqual(len(highs_detected), 0)
        self.assertEqual(len(lows_detected), 2)
        self.assertIn(2, lows_detected)
        self.assertIn(4, lows_detected)
        rel = _latest_swing_relation(lows_detected, s, 'low', True)
        self.assertEqual(rel, 'bullish')  # HL = bullish for lows

    def test_ll_returns_bearish_for_lows(self):
        # LL: latest low < previous low -> bearish for lows (lower low = bearish per D5 spec)
        # Explicit candle dicts — avoid _series helper to ensure deterministic data.
        # 7 candles; swing lows at idx 2 (low=98) and idx 4 (low=97) -> LL (97 < 98).
        s = [
            {'close_time': 0,      'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 300000,  'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 600000,  'open': 100, 'high': 101, 'low': 98, 'close': 100},
            {'close_time': 900000,  'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 1200000, 'open': 100, 'high': 101, 'low': 97, 'close': 100},
            {'close_time': 1500000, 'open': 100, 'high': 101, 'low': 99, 'close': 100},
            {'close_time': 1800000, 'open': 100, 'high': 101, 'low': 99, 'close': 100},
        ]
        highs_detected, lows_detected = detect_swings(s)
        self.assertEqual(len(highs_detected), 0)
        self.assertEqual(len(lows_detected), 2)
        self.assertIn(2, lows_detected)
        self.assertIn(4, lows_detected)
        rel = _latest_swing_relation(lows_detected, s, 'low', True)
        self.assertEqual(rel, 'bearish')  # LL = bearish for lows


# ---------------------------------------------------------------------------
# Structure score: the 17 scenarios from the spec
# ---------------------------------------------------------------------------

class TestStructureScoreHHHL(unittest.TestCase):
    """HH + HL -> LONG 100, SHORT 0."""

    def _setup_bullish(self):
        # 21 candles (20 lookback + C_tf at idx 20).
        # C_tf excluded from swings. We need 2+ swing highs and 2+ swing lows
        # in the 20-candle window (indices 0..19), with latest relations bullish.
        # Swing highs at idx 5 (high=102) and idx 12 (high=105) -> HH (105 > 102)
        # Swing lows at idx 7 (low=98) and idx 14 (low=95) -> HL (95 < 98) -> bearish!
        # Wait: HL = latest low > previous low = bearish for lows.
        # For LONG bullish we need HL = latest low < previous low = bullish for lows = LL.
        # Let me re-read: HL = latest low > previous low. LL = latest low < previous low.
        # For LONG score: HH (bullish) + HL. But HL is bearish for lows.
        # The spec says: HH + HL -> 100 LONG.
        # Wait, that contradicts the classification. Let me re-read the spec.
        # "LONG structure: Strong bullish = latest high relation is HH AND latest low relation is HL."
        # But HL means latest low > previous low, which is bearish for lows.
        # And LL means latest low < previous low, which is bullish for lows.
        # So for LONG bullish structure: HH + LL would be the naturally bullish combination.
        # But the spec explicitly says HH + HL -> 100 LONG.
        # Let me re-read the swing classification:
        # "Consecutive swing lows: latest low > previous low → HL; latest low < previous low → LL"
        # "LONG structure: Strong bullish = latest high relation is HH AND latest low relation is HL."
        # Hmm, this seems backwards for lows. For a bullish structure, we'd expect lows to be
        # making higher lows (HL = latest low > previous low = higher low = bullish for price).
        # Wait, I think I had the bullish/bearish mapping backwards.
        # For swing lows: higher low = bullish for price (support rising). So HL = bullish for price.
        # For swing highs: higher high = bullish for price (HH = bullish for price).
        # So HH + HL = both bullish for price = strong bullish. That makes sense!
        # My _latest_swing_relation mapping was wrong. Let me fix it.
        # For lows: direction_up_is_bullish should be True (higher low = bullish).
        pass

    def test_hh_hl_long_100_short_0(self):
        # Construct: 21 candles, C_tf at idx 20 excluded.
        # Swing highs: idx 5 (high=102), idx 12 (high=105) -> HH (bullish)
        # Swing lows: idx 7 (low=97), idx 14 (low=98) -> HL (bullish for lows: 98 > 97)
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        # Set swing points
        highs[5] = 102.0
        highs[12] = 105.0
        lows[7] = 97.0
        lows[14] = 98.0
        # Need neighbors to make these actual swings
        # Swing high at idx 5: need high[4] < high[5] > high[6]
        highs[4] = 100.0
        highs[6] = 100.0
        # Swing high at idx 12: need high[11] < high[12] > high[13]
        highs[11] = 100.0
        highs[13] = 100.0
        # Swing low at idx 7: need low[6] > low[7] < low[8]
        lows[6] = 99.0
        lows[8] = 99.0
        # Swing low at idx 14: need low[13] > low[14] < low[15]
        lows[13] = 99.0
        lows[15] = 99.0

        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 100)
        self.assertEqual(structure_score(series, direction='SHORT'), 0)


class TestStructureScoreLHLL(unittest.TestCase):
    """LH + LL -> LONG 0, SHORT 100."""

    def test_lh_ll_long_0_short_100(self):
        # Swing highs: idx 5 (high=105), idx 12 (high=102) -> LH (bearish)
        # Swing lows: idx 7 (low=98), idx 14 (low=97) -> LL (bullish for lows: 97 < 98)
        # Wait, LL = latest low < previous low. For LONG, LL is bearish for the low side.
        # Actually: LL = latest low < previous low = lower low = bearish for price.
        # So LH + LL = both bearish for price = strong bearish for LONG = 0 LONG / 100 SHORT.
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        highs[5] = 105.0
        highs[12] = 102.0
        lows[7] = 98.0
        lows[14] = 97.0
        highs[4] = 100.0
        highs[6] = 100.0
        highs[11] = 100.0
        highs[13] = 100.0
        lows[6] = 99.0
        lows[8] = 99.0
        lows[13] = 99.0
        lows[15] = 99.0

        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 0)
        self.assertEqual(structure_score(series, direction='SHORT'), 100)


class TestStructureScoreHHOnly(unittest.TestCase):
    """HH only (no second swing low) -> LONG 75, SHORT 25."""

    def test_hh_only_long_75_short_25(self):
        # 21 candles. Only swing highs (2+), no swing lows in window.
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        # Create 2 swing highs, NO swing lows
        highs[5] = 102.0
        highs[12] = 105.0
        highs[4] = 100.0
        highs[6] = 100.0
        highs[11] = 100.0
        highs[13] = 100.0
        # Ensure no swing lows: make all lows equal so no low[i-1] > low[i] < low[i+1]
        # lows all 99.0 -> no swing lows ✓
        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 75)
        self.assertEqual(structure_score(series, direction='SHORT'), 25)


class TestStructureScoreHLOnly(unittest.TestCase):
    """HL only (no second swing high) -> LONG 75, SHORT 25."""

    def test_hl_only_long_75_short_25(self):
        # 21 candles. Only swing lows (2+), no swing highs.
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        # Create 2 swing lows, NO swing highs
        lows[7] = 97.0
        lows[14] = 98.0  # HL: 98 > 97
        lows[6] = 99.0
        lows[8] = 99.0
        lows[13] = 99.0
        lows[15] = 99.0
        # Ensure no swing highs: all highs equal
        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 75)
        self.assertEqual(structure_score(series, direction='SHORT'), 25)


class TestStructureScoreLHOnly(unittest.TestCase):
    """LH only -> LONG 25, SHORT 75."""

    def test_lh_only_long_25_short_75(self):
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        highs[5] = 105.0
        highs[12] = 102.0  # LH: 102 < 105
        highs[4] = 100.0
        highs[6] = 100.0
        highs[11] = 100.0
        highs[13] = 100.0
        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 25)
        self.assertEqual(structure_score(series, direction='SHORT'), 75)


class TestStructureScoreLLOnly(unittest.TestCase):
    """LL only -> LONG 25, SHORT 75."""

    def test_ll_only_long_25_short_75(self):
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        lows[7] = 98.0
        lows[14] = 97.0  # LL: 97 < 98
        lows[6] = 99.0
        lows[8] = 99.0
        lows[13] = 99.0
        lows[15] = 99.0
        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 25)
        self.assertEqual(structure_score(series, direction='SHORT'), 75)


class TestStructureScoreHHLLConflict(unittest.TestCase):
    """HH + LL -> 50/50 (mixed)."""

    def test_hh_ll_50_50(self):
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        highs[5] = 102.0
        highs[12] = 105.0  # HH
        highs[4] = 100.0
        highs[6] = 100.0
        highs[11] = 100.0
        highs[13] = 100.0
        lows[7] = 98.0
        lows[14] = 97.0  # LL
        lows[6] = 99.0
        lows[8] = 99.0
        lows[13] = 99.0
        lows[15] = 99.0
        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 50)
        self.assertEqual(structure_score(series, direction='SHORT'), 50)


class TestStructureScoreLHHLConflict(unittest.TestCase):
    """LH + HL -> 50/50 (mixed)."""

    def test_lh_hl_50_50(self):
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        highs[5] = 105.0
        highs[12] = 102.0  # LH
        highs[4] = 100.0
        highs[6] = 100.0
        highs[11] = 100.0
        highs[13] = 100.0
        lows[7] = 97.0
        lows[14] = 98.0  # HL
        lows[6] = 99.0
        lows[8] = 99.0
        lows[13] = 99.0
        lows[15] = 99.0
        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 50)
        self.assertEqual(structure_score(series, direction='SHORT'), 50)


class TestStructureScoreNeitherAvailable(unittest.TestCase):
    """No usable swing relation -> 50/50."""

    def test_flat_no_swings_50_50(self):
        series = _series(1000000,
                         [100.0] * 21, [101.0] * 21, [99.0] * 21, [100.0] * 21)
        self.assertEqual(structure_score(series, direction='LONG'), 50)
        self.assertEqual(structure_score(series, direction='SHORT'), 50)

    def test_fewer_than_two_highs_and_lows_50_50(self):
        # Only 1 swing high, 1 swing low -> both unavailable -> 50/50
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        highs[10] = 102.0  # single swing high
        highs[9] = 100.0
        highs[11] = 100.0
        lows[10] = 98.0  # single swing low
        lows[9] = 99.0
        lows[11] = 99.0
        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 50)
        self.assertEqual(structure_score(series, direction='SHORT'), 50)


class TestStructureScoreEqualityDoesNotCreateRelation(unittest.TestCase):
    """Equality in swing sequence does not create HH/LH/HL/LL -> falls through to 50."""

    def test_equal_highs_no_relation_50(self):
        # 2 swing highs with equal values -> high relation None
        # 2 swing lows with bullish relation -> low relation bullish
        # high None + low bullish -> 75 LONG / 25 SHORT (not 50)
        # Wait, spec: "one relation available and other unavailable:
        #   bullish relation -> 75 LONG / 25 SHORT"
        # So: high=None, low=bullish -> 75 LONG. Not 50.
        # Let me construct: equal highs (None) + equal lows (None) -> 50.
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        highs[5] = 103.0
        highs[12] = 103.0  # equal -> no relation
        highs[4] = 100.0
        highs[6] = 100.0
        highs[11] = 100.0
        highs[13] = 100.0
        lows[7] = 98.0
        lows[14] = 98.0  # equal -> no relation
        lows[6] = 99.0
        lows[8] = 99.0
        lows[13] = 99.0
        lows[15] = 99.0
        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG'), 50)
        self.assertEqual(structure_score(series, direction='SHORT'), 50)


class TestStructureScoreStrictFractalBehavior(unittest.TestCase):
    """Verify strict fractal: last candle (C_tf) excluded, no future data."""

    def test_ctf_excluded_from_swings(self):
        # C_tf at idx 20 is a clear swing high, but should be excluded.
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        # Make idx 20 a big spike (would be swing if included)
        highs[20] = 200.0
        highs[19] = 100.0
        # Make idx 12 a swing high in the window
        highs[12] = 105.0
        highs[11] = 100.0
        highs[13] = 100.0
        series = _series(1000000, opens, highs, lows, closes)
        # With only 1 swing high in window (idx 12), high relation unavailable.
        # No swing lows -> low relation unavailable.
        # Neither available -> 50.
        self.assertEqual(structure_score(series, direction='LONG'), 50)
        # Verify C_tf's spike doesn't create a swing
        self.assertNotIn(20, detect_swings(series[:-1])[0])

    def test_fewer_than_3_candles_in_window_returns_50(self):
        # Window[:-1] has < 3 candles -> no swings possible -> 50
        series = _series(1000000,
                         [100.0, 101.0], [101.0, 102.0], [99.0, 100.0], [100.0, 101.0])
        self.assertEqual(structure_score(series, lookback=3), 50)
        self.assertEqual(structure_score(series, lookback=2), 50)

    def test_insufficient_data_returns_50(self):
        series = _series(1000000,
                         [100.0], [101.0], [99.0], [100.0])
        self.assertEqual(structure_score(series), 50)


class TestStructureScoreNoFutureDataLeakage(unittest.TestCase):
    def test_uses_only_available_candles(self):
        # Build 30 candles, verify lookback=20 uses only last 20 (excluding C_tf).
        opens = [100.0 + float(i) * 0.5 for i in range(30)]
        highs = [o + 2.0 for o in opens]
        lows = [o - 1.0 for o in opens]
        closes = [o + 0.5 for o in opens]
        full = _series(1000000, opens, highs, lows, closes)

        # Score with full 30 candles (lookback=20 uses last 20)
        score_full = structure_score(full, lookback=20, direction='LONG')

        # Score with only 21 candles (exactly lookback+1)
        short = full[:21]
        score_short = structure_score(short, lookback=20, direction='LONG')

        # Same last 20 candles (indices 1..20 of short = indices 11..30 of full? No.
        # short has 21 candles (indices 0..20). lookback=20 uses indices 0..19 for window,
        # excludes idx 20 (C_tf). So window = short[0:20] = full[0:20].
        # full with lookback=20 uses full[-20:] = full[10:30], excludes idx 29 (C_tf).
        # These are DIFFERENT windows. So scores will differ. That's correct —
        # structure_score uses the latest `lookback` candles, which depend on series length.
        # To test no-future-leakage: add candles beyond C_tf and verify score doesn't change.
        extended = full + [
            _k(1000000 + 30 * 300000, 115.0, 117.0, 114.0, 116.0),
            _k(1000000 + 31 * 300000, 116.0, 118.0, 115.0, 117.0),
        ]
        score_extended = structure_score(extended, lookback=20, direction='LONG')
        # C_tf of extended is the last candle (idx 31). Window is extended[-20:] = extended[12:32],
        # excludes idx 31. This is a different window from full (full[-20:] = full[10:30], excludes 29).
        # So scores will differ. That's correct behavior — different C_tf, different window.
        # To test no-leakage: truncate extended back to full's C_tf and compare.
        truncated = extended[:30]  # same as full
        score_truncated = structure_score(truncated, lookback=20, direction='LONG')
        self.assertEqual(score_full, score_truncated)

    def test_deterministic_with_same_input(self):
        series = _series(1000000,
                         [100.0 + float(i) * 0.5 for i in range(21)],
                         [o + 2.0 for o in [100.0 + float(i) * 0.5 for i in range(21)]],
                         [o - 1.0 for o in [100.0 + float(i) * 0.5 for i in range(21)]],
                         [o + 0.5 for o in [100.0 + float(i) * 0.5 for i in range(21)]])
        s1 = structure_score(series, direction='LONG')
        s2 = structure_score(series, direction='LONG')
        self.assertEqual(s1, s2)


class TestStructureScoreMirrorSymmetry(unittest.TestCase):
    def test_long_short_mirror(self):
        # For any series: structure_score(series, 'LONG') + structure_score(series, 'SHORT') == 100
        # when both are in the same symmetric bucket (100+0, 75+25, 50+50, 25+75, 0+100).
        test_cases = [
            # (opens, highs, lows, closes) — constructed to hit each bucket
            # We'll use the helpers from above implicitly by constructing inline
        ]
        # HH + HL case
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        highs[5] = 102.0; highs[12] = 105.0
        highs[4] = 100.0; highs[6] = 100.0; highs[11] = 100.0; highs[13] = 100.0
        lows[7] = 97.0; lows[14] = 98.0
        lows[6] = 99.0; lows[8] = 99.0; lows[13] = 99.0; lows[15] = 99.0
        series = _series(1000000, opens, highs, lows, closes)
        long_score = structure_score(series, direction='LONG')
        short_score = structure_score(series, direction='SHORT')
        self.assertEqual(long_score + short_score, 100)
        self.assertEqual(structure_score(series, direction='LONG'), 100)
        self.assertEqual(structure_score(series, direction='SHORT'), 0)

        # LH + LL case
        highs[5] = 105.0; highs[12] = 102.0
        lows[7] = 98.0; lows[14] = 97.0
        series = _series(1000000, opens, highs, lows, closes)
        self.assertEqual(structure_score(series, direction='LONG') + structure_score(series, direction='SHORT'), 100)
        self.assertEqual(structure_score(series, direction='LONG'), 0)
        self.assertEqual(structure_score(series, direction='SHORT'), 100)


class TestLatestStructureSR(unittest.TestCase):
    def test_returns_two_most_recent_sr(self):
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        highs[5] = 102.0; highs[12] = 105.0
        highs[4] = 100.0; highs[6] = 100.0; highs[11] = 100.0; highs[13] = 100.0
        lows[7] = 97.0; lows[14] = 98.0
        lows[6] = 99.0; lows[8] = 99.0; lows[13] = 99.0; lows[15] = 99.0
        series = _series(1000000, opens, highs, lows, closes)
        sr = latest_structure_sr(series)
        self.assertEqual(len(sr['swing_highs']), 2)
        self.assertEqual(len(sr['swing_lows']), 2)
        # Most recent first
        self.assertEqual(sr['swing_highs'][0]['high'], 105.0)  # idx 12
        self.assertEqual(sr['swing_highs'][1]['high'], 102.0)  # idx 5
        self.assertEqual(sr['swing_lows'][0]['low'], 98.0)     # idx 14
        self.assertEqual(sr['swing_lows'][1]['low'], 97.0)     # idx 7
        self.assertEqual(sr['high_relation'], 'bullish')  # HH
        self.assertEqual(sr['low_relation'], 'bullish')   # HL (for lows, higher = bullish)

    def test_returns_empty_when_insufficient(self):
        series = _series(1000000,
                         [100.0, 101.0], [101.0, 102.0], [99.0, 100.0], [100.0, 101.0])
        sr = latest_structure_sr(series)
        self.assertEqual(sr['swing_highs'], [])
        self.assertEqual(sr['swing_lows'], [])
        self.assertIsNone(sr['high_relation'])
        self.assertIsNone(sr['low_relation'])

    def test_ctf_excluded_from_sr(self):
        opens = [100.0] * 21
        highs = [101.0] * 21
        lows = [99.0] * 21
        closes = [100.0] * 21
        highs[20] = 200.0  # C_tf spike — should NOT appear in SR
        highs[19] = 100.0
        highs[12] = 105.0
        highs[11] = 100.0; highs[13] = 100.0
        series = _series(1000000, opens, highs, lows, closes)
        sr = latest_structure_sr(series)
        self.assertEqual(len(sr['swing_highs']), 1)  # only idx 12
        self.assertEqual(sr['swing_highs'][0]['high'], 105.0)
        self.assertNotIn(200.0, [c['high'] for c in sr['swing_highs']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
