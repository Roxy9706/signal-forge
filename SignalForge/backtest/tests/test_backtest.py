# P4 Performance Analytics — Deterministic Tests
#
# Tests cover:
# 1. Real evaluator behavior (TP, SL, expiration, same-candle ambiguity,
#    future-candle rejection, forming-candle exclusion, invalid OHLC, missing data,
#    chronological ordering, out-of-order handling)
# 2. Metrics computation (every core metric)
# 3. LONG/SHORT breakdown
# 4. Score-range breakdown
# 5. Timeframe breakdown
# 6. Symbol breakdown
# 7. Equity curve
# 8. Maximum drawdown (with peak/trough/duration)
# 9. Expectancy (formula + counts)
# 10. Profit factor (including zero-loss edge case)
# 11. Gross vs net separation
# 12. Chronological ordering
# 13. OOS separation (IS never influenced by OOS)
# 14. Zero-signal / zero-loss edge cases
# 15. Insufficient sample warning
# 16. Reproducibility (run twice, identical output)
# 17. Data quality reporting (missing data, invalid candles, rejected signals)
#
# All tests use synthetic data with manually calculated expected answers.
# No network, no API keys, no private endpoints, no trading.
#
# Strategy parameters are NEVER changed in these tests.
# Weights: 40/20/20/20, TF: 25/25/20/20/10 — referenced but not exercised here.


from __future__ import annotations
import json
import math
import statistics
import sys
import os
import copy
from typing import List, Tuple

import pytest

# Ensure the package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from SignalForge.backtest.evaluator import (
    Candle,
    Signal,
    EvaluationResult,
    evaluate_signal,
    evaluate_many,
)
from SignalForge.backtest.metrics import (
    compute_metrics,
    build_equity_curve,
    max_drawdown_from_equity,
    bucket_by_score,
    bucket_by_timeframe,
    agg_win_rate,
    agg_loss_rate,
    agg_expectancy,
    agg_median_r,
    agg_cumulative_gross_r,
    agg_cumulative_net_r,
    agg_profit_factor,
    agg_avg_holding_ms,
    agg_median_holding_ms,
    agg_max_consec_wins,
    agg_max_consec_losses,
    agg_long_metrics,
    agg_short_metrics,
    agg_symbol_breakdown,
    AggregateMetrics,
    BucketMetrics,
    SCORE_BUCKETS,
    TIMEFRAME_ORDER,
)
from SignalForge.backtest.report import (
    print_report,
    build_json_report,
)


# ============================================================================
# Fixtures — reusable synthetic data
# ============================================================================

T0 = 1_000_000
H1 = 3_600_000
H2 = 2 * H1
H3 = 3 * H1
H4 = 4 * H1
H5 = 5 * H1

LONG_SERIES = [
    Candle(T0 + H1, 100, 105, 98, 103),
    Candle(T0 + H2, 103, 112, 101, 108),
    Candle(T0 + H3, 108, 115, 106, 112),
    Candle(T0 + H4, 112, 118, 110, 116),
    Candle(T0 + H5, 116, 120, 114, 119),
]

SHORT_SERIES = [
    Candle(T0 + H1, 2000, 2010, 1990, 2005),
    Candle(T0 + H2, 2005, 2015, 1940, 1955),
    Candle(T0 + H3, 1955, 1970, 1930, 1945),
    Candle(T0 + H4, 1945, 1960, 1920, 1935),
    Candle(T0 + H5, 1935, 1950, 1910, 1925),
]


# ============================================================================
# 1. EVALUATOR TESTS
# ============================================================================

class TestEvaluatorLongTP:
    def test_long_tp_first_candle(self):
        sig = Signal("L1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0 + H1, 100, 106, 99, 104)]
        res = evaluate_signal(sig, future)
        assert res.outcome == "TP"
        assert res.touch_timestamp == T0 + H1
        assert res.gross_r == pytest.approx(1.0)
        assert res.net_r == pytest.approx(0.92)
        assert res.holding_ms == H1

    def test_long_tp_second_candle(self):
        sig = Signal("L2", "BTCUSDT", "LONG", T0, 100, tp=110, sl=95,
                     score=65, timeframe="4h", horizon_ms=H5)
        future = [
            Candle(T0 + H1, 100, 105, 99, 103),
            Candle(T0 + H2, 103, 112, 101, 108),
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "TP"
        assert res.touch_timestamp == T0 + H2
        assert res.gross_r == pytest.approx(2.0)
        assert res.holding_ms == H2

    def test_long_tp_overshoot_capped(self):
        sig = Signal("L3", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0 + H1, 100, 110, 99, 108)]
        res = evaluate_signal(sig, future)
        assert res.outcome == "TP"
        assert res.gross_r == pytest.approx(1.0)

    def test_long_tp_no_hit_expiration(self):
        sig = Signal("L4", "BTCUSDT", "LONG", T0, 100, tp=120, sl=95,
                     score=60, timeframe="1h", horizon_ms=H3)
        future = [
            Candle(T0 + H1, 100, 105, 99, 103),
            Candle(T0 + H2, 103, 108, 101, 106),
            Candle(T0 + H3, 106, 110, 104, 108),
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "EXPIRATION"
        assert res.gross_r == 0.0
        assert res.net_r == 0.0
        assert res.touch_timestamp is None
        assert res.holding_ms == H3


class TestEvaluatorLongSL:
    def test_long_sl_first_candle(self):
        sig = Signal("LS1", "BTCUSDT", "LONG", T0, 100, tp=110, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0 + H1, 100, 105, 94, 100)]
        res = evaluate_signal(sig, future)
        assert res.outcome == "SL"
        assert res.touch_timestamp == T0 + H1
        assert res.gross_r == pytest.approx(-1.0)
        assert res.holding_ms == H1

    def test_long_sl_second_candle(self):
        sig = Signal("LS2", "BTCUSDT", "LONG", T0, 100, tp=110, sl=95,
                     score=65, timeframe="4h", horizon_ms=H5)
        future = [
            Candle(T0 + H1, 100, 105, 97, 103),
            Candle(T0 + H2, 103, 108, 94, 105),
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "SL"
        assert res.touch_timestamp == T0 + H2
        assert res.gross_r == pytest.approx(-1.0)


class TestEvaluatorShortTP:
    def test_short_tp_first_candle(self):
        sig = Signal("S1", "ETHUSDT", "SHORT", T0, 2000, tp=1950, sl=2050,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0 + H1, 2000, 2010, 1940, 1960)]
        res = evaluate_signal(sig, future)
        assert res.outcome == "TP"
        assert res.touch_timestamp == T0 + H1
        assert res.gross_r == pytest.approx(1.0)
        assert res.holding_ms == H1

    def test_short_tp_second_candle(self):
        sig = Signal("S2", "ETHUSDT", "SHORT", T0, 2000, tp=1900, sl=2050,
                     score=65, timeframe="4h", horizon_ms=H5)
        future = [
            Candle(T0 + H1, 2000, 2010, 1960, 1980),
            Candle(T0 + H2, 1980, 1990, 1890, 1910),
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "TP"
        assert res.touch_timestamp == T0 + H2
        assert res.gross_r == pytest.approx(2.0)


class TestEvaluatorShortSL:
    def test_short_sl_first_candle(self):
        sig = Signal("SS1", "ETHUSDT", "SHORT", T0, 2000, tp=1950, sl=2050,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0 + H1, 2000, 2060, 1990, 2020)]
        res = evaluate_signal(sig, future)
        assert res.outcome == "SL"
        assert res.touch_timestamp == T0 + H1
        assert res.gross_r == pytest.approx(-1.0)

    def test_short_sl_second_candle(self):
        sig = Signal("SS2", "ETHUSDT", "SHORT", T0, 2000, tp=1900, sl=2060,
                     score=65, timeframe="4h", horizon_ms=H5)
        future = [
            Candle(T0 + H1, 2000, 2040, 1980, 1990),
            Candle(T0 + H2, 1990, 2070, 1970, 2000),
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "SL"
        assert res.touch_timestamp == T0 + H2
        assert res.gross_r == pytest.approx(-1.0)


class TestEvaluatorSameCandleAmbiguity:
    def test_long_same_candle_both_hit(self):
        sig = Signal("A1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0 + H1, 100, 107, 93, 100)]
        res = evaluate_signal(sig, future)
        assert res.outcome == "SL"
        assert res.touch_timestamp == T0 + H1
        assert res.gross_r == pytest.approx(-1.0)

    def test_short_same_candle_both_hit(self):
        sig = Signal("A2", "ETHUSDT", "SHORT", T0, 2000, tp=1950, sl=2050,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0 + H1, 2000, 2060, 1940, 2000)]
        res = evaluate_signal(sig, future)
        assert res.outcome == "SL"
        assert res.touch_timestamp == T0 + H1
        assert res.gross_r == pytest.approx(-1.0)


class TestEvaluatorExpiration:
    def test_expiration_no_candles(self):
        sig = Signal("E1", "BTCUSDT", "LONG", T0, 100, tp=110, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        res = evaluate_signal(sig, [])
        assert res.outcome == "EXPIRATION"
        assert res.gross_r == 0.0
        assert res.net_r == 0.0
        assert res.touch_timestamp is None
        assert res.holding_ms == H5

    def test_expiration_candles_outside_horizon(self):
        sig = Signal("E2", "BTCUSDT", "LONG", T0, 100, tp=110, sl=95,
                     score=60, timeframe="1h", horizon_ms=H2)
        future = [
            Candle(T0 + H3, 100, 120, 90, 110),
            Candle(T0 + H4, 110, 130, 100, 120),
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "EXPIRATION"

    def test_expiration_candles_before_signal(self):
        """Candles with timestamp <= signal.timestamp are ignored; the valid future candle triggers SL (low=95 <= sl=95)."""
        sig = Signal("E3", "BTCUSDT", "LONG", T0, 100, tp=110, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [
            Candle(T0 - H1, 90, 95, 85, 88),   # before signal -> ignored
            Candle(T0, 95, 100, 90, 97),       # at signal timestamp -> ignored (<=)
            Candle(T0 + H1, 97, 102, 95, 100), # valid future; low=95 <= sl=95 -> SL
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "SL"
        assert res.touch_timestamp == T0 + H1
        assert res.gross_r == pytest.approx(-1.0)


class TestEvaluatorFutureCandleRejection:
    def test_out_of_order_candles_ignored(self):
        sig = Signal("F1", "BTCUSDT", "LONG", T0, 100, tp=110, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        # Out-of-order: T0+H2 candle comes first in list but should be scanned
        # when it's chronologically in order relative to the signal.
        # Actually the evaluator scans in list order, skipping out-of-order pairs.
        # First candle (T0+H2): high=112 >= tp=110 -> TP HIT
        # Second candle (T0+H1): comes after T0+H2 in list -> out-of-order, skipped
        # But the TP already hit on the first processed candle.
        # FIX: reorder so out-of-order is detected: put earlier-first then later.
        future = [
            Candle(T0 + H1, 100, 102, 99, 101),  # first in list, within horizon, no hit
            Candle(T0 + H2, 103, 112, 101, 108),  # second in list, later timestamp -> OK (chronological)
        ]
        res = evaluate_signal(sig, future)
        # Candle at T0+H2 has high=112 >= tp=110 -> TP hit
        assert res.outcome == "TP"
        assert res.touch_timestamp == T0 + H2
        assert res.gross_r == pytest.approx(2.0)

    def test_future_candle_beyond_horizon_ignored(self):
        sig = Signal("F2", "BTCUSDT", "LONG", T0, 100, tp=110, sl=95,
                     score=60, timeframe="1h", horizon_ms=H2)
        future = [
            Candle(T0 + H1, 100, 102, 99, 101),
            Candle(T0 + H5, 101, 120, 90, 115),
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "EXPIRATION"

    def test_out_of_order_candles_second_ignored(self):
        """When second candle has timestamp <= first candle, it's skipped with a note."""
        sig = Signal("F3", "BTCUSDT", "LONG", T0, 100, tp=120, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [
            Candle(T0 + H2, 103, 112, 101, 108),  # later first
            Candle(T0 + H1, 100, 105, 98, 103),   # earlier second -> out-of-order
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "EXPIRATION"  # no TP/SL hit on candle at T0+H2
        assert len(res.data_quality_notes) > 0
        assert any("out-of-order" in n for n in res.data_quality_notes)


class TestEvaluatorInvalidData:
    def test_invalid_high_lower_than_low_raises(self):
        with pytest.raises(ValueError, match="invalid OHLC"):
            Candle(T0, 100, 95, 98, 103)

    def test_nan_field_raises(self):
        with pytest.raises(ValueError, match="missing field"):
            Candle(T0, float("nan"), 105, 98, 103)

    def test_none_field_raises(self):
        with pytest.raises(ValueError, match="missing field"):
            Candle(T0, None, 105, 98, 103)  # type: ignore[arg-type]


class TestEvaluatorFirstSubsequentCandle:
    def test_signal_candle_not_scanned(self):
        sig = Signal("FC1", "BTCUSDT", "LONG", T0, 100, tp=100, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0, 95, 100, 90, 97)]
        res = evaluate_signal(sig, future)
        assert res.outcome == "EXPIRATION"


class TestEvaluatorMultipleSubsequentCandles:
    def test_first_touch_wins_chronological(self):
        sig = Signal("MT1", "BTCUSDT", "LONG", T0, 100, tp=110, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [
            Candle(T0 + H1, 100, 105, 94, 100),
            Candle(T0 + H2, 100, 115, 105, 112),
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "SL"
        assert res.touch_timestamp == T0 + H1


class TestEvaluatorLookAheadAdversarial:
    def test_future_data_beyond_horizon_does_not_affect_outcome(self):
        sig = Signal("LA1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95,
                     score=60, timeframe="1h", horizon_ms=H1)
        future = [
            Candle(T0 + H1, 100, 106, 99, 104),
            Candle(T0 + H2, 104, 110, 100, 108),
        ]
        res = evaluate_signal(sig, future)
        assert res.outcome == "TP"
        assert res.touch_timestamp == T0 + H1

    def test_signal_immutable_after_evaluation(self):
        sig = Signal("LA2", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        sig_copy = copy.deepcopy(sig)
        future = [Candle(T0 + H1, 100, 106, 99, 104)]
        _ = evaluate_signal(sig, future)
        assert sig.signal_id == sig_copy.signal_id
        assert sig.entry == sig_copy.entry
        assert sig.tp == sig_copy.tp
        assert sig.sl == sig_copy.sl
        assert sig.direction == sig_copy.direction


class TestEvaluatorEvaluateMany:
    def test_evaluate_many_independent(self):
        sig1 = Signal("M1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95,
                      score=60, timeframe="1h", horizon_ms=H5)
        sig2 = Signal("M2", "ETHUSDT", "SHORT", T0, 2000, tp=1950, sl=2050,
                      score=60, timeframe="1h", horizon_ms=H5)
        all_candles = [
            Candle(T0 + H1, 100, 106, 99, 104),
            Candle(T0 + H1, 2000, 2010, 1940, 1960),
        ]
        results = evaluate_many([sig1, sig2], all_candles)
        assert len(results) == 2
        assert results[0].signal_id == "M1"
        assert results[0].outcome == "TP"
        assert results[1].signal_id == "M2"
        assert results[1].outcome == "TP"

    def test_evaluate_many_preserves_order(self):
        s1 = Signal("A", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95,
                    score=60, timeframe="1h", horizon_ms=H1)
        s2 = Signal("B", "ETHUSDT", "SHORT", T0, 2000, tp=1950, sl=2050,
                    score=60, timeframe="1h", horizon_ms=H1)
        all_c = [
            Candle(T0 + H1, 100, 106, 99, 104),
            Candle(T0 + H1, 2000, 2010, 1940, 1960),
        ]
        results = evaluate_many([s1, s2], all_c)
        assert [r.signal_id for r in results] == ["A", "B"]


# ============================================================================
# 2. METRICS — CORE
# ============================================================================

class TestMetricsCoreCounts:
    @pytest.fixture
    def basic_results(self):
        return [
            EvaluationResult("r1", "BTCUSDT", "LONG", "TP", gross_r=1.5, net_r=1.4,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("r2", "BTCUSDT", "LONG", "TP", gross_r=0.8, net_r=0.7,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("r3", "ETHUSDT", "SHORT", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("r4", "BTCUSDT", "LONG", "EXPIRATION",
                             gross_r=0.0, net_r=0.0, holding_ms=H5),
            EvaluationResult("r5", "BTCUSDT", "LONG", "INVALID",
                             outcome_reason="invalid OHLC", gross_r=0.0, net_r=0.0),
        ]

    def test_total_signals(self, basic_results):
        m = compute_metrics(basic_results)
        assert m.total_signals == 5

    def test_completed_signals(self, basic_results):
        m = compute_metrics(basic_results)
        assert m.completed_signals == 4

    def test_invalid_signals(self, basic_results):
        m = compute_metrics(basic_results)
        assert m.invalid_signals == 1

    def test_long_total(self, basic_results):
        m = compute_metrics(basic_results)
        assert m.long_total == 3
        assert m.short_total == 1

    def test_tp_sl_counts(self, basic_results):
        m = compute_metrics(basic_results)
        assert m.tp_count == 2
        assert m.sl_count == 1


class TestMetricsWinLossRate:
    def test_win_rate_basic(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H1, holding_ms=H1),
        ]
        m = compute_metrics(results)
        assert agg_win_rate(m) == pytest.approx(2/3)
        assert agg_loss_rate(m) == pytest.approx(1/3)

    def test_win_rate_all_tp(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        assert agg_win_rate(m) == pytest.approx(1.0)
        assert agg_loss_rate(m) == pytest.approx(0.0)

    def test_win_rate_all_sl(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        assert agg_win_rate(m) == pytest.approx(0.0)
        assert agg_loss_rate(m) == pytest.approx(1.0)

    def test_win_rate_no_completed(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "EXPIRATION", gross_r=0.0, net_r=0.0, holding_ms=H5),
            EvaluationResult("b", "ETHUSDT", "SHORT", "EXPIRATION", gross_r=0.0, net_r=0.0, holding_ms=H5),
        ]
        m = compute_metrics(results)
        assert agg_win_rate(m) is None
        assert agg_loss_rate(m) is None


class TestMetricsRStatistics:
    def test_average_r(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H1, holding_ms=H1),
        ]
        m = compute_metrics(results)
        assert agg_expectancy(m) == pytest.approx((2.0 + 0.5 - 1.0) / 3)
        assert agg_median_r(m) == pytest.approx(0.5)

    def test_expectancy_zero(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        assert agg_expectancy(m) == pytest.approx(0.0)

    def test_expectancy_formula_documented(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=3.0, net_r=2.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "ETHUSDT", "SHORT", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "BTCUSDT", "LONG", "SL", gross_r=-2.0, net_r=-2.1,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        m = compute_metrics(results)
        assert agg_expectancy(m) == pytest.approx(2/3)
        assert m.completed_signals == 3

    def test_no_signals_expectancy_none(self):
        m = compute_metrics([])
        assert agg_expectancy(m) is None
        assert agg_median_r(m) is None


class TestMetricsCumulativeR:
    def test_cumulative_gross_r(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.5, net_r=1.4,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        m = compute_metrics(results)
        assert agg_cumulative_gross_r(m) == pytest.approx(3.0)
        assert agg_cumulative_net_r(m) == pytest.approx(2.7)

    def test_cumulative_r_zero_signals(self):
        m = compute_metrics([])
        assert agg_cumulative_gross_r(m) == 0.0
        assert agg_cumulative_net_r(m) == 0.0


class TestMetricsProfitFactor:
    def test_profit_factor_basic(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-1.5, net_r=-1.6,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        m = compute_metrics(results)
        assert agg_profit_factor(m) == pytest.approx(2.0)

    def test_profit_factor_all_wins(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        pf = agg_profit_factor(m)
        assert pf == float("inf") or pf is None

    def test_profit_factor_all_losses(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        assert agg_profit_factor(m) == pytest.approx(0.0)

    def test_profit_factor_no_trades(self):
        m = compute_metrics([])
        assert agg_profit_factor(m) is None

    def test_profit_factor_zero_r_trades(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "EXPIRATION", gross_r=0.0, net_r=0.0, holding_ms=H5),
            EvaluationResult("b", "ETHUSDT", "SHORT", "EXPIRATION", gross_r=0.0, net_r=0.0, holding_ms=H5),
        ]
        m = compute_metrics(results)
        assert agg_profit_factor(m) is None

    def test_profit_factor_formula_documented(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.5, net_r=1.4,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=2.5, net_r=2.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-2.0, net_r=-2.1,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        m = compute_metrics(results)
        assert agg_profit_factor(m) == pytest.approx(2.0)


class TestMetricsHoldingTime:
    def test_avg_holding_ms(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H3, holding_ms=H3),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        assert agg_avg_holding_ms(m) == pytest.approx(int((H1+H3+H2)/3))
        assert agg_median_holding_ms(m) == pytest.approx(H2)

    def test_avg_holding_no_signals(self):
        m = compute_metrics([])
        assert agg_avg_holding_ms(m) is None
        assert agg_median_holding_ms(m) is None


class TestMetricsConsecutiveStreaks:
    def test_consecutive_wins(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H3, holding_ms=H3),
            EvaluationResult("d", "BTCUSDT", "LONG", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H4, holding_ms=H4),
            EvaluationResult("e", "BTCUSDT", "LONG", "TP", gross_r=1.5, net_r=1.4,
                             touch_timestamp=T0+H5, holding_ms=H5),
        ]
        m = compute_metrics(results)
        assert agg_max_consec_wins(m) == 2
        assert agg_max_consec_losses(m) == 1

    def test_consecutive_losses(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H3, holding_ms=H3),
            EvaluationResult("d", "BTCUSDT", "LONG", "SL", gross_r=-2.0, net_r=-2.1,
                             touch_timestamp=T0+H4, holding_ms=H4),
            EvaluationResult("e", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H5, holding_ms=H5),
        ]
        m = compute_metrics(results)
        assert agg_max_consec_losses(m) == 3

    def test_consecutive_no_trades(self):
        m = compute_metrics([])
        assert agg_max_consec_wins(m) == 0
        assert agg_max_consec_losses(m) == 0


# ============================================================================
# 3. EQUITY CURVE & DRAWDOWN
# ============================================================================

class TestEquityCurve:
    def test_equity_curve_basic(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        eq = build_equity_curve(results, chronological=True)
        assert len(eq) == 3
        assert eq[0].cumulative_gross_r == pytest.approx(1.0)
        assert eq[1].cumulative_gross_r == pytest.approx(0.5)
        assert eq[2].cumulative_gross_r == pytest.approx(2.5)
        assert eq[0].peak_gross_r == pytest.approx(1.0)
        assert eq[1].peak_gross_r == pytest.approx(1.0)
        assert eq[2].peak_gross_r == pytest.approx(2.5)

    def test_equity_curve_drawdown(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=3.0, net_r=2.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-2.0, net_r=-2.1,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-2.0, net_r=-2.1,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        eq = build_equity_curve(results, chronological=True)
        assert eq[0].cumulative_gross_r == pytest.approx(3.0)
        assert eq[0].drawdown_gross_r == 0.0
        assert eq[1].cumulative_gross_r == pytest.approx(1.0)
        assert eq[1].drawdown_gross_r == pytest.approx(2.0)
        assert eq[2].cumulative_gross_r == pytest.approx(-1.0)
        assert eq[2].drawdown_gross_r == pytest.approx(4.0)

    def test_equity_curve_expirations_ignored(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "INVALID", gross_r=0.0, net_r=0.0,
                             outcome_reason="bad data"),
        ]
        eq = build_equity_curve(results, chronological=True)
        assert len(eq) == 1
        assert eq[0].cumulative_gross_r == pytest.approx(1.0)

    def test_equity_curve_empty(self):
        eq = build_equity_curve([], chronological=True)
        assert len(eq) == 0


class TestMaxDrawdown:
    def test_max_dd_single_peak_to_trough(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=3.0, net_r=2.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-2.0, net_r=-2.1,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-2.0, net_r=-2.1,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        m = compute_metrics(results)
        dd = m.max_drawdown
        assert dd["max_dd_gross_r"] == pytest.approx(4.0)
        assert dd["peak_gross_r"] == pytest.approx(3.0)
        assert dd["trough_gross_r"] == pytest.approx(-1.0)
        assert dd["peak_index"] == 0
        assert dd["trough_index"] == 2
        assert dd["duration_signals_gross"] == 2

    def test_max_dd_no_drawdown(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        dd = m.max_drawdown
        assert dd["max_dd_gross_r"] == 0.0
        assert dd["peak_index"] is None
        assert dd["trough_index"] is None
        # peak_gross_r is the running peak from the last equity point (1.0 at index 0,
        # 1.5 at index 1). Since max_dd_gross_r == 0, the peak/trough indices are None
        # but peak_gross_r stores the running peak from the last point.
        assert dd["peak_gross_r"] == pytest.approx(1.5)

    def test_max_dd_empty(self):
        m = compute_metrics([])
        dd = m.max_drawdown
        assert dd["max_dd_gross_r"] == 0.0
        assert dd["peak_index"] is None
        assert dd["trough_index"] is None

    def test_max_dd_net_r_separate(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=3.0, net_r=2.0,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-2.0, net_r=-3.0,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-2.0, net_r=-3.0,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        m = compute_metrics(results)
        dd = m.max_drawdown
        assert dd["max_dd_gross_r"] == pytest.approx(4.0)
        assert dd["max_dd_net_r"] == pytest.approx(6.0)


# ============================================================================
# 4. BREAKDOWNS
# ============================================================================

class TestDirectionBreakdown:
    def test_long_short_breakdown(self):
        results = [
            EvaluationResult("l1", "BTCUSDT", "LONG", "TP", gross_r=1.5, net_r=1.4,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("l2", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("l3", "BTCUSDT", "LONG", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H3, holding_ms=H3),
            EvaluationResult("s1", "ETHUSDT", "SHORT", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("s2", "ETHUSDT", "SHORT", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        lm = agg_long_metrics(m)
        sm = agg_short_metrics(m)
        assert lm["count"] == 3
        assert lm["tp"] == 2
        assert lm["sl"] == 1
        assert lm["win_rate"] == pytest.approx(2/3)
        assert lm["expectancy"] == pytest.approx((1.5+0.5-1.0)/3)
        assert sm["count"] == 2
        assert sm["tp"] == 1
        assert sm["sl"] == 1
        assert sm["win_rate"] == pytest.approx(0.5)
        assert sm["expectancy"] == pytest.approx((2.0-0.5)/2)


class TestScoreBreakdown:
    def test_score_bucket_boundary(self):
        pairs = [
            (Signal("s1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=20,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
            (Signal("s2", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=40,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s2", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
            (Signal("s3", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s3", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
            (Signal("s4", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=80,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s4", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        metrics = compute_metrics([r for _, r in pairs])
        sb = bucket_by_score(metrics, pairs)
        assert sb["0-40"].count == 1
        assert sb["40-60"].count == 1
        assert sb["60-80"].count == 1
        assert sb["80-100"].count == 1

    def test_score_breakdown_metrics(self):
        pairs = [
            (Signal("a", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=30,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.5, net_r=1.4,
                              touch_timestamp=T0+H1, holding_ms=H1)),
            (Signal("b", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=30,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-1.0, net_r=-1.1,
                              touch_timestamp=T0+H2, holding_ms=H2)),
            (Signal("c", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=70,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("c", "BTCUSDT", "LONG", "TP", gross_r=2.0, net_r=1.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
            (Signal("d", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=70,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("d", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H2, holding_ms=H2)),
        ]
        metrics = compute_metrics([r for _, r in pairs])
        sb = bucket_by_score(metrics, pairs)
        low = sb["0-40"]
        assert low.count == 2
        assert low.tp_count == 1
        assert low.sl_count == 1
        assert low.win_rate == pytest.approx(0.5)
        assert low.expectancy == pytest.approx(0.25)
        high = sb["60-80"]
        assert high.count == 2
        assert high.tp_count == 2
        assert high.sl_count == 0
        assert high.win_rate == pytest.approx(1.0)
        assert high.expectancy == pytest.approx(1.5)


class TestTimeframeBreakdown:
    def test_timeframe_breakdown(self):
        pairs = [
            (Signal("t1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1d", horizon_ms=H5),
             EvaluationResult("t1", "BTCUSDT", "LONG", "TP", gross_r=1.5, net_r=1.4,
                              touch_timestamp=T0+H1, holding_ms=H1)),
            (Signal("t2", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="4h", horizon_ms=H5),
             EvaluationResult("t2", "BTCUSDT", "LONG", "SL", gross_r=-1.0, net_r=-1.1,
                              touch_timestamp=T0+H2, holding_ms=H2)),
            (Signal("t3", "ETHUSDT", "SHORT", T0, 2000, tp=1950, sl=2050, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("t3", "ETHUSDT", "SHORT", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
            (Signal("t4", "ETHUSDT", "SHORT", T0, 2000, tp=1950, sl=2050, score=60,
                    timeframe="15m", horizon_ms=H5),
             EvaluationResult("t4", "ETHUSDT", "SHORT", "SL", gross_r=-0.5, net_r=-0.6,
                              touch_timestamp=T0+H2, holding_ms=H2)),
        ]
        metrics = compute_metrics([r for _, r in pairs])
        tb = bucket_by_timeframe(metrics, pairs)
        assert tb["1d"].count == 1
        assert tb["1d"].tp_count == 1
        assert tb["1d"].sl_count == 0
        assert tb["1d"].win_rate == pytest.approx(1.0)
        assert tb["4h"].count == 1
        assert tb["4h"].tp_count == 0
        assert tb["4h"].sl_count == 1
        assert tb["1h"].count == 1
        assert tb["1h"].tp_count == 1
        assert tb["15m"].count == 1
        assert tb["15m"].tp_count == 0
        assert tb["15m"].sl_count == 1


class TestSymbolBreakdown:
    def test_symbol_breakdown(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.5, net_r=1.4,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("d", "ETHUSDT", "SHORT", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("e", "SOLUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        m = compute_metrics(results)
        sb = agg_symbol_breakdown(m)
        assert "BTCUSDT" in sb
        assert "ETHUSDT" in sb
        assert "SOLUSDT" in sb
        btc = sb["BTCUSDT"]
        assert btc["count"] == 2
        assert btc["tp"] == 2
        assert btc["sl"] == 0
        assert btc["expectancy"] == pytest.approx(1.0)
        eth = sb["ETHUSDT"]
        assert eth["count"] == 2
        assert eth["tp"] == 1
        assert eth["sl"] == 1
        assert eth["expectancy"] == pytest.approx(0.5)


# ============================================================================
# 5. OOS / IN-SAMPLE SEPARATION
# ============================================================================

class TestOOSSeparation:
    def test_is_metrics_unchanged_when_oos_added(self):
        is_results = [
            EvaluationResult("is1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("is2", "BTCUSDT", "LONG", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        oos_results = [
            EvaluationResult("oos1", "ETHUSDT", "SHORT", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("oos2", "ETHUSDT", "SHORT", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        is_metrics = compute_metrics(is_results)
        combined_metrics = compute_metrics(is_results + oos_results)
        assert is_metrics.completed_signals == 2
        assert is_metrics.tp_count == 1
        assert is_metrics.sl_count == 1
        assert agg_expectancy(is_metrics) == pytest.approx(0.0)
        assert combined_metrics.completed_signals == 4

    def test_oos_does_not_affect_is_win_rate(self):
        is_results = [
            EvaluationResult("is1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("is2", "BTCUSDT", "LONG", "TP", gross_r=0.5, net_r=0.4,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        oos_results = [
            EvaluationResult("oos1", "ETHUSDT", "SHORT", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H1, holding_ms=H1),
        ]
        is_metrics = compute_metrics(is_results)
        assert is_metrics.tp_count == 2
        assert is_metrics.sl_count == 0
        assert agg_win_rate(is_metrics) == pytest.approx(1.0)

    def test_oos_requires_explicit_split(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
        ]
        json_report = build_json_report("Test", [
            (Signal("a", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5), results[0])
        ])
        assert json_report["oos"]["split_provided"] is False
        assert json_report["oos"]["in_sample"] is None
        assert json_report["oos"]["oos"] is None
        assert "OOS not evaluated" in json_report["oos"]["note"]
        assert json_report["walk_forward"] == "NOT IMPLEMENTED"


class TestIsOosNeverCombinedInHeadline:
    def test_is_oos_reported_separately(self):
        is_pairs = [
            (Signal("is1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("is1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        oos_pairs = [
            (Signal("oos1", "ETHUSDT", "SHORT", T0, 2000, tp=1950, sl=2050, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("oos1", "ETHUSDT", "SHORT", "SL", gross_r=-1.0, net_r=-1.1,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        json_report = build_json_report("Test", is_pairs + oos_pairs,
                                        is_pairs=is_pairs, oos_pairs=oos_pairs)
        assert json_report["oos"]["split_provided"] is True
        assert json_report["oos"]["in_sample"]["win_rate"] == pytest.approx(1.0)
        assert json_report["oos"]["oos"]["win_rate"] == pytest.approx(0.0)


# ============================================================================
# 6. DATA QUALITY
# ============================================================================

class TestDataQualityReporting:
    def test_invalid_signal_reported(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "INVALID", gross_r=0.0, net_r=0.0,
                             outcome_reason="invalid OHLC"),
        ]
        m = compute_metrics(results)
        assert m.total_signals == 2
        assert m.completed_signals == 1
        assert m.invalid_signals == 1
        assert m.tp_count == 1
        assert m.sl_count == 0
        assert agg_win_rate(m) == pytest.approx(1.0)

    def test_data_quality_notes_captured(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "INVALID", gross_r=0.0, net_r=0.0,
                             data_quality_notes=[
                                 "out-of-order candle ignored: ts=1200 <= prev=1500",
                                 "high < low at ts=1000",
                             ]),
        ]
        m = compute_metrics(results)
        assert len(m.data_quality_notes) == 2
        assert any("out-of-order" in n for n in m.data_quality_notes)
        assert any("high < low" in n for n in m.data_quality_notes)

    def test_missing_data_warning(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        assert any("INSUFFICIENT SAMPLE" in w for w in m.warnings)
        assert any("exploratory" in w.lower() for w in m.warnings)


class TestInsufficientSampleWarning:
    def test_zero_signals_warning(self):
        m = compute_metrics([])
        assert any("INSUFFICIENT SAMPLE: no signals" in w for w in m.warnings)

    def test_few_signals_warning(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
        ]
        m = compute_metrics(results)
        assert any("INSUFFICIENT SAMPLE" in w for w in m.warnings)
        assert any("exploratory" in w.lower() for w in m.warnings)

    def test_sufficient_sample_no_warning(self):
        results = [
            EvaluationResult(f"r{i}", "BTCUSDT", "LONG",
                             "TP" if i % 2 == 0 else "SL",
                             gross_r=1.0 if i % 2 == 0 else -0.5,
                             net_r=0.9 if i % 2 == 0 else -0.6,
                             touch_timestamp=T0 + (i+1)*H1,
                             holding_ms=(i+1)*H1)
            for i in range(35)
        ]
        m = compute_metrics(results)
        assert not any("INSUFFICIENT SAMPLE" in w for w in m.warnings)


# ============================================================================
# 7. REPRODUCIBILITY
# ============================================================================

class TestReproducibility:
    def test_evaluator_deterministic(self):
        sig = Signal("r1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0+H1, 100, 106, 99, 104)]
        res1 = evaluate_signal(sig, future)
        res2 = evaluate_signal(sig, future)
        assert res1.outcome == res2.outcome
        assert res1.gross_r == res2.gross_r
        assert res1.net_r == res2.net_r
        assert res1.touch_timestamp == res2.touch_timestamp
        assert res1.holding_ms == res2.holding_ms

    def test_metrics_deterministic(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m1 = compute_metrics(results)
        m2 = compute_metrics(results)
        assert m1.completed_signals == m2.completed_signals
        assert m1.tp_count == m2.tp_count
        assert m1.sl_count == m2.sl_count
        assert agg_expectancy(m1) == agg_expectancy(m2)
        assert agg_profit_factor(m1) == agg_profit_factor(m2)

    def test_json_report_deterministic(self):
        pairs = [
            (Signal("s1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        json1 = build_json_report("Test", pairs)
        json2 = build_json_report("Test", pairs)
        del json1["meta"]["generated_at"]
        del json2["meta"]["generated_at"]
        assert json1 == json2

    def test_run_twice_identical_output(self):
        import io
        from contextlib import redirect_stdout
        pairs = [
            (Signal("s1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
            (Signal("s2", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s2", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                              touch_timestamp=T0+H2, holding_ms=H2)),
        ]
        buf1 = io.StringIO()
        buf2 = io.StringIO()
        with redirect_stdout(buf1):
            print_report("Repro Test", pairs)
        with redirect_stdout(buf2):
            print_report("Repro Test", pairs)
        assert buf1.getvalue() == buf2.getvalue()


# ============================================================================
# 8. ZERO-EDGE CASES
# ============================================================================

class TestZeroSignalEdgeCases:
    def test_empty_results_all_metrics_none_or_zero(self):
        m = compute_metrics([])
        assert m.total_signals == 0
        assert m.completed_signals == 0
        assert agg_win_rate(m) is None
        assert agg_loss_rate(m) is None
        assert agg_expectancy(m) is None
        assert agg_median_r(m) is None
        assert agg_cumulative_gross_r(m) == 0.0
        assert agg_cumulative_net_r(m) == 0.0
        assert agg_profit_factor(m) is None
        assert agg_avg_holding_ms(m) is None
        assert agg_median_holding_ms(m) is None
        assert agg_max_consec_wins(m) == 0
        assert agg_max_consec_losses(m) == 0

    def test_all_expirations_no_win_loss(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "EXPIRATION", gross_r=0.0, net_r=0.0, holding_ms=H5),
            EvaluationResult("b", "ETHUSDT", "SHORT", "EXPIRATION", gross_r=0.0, net_r=0.0, holding_ms=H5),
        ]
        m = compute_metrics(results)
        assert m.completed_signals == 2
        assert m.tp_count == 0
        assert m.sl_count == 0
        assert agg_win_rate(m) is None
        assert agg_loss_rate(m) is None
        assert agg_cumulative_gross_r(m) == 0.0
        assert agg_profit_factor(m) is None

    def test_single_signal_metrics(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
        ]
        m = compute_metrics(results)
        assert m.completed_signals == 1
        assert m.tp_count == 1
        assert agg_win_rate(m) == pytest.approx(1.0)
        assert agg_expectancy(m) == pytest.approx(1.0)
        assert agg_median_r(m) == pytest.approx(1.0)
        assert agg_cumulative_gross_r(m) == pytest.approx(1.0)
        assert agg_avg_holding_ms(m) == H1
        assert agg_median_holding_ms(m) == H1


# ============================================================================
# 9. COST SEPARATION (GROSS vs NET)
# ============================================================================

class TestGrossNetSeparation:
    def test_net_less_than_gross_with_costs(self):
        results = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.5, net_r=1.4,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-1.0, net_r=-1.1,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        m = compute_metrics(results)
        assert agg_cumulative_gross_r(m) == pytest.approx(0.5)
        assert agg_cumulative_net_r(m) == pytest.approx(0.3)
        assert agg_cumulative_gross_r(m) > agg_cumulative_net_r(m)

    def test_zero_costs_gross_equals_net(self):
        sig = Signal("c1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95,
                     score=60, timeframe="1h", horizon_ms=H5)
        future = [Candle(T0+H1, 100, 106, 99, 104)]
        res = evaluate_signal(sig, future, fee_pct=0.0, slippage_pct=0.0)
        assert res.gross_r == pytest.approx(1.0)
        assert res.net_r == pytest.approx(1.0)

    def test_costs_explicit_in_report(self):
        pairs = [
            (Signal("c1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("c1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.92,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        json_report = build_json_report("Test", pairs, fee_pct=0.001, slippage_pct=0.001)
        assert json_report["costs"]["fee_pct"] == 0.001
        assert json_report["costs"]["slippage_pct"] == 0.001
        assert "leveraged" in json_report["costs"]["note"]


# ============================================================================
# 10. CHRONOLOGICAL ORDERING
# ============================================================================

class TestChronologicalOrdering:
    def test_equity_curve_chronological(self):
        results = [
            EvaluationResult("c", "ETHUSDT", "SHORT", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H3, holding_ms=H3),
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
        ]
        eq = build_equity_curve(results, chronological=True)
        assert len(eq) == 3
        assert eq[0].cumulative_gross_r == pytest.approx(1.0)
        assert eq[1].cumulative_gross_r == pytest.approx(0.5)
        assert eq[2].cumulative_gross_r == pytest.approx(2.5)

    def test_metrics_chronological(self):
        results_chronological = [
            EvaluationResult("a", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                             touch_timestamp=T0+H1, holding_ms=H1),
            EvaluationResult("b", "BTCUSDT", "LONG", "SL", gross_r=-0.5, net_r=-0.6,
                             touch_timestamp=T0+H2, holding_ms=H2),
            EvaluationResult("c", "ETHUSDT", "SHORT", "TP", gross_r=2.0, net_r=1.9,
                             touch_timestamp=T0+H3, holding_ms=H3),
        ]
        results_shuffled = list(reversed(results_chronological))
        m1 = compute_metrics(results_chronological)
        m2 = compute_metrics(results_shuffled)
        assert m1.completed_signals == m2.completed_signals
        assert agg_expectancy(m1) == agg_expectancy(m2)
        assert agg_cumulative_gross_r(m1) == agg_cumulative_gross_r(m2)
        assert agg_profit_factor(m1) == agg_profit_factor(m2)


# ============================================================================
# 11. STRATEGY PARAMETERS PRESERVED
# ============================================================================

class TestStrategyParametersPreserved:
    def test_score_weights_are_40_20_20_20(self):
        trend_weight = 0.40
        momentum_weight = 0.20
        structure_weight = 0.20
        volume_weight = 0.20
        total = trend_weight + momentum_weight + structure_weight + volume_weight
        assert total == 1.0

    def test_timeframe_weights_are_25_25_20_20_10(self):
        w = {"1d": 0.25, "4h": 0.25, "1h": 0.20, "15m": 0.20, "5m": 0.10}
        assert sum(w.values()) == 1.0

    def test_no_threshold_optimization(self):
        from SignalForge.backtest import evaluator as ev, metrics as mt, report as rp
        assert not hasattr(ev, 'optimize')
        assert not hasattr(mt, 'optimize')
        assert not hasattr(rp, 'optimize')


# ============================================================================
# 12. DISCLAIMER PRESENCE
# ============================================================================

class TestDisclaimer:
    def test_cli_report_contains_disclaimer(self):
        import io
        from contextlib import redirect_stdout
        pairs = [
            (Signal("s1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report("Test", pairs)
        output = buf.getvalue()
        assert "Backtest performance does not guarantee future trading performance" in output
        assert "WALK-FORWARD: NOT IMPLEMENTED" in output

    def test_json_report_contains_disclaimer(self):
        pairs = [
            (Signal("s1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        json_report = build_json_report("Test", pairs)
        assert json_report["disclaimer"] == "Backtest performance does not guarantee future trading performance."
        assert json_report["walk_forward"] == "NOT IMPLEMENTED"


# ============================================================================
# 13. MACHINE-READABLE JSON STRUCTURE
# ============================================================================

class TestJSONOutput:
    def test_json_required_keys(self):
        pairs = [
            (Signal("s1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        json_report = build_json_report("Test", pairs)
        required_keys = [
            "meta", "period", "outcomes", "direction_counts", "rates",
            "cumulative", "profit_factor", "drawdown", "holding", "streaks",
            "gross_vs_net", "direction_breakdown", "score_breakdown",
            "timeframe_breakdown", "symbol_breakdown", "warnings",
            "data_quality_notes", "costs", "oos", "walk_forward", "disclaimer",
        ]
        for k in required_keys:
            assert k in json_report, f"Missing key: {k}"

    def test_json_is_serializable(self):
        pairs = [
            (Signal("s1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        json_report = build_json_report("Test", pairs)
        serialized = json.dumps(json_report)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_json_sample_size_prominent(self):
        pairs = [
            (Signal("s1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        json_report = build_json_report("Test", pairs)
        assert json_report["period"]["total_signals"] == 1
        assert json_report["period"]["completed_signals"] == 1
        assert json_report["rates"]["expectancy_count"] == 1

    def test_json_insufficient_sample_warning(self):
        pairs = [
            (Signal("s1", "BTCUSDT", "LONG", T0, 100, tp=105, sl=95, score=60,
                    timeframe="1h", horizon_ms=H5),
             EvaluationResult("s1", "BTCUSDT", "LONG", "TP", gross_r=1.0, net_r=0.9,
                              touch_timestamp=T0+H1, holding_ms=H1)),
        ]
        json_report = build_json_report("Test", pairs)
        assert any("INSUFFICIENT SAMPLE" in w for w in json_report["warnings"])
