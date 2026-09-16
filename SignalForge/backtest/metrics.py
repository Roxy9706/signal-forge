# PERFORMANCE METRICS — computed from EvaluationResult lists
#
# All metrics are R-based. No account-size, leverage, or fixed-position-size
# assumptions are made. The equity curve is built from chronological completed
# signals only.
#
# Sample-size warnings are emitted where appropriate. Do not invent statistical
# confidence.
#
# Metric definitions (see README for full documentation):
#
#   Expectancy (R) = mean(gross_r over all completed signals)
#   Win rate         = count(outcome == "TP") / count(completed signals)
#   Loss rate        = count(outcome == "SL") / count(completed signals)
#   Profit factor    = sum(gross_r where gross_r > 0) /
#                       abs(sum(gross_r where gross_r < 0))
#                       Zero-loss case handled explicitly.
#   Max drawdown     = max peak-to-trough decline in chronological equity curve.
#   Consecutive wins/losses: max run lengths in chronological order.
#
# Signal classification buckets (only categories that already exist in the
# strategy/scanner; no arbitrary overfits):
#   - direction: LONG / SHORT
#   - score ranges: 0-40, 40-60, 60-80, 80-100  (fixed, documented boundaries)
#   - timeframe: as reported by the signal (1d/4h/1h/15m/5m)
#   - symbol: as reported by the signal
#
# OOS / in-sample separation: callers pass lists already split. This module does
# NOT split — split responsibility lives in the CLI/report layer that consumes
# pre-split evaluation results. If no split is provided, OOS fields are explicit
# nulls and a "WALK-FORWARD: NOT IMPLEMENTED" note is emitted by the report layer.


from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import defaultdict
import statistics
import math


# Re-import the types we need from evaluator.
try:
    from SignalForge.backtest.evaluator import EvaluationResult
except ImportError:
    # Allow standalone import for tests that use synthetic data.
    EvaluationResult = None  # type: ignore


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

SCORE_BUCKETS = [
    ("0-40", 0.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-100", 80.0, 100.0),
]

# Any score outside [0, 100] gets bucketed as "other".

TIMEFRAME_ORDER = ["1d", "4h", "1h", "15m", "5m"]


# ---------------------------------------------------------------------------
# Per-bucket metrics
# ---------------------------------------------------------------------------

@dataclass
class BucketMetrics:
    label: str
    count: int = 0
    tp_count: int = 0
    sl_count: int = 0
    expiration_count: int = 0
    invalid_count: int = 0
    gross_r_sum: float = 0.0
    gross_r_wins: float = 0.0
    gross_r_losses: float = 0.0
    gross_r_list: List[float] = field(default_factory=list)
    net_r_list: List[float] = field(default_factory=list)
    holding_ms_list: List[int] = field(default_factory=list)

    @property
    def win_rate(self) -> Optional[float]:
        denom = self.tp_count + self.sl_count
        if denom == 0:
            return None
        return self.tp_count / denom

    @property
    def loss_rate(self) -> Optional[float]:
        denom = self.tp_count + self.sl_count
        if denom == 0:
            return None
        return self.sl_count / denom

    @property
    def expectancy(self) -> Optional[float]:
        if not self.gross_r_list:
            return None
        return statistics.mean(self.gross_r_list)

    @property
    def median_r(self) -> Optional[float]:
        if not self.gross_r_list:
            return None
        return statistics.median(self.gross_r_list)

    @property
    def cumulative_r(self) -> float:
        return self.gross_r_sum

    @property
    def profit_factor(self) -> Optional[float]:
        if self.gross_r_losses == 0.0:
            if self.gross_r_wins > 0:
                return float("inf")
            return None  # no winners and no losers
        return self.gross_r_wins / abs(self.gross_r_losses)

    @property
    def avg_holding_ms(self) -> Optional[int]:
        if not self.holding_ms_list:
            return None
        return int(statistics.mean(self.holding_ms_list))

    @property
    def median_holding_ms(self) -> Optional[int]:
        if not self.holding_ms_list:
            return None
        return int(statistics.median(self.holding_ms_list))

    @property
    def max_consecutive_wins(self) -> int:
        return _max_consecutive(self.gross_r_list, positive=True)

    @property
    def max_consecutive_losses(self) -> int:
        return _max_consecutive(self.gross_r_list, positive=False)

    @property
    def sample_sufficient(self) -> bool:
        return (self.tp_count + self.sl_count) >= 30


def _max_consecutive(r_list: List[float], positive: bool) -> int:
    """Max length of consecutive wins (r>0) or losses (r<0) in chronological order."""
    best = 0
    current = 0
    for r in r_list:
        if positive and r > 0:
            current += 1
            best = max(best, current)
        elif not positive and r < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


# ---------------------------------------------------------------------------
# Equity curve (R-based, chronological)
# ---------------------------------------------------------------------------

@dataclass
class EquityPoint:
    index: int  # chronological position among completed signals
    cumulative_gross_r: float
    cumulative_net_r: float
    peak_gross_r: float  # running peak of cumulative gross R
    peak_net_r: float     # running peak of cumulative net R
    drawdown_gross_r: float  # current drawdown from peak (peak - current), >=0
    drawdown_net_r: float


def build_equity_curve(results: List[EvaluationResult],
                       chronological: bool = True) -> List[EquityPoint]:
    """
    Build an R-based equity curve from completed signals.

    Args:
        results: list of EvaluationResult (completed = outcome in TP/SL/EXPIRATION).
        chronological: if True, sort by touch_timestamp (or signal.timestamp for
            expirations) ascending before building. If False, use the order given.

    Returns:
        List of EquityPoint, one per completed signal, in chronological order.
    """
    completed = [r for r in results if r.outcome in ("TP", "SL", "EXPIRATION")]
    if chronological:
        # Sort by touch_timestamp; expirations use the signal's timestamp proxy.
        # Use a numeric key: touch_timestamp if available, else a large sentinel
        # that places expirations after all touched signals (or use signal_id hash).
        # For determinism, use (touch_timestamp or 0, signal_id) — but signal_id
        # is str; we need a fully numeric key. Use (touch_timestamp or -1, idx).
        indexed = list(enumerate(completed))
        indexed.sort(key=lambda ri: (
            ri[1].touch_timestamp if ri[1].touch_timestamp is not None else -1,
            ri[0],
        ))
        completed = [r for _, r in indexed]

    points: List[EquityPoint] = []
    cum_gross = 0.0
    cum_net = 0.0
    peak_gross = 0.0
    peak_net = 0.0

    for idx, r in enumerate(completed):
        gross = r.gross_r if r.gross_r is not None else 0.0
        net = r.net_r if r.net_r is not None else 0.0
        cum_gross += gross
        cum_net += net
        if cum_gross > peak_gross:
            peak_gross = cum_gross
        if cum_net > peak_net:
            peak_net = cum_net
        dd_gross = max(0.0, peak_gross - cum_gross)
        dd_net = max(0.0, peak_net - cum_net)
        points.append(EquityPoint(
            index=idx,
            cumulative_gross_r=cum_gross,
            cumulative_net_r=cum_net,
            peak_gross_r=peak_gross,
            peak_net_r=peak_net,
            drawdown_gross_r=dd_gross,
            drawdown_net_r=dd_net,
        ))

    return points


def max_drawdown_from_equity(equity: List[EquityPoint]) -> Dict[str, Any]:
    """
    Compute maximum drawdown from an equity curve (list of EquityPoint).

    Returns dict with:
      - "max_dd_gross_r": maximum drawdown in gross R (positive number)
      - "max_dd_net_r": maximum drawdown in net R (positive number)
      - "peak_gross_r": peak value at which the max DD started
      - "trough_gross_r": trough value at which the max DD ended
      - "peak_net_r": peak value at which the max DD started (net)
      - "trough_net_r": trough value at which the max DD ended (net)
      - "peak_index": index of peak
      - "trough_index": index of trough
      - "duration_signals": number of completed signals between peak and trough
      - "duration_candles_ms": approximate duration in ms (trough touch_ts - peak touch_ts)
    """
    if not equity:
        return {
            "max_dd_gross_r": 0.0,
            "max_dd_net_r": 0.0,
            "peak_gross_r": 0.0,
            "trough_gross_r": 0.0,
            "peak_net_r": 0.0,
            "trough_net_r": 0.0,
            "peak_index": None,
            "trough_index": None,
            "duration_signals": None,
            "duration_candles_ms": None,
        }

    max_dd_gross = 0.0
    max_dd_net = 0.0
    peak_idx_gross = 0
    trough_idx_gross = 0
    peak_idx_net = 0
    trough_idx_net = 0
    # Track the actual maximum cumulative value (running peak across all points).
    peak_val_gross = equity[0].cumulative_gross_r if equity else 0.0
    trough_val_gross = equity[0].cumulative_gross_r if equity else 0.0
    peak_val_net = equity[0].cumulative_net_r if equity else 0.0
    trough_val_net = equity[0].cumulative_net_r if equity else 0.0

    for i, p in enumerate(equity):
        # Always update the running peak value.
        if p.cumulative_gross_r > peak_val_gross:
            peak_val_gross = p.cumulative_gross_r
        if p.cumulative_net_r > peak_val_net:
            peak_val_net = p.cumulative_net_r
        if p.drawdown_gross_r > max_dd_gross:
            max_dd_gross = p.drawdown_gross_r
            peak_idx_gross = _find_peak_index(equity, i, "gross")
            trough_idx_gross = i
            trough_val_gross = p.cumulative_gross_r
        if p.drawdown_net_r > max_dd_net:
            max_dd_net = p.drawdown_net_r
            peak_idx_net = _find_peak_index(equity, i, "net")
            trough_idx_net = i
            trough_val_net = p.cumulative_net_r

    # Duration: approximate from touch timestamps of peak and trough points.
    # We need the original results to get timestamps; equity points don't carry them.
    # Return duration_signals as the number of signals between peak and trough.
    dur_signals_gross = None
    dur_signals_net = None
    if peak_idx_gross is not None and trough_idx_gross is not None:
        dur_signals_gross = trough_idx_gross - peak_idx_gross
    if peak_idx_net is not None and trough_idx_net is not None:
        dur_signals_net = trough_idx_net - peak_idx_net

    return {
        "max_dd_gross_r": max_dd_gross,
        "max_dd_net_r": max_dd_net,
        "peak_gross_r": peak_val_gross,
        "trough_gross_r": trough_val_gross,
        "peak_net_r": peak_val_net,
        "trough_net_r": trough_val_net,
        "peak_index": peak_idx_gross if max_dd_gross > 0 else None,
        "trough_index": trough_idx_gross if max_dd_gross > 0 else None,
        "duration_signals_gross": dur_signals_gross if max_dd_gross > 0 else None,
        "duration_signals_net": dur_signals_net if max_dd_net > 0 else None,
        "duration_candles_ms": None,  # requires original results; set by caller when available
    }


def _find_peak_index(equity: List[EquityPoint], trough_idx: int,
                     kind: str) -> Optional[int]:
    """Find the index of the running peak that preceded the trough at trough_idx."""
    if trough_idx <= 0:
        return 0
    target_field = "cumulative_gross_r" if kind == "gross" else "cumulative_net_r"
    best_idx = 0
    best_val = equity[0].cumulative_gross_r if kind == "gross" else equity[0].cumulative_net_r
    for i in range(1, trough_idx + 1):
        val = equity[i].cumulative_gross_r if kind == "gross" else equity[i].cumulative_net_r
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Aggregate metrics from a list of EvaluationResult
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    # Sample composition
    total_signals: int = 0
    completed_signals: int = 0
    invalid_signals: int = 0
    expired_signals: int = 0

    # Direction counts
    long_total: int = 0
    short_total: int = 0
    long_completed: int = 0
    short_completed: int = 0
    long_tp: int = 0
    long_sl: int = 0
    short_tp: int = 0
    short_sl: int = 0

    # Outcome counts
    tp_count: int = 0
    sl_count: int = 0

    # R metrics
    gross_r_list: List[float] = field(default_factory=list)
    net_r_list: List[float] = field(default_factory=list)
    holding_ms_list: List[int] = field(default_factory=list)

    # Data quality
    data_quality_notes: List[str] = field(default_factory=list)
    missing_data_count: int = 0
    rejected_signal_count: int = 0

    # Bucket breakdowns
    direction_buckets: Dict[str, BucketMetrics] = field(default_factory=lambda: {
        "LONG": BucketMetrics(label="LONG"),
        "SHORT": BucketMetrics(label="SHORT"),
    })
    score_buckets: Dict[str, BucketMetrics] = field(default_factory=dict)
    timeframe_buckets: Dict[str, BucketMetrics] = field(default_factory=dict)
    symbol_buckets: Dict[str, BucketMetrics] = field(default_factory=dict)

    # Equity curve
    equity_curve: List[EquityPoint] = field(default_factory=list)
    max_drawdown: Dict[str, Any] = field(default_factory=dict)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # OOS fields (populated by caller/report layer)
    oos_in_sample_signals: Optional[int] = None
    oos_oos_signals: Optional[int] = None
    oos_combined_in_headline: bool = False


def compute_metrics(results: List[EvaluationResult],
                    chronological_equity: bool = True) -> AggregateMetrics:
    """
    Compute all aggregate and breakdown metrics from a list of EvaluationResult.

    Args:
        results: all evaluation results (completed + invalid).
        chronological_equity: whether to sort for equity curve.

    Returns:
        AggregateMetrics with all fields populated.
    """
    m = AggregateMetrics()

    for r in results:
        m.total_signals += 1

        # Track data quality.
        if r.outcome == "INVALID":
            m.invalid_signals += 1
            m.rejected_signal_count += 1
            if r.data_quality_notes:
                m.data_quality_notes.extend(r.data_quality_notes)
            continue

        if r.outcome == "EXPIRATION":
            m.expired_signals += 1

        # Only completed signals (TP/SL/EXPIRATION) count for performance.
        if r.outcome in ("TP", "SL", "EXPIRATION"):
            m.completed_signals += 1
        else:
            # UNKNOWN outcome: treat as not completed but not invalid.
            continue

        # Direction counts.
        if r.direction == "LONG":
            m.long_total += 1
            if r.outcome in ("TP", "SL", "EXPIRATION"):
                m.long_completed += 1
            if r.outcome == "TP":
                m.long_tp += 1
            elif r.outcome == "SL":
                m.long_sl += 1
        elif r.direction == "SHORT":
            m.short_total += 1
            if r.outcome in ("TP", "SL", "EXPIRATION"):
                m.short_completed += 1
            if r.outcome == "TP":
                m.short_tp += 1
            elif r.outcome == "SL":
                m.short_sl += 1

        # Outcome counts (across both directions).
        if r.outcome == "TP":
            m.tp_count += 1
        elif r.outcome == "SL":
            m.sl_count += 1

        # R values.
        gross = r.gross_r if r.gross_r is not None else 0.0
        net = r.net_r if r.net_r is not None else 0.0
        m.gross_r_list.append(gross)
        m.net_r_list.append(net)
        if r.holding_ms is not None:
            m.holding_ms_list.append(r.holding_ms)

        # Direction bucket.
        if r.direction in m.direction_buckets:
            db = m.direction_buckets[r.direction]
            db.count += 1
            if r.outcome == "TP":
                db.tp_count += 1
                db.gross_r_wins += max(0.0, gross)
            elif r.outcome == "SL":
                db.sl_count += 1
                db.gross_r_losses += min(0.0, gross)  # negative or zero
            elif r.outcome == "EXPIRATION":
                db.expiration_count += 1
            db.gross_r_sum += gross
            db.gross_r_list.append(gross)
            db.net_r_list.append(net)
            if r.holding_ms is not None:
                db.holding_ms_list.append(r.holding_ms)

        # Score bucket.
        score = r.score if hasattr(r, 'score') else 0.0
        # EvaluationResult doesn't carry score; we attach it via a workaround.
        # For now score bucket is populated from a separate signal->result mapping.
        # The report layer handles score bucketing using signal objects.

        # Symbol bucket.
        sym = r.symbol
        if sym not in m.symbol_buckets:
            m.symbol_buckets[sym] = BucketMetrics(label=sym)
        sb = m.symbol_buckets[sym]
        sb.count += 1
        if r.outcome == "TP":
            sb.tp_count += 1
            sb.gross_r_wins += max(0.0, gross)
        elif r.outcome == "SL":
            sb.sl_count += 1
            sb.gross_r_losses += min(0.0, gross)
        elif r.outcome == "EXPIRATION":
            sb.expiration_count += 1
        sb.gross_r_sum += gross
        sb.gross_r_list.append(gross)
        sb.net_r_list.append(net)
        if r.holding_ms is not None:
            sb.holding_ms_list.append(r.holding_ms)

        # Timeframe bucket (if available on result; otherwise skip).
        # EvaluationResult doesn't carry timeframe; populated by report layer.

    # Equity curve.
    m.equity_curve = build_equity_curve(results, chronological=chronological_equity)
    m.max_drawdown = max_drawdown_from_equity(m.equity_curve)

    # Warnings.
    if m.total_signals == 0:
        m.warnings.append("INSUFFICIENT SAMPLE: no signals evaluated")
    elif m.completed_signals < 30:
        m.warnings.append(
            f"INSUFFICIENT SAMPLE: only {m.completed_signals} completed signals "
            f"(< 30). Results are exploratory, not statistically reliable."
        )
    if m.invalid_signals > 0:
        m.warnings.append(
            f"DATA QUALITY: {m.invalid_signals} signal(s) rejected as invalid. "
            f"Review data_quality_notes."
        )
    if m.expired_signals > 0:
        m.warnings.append(
            f"EXPIRATION: {m.expired_signals} signal(s) expired without TP/SL hit."
        )
    m.warnings.append("OOS NOT EVALUATED: no out-of-sample split provided to metrics layer.")
    m.warnings.append("WALK-FORWARD: NOT IMPLEMENTED.")

    return m


# ---------------------------------------------------------------------------
# Score and timeframe bucketing (needs signal objects, done by report layer)
# ---------------------------------------------------------------------------

def bucket_by_score(metrics: AggregateMetrics,
                    signal_results: List[tuple]) -> Dict[str, BucketMetrics]:
    """
    Build score-bucketed metrics from a list of (signal, result) tuples.

    Args:
        metrics: existing aggregate (direction/symbol buckets already computed).
        signal_results: list of (signal, evaluation_result) tuples.

    Returns:
        Dict mapping bucket label -> BucketMetrics.
    """
    buckets: Dict[str, BucketMetrics] = {}
    for label, lo, hi in SCORE_BUCKETS:
        buckets[label] = BucketMetrics(label=label)

    for signal, result in signal_results:
        score = getattr(signal, 'score', 0.0)
        label = _score_to_bucket(score)
        if label not in buckets:
            buckets[label] = BucketMetrics(label=label)
        bm = buckets[label]
        bm.count += 1
        gross = result.gross_r if result.gross_r is not None else 0.0
        if result.outcome == "TP":
            bm.tp_count += 1
            bm.gross_r_wins += max(0.0, gross)
        elif result.outcome == "SL":
            bm.sl_count += 1
            bm.gross_r_losses += min(0.0, gross)
        elif result.outcome == "EXPIRATION":
            bm.expiration_count += 1
        bm.gross_r_sum += gross
        bm.gross_r_list.append(gross)
        bm.net_r_list.append(result.net_r if result.net_r is not None else 0.0)
        if result.holding_ms is not None:
            bm.holding_ms_list.append(result.holding_ms)

    return buckets


def bucket_by_timeframe(metrics: AggregateMetrics,
                        signal_results: List[tuple]) -> Dict[str, BucketMetrics]:
    """Build timeframe-bucketed metrics from (signal, result) tuples."""
    buckets: Dict[str, BucketMetrics] = {}
    for signal, result in signal_results:
        tf = getattr(signal, 'timeframe', 'unknown')
        if tf not in buckets:
            buckets[tf] = BucketMetrics(label=tf)
        bm = buckets[tf]
        bm.count += 1
        gross = result.gross_r if result.gross_r is not None else 0.0
        if result.outcome == "TP":
            bm.tp_count += 1
            bm.gross_r_wins += max(0.0, gross)
        elif result.outcome == "SL":
            bm.sl_count += 1
            bm.gross_r_losses += min(0.0, gross)
        elif result.outcome == "EXPIRATION":
            bm.expiration_count += 1
        bm.gross_r_sum += gross
        bm.gross_r_list.append(gross)
        bm.net_r_list.append(result.net_r if result.net_r is not None else 0.0)
        if result.holding_ms is not None:
            bm.holding_ms_list.append(result.holding_ms)
    return buckets


def _score_to_bucket(score: float) -> str:
    for label, lo, hi in SCORE_BUCKETS:
        if lo <= score < hi:
            return label
    if score >= 100.0:
        return "80-100"
    return "0-40"  # scores below 0 land here; shouldn't happen normally


# ---------------------------------------------------------------------------
# Aggregate-level convenience properties
# ---------------------------------------------------------------------------

def agg_win_rate(m: AggregateMetrics) -> Optional[float]:
    denom = m.tp_count + m.sl_count
    if denom == 0:
        return None
    return m.tp_count / denom


def agg_loss_rate(m: AggregateMetrics) -> Optional[float]:
    denom = m.tp_count + m.sl_count
    if denom == 0:
        return None
    return m.sl_count / denom


def agg_expectancy(m: AggregateMetrics) -> Optional[float]:
    if not m.gross_r_list:
        return None
    return statistics.mean(m.gross_r_list)


def agg_median_r(m: AggregateMetrics) -> Optional[float]:
    if not m.gross_r_list:
        return None
    return statistics.median(m.gross_r_list)


def agg_cumulative_gross_r(m: AggregateMetrics) -> float:
    return sum(m.gross_r_list)


def agg_cumulative_net_r(m: AggregateMetrics) -> float:
    return sum(m.net_r_list)


def agg_profit_factor(m: AggregateMetrics) -> Optional[float]:
    wins = sum(r for r in m.gross_r_list if r > 0)
    losses = sum(r for r in m.gross_r_list if r < 0)
    if losses == 0.0:
        if wins > 0:
            return float("inf")
        return None  # no trades with R != 0
    return wins / abs(losses)


def agg_avg_holding_ms(m: AggregateMetrics) -> Optional[int]:
    if not m.holding_ms_list:
        return None
    return int(statistics.mean(m.holding_ms_list))


def agg_median_holding_ms(m: AggregateMetrics) -> Optional[int]:
    if not m.holding_ms_list:
        return None
    return int(statistics.median(m.holding_ms_list))


def agg_max_consec_wins(m: AggregateMetrics) -> int:
    return _max_consecutive(m.gross_r_list, positive=True)


def agg_max_consec_losses(m: AggregateMetrics) -> int:
    return _max_consecutive(m.gross_r_list, positive=False)


def agg_long_metrics(m: AggregateMetrics) -> Dict[str, Any]:
    lb = m.direction_buckets.get("LONG")
    if lb is None:
        return {}
    return {
        "count": lb.count,
        "tp": lb.tp_count,
        "sl": lb.sl_count,
        "expiration": lb.expiration_count,
        "win_rate": lb.win_rate,
        "loss_rate": lb.loss_rate,
        "expectancy": lb.expectancy,
        "median_r": lb.median_r,
        "cumulative_r": lb.cumulative_r,
        "profit_factor": lb.profit_factor,
        "avg_holding_ms": lb.avg_holding_ms,
        "median_holding_ms": lb.median_holding_ms,
        "max_consec_wins": lb.max_consecutive_wins,
        "max_consec_losses": lb.max_consecutive_losses,
    }


def agg_short_metrics(m: AggregateMetrics) -> Dict[str, Any]:
    sb = m.direction_buckets.get("SHORT")
    if sb is None:
        return {}
    return {
        "count": sb.count,
        "tp": sb.tp_count,
        "sl": sb.sl_count,
        "expiration": sb.expiration_count,
        "win_rate": sb.win_rate,
        "loss_rate": sb.loss_rate,
        "expectancy": sb.expectancy,
        "median_r": sb.median_r,
        "cumulative_r": sb.cumulative_r,
        "profit_factor": sb.profit_factor,
        "avg_holding_ms": sb.avg_holding_ms,
        "median_holding_ms": sb.median_holding_ms,
        "max_consec_wins": sb.max_consecutive_wins,
        "max_consec_losses": sb.max_consecutive_losses,
    }


def agg_symbol_breakdown(m: AggregateMetrics) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sym, bm in m.symbol_buckets.items():
        out[sym] = {
            "count": bm.count,
            "tp": bm.tp_count,
            "sl": bm.sl_count,
            "expiration": bm.expiration_count,
            "win_rate": bm.win_rate,
            "loss_rate": bm.loss_rate,
            "expectancy": bm.expectancy,
            "median_r": bm.median_r,
            "cumulative_r": bm.cumulative_r,
            "profit_factor": bm.profit_factor,
            "avg_holding_ms": bm.avg_holding_ms,
            "median_holding_ms": bm.median_holding_ms,
        }
    return out
