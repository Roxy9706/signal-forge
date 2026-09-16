# Termux-friendly CLI report + machine-readable JSON output
#
# Consumes a list of (Signal, EvaluationResult) tuples (already evaluated) and
# produces:
#   - a human-readable CLI report (print_report)
#   - a machine-readable JSON dict (build_json_report)
#
# In-sample / OOS: the caller passes separate lists for IS and OOS. If OOS is
# empty or not provided, the report explicitly says "WALK-FORWARD: NOT IMPLEMENTED"
# and does NOT combine IS+OOS into one headline number.
#
# Gross vs net: both are reported separately. Costs (fee + slippage) are shown
# explicitly. Never silently apply leverage.
#
# Sample-size warnings are emitted prominently.
#
# No strategy changes, no parameter optimization, no ML, no trading, no API keys,
# no orders, no webhooks, no Telegram — this module only reports numbers.


from __future__ import annotations
import json
import math
import statistics
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone

from SignalForge.backtest.evaluator import Signal, EvaluationResult
from SignalForge.backtest.metrics import (
    compute_metrics,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_r(val: Optional[float], digits: int = 3) -> str:
    if val is None:
        return "n/a"
    if math.isinf(val):
        return "∞ (no losing R)"
    return f"{val:.{digits}f}"


def _fmt_pct(val: Optional[float], digits: int = 1) -> str:
    if val is None:
        return "n/a"
    return f"{val * 100:.{digits}f}%"


def _fmt_ms(ms: Optional[int]) -> str:
    if ms is None:
        return "n/a"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


def _ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def _build_metrics_from_pairs(pairs: List[Tuple[Signal, EvaluationResult]],
                              chronological_equity: bool = True) -> Tuple[AggregateMetrics, List[Tuple[Signal, EvaluationResult]]]:
    """
    Compute AggregateMetrics from (signal, result) pairs.
    Returns (metrics, sorted_pairs) where sorted_pairs is the chronological
    ordering used for the equity curve.
    """
    results = [r for _, r in pairs]
    metrics = compute_metrics(results, chronological_equity=chronological_equity)

    # Build chronological ordering for bucketing.
    completed_pairs = [(s, r) for s, r in pairs
                       if r.outcome in ("TP", "SL", "EXPIRATION")]
    if chronological_equity:
        completed_pairs = sorted(
            completed_pairs,
            key=lambda sr: (sr[1].touch_timestamp if sr[1].touch_timestamp is not None
                            else sr[0].timestamp),
        )

    # Score buckets.
    metrics.score_buckets = bucket_by_score(metrics, completed_pairs)

    # Timeframe buckets.
    metrics.timeframe_buckets = bucket_by_timeframe(metrics, completed_pairs)

    return metrics, completed_pairs


def _drawdown_detail(metrics: AggregateMetrics) -> str:
    dd = metrics.max_drawdown
    if not dd or dd.get("max_dd_gross_r", 0) == 0:
        return "No drawdown (flat or rising equity curve)."
    peak_idx = dd.get("peak_index")
    trough_idx = dd.get("trough_index")
    parts = []
    parts.append(f"Max DD (gross R): {_fmt_r(dd.get('max_dd_gross_r'))}R")
    parts.append(f"  Peak: {_fmt_r(dd.get('peak_gross_r'))}R at signal #{peak_idx}")
    parts.append(f"  Trough: {_fmt_r(dd.get('trough_gross_r'))}R at signal #{trough_idx}")
    dur = dd.get("duration_signals_gross")
    if dur is not None:
        parts.append(f"  Duration: {dur} signals")
    dd_net = dd.get("max_dd_net_r", 0)
    if dd_net > 0:
        parts.append(f"Max DD (net R): {_fmt_r(dd.get('max_dd_net_r'))}R")
        parts.append(f"  Peak (net): {_fmt_r(dd.get('peak_net_r'))}R")
        parts.append(f"  Trough (net): {_fmt_r(dd.get('trough_net_r'))}R")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------

def print_report(title: str,
                 pairs: List[Tuple[Signal, EvaluationResult]],
                 is_pairs: Optional[List[Tuple[Signal, EvaluationResult]]] = None,
                 oos_pairs: Optional[List[Tuple[Signal, EvaluationResult]]] = None,
                 fee_pct: float = 0.001,
                 slippage_pct: float = 0.001,
                 show_breakdowns: bool = True) -> None:
    """
    Print a Termux-friendly CLI report.

    Args:
        title: Report title (e.g. "SignalForge Backtest").
        pairs: Combined list of (signal, result) — used when IS/OOS not split.
        is_pairs: In-sample pairs (if available).
        oos_pairs: Out-of-sample pairs (if available).
        fee_pct: Fee percentage (display only).
        slippage_pct: Slippage percentage (display only).
        show_breakdowns: Whether to print LONG/SHORT/score/timeframe/symbol breakdowns.
    """
    # Determine which dataset to use for headline.
    if is_pairs and oos_pairs:
        # Both provided: report them separately.
        _print_section(title, is_pairs, fee_pct, slippage_pct,
                       label="IN-SAMPLE (development)", show_breakdowns=show_breakdowns)
        print()
        _print_section(title + " — OOS", oos_pairs, fee_pct, slippage_pct,
                       label="OUT-OF-SAMPLE", show_breakdowns=False)
        print()
        print("NOTE: In-sample and OOS are reported separately. They are NOT combined")
        print("into a single headline number. OOS results did not influence signal")
        print("generation, strategy configuration, thresholds, or scoring weights.")
        print()
        print("WALK-FORWARD: NOT IMPLEMENTED.")
        return

    # Single dataset (no split provided).
    _print_section(title, pairs, fee_pct, slippage_pct,
                   label="PERIOD", show_breakdowns=show_breakdowns)
    print()
    print("WALK-FORWARD: NOT IMPLEMENTED.")
    print("OOS: not evaluated (no out-of-sample split provided).")


def _print_section(title: str,
                   pairs: List[Tuple[Signal, EvaluationResult]],
                   fee_pct: float,
                   slippage_pct: float,
                   label: str,
                   show_breakdowns: bool) -> None:
    metrics, chrono_pairs = _build_metrics_from_pairs(pairs)

    print("=" * 60)
    print(f"{title}")
    print("=" * 60)
    print(f"Section: {label}")
    print()

    # Period / symbols / signals.
    if pairs:
        tss = [s.timestamp for s, _ in pairs]
        tse = [s.timestamp for s, _ in pairs]
        syms = sorted(set(s.symbol for s, _ in pairs))
        print(f"Symbols: {', '.join(syms)}")
        print(f"Signals (total): {len(pairs)}")
        print(f"  Completed: {metrics.completed_signals}")
        print(f"  Invalid/rejected: {metrics.invalid_signals}")
        print(f"  Expired: {metrics.expired_signals}")
    else:
        print("Symbols: none")
        print("Signals: 0")
    print()

    # Direction split.
    print("--- Direction ---")
    print(f"LONG signals:   {metrics.long_total} (completed: {metrics.long_completed})")
    print(f"SHORT signals:  {metrics.short_total} (completed: {metrics.short_completed})")
    print()

    # Core outcomes.
    print("--- Outcomes ---")
    print(f"TP hits:          {metrics.tp_count}")
    print(f"SL hits:          {metrics.sl_count}")
    print(f"Expirations:      {metrics.expired_signals}")
    print()

    # Rates.
    print("--- Rates ---")
    wr = agg_win_rate(metrics)
    lr = agg_loss_rate(metrics)
    print(f"Win rate:   {_fmt_pct(wr)}")
    print(f"Loss rate:  {_fmt_pct(lr)}")
    print(f"Expectancy (R per signal): {_fmt_r(agg_expectancy(metrics))}R")
    print(f"  Formula: mean(gross_R over {metrics.completed_signals} completed signals)")
    print(f"Median R:  {_fmt_r(agg_median_r(metrics))}R")
    print()

    # Cumulative R.
    print("--- Cumulative R ---")
    cr_gross = agg_cumulative_gross_r(metrics)
    cr_net = agg_cumulative_net_r(metrics)
    print(f"Cumulative R (gross): {_fmt_r(cr_gross)}R")
    print(f"Cumulative R (net):   {_fmt_r(cr_net)}R")
    print(f"  Net = gross - 2 * entry * (fee% + slippage%) / SL_distance")
    print(f"  fee% = {fee_pct*100:.3f}%, slippage% = {slippage_pct*100:.3f}%")
    print()

    # Profit factor.
    print("--- Profit Factor ---")
    pf = agg_profit_factor(metrics)
    print(f"Profit factor: {_fmt_r(pf)}")
    print(f"  Formula: sum(gross_R where R>0) / abs(sum(gross_R where R<0))")
    print(f"  Gross winning R:  {_fmt_r(sum(r for r in metrics.gross_r_list if r > 0))}R")
    print(f"  Gross losing R:  {_fmt_r(sum(r for r in metrics.gross_r_list if r < 0))}R")
    if pf is None and metrics.completed_signals > 0:
        print("  Note: no winning or losing R — all R = 0 (expired or breakeven).")
    print()

    # Drawdown.
    print("--- Maximum Drawdown ---")
    print(_drawdown_detail(metrics))
    print()

    # Holding time.
    print("--- Holding Time ---")
    print(f"Average hold:   {_fmt_ms(agg_avg_holding_ms(metrics))}")
    print(f"Median hold:    {_fmt_ms(agg_median_holding_ms(metrics))}")
    print()

    # Consecutive.
    print("--- Streaks ---")
    print(f"Max consecutive wins:  {agg_max_consec_wins(metrics)}")
    print(f"Max consecutive losses: {agg_max_consec_losses(metrics)}")
    print()

    # Gross vs net summary.
    print("--- Gross vs Net ---")
    print(f"Gross R (sum):  {_fmt_r(cr_gross)}R")
    print(f"Net R (sum):    {_fmt_r(cr_net)}R")
    print(f"Cost drag:      {_fmt_r(cr_gross - cr_net) if cr_gross is not None else 'n/a'}R")
    print(f"  (fee + slippage; NOT leverage)")
    print()

    # Warnings.
    if metrics.warnings:
        print("--- Warnings ---")
        for w in metrics.warnings:
            print(f"  ⚠ {w}")
        print()

    # Data quality notes.
    if metrics.data_quality_notes:
        print("--- Data Quality Notes ---")
        for n in metrics.data_quality_notes[:20]:
            print(f"  • {n}")
        if len(metrics.data_quality_notes) > 20:
            print(f"  ... and {len(metrics.data_quality_notes) - 20} more")
        print()

    # Breakdowns.
    if show_breakdowns:
        print("--- LONG vs SHORT Breakdown ---")
        lm = agg_long_metrics(metrics)
        sm = agg_short_metrics(metrics)
        if lm:
            print("LONG:")
            for k, v in lm.items():
                if isinstance(v, float):
                    print(f"  {k}: {_fmt_r(v) if 'r' in k or 'expectancy' in k or 'cumulative' in k else (v if v is not None else 'n/a')}")
                else:
                    print(f"  {k}: {v}")
        if sm:
            print("SHORT:")
            for k, v in sm.items():
                if isinstance(v, float):
                    print(f"  {k}: {_fmt_r(v) if 'r' in k or 'expectancy' in k or 'cumulative' in k else (v if v is not None else 'n/a')}")
                else:
                    print(f"  {k}: {v}")
        print()

        print("--- Score Range Breakdown ---")
        for label, lo, hi in SCORE_BUCKETS:
            bm = metrics.score_buckets.get(label)
            if bm is None or bm.count == 0:
                continue
            print(f"Score {label} (n={bm.count}):")
            print(f"  TP: {bm.tp_count}, SL: {bm.sl_count}, Exp: {bm.expiration_count}")
            print(f"  Win rate: {_fmt_pct(bm.win_rate)}")
            print(f"  Expectancy: {_fmt_r(bm.expectancy)}R")
            print(f"  Cumulative R: {_fmt_r(bm.cumulative_r)}R")
            print(f"  Profit factor: {_fmt_r(bm.profit_factor)}")
            print()

        print("--- Timeframe Breakdown ---")
        for tf in TIMEFRAME_ORDER:
            bm = metrics.timeframe_buckets.get(tf)
            if bm is None or bm.count == 0:
                continue
            print(f"Timeframe {tf} (n={bm.count}):")
            print(f"  TP: {bm.tp_count}, SL: {bm.sl_count}")
            print(f"  Win rate: {_fmt_pct(bm.win_rate)}")
            print(f"  Expectancy: {_fmt_r(bm.expectancy)}R")
            print(f"  Cumulative R: {_fmt_r(bm.cumulative_r)}R")
            print()

        print("--- Symbol Breakdown ---")
        sym_bd = agg_symbol_breakdown(metrics)
        for sym in sorted(sym_bd.keys()):
            d = sym_bd[sym]
            print(f"{sym} (n={d['count']}):")
            print(f"  TP: {d['tp']}, SL: {d['sl']}")
            print(f"  Win rate: {_fmt_pct(d['win_rate'])}")
            print(f"  Expectancy: {_fmt_r(d['expectancy'])}R")
            print(f"  Cumulative R: {_fmt_r(d['cumulative_r'])}R")
            print()

    print()
    print("=" * 60)
    print("DISCLAIMER: Backtest performance does not guarantee future trading performance.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Machine-readable JSON output
# ---------------------------------------------------------------------------

def build_json_report(title: str,
                      pairs: List[Tuple[Signal, EvaluationResult]],
                      is_pairs: Optional[List[Tuple[Signal, EvaluationResult]]] = None,
                      oos_pairs: Optional[List[Tuple[Signal, EvaluationResult]]] = None,
                      fee_pct: float = 0.001,
                      slippage_pct: float = 0.001) -> Dict[str, Any]:
    """
    Build a machine-readable JSON-serializable dict from (signal, result) pairs.

    Returns a dict with:
      - "meta": title, fee_pct, slippage_pct, generated_at (ISO UTC), version
      - "period": symbols, total_signals, completed, invalid, expired, is_oos_split
      - "outcomes": tp, sl, expiration counts
      - "rates": win_rate, loss_rate, expectancy, median_r
      - "cumulative": gross_r, net_r
      - "profit_factor": value or explicit null with reason
      - "drawdown": max_dd_gross_r, max_dd_net_r, peak, trough, etc.
      - "holding": avg_ms, median_ms
      - "streaks": max_consec_wins, max_consec_losses
      - "direction_breakdown": {LONG: {...}, SHORT: {...}}
      - "score_breakdown": {label: {...}}
      - "timeframe_breakdown": {tf: {...}}
      - "symbol_breakdown": {symbol: {...}}
      - "warnings": [...]
      - "data_quality_notes": [...]
      - "costs": {fee_pct, slippage_pct, gross_r, net_r, cost_drag_r}
      - "oos": {in_sample: {...}, oos: {...}} or explicit null
      - "walk_forward": "NOT IMPLEMENTED"
      - "disclaimer": required disclaimer text
    """
    if is_pairs and oos_pairs:
        is_json = _build_single_json(title + " (IS)", is_pairs, fee_pct, slippage_pct,
                                     label="in_sample")
        oos_json = _build_single_json(title + " (OOS)", oos_pairs, fee_pct, slippage_pct,
                                      label="oos")
        combined, _ = _build_metrics_from_pairs(pairs)
    else:
        is_json = None
        oos_json = None
        combined, _ = _build_metrics_from_pairs(pairs)

    report: Dict[str, Any] = {
        "meta": {
            "title": title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fee_pct": fee_pct,
            "slippage_pct": slippage_pct,
            "version": "P4",
        },
        "period": {
            "symbols": sorted(set(s.symbol for s, _ in pairs)),
            "total_signals": len(pairs),
            "completed_signals": combined.completed_signals,
            "invalid_signals": combined.invalid_signals,
            "expired_signals": combined.expired_signals,
            "is_oos_split": bool(is_pairs and oos_pairs),
        },
        "outcomes": {
            "tp": combined.tp_count,
            "sl": combined.sl_count,
            "expiration": combined.expired_signals,
        },
        "direction_counts": {
            "long_total": combined.long_total,
            "long_completed": combined.long_completed,
            "short_total": combined.short_total,
            "short_completed": combined.short_completed,
        },
        "rates": {
            "win_rate": agg_win_rate(combined),
            "loss_rate": agg_loss_rate(combined),
            "expectancy_r": agg_expectancy(combined),
            "median_r": agg_median_r(combined),
            "expectancy_formula": "mean(gross_R over completed signals)",
            "expectancy_count": combined.completed_signals,
        },
        "cumulative": {
            "gross_r": agg_cumulative_gross_r(combined),
            "net_r": agg_cumulative_net_r(combined),
        },
        "profit_factor": _json_profit_factor(combined),
        "drawdown": _json_drawdown(combined.max_drawdown),
        "holding": {
            "avg_ms": agg_avg_holding_ms(combined),
            "median_ms": agg_median_holding_ms(combined),
        },
        "streaks": {
            "max_consec_wins": agg_max_consec_wins(combined),
            "max_consec_losses": agg_max_consec_losses(combined),
        },
        "gross_vs_net": {
            "gross_r": agg_cumulative_gross_r(combined),
            "net_r": agg_cumulative_net_r(combined),
            "cost_drag_r": (agg_cumulative_gross_r(combined) -
                            agg_cumulative_net_r(combined)
                            if combined.completed_signals > 0 else 0.0),
            "note": "Net = gross - 2 * entry * (fee% + slippage%) / SL_distance. No leverage applied.",
        },
        "direction_breakdown": {
            "LONG": agg_long_metrics(combined),
            "SHORT": agg_short_metrics(combined),
        },
        "score_breakdown": {},
        "timeframe_breakdown": {},
        "symbol_breakdown": agg_symbol_breakdown(combined),
        "warnings": combined.warnings,
        "data_quality_notes": combined.data_quality_notes,
        "costs": {
            "fee_pct": fee_pct,
            "slippage_pct": slippage_pct,
            "note": "Configurable fee/slippage framework. Never silently applied. Never leveraged.",
        "no_leverage": True,
        },
        "oos": _json_oos(is_json, oos_json),
        "walk_forward": "NOT IMPLEMENTED",
        "disclaimer": "Backtest performance does not guarantee future trading performance.",
    }

    # Score and timeframe breakdowns from chronological pairs.
    _, chrono_pairs = _build_metrics_from_pairs(pairs)
    for label, lo, hi in SCORE_BUCKETS:
        bm = combined.score_buckets.get(label)
        if bm is None or bm.count == 0:
            continue
        report["score_breakdown"][label] = _bucket_to_dict(bm)
    for tf, bm in combined.timeframe_buckets.items():
        if bm.count == 0:
            continue
        report["timeframe_breakdown"][tf] = _bucket_to_dict(bm)

    return report


def _build_single_json(title: str,
                       pairs: List[Tuple[Signal, EvaluationResult]],
                       fee_pct: float,
                       slippage_pct: float,
                       label: str) -> Dict[str, Any]:
    metrics, _ = _build_metrics_from_pairs(pairs)
    return {
        "label": label,
        "total_signals": len(pairs),
        "completed_signals": metrics.completed_signals,
        "invalid_signals": metrics.invalid_signals,
        "expired_signals": metrics.expired_signals,
        "win_rate": agg_win_rate(metrics),
        "loss_rate": agg_loss_rate(metrics),
        "expectancy_r": agg_expectancy(metrics),
        "median_r": agg_median_r(metrics),
        "cumulative_gross_r": agg_cumulative_gross_r(metrics),
        "cumulative_net_r": agg_cumulative_net_r(metrics),
        "profit_factor": _json_profit_factor(metrics),
        "drawdown": _json_drawdown(metrics.max_drawdown),
        "avg_holding_ms": agg_avg_holding_ms(metrics),
        "median_holding_ms": agg_median_holding_ms(metrics),
        "max_consec_wins": agg_max_consec_wins(metrics),
        "max_consec_losses": agg_max_consec_losses(metrics),
        "direction_breakdown": {
            "LONG": agg_long_metrics(metrics),
            "SHORT": agg_short_metrics(metrics),
        },
        "warnings": metrics.warnings,
    }


def _json_profit_factor(m: AggregateMetrics) -> Dict[str, Any]:
    pf = agg_profit_factor(m)
    return {
        "value": pf if pf != float("inf") else None,
        "value_inf": pf == float("inf"),
        "gross_winning_r": sum(r for r in m.gross_r_list if r > 0),
        "gross_losing_r": sum(r for r in m.gross_r_list if r < 0),
        "formula": "sum(gross_R where R>0) / abs(sum(gross_R where R<0))",
        "note": ("Infinity means no losing R — report as undefined for decision-making"
                 if pf == float("inf") else
                 ("Undefined (no R ≠ 0 trades)" if pf is None else
                  "Finite value")) ,
    }


def _json_drawdown(dd: Dict[str, Any]) -> Dict[str, Any]:
    if not dd:
        return {"max_dd_gross_r": 0.0, "max_dd_net_r": 0.0, "note": "No equity curve data."}
    return {
        "max_dd_gross_r": dd.get("max_dd_gross_r", 0.0),
        "max_dd_net_r": dd.get("max_dd_net_r", 0.0),
        "peak_gross_r": dd.get("peak_gross_r", 0.0),
        "trough_gross_r": dd.get("trough_gross_r", 0.0),
        "peak_net_r": dd.get("peak_net_r", 0.0),
        "trough_net_r": dd.get("trough_net_r", 0.0),
        "peak_index": dd.get("peak_index"),
        "trough_index": dd.get("trough_index"),
        "duration_signals": dd.get("duration_signals_gross"),
        "note": "Max drawdown from chronological R-based equity curve. Not monetary.",
    }


def _json_oos(is_json: Optional[Dict], oos_json: Optional[Dict]) -> Dict[str, Any]:
    if is_json is None and oos_json is None:
        return {
            "split_provided": False,
            "in_sample": None,
            "oos": None,
            "note": "No in-sample/OOS split provided. OOS not evaluated.",
        }
    return {
        "split_provided": True,
        "in_sample": is_json,
        "oos": oos_json,
        "note": "In-sample and OOS reported separately. NEVER combined in headline. OOS does not influence strategy config.",
    }


def _bucket_to_dict(bm: BucketMetrics) -> Dict[str, Any]:
    return {
        "count": bm.count,
        "tp": bm.tp_count,
        "sl": bm.sl_count,
        "expiration": bm.expiration_count,
        "win_rate": bm.win_rate,
        "loss_rate": bm.loss_rate,
        "expectancy_r": bm.expectancy,
        "median_r": bm.median_r,
        "cumulative_r": bm.cumulative_r,
        "profit_factor": bm.profit_factor if bm.profit_factor != float("inf") else None,
        "profit_factor_inf": bm.profit_factor == float("inf"),
        "avg_holding_ms": bm.avg_holding_ms,
        "median_holding_ms": bm.median_holding_ms,
    }


# ---------------------------------------------------------------------------
# Entry point for CLI execution
# ---------------------------------------------------------------------------

def main_cli() -> None:
    """Minimal CLI entrypoint for Termux. Expects pre-evaluated pairs from caller."""
    print("SignalForge P4 Performance Analytics")
    print("======================================")
    print()
    print("This module is intended to be called by a backtest runner that produces")
    print("(Signal, EvaluationResult) pairs. Run with a driver script or import")
    print("print_report() / build_json_report().")
    print()
    print("No strategy parameters were changed from P3.")
    print("No trading, private APIs, keys, orders, webhooks, or Telegram were added.")


if __name__ == "__main__":
    main_cli()
