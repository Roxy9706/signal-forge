
"""
P7 Performance Analyzer — deterministic read-only research entry point.

CLI:
  python -m research.analyze --log data/signals/paper_trade_d15.jsonl
  python -m research.analyze --log data/signals/paper_trade_d15.jsonl --json

Scope: analysis ONLY. No mutation of D8-D15. No trading. No network.

Import note:
  research/ lives at the repository root alongside the top-level packages it
  imports (storage.jsonl, paper.models).  It does NOT use a SignalForge.*
  prefix — invoke from the repository root so the root is on sys.path
  (e.g. `python -m research.analyze` or pytest from the repo).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from research.models import (
    DataState,
    GroupMetrics,
    MIN_ANALYSIS_SAMPLES,
    ResearchMetrics,
    ResearchReport,
    SCORE_BUCKETS,
    UtcStats,
    score_bucket,
)
from storage.jsonl import jsonl_read
from paper.models import (
    PaperSignal,
    PaperOutcome,
    event_from_dict,
)


def _iso_ms(ms: int) -> Optional[str]:
    if ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_div(num: float, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def _build_group(
    total: int,
    wins: int,
    losses: int,
    expired: int,
    unresolved: int,
    total_r: float,
    scores: List[float],
) -> GroupMetrics:
    resolved = wins + losses + expired
    return GroupMetrics(
        total=total,
        wins=wins,
        losses=losses,
        expired=expired,
        unresolved=unresolved,
        total_r=total_r,
        win_rate=_safe_div(wins, resolved) if resolved else None,
        average_r=_safe_div(total_r, resolved) if resolved else None,
        average_score=_avg(scores) if scores else None,
        min_score=min(scores) if scores else None,
        max_score=max(scores) if scores else None,
        sample=total,
        insufficient_sample=total < MIN_ANALYSIS_SAMPLES,
    )


def _utc_from_times(
    signal_times: List[int],
    outcome_times: List[int],
) -> Optional[UtcStats]:
    if not signal_times and not outcome_times:
        return None
    first_sig = _iso_ms(min(signal_times)) if signal_times else None
    last_sig = _iso_ms(max(signal_times)) if signal_times else None
    first_out = _iso_ms(min(outcome_times)) if outcome_times else None
    last_out = _iso_ms(max(outcome_times)) if outcome_times else None
    spread = None
    if outcome_times:
        lo = min(outcome_times)
        hi = max(outcome_times)
        if hi > lo:
            spread = (hi - lo) / 3_600_000.0
    return UtcStats(
        first_signal_utc=first_sig,
        last_signal_utc=last_sig,
        first_outcome_utc=first_out,
        last_outcome_utc=last_out,
        outcome_spread_hours=spread,
    )


def _finalize(store: Dict, unresolved_map: Dict[str, int]) -> dict:
    out = {}
    for key, entry in store.items():
        label = key[1]
        g = _build_group(
            total=entry["total"],
            wins=entry["wins"],
            losses=entry["losses"],
            expired=entry["expired"],
            unresolved=unresolved_map.get(label, 0),
            total_r=entry["total_r"],
            scores=entry["scores"],
        )
        out[label] = g
    return out


def run_analysis(log: str, json_out: bool = False) -> ResearchReport:
    path = log or ""
    raw = jsonl_read(path) if path else []
    if not raw:
        r = ResearchReport(data_state=DataState.NO_DATA)
        if json_out:
            _emit_json(r)
        return r

    signals: Dict[str, dict] = {}
    malformed = 0

    for rec in raw:
        try:
            ev = event_from_dict(rec)
        except Exception:
            malformed += 1
            continue
        if isinstance(ev, PaperSignal):
            sid = ev.signal_id
            if not sid:
                continue
            if sid in signals:
                continue
            signals[sid] = {
                "symbol": ev.symbol or "",
                "direction": ev.direction or "",
                "score": ev.score,
                "strategy_version": ev.strategy_version or "",
                "signal_time": ev.signal_time,
            }

    if not signals:
        r = ResearchReport(data_state=DataState.NO_DATA, malformed_count=malformed)
        if json_out:
            _emit_json(r)
        return r

    total_signals = len(signals)
    long_count = 0
    short_count = 0
    wins = 0
    losses = 0
    expired = 0
    unresolved_non_signal = 0
    total_r = 0.0
    all_scores: List[float] = []
    signal_times: List[int] = []
    outcome_times: List[int] = []

    by_direction: Dict[str, dict] = {}
    by_symbol: Dict[str, dict] = {}
    by_strategy: Dict[str, dict] = {}

    for sid, sinfo in signals.items():
        score = sinfo["score"] if isinstance(sinfo["score"], (int, float)) else 0.0
        direction = sinfo["direction"]
        symbol = sinfo["symbol"]
        sv = sinfo["strategy_version"]
        stime = sinfo["signal_time"] if isinstance(sinfo["signal_time"], int) else 0
        all_scores.append(float(score))
        if stime:
            signal_times.append(stime)
        if direction == "LONG":
            long_count += 1
        elif direction == "SHORT":
            short_count += 1
        for key, store in (
            (("dir", direction), by_direction),
            (("sym", symbol), by_symbol),
            (("sv", sv), by_strategy),
        ):
            entry = store.setdefault(key, {
                "total": 0, "wins": 0, "losses": 0, "expired": 0,
                "unresolved": 0, "total_r": 0.0, "scores": [],
            })
            entry["total"] += 1
            entry["scores"].append(float(score))

    outcome_results: Dict[str, Optional[str]] = {sid: None for sid in signals}
    skipped_outcome = 0

    for rec in raw:
        try:
            ev = event_from_dict(rec)
        except Exception:
            continue
        if isinstance(ev, PaperOutcome):
            sid = ev.signal_id
            if not sid or sid not in signals:
                skipped_outcome += 1
                continue
            if outcome_results[sid] is not None:
                continue
            outcome_results[sid] = ev.outcome or ""

    # Build a lookup: signal_id -> PaperOutcome (first occurrence)
    outcome_lookup: Dict[str, PaperOutcome] = {}
    for rec in raw:
        try:
            ev = event_from_dict(rec)
        except Exception:
            continue
        if isinstance(ev, PaperOutcome):
            sid = ev.signal_id
            if sid in signals and sid not in outcome_lookup:
                outcome_lookup[sid] = ev

    for sid, otype in outcome_results.items():
        if otype == "tp_hit":
            wins += 1
        elif otype == "sl_hit":
            losses += 1
        elif otype == "expired":
            expired += 1
        else:
            unresolved_non_signal += 1
            continue

        sinfo = signals[sid]
        outcome = outcome_lookup.get(sid)
        net = float(outcome.net_r) if outcome and isinstance(outcome.net_r, (int, float)) else 0.0
        ot = int(outcome.outcome_time) if outcome and isinstance(outcome.outcome_time, int) else 0
        total_r += net
        if ot:
            outcome_times.append(ot)

        direction = sinfo["direction"]
        symbol = sinfo["symbol"]
        sv = sinfo["strategy_version"]
        for key, store in (
            (("dir", direction), by_direction),
            (("sym", symbol), by_symbol),
            (("sv", sv), by_strategy),
        ):
            entry = store[key]
            if otype == "tp_hit":
                entry["wins"] += 1
            elif otype == "sl_hit":
                entry["losses"] += 1
            elif otype == "expired":
                entry["expired"] += 1
            entry["total_r"] += net

    unresolved_count = unresolved_non_signal
    resolved_count = wins + losses + expired

    win_rate = _safe_div(wins, resolved_count) if resolved_count else None
    average_r = _safe_div(total_r, resolved_count) if resolved_count else None
    average_score = _avg(all_scores) if all_scores else None
    min_score = min(all_scores) if all_scores else None
    max_score = max(all_scores) if all_scores else None

    score_buckets = {b: 0 for b in SCORE_BUCKETS}
    for s in all_scores:
        score_buckets[score_bucket(float(s))] += 1

    unresolved_by_dir: Dict[str, int] = {}
    unresolved_by_sym: Dict[str, int] = {}
    unresolved_by_sv: Dict[str, int] = {}
    for sid, otype in outcome_results.items():
        if not otype:
            sinfo = signals[sid]
            d = sinfo["direction"]
            s = sinfo["symbol"]
            sv = sinfo["strategy_version"]
            unresolved_by_dir[d] = unresolved_by_dir.get(d, 0) + 1
            unresolved_by_sym[s] = unresolved_by_sym.get(s, 0) + 1
            unresolved_by_sv[sv] = unresolved_by_sv.get(sv, 0) + 1

    by_direction = _finalize(by_direction, unresolved_by_dir)
    by_symbol = _finalize(by_symbol, unresolved_by_sym)
    by_strategy_version = _finalize(by_strategy, unresolved_by_sv)

    utc = _utc_from_times(signal_times, outcome_times)

    metrics = ResearchMetrics(
        total_signals=total_signals,
        long_count=long_count,
        short_count=short_count,
        resolved_count=resolved_count,
        unresolved_count=unresolved_count,
        wins=wins,
        losses=losses,
        expired=expired,
        win_rate=win_rate,
        total_r=total_r,
        average_r=average_r,
        average_score=average_score,
        min_score=min_score,
        max_score=max_score,
        score_buckets=score_buckets,
        by_direction=by_direction,
        by_symbol=by_symbol,
        by_strategy_version=by_strategy_version,
        utc=utc,
    )

    r = ResearchReport(
        data_state=DataState.PRESENT,
        metrics=metrics,
        total_signals=total_signals,
        long_count=long_count,
        short_count=short_count,
        resolved_count=resolved_count,
        unresolved_count=unresolved_count,
        wins=wins,
        losses=losses,
        expired=expired,
        win_rate=win_rate,
        total_r=total_r,
        average_r=average_r,
        average_score=average_score,
        min_score=min_score,
        max_score=max_score,
        score_buckets=score_buckets,
        by_direction=by_direction,
        by_symbol=by_symbol,
        by_strategy_version=by_strategy_version,
        utc=utc,
        wins_status=DataState.PRESENT,
        losses_status=DataState.PRESENT,
        expired_status=DataState.PRESENT,
        win_rate_status=(DataState.PRESENT if win_rate is not None else DataState.UNAVAILABLE),
        average_r_status=(DataState.PRESENT if average_r is not None else DataState.UNAVAILABLE),
        total_r_status=DataState.PRESENT,
        malformed_count=malformed,
        skipped_outcome_count=skipped_outcome,
    )

    if json_out:
        _emit_json(r)

    return r


def _utc_to_dict(utc: Optional[UtcStats]) -> Optional[Dict[str, Any]]:
    if utc is None:
        return None
    return {
        "first_signal_utc": utc.first_signal_utc,
        "last_signal_utc": utc.last_signal_utc,
        "first_outcome_utc": utc.first_outcome_utc,
        "last_outcome_utc": utc.last_outcome_utc,
        "outcome_spread_hours": utc.outcome_spread_hours,
    }


def _emit_json(report: ResearchReport) -> None:
    payload = {
        "research_report": {
            "data_state": report.data_state.value,
            "metrics": None if report.metrics is None else {
                "total_signals": report.metrics.total_signals,
                "long_count": report.metrics.long_count,
                "short_count": report.metrics.short_count,
                "resolved_count": report.metrics.resolved_count,
                "unresolved_count": report.metrics.unresolved_count,
                "wins": report.metrics.wins,
                "losses": report.metrics.losses,
                "expired": report.metrics.expired,
                "win_rate": report.metrics.win_rate,
                "total_r": report.metrics.total_r,
                "average_r": report.metrics.average_r,
                "average_score": report.metrics.average_score,
                "min_score": report.metrics.min_score,
                "max_score": report.metrics.max_score,
                "score_buckets": report.metrics.score_buckets,
                "by_direction": {k: {
                    "total": v.total,
                    "wins": v.wins,
                    "losses": v.losses,
                    "expired": v.expired,
                    "unresolved": v.unresolved,
                    "total_r": v.total_r,
                    "win_rate": v.win_rate,
                    "average_r": v.average_r,
                    "average_score": v.average_score,
                    "min_score": v.min_score,
                    "max_score": v.max_score,
                    "sample": v.sample,
                    "insufficient_sample": v.insufficient_sample,
                } for k, v in report.metrics.by_direction.items()},
                "by_symbol": {k: {
                    "total": v.total,
                    "wins": v.wins,
                    "losses": v.losses,
                    "expired": v.expired,
                    "unresolved": v.unresolved,
                    "total_r": v.total_r,
                    "win_rate": v.win_rate,
                    "average_r": v.average_r,
                    "average_score": v.average_score,
                    "min_score": v.min_score,
                    "max_score": v.max_score,
                    "sample": v.sample,
                    "insufficient_sample": v.insufficient_sample,
                } for k, v in report.metrics.by_symbol.items()},
                "by_strategy_version": {k: {
                    "total": v.total,
                    "wins": v.wins,
                    "losses": v.losses,
                    "expired": v.expired,
                    "unresolved": v.unresolved,
                    "total_r": v.total_r,
                    "win_rate": v.win_rate,
                    "average_r": v.average_r,
                    "average_score": v.average_score,
                    "min_score": v.min_score,
                    "max_score": v.max_score,
                    "sample": v.sample,
                    "insufficient_sample": v.insufficient_sample,
                } for k, v in report.metrics.by_strategy_version.items()},
                "utc": {
                    "first_signal_utc": report.metrics.utc.first_signal_utc if report.metrics.utc else None,
                    "last_signal_utc": report.metrics.utc.last_signal_utc if report.metrics.utc else None,
                    "first_outcome_utc": report.metrics.utc.first_outcome_utc if report.metrics.utc else None,
                    "last_outcome_utc": report.metrics.utc.last_outcome_utc if report.metrics.utc else None,
                    "outcome_spread_hours": report.metrics.utc.outcome_spread_hours if report.metrics.utc else None,
                },
                "mfe": report.metrics.mfe,
                "mae": report.metrics.mae,
                "market_regime": report.metrics.market_regime,
                "drawdown": report.metrics.drawdown,
            },
            "mfe_status": report.mfe_status.value,
            "mae_status": report.mae_status.value,
            "market_regime_status": report.market_regime_status.value,
            "drawdown_status": report.drawdown_status.value,
            "malformed_count": report.malformed_count,
            "skipped_outcome_count": report.skipped_outcome_count,
        }
    }
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _run_cli() -> None:
    ap = argparse.ArgumentParser(prog="research.analyze")
    ap.add_argument("--log", required=True, help="Path to JSONL ledger")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = ap.parse_args()
    report = run_analysis(log=args.log, json_out=False)
    if args.json:
        _emit_json(report)
    else:
        _emit_text(report)


def _emit_text(report: ResearchReport) -> None:
    print(f"data_state={report.data_state.value}")
    print(f"total_signals={report.total_signals}")
    print(f"long_count={report.long_count}")
    print(f"short_count={report.short_count}")
    print(f"resolved_count={report.resolved_count}")
    print(f"unresolved_count={report.unresolved_count}")
    print(f"wins={report.wins}")
    print(f"losses={report.losses}")
    print(f"expired={report.expired}")
    if report.win_rate is not None:
        print(f"win_rate={report.win_rate:.4f}")
    else:
        print("win_rate=unavailable")
    if report.average_r is not None:
        print(f"average_r={report.average_r:.4f}")
    else:
        print("average_r=unavailable")
    print(f"total_r={report.total_r if report.total_r is not None else 'unavailable'}")
    if report.average_score is not None:
        print(f"average_score={report.average_score:.2f}")
    else:
        print("average_score=unavailable")
    if report.min_score is not None:
        print(f"min_score={report.min_score:.2f}")
    else:
        print("min_score=unavailable")
    if report.max_score is not None:
        print(f"max_score={report.max_score:.2f}")
    else:
        print("max_score=unavailable")
    print("score_buckets:")
    for bucket, count in report.score_buckets.items():
        print(f"  {bucket}: {count}")
    print(f"mfe={'unavailable' if report.mfe_status is DataState.UNAVAILABLE else report.mfe}")
    print(f"mae={'unavailable' if report.mae_status is DataState.UNAVAILABLE else report.mae}")
    print(f"market_regime={'unavailable' if report.market_regime_status is DataState.UNAVAILABLE else report.market_regime}")
    print(f"drawdown={'unavailable' if report.drawdown_status is DataState.UNAVAILABLE else report.drawdown}")
    if report.utc:
        print(f"first_signal_utc={report.utc.first_signal_utc}")
        print(f"last_signal_utc={report.utc.last_signal_utc}")
        if report.utc.first_outcome_utc:
            print(f"first_outcome_utc={report.utc.first_outcome_utc}")
        if report.utc.last_outcome_utc:
            print(f"last_outcome_utc={report.utc.last_outcome_utc}")
        if report.utc.outcome_spread_hours is not None:
            print(f"outcome_spread_hours={report.utc.outcome_spread_hours:.4f}")
    print("by_direction:")
    for d, g in report.by_direction.items():
        print(f"  {d}: total={g.total} wins={g.wins} losses={g.losses} expired={g.expired} unresolved={g.unresolved} win_rate={g.win_rate} avg_r={g.average_r} avg_score={g.average_score}")
    print("by_symbol:")
    for s, g in report.by_symbol.items():
        print(f"  {s}: total={g.total} wins={g.wins} losses={g.losses} expired={g.expired} unresolved={g.unresolved} win_rate={g.win_rate} avg_r={g.average_r} avg_score={g.average_score}")
    print("by_strategy_version:")
    for v, g in report.by_strategy_version.items():
        insuff = " insufficient_sample" if g.insufficient_sample else ""
        print(f"  {v}: total={g.total} wins={g.wins} losses={g.losses} expired={g.expired} unresolved={g.unresolved} win_rate={g.win_rate} avg_r={g.average_r} avg_score={g.average_score}{insuff}")
    print(f"malformed_count={report.malformed_count}")
    print(f"skipped_outcome_count={report.skipped_outcome_count}")


if __name__ == "__main__":
    _run_cli()
