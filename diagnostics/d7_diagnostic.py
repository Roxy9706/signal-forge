"""D7 rejection diagnostics — read-only classification of WHY D7 rejects.

Mirrors the existing D7 decision predicates from indicators.pullback exactly,
without altering production D7 behavior. Classifies each D7 evaluation that
reaches D7 into deterministic diagnostic categories.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import json

from indicators.pullback import (
    _select_retest_level_by_direction,
    find_prior_trend_side_candle,
    validate_pullback_long,
    validate_pullback_short,
    d7_pullback_pass,
)
from indicators.ema import compute_ema
from indicators.structure import most_recent_swing_high, most_recent_swing_low

D7_DIAGNOSTIC_CATEGORIES = [
    "RETEST_LEVEL_UNAVAILABLE",
    "NO_EMA20_RETEST",
    "NO_STRUCTURE_RETEST",
    "RETEST_FOUND_NO_REJECTION",
    "REJECTION_CLOSE_INVALID",
    "REJECTION_BODY_INVALID",
    "OTHER_EXISTING_D7_REJECTION",
]


def _classify_retest_level_source(
    candles_5m: List[dict],
    setup_index: int,
    direction: str,
) -> Tuple[Optional[float], str]:
    """Mirror _select_retest_level_by_direction and return (L, source)."""
    if setup_index < 0 or setup_index >= len(candles_5m):
        return None, "none"

    series_through_c = candles_5m[:setup_index + 1]
    ema20 = compute_ema(series_through_c, 20)
    if ema20 is not None:
        return ema20, "ema20"

    start_idx = max(0, setup_index - 20)
    window = candles_5m[start_idx:setup_index]
    if not window:
        return None, "none"

    if direction == 'LONG':
        swing = most_recent_swing_low(window, exclude_index=None)
        if swing is not None:
            return swing['low'], "structure"
    elif direction == 'SHORT':
        swing = most_recent_swing_high(window, exclude_index=None)
        if swing is not None:
            return swing['high'], "structure"

    return None, "none"


def classify_d7_rejection(
    candles_5m: List[dict],
    setup_index: int,
    direction: str,
) -> Dict[str, Any]:
    """Classify WHY D7 rejects a candidate into a deterministic category."""
    if direction not in ('LONG', 'SHORT'):
        return {
            "category": "OTHER_EXISTING_D7_REJECTION",
            "retest_level": None,
            "retest_source": "none",
            "retest_available": False,
            "pullback_found": False,
            "rejection_details": {"invalid_direction": True},
        }

    L, source = _classify_retest_level_source(candles_5m, setup_index, direction)

    if L is None:
        return {
            "category": "RETEST_LEVEL_UNAVAILABLE",
            "retest_level": None,
            "retest_source": "none",
            "retest_available": False,
            "pullback_found": False,
            "rejection_details": {"retest_level_unavailable": True},
        }

    P = find_prior_trend_side_candle(candles_5m, setup_index, L, direction)
    pullback_found = P is not None

    if not pullback_found:
        if source == "ema20":
            category = "NO_EMA20_RETEST"
        else:
            category = "NO_STRUCTURE_RETEST"
        return {
            "category": category,
            "retest_level": L,
            "retest_source": source,
            "retest_available": True,
            "pullback_found": False,
            "rejection_details": {
                "retest_source": source,
                "no_prior_trend_side_candle": True,
            },
        }

    C = candles_5m[setup_index]
    if direction == 'LONG':
        tested = C['low'] <= L
        closed_back = C['close'] > L
        body_valid = C['close'] > C['open']
    else:
        tested = C['high'] >= L
        closed_back = C['close'] < L
        body_valid = C['close'] < C['open']

    rejection_details = {
        "tested": tested,
        "closed_back": closed_back,
        "body_valid": body_valid,
        "retest_source": source,
    }

    if not tested:
        category = "RETEST_FOUND_NO_REJECTION"
    elif not closed_back:
        category = "REJECTION_CLOSE_INVALID"
    elif not body_valid:
        category = "REJECTION_BODY_INVALID"
    else:
        category = "OTHER_EXISTING_D7_REJECTION"

    return {
        "category": category,
        "retest_level": L,
        "retest_source": source,
        "retest_available": True,
        "pullback_found": True,
        "rejection_details": rejection_details,
    }


def build_d7_diagnostic_report(
    records: List[Dict],
) -> Dict[str, Any]:
    """Aggregate D7 diagnostic classification from stage records."""
    report: Dict[str, Any] = {
        "d7_evaluations": 0,
        "d7_pass": 0,
        "d7_rejected": 0,
        "d7_data_unavailable": 0,
        "d7_diagnostic_breakdown": {cat: 0 for cat in D7_DIAGNOSTIC_CATEGORIES},
        "d7_diagnostic_percentages": {cat: 0.0 for cat in D7_DIAGNOSTIC_CATEGORIES},
        "d7_long_breakdown": {cat: 0 for cat in D7_DIAGNOSTIC_CATEGORIES},
        "d7_short_breakdown": {cat: 0 for cat in D7_DIAGNOSTIC_CATEGORIES},
        "d7_retest_available_count": 0,
    }

    long_count = 0
    short_count = 0

    for rec in records:
        d7_pass = rec.get("d7_pass")
        d7_retest_available = rec.get("d7_retest_available")
        d7_category = rec.get("d7_diagnostic_category")
        d8_dir = rec.get("d8_direction")

        if d7_pass is None:
            continue

        report["d7_evaluations"] += 1

        if d7_pass is True:
            report["d7_pass"] += 1
        else:
            report["d7_rejected"] += 1
            if d7_retest_available is False:
                report["d7_data_unavailable"] += 1
            if d7_category and d7_category in report["d7_diagnostic_breakdown"]:
                report["d7_diagnostic_breakdown"][d7_category] += 1

        if d7_retest_available is True:
            report["d7_retest_available_count"] += 1

        if d8_dir == "LONG":
            long_count += 1
            if d7_category and d7_category in report["d7_long_breakdown"]:
                report["d7_long_breakdown"][d7_category] += 1
        elif d8_dir == "SHORT":
            short_count += 1
            if d7_category and d7_category in report["d7_short_breakdown"]:
                report["d7_short_breakdown"][d7_category] += 1

    total = report["d7_evaluations"]
    if total > 0:
        for cat in D7_DIAGNOSTIC_CATEGORIES:
            count = report["d7_diagnostic_breakdown"][cat]
            report["d7_diagnostic_percentages"][cat] = round(count / total * 100, 2)

    report["d7_long_count"] = long_count
    report["d7_short_count"] = short_count

    return report


def format_d7_diagnostic_text(report: Dict[str, Any]) -> str:
    """Render D7 diagnostic report as deterministic plain text."""
    lines = []
    lines.append("D7 Rejection Diagnostics (read-only)")
    lines.append("")
    lines.append(f"Total D7 evaluations: {report['d7_evaluations']}")
    lines.append(f"D7 passes: {report['d7_pass']}")
    lines.append(f"D7 rejected: {report['d7_rejected']}")
    lines.append(f"D7 data_unavailable: {report['d7_data_unavailable']}")
    lines.append(f"Retest available count: {report['d7_retest_available_count']}")
    lines.append("")

    if report['d7_evaluations'] > 0:
        lines.append("Diagnostic category breakdown:")
        for cat in D7_DIAGNOSTIC_CATEGORIES:
            count = report['d7_diagnostic_breakdown'][cat]
            pct = report['d7_diagnostic_percentages'][cat]
            lines.append(f"  {cat:<35} {count:>6} {pct:>5.1f}%")
        lines.append("")

    return "\n".join(lines)


def format_d7_diagnostic_json(report: Dict[str, Any]) -> str:
    """Serialize D7 diagnostic report as deterministic JSON."""
    return json.dumps(report, indent=2, sort_keys=True)
