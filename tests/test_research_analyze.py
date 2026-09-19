"""P7 Phase 1: Deterministic read-only Performance Analyzer — tests (RED first)."""
from __future__ import annotations
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List

import pytest

# The P7 research package lives at the repo root (research/). Add the repo root
# to sys.path so `research` and its top-level dependencies (storage.jsonl,
# paper.models) import regardless of the working directory pytest runs from.
_TEST_FILE_DIR = Path(__file__).resolve().parent   # SignalForge/tests/
_SIGNALFORGE_ROOT = _TEST_FILE_DIR.parent          # SignalForge/ (repo root)
sys.path.insert(0, str(_SIGNALFORGE_ROOT))
os.environ["PYTHONPATH"] = str(_SIGNALFORGE_ROOT)

from research.models import DataState, ResearchReport, ResearchMetrics
from research.analyze import run_analysis


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _signal(
    signal_id="s1",
    symbol="BTCUSDT",
    direction="LONG",
    signal_time=1_700_000_000_000,
    score=80.0,
    strategy_version="v1",
    **kw,
) -> dict:
    d = {
        "event": "signal",
        "signal_id": signal_id,
        "symbol": symbol,
        "direction": direction,
        "signal_time": signal_time,
        "entry": 100.0,
        "sl": 95.0,
        "tp": 110.0,
        "score": score,
        "timeframe": "5m",
        "horizon_ms": 4 * 3600 * 1000,
        "strategy_version": strategy_version,
        "generated_at": signal_time,
    }
    d.update(kw)
    return d


def _outcome(
    signal_id="s1",
    symbol="BTCUSDT",
    direction="LONG",
    outcome="tp_hit",
    outcome_time=1_700_000_040_000,
    gross_r=2.0,
    net_r=1.8,
    holding_ms=40_000,
    strategy_version="v1",
    **kw,
) -> dict:
    d = {
        "event": outcome,
        "signal_id": signal_id,
        "symbol": symbol,
        "direction": direction,
        "outcome": outcome,
        "outcome_time": outcome_time,
        "entry": 100.0,
        "sl": 95.0,
        "tp": 110.0,
        "gross_r": gross_r,
        "net_r": net_r,
        "holding_ms": holding_ms,
        "fee_pct": 0.001,
        "slippage_pct": 0.001,
        "horizon_ms": 4 * 3600 * 1000,
        "strategy_version": strategy_version,
        "close_time": outcome_time,
    }
    d.update(kw)
    return d


def _write_jsonl(path: str, records: List[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


# ---------------------------------------------------------------------------
# NO DATA
# ---------------------------------------------------------------------------

def test_missing_ledger_is_no_data():
    report = run_analysis(log="")
    assert report.data_state is DataState.NO_DATA
    assert report.total_signals is None


def test_empty_ledger_is_no_data(tmp_path):
    p = str(tmp_path / "empty.jsonl")
    Path(p).touch()
    report = run_analysis(log=p)
    assert report.data_state is DataState.NO_DATA
    assert report.total_signals is None


# ---------------------------------------------------------------------------
# SCOPE: metrics not in ledger are UNAVAILABLE, never zero
# ---------------------------------------------------------------------------

def test_unavailable_metrics_when_not_in_ledger(tmp_path):
    p = str(tmp_path / "x.jsonl")
    _write_jsonl(p, [_signal("s1"), _outcome("s1")])
    report = run_analysis(log=p)
    assert report.data_state is DataState.PRESENT
    assert report.mfe is None
    assert report.mae is None
    assert report.market_regime is None
    assert report.drawdown is None
    assert report.mfe_status is DataState.UNAVAILABLE
    assert report.mae_status is DataState.UNAVAILABLE
    assert report.market_regime_status is DataState.UNAVAILABLE
    assert report.drawdown_status is DataState.UNAVAILABLE


# ---------------------------------------------------------------------------
# ZERO vs UNAVAILABLE — ledger has signals but NO outcomes
# ---------------------------------------------------------------------------

def test_zero_counts_when_no_outcomes(tmp_path):
    p = str(tmp_path / "x.jsonl")
    _write_jsonl(p, [_signal("s1"), _signal("s2"), _signal("s3")])
    report = run_analysis(log=p)
    assert report.data_state is DataState.PRESENT
    assert report.total_signals == 3
    assert report.resolved_count == 0
    assert report.wins == 0
    assert report.losses == 0
    assert report.expired == 0
    assert report.win_rate is None  # cannot divide 0/0
    assert report.average_r is None
    assert report.total_r == 0.0
    # counts exist and are zero — not unavailable
    assert report.wins_status is not DataState.UNAVAILABLE
    assert report.average_r_status is DataState.UNAVAILABLE


# ---------------------------------------------------------------------------
# CLEAN HAPPY PATH
# ---------------------------------------------------------------------------

def test_clean_ledger_metrics(tmp_path):
    p = str(tmp_path / "x.jsonl")
    recs = [
        _signal("s1", "BTCUSDT", "LONG", score=80.0, strategy_version="v1"),
        _outcome("s1", "BTCUSDT", "LONG", "tp_hit", gross_r=2.0, net_r=1.8),
        _signal("s2", "BTCUSDT", "LONG", score=70.0, strategy_version="v1"),
        _outcome("s2", "BTCUSDT", "LONG", "sl_hit", gross_r=-1.0, net_r=-1.2),
        _signal("s3", "BTCUSDT", "SHORT", score=90.0, strategy_version="v1"),
        _outcome("s3", "BTCUSDT", "SHORT", "expired", gross_r=0.0, net_r=0.0),
        _signal("s4", "ETHUSDT", "LONG", score=60.0, strategy_version="v2"),
        _outcome("s4", "ETHUSDT", "LONG", "tp_hit", gross_r=1.0, net_r=0.9),
        _signal("s5", "ETHUSDT", "SHORT", score=85.0, strategy_version="v2"),
    ]
    _write_jsonl(p, recs)
    r = run_analysis(log=p)
    assert r.data_state is DataState.PRESENT
    assert r.total_signals == 5
    assert r.long_count == 3
    assert r.short_count == 2
    assert r.resolved_count == 4
    assert r.unresolved_count == 1
    assert r.wins == 2
    assert r.losses == 1
    assert r.expired == 1
    assert r.win_rate == pytest.approx(0.5)
    assert r.total_r == pytest.approx(1.5)  # 1.8 - 1.2 + 0.0 + 0.9
    assert r.average_r == pytest.approx(0.375)
    assert r.average_score == pytest.approx((80 + 70 + 90 + 60 + 85) / 5)
    assert r.min_score == 60.0
    assert r.max_score == 90.0


# ---------------------------------------------------------------------------
# SCORE BUCKETS
# ---------------------------------------------------------------------------

def test_score_buckets(tmp_path):
    p = str(tmp_path / "x.jsonl")
    recs = []
    for i, (s, v) in enumerate(
        [("40", "<65"), ("67", "65-69.9"), ("72", "70-74.9"), ("77", "75-79.9"),
         ("82", "80-84.9"), ("87", "85-89.9"), ("95", "90+")]
    ):
        recs.append(_signal(f"s{i}", score=float(s)))
        recs.append(_outcome(f"s{i}", outcome="tp_hit"))
    _write_jsonl(p, recs)
    r = run_analysis(log=p)
    assert r.data_state is DataState.PRESENT
    assert r.score_buckets["<65"] == 1
    assert r.score_buckets["65-69.9"] == 1
    assert r.score_buckets["70-74.9"] == 1
    assert r.score_buckets["75-79.9"] == 1
    assert r.score_buckets["80-84.9"] == 1
    r.score_buckets["85-89.9"] == 1
    r.score_buckets["90+"] == 1


# ---------------------------------------------------------------------------
# BREAKDOWNS
# ---------------------------------------------------------------------------

def test_breakdown_by_direction(tmp_path):
    p = str(tmp_path / "x.jsonl")
    recs = [
        _signal("s1", "BTCUSDT", "LONG", score=80.0),
        _outcome("s1", outcome="tp_hit", gross_r=2.0, net_r=1.8),
        _signal("s2", "BTCUSDT", "SHORT", score=70.0),
        _outcome("s2", outcome="sl_hit", gross_r=-1.0, net_r=-1.2),
    ]
    _write_jsonl(p, recs)
    r = run_analysis(log=p)
    long = r.by_direction["LONG"]
    short = r.by_direction["SHORT"]
    assert long.total == 1
    assert long.wins == 1
    assert long.losses == 0
    assert long.expired == 0
    assert long.win_rate == pytest.approx(1.0)
    assert long.average_r == pytest.approx(1.8)
    assert short.total == 1
    assert short.wins == 0
    assert short.losses == 1
    assert short.win_rate == pytest.approx(0.0)
    assert short.average_r == pytest.approx(-1.2)


def test_breakdown_by_symbol(tmp_path):
    p = str(tmp_path / "x.jsonl")
    recs = [
        _signal("s1", "BTCUSDT", "LONG", score=80.0),
        _outcome("s1", "BTCUSDT", "LONG", "tp_hit", gross_r=2.0, net_r=1.8),
        _signal("s2", "ETHUSDT", "LONG", score=70.0),
        _outcome("s2", "ETHUSDT", "LONG", "sl_hit", gross_r=-1.0, net_r=-1.2),
        _signal("s3", "BTCUSDT", "SHORT", score=75.0),
        _outcome("s3", "BTCUSDT", "SHORT", "expired"),
    ]
    _write_jsonl(p, recs)
    r = run_analysis(log=p)
    btc = r.by_symbol["BTCUSDT"]
    assert btc.total == 2
    assert btc.wins == 1
    assert btc.losses == 0
    assert btc.expired == 1
    assert btc.win_rate == pytest.approx(0.5)


def test_breakdown_by_strategy_version(tmp_path):
    p = str(tmp_path / "x.jsonl")
    recs = [
        _signal("s1", strategy_version="v1", score=80.0),
        _outcome("s1", strategy_version="v1", outcome="tp_hit", gross_r=2.0, net_r=1.8),
        _signal("s2", strategy_version="v2", score=70.0),
        _outcome("s2", strategy_version="v2", outcome="sl_hit", gross_r=-1.0, net_r=-1.2),
    ]
    _write_jsonl(p, recs)
    r = run_analysis(log=p)
    v1 = r.by_strategy_version["v1"]
    assert v1.total == 1
    assert v1.wins == 1
    assert v1.average_r == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# SAMPLE SIZE GUARD
# ---------------------------------------------------------------------------

def test_small_group_is_marked_insufficient_sample(tmp_path):
    p = str(tmp_path / "x.jsonl")
    recs = []
    for i in range(5):
        recs.append(_signal(f"s{i}", strategy_version="vsmall"))
        recs.append(_outcome(f"s{i}", strategy_version="vsmall", outcome="tp_hit"))
    recs.append(_signal("big1", strategy_version="vbig"))
    recs.append(_outcome("big1", strategy_version="vbig", outcome="tp_hit"))
    _write_jsonl(p, recs)
    r = run_analysis(log=p)
    small = r.by_strategy_version["vsmall"]
    big = r.by_strategy_version["vbig"]
    assert small.insufficient_sample is True
    assert small.sample == 5
    assert big.insufficient_sample is True  # 1 < 30
    assert big.sample == 1


# ---------------------------------------------------------------------------
# MALFORMED RECORDS
# ---------------------------------------------------------------------------

def test_malformed_records_skipped_safely(tmp_path):
    p = str(tmp_path / "x.jsonl")
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(_signal("s1")) + "\n")
        f.write("{not json\n")
        f.write(json.dumps(_outcome("s1", outcome="tp_hit")) + "\n")
        f.write("\n")
        f.write("[]\n")
    r = run_analysis(log=p)
    assert r.data_state is DataState.PRESENT
    assert r.total_signals == 1
    assert r.resolved_count == 1


def test_records_missing_signal_id_skipped(tmp_path):
    p = str(tmp_path / "x.jsonl")
    _write_jsonl(p, [
        {"event": "signal", "symbol": "BTCUSDT"},
        _signal("s1"),
        _outcome("s1", outcome="tp_hit"),
    ])
    r = run_analysis(log=p)
    assert r.total_signals == 1


def test_outcome_without_matching_signal_is_skipped(tmp_path):
    p = str(tmp_path / "x.jsonl")
    _write_jsonl(p, [
        _outcome("ghost", outcome="tp_hit", gross_r=2.0, net_r=1.8),
        _signal("s1"),
    ])
    r = run_analysis(log=p)
    assert r.total_signals == 1
    assert r.resolved_count == 0


# ---------------------------------------------------------------------------
# UTC TIME STATS
# ---------------------------------------------------------------------------

def test_utc_time_stats(tmp_path):
    p = str(tmp_path / "x.jsonl")
    recs = [
        _signal("s1", signal_time=1_700_000_000_000),
        _outcome("s1", outcome_time=1_700_000_040_000),
        _signal("s2", signal_time=1_700_000_100_000),
        _outcome("s2", outcome_time=1_700_000_140_000),
    ]
    _write_jsonl(p, recs)
    r = run_analysis(log=p)
    assert r.utc is not None
    assert r.utc.first_signal_utc is not None
    assert r.utc.last_signal_utc is not None
    assert r.utc.outcome_spread_hours == pytest.approx(100 / 3600)  # 100000ms = 100s


def test_no_outcomes_means_no_outcome_time_stats(tmp_path):
    p = str(tmp_path / "x.jsonl")
    _write_jsonl(p, [_signal("s1", signal_time=1_700_000_000_000)])
    r = run_analysis(log=p)
    assert r.utc.first_signal_utc is not None
    # outcome spread unavailable because no outcomes
    assert r.utc.outcome_spread_hours is None


# ---------------------------------------------------------------------------
# JSON OUTPUT PATH
# ---------------------------------------------------------------------------

def test_json_output(tmp_path, capsys):
    p = str(tmp_path / "x.jsonl")
    _write_jsonl(p, [_signal("s1"), _outcome("s1", outcome="tp_hit")])
    run_analysis(log=p, json_out=True)
    captured = capsys.readouterr()
    # Should be valid JSON with research_report key
    import json as j
    payload = j.loads(captured.out)
    assert "research_report" in payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_missing_file_exits_safely(tmp_path, capsys):
    p = str(tmp_path / "nope.jsonl")
    proc = subprocess.run(
        [sys.executable, "-m", "research.analyze", "--log", p],
        capture_output=True,
        text=True,
        cwd=_SIGNALFORGE_ROOT,
        env={**os.environ, "PYTHONPATH": str(_SIGNALFORGE_ROOT)},
    )
    # Should not crash
    assert proc.returncode == 0
    proc = subprocess.run(
        [sys.executable, "-m", "research.analyze", "--log", p, "--json"],
        capture_output=True,
        text=True,
        cwd=_SIGNALFORGE_ROOT,
        env={**os.environ, "PYTHONPATH": str(_SIGNALFORGE_ROOT)},
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["research_report"]["data_state"] == DataState.NO_DATA.value


def test_cli_json_flag(tmp_path):
    p = str(tmp_path / "x.jsonl")
    _write_jsonl(p, [_signal("s1"), _outcome("s1", outcome="tp_hit")])
    proc = subprocess.run(
        [sys.executable, "-m", "research.analyze", "--log", p, "--json"],
        capture_output=True,
        text=True,
        cwd=_SIGNALFORGE_ROOT,
        env={**os.environ, "PYTHONPATH": str(_SIGNALFORGE_ROOT)},
    )
    assert proc.returncode == 0
    import json as j
    payload = j.loads(proc.stdout)
    assert "research_report" in payload
