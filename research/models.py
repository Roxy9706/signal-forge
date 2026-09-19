
"""
P7 Performance Analyzer — deterministic read-only research models.

Immutable/frozen dataclasses only. No trading. No network. No credentials.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Data quality states
# ---------------------------------------------------------------------------

class DataState(Enum):
    NO_DATA = "no_data"
    PRESENT = "present"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Score buckets (exact spec)
# ---------------------------------------------------------------------------

SCORE_BUCKETS = ["<65", "65-69.9", "70-74.9", "75-79.9", "80-84.9", "85-89.9", "90+"]

MIN_ANALYSIS_SAMPLES = 30


def score_bucket(score: float) -> str:
    if score < 65:
        return "<65"
    if score < 70:
        return "65-69.9"
    if score < 75:
        return "70-74.9"
    if score < 80:
        return "75-79.9"
    if score < 85:
        return "80-84.9"
    if score < 90:
        return "85-89.9"
    return "90+"


# ---------------------------------------------------------------------------
# UTC time stats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UtcStats:
    first_signal_utc: Optional[str] = None
    last_signal_utc: Optional[str] = None
    first_outcome_utc: Optional[str] = None
    last_outcome_utc: Optional[str] = None
    outcome_spread_hours: Optional[float] = None


# ---------------------------------------------------------------------------
# Single-group breakdown (direction / symbol / strategy_version)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupMetrics:
    total: int = 0
    wins: int = 0
    losses: int = 0
    expired: int = 0
    unresolved: int = 0
    total_r: float = 0.0
    win_rate: Optional[float] = None
    average_r: Optional[float] = None
    average_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    sample: int = 0
    insufficient_sample: bool = False


# ---------------------------------------------------------------------------
# Full metrics (nested)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResearchMetrics:
    total_signals: int = 0
    long_count: int = 0
    short_count: int = 0
    resolved_count: int = 0
    unresolved_count: int = 0
    wins: int = 0
    losses: int = 0
    expired: int = 0
    win_rate: Optional[float] = None
    total_r: float = 0.0
    average_r: Optional[float] = None
    average_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None

    mfe: Optional[float] = None
    mae: Optional[float] = None
    market_regime: Optional[str] = None
    drawdown: Optional[float] = None

    score_buckets: dict = field(default_factory=dict)
    by_direction: dict = field(default_factory=dict)
    by_symbol: dict = field(default_factory=dict)
    by_strategy_version: dict = field(default_factory=dict)

    utc: Optional[UtcStats] = None


# ---------------------------------------------------------------------------
# Flat report — mirrors the test contract so analysis consumers see a
# single namespace.  Nested `metrics` is also available for structured
# JSON output.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResearchReport:
    data_state: DataState = DataState.NO_DATA

    # flat copies of the headline metrics — None means "not available"
    total_signals: Optional[int] = None
    long_count: Optional[int] = None
    short_count: Optional[int] = None
    resolved_count: Optional[int] = None
    unresolved_count: Optional[int] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    expired: Optional[int] = None
    win_rate: Optional[float] = None
    total_r: Optional[float] = None
    average_r: Optional[float] = None
    average_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None

    # unavailable research metrics — always None for P7 Phase 1
    mfe: Optional[float] = None
    mae: Optional[float] = None
    market_regime: Optional[str] = None
    drawdown: Optional[float] = None

    score_buckets: dict = field(default_factory=dict)
    by_direction: dict = field(default_factory=dict)
    by_symbol: dict = field(default_factory=dict)
    by_strategy_version: dict = field(default_factory=dict)
    utc: Optional[UtcStats] = None

    # nested structured metrics (also serialised in JSON output)
    metrics: Optional[ResearchMetrics] = None

    # per-metric data-state guards (ZERO vs UNAVAILABLE)
    wins_status: DataState = DataState.NO_DATA
    losses_status: DataState = DataState.NO_DATA
    expired_status: DataState = DataState.NO_DATA
    win_rate_status: DataState = DataState.NO_DATA
    average_r_status: DataState = DataState.NO_DATA
    total_r_status: DataState = DataState.NO_DATA

    mfe_status: DataState = DataState.UNAVAILABLE
    mae_status: DataState = DataState.UNAVAILABLE
    market_regime_status: DataState = DataState.UNAVAILABLE
    drawdown_status: DataState = DataState.UNAVAILABLE

    malformed_count: int = 0
    skipped_outcome_count: int = 0


# ---------------------------------------------------------------------------
# Flat report — mirrors the test contract so analysis consumers see a
# single namespace.  Nested `metrics` is also available for structured
# JSON output.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResearchReport:
    data_state: DataState = DataState.NO_DATA

    # flat copies of the headline metrics — None means "not available"
    total_signals: Optional[int] = None
    long_count: Optional[int] = None
    short_count: Optional[int] = None
    resolved_count: Optional[int] = None
    unresolved_count: Optional[int] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    expired: Optional[int] = None
    win_rate: Optional[float] = None
    total_r: Optional[float] = None
    average_r: Optional[float] = None
    average_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None

    # unavailable research metrics — always None for P7 Phase 1
    mfe: Optional[float] = None
    mae: Optional[float] = None
    market_regime: Optional[str] = None
    drawdown: Optional[float] = None

    score_buckets: dict = field(default_factory=dict)
    by_direction: dict = field(default_factory=dict)
    by_symbol: dict = field(default_factory=dict)
    by_strategy_version: dict = field(default_factory=dict)
    utc: Optional[UtcStats] = None

    # nested structured metrics (also serialised in JSON output)
    metrics: Optional[ResearchMetrics] = None

    # per-metric data-state guards (ZERO vs UNAVAILABLE)
    wins_status: DataState = DataState.NO_DATA
    losses_status: DataState = DataState.NO_DATA
    expired_status: DataState = DataState.NO_DATA
    win_rate_status: DataState = DataState.NO_DATA
    average_r_status: DataState = DataState.NO_DATA
    total_r_status: DataState = DataState.NO_DATA

    mfe_status: DataState = DataState.UNAVAILABLE
    mae_status: DataState = DataState.UNAVAILABLE
    market_regime_status: DataState = DataState.UNAVAILABLE
    drawdown_status: DataState = DataState.UNAVAILABLE

    malformed_count: int = 0
    skipped_outcome_count: int = 0
