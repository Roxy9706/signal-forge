"""D6 Volume & Liquidity — RVOL score + Binance USDT liquidity gate.

Per-timeframe RVOL:
  RVOL = current_volume / mean(previous_20_closed_volumes), excluding C_tf.
  Score buckets: >=2.0 -> 100, 1.5-1.99 -> 80, 1.0-1.49 -> 60,
                 0.75-0.99 -> 40, <0.75 -> 20.
  Insufficient history (<20 preceding candles) -> None (unavailable).

Liquidity gate (separate from score):
  Binance Futures USDT-margined only.
  GET fapi.binance.com/fapi/v1/ticker/24hr?symbol=<SYMBOL>
  liquidity_ok = float(quoteVolume) >= 50,000,000 USDT.
  Missing/malformed/unavailable -> False (data-quality issue, no invention).

D6 does NOT implement D8 aggregation, final direction, SL/TP, or trading.
D6 does NOT let quoteVolume alter the 0-100 RVOL score.
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple
import urllib.request
import urllib.error
import urllib.parse
import json

# Binance Futures 24hr ticker endpoint
_BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
_LIQUIDITY_THRESHOLD = 50_000_000.0  # USDT


# ---------------------------------------------------------------------------
# RVOL — pure, deterministic, closed-candle only
# ---------------------------------------------------------------------------

def compute_rvol(candles: List[dict]) -> Optional[float]:
    """RVOL at the latest closed candle (last in list).

    candles: list of dicts with 'volume' key, sorted oldest-to-newest,
             all closed (close_time <= T). The last candle is C_tf.

    Returns:
        RVOL = C_tf.volume / mean(previous_20 volumes), or None if
        fewer than 20 preceding candles exist, or if the baseline is
        non-positive.
    """
    if not candles or len(candles) < 21:
        return None

    current_vol = _safe_volume(candles[-1])
    if current_vol is None or current_vol <= 0:
        return None

    # Previous 20 candles (indices -21 .. -2), excluding C_tf at -1
    prev = candles[-21:-1]
    if len(prev) != 20:
        return None

    vols = []
    for c in prev:
        v = _safe_volume(c)
        if v is None or v < 0:
            return None
        vols.append(v)

    baseline = sum(vols) / 20.0
    if baseline <= 0:
        return None

    return current_vol / baseline


def volume_score(rvol: Optional[float]) -> Optional[int]:
    """Map RVOL to the approved 0-100 Volume score.

    >= 2.0  -> 100
    1.5-1.99 -> 80
    1.0-1.49 -> 60
    0.75-0.99 -> 40
    < 0.75 -> 20
    None -> None (unavailable)
    """
    if rvol is None:
        return None
    if rvol >= 2.0:
        return 100
    if rvol >= 1.5:
        return 80
    if rvol >= 1.0:
        return 60
    if rvol >= 0.75:
        return 40
    return 20


def _safe_volume(candle: dict) -> Optional[float]:
    """Extract volume as float, returning None on missing/invalid."""
    try:
        v = candle.get("volume")
        if v is None:
            return None
        f = float(v)
        if f < 0 or f != f or f == f + 1:  # negative, NaN, or Inf
            return None
        return f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Liquidity — pure parser + live client wrapper
# ---------------------------------------------------------------------------

def parse_liquidity_from_ticker(ticker_response: Any) -> Tuple[bool, bool]:
    """Parse liquidity_ok and data_ok from a Binance 24hr ticker response.

    Args:
        ticker_response: the parsed JSON from
            GET fapi.binance.com/fapi/v1/ticker/24hr?symbol=<SYMBOL>
            Either a dict (single symbol) or a list of dicts (all symbols).

    Returns:
        (liquidity_ok, data_ok)
          liquidity_ok: True iff quoteVolume >= 50M USDT.
          data_ok:      True iff quoteVolume was present, parseable, and
                        the symbol was found in the response.
    """
    # Normalize to a single symbol dict
    sym_dict = _extract_symbol_dict(ticker_response)
    if sym_dict is None:
        return (False, False)

    qv = _safe_quote_volume(sym_dict)
    if qv is None:
        return (False, False)

    return (qv >= _LIQUIDITY_THRESHOLD, True)


def _extract_symbol_dict(response: Any) -> Optional[dict]:
    """Extract the dict for our symbol from a 24hr ticker response.

    Binance returns either:
      - a dict (when ?symbol= was given)
      - a list of dicts (when no symbol filter)
    We accept either form.
    """
    if isinstance(response, dict):
        return response
    if isinstance(response, list):
        # Return the first entry — callers should filter by symbol before
        # passing in; if they pass the unfiltered list, take the first.
        if response:
            return response[0]
    return None


def _safe_quote_volume(sym_dict: dict) -> Optional[float]:
    """Extract quoteVolume as float, returning None on missing/invalid."""
    try:
        qv = sym_dict.get("quoteVolume")
        if qv is None:
            return None
        f = float(qv)
        if f < 0:
            return None
        return f
    except (TypeError, ValueError):
        return None


def check_liquidity(symbol: str) -> Tuple[bool, bool]:
    """Live Binance Futures 24hr ticker liquidity check.

    Args:
        symbol: e.g. "BTCUSDT" (must be USDT-margined).

    Returns:
        (liquidity_ok, data_ok)
    """
    url = f"{_BINANCE_FAPI}/ticker/24hr?symbol={urllib.parse.quote(symbol)}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SignalForge-D6/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except Exception:
        return (False, False)

    return parse_liquidity_from_ticker(data)
