# Binance contracts filter
#
# Filter Binance Futures contracts for paper observation.
# Public endpoints only. No API keys.
#
# Filters:
#   - status == "TRADING"
#   - quoteAsset == "USDT"
#   - quoteMargin == false (spot-like, not portfolio margin)
#   - not in exclude list (e.g. leveraged tokens, BTCUP/BTCDOWN)
#   - optional volume gate (optional)
#
# symbol_universe.py  —  automatic discovery of liquid USDT-M contracts
#   Uses Binance USDⓈ-M public market data only:
#     GET /fapi/v1/exchangeInfo  →  symbol list, status, quoteAsset
#     GET /fapi/v1/ticker/24hr  →  24h quoteVolume (liquidity gate)
#   No private endpoints. No API keys. No orders. Paper-only.

from __future__ import annotations
import math
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import logging


if TYPE_CHECKING:
    from binance.market_data import BinancePublicClient


LOG = logging.getLogger(__name__)

# Default exclude symbols for paper observation
DEFAULT_EXCLUDE = frozenset({
    # Leveraged tokens (not suitable for spot-like paper trading)
    "BTCUP", "BTCDOWN", "ETHUP", "ETHDOWN",
    "SOLUP", "SOLDOWN", "AXSUP", "AXSDOWN",
    # Basket/other tokens
    "BCDN", "DEI", "FORB",
})

# Frozen liquidity threshold (USDT 24h quote volume)
MIN_QUOTE_VOLUME_USDT = 50_000_000


def is_valid_symbol(symbol: str) -> bool:
    """
    Check if a symbol is valid for paper observation.

    Criteria:
    - Not in the exclude list
    - Ends with "USDT" (quote asset is USDT) — known stablecoin pairs
    - Not a leveraged token ticker (UP/DOWN suffix after non-USDT base)

    NOTE: In production, this would call exchangeInfo and check
    status=="TRADING" and quoteAsset=="USDT". Here we use a whitelist
    approach for deterministic testing.
    """
    if symbol in DEFAULT_EXCLUDE:
        return False

    # USDT-margined contracts / spot USDT pairs
    if symbol.endswith("USDT") and len(symbol) > 4:
        # Avoid leveraged tokens: BTCUP, BTCDOWN etc.
        base = symbol[:-4]  # strip USDT
        if base.endswith("UP") or base.endswith("DOWN"):
            return False
        return True

    return False


def get_active_usdt_symbols() -> List[str]:
    """
    Return a list of active USDT-margined symbols.

    In production this calls exchangeInfo and filters.
    For testing/offline use, return a small defaults list.
    """
    return [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "MATICUSDT",
        "LINKUSDT",
    ]


def build_symbol_universe(
    from_exchange_info: Optional[List[dict]] = None,
    max_symbols: int = 10,
) -> List[str]:
    """
    Build the paper-observation symbol universe.

    Args:
        from_exchange_info: optional list of exchangeInfo symbols.
            If None, uses get_active_usdt_symbols() defaults.
        max_symbols: maximum number of symbols to include (Termux safety cap).

    Returns:
        List of symbol strings, sorted alphabetically, capped at max_symbols.
    """
    if from_exchange_info is not None:
        symbols = [
            s.get("symbol", "")
            for s in from_exchange_info
            if is_valid_symbol(s.get("symbol", ""))
        ]
    else:
        symbols = get_active_usdt_symbols()

    # Deduplicate and sort
    symbols = sorted(set(symbols))
    return symbols[:max_symbols]


def _passes_exchange_info_filters(sym_info: Dict[str, Any]) -> bool:
    """
    Check whether a symbol in exchangeInfo passes the contract filters.

    Filters:
    - status == "TRADING"
    - quoteAsset == "USDT"
    - contractType == "PERPETUAL" (authoritative perpetual test;
      missing / CURRENT_QUARTER / NEXT_QUARTER / TRADIFI_PERPETUAL →
      rejected, fail closed)
    - underlyingType == "COIN" (secondary guard: hard-rejects INDEX-style
      products such as BTCDOMUSDT / ALLUSDT, EQUITY, and missing values —
      fail closed)
    - symbol not in DEFAULT_EXCLUDE
    - symbol ends with "USDT" and base not a leveraged token (UP/DOWN)
    - len(symbol) > 4
    """
    symbol = sym_info.get("symbol", "")
    if not isinstance(symbol, str) or not symbol:
        return False
    if symbol in DEFAULT_EXCLUDE:
        return False
    if not symbol.endswith("USDT") or len(symbol) <= 4:
        return False
    base = symbol[:-4]
    if base.endswith("UP") or base.endswith("DOWN"):
        return False
    if sym_info.get("status") != "TRADING":
        return False
    if sym_info.get("quoteAsset") != "USDT":
        return False
    if sym_info.get("contractType") != "PERPETUAL":
        return False
    if sym_info.get("underlyingType") != "COIN":
        return False
    return True


def _parse_quote_volume(ticker: Dict[str, Any]) -> Optional[float]:
    """
    Extract 24h quote volume from a ticker dict. Fails closed on malformed/missing.
    """
    qv = ticker.get("quoteVolume")
    if qv is None:
        return None
    # Reject booleans explicitly: float(True) == 1.0 would otherwise pass parsing.
    if isinstance(qv, bool):
        return None
    try:
        val = float(qv)
    except (TypeError, ValueError):
        return None
    if math.isinf(val) or math.isnan(val):
        return None
    return val


def discover_liquid_usdt_symbols(
    client: "BinancePublicClient",
    max_symbols: int = 10,
    min_quote_volume_usdt: float = MIN_QUOTE_VOLUME_USDT,
) -> List[str]:
    """
    Discover active/liquid Binance USDⓈ-M USDT Futures pairs using public
    market data only.

    Uses:
      - GET /fapi/v1/exchangeInfo  (symbol metadata: status, quoteAsset)
      - GET /fapi/v1/ticker/24hr   (24h quoteVolume for liquidity gate)

    Filtering:
      - status == "TRADING"
      - quoteAsset == "USDT"
      - not in DEFAULT_EXCLUDE
      - not a leveraged token (base ends with UP/DOWN)
      - 24h quoteVolume >= min_quote_volume_usdt
      - malformed/missing quoteVolume → excluded (fails closed)

    Returns:
        List of symbol strings, deduplicated, sorted alphabetically,
        capped at max_symbols.

    Raises:
        ValueError: if exchangeInfo or ticker responses are structurally
            malformed in a way that prevents filtering.
        urllib.error.URLError / HTTPError: propagated from the client on
            network/HTTP failure after retries exhausted.
        json.JSONDecodeError: propagated if the response body is not valid JSON.

    Paper-only: no private endpoints, no API keys, no orders, no execution.
    """
    # --- Step 1: fetch exchangeInfo ---
    exchange_info = client.get_exchange_info()
    if not isinstance(exchange_info, dict):
        raise ValueError(
            f"exchangeInfo: expected dict, got {type(exchange_info).__name__}"
        )

    symbols_list = exchange_info.get("symbols")
    if not isinstance(symbols_list, list):
        raise ValueError(
            f"exchangeInfo.symbols: expected list, got "
            f"{type(symbols_list).__name__ if symbols_list is not None else 'None'}"
        )

    # --- Step 2: filter by exchangeInfo fields ---
    candidate_symbols: List[str] = []
    for sym_info in symbols_list:
        if not isinstance(sym_info, dict):
            continue
        if _passes_exchange_info_filters(sym_info):
            candidate_symbols.append(sym_info["symbol"])

    if not candidate_symbols:
        # Valid result: no symbols passed the exchangeInfo filters.
        return []

    # --- Step 3: fetch bulk 24h ticker (all symbols, no symbol param) ---
    ticker_response = client.get_24hr_ticker()
    if not isinstance(ticker_response, list):
        raise ValueError(
            f"ticker/24hr: expected list, got {type(ticker_response).__name__}"
        )

    # Build a lookup: symbol -> ticker dict
    ticker_by_symbol: Dict[str, Dict[str, Any]] = {}
    for t in ticker_response:
        if not isinstance(t, dict):
            continue
        sym = t.get("symbol")
        if isinstance(sym, str) and sym:
            ticker_by_symbol[sym] = t

    # --- Step 4: apply liquidity gate ---
    liquid_symbols: List[str] = []
    for sym in candidate_symbols:
        ticker = ticker_by_symbol.get(sym)
        if ticker is None:
            # No ticker data for this symbol -> fails closed (exclude)
            continue
        qv = _parse_quote_volume(ticker)
        if qv is None:
            # Malformed/missing quoteVolume -> fails closed (exclude)
            continue
        if qv >= min_quote_volume_usdt:
            liquid_symbols.append(sym)

    # --- Step 5: rank by 24h quoteVolume descending, select top max_symbols,
    #     then sort those selected symbols alphabetically for deterministic output.
    liquid_with_volume: List[tuple[str, float]] = []
    for sym in liquid_symbols:
        ticker = ticker_by_symbol.get(sym)
        if ticker is None:
            continue
        qv = _parse_quote_volume(ticker)
        if qv is None:
            continue
        liquid_with_volume.append((sym, qv))

    # Deduplicate (should already be unique, but guard) and rank.
    seen: set[str] = set()
    ranked: list[tuple[str, float]] = []
    for sym, qv in liquid_with_volume:
        if sym in seen:
            continue
        seen.add(sym)
        ranked.append((sym, qv))

    # Stable sort: descending by quoteVolume; on ties, ascending by symbol
    # so the selection is deterministic regardless of input order.
    ranked.sort(key=lambda item: (-item[1], item[0]))

    top_n = ranked[:max_symbols]
    selected = sorted([sym for sym, _qv in top_n])
    return selected


# Re-export MAX_SYMBOLS from market_data so discovery tests and callers can
# import it from one place (binance.contracts) without reaching into market_data.
# This is a convenience re-export; the authoritative constant stays in
# binance.market_data.MAX_SYMBOLS.
try:
    from binance.market_data import MAX_SYMBOLS as _market_max_symbols  # type: ignore[import]
except ImportError:
    _market_max_symbols = 10

# Keep a local alias so the name exists in binance.contracts.  If
# binance.market_data is unavailable for any reason, fall back to 10 (the
# Termux safety cap documented in market_data.py).
MAX_SYMBOLS: int = _market_max_symbols
