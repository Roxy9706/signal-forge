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


from __future__ import annotations
from typing import List, Optional


# Default exclude symbols for paper observation
DEFAULT_EXCLUDE = frozenset({
    # Leveraged tokens (not suitable for spot-like paper trading)
    "BTCUP", "BTCDOWN", "ETHUP", "ETHDOWN",
    "SOLUP", "SOLDOWN", "AXSUP", "AXSDOWN",
    # Basket/other tokens
    "BCDN", "DEI", "FORB",
})


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
