# Symbol-specific exchange-filter precision for LIMIT orders (execution only).
#
# Pure functions: NO network access, NO credentials, NO strategy imports.
# Quantization is deterministic FLOOR toward the step grid:
#   - price is floored to PRICE_FILTER.tickSize
#   - quantity is floored to LOT_SIZE.stepSize
# Flooring never raises a BUY price above the signal entry and never
# increases exposure; geometry must be revalidated after rounding.
#
# Fail-closed: if filter information is missing or unusable, raise
# PrecisionDataUnavailableError and let the caller reject the order.

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional


class PrecisionDataUnavailableError(Exception):
    """Required exchange filter data is missing or invalid (fail closed)."""


def _to_decimal(value: Any, field: str) -> Decimal:
    if value is None:
        raise PrecisionDataUnavailableError(f"{field} is missing")
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise PrecisionDataUnavailableError(f"{field}={value!r} is not numeric")
    return d


def get_symbol_filters(exchange_info: Dict[str, Any], symbol: str) -> Dict[str, Optional[Decimal]]:
    """Extract PRICE_FILTER/LOT_SIZE for one symbol from an exchangeInfo payload.

    Returns dict with keys: tick_size, step_size, min_qty, min_notional
    (min_qty/min_notional may be None when the exchange omits them).

    Raises PrecisionDataUnavailableError if the symbol, PRICE_FILTER, or
    LOT_SIZE with a usable positive tick/step is absent (fail closed).
    """
    if not isinstance(exchange_info, dict):
        raise PrecisionDataUnavailableError("exchangeInfo payload is not a dict")
    symbols = exchange_info.get("symbols")
    if not isinstance(symbols, list):
        raise PrecisionDataUnavailableError("exchangeInfo has no 'symbols' list")

    entry = next((s for s in symbols if isinstance(s, dict) and s.get("symbol") == symbol), None)
    if entry is None:
        raise PrecisionDataUnavailableError(f"symbol {symbol!r} not present in exchangeInfo")

    filters = entry.get("filters") or []
    price_filter = next(
        (f for f in filters if isinstance(f, dict) and f.get("filterType") == "PRICE_FILTER"), None
    )
    lot_filter = next(
        (f for f in filters if isinstance(f, dict) and f.get("filterType") == "LOT_SIZE"), None
    )
    if price_filter is None:
        raise PrecisionDataUnavailableError(f"{symbol}: PRICE_FILTER missing")
    if lot_filter is None:
        raise PrecisionDataUnavailableError(f"{symbol}: LOT_SIZE filter missing")

    tick = _to_decimal(price_filter.get("tickSize"), f"{symbol}.tickSize")
    step = _to_decimal(lot_filter.get("stepSize"), f"{symbol}.stepSize")
    if tick <= 0:
        raise PrecisionDataUnavailableError(f"{symbol}: tickSize must be > 0, got {tick}")
    if step <= 0:
        raise PrecisionDataUnavailableError(f"{symbol}: stepSize must be > 0, got {step}")

    min_qty_raw = lot_filter.get("minQty")
    min_qty = _to_decimal(min_qty_raw, f"{symbol}.minQty") if min_qty_raw is not None else None

    min_notional = None
    for f in filters:
        if isinstance(f, dict) and f.get("filterType") == "MIN_NOTIONAL":
            raw = f.get("notional") or f.get("minNotional")
            if raw is not None:
                min_notional = _to_decimal(raw, f"{symbol}.minNotional")
            break

    return {
        "tick_size": tick,
        "step_size": step,
        "min_qty": min_qty,
        "min_notional": min_notional,
    }


def floor_to_step(value: float, step: Decimal) -> Decimal:
    """Deterministically floor `value` onto the `step` grid (no float drift)."""
    v = _to_decimal(value, "value")
    if step <= 0:
        raise PrecisionDataUnavailableError(f"step must be > 0, got {step}")
    return (v // step) * step


def quantize_price(price: float, tick_size: Decimal) -> float:
    """Floor a LIMIT price to the tick grid."""
    return float(floor_to_step(price, tick_size))


def quantize_quantity(quantity: float, step_size: Decimal) -> float:
    """Floor an order quantity to the lot-step grid."""
    return float(floor_to_step(quantity, step_size))


def validate_quantized_quantity(
    quantity: float,
    price: float,
    filters: Dict[str, Optional[Decimal]],
) -> None:
    """Post-quantization checks. Raises PrecisionDataUnavailableError to fail closed.

    - quantity > 0 and quantity >= min_qty (when provided)
    - quantity is exactly on the step grid
    - price > 0 and price is exactly on the tick grid
    - notional quantity*price >= min_notional (when provided)
    """
    if not isinstance(price, (int, float)) or price <= 0:
        raise PrecisionDataUnavailableError(f"price must be a positive number, got {price!r}")
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        raise PrecisionDataUnavailableError(f"quantity must be a positive number, got {quantity!r}")

    q = _to_decimal(quantity, "quantity")
    p = _to_decimal(price, "price")

    tick = filters.get("tick_size")
    step = filters.get("step_size")
    if tick is None or step is None:
        raise PrecisionDataUnavailableError("tick_size/step_size unavailable")
    if q % step != 0:
        raise PrecisionDataUnavailableError(f"quantity {q} not on step grid {step}")
    if p % tick != 0:
        raise PrecisionDataUnavailableError(f"price {p} not on tick grid {tick}")

    min_qty = filters.get("min_qty")
    if min_qty is not None and q < min_qty:
        raise PrecisionDataUnavailableError(f"quantity {q} below minQty {min_qty}")

    min_notional = filters.get("min_notional")
    if min_notional is not None and q * p < min_notional:
        raise PrecisionDataUnavailableError(
            f"notional {q * p} below minNotional {min_notional}"
        )


def price_preserves_geometry(direction: str, price: float, sl: float, tp: float) -> bool:
    """True when the rounded LIMIT price keeps the signal's risk geometry intact.

    LONG: SL < price < TP   SHORT: TP < price < SL
    (same semantics as RiskValidator geometry, applied to the order price).
    """
    if direction == "LONG":
        return sl < price < tp
    if direction == "SHORT":
        return tp < price < sl
    return False


__all__ = [
    "PrecisionDataUnavailableError",
    "get_symbol_filters",
    "floor_to_step",
    "quantize_price",
    "quantize_quantity",
    "validate_quantized_quantity",
    "price_preserves_geometry",
]
