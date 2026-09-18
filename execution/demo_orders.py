# execution/demo_orders.py — Binance Demo Futures order placement (Phase 1).
#
# DEMO ONLY. Only https://demo-fapi.binance.com. No live Binance URL, no
# live credentials, no cancel/modify/batch/leverage/margin/position-mode
# endpoints.
#
# Reuses signing helpers from execution.demo_client (build_query_string,
# sign_query_string, build_signed_query). Does NOT reuse execute_get
# (GET-only). Implements POST /fapi/v1/order.
#
# Phase 1 scope:
#   - LIMIT orders only (no MARKET)
#   - LONG -> BUY, SHORT -> SELL
#   - quantity, price, clientOrderId required
#   - timeInForce=GTC
#   - explicit SignalApproval required (approved=True)
#   - ApprovedSignalRef present and risk-validated by existing RiskValidator
#   - SL/TP are metadata only; NOT submitted as separate orders in this phase
#   - unknown execution state (timeout/network ambiguity) -> ERROR result,
#     never auto-retry
#   - duplicate clientOrderId rejected by local guard
#
# All network calls are POST only. No GET trading endpoints.

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from execution.config import DEMO_BASE_URL, ExecutionConfig
from execution.demo_client import (
    build_query_string,
    build_signed_query,
    _check_demo_url,
    _classify_http_status,
    _parse_binance_error_body,
    maybe_code,
)
from execution.ledger import ExecutionLedger
from execution.models import (
    ApprovedSignalRef,
    ExecutionResult,
    OrderRequest,
    SignalApproval,
)
from execution.risk import RiskValidationError, RiskValidator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_ORDER_BASE = "https://demo-fapi.binance.com"
ORDER_PATH = "/fapi/v1/order"
DEFAULT_ORDER_TIMEOUT: Tuple[float, float] = (10.0, 30.0)
DEFAULT_RECV_WINDOW = 5000

# Supported order types in this phase.
ALLOWED_ORDER_TYPES = frozenset(["LIMIT"])

# Binance side mapping.
_SIDE_MAP: Dict[str, str] = {
    "LONG": "BUY",
    "SHORT": "SELL",
}

# Time in force for LIMIT orders in this phase.
_DEFAULT_TIF = "GTC"

# ---------------------------------------------------------------------------
# Public helpers ( reusable, testable )
# ---------------------------------------------------------------------------


def direction_to_side(direction: str) -> str:
    """Map LONG/SHORT to BUY/SELL.

    Raises ValueError for anything else.
    """
    try:
        return _SIDE_MAP[direction]
    except KeyError:
        raise ValueError(
            f"Unsupported direction {direction!r}; expected LONG or SHORT"
        )


def build_order_params(
    signal: ApprovedSignalRef,
    client_order_id: str,
    quantity: float,
    price: float,
) -> Dict[str, Any]:
    """Build the parameter dict for POST /fapi/v1/order (LIMIT, GTC).

    Does NOT include SL/TP as separate orders. SL/TP remain in metadata.

    Parameters:
        signal: ApprovedSignalRef (approved signal metadata).
        client_order_id: unique client-side order id.
        quantity: order quantity (positive).
        price: LIMIT price (positive).

    Returns:
        Dict suitable for signing and sending to Binance.
    """
    if not client_order_id:
        raise ValueError("client_order_id must be non-empty")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if price <= 0:
        raise ValueError("price must be positive")

    side = direction_to_side(signal.direction)

    return {
        "symbol": signal.symbol,
        "side": side,
        "type": "LIMIT",
        "timeInForce": _DEFAULT_TIF,
        "quantity": str(quantity),
        "price": str(price),
        "newClientOrderId": client_order_id,
    }


def build_order_query_string(
    params: Dict[str, Any],
    api_secret: str,
) -> str:
    """Build a signed query string for a POST /fapi/v1/order request.

    Reuses signing helpers from execution.demo_client.

    Parameters are NOT modified in place.
    """
    signed_query, _raw = build_signed_query(params, api_secret)
    return signed_query


def build_order_url(url: str) -> str:
    """Build/validate the full Demo order URL and enforce Demo-only hostname.

    If ``url`` already contains a full URL (with scheme + hostname), it is
    validated via _check_demo_url and returned unchanged. If ``url`` is only
    a query string (no '://'), the Demo base URL is prepended.

    Raises ValueError for any non-Demo hostname.
    """
    if "://" not in url:
        # query string only — prepend Demo base
        base = _check_demo_url(DEMO_ORDER_BASE)
        return f"{base}{ORDER_PATH}?{url}"
    # full URL — validate hostname
    return _check_demo_url(url)


def build_order_request(
    url: str,
    api_key: str,
) -> Request:
    """Build a POST Request for the Demo order endpoint.

    Secrets never appear in the URL or in the request body. API key goes
    only in X-MBX-APIKEY header.
    """
    if not api_key:
        raise ValueError("API key required for signed order placement")
    headers: Dict[str, str] = {
        "User-Agent": "SignalForge-Demo/1.0",
        "Accept": "application/json",
        "X-MBX-APIKEY": api_key,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    return Request(url, method="POST", headers=headers, data=b"")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class DemoOrderError(Exception):
    """Base exception for demo order placement failures.

    Never contains API secret, signed URL, or auth headers.
    """


class ApprovalRequiredError(DemoOrderError):
    """Raised when SignalApproval is missing or not approved."""


class RiskCheckFailedError(DemoOrderError):
    """Raised when RiskValidator rejects the signal/order."""


class DuplicateOrderIdError(DemoOrderError):
    """Raised when client_order_id was already used locally."""


class UnknownExecutionStateError(DemoOrderError):
    """Raised when the HTTP request may have reached Binance but the response
    status is unknown (timeout/network failure). The caller MUST NOT retry
    automatically; they should query order status via openOrders instead.
    """

    def __init__(
        self,
        message: str,
        *,
        client_order_id: str = "",
        signal_id: str = "",
    ):
        super().__init__(message)
        self.client_order_id = client_order_id
        self.signal_id = signal_id

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.client_order_id:
            parts.append(f"client_order_id={self.client_order_id}")
        if self.signal_id:
            parts.append(f"signal_id={self.signal_id}")
        return " | ".join(parts)


def place_limit_order(
    config: ExecutionConfig,
    signal: ApprovedSignalRef,
    approval: Optional[SignalApproval],
    client_order_id: str,
    quantity: float,
    price: float,
    ledger: Optional[ExecutionLedger] = None,
    timeout: Optional[Tuple[float, float]] = None,
) -> ExecutionResult:
    """Place a LIMIT order on the Binance Demo Futures API.

    Pre-conditions (all enforced, fail-closed):
      - config.mode == "demo"
      - approval is not None AND approval.is_approved() is True
      - signal is an ApprovedSignalRef with valid geometry/score/RR
      - symbol in allowlist
      - quantity within risk limits
      - client_order_id non-empty and not previously used (local guard)

    SL/TP are NOT submitted to Binance in this phase; they remain in
    SignalForge metadata.

    Args:
        config: ExecutionConfig from env.
        signal: ApprovedSignalRef for the signal being executed.
        approval: SignalApproval with approved=True; None is rejected.
        client_order_id: unique client-side order id.
        quantity: order quantity (units of base asset).
        price: LIMIT price.
        ledger: optional ExecutionLedger for audit trail.
        timeout: optional (connect, read) timeout tuple.

    Returns:
        ExecutionResult with status ACCEPTED/REJECTED/ERROR/UNKNOWN.

    Raises:
        ApprovalRequiredError: if approval is None or not approved.
        RiskCheckFailedError: if RiskValidator rejects.
        DuplicateOrderIdError: if client_order_id already used locally.
        UnknownExecutionStateError: on timeout/network ambiguity.
        DemoOrderError: on other placement failures.
    """
    if timeout is None:
        timeout = DEFAULT_ORDER_TIMEOUT

    now_ms = int(time.time() * 1000)

    # --- Approval gate ---
    if approval is None:
        raise ApprovalRequiredError(
            "SignalApproval is required; none provided"
        )
    if not approval.is_approved():
        raise ApprovalRequiredError(
            "SignalApproval.approved must be True; default is False"
        )

    # --- Risk validation ---
    validator = RiskValidator(config)
    try:
        validator.validate_signal(signal)
    except RiskValidationError as exc:
        raise RiskCheckFailedError(
            f"Signal risk check failed: [{exc.code}] {exc.reason}"
        ) from exc

    if not client_order_id:
        raise DemoOrderError("client_order_id must be non-empty")

    # Build OrderRequest for order-level risk validation.
    order_req = OrderRequest(
        client_order_id=client_order_id,
        symbol=signal.symbol,
        side=direction_to_side(signal.direction),
        order_type="LIMIT",
        quantity=quantity,
        price=price,
        stop_price=None,
        time_in_force=_DEFAULT_TIF,
        reduce_only=False,
        close_position=False,
        signal_id=signal.signal_id,
        strategy_version=signal.strategy_version,
        requested_at_ms=now_ms,
    )

    try:
        validator.validate_order(order_req)
    except RiskValidationError as exc:
        raise RiskCheckFailedError(
            f"Order risk check failed: [{exc.code}] {exc.reason}"
        ) from exc

    # --- Duplicate client order ID guard (local, in-memory) ---
    if _is_duplicate_local(client_order_id, signal.signal_id):
        raise DuplicateOrderIdError(
            f"client_order_id {client_order_id!r} already used for signal "
            f"{signal.signal_id!r}"
        )
    _record_local_order_id(client_order_id, signal.signal_id)

    # --- Build signed request ---
    params = build_order_params(signal, client_order_id, quantity, price)
    signed_query = build_order_query_string(params, config.api_secret)
    url = build_order_url(signed_query)
    req = build_order_request(url, config.api_key)

    # --- Execute POST ---
    try:
        with urlopen(req, timeout=timeout[1]) as resp:
            status_code = resp.status
            body_bytes = resp.read()
            body = body_bytes.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as http_err:
        # HTTP error with body from server.
        err_status = http_err.code
        try:
            err_body = http_err.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        if err_body:
            try:
                binance_err = _parse_binance_error_body(
                    err_body, url, err_status
                )
                return _error_result(
                    client_order_id=client_order_id,
                    signal=signal,
                    order_req=order_req,
                    status="REJECTED",
                    message=(
                        f"Binance API error{maybe_code(binance_err.code)}: "
                        f"{binance_err}"
                    ),
                    error_code=binance_err.code,
                    executed_at_ms=now_ms,
                    ledger=ledger,
                )
            except Exception:
                # Non-Binance error body; treat as HTTP error.
                return _error_result(
                    client_order_id=client_order_id,
                    signal=signal,
                    order_req=order_req,
                    status="REJECTED",
                    message=(
                        f"HTTP {_classify_http_status(err_status)}: "
                        f"{err_body[:200]!r}"
                    ),
                    error_code=None,
                    executed_at_ms=now_ms,
                    ledger=ledger,
                )
        else:
            return _error_result(
                client_order_id=client_order_id,
                signal=signal,
                order_req=order_req,
                status="REJECTED",
                message=f"HTTP {_classify_http_status(err_status)}",
                error_code=None,
                executed_at_ms=now_ms,
                ledger=ledger,
            )
    except (urllib.error.URLError, TimeoutError) as network_err:
        # Timeout or network failure — state UNKNOWN, MUST NOT retry.
        raise UnknownExecutionStateError(
            "Request may have reached Binance but response status is unknown "
            f"({type(network_err).__name__}); do not retry automatically",
            client_order_id=client_order_id,
            signal_id=signal.signal_id,
        ) from network_err

    # --- Parse response ---
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError) as parse_err:
        return _error_result(
            client_order_id=client_order_id,
            signal=signal,
            order_req=order_req,
            status="ERROR",
            message=f"Response parse error: {parse_err}",
            error_code=None,
            executed_at_ms=now_ms,
            ledger=ledger,
        )

    # Binance order response can be a dict (new order) or list (batch, not used here).
    if not isinstance(data, dict):
        return _error_result(
            client_order_id=client_order_id,
            signal=signal,
            order_req=order_req,
            status="ERROR",
            message=f"Unexpected response type: {type(data).__name__}",
            error_code=None,
            executed_at_ms=now_ms,
            ledger=ledger,
        )

    # Check for Binance error code in successful-looking response
    error_code = data.get("code")
    error_msg = data.get("msg", "")
    if error_code is not None or error_msg:
        return _error_result(
            client_order_id=client_order_id,
            signal=signal,
            order_req=order_req,
            status="REJECTED",
            message=(
                f"Binance API error{maybe_code(error_code)}: "
                f"{error_msg or body[:200]!r}"
            ),
            error_code=error_code,
            executed_at_ms=now_ms,
            ledger=ledger,
        )

    # Extract order fields.
    order_id = data.get("orderId")
    symbol_resp = data.get("symbol", signal.symbol)
    side_resp = data.get("side", direction_to_side(signal.direction))
    order_type_resp = data.get("type", "LIMIT")
    status_resp = data.get("status", "UNKNOWN")
    filled_qty = data.get("executedQty")
    fill_price = data.get("price")
    fee = data.get("fee")

    # Map Binance order status to our ExecutionResult status.
    mapped_status = _map_binance_order_status(status_resp)

    result = ExecutionResult(
        client_order_id=client_order_id,
        symbol=symbol_resp,
        side=side_resp,
        order_type=order_type_resp,
        quantity=quantity,
        signal_id=signal.signal_id,
        status=mapped_status,
        message=(
            f"Demo LIMIT order {status_resp}; "
            f"orderId={order_id!r}; "
            f"price={fill_price!r}"
        ),
        binance_order_id=(int(order_id) if order_id is not None else None),
        filled_qty=(float(filled_qty) if filled_qty is not None else None),
        fill_price=(float(fill_price) if fill_price is not None else None),
        fee=(float(fee) if fee is not None else None),
        error_code=None,
        executed_at_ms=now_ms,
    )

    if ledger is not None:
        _log_result(ledger, order_req, result)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _map_binance_order_status(status: str) -> str:
    """Map Binance order status string to our ExecutionResult status.

    Binance statuses: NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED,
    EXPIRED, FILLED_BY_TRIGGER, etc.
    """
    mapping = {
        "NEW": "ACCEPTED",
        "PARTIALLY_FILLED": "ACCEPTED",
        "FILLED": "ACCEPTED",
        "CANCELED": "REJECTED",
        "REJECTED": "REJECTED",
        "EXPIRED": "REJECTED",
        "FILLED_BY_TRIGGER": "ACCEPTED",
        "UNKNOWN": "UNKNOWN",
    }
    return mapping.get(status, "ERROR")


def _error_result(
    *,
    client_order_id: str,
    signal: ApprovedSignalRef,
    order_req: OrderRequest,
    status: str,
    message: str,
    error_code: Optional[int],
    executed_at_ms: int,
    ledger: Optional[ExecutionLedger],
) -> ExecutionResult:
    """Build an ExecutionResult for a failed order attempt."""
    result = ExecutionResult(
        client_order_id=client_order_id,
        symbol=signal.symbol,
        side=direction_to_side(signal.direction),
        order_type="LIMIT",
        quantity=order_req.quantity,
        signal_id=signal.signal_id,
        status=status,
        message=message,
        binance_order_id=None,
        filled_qty=None,
        fill_price=None,
        fee=None,
        error_code=error_code,
        executed_at_ms=executed_at_ms,
    )
    if ledger is not None:
        _log_result(ledger, order_req, result)
    return result


def _log_result(
    ledger: ExecutionLedger,
    order_req: OrderRequest,
    result: ExecutionResult,
) -> None:
    """Log order_request_prepared and order_result to the ledger."""
    try:
        ledger.log_order_request_prepared(order_req)
    except Exception:
        pass  # never let ledger failure block the result
    try:
        ledger.log_order_result(result)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Local duplicate client order ID guard (in-memory, per process)
# ---------------------------------------------------------------------------

_LOCAL_ORDER_ID_REGISTRY: Dict[str, str] = {}  # client_order_id -> signal_id


def _is_duplicate_local(client_order_id: str, signal_id: str) -> bool:
    """Check whether client_order_id was already used in this process."""
    return client_order_id in _LOCAL_ORDER_ID_REGISTRY


def _record_local_order_id(client_order_id: str, signal_id: str) -> None:
    """Record a client_order_id as used for duplicate protection."""
    _LOCAL_ORDER_ID_REGISTRY[client_order_id] = signal_id


def reset_local_order_registry() -> None:
    """Clear the in-memory duplicate guard (for testing only)."""
    _LOCAL_ORDER_ID_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "DEMO_ORDER_BASE",
    "ORDER_PATH",
    "DEFAULT_ORDER_TIMEOUT",
    "ALLOWED_ORDER_TYPES",
    "place_limit_order",
    "direction_to_side",
    "build_order_params",
    "build_order_query_string",
    "build_order_url",
    "build_order_request",
    "DemoOrderError",
    "ApprovalRequiredError",
    "RiskCheckFailedError",
    "DuplicateOrderIdError",
    "UnknownExecutionStateError",
    "reset_local_order_registry",
]