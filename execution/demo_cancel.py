# execution/demo_cancel.py — Binance Demo Futures order cancellation (Phase 1).
#
# DEMO ONLY. Only https://demo-fapi.binance.com. No live Binance URL, no
# live credentials, no batch/leverage/margin/position-mode endpoints.
#
# Reuses signing helpers from execution.demo_client (build_signed_query).
# Implements DELETE /fapi/v1/order.
#
# Phase 1 scope:
#   - Manual cancellation only — no automatic cancellation
#   - Requires symbol and at least one of order_id / client_order_id
#   - Demo URL only — no live Binance support
#   - No retry on ambiguous network failure (CancelUnknownStateError)
#   - No scanner/PaperObserver integration
#   - Credentials only from environment
#   - Never prints credentials, signed URLs, or secrets
#
# All network calls are DELETE only. No GET/POST trading endpoints.

from __future__ import annotations
import json
import time
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.request import Request, urlopen

from execution.config import ExecutionConfig, DEMO_BASE_URL
from execution.demo_client import (
    build_signed_query,
    _check_demo_url,
    _classify_http_status,
    _parse_binance_error_body,
    maybe_code,
)
from execution.ledger import ExecutionLedger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_CANCEL_BASE = "https://demo-fapi.binance.com"
CANCEL_PATH = "/fapi/v1/order"
DEFAULT_CANCEL_TIMEOUT: Tuple[float, float] = (10.0, 30.0)

# ---------------------------------------------------------------------------
# Public helpers (reusable, testable)
# ---------------------------------------------------------------------------


def build_cancel_params(
    symbol: str,
    order_id: Optional[int] = None,
    client_order_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the parameter dict for DELETE /fapi/v1/order.

    Requires symbol. At least one of order_id / client_order_id should be
    provided by the caller (validated in cancel_order).
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("symbol must be a non-empty string")
    params: Dict[str, Any] = {"symbol": symbol}
    if order_id is not None:
        params["orderId"] = str(order_id)
    if client_order_id is not None:
        params["origClientOrderId"] = client_order_id
    return params


def build_cancel_query_string(
    params: Dict[str, Any],
    api_secret: str,
) -> str:
    """Build a signed query string for a DELETE /fapi/v1/order request.

    Reuses signing helpers from execution.demo_client.
    """
    signed_query, _raw = build_signed_query(params, api_secret)
    return signed_query


def build_cancel_url(query_string: str) -> str:
    """Build/validate the full Demo cancel URL and enforce Demo-only hostname.

    If ``query_string`` already contains a full URL (with scheme + hostname),
    it is validated via _check_demo_url and returned unchanged. If it is only
    a query string (no '://'), the Demo base URL is prepended.
    """
    if "://" not in query_string:
        base = _check_demo_url(DEMO_CANCEL_BASE)
        return f"{base}{CANCEL_PATH}?{query_string}"
    return _check_demo_url(query_string)


def build_cancel_request(url: str, api_key: str) -> Request:
    """Build a DELETE Request for the Demo cancel endpoint.

    Secrets never appear in the URL or in the request body. API key goes
    only in X-MBX-APIKEY header.
    """
    if not api_key:
        raise ValueError("API key required for signed cancel request")
    headers: Dict[str, str] = {
        "User-Agent": "SignalForge-Demo/1.0",
        "Accept": "application/json",
        "X-MBX-APIKEY": api_key,
    }
    return Request(url, method="DELETE", headers=headers, data=b"")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class CancelError(Exception):
    """Base exception for demo cancel failures.

    Never contains API secret, signed URL, or auth headers.
    """


class CancelUnknownStateError(CancelError):
    """Raised when the HTTP request may have reached Binance but the response
    status is unknown (timeout/network failure). The caller MUST NOT retry.
    """

    def __init__(
        self,
        message: str,
        *,
        order_id: Optional[int] = None,
        client_order_id: str = "",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.order_id = order_id
        self.client_order_id = client_order_id
        self.original_error = original_error

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.original_error is not None:
            parts.append(f"({type(self.original_error).__name__}: {self.original_error})")
        if self.order_id is not None:
            parts.append(f"order_id={self.order_id}")
        if self.client_order_id:
            parts.append(f"client_order_id={self.client_order_id}")
        return " | ".join(parts)


@dataclass(frozen=True)
class CancelResult:
    """Outcome of a demo cancel attempt.

    Never contains API secrets.
    """

    symbol: str
    side: str
    status: str
    message: str
    order_id: Optional[int]
    client_order_id: str
    executed_at_ms: int
    error_code: Optional[int] = None


def cancel_order(
    config: ExecutionConfig,
    symbol: str,
    order_id: Optional[int] = None,
    client_order_id: Optional[str] = None,
    ledger: Optional[ExecutionLedger] = None,
    timeout: Optional[Tuple[float, float]] = None,
) -> CancelResult:
    """Cancel an order on the Binance Demo Futures API.

    Pre-conditions (all enforced, fail-closed):
      - config.mode == "demo" (enforced at ExecutionConfig construction)
      - symbol non-empty and in config.symbol_allowlist
      - at least one of order_id / client_order_id provided
      - signed request (HMAC-SHA256)
      - Demo URL only

    Args:
        config: ExecutionConfig from env.
        symbol: symbol of the order to cancel (e.g. "BTCUSDT").
        order_id: Binance order ID (optional if client_order_id provided).
        client_order_id: client order ID (optional if order_id provided).
        ledger: optional ExecutionLedger for audit trail.
        timeout: optional (connect, read) timeout tuple.

    Returns:
        CancelResult with cancel status.

    Raises:
        CancelError: on validation failures or HTTP errors with Binance body.
        CancelUnknownStateError: on timeout/network ambiguity (NO retry).
    """
    if timeout is None:
        timeout = DEFAULT_CANCEL_TIMEOUT

    now_ms = int(time.time() * 1000)

    # --- Symbol validation ---
    if not symbol or not isinstance(symbol, str):
        raise CancelError("symbol must be a non-empty string")
    if not config.is_symbol_allowed(symbol):
        raise CancelError(
            f"Symbol {symbol!r} not in allowlist {config.symbol_allowlist}"
        )

    # --- At least one ID required ---
    if order_id is None and client_order_id is None:
        raise CancelError(
            "At least one of order_id or client_order_id must be provided"
        )

    # --- Build signed request ---
    params = build_cancel_params(symbol, order_id, client_order_id)
    signed_query = build_cancel_query_string(params, config.api_secret)
    url = build_cancel_url(signed_query)
    req = build_cancel_request(url, config.api_key)

    # --- Execute DELETE ---
    try:
        with urlopen(req, timeout=timeout[1]) as resp:
            status_code = resp.status
            body_bytes = resp.read()
            body = body_bytes.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as http_err:
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
                return _cancel_result(
                    symbol=symbol,
                    side="",
                    status="REJECTED",
                    message=(
                        f"Binance API error{maybe_code(binance_err.code)}: "
                        f"{binance_err}"
                    ),
                    error_code=binance_err.code,
                    order_id=order_id,
                    client_order_id=client_order_id or "",
                    executed_at_ms=now_ms,
                )
            except Exception:
                return _cancel_result(
                    symbol=symbol,
                    side="",
                    status="REJECTED",
                    message=(
                        f"HTTP {_classify_http_status(err_status)}: "
                        f"{err_body[:200]!r}"
                    ),
                    error_code=None,
                    order_id=order_id,
                    client_order_id=client_order_id or "",
                    executed_at_ms=now_ms,
                )
        else:
            return _cancel_result(
                symbol=symbol,
                side="",
                status="REJECTED",
                message=f"HTTP {_classify_http_status(err_status)}",
                error_code=None,
                order_id=order_id,
                client_order_id=client_order_id or "",
                executed_at_ms=now_ms,
            )
    except (urllib.error.URLError, TimeoutError) as network_err:
        raise CancelUnknownStateError(
            "Request may have reached Binance but response status is unknown "
            f"({type(network_err).__name__}); do not retry automatically",
            order_id=order_id,
            client_order_id=client_order_id or "",
            original_error=network_err,
        ) from network_err

    # --- Parse response ---
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError) as parse_err:
        return _cancel_result(
            symbol=symbol,
            side="",
            status="ERROR",
            message=f"Response parse error: {parse_err}",
            error_code=None,
            order_id=order_id,
            client_order_id=client_order_id or "",
            executed_at_ms=now_ms,
        )

    if not isinstance(data, dict):
        return _cancel_result(
            symbol=symbol,
            side="",
            status="ERROR",
            message=f"Unexpected response type: {type(data).__name__}",
            error_code=None,
            order_id=order_id,
            client_order_id=client_order_id or "",
            executed_at_ms=now_ms,
        )

    # Check for Binance error code in response
    error_code = data.get("code")
    error_msg = data.get("msg", "")
    if error_code is not None or error_msg:
        return _cancel_result(
            symbol=symbol,
            side="",
            status="REJECTED",
            message=(
                f"Binance API error{maybe_code(error_code)}: "
                f"{error_msg or body[:200]!r}"
            ),
            error_code=error_code,
            order_id=order_id,
            client_order_id=client_order_id or "",
            executed_at_ms=now_ms,
        )

    # Extract order fields from successful cancel response
    order_id_resp = data.get("orderId")
    symbol_resp = data.get("symbol", symbol)
    side_resp = data.get("side", "")
    status_resp = data.get("status", "UNKNOWN")
    client_order_id_resp = data.get("clientOrderId", client_order_id or "")

    result = CancelResult(
        symbol=symbol_resp,
        side=side_resp,
        status=status_resp,
        message=f"Demo order cancellation {status_resp}",
        order_id=(int(order_id_resp) if order_id_resp is not None else None),
        client_order_id=client_order_id_resp,
        executed_at_ms=now_ms,
        error_code=None,
    )

    if ledger is not None:
        ledger.log_cancel_result(result)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cancel_result(
    *,
    symbol: str,
    side: str,
    status: str,
    message: str,
    error_code: Optional[int],
    order_id: Optional[int],
    client_order_id: str,
    executed_at_ms: int,
) -> CancelResult:
    """Build a CancelResult for a cancel attempt (success or failure)."""
    return CancelResult(
        symbol=symbol,
        side=side,
        status=status,
        message=message,
        order_id=order_id,
        client_order_id=client_order_id,
        executed_at_ms=executed_at_ms,
        error_code=error_code,
    )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "DEMO_CANCEL_BASE",
    "CANCEL_PATH",
    "DEFAULT_CANCEL_TIMEOUT",
    "cancel_order",
    "build_cancel_params",
    "build_cancel_query_string",
    "build_cancel_url",
    "build_cancel_request",
    "CancelResult",
    "CancelError",
    "CancelUnknownStateError",
]
