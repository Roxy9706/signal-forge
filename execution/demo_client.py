# Binance Demo REST Client — read-only, demo-only foundation.

from __future__ import annotations
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from execution.config import ExecutionConfig, DEMO_BASE_URL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_BASE = "https://demo-fapi.binance.com"

# API paths
PATH_PING = "/fapi/v1/ping"
PATH_TIME = "/fapi/v1/time"
PATH_EXCHANGE_INFO = "/fapi/v1/exchangeInfo"
PATH_BALANCE = "/fapi/v2/balance"
PATH_ACCOUNT = "/fapi/v2/account"
PATH_OPEN_ORDERS = "/fapi/v1/openOrders"

# Default timeouts (connect, read) in seconds
DEFAULT_TIMEOUT = (10.0, 30.0)

# Default recvWindow (ms) for signed requests
DEFAULT_RECV_WINDOW = 5000

# Explicit allowlist of read-only endpoint paths.
# Arbitrary GET paths are rejected in execute_get.
ALLOWED_PATHS = frozenset([
    PATH_PING,
    PATH_TIME,
    PATH_EXCHANGE_INFO,
    PATH_BALANCE,
    PATH_ACCOUNT,
    PATH_OPEN_ORDERS,
])


# ---------------------------------------------------------------------------
# Signing helpers (deterministic, testable, no network)
# ---------------------------------------------------------------------------

def build_query_string(params: Dict[str, Any]) -> str:
    """Build a sorted query string from a dict of parameters.

    Boolean values are normalized to lowercase 'true'/'false' for Binance API
    compatibility (Python's urlencode produces 'True'/'False' by default).

    Args:
        params: dictionary of query parameters (all values must be stringifiable).

    Returns:
        Sorted query string (no leading '?').
    """
    if not params:
        return ""
    # Normalize booleans to lowercase strings before encoding
    normalized = {
        k: ("true" if v is True else "false" if v is False else v)
        for k, v in params.items()
    }
    sorted_items = sorted(normalized.items(), key=lambda kv: kv[0])
    return urllib.parse.urlencode(sorted_items, doseq=True)


def sign_query_string(query_string: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature for a query string.

    Args:
        query_string: the query string to sign (without the 'signature' param itself).
        secret: the API secret (kept out of logs/results).

    Returns:
        Hex-encoded HMAC-SHA256 signature.

    Raises:
        ValueError: if secret is empty or not a string.
    """
    if not isinstance(secret, str) or not secret:
        raise ValueError(
            "API secret must be a non-empty string. "
            "Missing or empty secret: fail-closed."
        )
    return hmac.new(
        secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def build_signed_query(
    params: Dict[str, Any],
    secret: str,
) -> Tuple[str, str]:
    """Build a signed query string and the raw query string before signing.

    Adds timestamp and recvWindow if not present, builds the query string,
    computes the signature, and appends signature to the query string.

    Args:
        params: query parameters (must include at least 'timestamp' for signed requests).
        secret: API secret.

    Returns:
        Tuple of (full_query_string_with_signature, raw_query_string_before_signature).
    """
    params = dict(params)  # don't mutate caller's dict
    if "timestamp" not in params:
        params["timestamp"] = int(time.time() * 1000)
    if "recvWindow" not in params:
        params["recvWindow"] = DEFAULT_RECV_WINDOW

    raw_query = build_query_string(params)
    signature = sign_query_string(raw_query, secret)
    full_query = raw_query + ("&" if raw_query else "") + f"signature={signature}"
    return full_query, raw_query


# ---------------------------------------------------------------------------
# URL construction (demo-only enforcement)
# ---------------------------------------------------------------------------

def _check_demo_url(url: str) -> str:
    """Validate that the URL targets the Demo endpoint.

    Args:
        url: full URL to validate (may include path and query string).

    Returns:
        The URL unchanged if valid.

    Raises:
        ValueError: if the URL does not target the Demo endpoint.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "demo-fapi.binance.com":
        raise ValueError(
            f"Demo client only permits https://demo-fapi.binance.com. "
            f"Got {url!r}. No live fallback permitted."
        )
    return url


def build_demo_url(path: str, query_string: str = "") -> str:
    """Build a full Demo Binance API URL.

    Args:
        path: API path, e.g. '/fapi/v1/ping'.
        query_string: optional query string (with signature if signed).

    Returns:
        Full URL string.
    """
    base = _check_demo_url(DEMO_BASE)
    if query_string:
        return f"{base}{path}?{query_string}"
    return f"{base}{path}"


# ---------------------------------------------------------------------------
# Path allowlist (explicit read-only endpoints only)
# ---------------------------------------------------------------------------

ALLOWED_PATHS = frozenset([
    PATH_PING,
    PATH_TIME,
    PATH_EXCHANGE_INFO,
    PATH_BALANCE,
    PATH_ACCOUNT,
    PATH_OPEN_ORDERS,
])

# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def build_request(
    method: str,
    url: str,
    api_key: str,
    signed: bool = False,
) -> Request:
    """Build a urllib Request with appropriate headers.

    Args:
        method: HTTP method (only GET supported in this phase).
        url: full URL — must be a Demo Binance URL.
        api_key: DEMO_API_KEY (sent in X-MBX-APIKEY header for signed requests).
        signed: whether this is a signed request.

    Returns:
        urllib.request.Request ready for urlopen.

    Raises:
        ValueError: if url is not a Demo Binance URL, or method is not GET.
    """
    if method.upper() != "GET":
        raise ValueError(f"Only GET is supported in this demo phase; got {method!r}")

    # Hard gate: only approved Demo URLs may be requested
    _check_demo_url(url)

    headers: Dict[str, str] = {
        "User-Agent": "SignalForge-Demo/1.0",
        "Accept": "application/json",
    }
    if signed:
        if not api_key:
            raise ValueError("API key required for signed requests")
        headers["X-MBX-APIKEY"] = api_key

    return Request(url, method=method.upper(), headers=headers)


# ---------------------------------------------------------------------------
# HTTP response handling
# ---------------------------------------------------------------------------

class DemoHttpError(Exception):
    """Raised when the Demo HTTP request fails at the transport level."""

    def __init__(
        self,
        message: str,
        url: str = "",
        status_code: Optional[int] = None,
        reason: str = "",
    ):
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.reason = reason

    def __str__(self) -> str:
        parts: List[str] = []
        if self.status_code is not None:
            parts.append(f"DemoHttpError({self.status_code})")
        else:
            parts.append("DemoHttpError")
        if self.reason:
            parts.append(f"({self.reason})")
        if self.url:
            parts.append(f"url={self.url}")
        parts.append(f": {super().__str__()}")
        return " ".join(parts)


class BinanceApiError(Exception):
    """Raised when Binance returns an API error response.

    Contains the error code and message from the Binance response, but NEVER
    the API secret or key.
    """

    def __init__(
        self,
        message: str,
        code: Optional[int] = None,
        url: str = "",
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.code = code
        self.url = url
        self.status_code = status_code

    def __str__(self) -> str:
        parts: List[str] = ["BinanceApiError"]
        if self.code is not None:
            parts.append(f"code={self.code}")
        if self.status_code is not None:
            parts.append(f"http={self.status_code}")
        if self.url:
            parts.append(f"url={self.url}")
        parts.append(f": {super().__str__()}")
        return " ".join(parts)


class BinanceResponseParseError(Exception):
    """Raised when the response body cannot be parsed as JSON."""

    def __init__(self, message: str, url: str = "", status_code: Optional[int] = None):
        super().__init__(message)
        self.url = url
        self.status_code = status_code

    def __str__(self) -> str:
        parts: List[str] = ["BinanceResponseParseError"]
        if self.status_code is not None:
            parts.append(f"http={self.status_code}")
        if self.url:
            parts.append(f"url={self.url}")
        parts.append(f": {super().__str__()}")
        return " ".join(parts)


def _classify_http_status(code: int) -> str:
    """Return a human-readable classification for an HTTP status code."""
    if code == 429:
        return "429 (rate limit)"
    if code == 418:
        return "418 (anti-bot / proxy auth)"
    if 500 <= code < 600:
        return f"5xx server error ({code})"
    if 400 <= code < 500:
        return f"4xx client error ({code})"
    if code < 200 or code >= 600:
        return f"unexpected status ({code})"
    return f"success ({code})"


def _parse_binance_error_body(body: str, url: str, status_code: int) -> BinanceApiError:
    """Try to parse a Binance error JSON body into a BinanceApiError."""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        raise BinanceResponseParseError(
            f"Non-JSON response from Binance ({status_code}): {body[:200]!r}",
            url=url,
            status_code=status_code,
        )
    error_code = data.get("code")
    error_msg = data.get("msg", "")
    if error_code is not None or error_msg:
        full_msg = f"Binance API error{maybe_code(error_code)}: {error_msg or body[:200]!r}"
        return BinanceApiError(
            message=full_msg,
            code=error_code,
            url=url,
            status_code=status_code,
        )
    # JSON but no error fields — treat as parse error
    raise BinanceResponseParseError(
        f"Unrecognized Binance response ({status_code}): {body[:200]!r}",
        url=url,
        status_code=status_code,
    )


def maybe_code(code: Optional[int]) -> str:
    """Format an optional error code for display."""
    if code is None:
        return ""
    return f" (code={code})"


# ---------------------------------------------------------------------------
# HTTP execution
# ---------------------------------------------------------------------------

def execute_get(
    config: ExecutionConfig,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    signed: bool = False,
    timeout: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Execute a GET request against the Binance Demo API.

    This is the main entry point. It handles:
    - Path whitelist enforcement (explicit read-only endpoints only)
    - URL construction (Demo-only enforcement)
    - Query string building (including signing for signed requests)
    - HTTP request execution with timeout
    - HTTP status classification (429, 418, 5xx, 4xx)
    - Binance JSON error parsing
    - Malformed/non-JSON response handling
    - Network failure handling

    Args:
        config: ExecutionConfig (provides api_key, api_secret, base_url).
        path: API path, e.g. '/fapi/v1/ping'. Must be in ALLOWED_PATHS.
        params: optional query parameters.
        signed: whether the request requires HMAC-SHA256 signing.
        timeout: optional (connect_timeout, read_timeout) tuple in seconds.

    Returns:
        Parsed JSON response as a dict (or list if the endpoint returns a list).

    Raises:
        ValueError: if path is not an allowed read-only endpoint.
        DemoHttpError: transport-level failure (timeout, network, HTTP error).
        BinanceApiError: Binance returned an error in the response body.
        BinanceResponseParseError: response body is not valid JSON.
    """
    # Enforce explicit path allowlist before any other processing
    if path not in ALLOWED_PATHS:
        raise ValueError(
            f"Path {path!r} is not in the explicit read-only endpoint allowlist. "
            f"Permitted paths: {sorted(ALLOWED_PATHS)}. "
            f"No arbitrary GET paths permitted."
        )

    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    connect_timeout, read_timeout = timeout

    # Build params dict
    query_params: Dict[str, Any] = {}
    if params:
        query_params.update(params)

    # For signed requests, add timestamp and recvWindow if not present
    if signed:
        if "timestamp" not in query_params:
            query_params["timestamp"] = int(time.time() * 1000)
        if "recvWindow" not in query_params:
            query_params["recvWindow"] = DEFAULT_RECV_WINDOW

    # Build URL
    if signed:
        query_string_with_sig, _raw = build_signed_query(query_params, config.api_secret)
        url = build_demo_url(path, query_string_with_sig)
    elif query_params:
        query_string = build_query_string(query_params)
        url = build_demo_url(path, query_string)
    else:
        url = build_demo_url(path)

    # Build request
    req = build_request(
        method="GET",
        url=url,
        api_key=config.api_key,
        signed=signed,
    )

    # Execute
    try:
        with urlopen(req, timeout=read_timeout) as resp:
            status_code = resp.status
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        # HTTPError has a .code and .read() for the body
        status_code = e.code
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        # Classify the error
        classification = _classify_http_status(status_code)
        # Try to parse as Binance error
        if body:
            try:
                raise _parse_binance_error_body(body, url, status_code)
            except BinanceResponseParseError:
                # Not a Binance error format — raise transport error with body
                raise DemoHttpError(
                    f"HTTP {classification}: {body[:200]!r}",
                    url=url,
                    status_code=status_code,
                    reason=classification,
                ) from None
        else:
            raise DemoHttpError(
                f"HTTP {classification}",
                url=url,
                status_code=status_code,
                reason=classification,
            )
    except urllib.error.URLError as e:
        # Network-level error (timeout, DNS, connection refused, etc.)
        raise DemoHttpError(
            f"Network error: {str(e)!r}",
            url=url,
        ) from e
    except TimeoutError as e:
        raise DemoHttpError(
            f"Timeout after {read_timeout}s reading {url!r}",
            url=url,
            reason="timeout",
        ) from e

    # Parse JSON
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError) as e:
        raise BinanceResponseParseError(
            f"Failed to parse JSON response from {url!r}: {str(e)}",
            url=url,
            status_code=status_code,
        ) from e

    # Success — return parsed data (could be dict or list)
    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Endpoint wrappers (read-only only)
# ---------------------------------------------------------------------------

def ping(config: ExecutionConfig, timeout: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """GET /fapi/v1/ping — verify Demo server connectivity.

    Unsigned. Returns the server timestamp.
    """
    return execute_get(config, PATH_PING, timeout=timeout)


def get_server_time(config: ExecutionConfig, timeout: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """GET /fapi/v1/time — get server time.

    Unsigned. Returns {"serverTime": 1234567890000}.
    """
    return execute_get(config, PATH_TIME, timeout=timeout)


def get_exchange_info(config: ExecutionConfig, timeout: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """GET /fapi/v1/exchangeInfo — get exchange trading rules and symbol info.

    Unsigned. Returns full exchangeInfo object.
    """
    return execute_get(config, PATH_EXCHANGE_INFO, timeout=timeout)


def get_balance(config: ExecutionConfig, timeout: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
    """GET /fapi/v2/balance — get account balances.

    Signed. Requires api_key + api_secret.
    Returns a list of balance objects: [{"asset": "USDT", "free": "1000.00", "locked": "0.00"}, ...].
    """
    data = execute_get(config, PATH_BALANCE, signed=True, timeout=timeout)
    # Binance returns a list for /fapi/v2/balance
    if not isinstance(data, list):
        raise BinanceApiError(
            f"Expected list from /fapi/v2/balance, got {type(data).__name__}",
            url=build_demo_url(PATH_BALANCE),
        )
    return data  # type: ignore[return-value]


def get_account(config: ExecutionConfig, timeout: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """GET /fapi/v2/account — get account information.

    Signed. Requires api_key + api_secret.
    Returns account object with balances, permissions, etc.
    """
    return execute_get(config, PATH_ACCOUNT, signed=True, timeout=timeout)


def get_open_orders(
    config: ExecutionConfig,
    symbol: Optional[str] = None,
    timeout: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """GET /fapi/v1/openOrders — get open orders.

    Signed. Optionally filter by symbol.
    Returns a list of open order objects.
    """
    params: Dict[str, Any] = {}
    if symbol:
        params["symbol"] = symbol
    data = execute_get(config, PATH_OPEN_ORDERS, params=params, signed=True, timeout=timeout)
    if not isinstance(data, list):
        raise BinanceApiError(
            f"Expected list from /fapi/v1/openOrders, got {type(data).__name__}",
            url=build_demo_url(PATH_OPEN_ORDERS),
        )
    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "DEMO_BASE",
    "PATH_PING",
    "PATH_TIME",
    "PATH_EXCHANGE_INFO",
    "PATH_BALANCE",
    "PATH_ACCOUNT",
    "PATH_OPEN_ORDERS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RECV_WINDOW",
    "build_query_string",
    "sign_query_string",
    "build_signed_query",
    "build_demo_url",
    "build_request",
    "DemoHttpError",
    "BinanceApiError",
    "BinanceResponseParseError",
    "execute_get",
    "ping",
    "get_server_time",
    "get_exchange_info",
    "get_balance",
    "get_account",
    "get_open_orders",
]
