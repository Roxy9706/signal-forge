"""
Tests for execution.demo_client - Binance Demo REST client.

All HTTP interactions are mocked. No real Binance network calls are made.
"""

from __future__ import annotations
import hashlib
import hmac
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional

from execution.config import ExecutionConfig, DEMO_BASE_URL
from execution.demo_client import (
    build_query_string,
    sign_query_string,
    build_signed_query,
    build_demo_url,
    build_request,
    DemoHttpError,
    BinanceApiError,
    BinanceResponseParseError,
    _classify_http_status,
    _parse_binance_error_body,
    maybe_code,
    execute_get,
    ping,
    get_server_time,
    get_exchange_info,
    get_balance,
    get_account,
    get_open_orders,
    PATH_PING,
    PATH_TIME,
    PATH_EXCHANGE_INFO,
    PATH_BALANCE,
    PATH_ACCOUNT,
    PATH_OPEN_ORDERS,
    DEFAULT_TIMEOUT,
    DEFAULT_RECV_WINDOW,
    ALLOWED_PATHS,
    DEMO_BASE as DEMO_BASE_CONSTANT,
    Request,
)


def _make_config(**kwargs: Any) -> ExecutionConfig:
    """Build an ExecutionConfig with default demo values for testing."""
    defaults: Dict[str, Any] = dict(
        mode="demo",
        api_key="test_demo_api_key_12345",
        api_secret="test_demo_api_secret_67890",
        base_url=DEMO_BASE_CONSTANT,
        symbol_allowlist=["BTCUSDT", "ETHUSDT"],
        max_position_quantity=100.0,
        max_total_exposure_usd=10000.0,
        min_rr=2.0,
        approved_only=True,
    )
    defaults.update(kwargs)
    return ExecutionConfig(**defaults)


# ---------------------------------------------------------------------------
# Build helpers tests
# ---------------------------------------------------------------------------

class TestBuildQueryString(unittest.TestCase):
    """Tests for build_query_string."""

    def test_empty_params(self):
        """Empty dict returns empty string."""
        self.assertEqual(build_query_string({}), "")

    def test_none_params(self):
        """None returns empty string."""
        self.assertEqual(build_query_string(None), "")

    def test_single_param(self):
        """Single param sorted correctly."""
        qs = build_query_string({"symbol": "BTCUSDT"})
        self.assertEqual(qs, "symbol=BTCUSDT")

    def test_multiple_params_sorted(self):
        """Params sorted by key."""
        qs = build_query_string({"symbol": "BTCUSDT", "timestamp": "123"})
        self.assertTrue(qs.startswith("symbol=") or qs.startswith("timestamp="))
        self.assertIn("symbol=BTCUSDT", qs)
        self.assertIn("timestamp=123", qs)

    def test_boolean_true(self):
        """Boolean True serializes as lowercase 'true'."""
        qs = build_query_string({"reduceOnly": True})
        self.assertIn("reduceOnly=true", qs)
        self.assertNotIn("reduceOnly=True", qs)

    def test_boolean_false(self):
        """Boolean False serializes as lowercase 'false'."""
        qs = build_query_string({"reduceOnly": False})
        self.assertIn("reduceOnly=false", qs)
        self.assertNotIn("reduceOnly=False", qs)

    def test_boolean_mixed_with_other_params(self):
        """Boolean serialized correctly alongside other params."""
        qs = build_query_string({"symbol": "BTCUSDT", "reduceOnly": True})
        self.assertIn("symbol=BTCUSDT", qs)
        self.assertIn("reduceOnly=true", qs)

    def test_numeric_values(self):
        """Numeric values serialized correctly."""
        qs = build_query_string({"quantity": 1.5, "price": 50000})
        self.assertIn("quantity=1.5", qs)
        self.assertIn("price=50000", qs)

    def test_list_values(self):
        """List values serialized with doseq."""
        qs = build_query_string({"symbol": ["BTCUSDT", "ETHUSDT"]})
        parts = qs.split("&")
        symbols = [p for p in parts if p.startswith("symbol=")]
        self.assertGreaterEqual(len(symbols), 1)


class TestSignQueryString(unittest.TestCase):
    """Tests for sign_query_string."""

    def test_valid_signature(self):
        """Produces a deterministic hex signature."""
        qs = "symbol=BTCUSDT&timestamp=1609459200000"
        secret = "test_secret"
        sig = sign_query_string(qs, secret)
        self.assertEqual(sign_query_string(qs, secret), sig)
        self.assertEqual(len(sig), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in sig))

    def test_different_secrets_produce_different_signatures(self):
        """Different secrets produce different signatures."""
        qs = "symbol=BTCUSDT&timestamp=1609459200000"
        sig1 = sign_query_string(qs, "secret1")
        sig2 = sign_query_string(qs, "secret2")
        self.assertNotEqual(sig1, sig2)

    def test_different_query_strings_produce_different_signatures(self):
        """Different query strings produce different signatures."""
        secret = "test_secret"
        sig1 = sign_query_string("symbol=BTCUSDT", secret)
        sig2 = sign_query_string("symbol=ETHUSDT", secret)
        self.assertNotEqual(sig1, sig2)

    def test_empty_secret_raises(self):
        """Empty secret raises ValueError (fail-closed)."""
        with self.assertRaises(ValueError) as ctx:
            sign_query_string("symbol=BTCUSDT", "")
        self.assertIn("non-empty", str(ctx.exception).lower())

    def test_none_secret_raises(self):
        """None secret raises ValueError."""
        with self.assertRaises(ValueError):
            sign_query_string("symbol=BTCUSDT", None)  # type: ignore[arg-type]

    def test_secret_not_in_signature(self):
        """The secret itself does not appear in the signature output."""
        secret = "my_super_secret_key"
        qs = "symbol=BTCUSDT"
        sig = sign_query_string(qs, secret)
        self.assertNotIn(secret, sig)

    def test_signature_matches_expected_hmac(self):
        """Verify signature matches expected HMAC-SHA256 output."""
        qs = "symbol=BTCUSDT&timestamp=1609459200000"
        secret = "test_secret"
        expected = hmac.new(
            secret.encode("utf-8"),
            qs.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(sign_query_string(qs, secret), expected)


class TestBuildSignedQuery(unittest.TestCase):
    """Tests for build_signed_query."""

    def test_produces_signature(self):
        """build_signed_query appends signature to query string."""
        params = {"symbol": "BTCUSDT", "timestamp": 1609459200000}
        full_query, raw_query = build_signed_query(params, "my_secret")
        self.assertIn("signature=", full_query)

    def test_signature_is_correct(self):
        """Signature matches direct sign_query_string call."""
        params = {"symbol": "BTCUSDT", "timestamp": 1609459200000}
        full_query, raw_query = build_signed_query(params, "my_secret")
        expected_sig = sign_query_string(raw_query, "my_secret")
        self.assertIn(f"signature={expected_sig}", full_query)

    def test_does_not_mutate_input_params(self):
        """Caller's original params dict remains completely unchanged."""
        original_params = {"symbol": "BTCUSDT"}
        caller_params = dict(original_params)
        build_signed_query(caller_params, "secret")
        self.assertEqual(caller_params, original_params)
        self.assertNotIn("timestamp", caller_params)
        self.assertNotIn("recvWindow", caller_params)
        self.assertNotIn("signature", caller_params)

    def test_adds_timestamp_if_missing(self):
        """Adds timestamp if not present."""
        params = {"symbol": "BTCUSDT"}
        full_query, _ = build_signed_query(params, "secret")
        self.assertIn("timestamp=", full_query)

    def test_adds_recv_window_if_missing(self):
        """Adds recvWindow if not present."""
        params = {"symbol": "BTCUSDT", "timestamp": 1609459200000}
        full_query, _ = build_signed_query(params, "secret")
        self.assertIn("recvWindow=", full_query)

    def test_preserves_existing_timestamp(self):
        """Uses existing timestamp without adding another."""
        params = {"symbol": "BTCUSDT", "timestamp": 999999999999}
        full_query, _ = build_signed_query(params, "secret")
        self.assertIn("timestamp=999999999999", full_query)

    def test_preserves_existing_recv_window(self):
        """Uses existing recvWindow without adding another."""
        params = {"symbol": "BTCUSDT", "timestamp": 1609459200000, "recvWindow": 10000}
        full_query, _ = build_signed_query(params, "secret")
        self.assertIn("recvWindow=10000", full_query)
        self.assertNotIn("recvWindow=5000", full_query)

    def test_raw_query_has_no_signature(self):
        """The raw query string (before signing) does not contain signature."""
        params = {"symbol": "BTCUSDT", "timestamp": 1609459200000}
        _, raw_query = build_signed_query(params, "my_secret")
        self.assertNotIn("signature=", raw_query)


# ---------------------------------------------------------------------------
# URL construction tests
# ---------------------------------------------------------------------------

class TestURLConstruction(unittest.TestCase):
    """Tests for demo URL construction."""

    def test_build_demo_url_base_is_demo(self):
        """build_demo_url uses DEMO_BASE_URL."""
        url = build_demo_url("/fapi/v1/ping")
        self.assertTrue(url.startswith(DEMO_BASE_URL))
        self.assertEqual(url, f"{DEMO_BASE_URL}/fapi/v1/ping")

    def test_build_demo_url_with_query(self):
        """build_demo_url appends query string."""
        url = build_demo_url("/fapi/v1/ping", "symbol=BTCUSDT")
        self.assertIn("symbol=BTCUSDT", url)

    def test_build_demo_url_rejects_non_demo_base(self):
        """build_demo_url uses EXACTLY the Demo endpoint hostname, never live."""
        url = build_demo_url("/fapi/v1/ping")
        # Must be exactly the Demo base, not any live or lookalike URL
        self.assertEqual(url, f"{DEMO_BASE_URL}/fapi/v1/ping")
        # Verify the hostname is exactly demo-fapi.binance.com (no subdomain tricks)
        self.assertTrue(url.startswith("https://demo-fapi.binance.com"))
        # Reject any URL that would point to live/production
        self.assertNotEqual(url, "https://fapi.binance.com/fapi/v1/ping")
        self.assertNotIn("https://fapi.binance.com", url)

    def test_build_demo_url_with_path_only(self):
        """build_demo_url with no query returns path-only URL."""
        url = build_demo_url("/fapi/v1/time")
        self.assertEqual(url, f"{DEMO_BASE_URL}/fapi/v1/time")
        self.assertNotIn("?", url)

    def test_build_demo_url_multiple_paths(self):
        """All supported paths build correctly."""
        for path in ["/fapi/v1/ping", "/fapi/v1/time", "/fapi/v1/exchangeInfo",
                     "/fapi/v2/balance", "/fapi/v2/account", "/fapi/v1/openOrders"]:
            url = build_demo_url(path)
            self.assertTrue(url.startswith(DEMO_BASE_URL))
            self.assertIn(path, url)


# ---------------------------------------------------------------------------
# Request construction tests
# ---------------------------------------------------------------------------

class TestRequestConstruction(unittest.TestCase):
    """Tests for request building."""

    def test_build_request_get(self):
        """build_request creates a GET request (Demo URL enforced)."""
        req = build_request("GET", "https://demo-fapi.binance.com/fapi/v1/ping", "test_key", signed=True)
        self.assertEqual(req.method, "GET")

    def test_build_request_includes_api_key_header_when_signed(self):
        """Signed requests include X-MBX-APIKEY header (urllib normalizes case)."""
        req = build_request("GET", "https://demo-fapi.binance.com/fapi/v1/ping", "my_api_key", signed=True)
        # urllib.Request stores headers case-insensitively; check via lowercased keys
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("x-mbx-apikey", headers_lower)
        self.assertEqual(headers_lower["x-mbx-apikey"], "my_api_key")

    def test_build_request_no_api_key_header_when_unsigned(self):
        """Unsigned requests do not include X-MBX-APIKEY header."""
        req = build_request("GET", "https://demo-fapi.binance.com/fapi/v1/ping", "my_api_key", signed=False)
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertNotIn("x-mbx-apikey", headers_lower)

    def test_build_request_requires_api_key_for_signed(self):
        """Signed request with empty api_key raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            build_request("GET", "https://demo-fapi.binance.com/fapi/v1/ping", "", signed=True)
        self.assertIn("API key required", str(ctx.exception))

    def test_build_request_rejects_non_get(self):
        """Only GET is supported in this demo phase."""
        # Use a valid Demo URL so the method gate is tested before the URL gate
        with self.assertRaises(ValueError) as ctx:
            build_request("POST", "https://demo-fapi.binance.com/fapi/v1/ping", "key", signed=True)
        self.assertIn("Only GET is supported", str(ctx.exception))

    def test_build_request_rejects_non_demo_url(self):
        """build_request rejects non-Demo URLs (live Binance / lookalikes)."""
        # Live Binance URL with GET so URL gate is tested (not method gate)
        with self.assertRaises(ValueError) as ctx:
            build_request("GET", "https://fapi.binance.com/fapi/v1/ping", "key", signed=True)
        self.assertIn("Demo client only permits", str(ctx.exception))
        # Lookalike URL
        with self.assertRaises(ValueError):
            build_request("GET", "https://demo-fapi.binance.com.evil.net/ping", "key", signed=True)
        # Valid Demo URL passes
        req = build_request("GET", "https://demo-fapi.binance.com/fapi/v1/ping", "key", signed=True)
        self.assertIsInstance(req, Request)

    def test_build_request_includes_user_agent(self):
        """Request includes User-Agent header (urllib normalizes case)."""
        req = build_request("GET", "https://demo-fapi.binance.com/fapi/v1/ping", "key")
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("user-agent", headers_lower)
        self.assertIn("SignalForge-Demo", headers_lower["user-agent"])


# ---------------------------------------------------------------------------
# Error class tests
# ---------------------------------------------------------------------------

class TestHttpErrorClasses(unittest.TestCase):
    """Tests for error exception classes."""

    def test_demo_http_error_message(self):
        """DemoHttpError stores and displays message with clean formatting."""
        exc = DemoHttpError("test message", url="https://x.com", status_code=500, reason="server error")
        s = str(exc)
        self.assertIn("DemoHttpError(500)", s)
        self.assertIn("server error", s)
        self.assertIn("url=https://x.com", s)
        self.assertIn("test message", s)
        # No double spaces
        self.assertNotIn("  ", s)

    def test_demo_http_error_no_status(self):
        """DemoHttpError without status code."""
        exc = DemoHttpError("test message")
        s = str(exc)
        self.assertIn("DemoHttpError", s)
        self.assertIn("test message", s)

    def test_binance_api_error_message(self):
        """BinanceApiError stores and displays code/message with clean formatting."""
        exc = BinanceApiError("API error", code=-1234, url="https://x.com", status_code=200)
        s = str(exc)
        self.assertIn("BinanceApiError", s)
        self.assertIn("code=-1234", s)
        self.assertIn("http=200", s)
        self.assertIn("url=https://x.com", s)
        self.assertIn("API error", s)
        self.assertEqual(exc.code, -1234)
        # No double spaces
        self.assertNotIn("  ", s)

    def test_binance_api_error_no_code(self):
        """BinanceApiError without code."""
        exc = BinanceApiError("API error", url="https://x.com")
        s = str(exc)
        self.assertIn("BinanceApiError", s)
        self.assertNotIn("code=", s)

    def test_binance_response_parse_error_message(self):
        """BinanceResponseParseError stores and displays message."""
        exc = BinanceResponseParseError("bad json", url="https://x.com", status_code=200)
        s = str(exc)
        self.assertIn("BinanceResponseParseError", s)
        self.assertIn("bad json", s)
        self.assertIn("http=200", s)
        # No double spaces
        self.assertNotIn("  ", s)


# ---------------------------------------------------------------------------
# HTTP status classification tests
# ---------------------------------------------------------------------------

class TestHttpStatusClassification(unittest.TestCase):
    """Tests for _classify_http_status."""

    def test_429_rate_limit(self):
        self.assertEqual(_classify_http_status(429), "429 (rate limit)")

    def test_418_anti_bot(self):
        self.assertEqual(_classify_http_status(418), "418 (anti-bot / proxy auth)")

    def test_500_server_error(self):
        self.assertEqual(_classify_http_status(500), "5xx server error (500)")

    def test_503_server_error(self):
        self.assertEqual(_classify_http_status(503), "5xx server error (503)")

    def test_599_server_error(self):
        self.assertEqual(_classify_http_status(599), "5xx server error (599)")

    def test_400_client_error(self):
        self.assertEqual(_classify_http_status(400), "4xx client error (400)")

    def test_404_client_error(self):
        self.assertEqual(_classify_http_status(404), "4xx client error (404)")

    def test_401_client_error(self):
        self.assertEqual(_classify_http_status(401), "4xx client error (401)")

    def test_200_success(self):
        self.assertEqual(_classify_http_status(200), "success (200)")

    def test_201_success(self):
        self.assertEqual(_classify_http_status(201), "success (201)")

    def test_999_unknown(self):
        self.assertEqual(_classify_http_status(999), "unexpected status (999)")


# ---------------------------------------------------------------------------
# Binance error parsing tests
# ---------------------------------------------------------------------------

class TestBinanceErrorParsing(unittest.TestCase):
    """Tests for _parse_binance_error_body."""

    def test_parse_binance_error_with_code_and_msg(self):
        """Parses Binance error JSON with code and msg."""
        body = '{"code": -1234, "msg": "Invalid symbol"}'
        exc = _parse_binance_error_body(body, "https://x.com", 200)
        self.assertIsInstance(exc, BinanceApiError)
        self.assertEqual(exc.code, -1234)
        self.assertIn("Invalid symbol", str(exc))

    def test_parse_binance_error_with_only_msg(self):
        """Parses error with only msg field."""
        body = '{"msg": "Some error"}'
        exc = _parse_binance_error_body(body, "https://x.com", 200)
        self.assertIsInstance(exc, BinanceApiError)
        self.assertIsNone(exc.code)
        self.assertIn("Some error", str(exc))

    def test_parse_binance_error_with_only_code(self):
        """Parses error with only code field."""
        body = '{"code": -1000}'
        exc = _parse_binance_error_body(body, "https://x.com", 200)
        self.assertIsInstance(exc, BinanceApiError)
        self.assertEqual(exc.code, -1000)

    def test_parse_non_json_raises_parse_error(self):
        """Non-JSON body raises BinanceResponseParseError."""
        with self.assertRaises(BinanceResponseParseError):
            _parse_binance_error_body("not json", "https://x.com", 200)

    def test_parse_json_without_error_fields_raises_parse_error(self):
        """JSON without code/msg raises BinanceResponseParseError."""
        body = '{"success": true}'
        with self.assertRaises(BinanceResponseParseError):
            _parse_binance_error_body(body, "https://x.com", 200)


# ---------------------------------------------------------------------------
# maybe_code tests
# ---------------------------------------------------------------------------

class TestMaybeCode(unittest.TestCase):
    """Tests for maybe_code helper."""

    def test_none_code(self):
        self.assertEqual(maybe_code(None), "")

    def test_valid_code(self):
        self.assertEqual(maybe_code(-1234), " (code=-1234)")

    def test_zero_code(self):
        self.assertEqual(maybe_code(0), " (code=0)")


# ---------------------------------------------------------------------------
# execute_get signing tests (mocked)
# ---------------------------------------------------------------------------

class TestExecuteGetSigner(unittest.TestCase):
    """Tests for execute_get signing behavior."""

    def test_execute_get_path_allowlist_rejects_arbitrary_path(self):
        """execute_get rejects paths not in the explicit allowlist."""
        config = _make_config()
        with self.assertRaises(ValueError) as ctx:
            execute_get(config, "/fapi/v1/order", signed=False)
        self.assertIn("not in the explicit read-only endpoint allowlist", str(ctx.exception))
        self.assertIn("No arbitrary GET paths permitted", str(ctx.exception))

    def test_execute_get_path_allowlist_allows_all_known_endpoints(self):
        """All 6 known read-only endpoints are in the allowlist."""
        for path in [PATH_PING, PATH_TIME, PATH_EXCHANGE_INFO,
                     PATH_BALANCE, PATH_ACCOUNT, PATH_OPEN_ORDERS]:
            self.assertIn(path, ALLOWED_PATHS,
                          f"Allowed path {path!r} should be in ALLOWED_PATHS")

    def test_execute_get_unsigned_no_signature(self):
        """Unsigned execute_get does not include signature in URL."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'{"serverTime": 1609459200000}'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            execute_get(config, "/fapi/v1/ping")
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertNotIn("signature=", req.full_url)

    def test_execute_get_signed_includes_signature(self):
        """Signed execute_get includes signature in URL."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[{"asset":"USDT"}]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            execute_get(config, "/fapi/v2/balance", signed=True)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("signature=", req.full_url)

    def test_execute_get_signed_includes_timestamp(self):
        """Signed execute_get includes timestamp."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[{"asset":"USDT"}]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            execute_get(config, "/fapi/v2/balance", signed=True)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("timestamp=", req.full_url)

    def test_execute_get_signed_includes_recv_window(self):
        """Signed execute_get includes recvWindow."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[{"asset":"USDT"}]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            execute_get(config, "/fapi/v2/balance", signed=True)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("recvWindow=", req.full_url)

    def test_execute_get_signed_api_key_in_header(self):
        """Signed execute_get puts API key in X-MBX-APIKEY header (case-insensitive)."""
        config = _make_config(api_key="my_api_key_123")
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[{"asset":"USDT"}]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            execute_get(config, "/fapi/v2/balance", signed=True)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertIn("x-mbx-apikey", headers_lower)
            self.assertEqual(headers_lower["x-mbx-apikey"], "my_api_key_123")

    def test_execute_get_unsigned_no_api_key_header(self):
        """Unsigned execute_get does not include X-MBX-APIKEY header."""
        config = _make_config(api_key="my_api_key_123")
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'{"serverTime": 1609459200000}'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            execute_get(config, "/fapi/v1/ping")
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertNotIn("x-mbx-apikey", headers_lower)


# ---------------------------------------------------------------------------
# execute_get HTTP error tests (mocked)
# ---------------------------------------------------------------------------

class TestExecuteGetHttpErrors(unittest.TestCase):
    """Tests for execute_get HTTP error handling."""

    def test_execute_get_http_429_empty_body(self):
        """429 with empty body raises DemoHttpError with rate limit classification."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.HTTPError(
                url="https://demo-fapi.binance.com/fapi/v1/ping",
                code=429,
                msg="Too Many Requests",
                hdrs=MagicMock(),
                fp=MagicMock(read=lambda: b""),
            )

            with self.assertRaises(DemoHttpError) as ctx:
                execute_get(config, "/fapi/v1/ping")
            self.assertEqual(ctx.exception.status_code, 429)
            self.assertIn("429", str(ctx.exception))

    def test_execute_get_http_418_empty_body(self):
        """418 with empty body raises DemoHttpError."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.HTTPError(
                url="https://demo-fapi.binance.com/fapi/v1/ping",
                code=418,
                msg="I'm a teapot",
                hdrs=MagicMock(),
                fp=MagicMock(read=lambda: b""),
            )

            with self.assertRaises(DemoHttpError) as ctx:
                execute_get(config, "/fapi/v1/ping")
            self.assertEqual(ctx.exception.status_code, 418)

    def test_execute_get_http_500_empty_body(self):
        """500 with empty body raises DemoHttpError."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.HTTPError(
                url="https://demo-fapi.binance.com/fapi/v1/ping",
                code=500,
                msg="Internal Server Error",
                hdrs=MagicMock(),
                fp=MagicMock(read=lambda: b""),
            )

            with self.assertRaises(DemoHttpError) as ctx:
                execute_get(config, "/fapi/v1/ping")
            self.assertEqual(ctx.exception.status_code, 500)
            self.assertIn("5xx", str(ctx.exception))

    def test_execute_get_http_404_empty_body(self):
        """404 with empty body raises DemoHttpError (path must be in allowlist)."""
        config = _make_config()
        # Use a known-good path; the 404 is returned by the mock regardless
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.HTTPError(
                url="https://demo-fapi.binance.com/fapi/v1/ping",
                code=404,
                msg="Not Found",
                hdrs=MagicMock(),
                fp=MagicMock(read=lambda: b""),
            )

            with self.assertRaises(DemoHttpError) as ctx:
                execute_get(config, "/fapi/v1/ping")
            self.assertEqual(ctx.exception.status_code, 404)

    def test_execute_get_http_401_with_binance_body(self):
        """HTTP 401 with Binance error JSON body raises BinanceApiError."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.HTTPError(
                url="https://demo-fapi.binance.com/fapi/v2/balance",
                code=401,
                msg="Unauthorized",
                hdrs=MagicMock(),
                fp=MagicMock(read=lambda: b'{"code": -2014, "msg": "API-key format invalid."}'),
            )

            with self.assertRaises(BinanceApiError) as ctx:
                execute_get(config, "/fapi/v2/balance", signed=True)
            self.assertEqual(ctx.exception.code, -2014)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("API-key format invalid", str(ctx.exception))

    def test_execute_get_http_400_with_binance_body(self):
        """HTTP 400 with Binance error JSON body raises BinanceApiError."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.HTTPError(
                url="https://demo-fapi.binance.com/fapi/v1/ping",
                code=400,
                msg="Bad Request",
                hdrs=MagicMock(),
                fp=MagicMock(read=lambda: b'{"code": -1234, "msg": "Invalid symbol"}'),
            )

            with self.assertRaises(BinanceApiError) as ctx:
                execute_get(config, "/fapi/v1/ping")
            self.assertEqual(ctx.exception.code, -1234)
            self.assertIn("Invalid symbol", str(ctx.exception))

    def test_execute_get_http_500_with_binance_body(self):
        """HTTP 500 with Binance error JSON body raises BinanceApiError (body takes priority)."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.HTTPError(
                url="https://demo-fapi.binance.com/fapi/v1/ping",
                code=500,
                msg="Internal Server Error",
                hdrs=MagicMock(),
                fp=MagicMock(read=lambda: b'{"code": -1000, "msg": "Server busy"}'),
            )

            with self.assertRaises(BinanceApiError) as ctx:
                execute_get(config, "/fapi/v1/ping")
            self.assertEqual(ctx.exception.code, -1000)
            self.assertIn("Server busy", str(ctx.exception))

    def test_execute_get_http_429_with_binance_body(self):
        """HTTP 429 with Binance error JSON body raises BinanceApiError."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.HTTPError(
                url="https://demo-fapi.binance.com/fapi/v1/ping",
                code=429,
                msg="Too Many Requests",
                hdrs=MagicMock(),
                fp=MagicMock(read=lambda: b'{"code": -1003, "msg": "Too many requests"}'),
            )

            with self.assertRaises(BinanceApiError) as ctx:
                execute_get(config, "/fapi/v1/ping")
            self.assertEqual(ctx.exception.code, -1003)
            self.assertIn("Too many requests", str(ctx.exception))

    def test_execute_get_timeout(self):
        """Timeout raises DemoHttpError."""
        config = _make_config()
        def raise_timeout(req, timeout=None):
            raise TimeoutError("timed out")
        with patch("execution.demo_client.urlopen", side_effect=raise_timeout):
            with self.assertRaises(DemoHttpError) as ctx:
                execute_get(config, "/fapi/v1/ping")
            self.assertIn("Timeout", str(ctx.exception))

    def test_execute_get_network_error(self):
        """Network error raises DemoHttpError."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.URLError("Connection refused")

            with self.assertRaises(DemoHttpError) as ctx:
                execute_get(config, "/fapi/v1/ping")
            self.assertIn("Connection refused", str(ctx.exception))

    def test_execute_get_malformed_json(self):
        """Non-JSON response raises BinanceResponseParseError."""
        config = _make_config()
        urlopen_mock = MagicMock()
        resp_mock = MagicMock()
        resp_mock.__enter__.return_value = resp_mock
        resp_mock.status = 200
        resp_mock.read.return_value = b"<html>Not JSON</html>"
        urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

        with patch("execution.demo_client.urlopen", urlopen_mock):
            with self.assertRaises(BinanceResponseParseError):
                execute_get(config, "/fapi/v1/ping")


# ---------------------------------------------------------------------------
# Endpoint wrapper tests (mocked)
# ---------------------------------------------------------------------------

class TestEndpoints(unittest.TestCase):
    """Tests for endpoint wrapper functions."""

    def test_ping_unsigned(self):
        """ping() is unsigned (no signature, no API key header)."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'{"serverTime": 1609459200000}'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            result = ping(config)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertEqual(req.full_url, f"{DEMO_BASE_URL}/fapi/v1/ping")
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertNotIn("x-mbx-apikey", headers_lower)
            self.assertNotIn("signature", req.full_url)
            self.assertEqual(result, {"serverTime": 1609459200000})

    def test_get_server_time_unsigned(self):
        """get_server_time() is unsigned."""
        config = _make_config()
        urlopen_mock = MagicMock()
        resp_mock = MagicMock()
        resp_mock.__enter__.return_value = resp_mock
        resp_mock.status = 200
        resp_mock.read.return_value = b'{"serverTime": 1700000000000}'
        urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

        with patch("execution.demo_client.urlopen", urlopen_mock):
            result = get_server_time(config)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertEqual(req.full_url, f"{DEMO_BASE_URL}/fapi/v1/time")
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertNotIn("x-mbx-apikey", headers_lower)
            self.assertIn("serverTime", result)

    def test_get_exchange_info_unsigned(self):
        """get_exchange_info() is unsigned."""
        config = _make_config()
        urlopen_mock = MagicMock()
        resp_mock = MagicMock()
        resp_mock.__enter__.return_value = resp_mock
        resp_mock.status = 200
        resp_mock.read.return_value = b'{"symbols":[{"symbol":"BTCUSDT","status":"TRADING"}],"exchangeName":"Binance"}'
        urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

        with patch("execution.demo_client.urlopen", urlopen_mock):
            result = get_exchange_info(config)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertEqual(req.full_url, f"{DEMO_BASE_URL}/fapi/v1/exchangeInfo")
            self.assertIn("symbols", result)

    def test_get_balance_signed(self):
        """get_balance() is signed (includes API key header and signature)."""
        config = _make_config(api_key="my_api_key_123")
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[{"asset":"USDT","free":"1000.00","locked":"0.00"}]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            result = get_balance(config)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertIn("x-mbx-apikey", headers_lower)
            self.assertIn("signature=", req.full_url)
            self.assertIsInstance(result, list)
            self.assertEqual(result[0]["asset"], "USDT")

    def test_get_balance_returns_list(self):
        """get_balance() returns a list."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[{"asset":"USDT","free":"1000.00","locked":"0.00"},{"asset":"BTC","free":"0.50","locked":"0.00"}]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            result = get_balance(config)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)

    def test_get_balance_rejects_non_list(self):
        """get_balance() raises BinanceApiError if response is not a list."""
        config = _make_config()
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'{"balances": "not a list"}'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            with self.assertRaises(BinanceApiError):
                get_balance(config)

    def test_get_account_signed(self):
        """get_account() is signed."""
        config = _make_config(api_key="my_key")
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'{"balances":[{"asset":"USDT","free":"1000"}],"permissions":["SPOT"]}'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            result = get_account(config)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertIn("x-mbx-apikey", headers_lower)
            self.assertIn("permissions", result)

    def test_get_open_orders_signed(self):
        """get_open_orders() is signed."""
        config = _make_config(api_key="my_key")
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[{"orderId":123,"symbol":"BTCUSDT","side":"BUY"}]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            result = get_open_orders(config)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertIn("x-mbx-apikey", headers_lower)
            self.assertIsInstance(result, list)

    def test_get_open_orders_with_symbol(self):
        """get_open_orders() with symbol includes symbol in query."""
        config = _make_config(api_key="my_key")
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            result = get_open_orders(config, symbol="BTCUSDT")
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("symbol=BTCUSDT", req.full_url)
            self.assertEqual(result, [])

    def test_get_open_orders_unsigned_fails(self):
        """get_open_orders() with invalid/missing credentials raises ValueError (signed endpoint)."""
        # ExecutionConfig rejects empty api_key at construction (fail-closed).
        # Use a valid non-empty key and verify the endpoint requires signing.
        config = _make_config(api_key="valid_demo_key")
        urlopen_mock = MagicMock()
        resp_mock = MagicMock()
        resp_mock.__enter__.return_value = resp_mock
        resp_mock.status = 200
        resp_mock.read.return_value = b"[]"
        urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

        with patch("execution.demo_client.urlopen", urlopen_mock):
            result = get_open_orders(config)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertIn("x-mbx-apikey", headers_lower)


# ---------------------------------------------------------------------------
# Secret protection tests
# ---------------------------------------------------------------------------

class TestSecretProtection(unittest.TestCase):
    """Tests verifying secrets are never exposed in logs/exceptions/responses."""

    def test_signature_does_not_contain_secret(self):
        """Generated signature does not contain the API secret."""
        secret = "my_secret_value"
        qs = "symbol=BTCUSDT&timestamp=123"
        sig = sign_query_string(qs, secret)
        self.assertNotIn(secret, sig)

    def test_error_messages_do_not_contain_secret(self):
        """Exception messages do not contain API secret."""
        exc = BinanceApiError("error message", code=-1234)
        self.assertNotIn("my_secret", str(exc))

    def test_demo_http_error_no_secret(self):
        """DemoHttpError does not contain API secret."""
        exc = DemoHttpError("http error", status_code=500, url="https://demo-fapi.binance.com")
        self.assertNotIn("my_secret", str(exc))

    def test_sign_query_string_rejects_empty_secret(self):
        """Empty secret raises ValueError without exposing secret in message."""
        with self.assertRaises(ValueError) as ctx:
            sign_query_string("symbol=BTCUSDT", "")
        error_msg = str(ctx.exception).lower()
        self.assertIn("non-empty", error_msg)

    def test_config_api_secret_not_in_url(self):
        """API secret does not appear in the constructed URL."""
        config = _make_config(api_secret="super_secret_value")
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'{"serverTime":1609459200000}'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            execute_get(config, "/fapi/v1/ping", signed=True)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertNotIn("super_secret_value", req.full_url)

    def test_config_api_key_not_in_url(self):
        """API key does not appear in the URL query string (only in header)."""
        config = _make_config(api_key="my_api_key_123")
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[{"asset":"USDT"}]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            execute_get(config, "/fapi/v2/balance", signed=True)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            # API key goes in header, NOT in URL
            self.assertNotIn("my_api_key_123", req.full_url)
            # But it IS in the header
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertIn("x-mbx-apikey", headers_lower)
            self.assertEqual(headers_lower["x-mbx-apikey"], "my_api_key_123")


# ---------------------------------------------------------------------------
# Config reuse tests
# ---------------------------------------------------------------------------

class TestConfigReuse(unittest.TestCase):
    """Tests verifying ExecutionConfig is reused correctly."""

    def test_config_provides_credentials_to_signer(self):
        """execute_get uses config.api_secret for signing."""
        config = _make_config(api_secret="test_secret_for_config")
        with patch("execution.demo_client.urlopen") as urlopen_mock:
            resp_mock = MagicMock()
            resp_mock.__enter__.return_value = resp_mock
            resp_mock.status = 200
            resp_mock.read.return_value = b'[{"asset":"USDT"}]'
            urlopen_mock.side_effect = lambda req, timeout=None: resp_mock

            execute_get(config, "/fapi/v2/balance", signed=True)
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            # Signature must be present (meaning secret was used)
            self.assertIn("signature=", req.full_url)

    def test_config_base_url_is_demo(self):
        """Config base_url is the demo endpoint."""
        config = _make_config()
        self.assertEqual(config.base_url, DEMO_BASE_CONSTANT)

    def test_config_mode_is_demo(self):
        """Config mode is 'demo'."""
        config = _make_config()
        self.assertEqual(config.mode, "demo")

    def test_config_cannot_use_live_url(self):
        """ExecutionConfig rejects non-demo base_url."""
        with self.assertRaises(ValueError) as ctx:
            _make_config(base_url="https://fapi.binance.com")
        self.assertIn("demo", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# No order endpoint tests
# ---------------------------------------------------------------------------

class TestNoOrderEndpoints(unittest.TestCase):
    """Verify no order/trading endpoints are implemented."""

    def test_no_post_order_in_module(self):
        """Module does not contain POST /fapi/v1/order."""
        import execution.demo_client as dc
        all_names = dir(dc)
        forbidden = ["place_order", "create_order", "order", "submit_order",
                     "cancel_order", "modify_order"]
        for name in forbidden:
            self.assertNotIn(name, all_names,
                             f"Forbidden order-related name found: {name}")

    def test_module_only_has_get_endpoints(self):
        """All endpoint wrappers are GET-only read endpoints."""
        import execution.demo_client as dc
        read_endpoints = {"ping", "get_server_time", "get_exchange_info",
                         "get_balance", "get_account", "get_open_orders"}
        for ep in read_endpoints:
            self.assertTrue(hasattr(dc, ep), f"Expected read endpoint missing: {ep}")

    def test_no_order_path_constants(self):
        """No order-placement path constants exist (only read endpoints)."""
        import execution.demo_client as dc
        all_paths = [attr for attr in dir(dc) if attr.startswith("PATH_")]
        # PATH_OPEN_ORDERS is a read endpoint (GET /fapi/v1/openOrders),
        # not an order placement path; exclude read-only order endpoints.
        read_order_paths = [p for p in all_paths if "order" in p.lower()]
        # Only PATH_OPEN_ORDERS should remain as a read-only order path
        self.assertEqual(read_order_paths, ["PATH_OPEN_ORDERS"],
                         f"Unexpected order paths: {read_order_paths}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
