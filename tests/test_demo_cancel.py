# tests/test_demo_cancel.py — Binance Demo Futures order cancellation tests.
#
# All HTTP interactions are mocked. No real Binance network calls are made.
# No credentials are exposed in test output.
# No actual cancellation is performed.

from __future__ import annotations
import json
import time
import urllib.error
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch
import unittest

from execution.config import ExecutionConfig, DEMO_BASE_URL
from execution.demo_cancel import (
    CANCEL_PATH,
    DEMO_CANCEL_BASE,
    DEFAULT_CANCEL_TIMEOUT,
    CancelError,
    CancelResult,
    CancelUnknownStateError,
    build_cancel_params,
    build_cancel_query_string,
    build_cancel_request,
    build_cancel_url,
    cancel_order,
)
from execution.ledger import ExecutionLedger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, body: bytes) -> MagicMock:
    """Build a mock urlopen response context manager."""
    resp = MagicMock()
    resp.status = status_code
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    return resp


def _mock_urlopen_success(status_code: int = 200, body_dict: Optional[Dict[str, Any]] = None) -> MagicMock:
    """Build a mock urlopen that returns a successful cancel response."""
    if body_dict is None:
        body_dict = {
            "orderId": 28589596297,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "status": "CANCELED",
            "clientOrderId": "DEMO-TEST-1789733542449",
        }
    urlopen_mock = MagicMock()
    urlopen_mock.side_effect = None
    urlopen_mock.return_value = _mock_response(status_code, json.dumps(body_dict).encode("utf-8"))
    return urlopen_mock


def _mock_urlopen_http_error(
    status_code: int,
    body_dict: Optional[Dict[str, Any]] = None,
    *,
    body: Optional[bytes] = None,
) -> MagicMock:
    """Build a mock urlopen that raises an HTTPError.

    If ``body`` is provided, it is used as the response body (raw bytes).
    Otherwise, if ``body_dict`` is provided, it is JSON-encoded; if neither,
    the body is empty.
    """
    fp_read: Any
    if body is not None:
        fp_read = lambda: body
    elif body_dict is not None:
        fp_read = lambda: json.dumps(body_dict).encode("utf-8")
    else:
        fp_read = lambda: b""
    urlopen_mock = MagicMock()
    urlopen_mock.side_effect = urllib.error.HTTPError(
        url=f"{DEMO_CANCEL_BASE}{CANCEL_PATH}",
        code=status_code,
        msg="Error",
        hdrs=MagicMock(),
        fp=MagicMock(read=fp_read),
    )
    return urlopen_mock


def _mock_urlopen_network_error(error_class: type) -> MagicMock:
    """Build a mock urlopen that raises a network error."""
    urlopen_mock = MagicMock()
    urlopen_mock.side_effect = error_class("simulated network error")
    return urlopen_mock


def _make_config(**kwargs: Any) -> ExecutionConfig:
    """Build an ExecutionConfig with default demo values for testing."""
    defaults: Dict[str, Any] = dict(
        mode="demo",
        api_key="test_demo_api_key_12345",
        api_secret="test_demo_api_secret_67890",
        base_url=DEMO_BASE_URL,
        symbol_allowlist=["BTCUSDT", "ETHUSDT"],
        max_position_quantity=100.0,
        max_total_exposure_usd=10000.0,
        min_rr=2.0,
        approved_only=True,
    )
    defaults.update(kwargs)
    return ExecutionConfig(**defaults)


def _make_ledger(tmp_dir: str) -> ExecutionLedger:
    """Create a temporary ledger for testing."""
    import os
    log_path = os.path.join(tmp_dir, "cancel_test_ledger.jsonl")
    return ExecutionLedger(log_path)


# ---------------------------------------------------------------------------
# Tests: build_cancel_params
# ---------------------------------------------------------------------------


class TestBuildCancelParams(unittest.TestCase):
    """Tests for build_cancel_params."""

    def test_basic_params_with_order_id(self):
        params = build_cancel_params("BTCUSDT", order_id=28589596297)
        self.assertEqual(params["symbol"], "BTCUSDT")
        self.assertEqual(params["orderId"], "28589596297")
        self.assertNotIn("origClientOrderId", params)

    def test_basic_params_with_client_order_id(self):
        params = build_cancel_params("BTCUSDT", client_order_id="DEMO-TEST-1789733542449")
        self.assertEqual(params["symbol"], "BTCUSDT")
        self.assertNotIn("orderId", params)
        self.assertEqual(params["origClientOrderId"], "DEMO-TEST-1789733542449")

    def test_params_with_both_ids(self):
        params = build_cancel_params(
            "BTCUSDT",
            order_id=28589596297,
            client_order_id="DEMO-TEST-1789733542449",
        )
        self.assertEqual(params["symbol"], "BTCUSDT")
        self.assertEqual(params["orderId"], "28589596297")
        self.assertEqual(params["origClientOrderId"], "DEMO-TEST-1789733542449")

    def test_symbol_required(self):
        with self.assertRaises(ValueError):
            build_cancel_params("")
        with self.assertRaises(ValueError):
            build_cancel_params(None)  # type: ignore[arg-type]

    def test_params_not_mutated(self):
        original = {"symbol": "BTCUSDT", "timestamp": "123"}
        params = build_cancel_params("BTCUSDT", order_id=123)
        # params is a new dict, original unaffected
        params_with_timestamp = dict(params)
        params_with_timestamp["timestamp"] = "123"
        self.assertNotEqual(params, params_with_timestamp)


# ---------------------------------------------------------------------------
# Tests: build_cancel_query_string (signed)
# ---------------------------------------------------------------------------


class TestBuildCancelQueryString(unittest.TestCase):
    """Tests for build_cancel_query_string (signed query)."""

    def setUp(self):
        self.params = {
            "symbol": "BTCUSDT",
            "orderId": "28589596297",
        }

    def test_signed_query_contains_signature(self):
        signed_query = build_cancel_query_string(self.params, "test_secret")
        self.assertIn("signature=", signed_query)

    def test_signed_query_contains_timestamp(self):
        signed_query = build_cancel_query_string(self.params, "test_secret")
        self.assertIn("timestamp=", signed_query)

    def test_signed_query_contains_recv_window(self):
        signed_query = build_cancel_query_string(self.params, "test_secret")
        self.assertIn("recvWindow=", signed_query)

    def test_secret_not_in_output(self):
        signed_query = build_cancel_query_string(self.params, "test_secret")
        self.assertNotIn("test_secret", signed_query)

    def test_api_key_not_in_output(self):
        signed_query = build_cancel_query_string(self.params, "test_secret")
        self.assertNotIn("test_demo_api_key_12345", signed_query)

    def test_params_not_mutated(self):
        original_params = dict(self.params)
        build_cancel_query_string(self.params, "test_secret")
        self.assertEqual(self.params, original_params)

    def test_different_secrets_produce_different_signatures(self):
        sig1 = build_cancel_query_string(self.params, "secret1")
        sig2 = build_cancel_query_string(self.params, "secret2")
        self.assertNotEqual(sig1, sig2)


# ---------------------------------------------------------------------------
# Tests: build_cancel_url (Demo-only URL construction)
# ---------------------------------------------------------------------------


class TestBuildCancelUrl(unittest.TestCase):
    """Tests for build_cancel_url (Demo-only URL construction)."""

    def test_builds_correct_demo_url(self):
        signed_query = "symbol=BTCUSDT&orderId=28589596297&timestamp=1234567890&recvWindow=5000&signature=abc123"
        url = build_cancel_url(signed_query)
        expected = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?{signed_query}"
        self.assertEqual(url, expected)

    def test_builds_url_with_client_order_id(self):
        signed_query = "symbol=BTCUSDT&origClientOrderId=DEMO-TEST-1789733542449&timestamp=1234567890&recvWindow=5000&signature=abc123"
        url = build_cancel_url(signed_query)
        expected = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?{signed_query}"
        self.assertEqual(url, expected)

    def test_build_cancel_url_uses_demo_base(self):
        url = build_cancel_url("symbol=BTCUSDT&signature=abc123")
        self.assertTrue(url.startswith(DEMO_CANCEL_BASE))
        self.assertIn(CANCEL_PATH, url)

    def test_build_cancel_url_rejects_live_binance(self):
        from execution.demo_cancel import _check_demo_url
        with self.assertRaises(ValueError):
            _check_demo_url("https://fapi.binance.com/fapi/v1/order")

    def test_build_cancel_url_rejects_non_demo_base(self):
        from execution.demo_cancel import _check_demo_url
        with self.assertRaises(ValueError):
            _check_demo_url("https://not-demo.example.com/fapi/v1/order")

    def test_full_url_also_validated(self):
        from execution.demo_cancel import _check_demo_url
        # Full Demo URL should pass
        url = build_cancel_url(f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT")
        self.assertEqual(url, f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT")
        # Full live URL should fail
        with self.assertRaises(ValueError):
            _check_demo_url("https://fapi.binance.com/fapi/v1/order?symbol=BTCUSDT")


# ---------------------------------------------------------------------------
# Tests: build_cancel_request
# ---------------------------------------------------------------------------


class TestBuildCancelRequest(unittest.TestCase):
    """Tests for build_cancel_request (DELETE request construction)."""

    def test_builds_delete_request(self):
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT"
        req = build_cancel_request(url, "test_api_key")
        self.assertEqual(req.method, "DELETE")

    def test_includes_api_key_header(self):
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT"
        req = build_cancel_request(url, "test_api_key")
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("x-mbx-apikey", headers_lower)
        self.assertEqual(headers_lower["x-mbx-apikey"], "test_api_key")

    def test_includes_user_agent(self):
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT"
        req = build_cancel_request(url, "test_api_key")
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("user-agent", headers_lower)
        self.assertIn("SignalForge-Demo", headers_lower["user-agent"])

    def test_includes_accept_header(self):
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT"
        req = build_cancel_request(url, "test_api_key")
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("accept", headers_lower)
        self.assertEqual(headers_lower["accept"], "application/json")

    def test_empty_api_key_raises(self):
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT"
        with self.assertRaises(ValueError) as ctx:
            build_cancel_request(url, "")
        self.assertIn("API key required", str(ctx.exception))

    def test_request_data_is_empty_bytes(self):
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT"
        req = build_cancel_request(url, "test_api_key")
        self.assertEqual(req.data, b"")

    def test_url_contains_query_string(self):
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT&signature=abc123"
        req = build_cancel_request(url, "test_api_key")
        self.assertIn("symbol=BTCUSDT", req.full_url)

    def test_url_contains_signature(self):
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT&signature=abc123"
        req = build_cancel_request(url, "test_api_key")
        self.assertIn("signature=", req.full_url)

    def test_no_content_type_header(self):
        # DELETE requests don't need content-type; verify not set
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT"
        req = build_cancel_request(url, "test_api_key")
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertNotIn("content-type", headers_lower)

    def test_url_is_demo_only(self):
        approval = MagicMock()  # not used in request building
        url = f"{DEMO_CANCEL_BASE}{CANCEL_PATH}?symbol=BTCUSDT&signature=abc123"
        req = build_cancel_request(url, "test_api_key")
        from urllib.parse import urlparse
        parsed = urlparse(req.full_url)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "demo-fapi.binance.com")
        self.assertEqual(parsed.path, CANCEL_PATH)


# ---------------------------------------------------------------------------
# Tests: cancel_order — validation gates
# ---------------------------------------------------------------------------


class TestCancelOrderValidationGates(unittest.TestCase):
    """Tests verifying cancel_order validation gates."""

    def setUp(self):
        self.config = _make_config()

    def test_symbol_not_in_allowlist_rejected(self):
        with self.assertRaises(CancelError) as ctx:
            cancel_order(
                self.config,
                symbol="NOTALLOWED",
                order_id=123456789,
            )
        self.assertIn("not in allowlist", str(ctx.exception))

    def test_both_ids_missing_rejected(self):
        with self.assertRaises(CancelError) as ctx:
            cancel_order(
                self.config,
                symbol="BTCUSDT",
            )
        self.assertIn("At least one of order_id or client_order_id", str(ctx.exception))

    def test_symbol_empty_rejected(self):
        with self.assertRaises(CancelError) as ctx:
            cancel_order(
                self.config,
                symbol="",
                order_id=123456789,
            )
        self.assertIn("symbol must be", str(ctx.exception).lower())

    def test_config_mode_demo_enforced(self):
        # ExecutionConfig rejects non-demo mode at construction
        with self.assertRaises(ValueError):
            ExecutionConfig(
                mode="live",
                api_key="test_key",
                api_secret="test_secret",
                base_url=DEMO_BASE_URL,
                symbol_allowlist=["BTCUSDT"],
                max_position_quantity=100.0,
                max_total_exposure_usd=10000.0,
                min_rr=2.0,
                approved_only=True,
            )


# ---------------------------------------------------------------------------
# Tests: cancel_order — success cases
# ---------------------------------------------------------------------------


class TestCancelOrderSuccessCases(unittest.TestCase):
    """Tests for successful cancel with mocked HTTP."""

    def setUp(self):
        self.config = _make_config()
        self.symbol = "BTCUSDT"
        self.order_id = 28589596297
        self.client_order_id = "DEMO-TEST-1789733542449"

    def test_cancel_with_order_id(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 28589596297,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "status": "CANCELED",
                "clientOrderId": "DEMO-TEST-1789733542449",
            }).encode("utf-8"))
            result = cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            self.assertIsInstance(result, CancelResult)
            self.assertEqual(result.symbol, "BTCUSDT")
            self.assertEqual(result.side, "BUY")
            self.assertEqual(result.status, "CANCELED")
            self.assertEqual(result.order_id, 28589596297)
            self.assertEqual(result.client_order_id, "DEMO-TEST-1789733542449")
            self.assertIsNone(result.error_code)

    def test_cancel_with_client_order_id(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 28589597165,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "status": "CANCELED",
                "clientOrderId": "DEMO-TEST-1789733611865",
            }).encode("utf-8"))
            result = cancel_order(
                self.config,
                symbol=self.symbol,
                client_order_id=self.client_order_id,
            )
            self.assertIsInstance(result, CancelResult)
            self.assertEqual(result.symbol, "BTCUSDT")
            self.assertEqual(result.status, "CANCELED")
            self.assertEqual(result.client_order_id, "DEMO-TEST-1789733611865")

    def test_cancel_request_includes_order_id_param(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("orderId=28589596297", req.full_url)
            self.assertIn("symbol=BTCUSDT", req.full_url)

    def test_cancel_request_includes_client_order_id_param(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123,
                "symbol": self.symbol,
                "status": "CANCELED",
                "clientOrderId": self.client_order_id,
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                client_order_id=self.client_order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("origClientOrderId=DEMO-TEST-1789733542449", req.full_url)

    def test_cancel_with_ledger_logs_result(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger = _make_ledger(tmp_dir)
            with patch("execution.demo_cancel.urlopen") as urlopen_mock:
                urlopen_mock.return_value = _mock_response(200, json.dumps({
                    "orderId": self.order_id,
                    "symbol": self.symbol,
                    "side": "BUY",
                    "status": "CANCELED",
                    "clientOrderId": self.client_order_id,
                }).encode("utf-8"))
                result = cancel_order(
                    self.config,
                    symbol=self.symbol,
                    order_id=self.order_id,
                    ledger=ledger,
                )
                events = ledger.read_all()
                self.assertTrue(any(e.get("event") == "cancel_result" for e in events))
                cancel_event = next(e for e in events if e.get("event") == "cancel_result")
                self.assertEqual(cancel_event["symbol"], self.symbol)
                self.assertEqual(cancel_event["status"], "CANCELED")
                self.assertEqual(cancel_event["order_id"], self.order_id)

    def test_cancel_result_is_frozen(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            result = cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            with self.assertRaises(Exception):
                result.symbol = "ETHUSDT"  # type: ignore[assignment]

    def test_cancel_result_no_secrets(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            result = cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            result_dict = {
                "symbol": result.symbol,
                "side": result.side,
                "status": result.status,
                "order_id": result.order_id,
                "client_order_id": result.client_order_id,
                "message": result.message,
                "error_code": result.error_code,
            }
            # None of these should contain the API secret
            self.assertNotIn("test_demo_api_secret_67890", str(result_dict))
            self.assertNotIn("test_demo_api_key_12345", str(result_dict))


# ---------------------------------------------------------------------------
# Tests: cancel_order — Binance error responses
# ---------------------------------------------------------------------------


class TestCancelOrderBinanceErrors(unittest.TestCase):
    """Tests for Binance error responses."""

    def setUp(self):
        self.config = _make_config()
        self.symbol = "BTCUSDT"
        self.order_id = 28589596297

    def test_cancel_http_error_with_binance_body_rejected(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = _mock_urlopen_http_error(400, {
                "code": -1234,
                "msg": "Invalid symbol",
            })
            result = cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            self.assertIsInstance(result, CancelResult)
            self.assertEqual(result.status, "REJECTED")
            self.assertEqual(result.error_code, -1234)
            self.assertIn("Invalid symbol", result.message)

    def test_cancel_http_error_without_body_rejected(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = _mock_urlopen_http_error(404, body=b"")
            result = cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            self.assertIsInstance(result, CancelResult)
            self.assertEqual(result.status, "REJECTED")
            self.assertIsNone(result.error_code)

    def test_cancel_binance_error_code_in_response(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "code": -1000,
                "msg": "Server busy",
            }).encode("utf-8"))
            result = cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            self.assertIsInstance(result, CancelResult)
            self.assertEqual(result.status, "REJECTED")
            self.assertEqual(result.error_code, -1000)
            self.assertIn("Server busy", result.message)


# ---------------------------------------------------------------------------
# Tests: cancel_order — network failures (no retry)
# ---------------------------------------------------------------------------


class TestCancelOrderNetworkFailures(unittest.TestCase):
    """Tests for network failures — CancelUnknownStateError, no retry."""

    def setUp(self):
        self.config = _make_config()
        self.symbol = "BTCUSDT"
        self.order_id = 28589596297

    def test_url_error_raises_unknown_state(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.URLError("Connection refused")
            with self.assertRaises(CancelUnknownStateError) as ctx:
                cancel_order(
                    self.config,
                    symbol=self.symbol,
                    order_id=self.order_id,
                )
            self.assertIn("URLError", str(ctx.exception))
            self.assertIn("Connection refused", str(ctx.exception))
            self.assertEqual(ctx.exception.order_id, self.order_id)

    def test_timeout_error_raises_unknown_state(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = TimeoutError("timed out")
            with self.assertRaises(CancelUnknownStateError) as ctx:
                cancel_order(
                    self.config,
                    symbol=self.symbol,
                    order_id=self.order_id,
                )
            self.assertIn("TimeoutError", str(ctx.exception))
            self.assertIn("timed out", str(ctx.exception))
            self.assertEqual(ctx.exception.order_id, self.order_id)

    def test_unknown_state_error_message_does_not_contain_secret(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.URLError("Connection refused")
            try:
                cancel_order(
                    self.config,
                    symbol=self.symbol,
                    order_id=self.order_id,
                )
            except CancelUnknownStateError as e:
                self.assertNotIn("test_demo_api_secret_67890", str(e))
                self.assertNotIn("test_demo_api_key_12345", str(e))

    def test_network_error_does_not_retry(self):
        # Verify that on network error, the function raises immediately
        # (no retry loop). We verify by checking only one call was made.
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.side_effect = urllib.error.URLError("Connection refused")
            try:
                cancel_order(
                    self.config,
                    symbol=self.symbol,
                    order_id=self.order_id,
                )
            except CancelUnknownStateError:
                pass
            # Only one call was made (no retry)
            self.assertEqual(urlopen_mock.call_count, 1)


# ---------------------------------------------------------------------------
# Tests: cancel_order — signature and auth
# ---------------------------------------------------------------------------


class TestCancelOrderSignatureAndAuth(unittest.TestCase):
    """Tests verifying HMAC signature and API key header are present."""

    def setUp(self):
        self.config = _make_config()
        self.symbol = "BTCUSDT"
        self.order_id = 28589596297

    def test_signed_request_includes_signature_in_url(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("signature=", req.full_url)

    def test_signed_request_includes_timestamp_in_url(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("timestamp=", req.full_url)

    def test_signed_request_includes_recv_window_in_url(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("recvWindow=", req.full_url)

    def test_signed_request_includes_api_key_in_header(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertIn("x-mbx-apikey", headers_lower)
            self.assertEqual(headers_lower["x-mbx-apikey"], "test_demo_api_key_12345")

    def test_signature_does_not_contain_secret(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertNotIn("test_demo_api_secret_67890", req.full_url)

    def test_api_key_not_in_url_query_string(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertNotIn("test_demo_api_key_12345", req.full_url)

    def test_url_is_demo_only(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            from urllib.parse import urlparse
            parsed = urlparse(req.full_url)
            self.assertEqual(parsed.scheme, "https")
            self.assertEqual(parsed.hostname, "demo-fapi.binance.com")
            self.assertEqual(parsed.path, CANCEL_PATH)

    def test_request_method_is_delete(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertEqual(req.method, "DELETE")


# ---------------------------------------------------------------------------
# Tests: cancel_order — non-demo URL rejection
# ---------------------------------------------------------------------------


class TestCancelOrderDemoOnlyEnforcement(unittest.TestCase):
    """Tests verifying non-demo URLs are rejected."""

    def setUp(self):
        self.config = _make_config()
        self.symbol = "BTCUSDT"
        self.order_id = 28589596297

    def test_build_cancel_url_rejects_live_binance(self):
        from execution.demo_cancel import _check_demo_url
        with self.assertRaises(ValueError):
            _check_demo_url("https://fapi.binance.com/fapi/v1/order")

    def test_build_cancel_url_rejects_non_demo_base(self):
        from execution.demo_cancel import _check_demo_url
        with self.assertRaises(ValueError):
            _check_demo_url("https://not-demo.example.com/fapi/v1/order")


# ---------------------------------------------------------------------------
# Tests: cancel_result fields and behavior
# ---------------------------------------------------------------------------


class TestCancelResultFields(unittest.TestCase):
    """Tests for CancelResult fields and behavior."""

    def test_cancel_result_fields(self):
        result = CancelResult(
            symbol="BTCUSDT",
            side="BUY",
            status="CANCELED",
            message="Demo order cancellation CANCELED",
            order_id=28589596297,
            client_order_id="DEMO-TEST-1789733542449",
            executed_at_ms=1789733542449,
            error_code=None,
        )
        self.assertEqual(result.symbol, "BTCUSDT")
        self.assertEqual(result.side, "BUY")
        self.assertEqual(result.status, "CANCELED")
        self.assertEqual(result.order_id, 28589596297)
        self.assertEqual(result.client_order_id, "DEMO-TEST-1789733542449")
        self.assertIsNone(result.error_code)

    def test_cancel_result_with_error_code(self):
        result = CancelResult(
            symbol="BTCUSDT",
            side="",
            status="REJECTED",
            message="Binance API error (code=-1234): Invalid symbol",
            order_id=28589596297,
            client_order_id="DEMO-TEST-1789733542449",
            executed_at_ms=1789733542449,
            error_code=-1234,
        )
        self.assertEqual(result.error_code, -1234)
        self.assertEqual(result.status, "REJECTED")

    def test_cancel_result_str_safe(self):
        result = CancelResult(
            symbol="BTCUSDT",
            side="BUY",
            status="CANCELED",
            message="Demo order cancellation CANCELED",
            order_id=28589596297,
            client_order_id="DEMO-TEST-1789733542449",
            executed_at_ms=1789733542449,
        )
        s = str(result)
        self.assertNotIn("test_demo_api_secret", s)
        self.assertNotIn("test_demo_api_key", s)

    def test_cancel_result_repr_safe(self):
        result = CancelResult(
            symbol="BTCUSDT",
            side="BUY",
            status="CANCELED",
            message="Demo order cancellation CANCELED",
            order_id=28589596297,
            client_order_id="DEMO-TEST-1789733542449",
            executed_at_ms=1789733542449,
        )
        r = repr(result)
        self.assertNotIn("test_demo_api_secret", r)
        self.assertNotIn("test_demo_api_key", r)

    def test_cancel_result_no_secrets_in_any_field(self):
        result = CancelResult(
            symbol="BTCUSDT",
            side="BUY",
            status="CANCELED",
            message="Demo order cancellation CANCELED",
            order_id=28589596297,
            client_order_id="DEMO-TEST-1789733542449",
            executed_at_ms=1789733542449,
        )
        for field in ["symbol", "side", "status", "message", "client_order_id"]:
            value = getattr(result, field)
            self.assertNotIn("test_demo_api_secret", str(value))
            self.assertNotIn("test_demo_api_key", str(value))


# ---------------------------------------------------------------------------
# Tests: cancel_order — malformed responses
# ---------------------------------------------------------------------------


class TestCancelOrderMalformedResponses(unittest.TestCase):
    """Tests for malformed/non-JSON responses."""

    def setUp(self):
        self.config = _make_config()
        self.symbol = "BTCUSDT"
        self.order_id = 28589596297

    def test_non_json_response_error(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, b"<html>Not JSON</html>")
            result = cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            self.assertIsInstance(result, CancelResult)
            self.assertEqual(result.status, "ERROR")
            self.assertIn("Response parse error", result.message)

    def test_non_dict_response_error(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, b"[1, 2, 3]")
            result = cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            self.assertIsInstance(result, CancelResult)
            self.assertEqual(result.status, "ERROR")
            self.assertIn("Unexpected response type", result.message)


# ---------------------------------------------------------------------------
# Tests: cancel_order — timeout behavior
# ---------------------------------------------------------------------------


class TestCancelOrderTimeout(unittest.TestCase):
    """Tests for timeout handling."""

    def setUp(self):
        self.config = _make_config()
        self.symbol = "BTCUSDT"
        self.order_id = 28589596297

    def test_default_timeout_used(self):
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
            )
            call_args = urlopen_mock.call_args
            # timeout is passed as keyword argument
            self.assertEqual(call_args[1].get("timeout"), DEFAULT_CANCEL_TIMEOUT[1])

    def test_custom_timeout_passed(self):
        custom_timeout = (5.0, 10.0)
        with patch("execution.demo_cancel.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": self.order_id,
                "symbol": self.symbol,
                "status": "CANCELED",
            }).encode("utf-8"))
            cancel_order(
                self.config,
                symbol=self.symbol,
                order_id=self.order_id,
                timeout=custom_timeout,
            )
            call_args = urlopen_mock.call_args
            self.assertEqual(call_args[1].get("timeout"), custom_timeout[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
