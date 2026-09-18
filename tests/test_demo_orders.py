# tests/test_demo_orders.py — Binance Demo Futures order placement tests.
#
# All HTTP interactions are mocked. No real Binance network calls are made.
# No credentials are exposed in test output.

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import unittest

from execution.config import ExecutionConfig, DEMO_BASE_URL
from execution.demo_client import (
    build_query_string,
    build_signed_query,
    _check_demo_url,
    maybe_code,
)
from execution.demo_orders import (
    ALLOWED_ORDER_TYPES,
    DEFAULT_ORDER_TIMEOUT,
    DEMO_ORDER_BASE,
    ORDER_PATH,
    ApprovalRequiredError,
    DemoOrderError,
    DuplicateOrderIdError,
    RiskCheckFailedError,
    UnknownExecutionStateError,
    build_order_params,
    build_order_query_string,
    build_order_request,
    build_order_url,
    direction_to_side,
    place_limit_order,
    reset_local_order_registry,
)
from execution.ledger import ExecutionLedger
from execution.models import (
    ApprovedSignalRef,
    ExecutionResult,
    OrderRequest,
    SignalApproval,
)
from execution.risk import RiskValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_approved_signal(
    *,
    signal_id: str = "sig-001",
    symbol: str = "BTCUSDT",
    direction: str = "LONG",
    entry: float = 50000.0,
    sl: float = 49000.0,
    tp: float = 52000.0,
    score: float = 80.0,
    signal_time: int = 1700000000000,
    strategy_version: str = "v1.0",
    approved_at_ms: int = 1700000001000,
    approver: str = "manual",
) -> ApprovedSignalRef:
    """Build an ApprovedSignalRef with valid geometry for LONG."""
    return ApprovedSignalRef(
        signal_id=signal_id,
        symbol=symbol,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=tp,
        score=score,
        signal_time=signal_time,
        strategy_version=strategy_version,
        approved_at_ms=approved_at_ms,
        approver=approver,
    )


def _make_approval(
    *,
    signal_id: str = "sig-001",
    symbol: str = "BTCUSDT",
    direction: str = "LONG",
    approved_at_ms: int = 1700000001000,
    approver: str = "manual",
    approved: bool = True,
    reason: str = "Approved for demo execution",
) -> SignalApproval:
    """Build a SignalApproval."""
    return SignalApproval(
        signal_id=signal_id,
        symbol=symbol,
        direction=direction,
        approved_at_ms=approved_at_ms,
        approver=approver,
        reason=reason,
        approved=approved,
    )


def _mock_response(status_code: int, body: bytes) -> MagicMock:
    """Build a mock urlopen response context manager."""
    resp = MagicMock()
    resp.status = status_code
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    return resp


def _mock_urlopen_success(status_code: int = 200, body_dict: Optional[Dict[str, Any]] = None) -> MagicMock:
    """Build a mock urlopen that returns a successful response."""
    if body_dict is None:
        body_dict = {
            "orderId": 123456789,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "status": "NEW",
            "price": "50000.00",
            "executedQty": "0.000",
            "fee": "0.000",
        }
    urlopen_mock = MagicMock()
    urlopen_mock.side_effect = None
    urlopen_mock.return_value = _mock_response(status_code, json.dumps(body_dict).encode("utf-8"))
    return urlopen_mock


def _mock_urlopen_http_error(status_code: int, body_dict: Optional[Dict[str, Any]] = None) -> MagicMock:
    """Build a mock urlopen that raises an HTTPError."""
    if body_dict is None:
        body_dict = {"code": -2014, "msg": "API-key format invalid."}
    urlopen_mock = MagicMock()
    urlopen_mock.side_effect = urllib.error.HTTPError(
        url=f"{DEMO_ORDER_BASE}{ORDER_PATH}",
        code=status_code,
        msg="Error",
        hdrs=MagicMock(),
        fp=MagicMock(read=lambda: json.dumps(body_dict).encode("utf-8")),
    )
    return urlopen_mock


def _mock_urlopen_network_error(error_class: type) -> MagicMock:
    """Build a mock urlopen that raises a network error."""
    urlopen_mock = MagicMock()
    urlopen_mock.side_effect = error_class("simulated network error")
    return urlopen_mock


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------


class TestDirectionToSide(unittest.TestCase):
    """Tests for direction_to_side mapping."""

    def test_long_to_buy(self):
        self.assertEqual(direction_to_side("LONG"), "BUY")

    def test_short_to_sell(self):
        self.assertEqual(direction_to_side("SHORT"), "SELL")

    def test_invalid_direction_raises(self):
        with self.assertRaises(ValueError):
            direction_to_side("INVALID")


class TestBuildOrderParams(unittest.TestCase):
    """Tests for build_order_params."""

    def setUp(self):
        self.signal = _make_approved_signal(
            signal_id="sig-001",
            symbol="BTCUSDT",
            direction="LONG",
            entry=50000.0,
            sl=49000.0,
            tp=52000.0,
            score=80.0,
        )
        self.client_order_id = "COID-001"
        self.quantity = 0.1
        self.price = 50000.0

    def test_basic_params(self):
        params = build_order_params(
            self.signal,
            self.client_order_id,
            self.quantity,
            self.price,
        )
        self.assertEqual(params["symbol"], "BTCUSDT")
        self.assertEqual(params["side"], "BUY")
        self.assertEqual(params["type"], "LIMIT")
        self.assertEqual(params["timeInForce"], "GTC")
        self.assertEqual(params["quantity"], "0.1")
        self.assertEqual(params["price"], "50000.0")
        self.assertEqual(params["newClientOrderId"], "COID-001")

    def test_short_direction_maps_to_sell(self):
        short_signal = _make_approved_signal(direction="SHORT")
        params = build_order_params(short_signal, "COID-002", 0.1, 50000.0)
        self.assertEqual(params["side"], "SELL")

    def test_empty_client_order_id_raises(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, "", 0.1, 50000.0)

    def test_zero_quantity_raises(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, "COID-001", 0.0, 50000.0)

    def test_negative_quantity_raises(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, "COID-001", -0.1, 50000.0)

    def test_zero_price_raises(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, "COID-001", 0.1, 0.0)

    def test_negative_price_raises(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, "COID-001", 0.1, -50000.0)


class TestBuildOrderQueryString(unittest.TestCase):
    """Tests for build_order_query_string (signed query)."""

    def setUp(self):
        self.params = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": "0.1",
            "price": "50000.0",
            "newClientOrderId": "COID-001",
        }

    def test_signed_query_contains_signature(self):
        signed_query = build_order_query_string(self.params, "test_secret")
        self.assertIn("signature=", signed_query)

    def test_signed_query_contains_timestamp(self):
        signed_query = build_order_query_string(self.params, "test_secret")
        self.assertIn("timestamp=", signed_query)

    def test_signed_query_contains_recv_window(self):
        signed_query = build_order_query_string(self.params, "test_secret")
        self.assertIn("recvWindow=", signed_query)

    def test_unsigned_params_not_in_output(self):
        signed_query = build_order_query_string(self.params, "test_secret")
        # The raw query string should not contain the secret
        self.assertNotIn("test_secret", signed_query)

    def test_same_params_produce_same_signature(self):
        with patch("execution.demo_client.time") as time_mock:
            time_mock.time.return_value = 1700000000.0
            sig1 = build_order_query_string(self.params, "test_secret")
            sig2 = build_order_query_string(self.params, "test_secret")
            self.assertEqual(sig1, sig2)

    def test_different_secrets_produce_different_signatures(self):
        sig1 = build_order_query_string(self.params, "secret1")
        sig2 = build_order_query_string(self.params, "secret2")
        self.assertNotEqual(sig1, sig2)

    def test_params_not_mutated(self):
        original_params = dict(self.params)
        build_order_query_string(self.params, "test_secret")
        self.assertEqual(self.params, original_params)


class TestBuildOrderUrl(unittest.TestCase):
    """Tests for build_order_url (Demo-only URL construction)."""

    def test_builds_correct_demo_url(self):
        signed_query = "symbol=BTCUSDT&side=BUY&type=LIMIT&timeInForce=GTC&quantity=0.1&price=50000.0&newClientOrderId=COID-001&timestamp=1234567890&recvWindow=5000&signature=abc123"
        url = build_order_url(signed_query)
        expected = f"{DEMO_ORDER_BASE}{ORDER_PATH}?{signed_query}"
        self.assertEqual(url, expected)

    def test_build_order_url_uses_demo_base(self):
        """Verify build_order_url uses demo base."""
        url = build_order_url("symbol=BTCUSDT&signature=abc123")
        self.assertTrue(url.startswith(DEMO_ORDER_BASE))
        self.assertIn(ORDER_PATH, url)

    def test_build_order_url_rejects_non_demo_url(self):
        """Verify build_order_url rejects non-demo hostname."""
        from execution.demo_orders import _check_demo_url
        with self.assertRaises(ValueError):
            _check_demo_url("https://fapi.binance.com/fapi/v1/order")


class TestBuildOrderRequest(unittest.TestCase):
    """Tests for build_order_request (POST request construction)."""

    def test_builds_post_request(self):
        url = f"{DEMO_ORDER_BASE}{ORDER_PATH}?symbol=BTCUSDT"
        req = build_order_request(url, "test_api_key")
        self.assertEqual(req.method, "POST")

    def test_includes_api_key_header(self):
        url = f"{DEMO_ORDER_BASE}{ORDER_PATH}?symbol=BTCUSDT"
        req = build_order_request(url, "test_api_key")
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("x-mbx-apikey", headers_lower)
        self.assertEqual(headers_lower["x-mbx-apikey"], "test_api_key")

    def test_includes_user_agent(self):
        url = f"{DEMO_ORDER_BASE}{ORDER_PATH}?symbol=BTCUSDT"
        req = build_order_request(url, "test_api_key")
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("user-agent", headers_lower)
        self.assertIn("SignalForge-Demo", headers_lower["user-agent"])

    def test_includes_accept_header(self):
        url = f"{DEMO_ORDER_BASE}{ORDER_PATH}?symbol=BTCUSDT"
        req = build_order_request(url, "test_api_key")
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("accept", headers_lower)
        self.assertEqual(headers_lower["accept"], "application/json")

    def test_includes_content_type_header(self):
        url = f"{DEMO_ORDER_BASE}{ORDER_PATH}?symbol=BTCUSDT"
        req = build_order_request(url, "test_api_key")
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        self.assertIn("content-type", headers_lower)
        self.assertEqual(headers_lower["content-type"], "application/x-www-form-urlencoded")

    def test_empty_api_key_raises(self):
        url = f"{DEMO_ORDER_BASE}{ORDER_PATH}?symbol=BTCUSDT"
        with self.assertRaises(ValueError) as ctx:
            build_order_request(url, "")
        self.assertIn("API key required", str(ctx.exception))

    def test_url_contains_query_string(self):
        url = f"{DEMO_ORDER_BASE}{ORDER_PATH}?symbol=BTCUSDT&signature=abc123"
        req = build_order_request(url, "test_api_key")
        self.assertIn("symbol=BTCUSDT", req.full_url)

    def test_url_contains_signature(self):
        url = f"{DEMO_ORDER_BASE}{ORDER_PATH}?symbol=BTCUSDT&signature=abc123"
        req = build_order_request(url, "test_api_key")
        self.assertIn("signature=", req.full_url)

    def test_request_data_is_empty_bytes(self):
        url = f"{DEMO_ORDER_BASE}{ORDER_PATH}?symbol=BTCUSDT"
        req = build_order_request(url, "test_api_key")
        self.assertEqual(req.data, b"")


# ---------------------------------------------------------------------------
# Tests: place_limit_order — approval gate
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderApprovalGate(unittest.TestCase):
    """Tests verifying the explicit approval gate is enforced."""

    def setUp(self):
        self.config = _make_config()
        self.signal = _make_approved_signal()
        self.client_order_id = "COID-APPROVAL-001"
        self.quantity = 0.1
        self.price = 50000.0
        reset_local_order_registry()

    def test_requires_approval_object(self):
        with self.assertRaises(ApprovalRequiredError):
            place_limit_order(
                self.config,
                self.signal,
                None,  # no approval
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_requires_approved_true(self):
        unapproved = _make_approval(approved=False)
        with self.assertRaises(ApprovalRequiredError):
            place_limit_order(
                self.config,
                self.signal,
                unapproved,
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_default_approval_false(self):
        # SignalApproval default is approved=False
        default_approval = SignalApproval(
            signal_id="sig-001",
            symbol="BTCUSDT",
            direction="LONG",
            approved_at_ms=int(time.time() * 1000),
            approver="manual",
        )
        self.assertFalse(default_approval.is_approved())
        with self.assertRaises(ApprovalRequiredError):
            place_limit_order(
                self.config,
                self.signal,
                default_approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_approved_true_passes_gate(self):
        approval = _make_approval(approved=True)
        self.assertTrue(approval.is_approved())
        # Should NOT raise ApprovalRequiredError (mock the HTTP call)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123456789,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "50000.00",
                "executedQty": "0.000",
                "fee": "0.000",
            }).encode("utf-8"))
            result = place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            self.assertIsInstance(result, ExecutionResult)
            self.assertIn(result.status, ("ACCEPTED", "REJECTED", "ERROR", "UNKNOWN"))


# ---------------------------------------------------------------------------
# Tests: place_limit_order — risk gate
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderRiskGate(unittest.TestCase):
    """Tests verifying RiskValidator is called and its failures propagate."""

    def setUp(self):
        self.config = _make_config()
        self.signal = _make_approved_signal(direction="LONG")
        self.client_order_id = "COID-RISK-001"
        self.quantity = 0.1
        self.price = 50000.0
        reset_local_order_registry()

    def test_invalid_geometry_rejected(self):
        # LONG with SL > entry (invalid geometry)
        bad_signal = _make_approved_signal(
            direction="LONG",
            entry=50000.0,
            sl=51000.0,  # SL above entry — invalid
            tp=52000.0,
        )
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                bad_signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_symbol_not_in_allowlist_rejected(self):
        bad_signal = _make_approved_signal(symbol="NOTALLOWED")
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                bad_signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_rr_below_minimum_rejected(self):
        # R:R = (52000-50000)/(50000-49000) = 2000/1000 = 2.0, exactly at minimum
        # Make it 1.5: tp=49500 (below entry!) — invalid geometry anyway
        # Use valid geometry but low RR: entry=50000, sl=49000, tp=51000 → R:R=1.0
        low_rr_signal = _make_approved_signal(
            entry=50000.0,
            sl=49000.0,
            tp=51000.0,  # R:R = 1000/1000 = 1.0 < 2.0
        )
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                low_rr_signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_invalid_score_rejected(self):
        bad_signal = _make_approved_signal(score=150.0)  # > 100
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                bad_signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_quantity_exceeds_max_rejected(self):
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                200.0,  # > max_position_quantity=100.0
                self.price,
            )

    def test_zero_quantity_rejected(self):
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                0.0,
                self.price,
            )

    def test_negative_quantity_rejected(self):
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                -0.1,
                self.price,
            )

    def test_invalid_side_rejected(self):
        # direction_to_side will reject invalid direction in build_order_params
        bad_signal = _make_approved_signal(direction="INVALID")
        approval = _make_approval(approved=True)
        with self.assertRaises(ValueError):
            build_order_params(bad_signal, "COID", 0.1, 50000.0)


# ---------------------------------------------------------------------------
# Tests: place_limit_order — valid LONG/SHORT orders
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderValidCases(unittest.TestCase):
    """Tests for valid LIMIT order placement with mocked HTTP."""

    def setUp(self):
        self.config = _make_config()
        self.signal = _make_approved_signal(direction="LONG")
        self.client_order_id = "COID-VALID-001"
        self.quantity = 0.1
        self.price = 50000.0
        reset_local_order_registry()

    def _mock_success(
        self,
        urlopen_mock,
        *,
        order_id: int = 123456789,
        symbol: str = "BTCUSDT",
        side: str = "BUY",
        price: str = "50000.00",
    ) -> None:
        urlopen_mock.return_value = _mock_response(
            200,
            json.dumps({
                "orderId": order_id,
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "status": "NEW",
                "price": price,
                "executedQty": "0.000",
                "fee": "0.000",
                "timeInForce": "GTC",
            }).encode("utf-8"),
        )

    def test_valid_long_limit_order(self):
        signal = _make_approved_signal(direction="LONG")
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            self._mock_success(urlopen_mock)
            result = place_limit_order(
                self.config,
                signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            self.assertIsInstance(result, ExecutionResult)
            self.assertEqual(result.side, "BUY")
            self.assertEqual(result.order_type, "LIMIT")
            self.assertEqual(result.quantity, 0.1)
            self.assertEqual(result.symbol, "BTCUSDT")
            self.assertEqual(result.client_order_id, self.client_order_id)
            self.assertEqual(result.signal_id, signal.signal_id)
            self.assertIsNotNone(result.binance_order_id)
            self.assertEqual(result.binance_order_id, 123456789)
            self.assertIn(result.status, ("ACCEPTED", "REJECTED", "ERROR", "UNKNOWN"))

    def test_valid_short_limit_order(self):
        signal = _make_approved_signal(
            direction="SHORT",
            entry=50000.0,
            sl=51000.0,
            tp=48000.0,
        )
        approval = _make_approval(
            approved=True,
            direction="SHORT",
        )
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            self._mock_success(urlopen_mock, side="SELL")
            result = place_limit_order(
                self.config,
                signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            self.assertIsInstance(result, ExecutionResult)
            self.assertEqual(result.side, "SELL")
            self.assertEqual(result.order_type, "LIMIT")
            self.assertEqual(result.symbol, "BTCUSDT")
            self.assertEqual(result.client_order_id, self.client_order_id)

    def test_order_params_include_correct_symbol(self):
        signal = _make_approved_signal(symbol="ETHUSDT")
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            self._mock_success(urlopen_mock, symbol="ETHUSDT", side="SELL")
            result = place_limit_order(
                self.config,
                signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            self.assertEqual(result.symbol, "ETHUSDT")

    def test_order_params_include_quantity(self):
        signal = _make_approved_signal()
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            self._mock_success(urlopen_mock)
            result = place_limit_order(
                self.config,
                signal,
                approval,
                self.client_order_id,
                0.5,
                self.price,
            )
            self.assertEqual(result.quantity, 0.5)

    def test_order_params_include_price(self):
        signal = _make_approved_signal()
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            self._mock_success(urlopen_mock, price="51000.00")
            result = place_limit_order(
                self.config,
                signal,
                approval,
                self.client_order_id,
                self.quantity,
                51000.0,
            )
            self.assertEqual(result.fill_price, 51000.0)


# ---------------------------------------------------------------------------
# Tests: place_limit_order — Binance parameters
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderBinanceParameters(unittest.TestCase):
    """Tests verifying the Binance request parameters are correct."""

    def setUp(self):
        self.config = _make_config()
        self.signal = _make_approved_signal(direction="LONG")
        self.client_order_id = "COID-PARAMS-001"
        self.quantity = 0.1
        self.price = 50000.0
        reset_local_order_registry()

    def test_built_params_have_correct_symbol(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertEqual(params["symbol"], "BTCUSDT")

    def test_built_params_have_correct_side_for_long(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertEqual(params["side"], "BUY")

    def test_built_params_have_correct_side_for_short(self):
        short_signal = _make_approved_signal(direction="SHORT", entry=50000.0, sl=51000.0, tp=49000.0)
        params = build_order_params(short_signal, self.client_order_id, self.quantity, self.price)
        self.assertEqual(params["side"], "SELL")

    def test_built_params_have_limit_type(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertEqual(params["type"], "LIMIT")

    def test_built_params_have_gtc_time_in_force(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertEqual(params["timeInForce"], "GTC")

    def test_built_params_have_quantity_as_string(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertEqual(params["quantity"], "0.1")

    def test_built_params_have_price_as_string(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertEqual(params["price"], "50000.0")

    def test_built_params_have_client_order_id(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertEqual(params["newClientOrderId"], self.client_order_id)

    def test_built_params_dont_include_sl(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertNotIn("stopPrice", params)
        self.assertNotIn("sl", params)

    def test_built_params_dont_include_tp(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertNotIn("stopPrice2", params)
        self.assertNotIn("tp", params)

    def test_built_params_dont_include_reduce_only(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertNotIn("reduceOnly", params)

    def test_built_params_dont_include_close_position(self):
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertNotIn("closePosition", params)


# ---------------------------------------------------------------------------
# Tests: place_limit_order — signature and auth
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderSignatureAndAuth(unittest.TestCase):
    """Tests verifying HMAC signature and API key header are present."""

    def setUp(self):
        self.config = _make_config()
        self.signal = _make_approved_signal(direction="LONG")
        self.client_order_id = "COID-AUTH-001"
        self.quantity = 0.1
        self.price = 50000.0
        reset_local_order_registry()

    def test_signed_request_includes_signature_in_url(self):
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123456789,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "50000.00",
                "executedQty": "0.000",
                "fee": "0.000",
            }).encode("utf-8"))
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("signature=", req.full_url)

    def test_signed_request_includes_timestamp_in_url(self):
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123456789,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "50000.00",
                "executedQty": "0.000",
                "fee": "0.000",
            }).encode("utf-8"))
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("timestamp=", req.full_url)

    def test_signed_request_includes_recv_window_in_url(self):
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123456789,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "50000.00",
                "executedQty": "0.000",
                "fee": "0.000",
            }).encode("utf-8"))
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertIn("recvWindow=", req.full_url)

    def test_signed_request_includes_api_key_in_header(self):
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123456789,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "50000.00",
                "executedQty": "0.000",
                "fee": "0.000",
            }).encode("utf-8"))
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            headers_lower = {k.lower(): v for k, v in req.headers.items()}
            self.assertIn("x-mbx-apikey", headers_lower)
            self.assertEqual(headers_lower["x-mbx-apikey"], "test_demo_api_key_12345")

    def test_signed_request_uses_post_method(self):
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123456789,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "50000.00",
                "executedQty": "0.000",
                "fee": "0.000",
            }).encode("utf-8"))
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertEqual(req.method, "POST")

    def test_signature_does_not_contain_secret(self):
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123456789,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "50000.00",
                "executedQty": "0.000",
                "fee": "0.000",
            }).encode("utf-8"))
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertNotIn("test_demo_api_secret_67890", req.full_url)

    def test_api_key_not_in_url_query_string(self):
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123456789,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "50000.00",
                "executedQty": "0.000",
                "fee": "0.000",
            }).encode("utf-8"))
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            self.assertNotIn("test_demo_api_key_12345", req.full_url)

    def test_url_is_demo_only(self):
        approval = _make_approval(approved=True)
        with patch("execution.demo_orders.urlopen") as urlopen_mock:
            urlopen_mock.return_value = _mock_response(200, json.dumps({
                "orderId": 123456789,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "price": "50000.00",
                "executedQty": "0.000",
                "fee": "0.000",
            }).encode("utf-8"))
            place_limit_order(
                self.config,
                self.signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )
            call_args = urlopen_mock.call_args
            req = call_args[0][0]
            parsed = urllib.parse.urlparse(req.full_url)
            self.assertEqual(parsed.scheme, "https")
            self.assertEqual(parsed.hostname, "demo-fapi.binance.com")
            self.assertEqual(parsed.path, "/fapi/v1/order")


# ---------------------------------------------------------------------------
# Tests: place_limit_order — non-demo URL rejection
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderDemoOnlyEnforcement(unittest.TestCase):
    """Tests verifying non-demo URLs are rejected."""

    def setUp(self):
        self.config = _make_config()
        self.signal = _make_approved_signal(direction="LONG")
        self.client_order_id = "COID-DEMO-001"
        self.quantity = 0.1
        self.price = 50000.0
        reset_local_order_registry()

    def test_build_order_url_rejects_live_binance(self):
        from execution.demo_orders import _check_demo_url
        with self.assertRaises(ValueError):
            _check_demo_url("https://fapi.binance.com/fapi/v1/order")

    def test_build_order_url_rejects_non_demo_base(self):
        from execution.demo_orders import _check_demo_url
        with self.assertRaises(ValueError):
            _check_demo_url("https://not-demo.example.com/fapi/v1/order")

    def test_execution_fails_for_non_demo_config(self):
        # Config with mode != "demo" cannot be constructed because
        # ExecutionConfig.__post_init__ rejects it. So this is tested
        # implicitly by the config construction itself.
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
# Tests: place_limit_order — unsupported order types
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderUnsupportedTypes(unittest.TestCase):
    """Tests verifying unsupported order types are rejected."""

    def setUp(self):
        self.config = _make_config()
        self.signal = _make_approved_signal(direction="LONG")
        self.client_order_id = "COID-TYPE-001"
        self.quantity = 0.1
        self.price = 50000.0
        reset_local_order_registry()

    def test_market_orders_not_supported_in_params(self):
        # build_order_params always produces LIMIT, so MARKET is not creatable.
        # We test that the params always have type=LIMIT.
        params = build_order_params(self.signal, self.client_order_id, self.quantity, self.price)
        self.assertEqual(params["type"], "LIMIT")
        self.assertNotEqual(params["type"], "MARKET")

    def test_allowlist_contains_only_limit(self):
        self.assertIn("LIMIT", ALLOWED_ORDER_TYPES)
        self.assertNotIn("MARKET", ALLOWED_ORDER_TYPES)


# ---------------------------------------------------------------------------
# Tests: place_limit_order — invalid quantity/price
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderInvalidQuantityPrice(unittest.TestCase):
    """Tests verifying invalid quantity/price are rejected."""

    def setUp(self):
        self.config = _make_config()
        self.signal = _make_approved_signal(direction="LONG")
        self.client_order_id = "COID-INVP-001"
        reset_local_order_registry()

    def test_zero_quantity_raises_in_params(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, self.client_order_id, 0.0, 50000.0)

    def test_negative_quantity_raises_in_params(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, self.client_order_id, -0.1, 50000.0)

    def test_zero_price_raises_in_params(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, self.client_order_id, 0.1, 0.0)

    def test_negative_price_raises_in_params(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, self.client_order_id, 0.1, -50000.0)

    def test_empty_client_order_id_raises_in_params(self):
        with self.assertRaises(ValueError):
            build_order_params(self.signal, "", 0.1, 50000.0)


# ---------------------------------------------------------------------------
# Tests: place_limit_order — invalid geometry
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderInvalidGeometry(unittest.TestCase):
    """Tests verifying invalid entry/SL/TP geometry is rejected."""

    def setUp(self):
        self.config = _make_config()
        self.client_order_id = "COID-GEO-001"
        self.quantity = 0.1
        self.price = 50000.0
        reset_local_order_registry()

    def test_long_sl_above_entry_rejected(self):
        bad_signal = _make_approved_signal(
            direction="LONG",
            entry=50000.0,
            sl=51000.0,  # SL above entry — invalid for LONG
            tp=52000.0,
        )
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                bad_signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_long_tp_below_entry_rejected(self):
        bad_signal = _make_approved_signal(
            direction="LONG",
            entry=50000.0,
            sl=49000.0,
            tp=48000.0,  # TP below entry — invalid for LONG
        )
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                bad_signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_short_sl_below_entry_rejected(self):
        bad_signal = _make_approved_signal(
            direction="SHORT",
            entry=50000.0,
            sl=49000.0,  # SL below entry — invalid for SHORT
            tp=48000.0,
        )
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                bad_signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )

    def test_short_tp_above_entry_rejected(self):
        bad_signal = _make_approved_signal(
            direction="SHORT",
            entry=50000.0,
            sl=51000.0,
            tp=52000.0,  # TP above entry — invalid for SHORT
        )
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                bad_signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )


# ---------------------------------------------------------------------------
# Tests: place_limit_order — bad R:R
# ---------------------------------------------------------------------------


class TestPlaceLimitOrderBadRR(unittest.TestCase):
    """Tests verifying bad R:R is rejected."""

    def setUp(self):
        self.config = _make_config()
        self.client_order_id = "COID-RR-001"
        self.quantity = 0.1
        self.price = 50000.0
        reset_local_order_registry()

    def test_rr_below_minimum_rejected(self):
        # R:R = (51000-50000)/(50000-49000) = 1000/1000 = 1.0 < 2.0
        low_rr_signal = _make_approved_signal(
            entry=50000.0,
            sl=49000.0,
            tp=51000.0,
        )
        approval = _make_approval(approved=True)
        with self.assertRaises(RiskCheckFailedError):
            place_limit_order(
                self.config,
                low_rr_signal,
                approval,
                self.client_order_id,
                self.quantity,
                self.price,
            )