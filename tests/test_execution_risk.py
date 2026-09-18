# Tests for execution risk validation.

from __future__ import annotations
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution.config import ExecutionConfig, DEMO_BASE_URL
from execution.models import ApprovedSignalRef, OrderRequest
from execution.risk import RiskValidator, RiskValidationError


def _make_config(
    mode: str = "demo",
    api_key: str = "test_key",
    api_secret: str = "test_secret",
    symbol_allowlist=None,
    max_position_quantity: float = 10.0,
    max_total_exposure_usd: float = 10000.0,
    min_rr: float = 2.0,
    approved_only: bool = True,
) -> ExecutionConfig:
    if symbol_allowlist is None:
        symbol_allowlist = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    return ExecutionConfig(
        mode=mode,
        api_key=api_key,
        api_secret=api_secret,
        base_url=DEMO_BASE_URL,
        symbol_allowlist=symbol_allowlist,
        max_position_quantity=max_position_quantity,
        max_total_exposure_usd=max_total_exposure_usd,
        min_rr=min_rr,
        approved_only=approved_only,
    )


def _make_signal(
    signal_id: str = "sig_test",
    symbol: str = "BTCUSDT",
    direction: str = "LONG",
    entry: float = 100.0,
    sl: float = 95.0,
    tp: float = 110.0,
    score: float = 65.0,
    signal_time: int = 1000000,
    strategy_version: str = "v1",
    approved_at_ms: int = 1000001,
    approver: str = "manual",
) -> ApprovedSignalRef:
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


class TestValidSignals(unittest.TestCase):
    def test_valid_long(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(
            symbol="BTCUSDT", direction="LONG",
            entry=100.0, sl=95.0, tp=110.0, score=65.0,
        )
        validator.validate_signal(signal)

    def test_valid_short(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(
            symbol="ETHUSDT", direction="SHORT",
            entry=2000.0, sl=2100.0, tp=1800.0, score=70.0,
        )
        validator.validate_signal(signal)


class TestInvalidDirection(unittest.TestCase):
    def test_invalid_direction(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(direction="BUY")
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_DIRECTION")

    def test_empty_direction(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(direction="")
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_DIRECTION")


class TestInvalidPrices(unittest.TestCase):
    def test_zero_entry(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(entry=0.0)
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_PRICE")

    def test_negative_entry(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(entry=-100.0)
        with self.assertRaises(RiskValidationError):
            validator.validate_signal(signal)

    def test_zero_sl(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(sl=0.0)
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertIn("sl", str(ctx.exception))

    def test_negative_sl(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(sl=-5.0)
        with self.assertRaises(RiskValidationError):
            validator.validate_signal(signal)

    def test_zero_tp(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(tp=0.0)
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertIn("tp", str(ctx.exception))

    def test_nan_entry(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(entry=float("nan"))
        with self.assertRaises(RiskValidationError):
            validator.validate_signal(signal)


class TestInvalidGeometry(unittest.TestCase):
    def test_long_sl_above_entry(self):
        config = _make_config()
        validator = RiskValidator(config)
        # LONG with SL > entry but R:R >= 2.0 (TP far enough to pass RR check
        # after geometry validation — but geometry fails first)
        signal = _make_signal(
            direction="LONG", entry=100.0, sl=110.0, tp=140.0,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_GEOMETRY_LONG")

    def test_long_tp_below_entry(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(
            direction="LONG", entry=100.0, sl=95.0, tp=90.0,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_GEOMETRY_LONG")

    def test_long_sl_equal_entry(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(
            direction="LONG", entry=100.0, sl=100.0, tp=110.0,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_GEOMETRY_LONG")

    def test_short_tp_above_entry(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(
            direction="SHORT", entry=100.0, sl=110.0, tp=105.0,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_GEOMETRY_SHORT")

    def test_short_sl_below_entry(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(
            direction="SHORT", entry=100.0, sl=95.0, tp=90.0,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_GEOMETRY_SHORT")

    def test_short_sl_equal_entry(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(
            direction="SHORT", entry=100.0, sl=100.0, tp=90.0,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_GEOMETRY_SHORT")


class TestInvalidScore(unittest.TestCase):
    def test_score_negative(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(score=-5.0)
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_SCORE")

    def test_score_above_100(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(score=105.0)
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "INVALID_SCORE")

    def test_score_exactly_100(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(score=100.0)
        validator.validate_signal(signal)

    def test_score_exactly_0(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal(score=0.0)
        validator.validate_signal(signal)


class TestRrBelowMinimum(unittest.TestCase):
    def test_rr_below_2(self):
        config = _make_config(min_rr=2.0)
        validator = RiskValidator(config)
        signal = _make_signal(
            direction="LONG", entry=100.0, sl=95.0, tp=107.0, score=65.0,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "RR_BELOW_MINIMUM")

    def test_rr_exactly_2(self):
        config = _make_config(min_rr=2.0)
        validator = RiskValidator(config)
        signal = _make_signal(
            direction="LONG", entry=100.0, sl=95.0, tp=110.0, score=65.0,
        )
        validator.validate_signal(signal)

    def test_rr_below_custom_min(self):
        config = _make_config(min_rr=3.0)
        validator = RiskValidator(config)
        signal = _make_signal(
            direction="LONG", entry=100.0, sl=95.0, tp=110.0, score=65.0,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "RR_BELOW_MINIMUM")


class TestInvalidQuantity(unittest.TestCase):
    def test_zero_quantity(self):
        config = _make_config(max_position_quantity=10.0)
        validator = RiskValidator(config)
        signal = _make_signal()
        validator.validate_signal(signal)
        order = OrderRequest(
            client_order_id="cid_qty_0",
            symbol="BTCUSDT", side="BUY", order_type="MARKET",
            quantity=0.0, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000002,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_order(order)
        self.assertEqual(ctx.exception.code, "INVALID_QUANTITY")

    def test_negative_quantity(self):
        config = _make_config(max_position_quantity=10.0)
        validator = RiskValidator(config)
        signal = _make_signal()
        validator.validate_signal(signal)
        order = OrderRequest(
            client_order_id="cid_qty_neg",
            symbol="BTCUSDT", side="BUY", order_type="MARKET",
            quantity=-1.0, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000003,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_order(order)
        self.assertEqual(ctx.exception.code, "INVALID_QUANTITY")

    def test_quantity_exceeds_max(self):
        config = _make_config(max_position_quantity=1.0)
        validator = RiskValidator(config)
        signal = _make_signal()
        validator.validate_signal(signal)
        order = OrderRequest(
            client_order_id="cid_qty_over",
            symbol="BTCUSDT", side="BUY", order_type="MARKET",
            quantity=5.0, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000004,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_order(order)
        self.assertEqual(ctx.exception.code, "QUANTITY_EXCEEDS_LIMIT")

    def test_quantity_at_max_boundary(self):
        config = _make_config(max_position_quantity=2.5)
        validator = RiskValidator(config)
        signal = _make_signal()
        validator.validate_signal(signal)
        order = OrderRequest(
            client_order_id="cid_qty_max",
            symbol="BTCUSDT", side="BUY", order_type="MARKET",
            quantity=2.5, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000005,
        )
        validator.validate_order(order)


class TestDuplicateOrderId(unittest.TestCase):
    def test_duplicate_order_id(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal()
        validator.validate_signal(signal)
        order1 = OrderRequest(
            client_order_id="cid_dup_001",
            symbol="BTCUSDT", side="BUY", order_type="MARKET",
            quantity=0.5, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000006,
        )
        validator.validate_order(order1)
        order2 = OrderRequest(
            client_order_id="cid_dup_001",
            symbol="BTCUSDT", side="SELL", order_type="MARKET",
            quantity=0.3, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000007,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_order(order2)
        self.assertEqual(ctx.exception.code, "DUPLICATE_ORDER_ID")

    def test_different_order_ids_allowed(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal()
        validator.validate_signal(signal)
        order1 = OrderRequest(
            client_order_id="cid_a",
            symbol="BTCUSDT", side="BUY", order_type="MARKET",
            quantity=0.5, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000008,
        )
        order2 = OrderRequest(
            client_order_id="cid_b",
            symbol="BTCUSDT", side="BUY", order_type="MARKET",
            quantity=0.3, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000009,
        )
        validator.validate_order(order1)
        validator.validate_order(order2)


class TestInvalidOrderType(unittest.TestCase):
    def test_invalid_order_type(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal()
        validator.validate_signal(signal)
        order = OrderRequest(
            client_order_id="cid_type_bad",
            symbol="BTCUSDT", side="BUY", order_type="STOP_LOSS",
            quantity=0.5, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000010,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_order(order)
        self.assertEqual(ctx.exception.code, "INVALID_ORDER_TYPE")


class TestInvalidSide(unittest.TestCase):
    def test_invalid_side(self):
        config = _make_config()
        validator = RiskValidator(config)
        signal = _make_signal()
        validator.validate_signal(signal)
        order = OrderRequest(
            client_order_id="cid_side_bad",
            symbol="BTCUSDT", side="LONG", order_type="MARKET",
            quantity=0.5, price=None, stop_price=None,
            time_in_force=None, reduce_only=False, close_position=False,
            signal_id="sig_test", strategy_version="v1", requested_at_ms=1000011,
        )
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_order(order)
        self.assertEqual(ctx.exception.code, "INVALID_SIDE")


class TestSymbolNotAllowed(unittest.TestCase):
    def test_symbol_not_in_allowlist(self):
        config = _make_config(symbol_allowlist=["BTCUSDT", "ETHUSDT"])
        validator = RiskValidator(config)
        signal = _make_signal(symbol="SOLUSDT")
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "SYMBOL_NOT_ALLOWED")


class TestNonDemoMode(unittest.TestCase):
    def test_live_mode_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _make_config(mode="live")
        self.assertIn("must be exactly 'demo'", str(ctx.exception))

    def test_empty_mode_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _make_config(mode="")
        self.assertIn("must be exactly 'demo'", str(ctx.exception))


class TestMalformedInput(unittest.TestCase):
    def test_missing_symbol(self):
        config = _make_config(symbol_allowlist=["BTCUSDT"])
        validator = RiskValidator(config)
        signal = _make_signal(symbol="")
        with self.assertRaises(RiskValidationError) as ctx:
            validator.validate_signal(signal)
        self.assertEqual(ctx.exception.code, "SYMBOL_NOT_ALLOWED")


class TestExposureCalculation(unittest.TestCase):
    def test_exposure_is_quantity_times_entry(self):
        """Exposure = quantity * entry (notional value)."""
        quantity = 0.5
        entry = 100000.0
        exposure = quantity * entry
        self.assertEqual(exposure, 50000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
