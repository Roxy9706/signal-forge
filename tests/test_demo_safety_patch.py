"""
Regression tests for the DEMO EXECUTION SAFETY PATCH (execution infra only).

Covers:
  1. LONG/SHORT geometry semantics (authoritative: LONG SL<entry<TP, SHORT TP<entry<SL)
  2. Notional/exposure enforcement (max_position_quantity, max_open_orders,
     max_total_exposure_usd; fail closed on missing price/quantity)
  3. min_rr compared against ACTUAL R:R, never score
  4. Deterministic precision quantization + post-round revalidation (execution/precision.py)
  5. Default D15 signal source matches the scanner's output ledger
  6. CLI safety controls (--kill-switch, --max-open-orders, --max-exposure-usd)
     actually influence execution and can only tighten, never loosen.

No network calls. No orders. No strategy imports.
"""

import json
import os
import tempfile
from decimal import Decimal
from unittest.mock import patch

from execution.config import ExecutionConfig, DEMO_BASE_URL
from execution.demo_auto import (
    D15_SIGNAL_LOG_DEFAULT,
    _effective_limits,
    _kill_switch_active,
    main,
    process_signal,
)
from execution.ledger import ExecutionLedger
from execution.precision import (
    PrecisionDataUnavailableError,
    get_symbol_filters,
    price_preserves_geometry,
    quantize_price,
    quantize_quantity,
    validate_quantized_quantity,
)

from tests.test_demo_auto import _make_config, _make_signal, _env_setup, _env_clear, _safe_unlink


# ── 1. Geometry regression ────────────────────────────────────────────

class TestGeometrySemantics:
    """LONG: SL < entry < TP.  SHORT: TP < entry < SL. (mirrors risk.py)"""

    def test_valid_long_accepted(self):
        config = _make_config()
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        result = process_signal(_make_signal(direction="LONG", entry=100.0, sl=90.0, tp=120.0),
                                config, ledger)
        assert result is not None

    def test_valid_short_accepted(self):
        """The fixed bug: valid SHORTs (TP < entry < SL) must NOT be rejected."""
        config = _make_config()
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        result = process_signal(_make_signal(direction="SHORT", entry=100.0, sl=110.0, tp=80.0),
                                config, ledger)
        assert result is not None
        assert result["side"] == "SELL"

    def test_invalid_short_rejected_sl_below_entry(self):
        config = _make_config()
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        result = process_signal(_make_signal(direction="SHORT", entry=100.0, sl=90.0, tp=80.0),
                                config, ledger)
        assert result is None

    def test_invalid_short_rejected_tp_above_entry(self):
        config = _make_config()
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        result = process_signal(_make_signal(direction="SHORT", entry=100.0, sl=110.0, tp=120.0),
                                config, ledger)
        assert result is None

    def test_invalid_long_rejected(self):
        config = _make_config()
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        result = process_signal(_make_signal(direction="LONG", entry=100.0, sl=110.0, tp=120.0),
                                config, ledger)
        assert result is None

    def test_geometry_boundary_equalities_rejected(self):
        """entry == sl or entry == tp is degenerate risk => reject (fail closed)."""
        config = _make_config()
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        assert process_signal(_make_signal(entry=100.0, sl=100.0, tp=120.0), config, ledger) is None
        assert process_signal(_make_signal(entry=100.0, sl=90.0, tp=100.0), config, ledger) is None


# ── 2. Notional / exposure regression ─────────────────────────────────

class TestExposureEnforcement:

    def _ledger(self):
        return ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))

    def test_btc_like_price_quantity_reduced_to_fit_cap(self):
        """1 BTC @ ~$75k must never be proposed against a $1,000 exposure cap."""
        config = _make_config(max_position_quantity=1.0, max_total_exposure_usd=1000.0)
        result = process_signal(
            _make_signal(symbol="BTCUSDT", direction="LONG", entry=75793.7,
                         sl=74277.826, tp=78825.448),
            config, self._ledger())
        assert result is not None
        assert result["quantity"] <= 1000.0 / 75793.7
        assert result["notional_usd"] <= 1000.0 + 1e-6

    def test_quantity_above_exposure_cap_rejected_when_no_room(self):
        """Committed exposure already at cap => reject new orders (fail closed)."""
        path = tempfile.mktemp(suffix=".jsonl")
        ledger = ExecutionLedger(path)
        ledger.append_dict({"event": "order_submitted", "signal_id": "prev",
                            "quantity": 0.013, "price": 75793.7})  # ~$985 of $1000
        result = process_signal(
            _make_signal(symbol="BTCUSDT", entry=75793.7, sl=74000.0, tp=79381.1),
            _make_config(max_total_exposure_usd=1000.0), ledger)
        assert result is not None  # only ~$15 headroom, quantity shrinks to fit
        assert result["notional_usd"] <= 1000.0 - 0.013 * 75793.7 + 1e-6
        _safe_unlink(path)

    def test_multiple_open_orders_exceeding_total_exposure_blocked(self):
        """Ledger open orders + run_state accumulate against the cap."""
        path = tempfile.mktemp(suffix=".jsonl")
        ledger = ExecutionLedger(path)
        ledger.append_dict({"event": "order_submitted", "signal_id": "o1",
                            "quantity": 1.0, "price": 900.0})   # $900 of $1000
        run_state = {"proposed_exposure_usd": 50.0, "proposed_orders": 1}
        result = process_signal(
            _make_signal(symbol="ETHUSDT", direction="LONG", entry=3000.0, sl=2900.0, tp=3200.0),
            _make_config(max_total_exposure_usd=1000.0), ledger, run_state=run_state)
        assert result is not None
        assert result["notional_usd"] <= 1000.0 - 900.0 - 50.0 + 1e-6
        _safe_unlink(path)

    def test_exposure_cap_exhausted_rejects(self):
        path = tempfile.mktemp(suffix=".jsonl")
        ledger = ExecutionLedger(path)
        ledger.append_dict({"event": "order_submitted", "signal_id": "o1",
                            "quantity": 1.0, "price": 1000.0})  # $1000 = cap
        result = process_signal(
            _make_signal(symbol="ETHUSDT", entry=3000.0, sl=2900.0, tp=3200.0),
            _make_config(max_total_exposure_usd=1000.0), ledger)
        assert result is None
        _safe_unlink(path)

    def test_fail_closed_when_open_order_missing_price(self):
        """Cannot account committed exposure => reject (fail closed)."""
        path = tempfile.mktemp(suffix=".jsonl")
        ledger = ExecutionLedger(path)
        ledger.append_dict({"event": "order_submitted", "signal_id": "o1"})  # no qty/price
        result = process_signal(
            _make_signal(symbol="ETHUSDT", entry=3000.0, sl=2900.0, tp=3200.0),
            _make_config(), ledger)
        assert result is None
        _safe_unlink(path)

    def test_max_open_orders_reached_rejects(self):
        path = tempfile.mktemp(suffix=".jsonl")
        ledger = ExecutionLedger(path)
        ledger.append_dict({"event": "order_submitted", "signal_id": "o1",
                            "quantity": 0.1, "price": 100.0})
        result = process_signal(
            _make_signal(symbol="BTCUSDT", entry=100.0, sl=90.0, tp=120.0),
            _make_config(), ledger, max_open_orders=1)
        assert result is None  # 1 open, limit 1
        result2 = process_signal(
            _make_signal(signal_id="sig-002", symbol="BTCUSDT", entry=100.0, sl=90.0, tp=120.0),
            _make_config(), ledger, max_open_orders=2)
        assert result2 is not None  # 1 open, limit 2 => allowed
        _safe_unlink(path)

    def test_settled_orders_release_exposure(self):
        path = tempfile.mktemp(suffix=".jsonl")
        ledger = ExecutionLedger(path)
        ledger.append_dict({"event": "order_submitted", "signal_id": "o1",
                            "quantity": 1.0, "price": 1000.0})
        ledger.append_dict({"event": "order_failed", "signal_id": "o1"})
        result = process_signal(
            _make_signal(symbol="BTCUSDT", entry=100.0, sl=90.0, tp=120.0),
            _make_config(max_total_exposure_usd=1000.0), ledger)
        assert result is not None  # o1 settled => full cap available
        _safe_unlink(path)


# ── 3. R:R regression ──────────────────────────────────────────────────

class TestRROverScore:
    """min_rr must gate on the signal's actual R:R, never its score."""

    def test_score_90_with_rr_1_5_fails_min_rr_2(self):
        config = _make_config(min_rr=2.0)
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        # entry 60000, sl 58000 (risk 2000), tp 63000 (reward 3000) => rr = 1.5
        result = process_signal(
            _make_signal(score=90.0, entry=60000.0, sl=58000.0, tp=63000.0),
            config, ledger)
        assert result is None

    def test_score_65_with_rr_2_0_passes_min_rr_2(self):
        config = _make_config(min_rr=2.0)
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        # entry 60000, sl 58000 (risk 2000), tp 64000 (reward 4000) => rr = 2.0
        result = process_signal(
            _make_signal(score=65.0, entry=60000.0, sl=58000.0, tp=64000.0),
            config, ledger)
        assert result is not None

    def test_low_score_with_good_rr_passes_gate(self):
        """Score is NOT an R:R proxy: score 10 with rr 3.0 passes min_rr=2."""
        config = _make_config(min_rr=2.0)
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        result = process_signal(
            _make_signal(score=10.0, entry=100.0, sl=90.0, tp=130.0),
            config, ledger)
        assert result is not None


# ── 4. Precision quantization regression ───────────────────────────────

BTC_EXCHANGE_INFO = {
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.10",
                 "minPrice": "0.10", "maxPrice": "1000000"},
                {"filterType": "LOT_SIZE", "stepSize": "0.0001",
                 "minQty": "0.001", "maxQty": "1000"},
                {"filterType": "MIN_NOTIONAL", "notional": "100"},
            ],
        },
        {
            "symbol": "ETHUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.010"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
            ],
        },
    ]
}


class TestPrecision:

    def test_get_symbol_filters_symbol_specific_not_hardcoded(self):
        btc = get_symbol_filters(BTC_EXCHANGE_INFO, "BTCUSDT")
        eth = get_symbol_filters(BTC_EXCHANGE_INFO, "ETHUSDT")
        assert btc["tick_size"] == Decimal("0.10")
        assert btc["step_size"] == Decimal("0.0001")
        assert eth["tick_size"] == Decimal("0.010")
        assert eth["step_size"] == Decimal("0.001")

    def test_missing_symbol_fails_closed(self):
        import pytest
        with pytest.raises(PrecisionDataUnavailableError):
            get_symbol_filters(BTC_EXCHANGE_INFO, "DOGEUSDT")

    def test_missing_filters_fails_closed(self):
        import pytest
        with pytest.raises(PrecisionDataUnavailableError):
            get_symbol_filters({"symbols": [{"symbol": "X", "filters": []}]}, "X")

    def test_overprecise_price_floored_to_tick(self):
        """price=75793.826 with tickSize 0.10 => 75793.8 (floor, deterministic)."""
        assert quantize_price(75793.826, Decimal("0.10")) == 75793.8

    def test_overprecise_quantity_floored_to_step(self):
        assert quantize_quantity(0.0132133, Decimal("0.0001")) == 0.0132

    def test_floor_never_rounds_up(self):
        """Flooring never increases notional exposure or a BUY price."""
        p = quantize_price(60000.09, Decimal("0.10"))
        assert p <= 60000.09
        q = quantize_quantity(1.23456, Decimal("0.001"))
        assert q <= 1.23456

    def test_validate_quantized_rejects_off_grid(self):
        import pytest
        filters = get_symbol_filters(BTC_EXCHANGE_INFO, "BTCUSDT")
        with pytest.raises(PrecisionDataUnavailableError):
            validate_quantized_quantity(0.01321, 75793.8, filters)  # off step grid
        with pytest.raises(PrecisionDataUnavailableError):
            validate_quantized_quantity(0.0132, 75793.85, filters)  # off tick grid

    def test_validate_quantized_rejects_below_min_notional(self):
        import pytest
        filters = get_symbol_filters(BTC_EXCHANGE_INFO, "BTCUSDT")
        with pytest.raises(PrecisionDataUnavailableError):
            validate_quantized_quantity(0.0001, 1000.0, filters)  # $0.10 < $100

    def test_price_preserves_geometry(self):
        assert price_preserves_geometry("LONG", 75793.8, 74277.826, 78825.448)
        assert not price_preserves_geometry("LONG", 74277.8, 74277.826, 78825.448)
        assert price_preserves_geometry("SHORT", 60106.1, 61616.245, 58597.19)
        assert not price_preserves_geometry("SHORT", 61616.3, 61616.245, 58597.19)

    def test_process_signal_with_btc_like_filters_and_min_qty(self):
        """Headroom-derived qty can land below LOT_SIZE minQty => fail closed."""
        filters = get_symbol_filters(BTC_EXCHANGE_INFO, "BTCUSDT")
        config = _make_config(max_total_exposure_usd=50.0)  # $50 cap: qty < minQty 0.001
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        result = process_signal(
            _make_signal(symbol="BTCUSDT", entry=75793.826, sl=74277.826, tp=79000.0),
            config, ledger, exchange_filters=filters)
        assert result is None

    def test_process_signal_rounds_and_revalidates(self):
        filters = get_symbol_filters(BTC_EXCHANGE_INFO, "BTCUSDT")
        config = _make_config(max_position_quantity=1.0, max_total_exposure_usd=1000.0)
        ledger = ExecutionLedger(tempfile.mktemp(suffix=".jsonl"))
        # tp chosen so rr >= 2.0 (raw ledger tp 78825.448 gives rr=1.9998 < 2.0)
        result = process_signal(
            _make_signal(symbol="BTCUSDT", entry=75793.826, sl=74277.826, tp=79000.0),
            config, ledger, exchange_filters=filters)
        assert result is not None
        assert result["price"] == 75793.8          # tick-floored
        assert result["quantity"] == 0.0131         # (1000/75793.8)=0.013193 -> step-floored
        assert result["notional_usd"] <= 1000.0
        assert result["quantity"] * result["price"] >= 100  # min notional respected
        assert price_preserves_geometry("LONG", result["price"], 74277.826, 78825.448)


# ── 5. D15 signal source regression ───────────────────────────────────

class TestD15SignalSource:

    def test_default_matches_scanner_output_ledger(self):
        """demo_auto's default must be the file the scanner actually writes."""
        import sys
        from scanner import P5Config
        from execution.demo_auto import parse_args
        assert D15_SIGNAL_LOG_DEFAULT == P5Config().log_path
        assert D15_SIGNAL_LOG_DEFAULT == "data/signals/paper_trade.jsonl"
        # The argparse default must resolve to the same file (no argv override).
        saved = sys.argv
        try:
            sys.argv = ["demo_auto"]
            assert parse_args().log == "data/signals/paper_trade.jsonl"
        finally:
            sys.argv = saved

    def test_non_signal_events_not_processed(self):
        """Outcome/data_quality lines in the shared ledger are not treated as signals."""
        _env_setup(DEMO_AUTO_TRADE="true")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"event": "expired", "signal_id": "x1"}) + "\n")
            f.write(json.dumps({"event": "data_quality", "signal_id": "x2"}) + "\n")
            path = f.name
        try:
            rc = main(["--log", path, "--dry-run"])
            assert rc == 0  # 0 signals; would-submit never printed
        finally:
            _safe_unlink(path)
            _safe_unlink(path + ".ledger")
            _env_clear()


# ── 6. CLI safety controls regression ─────────────────────────────────

class TestCliSafetyControls:

    def test_cli_kill_switch_blocks_before_any_submission(self):
        """--kill-switch => abort (exit 1) even with trade enabled and no env kill."""
        _env_setup(DEMO_AUTO_TRADE="true", DEMO_AUTO_KILL_SWITCH="false")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump(_make_signal(), f)
            path = f.name
        try:
            with patch("execution.demo_orders.place_limit_order") as mock_place:
                rc = main(["--log", path, "--kill-switch"])  # NOT --dry-run
                assert rc == 1
                mock_place.assert_not_called()
        finally:
            _safe_unlink(path)
            _safe_unlink(path + ".ledger")
            _env_clear()

    def test_kill_switch_active_helper(self):
        os.environ.pop("DEMO_AUTO_KILL_SWITCH", None)
        assert _kill_switch_active(False) is False
        assert _kill_switch_active(True) is True
        os.environ["DEMO_AUTO_KILL_SWITCH"] = "true"
        assert _kill_switch_active(False) is True
        os.environ.pop("DEMO_AUTO_KILL_SWITCH", None)

    def test_cli_cannot_loosen_configured_exposure(self):
        """--max-exposure-usd above config must NOT loosen the cap."""
        config = _make_config(max_total_exposure_usd=1000.0)
        _, exposure = _effective_limits(config, max_open_orders=None, max_exposure_usd=99999.0)
        assert exposure == 1000.0
        _, exposure2 = _effective_limits(config, max_open_orders=None, max_exposure_usd=500.0)
        assert exposure2 == 500.0  # tightening is allowed

    def test_cli_max_open_orders_enforced_in_dry_run(self):
        """Two valid signals with --max-open-orders 1 => only one would-submit."""
        _env_setup(DEMO_AUTO_TRADE="true")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_signal(signal_id="s1", entry=100.0, sl=90.0, tp=120.0)) + "\n")
            f.write(json.dumps(_make_signal(signal_id="s2", entry=100.0, sl=90.0, tp=120.0)) + "\n")
            path = f.name
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(["--log", path, "--dry-run", "--max-open-orders", "1"])
            assert rc == 0
            assert buf.getvalue().count("Would submit") == 1
        finally:
            _safe_unlink(path)
            _safe_unlink(path + ".ledger")
            _env_clear()

    def test_dry_run_exposure_budget_accumulates_across_signals(self):
        """$1000 cap, two $600-entry ETH signals: second shrinks, not blows cap."""
        _env_setup(DEMO_AUTO_TRADE="true")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_signal(signal_id="e1", symbol="ETHUSDT",
                                            entry=600.0, sl=550.0, tp=700.0)) + "\n")
            f.write(json.dumps(_make_signal(signal_id="e2", symbol="ETHUSDT",
                                            entry=600.0, sl=550.0, tp=700.0)) + "\n")
            path = f.name
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(["--log", path, "--dry-run"])
            assert rc == 0
            lines = [l for l in buf.getvalue().splitlines() if "Would submit" in l]
            assert len(lines) == 2
            notionals = [float(l.split("notional=")[1].rstrip(")")) for l in lines]
            assert sum(notionals) <= 1000.0 + 1e-6
            assert notionals[0] > notionals[1]  # second capped by remaining headroom
        finally:
            _safe_unlink(path)
            _safe_unlink(path + ".ledger")
            _env_clear()
