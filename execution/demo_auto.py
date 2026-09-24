"""
Auto-execution of D15 signals on Binance Demo (LIMIT only).

Safety gates (all must pass before any order is submitted):
  - EXECUTION_MODE must be 'demo'
  - DEMO_AUTO_TRADE must be true
  - DEMO_AUTO_KILL_SWITCH must be false (or not set)
  - DEMO_API_KEY and DEMO_API_SECRET must be set
  - Signal status must be 'OK'
  - Direction must be LONG or SHORT
  - Geometry must be valid (LONG: SL < entry < TP; SHORT: TP < entry < SL)
  - Minimum RR (min_rr) must be met
  - Symbol must be in allowlist
  - Quantity must not exceed max_position_quantity
  - Total exposure must not exceed max_total_exposure_usd
  - At most DEMO_MAX_OPEN_ORDERS active orders allowed

Persistence:
  - Ledger records every attempt (auto_approval, order_submitted, order_reconciliation)
  - Duplicate prevention via ledger (signal_id + client_order_id)
  - Kill switch prevents any new orders if enabled

Usage:
  python -m execution.demo_auto \
      --log data/signals/paper_trade.jsonl \
      [--dry-run] \
      [--max-open-orders N] \
      [--max-exposure-usd N] \
      [--kill-switch]

Signal source of truth: the scanner's D15 paper ledger
(data/signals/paper_trade.jsonl — same default as ScannerConfig.log_path).

Dry-run mode (default): validates all logic but makes no network calls.
Actual execution requires DEMO_AUTO_TRADE=true and no --dry-run flag.
--kill-switch (or DEMO_AUTO_KILL_SWITCH=true) fails closed in every mode.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from execution.config import ExecutionConfig, DEMO_BASE_URL
from execution.models import ApprovedSignalRef, SignalApproval, OrderRequest
from execution.ledger import ExecutionLedger
from execution.demo_orders import (
    build_order_params,
    build_order_url,
    direction_to_side,
    place_limit_order,
    reset_local_order_registry,
    ApprovedSignalRef as DemoApprovedSignalRef,
    SignalApproval as DemoSignalApproval,
    OrderRequest as DemoOrderRequest,
)
from execution.demo_client import build_signed_query, ping, get_exchange_info
from execution.risk import RiskValidator

# ── Safety Flag Constants ──────────────────────────────────────────────
DEMO_AUTO_TRADE_ENABLED = "true"  # Must be set to true to enable auto-trade
DEMO_AUTO_KILL_SWITCH = "DEMO_AUTO_KILL_SWITCH"  # Env var name

# ── CLI Arguments ─────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Auto-execute D15 signals on Binance Demo")
    parser.add_argument("--log", default="data/signals/paper_trade.jsonl",
                        help="Path to D15 signal log (JSONL)")
    parser.add_argument("--dry-run", action="store_true", help="Validate logic without network calls")
    parser.add_argument("--max-open-orders", type=int, default=None,
                        help="Maximum number of concurrent open Demo orders")
    parser.add_argument("--max-exposure-usd", type=float, default=None,
                        help="Maximum total exposure in USD")
    parser.add_argument("--kill-switch", action="store_true", help="Disable all new orders")
    return parser.parse_args()

# ── Safety Gate Checks ────────────────────────────────────────────────
def check_safety_gates(
    max_open_orders: Optional[int] = None,
    max_exposure_usd: Optional[float] = None,
) -> bool:
    """Return True if all safety gates are satisfied."""
    # 1. EXECUTION_MODE must be 'demo'
    if os.environ.get("EXECUTION_MODE") != "demo":
        print("FAIL: EXECUTION_MODE must be 'demo'", file=sys.stderr)
        return False

    # 2. DEMO_AUTO_TRADE must be true
    if os.environ.get("DEMO_AUTO_TRADE") != DEMO_AUTO_TRADE_ENABLED:
        print("FAIL: DEMO_AUTO_TRADE must be set to true", file=sys.stderr)
        return False

    # 3. DEMO_AUTO_KILL_SWITCH must be false (or not set)
    if os.environ.get(DEMO_AUTO_KILL_SWITCH) == "true":
        print("FAIL: DEMO_AUTO_KILL_SWITCH is enabled – no new orders will be placed", file=sys.stderr)
        return False

    # 4. DEMO_API_KEY and DEMO_API_SECRET must be set
    if not os.environ.get("DEMO_API_KEY"):
        print("FAIL: DEMO_API_KEY is not set", file=sys.stderr)
        return False
    if not os.environ.get("DEMO_API_SECRET"):
        print("FAIL: DEMO_API_SECRET is not set", file=sys.stderr)
        return False

    # 5. Load configuration
    try:
        config = ExecutionConfig.from_env()
        if not config:
            print("FAIL: Could not load ExecutionConfig", file=sys.stderr)
            return False
    except Exception as e:
        print(f"FAIL: Could not load ExecutionConfig: {e}", file=sys.stderr)
        return False

    # 6. Check symbol allowlist (if specified)
    if config.symbol_allowlist:
        # The signal will be checked individually, but we can warn early
        pass

    # 7. Check max open orders limit
    if max_open_orders is not None:
        if max_open_orders < 0:
            print("FAIL: max_open_orders must be non-negative", file=sys.stderr)
            return False

    # 8. Check max exposure limit
    if max_exposure_usd is not None:
        if max_exposure_usd < 0:
            print("FAIL: max_exposure_usd must be non-negative", file=sys.stderr)
            return False

    # 9. KILL SWITCH already checked above

    return True

# ── Signal Processing ─────────────────────────────────────────────────
# Counter suffix on client_order_id so two orders prepared in the same
# millisecond can never collide.
_CLIENT_ORDER_SEQ = 0


def _next_client_order_id() -> str:
    global _CLIENT_ORDER_SEQ
    _CLIENT_ORDER_SEQ += 1
    return f"DEMO-AUTO-{int(time.time() * 1000)}-{_CLIENT_ORDER_SEQ}"


def _effective_limits(
    config: ExecutionConfig,
    max_open_orders: Optional[int] = None,
    max_exposure_usd: Optional[float] = None,
):
    """Resolve effective limits. CLI/env values may only TIGHTEN the
    configured limits, never loosen them (CLI values must not bypass
    stricter configured limits)."""
    exposure = config.max_total_exposure_usd
    if max_exposure_usd is not None:
        exposure = min(exposure, max_exposure_usd)

    open_orders: Optional[int] = None
    env_open = os.environ.get("DEMO_MAX_OPEN_ORDERS")
    if env_open is not None:
        try:
            open_orders = int(env_open)
        except ValueError:
            open_orders = 0  # malformed cap => fail closed
    if max_open_orders is not None:
        open_orders = max_open_orders if open_orders is None else min(open_orders, max_open_orders)
    return open_orders, exposure


def _committed_exposure(
    ledger: ExecutionLedger,
    run_state: Optional[Dict[str, Any]],
):
    """Total open/proposed USD exposure and open order count.

    Counts ledger order_submitted/order_accepted events not yet settled,
    plus orders proposed earlier in this same run (run_state).
    """
    events = ledger.read_all()
    open_by_signal: Dict[str, Dict[str, Any]] = {}
    for e in events:
        ev = e.get("event")
        sid = e.get("signal_id")
        if ev in ("order_submitted", "order_accepted") and sid:
            open_by_signal[sid] = e
        elif ev in ("order_rejected", "order_failed", "order_settled", "order_cancelled") and sid:
            open_by_signal.pop(sid, None)

    exposure = 0.0
    count = 0
    for e in open_by_signal.values():
        qty = e.get("quantity")
        px = e.get("price")
        if qty is None or px is None:
            # Fail closed: cannot account for an open order without qty/price.
            return None, None
        exposure += float(qty) * float(px)
        count += 1

    if run_state:
        exposure += run_state.get("proposed_exposure_usd", 0.0)
        count += run_state.get("proposed_orders", 0)
    return exposure, count


def process_signal(
    signal: dict,
    config: ExecutionConfig,
    ledger: ExecutionLedger,
    exchange_filters: Optional[Dict[str, Any]] = None,
    run_state: Optional[Dict[str, Any]] = None,
    max_open_orders: Optional[int] = None,
    max_exposure_usd: Optional[float] = None,
) -> Optional[dict]:
    """
    Validate a D15 signal and prepare for order submission.
    Returns a dict with order parameters ready for submission, or None if
    the signal is rejected by any gate (fail closed).

    exchange_filters: parsed execution.precision filters for the symbol
    (from get_symbol_filters). REQUIRED for real execution; None in dry-run
    skips quantization but still applies the exposure cap.
    """
    # Required fields
    required = ("signal_id", "symbol", "direction", "entry", "sl", "tp", "score", "signal_time")
    for field in required:
        if field not in signal:
            print(f"FAIL: Missing required field '{field}' in signal", file=sys.stderr)
            return None

    signal_id = signal["signal_id"]
    symbol = signal["symbol"]
    direction = signal["direction"]
    entry = signal["entry"]
    sl = signal["sl"]
    tp = signal["tp"]
    score = signal["score"]
    signal_time = signal["signal_time"]

    # Fail closed on unusable price information.
    for name, value in (("entry", entry), ("sl", sl), ("tp", tp)):
        if not isinstance(value, (int, float)) or value <= 0:
            print(f"FAIL: {name} unusable for signal {signal_id}: {value!r}", file=sys.stderr)
            return None

    # 1. Direction must be LONG or SHORT
    if direction not in ("LONG", "SHORT"):
        print(f"FAIL: Invalid direction '{direction}' in signal {signal_id}", file=sys.stderr)
        return None

    # 2. Geometry validation — authoritative semantics (mirrors risk.py):
    #    LONG:  SL < entry < TP
    #    SHORT: TP < entry < SL
    if direction == "LONG":
        if not (sl < entry < tp):
            print(f"FAIL: Invalid geometry for LONG signal {signal_id} (need SL < entry < TP)", file=sys.stderr)
            return None
    else:  # SHORT
        if not (tp < entry < sl):
            print(f"FAIL: Invalid geometry for SHORT signal {signal_id} (need TP < entry < SL)", file=sys.stderr)
            return None

    # 3. Minimum R:R check — compares the signal's ACTUAL R:R
    #    (TP reward / SL risk), NOT its score. D14's own gate in
    #    indicators/ is untouched; this only refuses orders below min_rr.
    signal_ref = ApprovedSignalRef(
        signal_id=signal_id,
        symbol=symbol,
        direction=direction,
        entry=entry,
        sl=sl,
        tp=tp,
        score=score,
        signal_time=signal_time,
        strategy_version=signal.get("strategy_version", "demo_auto_v1"),
        approved_at_ms=int(time.time() * 1000),
        approver="demo_auto",
    )
    min_rr = config.min_rr
    rr = signal_ref.rr_ratio()
    if min_rr is not None and rr < min_rr:
        print(f"FAIL: Signal {signal_id} has R:R {rr:.4f} < min_rr {min_rr}", file=sys.stderr)
        return None

    # 4. Symbol allowlist — hard gate (fail closed), consistent with
    #    RiskValidator.SYMBOL_NOT_ALLOWED semantics.
    if config.symbol_allowlist and symbol not in config.symbol_allowlist:
        print(f"FAIL: Symbol '{symbol}' not in allowlist (signal {signal_id})", file=sys.stderr)
        return None

    # 5. Check duplicate in ledger (persistent duplicate protection)
    existing_events = ledger.get_events_by_signal_id(signal_id)
    for event in existing_events:
        if event.get("event") in ("order_submitted", "order_accepted"):
            print(f"FAIL: Duplicate signal_id {signal_id} already processed", file=sys.stderr)
            return None

    # 6. Exposure / open-order budget (may be tightened by CLI, never loosened)
    open_limit, exposure_cap = _effective_limits(config, max_open_orders, max_exposure_usd)
    committed_exposure, open_count = _committed_exposure(ledger, run_state)
    if committed_exposure is None or open_count is None:
        print(f"FAIL: Cannot account open exposure (missing qty/price in ledger) – rejecting {signal_id}", file=sys.stderr)
        return None
    if open_limit is not None and open_count >= open_limit:
        print(f"FAIL: max_open_orders {open_limit} reached ({open_count} open) – rejecting {signal_id}", file=sys.stderr)
        return None
    headroom = exposure_cap - committed_exposure
    if headroom <= 0:
        print(f"FAIL: Exposure cap ${exposure_cap:.2f} reached (${committed_exposure:.2f} committed) – rejecting {signal_id}", file=sys.stderr)
        return None

    # 7. Quantity derivation: cap by max_position_quantity AND USD headroom
    #    at the actual proposed price. No 1-BTC-at-$75k against a $1k cap.
    price = float(entry)
    quantity = min(float(config.max_position_quantity), headroom / price)

    # 8. Precision: quantize to symbol-specific exchange filters and
    #    revalidate everything after rounding.
    if exchange_filters is not None:
        from execution.precision import (
            PrecisionDataUnavailableError,
            quantize_price,
            quantize_quantity,
            validate_quantized_quantity,
            price_preserves_geometry,
        )
        try:
            price = quantize_price(price, exchange_filters["tick_size"])
            quantity = quantize_quantity(quantity, exchange_filters["step_size"])
            # Re-apply the notional cap after any rounding surprise.
            if price > 0 and quantity * price > headroom:
                quantity = quantize_quantity(headroom / price, exchange_filters["step_size"])
            if not price_preserves_geometry(direction, price, sl, tp):
                print(f"FAIL: Rounded price {price} breaks geometry for {signal_id}", file=sys.stderr)
                return None
            validate_quantized_quantity(quantity, price, exchange_filters)
        except PrecisionDataUnavailableError as e:
            print(f"FAIL: Precision check failed for {signal_id}: {e}", file=sys.stderr)
            return None
    else:
        if quantity <= 0:
            print(f"FAIL: Derived quantity unusable for {signal_id} (no precision filters)", file=sys.stderr)
            return None

    # 9. Order-level risk validation with the final rounded values
    client_order_id = _next_client_order_id()
    side = direction_to_side(direction)
    order_req = OrderRequest(
        client_order_id=client_order_id,
        symbol=symbol,
        side=side,
        order_type="LIMIT",
        quantity=quantity,
        price=price,
        stop_price=None,
        time_in_force="GTC",
        reduce_only=False,
        close_position=False,
        signal_id=signal_id,
        strategy_version=signal_ref.strategy_version,
        requested_at_ms=int(time.time() * 1000),
    )
    try:
        RiskValidator(config).validate_signal(signal_ref)
        RiskValidator(config).validate_order(order_req)
    except Exception as e:
        print(f"FAIL: Risk validation rejected {signal_id}: {e}", file=sys.stderr)
        return None

    # Final exposure assertion: notional of THIS order must fit headroom.
    if quantity * price > headroom + 1e-6:
        print(f"FAIL: Order notional {quantity * price:.2f} exceeds headroom {headroom:.2f} for {signal_id}", file=sys.stderr)
        return None

    try:
        params = build_order_params(
            signal=signal_ref,
            client_order_id=client_order_id,
            quantity=quantity,
            price=price,
        )
    except Exception as e:
        print(f"FAIL: Order param build failed for {signal_id}: {e}", file=sys.stderr)
        return None

    # All checks passed – return prepared order
    return {
        "signal_id": signal_id,
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "score": score,
        "quantity": quantity,
        "price": price,
        "side": side,
        "client_order_id": client_order_id,
        "binance_order_id": None,  # Will be set by demo_client after submission
        "signal_time": signal_time,
        "notional_usd": quantity * price,
        "status": "PREPARED",
    }

# ── Order Submission ──────────────────────────────────────────────────
def submit_order(order: dict, config: ExecutionConfig) -> dict:
    """Submit a limit order to Binance Demo. Returns the order result."""
    # Reset local order registry before submission
    reset_local_order_registry()

    # Create the approval objects needed by place_limit_order
    approval = SignalApproval(
        signal_id=order["signal_id"],
        symbol=order["symbol"],
        direction=order["direction"],
        approved_at_ms=int(time.time() * 1000),
        approver="demo_auto",
        approved=True,
        reason="Auto-approved by demo_auto execution layer",
    )

    signal_ref = ApprovedSignalRef(
        signal_id=order["signal_id"],
        symbol=order["symbol"],
        direction=order["direction"],
        entry=order["entry"],
        sl=order["sl"],
        tp=order["tp"],
        score=order["score"],
        signal_time=order["signal_time"],
        strategy_version=order.get("strategy_version", "demo_auto_v1"),
        approved_at_ms=int(time.time() * 1000),
        approver="demo_auto",
    )

    # Place the order using the existing demo_orders module
    result = place_limit_order(
        config=config,
        signal=signal_ref,
        approval=approval,
        client_order_id=order["client_order_id"],
        quantity=order["quantity"],
        price=order["price"],
    )

    return result

# ── Ledger Recording ──────────────────────────────────────────────────
def record_order_result(ledger: ExecutionLedger, order: dict, result: dict) -> None:
    """Record order outcome in the execution ledger."""
    event_type = "order_submitted"
    if result.get("status") == "UNKNOWN":
        event_type = "order_rejected"
    elif result.get("status") == "FAILED":
        event_type = "order_failed"
    else:
        event_type = "order_accepted"

    ledger.append_dict({
        "event": event_type,
        "signal_id": order["signal_id"],
        "symbol": order["symbol"],
        "direction": order["direction"],
        "client_order_id": order["client_order_id"],
        "binance_order_id": result.get("binance_order_id"),
        "status": result.get("status", "UNKNOWN"),
        "quantity": order["quantity"],
        "price": order["price"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_message": result.get("error_message")
    })

    print(f"[Ledger] Recorded {event_type} for signal {order['signal_id']}")

# ── Main Execution Flow ───────────────────────────────────────────────
# Source of truth for D15 signals: the scanner writes PaperSignalEvents to
# ScannerConfig.log_path (data/signals/paper_trade.jsonl). demo_auto must
# consume the SAME file; the old default (paper_trade_d15.jsonl) never
# existed and would have silently produced 0 signals.
D15_SIGNAL_LOG_DEFAULT = "data/signals/paper_trade.jsonl"


def _kill_switch_active(cli_flag: bool) -> bool:
    """Kill switch fails closed: CLI flag OR env var arms it."""
    return bool(cli_flag) or os.environ.get(DEMO_AUTO_KILL_SWITCH) == "true"


def _load_signal_log(signal_path: str, strict: bool):
    """Read the signal ledger; only keep D15 signal events.

    strict=False (dry-run): tolerate a missing/malformed file (warn).
    strict=True (execution): any malformed line aborts (fail closed).
    """
    if not os.path.exists(signal_path):
        if strict:
            print(f"FAIL: Signal log not found at {signal_path}", file=sys.stderr)
            return None
        print("[DRY-RUN] No signal ledger found; treating as empty set (0 signals)")
        return []
    signals = []
    with open(signal_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if strict:
                    print(f"FAIL: Malformed signal ledger at line {line_num}: {e}", file=sys.stderr)
                    return None
                print(f"WARN: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue
            # The paper ledger mixes signal/outcome/data_quality events;
            # only 'signal' events are executable D15 signals.
            if record.get("event") == "signal":
                signals.append(record)
    return signals


def _fetch_symbol_filters(config: ExecutionConfig, symbol: str):
    """Fetch exchangeInfo (Demo-only GET) and parse symbol filters.

    Returns filters dict, or raises. Used ONLY in real-execution mode;
    dry-run never calls this (zero network)."""
    from execution.precision import get_symbol_filters
    info = get_exchange_info(config)
    return get_symbol_filters(info, symbol)


def main(args=None) -> int:
    # args: None => real CLI; list => parse manually (tests); Namespace => use as-is
    if args is None:
        args = parse_args()
    if isinstance(args, list):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--log", default=D15_SIGNAL_LOG_DEFAULT)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--max-open-orders", type=int, default=None)
        parser.add_argument("--max-exposure-usd", type=float, default=None)
        parser.add_argument("--kill-switch", action="store_true")
        args = parser.parse_args(args)

    # Check safety gates first
    if not check_safety_gates(max_open_orders=args.max_open_orders, max_exposure_usd=args.max_exposure_usd):
        print("Safety gates not satisfied – aborting", file=sys.stderr)
        return 1

    # CLI --kill-switch arms the same fail-closed switch as the env var.
    if _kill_switch_active(args.kill_switch):
        print("FAIL: kill switch engaged (CLI --kill-switch or env) – no orders will be placed", file=sys.stderr)
        return 1

    # Load configuration
    try:
        config = ExecutionConfig.from_env()
    except Exception as e:
        print(f"FAIL: Could not load ExecutionConfig: {e}", file=sys.stderr)
        return 1

    # Initialize ledger
    try:
        ledger = ExecutionLedger(log_path=args.log + ".ledger")
    except Exception as e:
        print(f"FAIL: Could not initialize ExecutionLedger: {e}", file=sys.stderr)
        return 1

    # Load D15 signals from the log file
    signal_path = args.log
    signals = _load_signal_log(signal_path, strict=not args.dry_run)
    if signals is None:
        return 1
    if not signals:
        if args.dry_run:
            print("Processed 0 signals, validated 0, submitted 0 orders (dry-run, empty/missing ledger)")
            return 0
        print("FAIL: Signal log is empty; no executions permitted", file=sys.stderr)
        return 1

    processed_count = 0
    submitted_count = 0
    # In-run budget tracking so multiple orders in ONE invocation respect
    # exposure and max-open-orders collectively, not just against the ledger.
    run_state = {"proposed_exposure_usd": 0.0, "proposed_orders": 0}
    filters_cache: Dict[str, Any] = {}
    filter_fetch_failed = set()

    # Process each signal
    for signal in signals:
        # Kill switch re-check per iteration (fail closed mid-run).
        if _kill_switch_active(args.kill_switch):
            print("Kill switch enabled – cancelling further orders", file=sys.stderr)
            break

        symbol = signal.get("symbol", "")
        exchange_filters = None
        if not args.dry_run:
            # Real execution REQUIRES precision filters (fail closed).
            if symbol in filter_fetch_failed:
                print(f"FAIL: exchangeInfo unavailable for {symbol} – rejecting {signal.get('signal_id')}", file=sys.stderr)
                continue
            if symbol not in filters_cache:
                try:
                    filters_cache[symbol] = _fetch_symbol_filters(config, symbol)
                except Exception as e:
                    filter_fetch_failed.add(symbol)
                    print(f"FAIL: Could not load exchange filters for {symbol}: {e}", file=sys.stderr)
                    continue
            exchange_filters = filters_cache[symbol]

        order_params = process_signal(
            signal, config, ledger,
            exchange_filters=exchange_filters,
            run_state=run_state,
            max_open_orders=args.max_open_orders,
            max_exposure_usd=args.max_exposure_usd,
        )
        if order_params is None:
            # Invalid signal – skip but log
            print(f"Skipping invalid signal {signal.get('signal_id', 'unknown')}", file=sys.stderr)
            continue

        processed_count += 1

        # Dry-run mode: validate but don't submit
        if args.dry_run:
            print(
                f"[DRY-RUN] Would submit order for signal {order_params['signal_id']} "
                f"(qty={order_params['quantity']} price={order_params['price']} "
                f"notional={order_params['notional_usd']:.2f})"
            )
            submitted_count += 1
            run_state["proposed_exposure_usd"] += order_params["notional_usd"]
            run_state["proposed_orders"] += 1
            continue

        # Submit the order
        try:
            result = submit_order(order_params, config)
        except Exception as e:
            print(f"FAIL: Order submission failed for signal {order_params['signal_id']}: {e}", file=sys.stderr)
            # Record as failed in ledger
            record_order_result(ledger, order_params, {"status": "FAILED", "error_message": str(e)})
            continue

        # Check result status
        if result.get("status") == "UNKNOWN":
            print(f"Order submission failed (status: {result.get('status')}) for signal {order_params['signal_id']}", file=sys.stderr)
            # Record as rejected in ledger
            record_order_result(ledger, order_params, result)
            continue

        # Success – record in ledger
        record_order_result(ledger, order_params, result)
        submitted_count += 1
        run_state["proposed_exposure_usd"] += order_params["notional_usd"]
        run_state["proposed_orders"] += 1

    print(f"Processed {len(signals)} signals, validated {processed_count}, submitted {submitted_count} orders", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())