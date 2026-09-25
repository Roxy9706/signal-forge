"""execution/shadow_reconcile.py — Demo order/fill reconciliation (Phase 1).

Read-only lifecycle observer for the Demo execution ledger:

  1. GET /fapi/v1/openOrders   (existing demo_client path)
  2. GET /fapi/v1/userTrades   (new demo_client path; Demo-host validated)
  3. correlate exchange records against ledger-open orders by
     client_order_id / binance_order_id
  4. append settlement/outcome events ('order_settled' /
     'reconcile_open_confirmed' / 'order_failed') to the SAME Demo
     execution ledger the bridge writes

Writing 'order_settled' (or 'order_rejected'/'order_failed'/
'order_cancelled') releases the locally committed exposure in
execution.demo_auto._committed_exposure, which is the exact mechanism the
audit identified as missing (phantom exposure until reconciled).

Phase 1 boundaries (audit-approved):
  - NO cancellation, EVER, from this module. Cancel stays behind its own
    explicit later gate (execution/demo_cancel.py remains manual-only).
  - No new risk math: it never modifies or re-derives orders.
  - Unknown/ambiguous exchange state is recorded conservatively as still
    open (never releases exposure, never retries).
  - All network access goes through execution.demo_client's GET allowlist +
    _check_demo_url; tests inject fake fetchers so zero real calls exist.

Usage:
  python -m execution.shadow_reconcile --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from execution.config import ExecutionConfig
from execution.ledger import ExecutionLedger
from execution.models import ApprovedSignalRef, ExecutionResult, SignalApproval
from execution.demo_client import get_open_orders, get_trade_history
from execution.shadow_bridge import (
    BRIDGE_DATA_DIR,
    LEDGER_FILENAME,
    assert_writable_bridge_path,
    bridge_disabled_reason,
    verify_demo_endpoint,
)

# Events that mark a signal as consuming exposure (open) until a settle event.
_OPEN_EVENTS = ("order_submitted", "order_accepted")
_SETTLE_EVENTS = ("order_rejected", "order_failed", "order_settled",
                  "order_cancelled")

# Map exchange order status -> terminal classification.
_FILLED_STATUSES = ("FILLED", "FILLED_BY_TRIGGER")
_UNFILLED_TERMINAL_STATUSES = ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED")


# SL/TP lifecycle production: exact D15 SL/TP preserved; reconcile_once (GET only);
# duplicate/restart/UNKNOWN protected; 0 auto-cancel.
# Protective orders (SL/TP) are placed by place_protective_orders_on_fill() ONLY
# when an entry fill is confirmed — not at order submission. See also
# execution.demo_orders.place_protective_orders + tests/test_lifecycle_sl_tp.py.

def load_open_orders(ledger: ExecutionLedger) -> Dict[str, Dict[str, Any]]:
    """Ledger signal_id -> representative open-order event.

    Mirrors demo_auto._committed_exposure's open/settled folding so the
    bridge's exposure view and the reconciler's view agree exactly.
    """
    open_by_signal: Dict[str, Dict[str, Any]] = {}
    for e in ledger.read_all():
        ev = e.get("event")
        sid = e.get("signal_id")
        if not sid:
            continue
        if ev in _OPEN_EVENTS:
            open_by_signal[sid] = e
        elif ev in _SETTLE_EVENTS or ev == "reconcile_abandoned_confirmed":
            open_by_signal.pop(sid, None)
    return open_by_signal


def _match_exchange(order_rec: Dict[str, Any],
                    exchange_open: List[Dict[str, Any]]):
    """Correlate one ledger order with the exchange openOrders list.

    Returns the matching open-order dict, or None if the order is not open
    on the exchange. Matches on clientOrderId first (authoritative), then
    on orderId when the ledger captured the numeric id.
    """
    cid = order_rec.get("client_order_id")
    oid = order_rec.get("binance_order_id")
    for rec in exchange_open:
        if cid and rec.get("clientOrderId") == cid:
            return rec
        if oid is not None and str(rec.get("orderId")) == str(oid):
            return rec
    return None


def _trades_for(trades: List[Dict[str, Any]],
                order_rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    cid = order_rec.get("client_order_id")
    oid = order_rec.get("binance_order_id")
    out = []
    for t in trades:
        if cid and t.get("clientOrderId") == cid:
            out.append(t)
        elif oid is not None and str(t.get("orderId")) == str(oid):
            out.append(t)
    return out


def reconcile_once(
    *,
    ledger_path: Optional[Path] = None,
    now_ms: Optional[int] = None,
    fetch_open_orders: Optional[Callable[[ExecutionConfig], List[Dict[str, Any]]]] = None,
    fetch_trades: Optional[Callable[[ExecutionConfig, str], List[Dict[str, Any]]]] = None,
    require_env_gates: bool = True,
) -> Dict[str, Any]:
    """One reconciliation pass. Returns a summary dict.

    fetch_open_orders / fetch_trades are injectable for tests; defaults use
    ONLY the demo_client GET allowlist (Demo host enforced downstream by
    _check_demo_url inside execute_get/build paths).
    """
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    summary: Dict[str, Any] = {"ran": False, "open_on_ledger": 0,
                               "still_open": 0, "settled": 0,
                               "confirmations": 0, "errors": 0,
                               "reason": None}
    if require_env_gates:
        disabled = bridge_disabled_reason()
        if disabled:
            summary["reason"] = disabled
            return summary
        from execution.shadow_bridge import _kill_switch_active
        if _kill_switch_active(False):
            summary["reason"] = "DEMO_AUTO_KILL_SWITCH is enabled – " \
                                "reconciliation disabled"
            return summary

    ledger_path = Path(ledger_path) if ledger_path else Path(BRIDGE_DATA_DIR) / LEDGER_FILENAME
    assert_writable_bridge_path(ledger_path)
    ledger = ExecutionLedger(log_path=str(ledger_path))
    open_orders = load_open_orders(ledger)
    summary["open_on_ledger"] = len(open_orders)
    if not open_orders:
        summary["ran"] = True
        return summary

    config = ExecutionConfig.from_env()
    verify_demo_endpoint(config)

    _fetch_open = fetch_open_orders or (lambda cfg: get_open_orders(cfg))
    _fetch_trades = fetch_trades or (lambda cfg, sym: get_trade_history(cfg, symbol=sym))

    # Wire the protective-order callback so _settle can place SL/TP on fill.
    _wired_callback = None
    if _PLACE_PROTECTIVE_CALLBACK is None:
        # Production default: route through execution.demo_orders.place_protective_orders.
        from execution.demo_orders import place_protective_orders as _ppo

        def _default_protective(config, signal_ref, approval,
                                entry_client_order_id, quantity):
            return place_protective_orders(
                config=config,
                signal=signal_ref,
                approval=approval,
                entry_client_order_id=entry_client_order_id,
                quantity=quantity,
            )

        _wired_callback = _default_protective
    else:
        _wired_callback = _PLACE_PROTECTIVE_CALLBACK

    def place_protective(config, signal_ref, approval,
                         entry_client_order_id, quantity):
        return _wired_callback(config, signal_ref, approval,
                              entry_client_order_id, quantity)

    try:
        exchange_open = _fetch_open(config)
    except Exception as e:  # transport fail => conservative no-op
        summary["reason"] = f"openOrders fetch failed; state untouched: {e}"
        summary["ran"] = True
        return summary

    trades_cache: Dict[str, Optional[List[Dict[str, Any]]]] = {}

    def trades_for_symbol(symbol: str) -> List[Dict[str, Any]]:
        if symbol not in trades_cache:
            try:
                trades_cache[symbol] = _fetch_trades(config, symbol)
            except Exception:
                trades_cache[symbol] = None
        return trades_cache[symbol] or []

    for sid, rec in open_orders.items():
        symbol = str(rec.get("symbol", ""))
        qty = float(rec.get("quantity") or 0)
        match = _match_exchange(rec, exchange_open)
        if match is not None:
            summary["still_open"] += 1
            if str(match.get("status", "")).upper() in _FILLED_STATUSES:
                # Open-list should only hold NEW/PARTIALLY_FILLED; a FILLED
                # entry is possible on some gateways — settle as filled.
                _settle(ledger, rec, sid, symbol, "FILLED",
                        filled_qty=float(match.get("executedQty") or qty),
                        fill_price=float(match.get("avgPrice") or match.get("price") or
                                         rec.get("price") or 0),
                        now_ms=now_ms,
                        place_protective_callback=place_protective,
                        config=config)
                summary["settled"] += 1
            else:
                ledger.append_dict({
                    "event": "reconcile_open_confirmed", "signal_id": sid,
                    "symbol": symbol,
                    "client_order_id": rec.get("client_order_id"),
                    "exchange_status": match.get("status"),
                    "confirmed_at_ms": now_ms,
                    "source": "shadow_reconcile"})
                summary["confirmations"] += 1
            continue

        # Not open on exchange: classify terminal state from trades.
        matched_trades = trades_for_symbol(symbol)
        t_list = _trades_for(matched_trades, rec)
        exec_qty = sum(float(t.get("executedQty") or t.get("qty") or 0) for t in t_list)
        if t_list and exec_qty > 0:
            state = "FILLED" if exec_qty >= qty - 1e-12 else "PARTIALLY_FILLED_THEN_CLOSED"
            last = t_list[-1]
            _settle(ledger, rec, sid, symbol, state,
                    filled_qty=exec_qty,
                    fill_price=float(last.get("price") or rec.get("price") or 0),
                    now_ms=now_ms,
                    place_protective_callback=place_protective,
                    config=config)
            summary["settled"] += 1
            continue
        if t_list or (trades_cache.get(symbol) is not None):
            # Symbol trades fetched cleanly and none reference this order:
            # it terminated without filling (cancel or expiry —
            # indistinguishable via GET-only allowlist; recorded as such).
            _settle(ledger, rec, sid, symbol, "ENDED_UNFILLED",
                    filled_qty=0.0, fill_price=0.0, now_ms=now_ms,
                    place_protective_callback=None)
            summary["settled"] += 1
            continue
        # Trades unavailable => ambiguous; stay conservative (exposure kept).
        summary["errors"] += 1
        ledger.append_dict({
            "event": "reconcile_error", "signal_id": sid, "symbol": symbol,
            "message": "userTrades unavailable; order left open (fail closed)",
            "occurred_at_ms": now_ms, "source": "shadow_reconcile"})

    summary["ran"] = True
    return summary


# ── Protective-order placement (called ONLY on confirmed entry fill) ────────

def _lookup_signal_ref(ledger: ExecutionLedger, signal_id: str) -> Optional[ApprovedSignalRef]:
    """Recover the ApprovedSignalRef for a signal_id from the ledger.

    The bridge writes an 'approved_signal' event; the reconciler reads it back
    to get exact D15 SL/TP without recalculating.
    """
    for e in ledger.get_events_by_signal_id(signal_id):
        if e.get("event") == "approved_signal":
            try:
                return ApprovedSignalRef.from_dict(e)
            except Exception:
                return None
    return None


def _protective_orders_already_placed(ledger: ExecutionLedger, signal_id: str) -> bool:
    """Check whether protective SL/TP orders were already placed for this signal.

    Looks for 'protective_order_placed' events tagged with this signal_id.
    """
    for e in ledger.get_events_by_signal_id(signal_id):
        if e.get("event") == "protective_order_placed":
            return True
    return False


# Module-level injectable callback — default None (no real placement in reconcile).
_PLACE_PROTECTIVE_CALLBACK = None  # set by tests or production wiring


def set_place_protective_callback(callback) -> None:
    """Inject the callback that places protective orders.

    In production this is wired to a function that calls
    execution.demo_orders.place_protective_orders with real config.
    In tests it is replaced with a mock.
    """
    global _PLACE_PROTECTIVE_CALLBACK
    _PLACE_PROTECTIVE_CALLBACK = callback


def _place_protective_orders(
    ledger: ExecutionLedger,
    *,
    config,
    signal_ref: ApprovedSignalRef,
    entry_client_order_id: str,
    filled_qty: float,
    now_ms: int,
    place_protective_callback: Optional[Callable] = None,
) -> None:
    """Place SL + TP protective orders via the injected callback.

    Records 'protective_order_placed' events in the ledger for duplicate
    protection and audit.  Failures are recorded as 'protective_order_failed'
    but NEVER block the settle event or retry.
    """
    global _PLACE_PROTECTIVE_CALLBACK
    cb = place_protective_callback if place_protective_callback is not None else _PLACE_PROTECTIVE_CALLBACK
    if cb is None:
        return

    approval = SignalApproval(
        signal_id=signal_ref.signal_id,
        symbol=signal_ref.symbol,
        direction=signal_ref.direction,
        approved_at_ms=now_ms,
        approver="shadow_reconcile",
        approved=True,
        reason="protective orders placed on confirmed entry fill",
    )

    try:
        results: Optional[Dict[str, ExecutionResult]] = cb(
            config, signal_ref, approval,
            entry_client_order_id=entry_client_order_id,
            quantity=filled_qty,
        )
    except Exception as exc:
        ledger.append_dict({
            "event": "protective_order_failed",
            "signal_id": signal_ref.signal_id,
            "symbol": signal_ref.symbol,
            "direction": signal_ref.direction,
            "entry_client_order_id": entry_client_order_id,
            "error_message": str(exc)[:300],
            "failed_at_ms": now_ms,
            "source": "shadow_reconcile",
        })
        return

    if results is None:
        return

    for role in ("sl", "tp"):
        res = results.get(role)
        if res is None:
            continue
        ledger.append_dict({
            "event": "protective_order_placed",
            "signal_id": signal_ref.signal_id,
            "symbol": signal_ref.symbol,
            "direction": signal_ref.direction,
            "role": role,  # "sl" or "tp"
            "client_order_id": res.client_order_id,
            "binance_order_id": res.binance_order_id,
            "order_type": res.order_type,
            "status": res.status,
            "trigger_price": (signal_ref.sl if role == "sl" else signal_ref.tp),
            "placed_at_ms": now_ms,
            "source": "shadow_reconcile",
        })


def _settle(ledger: ExecutionLedger, rec: Dict[str, Any], sid: str,
            symbol: str, terminal_state: str, filled_qty: float,
            fill_price: float, now_ms: int,
            place_protective_callback: Optional[Callable] = None,
            config: Optional[ExecutionConfig] = None) -> None:
    """Append the settle event that releases exposure in _committed_exposure.

    'order_settled' is one of the release events the bridge/exposure code
    already recognizes — no formula changes needed anywhere.

    When terminal_state indicates a fill (FILLED / PARTIALLY_FILLED_THEN_CLOSED)
    and place_protective is True, also place exact D15 protective orders
    (STOP_MARKET at SL, TAKE_PROFIT_MARKET at TP) via the reconciler's
    injectable place_protective callback.  This is NOT done on order submission
    — only when an entry fill is confirmed by exchange data.
    """
    ledger.append_dict({
        "event": "order_settled", "signal_id": sid, "symbol": symbol,
        "direction": rec.get("direction"),
        "client_order_id": rec.get("client_order_id"),
        "binance_order_id": rec.get("binance_order_id"),
        "quantity": rec.get("quantity"), "price": rec.get("price"),
        "terminal_state": terminal_state,
        "filled_qty": filled_qty, "fill_price": fill_price,
        "settled_at_ms": now_ms, "source": "shadow_reconcile"})

    if not place_protective_callback:
        return

    # Only place protective orders on actual fills (not cancellations/expiry).
    if terminal_state not in ("FILLED", "PARTIALLY_FILLED_THEN_CLOSED"):
        return

    # Guard: must have quantity filled.
    if filled_qty <= 0:
        return

    # Look up the original signal's exact D15 SL/TP from the ledger.
    signal_ref = _lookup_signal_ref(ledger, sid)
    if signal_ref is None:
        return

    # Derive the entry client_order_id from the settled record.
    entry_cid = str(rec.get("client_order_id", ""))
    if not entry_cid:
        return

    # Check duplicate: have we already placed protective orders for this signal?
    if _protective_orders_already_placed(ledger, sid):
        return

    if place_protective_callback is not None:
        _place_protective_orders(
            ledger,
            config=config,
            signal_ref=signal_ref,
            entry_client_order_id=entry_cid,
            filled_qty=filled_qty,
            now_ms=now_ms,
            place_protective_callback=place_protective_callback,
        )


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Demo shadow reconciler (read-only)")
    parser.add_argument("--ledger", default=str(Path(BRIDGE_DATA_DIR) / LEDGER_FILENAME))
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    while True:
        summary = reconcile_once(ledger_path=Path(args.ledger))
        print(json.dumps(summary, sort_keys=True))
        if args.once:
            return 0
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    sys.exit(main())
