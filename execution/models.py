# Execution data models — serializable, no secrets stored.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Explicit approval boundary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalApproval:
    """Explicit approval record for a signal.

    A signal MUST have an approval record before the execution adapter will
    consider it for order preparation. Mere existence in paper_trade_d15.jsonl
    does NOT constitute approval.

    Approval is explicit and auditable: each approval has an approver identity,
    a timestamp, and an optional reason.
    """

    signal_id: str
    symbol: str
    direction: str
    approved_at_ms: int
    approver: str  # e.g. 'manual', 'policy_v1', 'automated_gate_v1'
    reason: str = ""
    approved: bool = False  # NOT True by default — fail closed

    def to_dict(self) -> dict:
        return {
            "event": "approval",
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "approved_at_ms": self.approved_at_ms,
            "approver": self.approver,
            "reason": self.reason,
            "approved": self.approved,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignalApproval":
        return cls(
            signal_id=d.get("signal_id", ""),
            symbol=d.get("symbol", ""),
            direction=d.get("direction", ""),
            approved_at_ms=int(d.get("approved_at_ms", 0)),
            approver=d.get("approver", ""),
            reason=d.get("reason", ""),
            approved=bool(d.get("approved", False)),
        )

    def is_approved(self) -> bool:
        """True only if explicit approval is True."""
        return self.approved is True


# ---------------------------------------------------------------------------
# Approved signal reference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApprovedSignalRef:
    """A paper signal that has passed the approval boundary.

    Minimal reference passed from paper layer to execution layer.
    Does NOT contain API credentials, order parameters, or response data.
    """

    signal_id: str
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    score: float
    signal_time: int
    strategy_version: str
    approved_at_ms: int
    approver: str

    def risk_distance(self) -> float:
        """Absolute distance from entry to SL (positive)."""
        return abs(self.entry - self.sl)

    def rr_ratio(self) -> float:
        """Risk:reward ratio (TP reward / SL risk)."""
        risk = self.risk_distance()
        if risk <= 0:
            return 0.0
        reward = abs(self.tp - self.entry)
        return reward / risk

    def to_dict(self) -> dict:
        return {
            "event": "approved_signal",
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "score": self.score,
            "signal_time": self.signal_time,
            "strategy_version": self.strategy_version,
            "approved_at_ms": self.approved_at_ms,
            "approver": self.approver,
            "rr_ratio": self.rr_ratio(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ApprovedSignalRef":
        return cls(
            signal_id=d.get("signal_id", ""),
            symbol=d.get("symbol", ""),
            direction=d.get("direction", ""),
            entry=float(d.get("entry", 0.0)),
            sl=float(d.get("sl", 0.0)),
            tp=float(d.get("tp", 0.0)),
            score=float(d.get("score", 0.0)),
            signal_time=int(d.get("signal_time", 0)),
            strategy_version=d.get("strategy_version", ""),
            approved_at_ms=int(d.get("approved_at_ms", 0)),
            approver=d.get("approver", ""),
        )


# ---------------------------------------------------------------------------
# Order request (representation only — not yet placed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderRequest:
    """Representation of an order to be submitted to the demo API.

    No order placement is implemented. Data-only representation for audit
    logging and future order-placement wiring.
    """

    client_order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    order_type: str  # "MARKET", "LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"
    quantity: float
    price: Optional[float]  # None for MARKET orders
    stop_price: Optional[float]  # for STOP_MARKET / TAKE_PROFIT_MARKET
    time_in_force: Optional[str]  # GTC/IOC/FOK for LIMIT orders
    reduce_only: bool
    close_position: bool
    signal_id: str
    strategy_version: str
    requested_at_ms: int

    def to_dict(self) -> dict:
        return {
            "event": "order_request_prepared",
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "reduce_only": self.reduce_only,
            "close_position": self.close_position,
            "signal_id": self.signal_id,
            "strategy_version": self.strategy_version,
            "requested_at_ms": self.requested_at_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OrderRequest":
        return cls(
            client_order_id=d.get("client_order_id", ""),
            symbol=d.get("symbol", ""),
            side=d.get("side", ""),
            order_type=d.get("order_type", ""),
            quantity=float(d.get("quantity", 0.0)),
            price=float(d["price"]) if d.get("price") is not None else None,
            stop_price=float(d["stop_price"]) if d.get("stop_price") is not None else None,
            time_in_force=d.get("time_in_force"),
            reduce_only=bool(d.get("reduce_only", False)),
            close_position=bool(d.get("close_position", False)),
            signal_id=d.get("signal_id", ""),
            strategy_version=d.get("strategy_version", ""),
            requested_at_ms=int(d.get("requested_at_ms", 0)),
        )


# ---------------------------------------------------------------------------
# Execution result / event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of an execution attempt (order placement result or rejection).

    Never contains API secrets.
    """

    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    signal_id: str
    status: str  # "ACCEPTED", "REJECTED", "ERROR", "PENDING"
    message: str
    binance_order_id: Optional[int] = None
    filled_qty: Optional[float] = None
    fill_price: Optional[float] = None
    fee: Optional[float] = None
    error_code: Optional[int] = None
    executed_at_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "event": "order_result",
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "signal_id": self.signal_id,
            "status": self.status,
            "message": self.message,
            "binance_order_id": self.binance_order_id,
            "filled_qty": self.filled_qty,
            "fill_price": self.fill_price,
            "fee": self.fee,
            "error_code": self.error_code,
            "executed_at_ms": self.executed_at_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionResult":
        return cls(
            client_order_id=d.get("client_order_id", ""),
            symbol=d.get("symbol", ""),
            side=d.get("side", ""),
            order_type=d.get("order_type", ""),
            quantity=float(d.get("quantity", 0.0)),
            signal_id=d.get("signal_id", ""),
            status=d.get("status", "ERROR"),
            message=d.get("message", ""),
            binance_order_id=int(d["binance_order_id"]) if d.get("binance_order_id") is not None else None,
            filled_qty=float(d["filled_qty"]) if d.get("filled_qty") is not None else None,
            fill_price=float(d["fill_price"]) if d.get("fill_price") is not None else None,
            fee=float(d["fee"]) if d.get("fee") is not None else None,
            error_code=int(d["error_code"]) if d.get("error_code") is not None else None,
            executed_at_ms=int(d.get("executed_at_ms", 0)),
        )


# ---------------------------------------------------------------------------
# Execution event union (for ledger serialization)
# ---------------------------------------------------------------------------

ExecEvent = (
    SignalApproval
    | ApprovedSignalRef
    | OrderRequest
    | ExecutionResult
)


def event_to_dict(event: ExecEvent) -> Dict[str, Any]:
    """Serialize any execution event to a dict for JSONL."""
    if isinstance(event, SignalApproval):
        return event.to_dict()
    elif isinstance(event, ApprovedSignalRef):
        return event.to_dict()
    elif isinstance(event, OrderRequest):
        return event.to_dict()
    elif isinstance(event, ExecutionResult):
        return event.to_dict()
    else:
        raise TypeError(f"Unsupported execution event type: {type(event).__name__}")


def event_from_dict(d: Dict[str, Any]) -> ExecEvent:
    """Deserialize a JSONL dict to the appropriate execution event type."""
    event_type = d.get("event", "")
    if event_type == "approval":
        return SignalApproval.from_dict(d)
    elif event_type == "approved_signal":
        return ApprovedSignalRef.from_dict(d)
    elif event_type == "order_request_prepared":
        return OrderRequest.from_dict(d)
    elif event_type == "order_result":
        return ExecutionResult.from_dict(d)
    else:
        # Unknown event type — return as generic dict for audit
        return d  # type: ignore[return-value]
