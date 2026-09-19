# Execution audit ledger — append-only JSONL, no secrets stored.

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

from storage.jsonl import jsonl_append, jsonl_read, jsonl_find
from storage.jsonl import jsonl_read_reverse as _jsonl_read_reverse

from execution.models import (
    ExecEvent,
    event_to_dict,
    SignalApproval,
    ApprovedSignalRef,
    OrderRequest,
    ExecutionResult,
)


# Fields whose values look like secrets and must never appear in the ledger.
_SECRET_FIELD_NAMES = frozenset({
    "api_key", "apikey", "api_secret", "apisecret", "secret",
    "password", "token", "private_key", "private",
})

# Field names that legitimately contain long hex/ID strings (not secrets).
_SAFE_LONG_STRING_FIELDS = frozenset({
    "signal_id", "client_order_id", "binance_order_id",
    "strategy_version", "approver", "hash", "fingerprint", "signature",
})


class ExecutionLedger:
    """Append-only audit ledger for execution events.

    Uses storage/jsonl.py. NEVER stores API secrets.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: ExecEvent) -> int:
        """Append an execution event. Returns byte offset."""
        record = event_to_dict(event)
        self._assert_no_secrets(record)
        return jsonl_append(self.log_path, record)

    def append_dict(self, record: Dict[str, Any]) -> int:
        """Append a raw dict event (for signal_received, error, etc.)."""
        self._assert_no_secrets(record)
        return jsonl_append(self.log_path, record)

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all events from the ledger."""
        return jsonl_read(self.log_path)

    def read_reverse(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Read the most recent N events (newest first)."""
        return _jsonl_read_reverse(self.log_path, limit=limit)

    def find_by_event_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Find all events of a given type."""
        return jsonl_find(self.log_path, event_type)

    def find_by_signal_id(self, signal_id: str) -> List[Dict[str, Any]]:
        """Find all events related to a specific signal_id."""
        all_events = self.read_all()
        return [e for e in all_events if e.get("signal_id") == signal_id]

    def get_events_by_signal_id(self, signal_id: str) -> List[Dict[str, Any]]:
        return self.find_by_signal_id(signal_id)

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Return events representing pending/open orders."""
        events = self.read_all()
        return [e for e in events if e.get("event") in ("order_submitted", "order_accepted")]

    def count(self, event_type: Optional[str] = None) -> int:
        """Count events, optionally filtered by type."""
        records = self.read_all()
        if event_type is None:
            return len(records)
        return sum(1 for r in records if r.get("event") == event_type)

    def is_empty(self) -> bool:
        """True if ledger file doesn't exist or is empty."""
        p = Path(self.log_path)
        if not p.exists():
            return True
        return p.stat().st_size == 0

    # ------------------------------------------------------------------
    # Convenience event builders
    # ------------------------------------------------------------------

    def log_signal_received(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        score: float,
        signal_time: int,
        strategy_version: str,
        received_at_ms: int,
    ) -> int:
        """Log a signal_received event."""
        record = {
            "event": "signal_received",
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "score": score,
            "signal_time": signal_time,
            "strategy_version": strategy_version,
            "received_at_ms": received_at_ms,
        }
        return self.append_dict(record)

    def log_approval(self, approval: SignalApproval) -> int:
        """Log an approval event."""
        return self.append(approval)

    def log_approved_signal(self, signal: ApprovedSignalRef) -> int:
        """Log an approved_signal event."""
        return self.append(signal)

    def log_risk_rejected(
        self,
        signal_id: str,
        symbol: str,
        reason: str,
        risk_code: str,
        details: Optional[dict] = None,
        rejected_at_ms: int = 0,
    ) -> int:
        """Log a risk_rejected event."""
        record = {
            "event": "risk_rejected",
            "signal_id": signal_id,
            "symbol": symbol,
            "reason": reason,
            "risk_code": risk_code,
            "details": details or {},
            "rejected_at_ms": rejected_at_ms,
        }
        return self.append_dict(record)

    def log_order_request_prepared(self, order: OrderRequest) -> int:
        """Log an order_request_prepared event."""
        return self.append(order)

    def log_order_result(self, result: ExecutionResult) -> int:
        """Log an order_result event."""
        return self.append(result)

    def log_cancel_result(self, result: CancelResult) -> int:
        """Log a cancel_result event."""
        from execution.demo_cancel import CancelResult as _CancelResult

        record = {
            "event": "cancel_result",
            "symbol": result.symbol,
            "side": result.side,
            "status": result.status,
            "order_id": result.order_id,
            "client_order_id": result.client_order_id,
            "message": result.message,
            "executed_at_ms": result.executed_at_ms,
            "error_code": result.error_code,
        }
        return self.append_dict(record)

    def log_error(
        self,
        signal_id: str = "",
        symbol: str = "",
        error_type: str = "",
        message: str = "",
        error_code: Optional[int] = None,
        occurred_at_ms: int = 0,
        details: Optional[dict] = None,
    ) -> int:
        """Log a generic error event."""
        record = {
            "event": "error",
            "signal_id": signal_id,
            "symbol": symbol,
            "error_type": error_type,
            "message": message,
            "error_code": error_code,
            "occurred_at_ms": occurred_at_ms,
            "details": details or {},
        }
        return self.append_dict(record)

    # ------------------------------------------------------------------
    # Secret assertion
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_no_secrets(record: Dict[str, Any]) -> None:
        """Raise if any field name in the record looks like a secret field."""
        for key in record:
            key_lower = key.lower()
            if key_lower in _SECRET_FIELD_NAMES:
                raise ValueError(
                    f"Attempted to write sensitive field {key!r} to execution ledger. "
                    f"Secrets are never logged."
                )
            value = record[key]
            if isinstance(value, str) and len(value) > 32:
                if all(c in "0123456789abcdef" for c in value.lower()):
                    if key_lower not in _SAFE_LONG_STRING_FIELDS:
                        raise ValueError(
                            f"Potential secret value in field {key!r}: {value[:8]}... "
                            f"Secrets are never logged."
                        )
