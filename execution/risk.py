# Execution risk validation — pre-order, deterministic, fail-closed.

from __future__ import annotations
from typing import Set, Optional
from execution.config import ExecutionConfig
from execution.models import ApprovedSignalRef, OrderRequest


class RiskValidationError(Exception):
    """Raised when a pre-order risk check fails."""

    def __init__(self, code: str, reason: str, details: Optional[dict] = None):
        self.code = code
        self.reason = reason
        self.details = details or {}
        super().__init__(f"[{code}] {reason}")


class RiskValidator:
    """Deterministic pre-order risk validation.

    All checks are explicit and fail-closed: any violation raises
    RiskValidationError.
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self._used_order_ids: Set[str] = set()

    # ------------------------------------------------------------------
    # Signal-level validation
    # ------------------------------------------------------------------

    def validate_signal(self, signal: ApprovedSignalRef) -> None:
        """Validate an approved signal before order preparation."""
        # Direction must be LONG or SHORT
        if signal.direction not in ("LONG", "SHORT"):
            raise RiskValidationError(
                "INVALID_DIRECTION",
                f"Direction must be LONG or SHORT, got {signal.direction!r}",
                {"direction": signal.direction},
            )

        # Symbol must be in allowlist
        if not self.config.is_symbol_allowed(signal.symbol):
            raise RiskValidationError(
                "SYMBOL_NOT_ALLOWED",
                f"Symbol {signal.symbol!r} not in allowlist {self.config.symbol_allowlist}",
                {"symbol": signal.symbol, "allowlist": self.config.symbol_allowlist},
            )

        # Entry, SL, TP must be positive
        for field_name, value in [("entry", signal.entry), ("sl", signal.sl), ("tp", signal.tp)]:
            if not isinstance(value, (int, float)) or value <= 0:
                raise RiskValidationError(
                    "INVALID_PRICE",
                    f"{field_name} must be a positive number, got {value!r}",
                    {field_name: value},
                )

        # Geometry: LONG requires SL < entry < TP; SHORT requires TP < entry < SL
        if signal.direction == "LONG":
            if not (signal.sl < signal.entry < signal.tp):
                raise RiskValidationError(
                    "INVALID_GEOMETRY_LONG",
                    f"LONG requires SL < entry < TP: "
                    f"SL={signal.sl}, entry={signal.entry}, TP={signal.tp}",
                    {"sl": signal.sl, "entry": signal.entry, "tp": signal.tp},
                )
        else:
            if not (signal.tp < signal.entry < signal.sl):
                raise RiskValidationError(
                    "INVALID_GEOMETRY_SHORT",
                    f"SHORT requires TP < entry < SL: "
                    f"TP={signal.tp}, entry={signal.entry}, SL={signal.sl}",
                    {"tp": signal.tp, "entry": signal.entry, "sl": signal.sl},
                )

        # Score must be in [0, 100]
        if not isinstance(signal.score, (int, float)) or signal.score < 0 or signal.score > 100:
            raise RiskValidationError(
                "INVALID_SCORE",
                f"Score must be in [0, 100], got {signal.score!r}",
                {"score": signal.score},
            )

        # R:R must meet minimum
        rr = signal.rr_ratio()
        if rr < self.config.min_rr:
            raise RiskValidationError(
                "RR_BELOW_MINIMUM",
                f"R:R {rr:.2f} below minimum {self.config.min_rr:.2f}",
                {"rr": rr, "min_rr": self.config.min_rr},
            )

    # ------------------------------------------------------------------
    # Order-level validation
    # ------------------------------------------------------------------

    def validate_order(self, order: OrderRequest) -> None:
        """Validate an order request before it would be submitted."""
        # Quantity must be positive
        if not isinstance(order.quantity, (int, float)) or order.quantity <= 0:
            raise RiskValidationError(
                "INVALID_QUANTITY",
                f"Quantity must be > 0, got {order.quantity!r}",
                {"quantity": order.quantity},
            )

        # Quantity must not exceed max position quantity
        if order.quantity > self.config.max_position_quantity:
            raise RiskValidationError(
                "QUANTITY_EXCEEDS_LIMIT",
                f"Quantity {order.quantity} exceeds max {self.config.max_position_quantity}",
                {"quantity": order.quantity, "max": self.config.max_position_quantity},
            )

        # Side must be BUY or SELL
        if order.side not in ("BUY", "SELL"):
            raise RiskValidationError(
                "INVALID_SIDE",
                f"Side must be BUY or SELL, got {order.side!r}",
                {"side": order.side},
            )

        # Order type must be recognized
        valid_types = {"MARKET", "LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"}
        if order.order_type not in valid_types:
            raise RiskValidationError(
                "INVALID_ORDER_TYPE",
                f"Order type must be one of {valid_types}, got {order.order_type!r}",
                {"order_type": order.order_type},
            )

        # Duplicate client_order_id protection
        if order.client_order_id in self._used_order_ids:
            raise RiskValidationError(
                "DUPLICATE_ORDER_ID",
                f"Client order ID {order.client_order_id!r} already used",
                {"client_order_id": order.client_order_id},
            )

        # Register the order ID as used (so duplicates are caught)
        self._used_order_ids.add(order.client_order_id)

    def record_order_id(self, client_order_id: str) -> None:
        """Record a client order ID as used (for duplicate protection)."""
        if client_order_id in self._used_order_ids:
            raise RiskValidationError(
                "DUPLICATE_ORDER_ID",
                f"Client order ID {client_order_id!r} already used",
                {"client_order_id": client_order_id},
            )
        self._used_order_ids.add(client_order_id)
