# Execution configuration — demo-only, fixed URL, env credentials.

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional


DEMO_BASE_URL = "https://demo-fapi.binance.com"
REQUIRED_MODE = "demo"


@dataclass(frozen=True)
class ExecutionConfig:
    """Immutable execution configuration.

    DEMO ONLY. Any deviation from demo mode is rejected at construction.
    Credentials come ONLY from environment — never from source code.
    The base URL is FIXED and cannot be overridden.
    """

    mode: str
    api_key: str
    api_secret: str
    base_url: str
    symbol_allowlist: List[str]
    max_position_quantity: float
    max_total_exposure_usd: float
    min_rr: float
    approved_only: bool

    def __post_init__(self) -> None:
        # --- Mode gate ---
        if not isinstance(self.mode, str) or self.mode != REQUIRED_MODE:
            raise ValueError(
                f"EXECUTION_MODE must be exactly '{REQUIRED_MODE}'; "
                f"got {self.mode!r}. Fail-closed: no other mode permitted."
            )

        # --- Base URL gate ---
        if not isinstance(self.base_url, str) or self.base_url != DEMO_BASE_URL:
            raise ValueError(
                f"Execution base_url is FIXED to {DEMO_BASE_URL!r}. "
                f"Configured value {self.base_url!r} is rejected. "
                f"No live fallback permitted."
            )

        # --- Credential gates ---
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError(
                "DEMO_API_KEY must be a non-empty string from environment. "
                "Missing or empty: fail-closed."
            )
        if not isinstance(self.api_secret, str) or not self.api_secret.strip():
            raise ValueError(
                "DEMO_API_SECRET must be a non-empty string from environment. "
                "Missing or empty: fail-closed."
            )

        # --- Symbol allowlist ---
        if not isinstance(self.symbol_allowlist, list) or not self.symbol_allowlist:
            raise ValueError(
                "symbol_allowlist must be a non-empty list of symbols."
            )
        for sym in self.symbol_allowlist:
            if not isinstance(sym, str) or not sym.strip():
                raise ValueError(f"Invalid symbol in allowlist: {sym!r}")

        # --- Quantity / exposure gates ---
        if not isinstance(self.max_position_quantity, (int, float)) or self.max_position_quantity <= 0:
            raise ValueError(
                f"max_position_quantity must be > 0, got {self.max_position_quantity!r}"
            )
        if not isinstance(self.max_total_exposure_usd, (int, float)) or self.max_total_exposure_usd <= 0:
            raise ValueError(
                f"max_total_exposure_usd must be > 0, got {self.max_total_exposure_usd!r}"
            )
        if not isinstance(self.min_rr, (int, float)) or self.min_rr < 0:
            raise ValueError(
                f"min_rr must be >= 0, got {self.min_rr!r}"
            )
        if not isinstance(self.approved_only, bool):
            raise ValueError(
                f"approved_only must be a bool, got {type(self.approved_only).__name__}"
            )

    @staticmethod
    def from_env(
        symbol_allowlist: Optional[List[str]] = None,
        max_position_quantity: float = 1.0,
        max_total_exposure_usd: float = 1000.0,
        min_rr: float = 2.0,
        approved_only: bool = True,
    ) -> "ExecutionConfig":
        """Build config from environment only.

        Args:
            symbol_allowlist: optional explicit allowlist. If None, defaults to
                BTCUSDT, ETHUSDT, SOLUSDT (the current D8-D15 scan universe).
            max_position_quantity: maximum position quantity per order.
            max_total_exposure_usd: maximum total notional exposure across all positions.
            min_rr: minimum risk:reward ratio required for any order (default 2.0).
            approved_only: if True, only explicitly approved signals may proceed.

        Returns:
            ExecutionConfig with credentials from DEMO_API_KEY / DEMO_API_SECRET env vars.

        Raises:
            ValueError: if env vars missing/empty, or mode not set to 'demo'.
        """
        mode = os.environ.get("EXECUTION_MODE", "")
        api_key = os.environ.get("DEMO_API_KEY", "")
        api_secret = os.environ.get("DEMO_API_SECRET", "")

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

    @property
    def has_credentials(self) -> bool:
        """True if both API key and secret are non-empty."""
        return bool(self.api_key.strip() and self.api_secret.strip())

    def is_symbol_allowed(self, symbol: str) -> bool:
        """Check whether a symbol is in the allowlist."""
        return isinstance(symbol, str) and symbol in self.symbol_allowlist

    def redacted_dict(self) -> dict:
        """Serialize config with credentials redacted (for logging/debug).

        This is the ONLY approved serialization path for logging.
        """
        return {
            "mode": self.mode,
            "api_key": "<REDACTED>",
            "api_secret": "<REDACTED>",
            "base_url": self.base_url,
            "symbol_allowlist": self.symbol_allowlist,
            "max_position_quantity": self.max_position_quantity,
            "max_total_exposure_usd": self.max_total_exposure_usd,
            "min_rr": self.min_rr,
            "approved_only": self.approved_only,
        }

    def __repr__(self) -> str:
        """Safe repr that redacts credentials."""
        return (
            f"ExecutionConfig(\n"
            f"    mode={self.mode!r},\n"
            f"    api_key=<REDACTED>,\n"
            f"    api_secret=<REDACTED>,\n"
            f"    base_url={self.base_url!r},\n"
            f"    symbol_allowlist={self.symbol_allowlist!r},\n"
            f"    max_position_quantity={self.max_position_quantity!r},\n"
            f"    max_total_exposure_usd={self.max_total_exposure_usd!r},\n"
            f"    min_rr={self.min_rr!r},\n"
            f"    approved_only={self.approved_only!r}\n"
            f")"
        )
