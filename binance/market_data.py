# Binance public market data client
#
# Public endpoints only: klines (candlesticks), exchangeInfo, 24hr ticker.
# NO API keys. NO private endpoints. NO orders.
#
# Kline response fields (Binance public klines endpoint):
#   [0]  open_time       (ms)
#   [1]  open            (str)
#   [2]  high            (str)
#   [3]  low             (str)
#   [4]  close           (str)
#   [5]  volume          (str)
#   [6]  close_time      (ms)        ← CLOSEST CANDLE FIELD (NOT open_time)
#   [7]  quote_volume    (str)
#   [8]  count           (str)
#   [9]  taker_buy_vol   (str)
#  [10]  taker_buy_quote (str)
#  [11]  ignore
#
# CLOSEST CANDLE RULE (P5 correction #1):
#   A candle is eligible for signal generation ONLY when:
#       candle.close_time <= poll_time
#   NEVER use open_time as the production closed-candle criterion.
#   If close_time is missing, REJECT the candle — do NOT substitute open_time.


from __future__ import annotations
import time
import json
import hashlib
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from binance.contracts import is_valid_symbol, get_active_usdt_symbols

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POLL_INTERVAL_SEC = 300       # 5 minutes — appropriate for 5m candles
DEFAULT_CANDLE_BUFFER_SEC = 60        # 1 minute after close_time before signal
MAX_SYMBOLS = 10                      # Termux safety cap
MAX_OPEN_SIGNALS = 50                 # Safety cap on open paper signals
DEFAULT_KLINES_LIMIT = 1000           # Max klines per request (Binance max = 1000)
BINANCE_PUBLIC_API = "https://api.binance.com/api/v3"
BINANCE_FUTURES_API = "https://fapi.binance.com/fapi/v1"

# Interval → approximate ms duration (for staleness checks)
CANDLE_INTERVAL_MS = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "3d": 3 * 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OHLCV:
    """Single OHLCV data point."""
    open_time: int       # ms — when the candle opened
    close_time: int      # ms — when the candle closed (CANONICAL closure field)
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError(
                f"invalid OHLC: high ({self.high}) < low ({self.low}) "
                f"at close_time={self.close_time}"
            )
        for name, val in (("open", self.open), ("high", self.high),
                          ("low", self.low), ("close", self.close),
                          ("volume", self.volume)):
            if val is None or (isinstance(val, float) and val != val):
                raise ValueError(f"missing/null field '{name}' at close_time={self.close_time}")


@dataclass(frozen=True)
class Kline(OHLCV):
    """A Binance kline (candlestick) with symbol and interval metadata."""
    symbol: str = ""
    interval: str = ""

    def to_candle(self, timestamp: Optional[int] = None) -> "Candle":
        """Convert to a P3 evaluator Candle. Uses close_time as timestamp."""
        from SignalForge.backtest.evaluator import Candle
        ts = timestamp if timestamp is not None else self.close_time
        return Candle(
            timestamp=ts,
            open_=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
        )


# Re-export Candle for convenience, aliased from evaluator
from SignalForge.backtest.evaluator import Candle


# ---------------------------------------------------------------------------
# Kline parsing
# ---------------------------------------------------------------------------

def parse_kline_row(row: List[Any], symbol: str, interval: str) -> Kline:
    """
    Parse a raw Binance kline row into a Kline object.

    Binance returns klines as a list of lists:
        [open_time, open, high, low, close, volume, close_time, ...]

    Args:
        row: raw kline list from Binance API
        symbol: trading pair symbol (e.g. "BTCUSDT")
        interval: candle interval string (e.g. "5m", "1h", "4h", "1d")

    Returns:
        Kline dataclass instance

    Raises:
        ValueError: if the row is missing required fields, has invalid OHLC,
                    or if close_time is missing (we do NOT substitute open_time).
    """
    if not isinstance(row, (list, tuple)) or len(row) < 12:
        raise ValueError(f"malformed kline row: expected >=12 fields, got {type(row).__name__} len={len(row) if hasattr(row, '__len__') else 'N/A'}")

    raw = list(row)

    # Required fields
    try:
        open_time = int(raw[0])
        open_price = float(raw[1])
        high_price = float(raw[2])
        low_price = float(raw[3])
        close_price = float(raw[4])
        volume = float(raw[5])
        close_time = int(raw[6])   # ← CLOSEST CANDLE FIELD
    except (IndexError, ValueError, TypeError) as e:
        raise ValueError(f"failed to parse kline fields: {e}") from e

    # CLOSEST CANDLE RULE: close_time MUST be present and valid.
    # Do NOT fall back to open_time if close_time is missing.
    if close_time <= 0:
        raise ValueError(
            f"invalid close_time={close_time} for {symbol} {interval} "
            f"(open_time={open_time}). close_time must be > 0. "
            f"Do NOT substitute open_time."
        )

    # Validate close_time >= open_time (basic sanity)
    if close_time < open_time:
        raise ValueError(
            f"close_time ({close_time}) < open_time ({open_time}) "
            f"for {symbol} {interval} — data anomaly"
        )

    return Kline(
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        close_time=close_time,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
    )


# ---------------------------------------------------------------------------
# Closed-candle filtering
# ---------------------------------------------------------------------------

def filter_closed_klines(klines: List[Kline], poll_time_ms: int) -> List[Kline]:
    """
    Filter klines to those that are CLOSED at poll_time.

    CLOSEST CANDLE RULE:
        A candle is eligible ONLY when: candle.close_time <= poll_time
        NEVER use open_time as the production criterion.

    Args:
        klines: list of Kline objects (raw from API)
        poll_time_ms: current poll time in milliseconds (UTC)

    Returns:
        List of Kline objects whose close_time <= poll_time,
        sorted by close_time descending (newest first).
    """
    closed = [k for k in klines if k.close_time <= poll_time_ms]
    closed.sort(key=lambda k: k.close_time, reverse=True)
    return closed


def get_latest_closed_kline(klines: List[Kline], poll_time_ms: int) -> Optional[Kline]:
    """
    Get the single latest CLOSED kline (max close_time where close_time <= poll_time).

    Args:
        klines: list of Kline objects
        poll_time_ms: current poll time in ms

    Returns:
        The Kline with the largest close_time <= poll_time, or None if no closed candles.
    """
    closed = filter_closed_klines(klines, poll_time_ms)
    if not closed:
        return None
    return closed[0]  # already sorted desc by close_time


def get_latest_closed_per_timeframe(
    klines_by_tf: Dict[str, List[Kline]],
    signal_time_ms: int,
    include_eq: bool = False,
) -> Dict[str, Optional[Kline]]:
    """
    For each timeframe, get the latest CLOSED candle at or before signal_time.

    MTF ALIGNMENT RULE (P5 correction #2):
        latest_closed_candle[tf] = max(candle where candle.close_time <= signal_time)
        Never use a higher-timeframe candle whose close_time > signal_time.
        At close_time == signal_time, the candle IS eligible.

    Args:
        klines_by_tf: dict mapping timeframe string (e.g. "5m", "15m", "1h", "4h", "1d")
                      to list of Kline objects (raw, before filtering)
        signal_time_ms: the signal timestamp (close_time of setup candle) in ms
        include_eq: if True, include timeframe entries even when no closed candle exists
                    (value will be None). If False, omit the key entirely.

    Returns:
        dict: {timeframe: latest Closed Kline or None}
    """
    result: Dict[str, Optional[Kline]] = {}
    for tf, klines in klines_by_tf.items():
        latest = get_latest_closed_kline(klines, signal_time_ms)
        if latest is not None or include_eq:
            result[tf] = latest
    return result


def is_data_stale(latest_closed: Optional[Kline], poll_time_ms: int,
                  max_age_ms: int) -> bool:
    """
    Check if the latest closed candle is too old (stale data).

    Args:
        latest_closed: the most recent closed Kline, or None
        poll_time_ms: current poll time in ms
        max_age_ms: maximum age of the latest candle before considered stale

    Returns:
        True if no closed candle exists OR the latest closed candle is older
        than max_age_ms from poll_time.
    """
    if latest_closed is None:
        return True  # no data at all
    age_ms = poll_time_ms - latest_closed.close_time
    return age_ms > max_age_ms


# ---------------------------------------------------------------------------
# Binance public client
# ---------------------------------------------------------------------------

class BinancePublicClient:
    """
    Binance public API client (no authentication).

    Endpoints used:
        - GET /fapi/v1/klines (futures) or /api/v3/klines (spot)
        - GET /fapi/v1/exchangeInfo (futures) or /api/v3/exchangeInfo (spot)
        - GET /fapi/v1/ticker/24hr (futures) or /api/v3/ticker/24hr (spot)

    NO API keys. NO private endpoints. NO orders.
    """

    def __init__(
        self,
        timeout_sec: int = 10,
        max_retries: int = 5,
        backoff_base_sec: int = 2,
        use_futures: bool = True,
    ):
        """
        Initialize the public client.

        Args:
            timeout_sec: HTTP request timeout in seconds.
            max_retries: maximum retry attempts per request.
            backoff_base_sec: base for exponential backoff (2^n * base).
            use_futures: if True, use Binance Futures API (fapi.binance.com).
                         if False, use Spot API (api.binance.com).
        """
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self.use_futures = use_futures

        self._base_url = BINANCE_FUTURES_API if use_futures else BINANCE_PUBLIC_API
        self._session_cache: Dict[str, Any] = {}  # simple per-symbol cache

    def _build_url(self, path: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build a full API URL with query parameters."""
        url = f"{self._base_url}{path}"
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"
        return url

    def _make_request(self, url: str) -> Any:
        """
        Make a GET request with timeout, retry, and error classification.

        Args:
            url: full URL to request

        Returns:
            Parsed JSON response as Python object (list or dict)

        Raises:
            urllib.error.HTTPError: on non-200 responses (after retries exhausted)
            urllib.error.URLError: on network failures (after retries exhausted)
            json.JSONDecodeError: on malformed JSON responses
            TimeoutError: on timeout after all retries
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "SignalForge-P5/1.0 (paper-observation; +contact@example.com)",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw_bytes = resp.read()
                    raw_str = raw_bytes.decode("utf-8")
                    return json.loads(raw_str)

            except urllib.error.HTTPError as e:
                last_error = e
                status = e.code
                if status == 429:
                    # Rate limit — read Retry-After if present
                    retry_after = e.headers.get("Retry-After")
                    wait = (
                        float(retry_after) if retry_after
                        else self.backoff_base_sec * (2 ** (attempt - 1))
                    )
                    time.sleep(min(wait, 60))
                    continue
                elif 500 <= status < 600:
                    # Server error — exponential backoff
                    wait = self.backoff_base_sec * (2 ** (attempt - 1))
                    time.sleep(min(wait, 30))
                    continue
                else:
                    # Other HTTP error — don't retry (400 Bad Request, etc.)
                    raise

            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_error = e
                wait = self.backoff_base_sec * (2 ** (attempt - 1))
                time.sleep(min(wait, 30))
                continue

            except json.JSONDecodeError as e:
                # Malformed response — don't retry, raise immediately
                raise ValueError(f"malformed JSON from {url}: {e}") from e

        # All retries exhausted
        if last_error is not None:
            raise last_error
        raise RuntimeError("unexpected: max_retries exhausted without error")

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = DEFAULT_KLINES_LIMIT,
        startTime: Optional[int] = None,
        endTime: Optional[int] = None,
    ) -> List[Kline]:
        """
        Fetch klines (candlesticks) for a symbol and interval.

        Uses Binance Futures public endpoint (fapi.binance.com/fapi/v1/klines)
        by default. Set use_futures=False for spot.

        Args:
            symbol: trading pair (e.g. "BTCUSDT")
            interval: candle interval (e.g. "5m", "15m", "1h", "4h", "1d")
            limit: max number of candles to return (default 1000, Binance max)
            startTime: optional start time filter (ms)
            endTime: optional end time filter (ms)

        Returns:
            List of Kline objects, sorted ascending by close_time.

        Raises:
            ValueError: on malformed response or if symbol is invalid.
            urllib.error.HTTPError: on HTTP errors (429, 5xx, etc.) after retries.
            urllib.error.URLError: on network failures.
        """
        if not is_valid_symbol(symbol):
            raise ValueError(f"invalid symbol for paper observation: {symbol!r}")

        if interval not in CANDLE_INTERVAL_MS:
            raise ValueError(f"unsupported interval: {interval!r}")

        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, DEFAULT_KLINES_LIMIT),
        }
        if startTime is not None:
            params["startTime"] = startTime
        if endTime is not None:
            params["endTime"] = endTime

        path = "/klines"
        url = self._build_url(path, params)

        raw_klines = self._make_request(url)

        if not isinstance(raw_klines, list):
            raise ValueError(
                f"unexpected klines response type: {type(raw_klines).__name__} "
                f"for {symbol} {interval}"
            )

        klines: List[Kline] = []
        for i, row in enumerate(raw_klines):
            try:
                kline = parse_kline_row(row, symbol, interval)
                klines.append(kline)
            except ValueError as e:
                raise ValueError(f"row {i} in klines for {symbol} {interval}: {e}") from e

        # Sort ascending by close_time (oldest first)
        klines.sort(key=lambda k: k.close_time)
        return klines

    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Fetch exchangeinfo for contract filtering.

        Returns:
            Parsed JSON exchangeinfo object.

        Raises:
            Same as _make_request.
        """
        path = "/exchangeInfo"
        url = self._build_url(path, None)
        return self._make_request(url)

    def get_24hr_ticker(self, symbol: Optional[str] = None) -> Any:
        """
        Fetch 24hr ticker stats.

        Args:
            symbol: optional specific symbol (e.g. "BTCUSDT").
                    If None, returns all tickers.

        Returns:
            Parsed JSON ticker response (dict if symbol given, list otherwise).
        """
        path = "/ticker/24hr"
        params: Dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = symbol
        url = self._build_url(path, params if params else None)
        return self._make_request(url)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def make_fingerprint(symbols: List[str], poll_interval: int) -> str:
    """
    Create a deterministic fingerprint for a configuration.

    Used for caching/identification purposes.
    """
    raw = ",".join(sorted(symbols)) + "|" + str(poll_interval)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return h[:16]
