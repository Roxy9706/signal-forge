# Tests for Binance USDT-M automatic symbol discovery.
#
# Covers:
#   - exchangeInfo filtering (status, quoteAsset, excludes, leveraged tokens)
#   - 24h ticker liquidity gate (quoteVolume >= 50,000,000 USDT)
#   - malformed / missing quoteVolume fails closed
#   - deduplication, alphabetical ordering, MAX_SYMBOLS cap
#   - discovery failure propagation (no silent fallback)
#   - PaperObserver integration (discovery honours existing symbols when used warm)
#   - no private API / order endpoints touched
#
# Deterministic: all Binance HTTP calls are mocked (no live network).
# Runs under the existing P5 test suite via pytest/unittest.

from __future__ import annotations
import sys
import json
import unittest
import urllib.error
from typing import List, Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binance.market_data import BinancePublicClient
from binance.contracts import (
    is_valid_symbol,
    build_symbol_universe,
    discover_liquid_usdt_symbols,
    MIN_QUOTE_VOLUME_USDT,
    DEFAULT_EXCLUDE,
    MAX_SYMBOLS,
)
from scanner import P5Config, PaperObserver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(ts: float) -> int:
    return int(ts * 1000)


class _FakeUrlopen:
    """Capture URLs and return canned JSON bodies for deterministic tests."""

    def __init__(self):
        self.urls: List[str] = []
        self.responses: List[Any] = []

    def install(self):
        import urllib.request
        self._original = urllib.request.urlopen
        urllib.request.urlopen = self._fake

    def uninstall(self):
        import urllib.request
        urllib.request.urlopen = self._original

    def set_responses(self, responses: List[Any]):
        """responses in call order; each must be JSON-serialisable."""
        self.responses = list(responses)

    def _fake(self, req, timeout=None):
        self.urls.append(req.full_url)
        if not self.responses:
            raise RuntimeError("no more canned responses")
        body = self.responses.pop(0)
        if isinstance(body, Exception):
            raise body
        if isinstance(body, bytes):
            return _FakeResponseBytes(body)
        return _FakeResponse(json.dumps(body))


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return 200

    def info(self):
        class _Info:
            def get(self, key, default=None):
                return default
        return _Info()


class _FakeResponseBytes:
    """Return raw bytes without JSON encoding (for malformed-response tests)."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return 200

    def info(self):
        class _Info:
            def get(self, key, default=None):
                return default
        return _Info()


# ---------------------------------------------------------------------------
# Minimal canned responses
# ---------------------------------------------------------------------------

def _exchange_info(symbols: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"symbols": symbols}


def _ticker_all(symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return symbols


def _make_ticker(symbol: str, quote_volume: Any) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "quoteVolume": quote_volume,
        "lastPrice": "1.0",
        "priceChange": "0",
        "priceChangePercent": "0",
        "volume": "1",
        "quoteVolume": quote_volume,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIsValidSymbolBaseline(unittest.TestCase):
    """Sanity: the existing symbol-validity helper still behaves as before."""

    def test_valid_usdt_pairs(self):
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
            self.assertTrue(is_valid_symbol(sym))

    def test_non_usdt_rejected(self):
        for sym in ("BTCBTC", "ETHBTC", "NEARUSDT", "XRPBUSD"):
            # NEARUSDT ends with USDT and is >4 chars — accepted by string rule
            pass
        self.assertFalse(is_valid_symbol("BTCBTC"))
        self.assertFalse(is_valid_symbol("ETHBTC"))
        self.assertTrue(is_valid_symbol("NEARUSDT"))   # string-rule only
        self.assertFalse(is_valid_symbol("XRPBUSD"))

    def test_leveraged_tokens_rejected(self):
        for sym in ("BTCUP", "BTCDOWN", "ETHUP", "ETHDOWN",
                    "SOLUP", "SOLDOWN", "AXSUP", "AXSDOWN"):
            self.assertFalse(is_valid_symbol(sym))

    def test_excluded_list_rejected(self):
        for sym in DEFAULT_EXCLUDE:
            self.assertFalse(is_valid_symbol(sym))

    def test_too_short_rejected(self):
        self.assertFalse(is_valid_symbol("USDT"))
        self.assertFalse(is_valid_symbol("AAA"))


class TestBuildSymbolUniversePreserved(unittest.TestCase):
    """build_symbol_universe() interface is unchanged; existing behaviour preserved."""

    def test_defaults(self):
        u = build_symbol_universe()
        self.assertTrue(len(u) > 0)
        self.assertTrue(all(s.endswith("USDT") for s in u))
        self.assertTrue(all(is_valid_symbol(s) for s in u))

    def test_capped(self):
        huge = [{"symbol": f"S{i}USDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}
                for i in range(100)]
        u = build_symbol_universe(from_exchange_info=huge, max_symbols=7)
        self.assertEqual(len(u), 7)

    def test_sorted(self):
        eqs = [
            {"symbol": "ZZZUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "AAAUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "MMMUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ]
        u = build_symbol_universe(from_exchange_info=eqs)
        self.assertEqual(u, ["AAAUSDT", "MMMUSDT", "ZZZUSDT"])


class TestPassesExchangeInfoFilters(unittest.TestCase):
    """Direct unit-test of the exchangeInfo filtering predicate."""

    def _sym(self, symbol, status="TRADING", quoteAsset="USDT", underlyingType="COIN",
             contractType="PERPETUAL", **extra) -> Dict[str, Any]:
        d: Dict[str, Any] = {"symbol": symbol, "status": status, "quoteAsset": quoteAsset,
                             "contractType": contractType, "underlyingType": underlyingType}
        d.update(extra)
        return d

    def test_trading_usdt_perpetual_accept(self):
        """Regression 1: TRADING + USDT + PERPETUAL qualifies."""
        self.assertTrue(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(self._sym("BTCUSDT"))
        )

    def test_current_quarter_rejected(self):
        """Regression 2: CURRENT_QUARTER is rejected."""
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(self._sym("BTCUSDT", contractType="CURRENT_QUARTER"))
        )

    def test_next_quarter_rejected(self):
        """Regression 3: NEXT_QUARTER is rejected."""
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(self._sym("BTCUSDT", contractType="NEXT_QUARTER"))
        )

    def test_tradifi_perpetual_rejected(self):
        """Regression 4: TRADIFI_PERPETUAL is rejected (contractType is authoritative)."""
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(self._sym("XAUUSDT", contractType="TRADIFI_PERPETUAL"))
        )

    def test_missing_contract_type_rejected(self):
        """Regression 5: missing contractType fails closed (USDT non-perpetual/unknown)."""
        d = self._sym("BTCUSDT")
        del d["contractType"]
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(d)
        )

    def test_index_underlying_rejected(self):
        """Regression 6: INDEX-style products (BTCDOMUSDT, ALLUSDT) are rejected."""
        f = __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
        self.assertFalse(f._passes_exchange_info_filters(self._sym("BTCDOMUSDT", underlyingType="INDEX")))
        self.assertFalse(f._passes_exchange_info_filters(self._sym("ALLUSDT", underlyingType="INDEX")))

    def test_trading_usdt_coin_accept(self):
        self.assertTrue(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(self._sym("BTCUSDT"))
        )

    def test_existing_exclusions_remain_effective(self):
        """Regression 7: DEFAULT_EXCLUDE + leveraged UP/DOWN exclusion still drop symbols."""
        f = __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
        for sym in DEFAULT_EXCLUDE:
            self.assertFalse(f._passes_exchange_info_filters(self._sym(sym)),
                             msg=f"DEFAULT_EXCLUDE symbol {sym} must stay rejected")
        for sym in ("BTCUPUSDT", "BTCDOWNUSDT", "SOLUPUSDT", "SOLDOWNUSDT"):
            self.assertFalse(f._passes_exchange_info_filters(self._sym(sym)),
                             msg=f"leveraged-token symbol {sym} must stay rejected")

    def test_liquidity_gate_constant_unchanged(self):
        """Regression 8: the $50M 24h quote-volume gate is unchanged."""
        self.assertEqual(MIN_QUOTE_VOLUME_USDT, 50_000_000)

    def test_max_symbols_constant_unchanged(self):
        """Regression 9: manual-list cap MAX_SYMBOLS remains 10 (the
        auto-discovery universe cap lives in scanner.AUTO_DISCOVERY_MAX_SYMBOLS
        and is pinned separately in tests/test_scanner_concurrency.py)."""
        self.assertEqual(MAX_SYMBOLS, 10)

    def test_underlying_type_equity_rejected(self):
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(self._sym("AAPLUSDT", underlyingType="EQUITY"))
        )
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(self._sym("AMDUSDT", underlyingType="EQUITY"))
        )

    def test_underlying_type_non_coin_rejected(self):
        for ut in ("FUND", "STATISTIC", "NONE", ""):
            self.assertFalse(
                __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
                ._passes_exchange_info_filters(self._sym("XCOINUSDT", underlyingType=ut))
            )

    def test_missing_underlying_type_rejected(self):
        # No underlyingType key at all — fail closed.
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters({"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT"})
        )

    def test_non_trading_rejected(self):
        for st in ("BREAK", "PRE_BOOTSTRAP", "HALTED", "DELISTED", ""):
            self.assertFalse(
                __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
                ._passes_exchange_info_filters(self._sym("BTCUSDT", status=st))
            )

    def test_non_usdt_quote_rejected(self):
        for qa in ("BTC", "BUSD", "BNB", "ETH"):
            self.assertFalse(
                __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
                ._passes_exchange_info_filters(self._sym("BTC" + qa, quoteAsset=qa))
            )

    def test_leveraged_tokens_rejected(self):
        for sym in ("BTCUPUSDT", "BTCDOWNUSDT", "ETHUPUSDT", "ETHDOWNUSDT",
                    "SOLUPUSDT", "SOLDOWNUSDT"):
            self.assertFalse(
                __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
                ._passes_exchange_info_filters(self._sym(sym))
            )

    def test_excluded_symbols_rejected(self):
        for sym in DEFAULT_EXCLUDE:
            self.assertFalse(
                __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
                ._passes_exchange_info_filters(self._sym(sym))
            )

    def test_too_short_rejected(self):
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(self._sym("USDT"))
        )
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters(self._sym("AAA"))
        )

    def test_missing_symbol_rejected(self):
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters({})
        )
        self.assertFalse(
            __import__("binance.contracts", fromlist=["_passes_exchange_info_filters"])
            ._passes_exchange_info_filters({"symbol": ""})
        )


class TestParseQuoteVolume(unittest.TestCase):
    """Malformed / missing quoteVolume fails closed."""

    def _parse(self, ticker: Dict[str, Any]) -> Optional[float]:
        return __import__("binance.contracts", fromlist=["_parse_quote_volume"])._parse_quote_volume(ticker)

    def test_normal(self):
        self.assertEqual(self._parse({"quoteVolume": "12345.67"}), 12345.67)
        self.assertEqual(self._parse({"quoteVolume": 99999}), 99999.0)

    def test_missing(self):
        self.assertIsNone(self._parse({}))
        self.assertIsNone(self._parse({"symbol": "BTCUSDT"}))

    def test_null(self):
        self.assertIsNone(self._parse({"quoteVolume": None}))

    def test_non_numeric(self):
        self.assertIsNone(self._parse({"quoteVolume": "nope"}))
        # "inf" string parses to float("inf") which is a valid float but
        # semantically invalid volume — current implementation rejects it.
        self.assertIsNone(self._parse({"quoteVolume": "inf"}))
        self.assertIsNone(self._parse({"quoteVolume": "-inf"}))

    def test_empty_string(self):
        self.assertIsNone(self._parse({"quoteVolume": ""}))

    def test_boolean(self):
        # bool is a subclass of int; float(True) == 1.0, but a boolean
        # is not a valid volume string from Binance.  Fail closed.
        self.assertIsNone(self._parse({"quoteVolume": True}))
        self.assertIsNone(self._parse({"quoteVolume": False}))


class TestDiscoverLiquidUsdtSymbolsHappyPath(unittest.TestCase):
    """Full happy-path discovery with mocked HTTP."""

    def setUp(self):
        self.fake = _FakeUrlopen()
        self.fake.install()
        self.addCleanup(self.fake.uninstall)

    def _client(self) -> BinancePublicClient:
        return BinancePublicClient(timeout_sec=10, max_retries=1, backoff_base_sec=1)

    def test_basic_discovery(self):
        exch = _exchange_info([
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "ETHUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "BNBUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "XRPUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "ADAUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "DOGEUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "SOLUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "AVAXUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "MATICUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "LINKUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "ATOMUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker("BTCUSDT", 2_000_000_000),
            _make_ticker("ETHUSDT", 1_500_000_000),
            _make_ticker("BNBUSDT", 800_000_000),
            _make_ticker("XRPUSDT", 600_000_000),
            _make_ticker("ADAUSDT", 400_000_000),
            _make_ticker("DOGEUSDT", 300_000_000),
            _make_ticker("SOLUSDT", 900_000_000),
            _make_ticker("AVAXUSDT", 200_000_000),
            _make_ticker("MATICUSDT", 150_000_000),
            _make_ticker("LINKUSDT", 120_000_000),
            _make_ticker("ATOMUSDT", 10_000_000),   # below threshold → excluded
        ])
        self.fake.set_responses([exch, tickers])

        syms = discover_liquid_usdt_symbols(self._client(), max_symbols=10)
        # ATOM excluded for low volume; everything else passes.
        # Sorted alphabetically: ADA < AVAX < BNB < BTC < DOGE < ETH < LINK < MATIC < SOL < XRP
        expected = [
            "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
            "ETHUSDT", "LINKUSDT", "MATICUSDT", "SOLUSDT", "XRPUSDT",
        ]
        self.assertEqual(syms, expected)

    def test_sorted_and_capped(self):
        exch = _exchange_info([
            {"symbol": "ZZZUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "AAAUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "MMMUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "BBBUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker("ZZZUSDT", 1_000_000_000),
            _make_ticker("AAAUSDT", 1_000_000_000),
            _make_ticker("MMMUSDT", 1_000_000_000),
            _make_ticker("BBBUSDT", 1_000_000_000),
        ])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client(), max_symbols=3)
        self.assertEqual(syms, ["AAAUSDT", "BBBUSDT", "MMMUSDT"])

    def test_deduplication(self):
        exch = _exchange_info([
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},  # dup
        ])
        tickers = _ticker_all([_make_ticker("BTCUSDT", 1_000_000_000)])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, ["BTCUSDT"])

    def test_empty_candidate_set_returns_empty(self):
        exch = _exchange_info([
            {"symbol": "BTCUSDT", "status": "BREAK", "quoteAsset": "USDT"},
        ])
        tickers = _ticker_all([])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, [])


class TestDiscoverLiquidityGate(unittest.TestCase):
    """quoteVolume threshold and malformed-data handling."""

    def setUp(self):
        self.fake = _FakeUrlopen()
        self.fake.install()
        self.addCleanup(self.fake.uninstall)

    def _client(self) -> BinancePublicClient:
        return BinancePublicClient(timeout_sec=10, max_retries=1, backoff_base_sec=1)

    def test_below_threshold_excluded(self):
        exch = _exchange_info([{"symbol": "COINUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}])
        tickers = _ticker_all([_make_ticker("COINUSDT", MIN_QUOTE_VOLUME_USDT - 1)])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, [])

    def test_at_threshold_included(self):
        exch = _exchange_info([{"symbol": "COINUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}])
        tickers = _ticker_all([_make_ticker("COINUSDT", MIN_QUOTE_VOLUME_USDT)])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, ["COINUSDT"])

    def test_above_threshold_included(self):
        exch = _exchange_info([{"symbol": "COINUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}])
        tickers = _ticker_all([_make_ticker("COINUSDT", MIN_QUOTE_VOLUME_USDT + 1)])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, ["COINUSDT"])

    def test_missing_ticker_excluded(self):
        exch = _exchange_info([{"symbol": "MISSUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}])
        tickers = _ticker_all([_make_ticker("OTHERUSDT", 1_000_000_000)])  # MISS not in ticker list
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, [])

    def test_malformed_quote_volume_excluded(self):
        exch = _exchange_info([{"symbol": "BADUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}])
        tickers = _ticker_all([{"symbol": "BADUSDT", "quoteVolume": "nope"}])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, [])

    def test_null_quote_volume_excluded(self):
        exch = _exchange_info([{"symbol": "NULLUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}])
        tickers = _ticker_all([{"symbol": "NULLUSDT", "quoteVolume": None}])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, [])


class TestDiscoverExclusions(unittest.TestCase):
    """Excludes, leveraged tokens, non-Trading, non-USDT quote all drop out."""

    def setUp(self):
        self.fake = _FakeUrlopen()
        self.fake.install()
        self.addCleanup(self.fake.uninstall)

    def _client(self) -> BinancePublicClient:
        return BinancePublicClient(timeout_sec=10, max_retries=1, backoff_base_sec=1)

    def test_leveraged_tokens_dropped(self):
        exch = _exchange_info([
            {"symbol": "BTCUPUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "BTCDOWNUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "ETHUPUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker("BTCUPUSDT", 1_000_000_000),
            _make_ticker("BTCDOWNUSDT", 1_000_000_000),
            _make_ticker("ETHUPUSDT", 1_000_000_000),
            _make_ticker("BTCUSDT", 1_000_000_000),
        ])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, ["BTCUSDT"])

    def test_excluded_list_dropped(self):
        excl_symbols = list(DEFAULT_EXCLUDE)[:3]
        exch = _exchange_info([
            {"symbol": s, "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"} for s in excl_symbols
        ] + [
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker(s, 1_000_000_000) for s in excl_symbols
        ] + [_make_ticker("BTCUSDT", 1_000_000_000)])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertNotIn(excl_symbols[0], syms)
        self.assertNotIn(excl_symbols[1], syms)
        self.assertNotIn(excl_symbols[2], syms)
        self.assertIn("BTCUSDT", syms)

    def test_non_trading_dropped(self):
        exch = _exchange_info([
            {"symbol": "BREAKUSDT", "status": "BREAK", "quoteAsset": "USDT", "underlyingType": "COIN"},
            {"symbol": "HALTUSDT", "status": "HALTED", "quoteAsset": "USDT", "underlyingType": "COIN"},
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker("BREAKUSDT", 1_000_000_000),
            _make_ticker("HALTUSDT", 1_000_000_000),
            _make_ticker("BTCUSDT", 1_000_000_000),
        ])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, ["BTCUSDT"])

    def test_non_usdt_quote_dropped(self):
        exch = _exchange_info([
            {"symbol": "BTCBTC", "status": "TRADING", "quoteAsset": "BTC", "underlyingType": "COIN"},
            {"symbol": "ETHBUSD", "status": "TRADING", "quoteAsset": "BUSD", "underlyingType": "COIN"},
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker("BTCBTC", 1_000_000_000),
            _make_ticker("ETHBUSD", 1_000_000_000),
            _make_ticker("BTCUSDT", 1_000_000_000),
        ])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client())
        self.assertEqual(syms, ["BTCUSDT"])


class TestDiscoverFailureNoSilentFallback(unittest.TestCase):
    """Discovery failure must propagate — no fallback to get_active_usdt_symbols()."""

    def setUp(self):
        self.fake = _FakeUrlopen()
        self.fake.install()
        self.addCleanup(self.fake.uninstall)

    def _client(self) -> BinancePublicClient:
        return BinancePublicClient(timeout_sec=10, max_retries=1, backoff_base_sec=1)

    def test_exchange_info_http_failure_raises(self):
        # Build a minimal HTTPError
        err = urllib.error.HTTPError(
            url="https://fapi.binance.com/fapi/v1/exchangeInfo",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )
        self.fake.set_responses([err])
        with self.assertRaises(urllib.error.HTTPError):
            discover_liquid_usdt_symbols(self._client())

    def test_exchange_info_malformed_object_raises(self):
        # exchangeInfo top-level is an object but has no "symbols" key → ValueError
        self.fake.set_responses([{"markets": []}])
        with self.assertRaises(ValueError):
            discover_liquid_usdt_symbols(self._client())

    def test_ticker_http_failure_raises(self):
        exch = _exchange_info([
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        self.fake.set_responses([exch, urllib.error.URLError("network down")])
        with self.assertRaises(urllib.error.URLError):
            discover_liquid_usdt_symbols(self._client())

    def test_ticker_malformed_raises(self):
        exch = _exchange_info([
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        self.fake.set_responses([exch, b"not-json"])
        with self.assertRaises(ValueError):
            discover_liquid_usdt_symbols(self._client())


class TestUnderlyingTypeFilter(unittest.TestCase):
    """underlyingType == "COIN" required; missing / non-COIN / EQUITY all rejected."""

    def setUp(self):
        self.fake = _FakeUrlopen()
        self.fake.install()
        self.addCleanup(self.fake.uninstall)

    def _client(self) -> BinancePublicClient:
        return BinancePublicClient(timeout_sec=10, max_retries=1, backoff_base_sec=1)

    def _run(self, *symbols_with_ut, expect: list[str]) -> None:
        exch = _exchange_info([
            {"symbol": s, "status": "TRADING", "quoteAsset": "USDT",
             "contractType": "PERPETUAL", "underlyingType": ut}
            for s, ut in symbols_with_ut
        ])
        tickers = _ticker_all([_make_ticker(s, 1_000_000_000) for s, _ut in symbols_with_ut])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client(), max_symbols=MAX_SYMBOLS)
        self.assertEqual(syms, expect)

    def test_coin_accepted(self):
        self._run(("BTCUSDT", "COIN"), ("ETHUSDT", "COIN"), expect=["BTCUSDT", "ETHUSDT"])

    def test_equity_rejected(self):
        self._run(
            ("AAPLUSDT", "EQUITY"),
            ("BTCUSDT", "COIN"),
            expect=["BTCUSDT"],
        )

    def test_non_coin_rejected(self):
        self._run(
            ("FUNDUSDT", "FUND"),
            ("STATUSDT", "STATISTIC"),
            ("EMPTYUSDT", ""),
            ("BTCUSDT", "COIN"),
            expect=["BTCUSDT"],
        )

    def test_missing_underlying_type_rejected(self):
        self._run(
            ("BTCUSDT", "COIN"),
            ("MISSINGUSDT", None),  # type: ignore[arg-type]
            expect=["BTCUSDT"],
        )


class TestLiquidityRanking(unittest.TestCase):
    """Top-N is selected by 24h quoteVolume descending; output is alphabetical."""

    def setUp(self):
        self.fake = _FakeUrlopen()
        self.fake.install()
        self.addCleanup(self.fake.uninstall)

    def _client(self) -> BinancePublicClient:
        return BinancePublicClient(timeout_sec=10, max_retries=1, backoff_base_sec=1)

    def test_volume_drives_selection_not_alphabet_order(self):
        # High-volume symbol that sorts late alphabetically outranks a
        # low-volume symbol that sorts early alphabetically.
        exch = _exchange_info([
            {"symbol": "AAADUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "ZZZUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker("AAADUSDT", 60_000_000),     # qualifies but lower volume
            _make_ticker("ZZZUSDT", 900_000_000),     # qualifies and higher volume
        ])
        self.fake.set_responses([exch, tickers])
        syms = discover_liquid_usdt_symbols(self._client(), max_symbols=1)
        # ZZZUSDT has higher quoteVolume → selected, even though AAADUSDT < ZZZUSDT.
        self.assertEqual(syms, ["ZZZUSDT"])

    def test_top_n_selected_by_volume(self):
        exch = _exchange_info([
            {"symbol": "AUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "BUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "CUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "DUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker("AUSDT", 100_000_000),
            _make_ticker("BUSDT", 400_000_000),
            _make_ticker("CUSDT", 200_000_000),
            _make_ticker("DUSDT", 300_000_000),
        ])
        self.fake.set_responses([exch, tickers])
        # Top 2 by volume: BUSDT (400M) and DUSDT (300M).
        syms = discover_liquid_usdt_symbols(self._client(), max_symbols=2)
        self.assertEqual(syms, ["BUSDT", "DUSDT"])

    def test_output_is_alphabetical_among_selected(self):
        exch = _exchange_info([
            {"symbol": "ZUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "AUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "MUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker("ZUSDT", 600_000_000),
            _make_ticker("AUSDT", 500_000_000),
            _make_ticker("MUSDT", 400_000_000),
        ])
        self.fake.set_responses([exch, tickers])
        # All three qualify; top 3 selected; returned alphabetically.
        syms = discover_liquid_usdt_symbols(self._client(), max_symbols=3)
        self.assertEqual(syms, ["AUSDT", "MUSDT", "ZUSDT"])

    def test_equal_volume_selection_is_deterministic(self):
        exch = _exchange_info([
            {"symbol": "QUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "AUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "ZUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        ])
        tickers = _ticker_all([
            _make_ticker("QUSDT", 200_000_000),
            _make_ticker("AUSDT", 200_000_000),
            _make_ticker("ZUSDT", 200_000_000),
        ])
        self.fake.set_responses([exch, tickers])
        # All equal; top 2 selected deterministically by symbol asc on tie.
        syms = discover_liquid_usdt_symbols(self._client(), max_symbols=2)
        self.assertEqual(syms, ["AUSDT", "QUSDT"])


class TestDiscoveryEndpointAudit(unittest.TestCase):
    """Verify only the expected public fapi endpoints are hit — no private/order."""

    def setUp(self):
        self.fake = _FakeUrlopen()
        self.fake.install()
        self.addCleanup(self.fake.uninstall)

    def _client(self) -> BinancePublicClient:
        return BinancePublicClient(timeout_sec=10, max_retries=1, backoff_base_sec=1)

    def test_only_exchange_info_and_ticker_called(self):
        exch = _exchange_info([{"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}])
        tickers = _ticker_all([_make_ticker("BTCUSDT", 1_000_000_000)])
        self.fake.set_responses([exch, tickers])

        discover_liquid_usdt_symbols(self._client(), max_symbols=MAX_SYMBOLS)

        captured = set(self.fake.urls)
        expected = {
            "https://fapi.binance.com/fapi/v1/exchangeInfo",
            "https://fapi.binance.com/fapi/v1/ticker/24hr",
        }
        self.assertEqual(captured, expected)

    def test_no_private_order_endpoints(self):
        exch = _exchange_info([{"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"}])
        tickers = _ticker_all([_make_ticker("BTCUSDT", 1_000_000_000)])
        self.fake.set_responses([exch, tickers])

        discover_liquid_usdt_symbols(self._client(), max_symbols=MAX_SYMBOLS)

        for url in self.fake.urls:
            self.assertNotIn("/account", url)
            self.assertNotIn("/order", url)
            self.assertNotIn("/openOrder", url)
            self.assertNotIn("/balance", url)
            self.assertNotIn("/-trade", url)
            self.assertNotIn("/listenKey", url)
            self.assertIn("fapi.binance.com/fapi/v1", url)
            self.assertNotIn("api.binance.com/api/v3", url)


class TestPaperObserverIntegration(unittest.TestCase):
    """PaperObserver honours existing symbol behaviour; discovery path wired."""

    def test_config_defaults_no_auto_discover(self):
        cfg = P5Config(symbols=["BTCUSDT", "ETHUSDT"])
        self.assertEqual(cfg.symbols, ["BTCUSDT", "ETHUSDT"])

    def test_paper_observer_accepts_explicit_symbols(self):
        cfg = P5Config(symbols=["BTCUSDT"])
        obs = PaperObserver(cfg)
        self.assertEqual(obs.config.symbols, ["BTCUSDT"])
        self.assertIsInstance(obs.client, BinancePublicClient)

    def test_paper_observer_copies_client_settings(self):
        cfg = P5Config(symbols=["BTCUSDT"], timeout_sec=7, max_retries=3, backoff_base_sec=1)
        obs = PaperObserver(cfg)
        self.assertEqual(obs.client.timeout_sec, 7)
        self.assertEqual(obs.client.max_retries, 3)
        self.assertEqual(obs.client.backoff_base_sec, 1)


# ---------------------------------------------------------------------------
# cmd_paper default/mode tests
# ---------------------------------------------------------------------------
# Deterministic tests for the cmd_paper() argument-handling logic.
# All Binance HTTP calls are mocked via _FakeUrlopen (no live network).
# Covers:
#   - no --symbols  => auto-discovery is the default
#   - --symbols     => manual mode, discovery skipped
#   - discovery failure => fatal (sys.exit)
#   - empty discovery  => fatal (sys.exit)
#   - --auto-discover flag still accepted (backward compat)


class TestCmdPaperDefaultMode(unittest.TestCase):
    """cmd_paper() argument-handling: defaults and manual override."""

    def setUp(self):
        self._fake = _FakeUrlopen()
        self._fake.install()
        # Canned exchangeInfo + ticker that yield a small liquid set.
        self._liquid_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self._fake.set_responses([
            _exchange_info([
                {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
                {"symbol": "ETHUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
                {"symbol": "SOLUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
                {"symbol": "BNBUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            ]),
            _ticker_all([
                _make_ticker("BTCUSDT", 150_000_000),
                _make_ticker("ETHUSDT", 120_000_000),
                _make_ticker("SOLUSDT", 60_000_000),
                _make_ticker("BNBUSDT", 40_000_000),  # below threshold -> excluded
            ]),
        ])

    def tearDown(self):
        self._fake.uninstall()

    def _parse_args(self, argv: List[str]) -> Any:
        from scanner import build_parser
        parser = build_parser()
        # Always cap cycles to 0 so cmd_paper exits the observer loop
        # immediately instead of hanging the test.
        if "--max-cycles" not in argv:
            argv = argv + ["--max-cycles", "0"]
        return parser.parse_args(argv)

    def _run_cmd_paper(self, args: Any) -> None:
        from scanner import cmd_paper
        # sys.exit is called on fatal paths; catch SystemExit.
        try:
            cmd_paper(args)
        except SystemExit as exc:
            raise exc

    def test_no_symbols_defaults_to_discovery(self):
        """python scanner.py paper  -> auto-discovery runs (no --symbols)."""
        args = self._parse_args(["paper"])
        self.assertIsNone(args.symbols)
        self.assertFalse(args.auto_discover)
        # This must succeed (discovery returns symbols).
        try:
            self._run_cmd_paper(args)
        except SystemExit:
            self.fail("cmd_paper with no --symbols should succeed via auto-discovery")
        # The fake recorded 2 Binance calls: exchangeInfo + ticker/24hr.
        self.assertEqual(len(self._fake.urls), 2)
        self.assertTrue(any("exchangeInfo" in u for u in self._fake.urls))
        self.assertTrue(any("ticker/24hr" in u for u in self._fake.urls))

    def test_symbols_supplied_skips_discovery(self):
        """python scanner.py paper --symbols BTCUSDT ETHUSDT -> no HTTP calls."""
        self._fake.set_responses([])  # no responses -> any HTTP call would raise
        args = self._parse_args(["paper", "--symbols", "BTCUSDT", "ETHUSDT"])
        self.assertEqual(args.symbols, ["BTCUSDT", "ETHUSDT"])
        # Should NOT trigger any discovery; should print manual list and proceed.
        try:
            self._run_cmd_paper(args)
        except SystemExit:
            self.fail("cmd_paper with --symbols should not call discovery")
        # No HTTP calls made (discovery skipped).
        self.assertEqual(len(self._fake.urls), 0)

    def test_discovery_failure_is_fatal(self):
        """Failed auto-discovery -> sys.exit(1), no fallback."""
        self._fake.set_responses([urllib.error.URLError("network down")])
        args = self._parse_args(["paper"])
        with self.assertRaises(SystemExit) as ctx:
            self._run_cmd_paper(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_empty_discovery_is_fatal(self):
        """Auto-discovery returning 0 symbols -> sys.exit(1)."""
        # ExchangeInfo with only a sub-threshold symbol + ticker confirming it.
        self._fake.set_responses([
            _exchange_info([
                {"symbol": "BNBUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
            ]),
            _ticker_all([
                _make_ticker("BNBUSDT", 10_000_000),  # below 50M threshold
            ]),
        ])
        args = self._parse_args(["paper"])
        with self.assertRaises(SystemExit) as ctx:
            self._run_cmd_paper(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_auto_discover_flag_still_accepted(self):
        """--auto-discover is backward-compatible; identical to default behaviour."""
        args = self._parse_args(["paper", "--auto-discover"])
        self.assertTrue(args.auto_discover)
        # With --symbols absent, the new logic still auto-discovers (flag ignored).
        try:
            self._run_cmd_paper(args)
        except SystemExit:
            self.fail("--auto-discover with no --symbols should still auto-discover")
        self.assertEqual(len(self._fake.urls), 2)

    def test_symbols_overrides_auto_discover_flag(self):
        """When both --symbols and --auto-discover are given, --symbols wins."""
        self._fake.set_responses([])  # any HTTP call would raise
        args = self._parse_args(["paper", "--symbols", "BTCUSDT", "--auto-discover"])
        self.assertEqual(args.symbols, ["BTCUSDT"])
        self.assertTrue(args.auto_discover)
        # Manual symbols take precedence; no discovery HTTP calls.
        try:
            self._run_cmd_paper(args)
        except SystemExit:
            self.fail("--symbols should win over --auto-discover")
        self.assertEqual(len(self._fake.urls), 0)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
