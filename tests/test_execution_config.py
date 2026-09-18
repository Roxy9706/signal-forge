# Tests for execution configuration.

from __future__ import annotations
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution.config import ExecutionConfig, DEMO_BASE_URL, REQUIRED_MODE


class TestDemoModeRequired(unittest.TestCase):
    """DEMO mode is the only permitted mode."""

    def _clear_env(self) -> None:
        """Remove execution config environment variables."""
        for key in ("EXECUTION_MODE", "DEMO_API_KEY", "DEMO_API_SECRET"):
            os.environ.pop(key, None)

    def test_valid_demo_mode(self):
        """Valid demo mode with credentials passes."""
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = "test_key"
        os.environ["DEMO_API_SECRET"] = "test_secret"
        try:
            cfg = ExecutionConfig.from_env()
            self.assertEqual(cfg.mode, "demo")
            self.assertTrue(cfg.has_credentials)
        finally:
            self._clear_env()

    def test_missing_mode(self):
        """Missing EXECUTION_MODE fails closed."""
        self._clear_env()
        os.environ["DEMO_API_KEY"] = "k"
        os.environ["DEMO_API_SECRET"] = "s"
        try:
            with self.assertRaises(ValueError) as ctx:
                ExecutionConfig.from_env()
            self.assertIn("EXECUTION_MODE must be exactly 'demo'", str(ctx.exception))
        finally:
            self._clear_env()

    def test_wrong_mode_rejected(self):
        """Any mode other than 'demo' is rejected."""
        self._clear_env()
        os.environ["EXECUTION_MODE"] = "live"
        os.environ["DEMO_API_KEY"] = "k"
        os.environ["DEMO_API_SECRET"] = "s"
        try:
            with self.assertRaises(ValueError) as ctx:
                ExecutionConfig.from_env()
            self.assertIn("got 'live'", str(ctx.exception))
        finally:
            self._clear_env()

    def test_empty_mode_rejected(self):
        """Empty string mode is rejected."""
        self._clear_env()
        os.environ["EXECUTION_MODE"] = ""
        os.environ["DEMO_API_KEY"] = "k"
        os.environ["DEMO_API_SECRET"] = "s"
        try:
            with self.assertRaises(ValueError) as ctx:
                ExecutionConfig.from_env()
            self.assertIn("must be exactly 'demo'", str(ctx.exception))
        finally:
            self._clear_env()

    def test_live_mode_uppercase_rejected(self):
        """'LIVE' mode rejected (not just 'live')."""
        self._clear_env()
        os.environ["EXECUTION_MODE"] = "LIVE"
        os.environ["DEMO_API_KEY"] = "k"
        os.environ["DEMO_API_SECRET"] = "s"
        try:
            with self.assertRaises(ValueError):
                ExecutionConfig.from_env()
        finally:
            self._clear_env()


class TestFixedDemoUrl(unittest.TestCase):
    """Base URL is fixed to demo-fapi.binance.com."""

    def setUp(self):
        self._saved = {}
        for k in ("EXECUTION_MODE", "DEMO_API_KEY", "DEMO_API_SECRET"):
            self._saved[k] = os.environ.get(k)
            os.environ[k] = ""

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_default_is_demo_url(self):
        """Default base_url is the fixed demo URL."""
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = "key"
        os.environ["DEMO_API_SECRET"] = "secret"
        cfg = ExecutionConfig.from_env()
        self.assertEqual(cfg.base_url, DEMO_BASE_URL)
        self.assertEqual(cfg.base_url, "https://demo-fapi.binance.com")

    def test_url_is_immutable(self):
        """URL cannot be overridden — the __post_init__ gate enforces DEMO_BASE_URL.

        Constructing ExecutionConfig with any URL other than exactly
        https://demo-fapi.binance.com is rejected at construction time.
        """
        # Direct construction with a live URL is rejected by __post_init__
        with self.assertRaises(ValueError) as ctx:
            ExecutionConfig(
                mode="demo",
                api_key="k",
                api_secret="s",
                base_url="https://fapi.binance.com",  # live — rejected
                symbol_allowlist=["BTCUSDT"],
                max_position_quantity=1.0,
                max_total_exposure_usd=1000.0,
                min_rr=2.0,
                approved_only=True,
            )
        self.assertIn("FIXED", str(ctx.exception))

        # A slightly different URL is also rejected
        with self.assertRaises(ValueError):
            ExecutionConfig(
                mode="demo",
                api_key="k",
                api_secret="s",
                base_url="https://demo-fapi.binance.com:443",  # different — rejected
                symbol_allowlist=["BTCUSDT"],
                max_position_quantity=1.0,
                max_total_exposure_usd=1000.0,
                min_rr=2.0,
                approved_only=True,
            )

        # The correct demo URL is accepted
        cfg = ExecutionConfig(
            mode="demo",
            api_key="k",
            api_secret="s",
            base_url="https://demo-fapi.binance.com",
            symbol_allowlist=["BTCUSDT"],
            max_position_quantity=1.0,
            max_total_exposure_usd=1000.0,
            min_rr=2.0,
            approved_only=True,
        )
        self.assertEqual(cfg.base_url, "https://demo-fapi.binance.com")


class TestSecretRedaction(unittest.TestCase):
    """Credentials are never exposed via repr() or redacted_dict()."""

    def setUp(self):
        self._saved = {}
        for k in ("EXECUTION_MODE", "DEMO_API_KEY", "DEMO_API_SECRET"):
            self._saved[k] = os.environ.get(k)
            os.environ[k] = ""

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_redacted_dict(self):
        """redacted_dict replaces credentials with <REDACTED>."""
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = "super_secret_apikey_12345"
        os.environ["DEMO_API_SECRET"] = "super_secret_secret_67890"
        cfg = ExecutionConfig.from_env()
        d = cfg.redacted_dict()
        self.assertEqual(d["api_key"], "<REDACTED>")
        self.assertEqual(d["api_secret"], "<REDACTED>")
        self.assertNotIn("super_secret", str(d))

    def test_redacted_dict_excludes_raw_secrets(self):
        """redacted_dict() must NOT contain the raw DEMO_API_KEY or DEMO_API_SECRET.

        This is the intended safe serialization boundary: redacted_dict() is the
        approved serialization path for logging/debug. It must never expose
        the raw credential values.
        """
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = "real_key_value"
        os.environ["DEMO_API_SECRET"] = "real_secret_value"
        cfg = ExecutionConfig.from_env()
        d = cfg.redacted_dict()
        # Credentials must be redacted, not raw
        self.assertEqual(d["api_key"], "<REDACTED>")
        self.assertEqual(d["api_secret"], "<REDACTED>")
        # The raw values must NOT appear anywhere in the dict
        d_str = str(d)
        self.assertNotIn("real_key_value", d_str)
        self.assertNotIn("real_secret_value", d_str)

    def test_repr_does_not_leak_secret(self):
        """repr(config) is safe for logging — does NOT contain API key or secret.

        The ExecutionConfig.__repr__ override redacts both DEMO_API_KEY
        and DEMO_API_SECRET so repr() can be safely used in logs.
        """
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = "live_key_abc"
        os.environ["DEMO_API_SECRET"] = "live_sec_xzy"
        cfg = ExecutionConfig.from_env()
        r = repr(cfg)
        # The repr must NOT contain the raw API key
        self.assertNotIn("live_key_abc", r)
        # The repr must NOT contain the raw API secret
        self.assertNotIn("live_sec_xzy", r)
        # The repr should show <REDACTED> for credentials
        self.assertIn("<REDACTED>", r)


class TestMissingCredentials(unittest.TestCase):
    """Missing or empty credentials fail closed."""

    def setUp(self):
        self._saved = {}
        for k in ("EXECUTION_MODE", "DEMO_API_KEY", "DEMO_API_SECRET"):
            self._saved[k] = os.environ.get(k)
            os.environ[k] = ""

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_missing_key_fails(self):
        """DEMO_API_KEY missing => ValueError."""
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = ""
        os.environ["DEMO_API_SECRET"] = "s"
        with self.assertRaises(ValueError) as ctx:
            ExecutionConfig.from_env()
        self.assertIn("DEMO_API_KEY", str(ctx.exception))

    def test_missing_secret_fails(self):
        """DEMO_API_SECRET missing => ValueError."""
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = "k"
        os.environ["DEMO_API_SECRET"] = ""
        with self.assertRaises(ValueError) as ctx:
            ExecutionConfig.from_env()
        self.assertIn("DEMO_API_SECRET", str(ctx.exception))

    def test_empty_key_fails(self):
        """Empty DEMO_API_KEY => ValueError."""
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = "   "
        os.environ["DEMO_API_SECRET"] = "s"
        with self.assertRaises(ValueError):
            ExecutionConfig.from_env()

    def test_empty_secret_fails(self):
        """Empty DEMO_API_SECRET => ValueError."""
        os.environ["EXECUTION_MODE"] = "demo"
        os.environ["DEMO_API_KEY"] = "k"
        os.environ["DEMO_API_SECRET"] = "   "
        with self.assertRaises(ValueError):
            ExecutionConfig.from_env()


class TestBuiltConfig(unittest.TestCase):
    """Direct config construction (not from_env) still enforces gates."""

    def test_direct_demo_mode_ok(self):
        """Direct construction with mode='demo' passes."""
        cfg = ExecutionConfig(
            mode="demo",
            api_key="k",
            api_secret="s",
            base_url=DEMO_BASE_URL,
            symbol_allowlist=["BTCUSDT"],
            max_position_quantity=1.0,
            max_total_exposure_usd=1000.0,
            min_rr=2.0,
            approved_only=True,
        )
        self.assertEqual(cfg.mode, "demo")

    def test_direct_live_mode_fails(self):
        """Direct construction with mode='live' fails."""
        with self.assertRaises(ValueError) as ctx:
            ExecutionConfig(
                mode="live",
                api_key="k",
                api_secret="s",
                base_url=DEMO_BASE_URL,
                symbol_allowlist=["BTCUSDT"],
                max_position_quantity=1.0,
                max_total_exposure_usd=1000.0,
                min_rr=2.0,
                approved_only=True,
            )
        self.assertIn("must be exactly 'demo'", str(ctx.exception))

    def test_direct_live_url_fails(self):
        """Direct construction with live URL fails."""
        with self.assertRaises(ValueError) as ctx:
            ExecutionConfig(
                mode="demo",
                api_key="k",
                api_secret="s",
                base_url="https://fapi.binance.com",
                symbol_allowlist=["BTCUSDT"],
                max_position_quantity=1.0,
                max_total_exposure_usd=1000.0,
                min_rr=2.0,
                approved_only=True,
            )
        self.assertIn("FIXED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
