"""execution/shadow_bridge.py — isolated Demo shadow bridge (Phase 1).

Tail-reads the shadow observer's tenant observation log (READ-ONLY) and feeds
fresh D15 'signal' events through the EXISTING demo_auto safety gates into the
Demo order machinery, recording every decision in a SEPARATE Demo execution
ledger under data/demo_execution/.

Design invariants (audit-approved):
  - The observer (diagnostics/forward_observe_d15.py) and its tenant dirs are
    never written; the log is opened read-only and only its byte cursor lives
    in bridge_state.json.
  - No strategy/risk formulas are duplicated: every signal passes through
    execution.demo_auto.process_signal (mode/trade gates, kill switch,
    geometry, min-RR, allowlist, ledger duplicates, exposure headroom,
    precision, RiskValidator) and execution.demo_orders.submit machinery.
  - Entry/SL/TP come EXACTLY from the approved signal envelope; never
    recomputed. Only the LIMIT price is tick-floored by existing precision
    code (floor only, never raises above entry).
  - Explicit triple opt-in: EXECUTION_MODE=demo + DEMO_BRIDGE_ENABLED=true
    (+ DEMO_AUTO_TRADE=true required for REAL submissions).
  - Kill switch (DEMO_AUTO_KILL_SWITCH) checked before every cycle AND
    immediately before every submission.
  - Demo-only hard gate: ExecutionConfig.base_url and a direct hostname/scheme
    parse must both equal https://demo-fapi.binance.com, else fail closed.
  - Duplicate protection: the ledger is authoritative. A provisional
    'order_prepared' reservation is written BEFORE the POST so a crash between
    prepare and submit can never cause a silent resubmission; ambiguous state
    is handed to the reconciler instead of retried.
  - UnknownExecutionStateError from the POST is recorded conservatively as
    open exposure (order_submitted status=UNKNOWN) and NEVER retried.
  - Single-writer via flock on data/demo_execution/bridge.lock.

This phase is code + mocked tests only: nothing here is wired to place a real
order unless --execute + DEMO_AUTO_TRADE=true + demo env are all set.

Usage:
  python -m execution.shadow_bridge --once                 # dry-run one cycle
  python -m execution.shadow_bridge                        # dry-run daemon
  python -m execution.shadow_bridge --execute              # (not authorized phase 1)
  python -m execution.shadow_bridge status
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from execution.config import ExecutionConfig, DEMO_BASE_URL
from execution.ledger import ExecutionLedger
from execution.models import ApprovedSignalRef, SignalApproval
from execution.demo_auto import (
    _kill_switch_active,
    _fetch_symbol_filters,
    process_signal,
)
from execution.demo_orders import place_limit_order, UnknownExecutionStateError
from execution.demo_client import _check_demo_url

# ── Runtime constants (tests relocate these module constants) ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBSERVER_BASE = PROJECT_ROOT / "diagnostics" / "out" / "forward_observe_d15"
DEFAULT_TENANT = "d7-shadow-daemon"
BRIDGE_DATA_DIR = PROJECT_ROOT / "data" / "demo_execution"

STATE_FILENAME = "bridge_state.json"
LEDGER_FILENAME = "execution_ledger.jsonl"
LOCK_FILENAME = "bridge.lock"

# Bridge-specific env gates (mirrors DEMO_AUTO_TRADE explicitness)
BRIDGE_ENABLED_ENV = "DEMO_BRIDGE_ENABLED"
BRIDGE_MAX_AGE_ENV = "DEMO_BRIDGE_MAX_SIGNAL_AGE_SEC"
DEFAULT_MAX_SIGNAL_AGE_SEC = 900  # 3 closed 5m candles

# Audit requirement: a bridge tenant name is valid iff the OBSERVER would have
# created that directory (same regex semantics, re-implemented here because
# importing diagnostics would drag in its no-execution-import guard).
_TENANT_NAME_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")


def validate_tenant_name(name: str) -> str:
    """Mirror of the observer's tenant regex ^[a-zA-Z][a-zA-Z0-9-]{0,63}$."""
    if (not isinstance(name, str) or not (1 <= len(name) <= 64)
            or name[0] not in _TENANT_NAME_OK
            or name[0].isdigit()
            or any(ch not in _TENANT_NAME_OK for ch in name)):
        raise ValueError(f"invalid bridge tenant name: {name!r}")
    return name


def observation_log_path(tenant: str) -> Path:
    """Read-only source path. Validated tenant, fixed shadow base."""
    validate_tenant_name(tenant)
    return Path(OBSERVER_BASE) / "tenants" / tenant / "observation.jsonl"


# ── Write-path guards (mirror the observer's isolation in reverse) ─────────
_FORBIDDEN_WRITE_PREFIXES: Tuple[str, ...] = ()


def _forbidden_write_prefixes() -> Tuple[Path, ...]:
    return (
        PROJECT_ROOT / "data" / "signals",
        PROJECT_ROOT / "diagnostics",
    )


def assert_writable_bridge_path(path: Path) -> Path:
    """Fail closed if any bridge OUTPUT path lives under data/signals/ or
    diagnostics/ (the observer's territory)."""
    resolved = Path(path).resolve()
    for bad in _forbidden_write_prefixes():
        try:
            resolved.relative_to(bad.resolve())
        except ValueError:
            continue
        raise RuntimeError(
            f"SHADOW BRIDGE SAFETY: refusing to write {resolved} — "
            f"inside forbidden zone {bad}"
        )
    return resolved


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    assert_writable_bridge_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── Safety gates (bridge layer; execution gates stay in demo_auto) ─────────
def bridge_disabled_reason() -> Optional[str]:
    """Return why the bridge must not run, or None if env gates pass."""
    if os.environ.get("EXECUTION_MODE") != "demo":
        return "EXECUTION_MODE must be exactly 'demo'"
    if os.environ.get(BRIDGE_ENABLED_ENV) != "true":
        return f"{BRIDGE_ENABLED_ENV} must be set to true (default OFF)"
    return None


def verify_demo_endpoint(config: ExecutionConfig) -> None:
    """Double safety: config base_url constant + direct URL parse + the
    demo_client hostname validator. Any deviation fails closed."""
    parsed = urllib.parse.urlparse(config.base_url)
    if parsed.scheme != "https" or parsed.hostname != "demo-fapi.binance.com":
        raise RuntimeError(
            "SHADOW BRIDGE SAFETY: execution endpoint is not the Demo host "
            f"(scheme={parsed.scheme!r}, host={parsed.hostname!r})"
        )
    if config.base_url != DEMO_BASE_URL:
        raise RuntimeError(
            f"SHADOW BRIDGE SAFETY: base_url {config.base_url!r} != {DEMO_BASE_URL!r}"
        )
    # Belt and braces: reuse the client-level hard gate.
    _check_demo_url(f"{config.base_url}/fapi/v1/ping")


def max_signal_age_sec() -> int:
    raw = os.environ.get(BRIDGE_MAX_AGE_ENV)
    if raw is None:
        return DEFAULT_MAX_SIGNAL_AGE_SEC
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 0  # malformed staleness cap => fail closed (everything stale)
    return val if val >= 0 else 0


def is_stale(signal: Dict[str, Any], now_ms: int, max_age_sec: int) -> bool:
    """Deterministic staleness: age = now_ms - signal_time (candle close ms).
    Unknown/malformed signal_time is stale (fail closed)."""
    st = signal.get("signal_time")
    if not isinstance(st, (int, float)) or st <= 0:
        return True
    return (now_ms - int(st)) > max_age_sec * 1000


# ── Single-writer lock ──────────────────────────────────────────────────────
class BridgeLock:
    """flock-based single-writer lock on data/demo_execution/bridge.lock."""

    def __init__(self, path: Path):
        assert_writable_bridge_path(path)
        self.path = Path(path)
        self._fh = None

    def __enter__(self) -> "BridgeLock":
        return self.acquire()

    def __exit__(self, *exc_info) -> bool:
        self.release()
        return False

    def acquire(self) -> "BridgeLock":
        assert_writable_bridge_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "r+", encoding="utf-8") if self.path.exists() \
            else open(self.path, "w+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            fh.close()
            raise RuntimeError(
                f"shadow bridge lock held by another instance ({self.path}); "
                "refusing to run two writers"
            )
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh
        return self

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


# ── Cursor / state ──────────────────────────────────────────────────────────
def default_state() -> Dict[str, Any]:
    return {"cursor_offset": 0, "dev": None, "ino": None,
            "last_cycle_at": None, "cycles": 0,
            "signals_seen": 0, "signals_submitted": 0, "signals_rejected": 0,
            "malformed_lines": 0, "notes": []}


def load_state(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return default_state()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt state is NOT trusted as a cursor: full replay is safe
        # because the ledger is the duplicate authority.
        return default_state()
    merged = default_state()
    merged.update(data if isinstance(data, dict) else {})
    return merged


def tail_new_records(log_path: Path, state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, int]:
    """Read-only incremental tail from state cursor.

    Returns (records, new_offset, malformed_count). A line that fails JSON
    parse is counted and skipped (cursor still advances). File rotation
    (size < offset or dev/ino change) resets to a full replay from 0 — safe
    because the ledger dedups by signal_id.
    """
    p = Path(log_path)
    if not p.exists():
        return [], int(state.get("cursor_offset") or 0), 0
    size = p.stat().st_size
    offset = int(state.get("cursor_offset") or 0)
    st = p.stat()
    identity_matches = (state.get("dev") == st.st_dev and state.get("ino") == st.st_ino)
    if not identity_matches or size < offset:
        offset = 0  # rotation/new file: replay from start (ledger dedups)
    records: List[Dict[str, Any]] = []
    malformed = 0
    if size > offset:
        with open(p, "rb") as f:
            f.seek(offset)
            chunk = f.read(size - offset)
        new_offset = offset + len(chunk)
        # Drop a trailing partial line (writer may be mid-append); it will be
        # re-read next cycle once complete.
        if chunk and not chunk.endswith(b"\n"):
            cut = chunk.rfind(b"\n")
            if cut == -1:
                return [], offset, 0  # nothing complete yet: hold cursor
            chunk = chunk[:cut + 1]
            new_offset = offset + len(chunk)
        for raw in chunk.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                if isinstance(rec, dict):
                    records.append(rec)
                else:
                    malformed += 1
            except json.JSONDecodeError:
                malformed += 1
        state["dev"], state["ino"] = st.st_dev, st.st_ino
        return records, new_offset, malformed
    state["dev"], state["ino"] = st.st_dev, st.st_ino
    return [], offset, 0


# ── Ledger helpers ──────────────────────────────────────────────────────────
_RESERVATION_EVENTS = ("signal_received", "order_prepared", "order_submitted",
                       "order_accepted")


def _already_handled(ledger: ExecutionLedger, signal_id: str) -> bool:
    """Ledger-authoritative duplicate check. Includes the pre-POST
    'order_prepared' reservation and 'signal_received' ingest markers so a
    crash between any two steps can never produce a duplicate submission."""
    for ev in ledger.get_events_by_signal_id(signal_id):
        if ev.get("event") in _RESERVATION_EVENTS:
            return True
    return False


def _ingest_mark_exists(ledger: ExecutionLedger, signal_id: str) -> bool:
    for ev in ledger.get_events_by_signal_id(signal_id):
        if ev.get("event") in ("signal_received", "order_prepared",
                               "order_submitted", "order_accepted"):
            return True
    return False


# ── One bridge cycle ───────────────────────────────────────────────────────
def _fetch_filters_real(config: ExecutionConfig, symbol: str):
    """Real-mode precision filters via the existing exchangeInfo GET path."""
    return _fetch_symbol_filters(config, symbol)


def run_bridge_cycle(
    *,
    tenant: str = DEFAULT_TENANT,
    now_ms: Optional[int] = None,
    dry_run: bool = True,
    execute: bool = False,
    max_open_orders: Optional[int] = 1,
    max_exposure_usd: Optional[float] = 250.0,
    data_dir: Optional[Path] = None,
    kill_switch: bool = False,
    fetch_filters=None,
) -> Dict[str, Any]:
    """Run one tail->gate->(submit) cycle. Returns a summary dict.

    dry_run=True: full pipeline through process_signal; logs would_submit;
      ZERO network calls (no exchangeInfo, no POST) and NO ledger writes
      except a local decision log line.
    execute=True + DEMO_AUTO_TRADE=true: real submission path through the
      EXISTING gates (Phase 1 does not authorize this; tests mock transport).
    """
    summary: Dict[str, Any] = {
        "tenant": tenant, "dry_run": dry_run, "ran": False,
        "signals": 0, "would_submit": 0, "submitted": 0,
        "rejected_gates": 0, "stale": 0, "duplicates": 0,
        "ignored_non_signal": 0, "malformed": 0, "reason": None,
    }
    if execute and dry_run:
        raise ValueError("dry_run and execute are mutually exclusive")

    disabled = bridge_disabled_reason()
    if disabled:
        summary["reason"] = disabled
        return summary

    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    data_dir = Path(BRIDGE_DATA_DIR if data_dir is None else data_dir)
    state_path = data_dir / STATE_FILENAME
    ledger_path = data_dir / LEDGER_FILENAME
    lock_path = data_dir / LOCK_FILENAME
    assert_writable_bridge_path(state_path)
    assert_writable_bridge_path(ledger_path)
    assert_writable_bridge_path(lock_path)

    with BridgeLock(lock_path):
        summary["ran"] = True
        # Endpoint hard gate before anything else (real and dry-run alike).
        try:
            config = ExecutionConfig.from_env()
            verify_demo_endpoint(config)
        except Exception as e:
            summary["reason"] = f"endpoint/config gate failed closed: {e}"
            return summary

        ledger = ExecutionLedger(log_path=str(ledger_path))
        state = load_state(state_path)
        src = observation_log_path(tenant)
        records, new_offset, malformed = tail_new_records(src, state)
        summary["malformed"] = malformed

        # Kill switch BEFORE the cycle.
        if _kill_switch_active(kill_switch):
            summary["reason"] = "kill switch engaged before cycle"
            state["last_cycle_at"] = now_ms
            state["cycles"] = int(state.get("cycles") or 0) + 1
            _atomic_write_json(state_path, state)
            return summary

        real_trade = (not dry_run) and execute and \
            os.environ.get("DEMO_AUTO_TRADE") == "true"

        filters_cache: Dict[str, Any] = {}
        filter_fetch_failed: set = set()
        _fetch = fetch_filters or _fetch_filters_real

        run_state = {"proposed_exposure_usd": 0.0, "proposed_orders": 0}

        for rec in records:
            if rec.get("event") != "signal":
                summary["ignored_non_signal"] += 1
                continue
            sid = str(rec.get("signal_id") or "")
            summary["signals"] += 1
            if not sid:
                summary["malformed"] += 1
                summary["rejected_gates"] += 1
                continue
            # Replay-safety across cursor resets.
            if real_trade and _already_handled(ledger, sid):
                summary["duplicates"] += 1
                continue
            # Staleness (before the gates; recorded, never retried later).
            if is_stale(rec, now_ms, max_signal_age_sec()):
                summary["stale"] += 1
                if real_trade:
                    ledger.append_dict({
                        "event": "signal_stale", "signal_id": sid,
                        "signal_time": rec.get("signal_time"),
                        "observed_at_ms": now_ms, "source": "shadow_bridge",
                        "tenant": tenant})
                continue
            if real_trade:
                if not _ingest_mark_exists(ledger, sid):
                    ledger.log_signal_received(
                        signal_id=sid,
                        symbol=str(rec.get("symbol", "")),
                        direction=str(rec.get("direction", "")),
                        entry=float(rec.get("entry") or 0),
                        sl=float(rec.get("sl") or 0),
                        tp=float(rec.get("tp") or 0),
                        score=float(rec.get("score") or 0),
                        signal_time=int(rec.get("signal_time") or 0),
                        strategy_version=str(rec.get("strategy_version", "shadow_bridge_v1")),
                        received_at_ms=now_ms,
                    )
                symbol = str(rec.get("symbol", ""))
                if symbol in filter_fetch_failed:
                    summary["rejected_gates"] += 1
                    continue
                if symbol not in filters_cache:
                    try:
                        filters_cache[symbol] = _fetch(config, symbol)
                    except Exception:
                        filter_fetch_failed.add(symbol)
                        summary["rejected_gates"] += 1
                        continue

            order = process_signal(
                rec, config, ledger,
                exchange_filters=filters_cache.get(str(rec.get("symbol", ""))) if real_trade else None,
                run_state=run_state,
                max_open_orders=max_open_orders,
                max_exposure_usd=max_exposure_usd,
            )
            if order is None:
                summary["rejected_gates"] += 1
                continue

            if not real_trade:
                summary["would_submit"] += 1
                summary.setdefault("would_submit_ids", []).append(sid)
                # Accumulate budget in dry-run too (mirrors demo_auto.main
                # semantics; process_signal only READS run_state).
                run_state["proposed_exposure_usd"] += order["notional_usd"]
                run_state["proposed_orders"] += 1
                continue

            # ── REAL submission path (Phase 1: tests only) ──
            # Kill switch IMMEDIATELY before submission (checked BEFORE the
            # reservation so a kill does not strand a signal unexecuted).
            if _kill_switch_active(kill_switch):
                summary["reason"] = "kill switch engaged before submission"
                break

            # Pre-POST reservation: a crash after this point can never
            # silently resubmit (handled via _already_handled + reconciler).
            ledger.append_dict({
                "event": "order_prepared", "signal_id": sid,
                "symbol": order["symbol"], "direction": order["direction"],
                "client_order_id": order["client_order_id"],
                "quantity": order["quantity"], "price": order["price"],
                "notional_usd": order["notional_usd"],
                "prepared_at_ms": now_ms, "source": "shadow_bridge",
                "tenant": tenant})

            signal_ref = ApprovedSignalRef(
                signal_id=sid, symbol=order["symbol"],
                direction=order["direction"],
                # Geometry passed to the order layer EXACTLY as the signal
                # provided it (never recomputed):
                entry=order["entry"], sl=order["sl"], tp=order["tp"],
                score=order["score"], signal_time=order["signal_time"],
                strategy_version=str(rec.get("strategy_version", "shadow_bridge_v1")),
                approved_at_ms=now_ms, approver="shadow_bridge")
            approval = SignalApproval(
                signal_id=sid, symbol=order["symbol"],
                direction=order["direction"], approved_at_ms=now_ms,
                approver="shadow_bridge", approved=True,
                reason="shadow bridge auto-execution on approved D15 signal")
            try:
                result = place_limit_order(
                    config=config, signal=signal_ref, approval=approval,
                    client_order_id=order["client_order_id"],
                    quantity=order["quantity"], price=order["price"])
            except Exception as exc:  # noqa: BLE001 — recorded below
                # Unknown state (and any submit crash): conservatively keep
                # the exposure committed; NEVER retry. The reconciler
                # verifies against openOrders. UnknownExecutionStateError is
                # flagged explicitly for audit clarity.
                ledger.append_dict({
                    "event": "order_submitted", "signal_id": sid,
                    "symbol": order["symbol"], "direction": order["direction"],
                    "client_order_id": order["client_order_id"],
                    "status": "UNKNOWN", "quantity": order["quantity"],
                    "price": order["price"],
                    "error_message": str(exc)[:300],
                    "unknown_state": isinstance(exc, UnknownExecutionStateError),
                    "recorded_at_ms": now_ms, "source": "shadow_bridge",
                    "tenant": tenant})
                summary["submitted"] += 1
                continue

            ev = getattr(result, "status", None) or str(result)
            ledger.append_dict({
                "event": "order_submitted", "signal_id": sid,
                "symbol": order["symbol"], "direction": order["direction"],
                "client_order_id": order["client_order_id"],
                "binance_order_id": getattr(result, "binance_order_id", None),
                "status": ev,
                "quantity": order["quantity"], "price": order["price"],
                "recorded_at_ms": now_ms, "source": "shadow_bridge",
                "tenant": tenant})
            summary["submitted"] += 1
            run_state["proposed_exposure_usd"] += order["notional_usd"]
            run_state["proposed_orders"] += 1

        state["cursor_offset"] = new_offset
        state["last_cycle_at"] = now_ms
        state["cycles"] = int(state.get("cycles") or 0) + 1
        state["signals_seen"] = int(state.get("signals_seen") or 0) + summary["signals"]
        state["malformed_lines"] = int(state.get("malformed_lines") or 0) + malformed
        _atomic_write_json(state_path, state)

    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Demo shadow bridge (Phase 1)")
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="REAL submissions (requires DEMO_AUTO_TRADE=true; "
                             "NOT authorized in Phase 1)")
    parser.add_argument("--max-open-orders", type=int, default=1)
    parser.add_argument("--max-exposure-usd", type=float, default=250.0)
    parser.add_argument("--kill-switch", action="store_true")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "status"])
    return parser.parse_args(argv)


def status(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    data_dir = Path(BRIDGE_DATA_DIR if data_dir is None else data_dir)
    return load_state(data_dir / STATE_FILENAME)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(), indent=1, sort_keys=True))
        return 0
    dry = not args.execute
    try:
        validate_tenant_name(args.tenant)
    except ValueError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    while True:
        summary = run_bridge_cycle(
            tenant=args.tenant, dry_run=dry, execute=args.execute,
            max_open_orders=args.max_open_orders,
            max_exposure_usd=args.max_exposure_usd,
            kill_switch=args.kill_switch)
        print(json.dumps(summary, sort_keys=True))
        if args.once:
            return 0 if summary.get("ran") or summary.get("reason") else 0
        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    sys.exit(main())
