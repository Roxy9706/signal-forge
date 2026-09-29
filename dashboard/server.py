# SignalForge mobile dashboard — read-only JSONL viewer with auto-refresh.
#
# Read-only boundary (fail closed):
#   - This module only ever READS the scanner JSONL log. It never writes it.
#   - It never imports or calls anything under `execution/`.
#   - It never makes Binance API calls of any kind (the scanner fetches raw
#     klines; the dashboard only serves the scanner's own event log).
#   - It never stores, logs, or surfaces API keys, secrets, or account info.
#   - If the scanner is stopped, this dashboard does NOT start or modify it.
#
# Source of truth: scanner JSONL event log (e.g.
#   data/signals/paper_trade.jsonl). The dashboard is a pure view layer.

from __future__ import annotations
import argparse
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

try:
    import uvicorn
except ImportError:
    uvicorn = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Read-only guard (fail closed at import time)
# ---------------------------------------------------------------------------
_FORBIDDEN_MODULE_NAMES = frozenset({
    "execution",
    "execution.demo_auto",
    "execution.demo_orders",
    "execution.demo_cancel",
    "execution.demo_client",
    "execution.ledger",
    "execution.risk",
    "execution.config",
    "execution.models",
    "execution.precision",
    "execution.shadow_bridge",
    "execution.shadow_reconcile",
})

def _assert_read_only_imports() -> None:
    """Abort if any execution module is already loaded in this process."""
    breached = sorted(
        n for n in sys.modules.keys()
        if n in _FORBIDDEN_MODULE_NAMES or n.startswith("execution.")
    )
    if breached:
        raise SystemExit(
            "SignalForge dashboard refuses to start with execution modules "
            f"already loaded (loaded: {breached}). Restart in a clean process."
        )

_assert_read_only_imports()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="SignalForge Dashboard", debug=False)

# Load the mobile-first HTML template (embedded at startup for fast serving).
import pathlib
_TEMPLATE_PATH = pathlib.Path(__file__).parent / "templates" / "index.html"
_TEMPLATE_HTML = _TEMPLATE_PATH.read_text(encoding="utf-8") if _TEMPLATE_PATH.exists() else ""

app.state.log_path = "data/signals/paper_trade.jsonl"

# ---------------------------------------------------------------------------
# JSONL reading helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def _find_events(events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    """Filter events by type."""
    return [e for e in events if e.get("event") == event_type]


def _get_latest_event(events: List[Dict[str, Any]], event_type: str) -> Optional[Dict[str, Any]]:
    """Get the latest event of a given type."""
    matching = _find_events(events, event_type)
    return matching[-1] if matching else None


# ---------------------------------------------------------------------------
# Dashboard data helpers
# ---------------------------------------------------------------------------


def get_scanner_status(log_path: str) -> Dict[str, Any]:
    """Determine scanner status from JSONL events."""
    events = _read_jsonl(log_path)
    now_ms = int(time.time() * 1000)

    state_event = _get_latest_event(events, "scanner_state")
    cycle_event = _get_latest_event(events, "cycle_end")

    if state_event is None and cycle_event is None:
        return {
            "running": False,
            "state": "unknown",
            "last_update_ms": None,
            "last_update_age_s": None,
            "symbols": [],
            "interval_sec": 300,
            "max_cycles": None,
            "latest_cycle": None,
            "symbol_count": 0,
            "last_cycle_time": None,
            "age_since_last_cycle_s": None,
        }

    # --- Determine the latest scanner activity timestamp ---
    # Use the freshest of scanner_state and cycle_end events.  After a manual
    # Ctrl+C stop the last scanner_state still says "running" but its timestamp
    # can be hours old; a recent cycle_end is a better liveness signal.
    state_time = state_event.get("poll_time_ms", 0) if state_event else 0
    cycle_time = cycle_event.get("poll_time_ms", 0) if cycle_event else 0
    latest_activity_ms = max(state_time, cycle_time)

    # --- Nominal state from the latest scanner_state (fall back to cycle_end) ---
    # Use the freshest event as the primary source for state/interval/symbols.
    # scanner_state carries symbols/interval/max_cycles; cycle_end does not, so
    # when cycle_end is fresher we keep the state/interval from scanner_state
    # (it is still valid metadata) and only use cycle_end to establish liveness
    # and the cycle summary.
    if state_time >= cycle_time:
        # scanner_state is fresher (or equal)
        state = state_event.get("state", "unknown")
        interval_sec = state_event.get("interval_sec", 300)
        symbols = state_event.get("symbols", [])
        max_cycles = state_event.get("max_cycles")
    elif state_event:
        # cycle_end is fresher — state/interval/symbols still come from
        # scanner_state (valid metadata), liveness is driven by the recent
        # cycle_end timestamp.
        state = state_event.get("state", "unknown")
        interval_sec = state_event.get("interval_sec", 300)
        symbols = state_event.get("symbols", [])
        max_cycles = state_event.get("max_cycles")
    elif cycle_event:
        state = "running"
        interval_sec = 300
        symbols = []
        max_cycles = None
    else:
        state = "unknown"
        interval_sec = 300
        symbols = []
        max_cycles = None

    age_ms = now_ms - latest_activity_ms
    age_s = age_ms / 1000

    # --- Freshness gate ---
    # Treat the scanner as stopped when the latest activity is older than
    # 2 poll intervals (mirrors the existing heuristic but applies it to the
    # best available timestamp, not just scanner_state).
    threshold_ms = 2 * interval_sec * 1000
    if state == "running" and age_ms < threshold_ms:
        display_state = "running"
        is_running = True
    else:
        display_state = "stopped"
        is_running = False

    # --- Latest cycle summary (unchanged) ---
    latest_cycle: Optional[Dict[str, Any]] = None
    if cycle_event:
        latest_cycle = {
            "cycle_number": cycle_event.get("cycle_number"),
            "symbols_discovered": cycle_event.get("symbols_discovered"),
            "symbols_processed": cycle_event.get("symbols_processed"),
            "signals_generated": cycle_event.get("signals_generated"),
            "outcomes_resolved": cycle_event.get("outcomes_resolved"),
            "data_quality_events": cycle_event.get("data_quality_events"),
            "error_symbol_count": cycle_event.get("error_symbol_count"),
            "scan_duration_ms": cycle_event.get("scan_duration_ms"),
            "poll_time_ms": cycle_event.get("poll_time_ms"),
        }

    # --- Last cycle time + age (for the dashboard to display) ---
    last_cycle_time = cycle_event.get("poll_time_ms") if cycle_event else None
    if last_cycle_time:
        age_since_last_cycle_s = (now_ms - last_cycle_time) / 1000
    else:
        age_since_last_cycle_s = None

    return {
        "running": is_running,
        "state": display_state,
        "last_update_ms": latest_activity_ms,
        "last_update_age_s": age_s,
        "symbols": symbols,
        "interval_sec": interval_sec,
        "max_cycles": max_cycles,
        "latest_cycle": latest_cycle,
        "symbol_count": len(symbols),
        "last_cycle_time": last_cycle_time,
        "age_since_last_cycle_s": age_since_last_cycle_s,
    }


def get_open_signals(log_path: str) -> List[Dict[str, Any]]:
    """Get open (unresolved) signals — signals without outcomes."""
    events = _read_jsonl(log_path)
    signal_events = _find_events(events, "signal")
    outcome_events = (
        _find_events(events, "tp_hit")
        + _find_events(events, "sl_hit")
        + _find_events(events, "expired")
    )

    outcome_ids = {e.get("signal_id") for e in outcome_events if e.get("signal_id")}

    open_signals: List[Dict[str, Any]] = []
    for sig in signal_events:
        sid = sig.get("signal_id")
        if sid and sid not in outcome_ids:
            open_signals.append(sig)

    return open_signals


def get_recent_signals(log_path: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent signal events, newest first."""
    events = _read_jsonl(log_path)
    signal_events = _find_events(events, "signal")
    signal_events.sort(key=lambda e: e.get("generated_at", 0), reverse=True)
    return signal_events[:limit]


def get_recent_outcomes(log_path: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent outcome events, newest first."""
    events = _read_jsonl(log_path)
    outcome_types = ("tp_hit", "sl_hit", "expired")
    outcome_events: List[Dict[str, Any]] = []
    for et in outcome_types:
        outcome_events.extend(_find_events(events, et))
    outcome_events.sort(key=lambda e: e.get("outcome_time", 0), reverse=True)
    return outcome_events[:limit]


def signal_age_label(signal_time_ms: int) -> str:
    """Return a human-readable age label for a signal."""
    now_ms = int(time.time() * 1000)
    age_ms = now_ms - signal_time_ms
    if age_ms < 0:
        return "just now"
    if age_ms < 1000:
        return "just now"
    if age_ms < 60_000:
        return f"{age_ms // 1000}s ago"
    if age_ms < 3_600_000:
        return f"{age_ms // 60_000}m ago"
    if age_ms < 86_400_000:
        return f"{age_ms // 3_600_000}h ago"
    return f"{age_ms // 86_400_000}d ago"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def dashboard(request: Request) -> HTMLResponse:
    """Main dashboard page — read-only HTML from embedded template."""
    _ = get_scanner_status(app.state.log_path)  # ensure status is computed (for debugging)
    return HTMLResponse(content=_TEMPLATE_HTML, status_code=200)


@app.get("/api/status")
async def api_status(request: Request):
    """Scanner status endpoint — read-only view of scanner_state + cycle_end."""
    return get_scanner_status(request.app.state.log_path)


@app.get("/api/signals")
async def api_signals(request: Request):
    """Signals data endpoint — active + recent, read-only from JSONL."""
    log_path = request.app.state.log_path
    active = get_open_signals(log_path)
    recent = get_recent_signals(log_path, limit=20)
    status = get_scanner_status(log_path)
    return {
        "active": active,
        "recent": recent,
        "has_active": len(active) > 0,
        "status": status,
        "latest_cycle": status.get("latest_cycle"),
    }


@app.get("/api/history")
async def api_history(request: Request):
    """Recent signal + outcome history — read-only from JSONL."""
    log_path = request.app.state.log_path
    return {
        "signals": get_recent_signals(log_path, limit=20),
        "outcomes": get_recent_outcomes(log_path, limit=20),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SignalForge mobile dashboard (read-only JSONL viewer)",
        prog="dashboard.py",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="data/signals/paper_trade.jsonl",
        help="JSONL log path (default: data/signals/paper_trade.jsonl)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    app.state.log_path = args.log

    print(f"[Dashboard] Starting read-only dashboard")
    print(f"[Dashboard] Log: {args.log}")
    print(f"[Dashboard] Listening on http://{args.host}:{args.port}")
    print(f"[Dashboard] Press Ctrl+C to stop")

    if uvicorn is None:
        raise SystemExit("uvicorn is required. Install with: pip install uvicorn")

    uvicorn.run(app, host=args.host, port=args.port, log_level="error")


if __name__ == "__main__":
    main()
