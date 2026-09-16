# P5 JSONL Logger — source of truth for paper observation
#
# JSONL is the authoritative record of all paper events.
# Any state cache is disposable and rebuildable from JSONL.
#
# Events:
#   - "signal": paper signal generated from closed candles
#   - "tp_hit": TP outcome
#   - "sl_hit": SL outcome
#   - "expired": 4h expiration without TP/SL
#   - "data_quality": data quality / evaluation status event
#
# All events are immutable once written. Duplicates are prevented by
# checking signal_id existence before write.


from __future__ import annotations
import json
import os
import time
from typing import Any, List, Dict, Optional, Tuple, Iterable
from pathlib import Path


# ---------------------------------------------------------------------------
# JSONL append
# ---------------------------------------------------------------------------

def jsonl_append(path: str, record: dict) -> int:
    """Append a JSON record as a new line. Returns the byte offset."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        offset = f.tell()
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return offset


# ---------------------------------------------------------------------------
# JSONL read (batch / streaming)
# ---------------------------------------------------------------------------

def read_jsonl_lines(path: str, offset: int = 0, limit: Optional[int] = None) -> List[str]:
    """Read raw lines from a JSONL file. Supports offset + limit for partial reads."""
    p = Path(path)
    if not p.exists():
        return []
    lines: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        if offset > 0:
            f.seek(offset)
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            lines.append(line.rstrip("\n").rstrip("\r"))
    return lines


def jsonl_read(path: str) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file.

    Returns list of dicts, one per line. Skips blank/invalid lines with a
    warning printed to stderr.
    """
    p = Path(path)
    if not p.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
                if isinstance(event, dict):
                    records.append(event)
            except json.JSONDecodeError:
                import sys
                print(f"[jsonl] warning: skipping invalid JSON on line {line_num}", file=sys.stderr)
                continue
    return records


def jsonl_read_reverse(path: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Read the most recent N records from a JSONL file (newest first)."""
    p = Path(path)
    if not p.exists():
        return []

    # Read lines in batches from the end
    all_lines: List[str] = []
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if file_size == 0:
            return []
        chunk_size = 4096
        pos = file_size
        newline_count = 0
        # Count newlines from end to find batch boundary
        while pos > 0 and newline_count < limit + 10:
            pos -= chunk_size
            if pos < 0:
                pos = 0
            f.seek(pos)
            chunk = f.read(chunk_size).decode("utf-8", errors="replace")
            newline_count += chunk.count("\n")

        batch_start = max(0, pos - chunk_size)
        f.seek(batch_start)
        tail = f.read().decode("utf-8", errors="replace")
        all_lines = tail.split("\n")

    # Filter to valid JSON, newest first
    events: List[Dict[str, Any]] = []
    for line in reversed(all_lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
            if isinstance(event, dict):
                events.append(event)
                if len(events) >= limit:
                    break
        except json.JSONDecodeError:
            continue

    return events


# ---------------------------------------------------------------------------
# JSONL search / filtering
# ---------------------------------------------------------------------------

def jsonl_find(path: str, event_type: str) -> List[Dict[str, Any]]:
    """Find all events of a given type in the JSONL log."""
    lines = read_jsonl_lines(path)
    events: List[Dict[str, Any]] = []
    for line in lines:
        try:
            ev: Dict[str, Any] = json.loads(line)
            if ev.get("event") == event_type:
                events.append(ev)
        except json.JSONDecodeError:
            continue
    return events


def jsonl_has_signal(path: str, signal_id: str) -> bool:
    """Check if a signal_id has ever been written to the JSONL log."""
    signals = jsonl_find(path, event_type="signal")
    for s in signals:
        if s.get("signal_id") == signal_id:
            return True
    return False


def jsonl_has_outcome(path: str, signal_id: str) -> bool:
    """Check if an outcome (tp_hit, sl_hit, or expired) exists for signal_id."""
    outcome_events = jsonl_find(path, event_type="tp_hit")
    outcome_events += jsonl_find(path, event_type="sl_hit")
    outcome_events += jsonl_find(path, event_type="expired")
    for o in outcome_events:
        if o.get("signal_id") == signal_id:
            return True
    return False


# ---------------------------------------------------------------------------
# Open signal recovery
# ---------------------------------------------------------------------------

def jsonl_recover_open_signals(path: str) -> List[Dict[str, Any]]:
    """
    Recover open (unresolved) signals from a JSONL log.

    An open signal = a "signal" event with no corresponding outcome event
    (tp_hit, sl_hit, or expired).

    Returns list of signal event dicts that are still open.
    """
    signals = jsonl_find(path, event_type="signal")
    open_signals: List[Dict[str, Any]] = []
    for sig in signals:
        sid = sig.get("signal_id")
        if sid and not jsonl_has_outcome(path, sid):
            open_signals.append(sig)
    return open_signals


def jsonl_get_open_signal_ids(path: str) -> List[str]:
    """Get list of open signal_ids (unresolved)."""
    open_sigs = jsonl_recover_open_signals(path)
    return [s.get("signal_id") for s in open_sigs if s.get("signal_id")]  # type: ignore[return-value]


def jsonl_get_all_signal_ids(path: str) -> List[str]:
    """Get all signal_ids ever written."""
    signals = jsonl_find(path, event_type="signal")
    return [s.get("signal_id") for s in signals if s.get("signal_id")]  # type: ignore[return-value]


def jsonl_get_latest_processed_close_time(path: str) -> Dict[str, Dict[str, int]]:
    """Get latest processed close_time per (symbol, timeframe).

    Reads from outcome events; derived from data available in the log.
    Returns dict: {symbol: {timeframe: close_time_ms}}
    """
    results: Dict[str, Dict[str, int]] = {}

    # Read all outcome + evaluation events
    for event_type in ("tp_hit", "sl_hit", "expired"):
        events = jsonl_find(path, event_type=event_type)
        for evt in events:
            symbol = evt.get("symbol", "")
            close_time = evt.get("close_time", 0)
            if symbol and close_time:
                if symbol not in results:
                    results[symbol] = {}
                # Determine timeframe from signal or use 5m as default
                timeframe = evt.get("timeframe", "5m")
                if close_time > results[symbol].get(timeframe, 0):
                    results[symbol][timeframe] = close_time

    return results


def jsonl_count(path: str, event_type: Optional[str] = None) -> int:
    """Count records, optionally filtered by event_type."""
    records = jsonl_read(path)
    if event_type is None:
        return len(records)
    return sum(1 for r in records if r.get("event") == event_type)


def jsonl_is_empty(path: str) -> bool:
    """Check if the JSONL file is empty or doesn't exist."""
    p = Path(path)
    if not p.exists():
        return True
    return p.stat().st_size == 0
