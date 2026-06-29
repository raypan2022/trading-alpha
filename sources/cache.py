"""
Minimal file-based JSON cache with TTL.

This is the "persistent cache" layer for cold data (SEC fundamentals, the
ticker->CIK map). It is deliberately simple — a JSON file per key under
.cache/, keyed by a sanitized string, invalidated by file mtime age.

A SQLite-backed interceptor is a later upgrade (see ROADMAP); for single-ticker
runs this is enough to avoid re-hitting SEC on every call.
"""
import os
import json
import time
import hashlib

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")

# Short TTL for live data — long enough to dedupe duplicate tool calls within a
# single committee run (bull + bear both fetch the same news/price), short
# enough that back-to-back live runs aren't stale.
LIVE_TTL = 900                       # 15 minutes
# Point-in-time (as_of) data is immutable — a past date's price/news never
# changes — so it can be cached effectively forever.
BACKTEST_TTL = 30 * 24 * 3600        # 30 days


def ttl_for(as_of) -> int:
    """Pick a TTL based on whether the data is point-in-time (immutable) or live."""
    return BACKTEST_TTL if as_of else LIVE_TTL


def _path(key: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in key)
    # Filesystems cap filenames at ~255 bytes; long keys (e.g. search queries)
    # get truncated with a hash suffix to stay unique and within the limit.
    if len(safe) > 100:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        safe = f"{safe[:80]}_{digest}"
    return os.path.join(CACHE_DIR, f"{safe}.json")


def get(key: str, ttl_seconds: int):
    """Return cached value if present and younger than ttl_seconds, else None."""
    path = _path(key)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > ttl_seconds:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def set(key: str, value) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_path(key), "w") as f:
        json.dump(value, f)
