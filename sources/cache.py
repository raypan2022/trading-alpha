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

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")


def _path(key: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in key)
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
