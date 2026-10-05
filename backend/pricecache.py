"""Daily price history cached in MongoDB, shared by every server instance.

A cached history is reused while it is fresh (less than FRESH_SECONDS old). When it is stale, only the last
month is downloaded and merged in, instead of five years, so Yahoo is called far less and pages load faster.
Uses a synchronous PyMongo client because market data is fetched from worker threads.
"""

from __future__ import annotations

import time
from typing import Any

from backend.config import logger, settings

FRESH_SECONDS = 6 * 3600
_client: Any = None


def _collection() -> Any | None:
    global _client
    if settings.use_in_memory_db or not settings.cache_prices_in_db:
        return None
    try:
        if _client is None:
            from pymongo import MongoClient

            _client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
        return _client[settings.mongo_db_name]["price_cache"]
    except Exception as exc:  # the cache is optional; never fail a request because of it
        logger.warning("Price cache unavailable: %s", exc)
        return None


def get(symbol: str, range_value: str) -> tuple[dict[str, Any] | None, bool]:
    """(cached chart, still fresh?)"""
    collection = _collection()
    if collection is None:
        return None, False
    try:
        doc = collection.find_one({"_id": f"{symbol.upper()}:{range_value}"})
    except Exception as exc:
        logger.warning("Price cache read failed: %s", exc)
        return None, False
    if not doc:
        return None, False
    return doc["chart"], time.time() - doc["fetchedAt"] < FRESH_SECONDS


def put(symbol: str, range_value: str, chart: dict[str, Any]) -> None:
    collection = _collection()
    if collection is None:
        return
    try:
        collection.replace_one(
            {"_id": f"{symbol.upper()}:{range_value}"},
            {"chart": chart, "fetchedAt": time.time()},
            upsert=True,
        )
    except Exception as exc:
        logger.warning("Price cache write failed: %s", exc)


def merge(old: dict[str, Any], recent: dict[str, Any]) -> dict[str, Any]:
    """Old history with the recent bars appended (recent bars replace any for the same day)."""
    recent_days = {p["timestamp"][:10] for p in recent["points"]}
    points = [p for p in old["points"] if p["timestamp"][:10] not in recent_days] + recent["points"]
    points.sort(key=lambda p: p["timestamp"])
    # Keep the window the same length (drop as many old days as were added).
    extra = max(len(points) - len(old["points"]), 0)
    return old | recent | {"points": points[extra:], "range": old.get("range")}
