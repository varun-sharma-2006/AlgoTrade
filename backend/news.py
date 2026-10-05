"""News tone from GDELT (free, no key), as an optional machine-learning feature.

GDELT scores the tone of worldwide online news every day. For a company we fetch the daily average tone of
English articles mentioning its name, and the feature on day t is the average tone of the seven days *before*
t, so a decision at t's close never uses articles published after it. GDELT limits requests to one every
few seconds per IP, so results are cached for a day (in MongoDB too, when it is used) and the feature is
simply skipped when the service is busy.
"""

from __future__ import annotations

import re
import time
from datetime import date, timedelta

import requests

from backend.config import logger

API = "https://api.gdeltproject.org/api/v2/doc/doc"
CACHE_SECONDS = 24 * 3600
_cache: dict[str, tuple[float, dict[str, float] | None]] = {}
_SUFFIXES = r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings|group|class [a-z])\b\.?"


def company_query(name: str | None, symbol: str) -> str:
    """A news search phrase for a company: its short name without legal suffixes, or the ticker."""
    cleaned = re.sub(_SUFFIXES, "", (name or "").lower()).replace(",", " ").strip(" .&")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return f'"{cleaned}"' if len(cleaned) >= 3 else f'"{symbol.split(".")[0].split("-")[0].upper()}"'


def daily_tone(query: str, start: str = "20170101000000") -> dict[str, float] | None:
    """{YYYY-MM-DD: average tone} for English news matching `query`; None when GDELT is unavailable."""
    from backend import pricecache

    key = f"news:{query}"
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    stored, fresh = pricecache.get(key, "tone")
    if stored and fresh:
        _cache[key] = (time.time(), stored["days"])
        return stored["days"]
    try:
        response = requests.get(
            API,
            params={
                "query": f"{query} sourcelang:english",
                "mode": "timelinetone",
                "startdatetime": start,
                "enddatetime": date.today().strftime("%Y%m%d235959"),
                "format": "json",
            },
            timeout=20,
        )
        if response.status_code == 429 or not response.text.startswith("{"):
            raise requests.RequestException(f"GDELT busy ({response.status_code})")
        series = response.json()["timeline"][0]["data"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        logger.warning("News tone unavailable for %s: %s", query, exc)
        days = stored["days"] if stored else None  # a stale copy beats nothing
        _cache[key] = (time.time() - CACHE_SECONDS + 300, days)  # retry in 5 minutes
        return days
    days = {f"{p['date'][:4]}-{p['date'][4:6]}-{p['date'][6:8]}": float(p["value"]) for p in series}
    _cache[key] = (time.time(), days)
    pricecache.put(key, "tone", {"days": days})
    return days


def aligned(timestamps: list[str], tone: dict[str, float], window: int = 7) -> list[float | None]:
    """For each bar, the average tone of the `window` calendar days before the bar's date (None if no news)."""
    out: list[float | None] = []
    for ts in timestamps:
        day = date.fromisoformat(ts[:10])
        values = [tone[d] for k in range(1, window + 1) if (d := (day - timedelta(days=k)).isoformat()) in tone]
        out.append(sum(values) / len(values) if values else None)
    return out


def tone_feature(symbol: str, timestamps: list[str]) -> tuple[list[float | None] | None, str | None]:
    """(aligned tone series, the query used), or (None, reason) when news is unavailable."""
    from backend.market import MarketDataError, _get_json

    name = None
    try:
        quotes = _get_json("/v1/finance/search", {"q": symbol, "quotesCount": 1, "newsCount": 0}).get("quotes") or []
        name = (quotes[0].get("shortname") or quotes[0].get("longname")) if quotes else None
    except MarketDataError:
        pass
    query = company_query(name, symbol)
    tone = daily_tone(query)
    if not tone:
        return None, "GDELT news data is busy or unavailable right now; try again in a few minutes"
    series = aligned(timestamps, tone)
    if sum(1 for v in series if v is not None) < len(series) // 3:
        return None, f"Too little news coverage for {query}"
    return series, query
