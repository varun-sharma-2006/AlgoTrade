"""Earnings announcement dates for US stocks, free from SEC EDGAR.

Companies file a Form 8-K with Item 2.02 ("Results of Operations and Financial Condition") when they release
earnings, so the filing dates are the earnings dates. The SEC asks every client to identify itself with a
contact email in the User-Agent header (SEC_CONTACT_EMAIL); without it the feature is off.

A filing may come before the open, during the day or after the close, so the price reaction can land on the
filing day or the next trading day. Strategies can avoid holding through that window, and backtests report
how much of their return came from it.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from backend.config import logger, settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
CACHE_SECONDS = 24 * 3600
_cache: dict[str, tuple[float, Any]] = {}


def enabled() -> bool:
    return bool(settings.sec_contact_email)


def _get(url: str) -> Any:
    headers = {"User-Agent": f"AlgoTradeSimulator/1.0 {settings.sec_contact_email}", "Accept": "application/json"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()


def _cached(key: str, loader: Any) -> Any:
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    value = loader()
    _cache[key] = (time.time(), value)
    return value


def cik_for(symbol: str) -> int | None:
    table = _cached("tickers", lambda: {v["ticker"].upper(): int(v["cik_str"]) for v in _get(TICKERS_URL).values()})
    upper = symbol.upper()
    return table.get(upper) or table.get(upper.replace("-", "."))


def earnings_dates(symbol: str) -> list[str] | None:
    """Dates (YYYY-MM-DD) of earnings 8-K filings, oldest first; None if unavailable or not a US filer."""
    if not enabled():
        return None
    try:
        cik = cik_for(symbol)
        if cik is None:
            return None
        recent = _cached(f"sub:{cik}", lambda: _get(SUBMISSIONS_URL.format(cik=cik))["filings"]["recent"])
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning("SEC earnings dates unavailable for %s: %s", symbol, exc)
        return None
    dates = {
        day
        for form, day, items in zip(recent["form"], recent["filingDate"], recent["items"], strict=False)
        if form in {"8-K", "8-K/A"} and "2.02" in (items or "")
    }
    return sorted(dates)


def windows(timestamps: list[str], dates: list[str], next_open: bool = False) -> tuple[set[int], set[int]]:
    """(decision bars to stay flat on, bars where the price reacts) for each earnings date.

    Reaction bars: the first trading day on or after the filing date, and the day after. Staying flat at the
    closes of the day before and the filing day means no position is held during either reaction bar. When
    orders fill at the next open, the overnight gap into the filing day is earned by the position decided two
    closes earlier, so the blackout starts a day sooner.
    """
    first = 2 if next_open else 1
    days = [ts[:10] for ts in timestamps]
    blackout: set[int] = set()
    reaction: set[int] = set()
    k = 0
    for day in dates:
        while k < len(days) and days[k] < day:
            k += 1
        if k >= len(days):
            break
        reaction.update(i for i in (k, k + 1) if i < len(days))
        blackout.update(i for i in range(k - first, k + 1) if i >= 0)
    return blackout, reaction
