"""Survivorship bias: which stocks were really in the S&P 500 at the start of a backtest?

Testing today's index members over the past quietly picks the winners: companies that shrank, went bankrupt or
were taken over have already left the index, and fast risers were added later. This module rebuilds the
index's membership on any past date from Wikipedia's list of current constituents and its table of historical
changes (free), so a basket can be checked or drawn as of the start date.

Prices for companies that have since been delisted often aren't available for free; such stocks are reported
as missing rather than silently dropped.
"""

from __future__ import annotations

import html
import random
import re
import time
from datetime import date, datetime
from typing import Any

import requests

from backend.config import logger

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "AlgoTradeSimulator/1.0 (https://github.com/varun-sharma-2006/AlgoTrade)"}
CACHE_SECONDS = 24 * 3600
_cache: dict[str, tuple[float, Any]] = {}


def _page(title: str) -> str:
    response = requests.get(
        API,
        params={"action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": 2},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["parse"]["text"]


def _cells(row: str) -> list[str]:
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
    return [html.unescape(re.sub(r"\[\d+\]", "", re.sub(r"<[^>]+>", "", c))).strip() for c in cells]


def _table(page: str, table_id: str) -> list[list[str]]:
    start = page.find(f'id="{table_id}"')
    if start < 0:
        raise ValueError(f"table {table_id} not found")
    end = page.find("</table>", start)
    return [_cells(row) for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page[start:end], re.S)]


def _yahoo(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def _parse_date(text: str) -> str | None:
    for fmt in ("%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_constituents(page: str) -> list[str]:
    return [_yahoo(row[0]) for row in _table(page, "constituents") if row and row[0] and row[0] != "Symbol"]


def parse_changes(page: str) -> list[dict[str, Any]]:
    """Index changes, newest first: {date, added, removed}. Rows sharing a date omit the date cell."""
    changes: list[dict[str, Any]] = []
    current: str | None = None
    for row in _table(page, "changes"):
        if not row:
            continue
        day = _parse_date(row[0])
        if day:
            current, rest = day, row[1:]
        elif current and len(row) >= 4:
            rest = row
        else:
            continue  # header rows
        if len(rest) < 4:
            continue
        added, removed = _yahoo(rest[0]), _yahoo(rest[2])
        changes.append({"date": current, "added": added or None, "removed": removed or None})
    return changes


def load() -> tuple[list[str], list[dict[str, Any]]] | None:
    """(current members, changes newest first), cached for a day; None if Wikipedia can't be reached."""
    hit = _cache.get("sp500")
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    try:
        members = parse_constituents(_page("List of S&P 500 companies"))
        changes = parse_changes(_page("Historical components of the S&P 500"))
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning("S&P 500 history unavailable: %s", exc)
        return None
    _cache["sp500"] = (time.time(), (members, changes))
    return members, changes


def members_on(day: str, members: list[str], changes: list[dict[str, Any]]) -> set[str]:
    """Index members on `day`, by undoing every change made after it (changes are newest first)."""
    current = set(members)
    for change in changes:
        if change["date"] <= day:
            break
        if change["added"]:
            current.discard(change["added"])
        if change["removed"]:
            current.add(change["removed"])
    return current


def check(
    symbols: list[str], start_day: str, data: tuple[list[str], list[dict[str, Any]]] | None = None
) -> dict[str, Any] | None:
    """How a basket of symbols relates to the index's membership at the start date."""
    data = data or load()
    if not data:
        return None
    members, changes = data
    then = members_on(start_day, members, changes)
    now = set(members)
    added_later = {c["added"]: c["date"] for c in changes if c["added"] and c["date"] > start_day}
    removed_since = [
        {"symbol": c["removed"], "date": c["date"]}
        for c in changes
        if c["removed"] and c["date"] > start_day and c["removed"] in then
    ]
    return {
        "asOf": start_day,
        "membersThen": len(then),
        "inIndexThen": [s for s in symbols if s in then],
        "joinedLater": [{"symbol": s, "date": added_later.get(s)} for s in symbols if s in now and s not in then],
        "notInIndex": [s for s in symbols if s not in then and s not in now],
        "removedSince": removed_since,
        "survivorShare": len([s for s in then if s in now]) / len(then) if then else 0.0,
    }


def sample(start_day: str, size: int = 20, seed: int | None = None) -> dict[str, Any] | None:
    """A random sample of the index as it was on `start_day`, including companies that later left it."""
    data = load()
    if not data:
        return None
    members, changes = data
    then = sorted(members_on(start_day, members, changes))
    rng = random.Random(seed if seed is not None else date.today().toordinal())
    picked = rng.sample(then, min(size, len(then)))
    now = set(members)
    return {
        "asOf": start_day,
        "symbols": picked,
        "leftSince": [s for s in picked if s not in now],
        "membersThen": len(then),
    }
