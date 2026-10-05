"""Market data from Yahoo Finance's public chart API, with Stooq as a fallback for daily history and offline
fallbacks for local development.

Talking to the JSON endpoints directly (instead of through yfinance) keeps pandas/numpy out of the
dependency tree, which makes installs and serverless cold starts much lighter. Daily history is cached in
memory for a few minutes, since one backtest page can ask for the same symbol several times.
"""

from __future__ import annotations

import csv
import io
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import requests
from fastapi import HTTPException

from backend.config import logger, settings
from backend.stores import now

WATCHLIST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
YAHOO = "https://query1.finance.yahoo.com"

OFFLINE_QUOTES: dict[str, dict[str, Any]] = {
    "AAPL": {"symbol": "AAPL", "price": 182.54, "previousClose": 181.82, "currency": "USD"},
    "MSFT": {"symbol": "MSFT", "price": 327.31, "previousClose": 326.78, "currency": "USD"},
    "GOOGL": {"symbol": "GOOGL", "price": 141.05, "previousClose": 140.44, "currency": "USD"},
    "AMZN": {"symbol": "AMZN", "price": 135.13, "previousClose": 134.88, "currency": "USD"},
    "TSLA": {"symbol": "TSLA", "price": 253.24, "previousClose": 255.12, "currency": "USD"},
    "NVDA": {"symbol": "NVDA", "price": 448.67, "previousClose": 452.11, "currency": "USD"},
    "RELIANCE.NS": {"symbol": "RELIANCE.NS", "price": 2461.45, "previousClose": 2458.30, "currency": "INR"},
}

_CLOSE_CACHE_SECONDS = 600
_close_cache: dict[str, tuple[float, list[float]]] = {}
_HISTORY_CACHE_SECONDS = 600
_history_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
STOOQ = "https://stooq.com/q/d/l/"
STOOQ_INDICES = {"^GSPC": "^spx", "^DJI": "^dji", "^IXIC": "^ndq", "^NDX": "^ndx"}


class MarketDataError(Exception):
    """Yahoo returned no usable data (unknown symbol, outage, or blocked request)."""


def yahoo_headers() -> dict[str, str]:
    return {"User-Agent": settings.yahoo_user_agent, "Accept": "application/json"}


def _get_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.get(f"{YAHOO}{path}", params=params, headers=yahoo_headers(), timeout=10)
    except requests.RequestException as exc:
        raise MarketDataError(str(exc)) from exc
    if response.status_code == 404:
        raise MarketDataError("symbol not found")
    if not response.ok:
        raise MarketDataError(f"HTTP {response.status_code}")
    return response.json()


def parse_chart(payload: dict[str, Any], symbol: str, range_value: str, interval: str) -> dict[str, Any]:
    """Turn a /v8/finance/chart response into the API's chart shape, skipping incomplete bars."""
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        raise MarketDataError(str((payload.get("chart") or {}).get("error") or "no data"))
    result = results[0]
    meta = result.get("meta") or {}
    indicators = result.get("indicators") or {}
    quote_data = (indicators.get("quote") or [{}])[0]
    adjclose = ((indicators.get("adjclose") or [{}])[0]).get("adjclose") or []
    points: list[dict[str, Any]] = []
    for index, stamp in enumerate(result.get("timestamp") or []):
        bar = {key: (quote_data.get(key) or [None] * (index + 1))[index] for key in ("open", "high", "low", "close")}
        if any(value is None for value in bar.values()):
            continue
        # Adjust OHLC for splits and dividends (like yfinance's auto_adjust) so backtest returns include them.
        adjusted = adjclose[index] if index < len(adjclose) else None
        factor = adjusted / bar["close"] if adjusted and bar["close"] else 1.0
        volume = (quote_data.get("volume") or [None] * (index + 1))[index]
        points.append(
            {
                "timestamp": datetime.fromtimestamp(stamp, UTC).isoformat(),
                **{key: float(value) * factor for key, value in bar.items()},
                "volume": int(volume) if volume is not None else None,
            }
        )
    if not points:
        raise MarketDataError("no complete bars")
    return {
        "symbol": symbol.upper(),
        "points": points,
        "timezone": meta.get("exchangeTimezoneName") or "UTC",
        "currency": meta.get("currency"),
        "range": range_value,
        "interval": interval,
        "previousClose": points[0]["close"],
        "regularMarketPrice": meta.get("regularMarketPrice"),
    }


def yahoo_chart(symbol: str, range_value: str, interval: str) -> dict[str, Any]:
    payload = _get_json(
        f"/v8/finance/chart/{quote(symbol.upper(), safe='')}", {"range": range_value, "interval": interval}
    )
    return parse_chart(payload, symbol, range_value, interval)


def _quote_from_chart(chart: dict[str, Any]) -> dict[str, Any]:
    closes = [point["close"] for point in chart["points"]]
    price = float(chart.get("regularMarketPrice") or closes[-1])
    # Daily bars always end with the latest session, so the previous close is the bar before it.
    previous = closes[-2] if len(closes) > 1 else closes[-1]
    change = price - previous
    return {
        "symbol": chart["symbol"],
        "price": price,
        "change": change,
        "changePercent": change / previous * 100 if previous else 0.0,
        "previousClose": previous,
        "currency": chart.get("currency"),
        "updated": now().isoformat(),
    }


def fetch_quotes(symbols: list[str]) -> list[dict[str, Any]]:
    if not symbols:
        return []
    requested = [symbol.upper() for symbol in symbols]

    def one(symbol: str) -> dict[str, Any] | None:
        try:
            return _quote_from_chart(yahoo_chart(symbol, "5d", "1d"))
        except MarketDataError as exc:
            logger.warning("Quote for %s unavailable: %s", symbol, exc)
            return None

    with ThreadPoolExecutor(max_workers=min(8, len(requested))) as pool:
        collected = [quote_ for quote_ in pool.map(one, requested) if quote_]
    if not collected and settings.use_in_memory_db:
        return build_offline_quotes(requested)
    return collected


def _range_days(range_value: str) -> int:
    from backend.strategies import period_days

    return period_days(range_value)


def stooq_symbol(symbol: str) -> str | None:
    """Stooq's name for a symbol: US listings get ".us"; other exchanges aren't mapped."""
    upper = symbol.upper()
    if upper in STOOQ_INDICES:
        return STOOQ_INDICES[upper]
    if upper.isalpha() or (upper.replace("-", "").isalpha() and "-" in upper):
        return f"{upper.lower().replace('-', '.')}.us"
    return None


def stooq_chart(symbol: str, range_value: str) -> dict[str, Any]:
    """Daily OHLCV from Stooq's CSV download, trimmed to `range_value`."""
    code = stooq_symbol(symbol)
    if not code:
        raise MarketDataError("no Stooq mapping")
    try:
        response = requests.get(STOOQ, params={"s": code, "i": "d"}, headers=yahoo_headers(), timeout=10)
    except requests.RequestException as exc:
        raise MarketDataError(str(exc)) from exc
    if not response.ok or not response.text.startswith("Date"):
        raise MarketDataError(f"Stooq HTTP {response.status_code}")
    cutoff = (datetime.now(UTC) - timedelta(days=_range_days(range_value))).date().isoformat()
    points = []
    for row in csv.DictReader(io.StringIO(response.text)):
        try:
            if row["Date"] < cutoff:
                continue
            points.append(
                {
                    "timestamp": f"{row['Date']}T00:00:00+00:00",
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(float(row["Volume"])) if row.get("Volume") else None,
                }
            )
        except (KeyError, ValueError):
            continue
    if not points:
        raise MarketDataError("Stooq returned no rows")
    return {
        "symbol": symbol.upper(),
        "points": points,
        "timezone": "UTC",
        "currency": "USD",
        "range": range_value,
        "interval": "1d",
        "previousClose": points[0]["close"],
        "regularMarketPrice": points[-1]["close"],
        "source": "stooq",
    }


def fetch_chart(symbol: str, range_value: str = "1mo", interval: str = "1d") -> dict[str, Any]:
    daily_history = interval == "1d" and _range_days(range_value) >= 365
    key = (symbol.upper(), range_value)
    if daily_history:
        cached = _history_cache.get(key)
        if cached and time.time() - cached[0] < _HISTORY_CACHE_SECONDS:
            return cached[1]
    try:
        chart = yahoo_chart(symbol, range_value, interval)
    except MarketDataError as exc:
        chart = None
        if interval == "1d":
            try:
                chart = stooq_chart(symbol, range_value)
                logger.warning("Yahoo failed for %s (%s); using Stooq", symbol, exc)
            except MarketDataError as fallback_exc:
                logger.error("Chart fetch failed for %s: %s; Stooq: %s", symbol, exc, fallback_exc)
        else:
            logger.error("Chart fetch failed for %s: %s", symbol, exc)
        if chart is None:
            raise HTTPException(status_code=502, detail=f"No market data for {symbol.upper()}") from exc
    if daily_history:
        if len(_history_cache) > 64:
            _history_cache.clear()
        _history_cache[key] = (time.time(), chart)
    return chart


def fx_symbol(currency: str, base: str) -> tuple[str, float]:
    """Yahoo FX ticker for converting `currency` into `base`, and a unit factor (prices quoted in pence)."""
    factor = 1.0
    if currency in {"GBp", "GBX"}:
        currency, factor = "GBP", 0.01
    elif currency == "ZAc":
        currency, factor = "ZAR", 0.01
    elif currency == "ILA":
        currency, factor = "ILS", 0.01
    return f"{currency.upper()}{base.upper()}=X", factor


def fx_rates(currency: str | None, base: str, range_value: str) -> dict[str, float] | None:
    """Daily rate of one unit of `currency` in `base` currency, by date, or None if unavailable."""
    if not currency:
        return None
    ticker, factor = fx_symbol(currency, base)
    if ticker[:3] == base.upper():  # same currency; only a unit change such as pence -> pounds
        return None if factor == 1.0 else {"*": factor}
    chart = fetch_chart(ticker, range_value=range_value, interval="1d")
    return {point["timestamp"][:10]: point["close"] * factor for point in chart["points"]}


def search_symbols(query: str) -> list[dict[str, Any]]:
    try:
        data = _get_json("/v1/finance/search", {"q": query, "quotesCount": 10, "newsCount": 0})
    except MarketDataError as exc:
        if settings.use_in_memory_db:
            return build_offline_search(query)
        raise HTTPException(status_code=502, detail=f"Search service error: {exc}") from exc
    return [
        {
            "symbol": entry["symbol"],
            "shortName": entry.get("shortname"),
            "longName": entry.get("longname"),
            "exchange": entry.get("exchange"),
            "type": entry.get("quoteType"),
        }
        for entry in data.get("quotes") or []
        if entry.get("symbol")
    ]


def _spark_closes(symbols: list[str]) -> dict[str, list[float]]:
    """Daily closes for several symbols in one request (Yahoo's spark endpoint)."""
    data = _get_json("/v7/finance/spark", {"symbols": ",".join(symbols), "range": "6mo", "interval": "1d"})
    closes: dict[str, list[float]] = {}
    for item in (data.get("spark") or {}).get("result") or []:
        responses = item.get("response") or []
        if not responses:
            continue
        series = ((responses[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        values = [float(value) for value in series if value is not None]
        if values:
            closes[item["symbol"].upper()] = values
    return closes


def load_closes(symbols: Sequence[str]) -> dict[str, list[float]]:
    """Six months of daily closes per symbol, cached for 10 minutes (used by the chatbot analyst)."""
    stamp = time.time()
    result: dict[str, list[float]] = {}
    missing = []
    for symbol in dict.fromkeys(s.upper() for s in symbols):
        cached = _close_cache.get(symbol)
        if cached and stamp - cached[0] < _CLOSE_CACHE_SECONDS:
            result[symbol] = cached[1]
        else:
            missing.append(symbol)
    for start in range(0, len(missing), 10):
        try:
            result.update(_spark_closes(missing[start : start + 10]))
        except MarketDataError as exc:
            logger.warning("Bulk price download failed: %s", exc)
    for symbol in missing:
        if symbol not in result:
            try:
                result[symbol] = [point["close"] for point in yahoo_chart(symbol, "6mo", "1d")["points"]]
            except MarketDataError:  # unknown ticker or data outage
                continue
        _close_cache[symbol] = (stamp, result[symbol])
    return result


def build_offline_quotes(symbols: list[str]) -> list[dict[str, Any]]:
    timestamp = now().isoformat()
    fallback: list[dict[str, Any]] = []
    for symbol in symbols:
        base = OFFLINE_QUOTES.get(symbol.upper()) or {
            "symbol": symbol.upper(),
            "price": 100.0,
            "previousClose": 100.0,
            "currency": "USD",
        }
        price = float(base["price"])
        previous = float(base.get("previousClose", price))
        change = price - previous if previous else 0.0
        fallback.append(
            {
                "symbol": base["symbol"],
                "price": price,
                "change": change,
                "changePercent": (change / previous) * 100 if previous else 0.0,
                "previousClose": previous,
                "currency": base.get("currency", "USD"),
                "updated": timestamp,
            }
        )
    return fallback


def build_offline_search(query: str) -> list[dict[str, Any]]:
    lowered = query.lower()
    matches = [
        {"symbol": info["symbol"], "shortName": info["symbol"], "exchange": "OFFLINE", "type": "EQUITY"}
        for info in OFFLINE_QUOTES.values()
        if lowered in info["symbol"].lower()
    ]
    return matches or [{"symbol": query.upper(), "shortName": query.upper(), "exchange": "OFFLINE", "type": "EQUITY"}]
