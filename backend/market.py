"""Market data from Yahoo Finance (via yfinance), with offline fallbacks for local development."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from datetime import UTC, timedelta
from typing import Any

import requests
from fastapi import HTTPException

from backend.config import logger, settings
from backend.stores import now

try:
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - yfinance is in requirements
    yf = None

WATCHLIST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]

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


def yahoo_headers() -> dict[str, str]:
    return {"User-Agent": settings.yahoo_user_agent, "Accept": "application/json"}


def fetch_quotes(symbols: list[str]) -> list[dict[str, Any]]:
    if not symbols:
        return []
    collected: list[dict[str, Any]] = []
    if yf is not None:
        try:
            collected = fetch_quotes_with_yfinance([symbol.upper() for symbol in symbols])
        except Exception as exc:
            logger.error("yfinance quote fetch failed: %s", exc)
            raise HTTPException(status_code=502, detail="Quote service error") from exc
    if not collected and settings.use_in_memory_db:
        return build_offline_quotes(symbols)
    return collected


def fetch_chart(symbol: str, range_value: str = "1mo", interval: str = "1d") -> dict[str, Any]:
    if yf is not None:
        try:
            chart = fetch_chart_with_yfinance(symbol, range_value, interval)
            if chart["points"]:
                return chart
        except Exception as exc:
            logger.error("yfinance chart fetch failed for %s: %s", symbol, exc)
            raise HTTPException(status_code=502, detail="Chart service error") from exc
    return build_offline_chart(symbol, range_value, interval)


def search_symbols(query: str) -> list[dict[str, Any]]:
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 10, "newsCount": 0}
    try:
        response = requests.get(url, params=params, headers=yahoo_headers(), timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        if settings.use_in_memory_db:
            return build_offline_search(query)
        raise HTTPException(status_code=502, detail=f"Search service error: {exc}") from exc
    output: list[dict[str, Any]] = []
    for entry in response.json().get("quotes") or []:
        if not entry.get("symbol"):
            continue
        output.append(
            {
                "symbol": entry["symbol"],
                "shortName": entry.get("shortname"),
                "longName": entry.get("longname"),
                "exchange": entry.get("exchange"),
                "type": entry.get("quoteType"),
            }
        )
    return output


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
    if missing and yf is not None:
        try:
            frame = yf.download(missing, period="6mo", interval="1d", progress=False, auto_adjust=True, threads=True)
            closes = frame["Close"]
            for symbol in missing:
                series = closes[symbol] if hasattr(closes, "columns") and symbol in closes.columns else closes
                values = [float(v) for v in series.dropna().tolist()]
                if values:
                    result[symbol] = values
                    _close_cache[symbol] = (stamp, values)
        except Exception as exc:
            logger.warning("Bulk price download failed: %s", exc)
    for symbol in missing:
        if symbol in result:
            continue
        try:
            values = [p["close"] for p in fetch_chart(symbol, range_value="6mo", interval="1d")["points"]]
        except Exception:  # unknown ticker or data outage
            continue
        if values:
            result[symbol] = values
            _close_cache[symbol] = (stamp, values)
    return result


def fetch_quotes_with_yfinance(symbols: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    timestamp = now().isoformat()
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        info = getattr(ticker, "fast_info", {}) or {}
        price = info.get("last_price") or info.get("last_close") or info.get("previous_close")
        previous = info.get("previous_close") or price
        if price is None:
            history = ticker.history(period="5d", interval="1d")
            if not history.empty:
                price = float(history["Close"].iloc[-1])
                previous = float(history["Close"].iloc[-2]) if len(history) > 1 else price
        if price is None:
            continue
        change = price - previous if previous else 0.0
        results.append(
            {
                "symbol": symbol.upper(),
                "price": float(price),
                "change": float(change),
                "changePercent": float(change / previous * 100) if previous else 0.0,
                "previousClose": float(previous) if previous is not None else None,
                "currency": info.get("currency"),
                "updated": timestamp,
            }
        )
    return results


def fetch_chart_with_yfinance(symbol: str, range_value: str, interval: str) -> dict[str, Any]:
    ticker = yf.Ticker(symbol)
    history = ticker.history(period=range_value, interval=interval)
    if history.empty:
        raise ValueError("No history returned")
    points: list[dict[str, Any]] = []
    for timestamp, row in history.iterrows():
        open_price, high, low, close = (float(row.get(key, float("nan"))) for key in ("Open", "High", "Low", "Close"))
        if any(math.isnan(value) for value in (open_price, high, low, close)):
            continue
        volume_val = row.get("Volume", float("nan"))
        ts = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.tz_convert(UTC)
        points.append(
            {
                "timestamp": ts.isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": None if math.isnan(volume_val) else int(volume_val),
            }
        )
    info = getattr(ticker, "fast_info", None)
    return {
        "symbol": symbol.upper(),
        "points": points,
        "timezone": str(history.index.tz) if history.index.tz is not None else "UTC",
        "currency": info.get("currency") if info else None,
        "range": range_value,
        "interval": interval,
        "previousClose": points[0]["close"] if points else None,
    }


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


def build_offline_chart(symbol: str, range_value: str, interval: str) -> dict[str, Any]:
    base_price = float(OFFLINE_QUOTES.get(symbol.upper(), {}).get("price", 100.0))
    points: list[dict[str, Any]] = []
    for idx in range(60):
        close = base_price * (1 + 0.002 * (idx - 30) / 30)
        high, low = close * 1.01, close * 0.99
        points.append(
            {
                "timestamp": (now() - timedelta(days=60 - idx)).isoformat(),
                "open": (high + low) / 2,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000000 + idx * 2500,
            }
        )
    return {
        "symbol": symbol.upper(),
        "points": points,
        "timezone": "UTC",
        "currency": OFFLINE_QUOTES.get(symbol.upper(), {}).get("currency", "USD"),
        "range": range_value,
        "interval": interval,
        "previousClose": points[0]["close"],
    }


def build_offline_search(query: str) -> list[dict[str, Any]]:
    lowered = query.lower()
    matches = [
        {"symbol": info["symbol"], "shortName": info["symbol"], "exchange": "OFFLINE", "type": "EQUITY"}
        for info in OFFLINE_QUOTES.values()
        if lowered in info["symbol"].lower()
    ]
    return matches or [{"symbol": query.upper(), "shortName": query.upper(), "exchange": "OFFLINE", "type": "EQUITY"}]
