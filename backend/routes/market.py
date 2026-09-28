from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_current_user
from backend.market import WATCHLIST_SYMBOLS, fetch_chart, fetch_quotes, search_symbols

router = APIRouter(tags=["market"])


def parse_symbols(symbols: str | None) -> list[str]:
    raw = symbols.split(",") if symbols else WATCHLIST_SYMBOLS
    return [symbol.strip().upper() for symbol in raw if symbol.strip()]


@router.get("/market/watchlist")
async def get_watchlist(symbols: str | None = None) -> list[dict[str, Any]]:
    return fetch_quotes(parse_symbols(symbols))


@router.get("/market/quote/{symbol}")
async def get_quote(symbol: str) -> dict[str, Any]:
    quotes = fetch_quotes([symbol.upper()])
    if not quotes:
        raise HTTPException(status_code=404, detail=f"No quote for {symbol}")
    return quotes[0]


@router.get("/market/search", dependencies=[Depends(get_current_user)])
async def search_market(q: str) -> list[dict[str, Any]]:
    return search_symbols(q)


@router.get("/market/chart/{symbol}", dependencies=[Depends(get_current_user)])
async def get_chart(symbol: str, range: str = "1mo", interval: str = "1d") -> dict[str, Any]:
    return fetch_chart(symbol, range_value=range, interval=interval)
