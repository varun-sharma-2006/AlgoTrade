"""Alpaca paper trading: place a simulation's signals as real paper orders (admins only, US stocks).

When ALPACA_KEY_ID and ALPACA_SECRET_KEY are set (use paper-trading keys; the default endpoint is Alpaca's
paper API), the daily job sends a market order for each mirrored simulation that signals a trade: a buy of the
simulation's starting capital (as a notional, fractional order), or closing the position on a sell.
"""

from __future__ import annotations

from typing import Any

import requests

from backend.config import logger, settings


def configured() -> bool:
    return bool(settings.alpaca_key_id and settings.alpaca_secret_key)


def _headers() -> dict[str, str]:
    return {"APCA-API-KEY-ID": settings.alpaca_key_id, "APCA-API-SECRET-KEY": settings.alpaca_secret_key}


def mirror(symbol: str, signal: str, notional: float) -> dict[str, Any]:
    """Send the order for a buy or sell signal; returns {"ok", "detail"}."""
    if not configured():
        return {"ok": False, "detail": "Alpaca is not configured"}
    if "." in symbol or "-" in symbol or "^" in symbol:
        return {"ok": False, "detail": "Alpaca mirroring supports US stocks only"}
    base = settings.alpaca_base_url.rstrip("/")
    try:
        if signal == "buy":
            response = requests.post(
                f"{base}/v2/orders",
                json={
                    "symbol": symbol,
                    "notional": round(notional, 2),
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                },
                headers=_headers(),
                timeout=15,
            )
        elif signal == "sell":
            response = requests.delete(f"{base}/v2/positions/{symbol}", headers=_headers(), timeout=15)
        else:
            return {"ok": False, "detail": f"No order for a {signal} signal"}
    except requests.RequestException as exc:
        logger.warning("Alpaca order for %s failed: %s", symbol, exc)
        return {"ok": False, "detail": str(exc)[:200]}
    if not response.ok:
        return {"ok": False, "detail": f"Alpaca HTTP {response.status_code}: {response.text[:200]}"}
    return {"ok": True, "detail": f"{signal} order sent for {symbol}"}
