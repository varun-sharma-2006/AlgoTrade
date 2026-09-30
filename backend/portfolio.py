"""Values paper-trading simulations: replays each one's strategy on real daily closes from its start date.

The rules match the backtester: the position decided at a day's close is held through the next day, and
every buy and sell pays the trading fee. The money starts in cash on the start date, so a strategy that is
already "long" buys at that day's close.
"""

from __future__ import annotations

from typing import Any

from backend import strategies


def _day(timestamp: str) -> str:
    return timestamp[:10]


def value_simulation(
    closes: list[float],
    timestamps: list[str],
    *,
    strategy_id: str,
    params: dict[str, float],
    rules: dict[str, Any] | None,
    start_date: str,
    capital: float,
    fee_bps: float,
) -> dict[str, Any]:
    days = [_day(ts) for ts in timestamps]
    start = next((i for i, day in enumerate(days) if day >= start_date), len(days) - 1)
    target, _ = strategies.signals(strategy_id, closes, params, rules)
    fee = fee_bps / 10_000

    value, held, trades = capital, 0, 0
    history = []
    for t in range(start, len(closes)):
        if t > start and held:
            value *= closes[t] / closes[t - 1]
        if target[t] != held:
            value *= 1 - fee
            held = target[t]
            trades += 1
        history.append({"date": days[t], "value": value})

    first_close = closes[start]
    buy_hold_value = capital * (1 - fee) * closes[-1] / first_close
    previous = history[-2]["value"] if len(history) > 1 else capital
    return {
        "value": value,
        "pnl": value - capital,
        "pnlPct": value / capital - 1,
        "dayChange": value - previous,
        "dayChangePct": value / previous - 1 if previous else 0.0,
        "inMarket": bool(held),
        "shares": value / closes[-1] if held else 0.0,
        "lastPrice": closes[-1],
        "startDate": days[start],
        "startPrice": first_close,
        "buyHoldValue": buy_hold_value,
        "trades": trades,
        "history": history,
    }


def combine(positions: list[dict[str, Any]]) -> dict[str, Any]:
    """Portfolio totals and a daily value history across positions (each counts as its capital until it starts)."""
    all_days = sorted({point["date"] for p in positions for point in p.get("history") or []})
    # Walk every position's history forward in step with the calendar (histories are date-sorted).
    cursors = [0] * len(positions)
    latest = [p["startingCapital"] for p in positions]
    history = []
    for day in all_days:
        for k, p in enumerate(positions):
            points = p.get("history") or []
            while cursors[k] < len(points) and points[cursors[k]]["date"] <= day:
                latest[k] = points[cursors[k]]["value"]
                cursors[k] += 1
        history.append({"date": day, "value": sum(latest)})

    capital = sum(p["startingCapital"] for p in positions)
    value = sum(p["value"] for p in positions)
    day_change = sum(p.get("dayChange", 0.0) for p in positions)
    previous = value - day_change
    allocation = [
        {"symbol": p["symbol"], "value": p["value"], "weight": p["value"] / value if value else 0.0, "id": p["id"]}
        for p in sorted(positions, key=lambda p: p["value"], reverse=True)
    ]
    return {
        "summary": {
            "totalValue": value,
            "totalCapital": capital,
            "pnl": value - capital,
            "pnlPct": value / capital - 1 if capital else 0.0,
            "dayChange": day_change,
            "dayChangePct": day_change / previous if previous else 0.0,
            "positions": len(positions),
            "inMarket": sum(1 for p in positions if p.get("inMarket")),
        },
        "history": history,
        "allocation": allocation,
    }
