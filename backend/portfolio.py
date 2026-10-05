"""Values paper-trading simulations: replays each one's strategy on real daily prices from its start date.

The rules match the backtester: the position decided at a day's close is taken at that close or at the next
day's open (the server's EXECUTION setting), intraday stops fill at their level, and every buy and sell pays
the trading fee and slippage. The money starts in cash on the start date.

Starting capital is in the base currency (USD by default). A simulation on a stock quoted in another currency
converts the capital at the start date's exchange rate and converts the value back at each day's rate, so
the currency's move is part of the profit or loss, as it would be for a real investor.

Every fill is recorded in a ledger with its date, side, price, shares, fee and slippage.
"""

from __future__ import annotations

from typing import Any

from backend import strategies


def _day(timestamp: str) -> str:
    return timestamp[:10]


def align_fx(days: list[str], rates: dict[str, float] | None) -> list[float] | None:
    """The exchange rate on each day, carried forward over days the currency market didn't quote."""
    if not rates:
        return None
    if "*" in rates:
        return [rates["*"]] * len(days)
    known = sorted(rates)
    out: list[float] = []
    k, last = 0, rates[known[0]]
    for day in days:
        while k < len(known) and known[k] <= day:
            last = rates[known[k]]
            k += 1
        out.append(last)
    return out


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
    sell_fee_bps: float | None = None,
    slippage_bps: float = 0.0,
    bars: strategies.Bars | None = None,
    execution: str = "close",
    fx: list[float] | None = None,
) -> dict[str, Any]:
    days = [_day(ts) for ts in timestamps]
    start = next((i for i, day in enumerate(days) if day >= start_date), len(days) - 1)
    last = len(closes) - 1
    planned = strategies.plan(strategy_id, closes, params, rules, bars=bars, execution=execution)
    opens = (bars or {}).get("open")
    if opens is not None and None in opens:
        opens = None
    sell_fee = fee_bps if sell_fee_bps is None else sell_fee_bps
    run = strategies.simulate(
        closes,
        planned["target"],
        start,
        last,
        fee_bps + slippage_bps,
        sell_cost_bps=sell_fee + slippage_bps,
        opens=opens,
        execution=execution,
        fills=planned["fills"],
    )
    rate = fx or [1.0] * len(closes)
    capital_local = capital / rate[start]  # base-currency capital converted at the start date
    history = [
        {"date": days[t], "value": capital_local * run["equity"][k] * rate[t]}
        for k, t in enumerate(range(start, last + 1))
    ]

    ledger = []
    for fill in run["fills"]:
        traded = abs(fill["to"] - fill["from"]) * fill["valueBefore"] * capital_local
        buying = fill["to"] > fill["from"]
        ledger.append(
            {
                "date": days[fill["index"]],
                "side": "buy" if buying else "sell",
                "price": fill["price"],
                "shares": traded / fill["price"] if fill["price"] else 0.0,
                "notional": traded,
                "fee": traded * (fee_bps if buying else sell_fee) / 10_000,
                "slippage": traded * slippage_bps / 10_000,
                "stop": fill["to"] == 0 and planned["fills"].get(fill["index"]) == fill["price"],
            }
        )

    value = history[-1]["value"]
    first_close = closes[start]
    fx_move = rate[last] / rate[start]
    buy_hold_value = capital * (1 - (fee_bps + slippage_bps) / 10_000) * closes[-1] / first_close * fx_move
    previous = history[-2]["value"] if len(history) > 1 else capital
    held = run["held"]
    pending = None
    if execution == "next_open" and opens is not None and planned["target"][last] != held:
        pending = "buy" if planned["target"][last] > held else "sell"
    return {
        "value": value,
        "pnl": value - capital,
        "pnlPct": value / capital - 1,
        "dayChange": value - previous,
        "dayChangePct": value / previous - 1 if previous else 0.0,
        "inMarket": bool(held),
        "shares": held * capital_local * run["equity"][-1] / closes[-1] if held else 0.0,
        "lastPrice": closes[-1],
        "startDate": days[start],
        "startPrice": first_close,
        "buyHoldValue": buy_hold_value,
        "fxReturn": fx_move - 1 if fx else 0.0,
        "trades": len(ledger),
        "pendingOrder": pending,
        "ledger": ledger[-100:],
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
