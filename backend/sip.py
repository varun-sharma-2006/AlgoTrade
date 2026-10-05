"""SIP (systematic investment plan) simulator.

Invests a fixed amount on the first trading day of every month, optionally raising it each year (step-up),
and compares three ways of putting the same money to work:

* SIP: buy every month regardless of price.
* Lump sum: invest the SIP's total contributions on the first day.
* Strategy-timed SIP: each month's instalment waits in cash and is invested only while a strategy's signal is
  on (for example "price above its 200-day average"); it never sells.

Returns are reported as XIRR (the annual return that accounts for when each instalment went in), and the
tax due if everything were redeemed on the last day is worked out lot by lot (first in, first out).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from backend import tax


def _date(timestamp: str) -> date:
    return date.fromisoformat(timestamp[:10])


def xirr(flows: list[tuple[date, float]]) -> float | None:
    """Annual rate r with sum(cf / (1 + r) ** (days / 365)) = 0, by bisection."""
    if not flows or all(cf >= 0 for _, cf in flows) or all(cf <= 0 for _, cf in flows):
        return None
    first = flows[0][0]

    def npv(rate: float) -> float:
        return sum(cf / (1 + rate) ** ((day - first).days / 365) for day, cf in flows)

    low, high = -0.99, 10.0
    if npv(low) * npv(high) > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2
        if npv(low) * npv(mid) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def instalment_days(timestamps: list[str], start: int) -> list[int]:
    """Index of the first trading day of each month from `start`."""
    days = []
    for i in range(start, len(timestamps)):
        if i == start or timestamps[i][:7] != timestamps[i - 1][:7]:
            days.append(i)
    return days


def run(
    closes: list[float],
    timestamps: list[str],
    *,
    monthly: float,
    start: int,
    step_up: float = 0.0,
    fee_bps: float = 0.0,
    sell_fee_bps: float = 0.0,
    timing: list[int] | None = None,
    region: str = "US",
    crypto: bool = False,
    currency: str = "USD",
) -> dict[str, Any]:
    buy_cost, sell_cost = fee_bps / 10_000, sell_fee_bps / 10_000
    months = instalment_days(timestamps, start)
    if len(months) < 2:
        raise ValueError("A SIP needs at least two months of prices")
    last = len(closes) - 1
    first_day = _date(timestamps[months[0]])

    amounts = []
    for i in months:
        years_in = (_date(timestamps[i]) - first_day).days // 365
        amounts.append(monthly * (1 + step_up) ** years_in)
    total = sum(amounts)
    by_index = dict(zip(months, amounts, strict=True))

    lots: list[dict[str, Any]] = []  # SIP purchases: date, units, cost
    sip_units = timed_units = timed_cash = 0.0
    timed_lots: list[dict[str, Any]] = []
    lump_units = total * (1 - buy_cost) / closes[months[0]]
    invested = 0.0
    series = []
    for t in range(months[0], last + 1):
        price = closes[t]
        if t in by_index:
            amount = by_index[t]
            invested += amount
            units = amount * (1 - buy_cost) / price
            sip_units += units
            lots.append({"date": timestamps[t], "units": units, "cost": amount})
            timed_cash += amount
        if timing is not None and timed_cash > 0 and timing[t] > 0:
            units = timed_cash * (1 - buy_cost) / price
            timed_units += units
            timed_lots.append({"date": timestamps[t], "units": units, "cost": timed_cash})
            timed_cash = 0.0
        if t in by_index or t == last:
            series.append(
                {
                    "date": timestamps[t][:10],
                    "invested": invested,
                    "sip": sip_units * price,
                    "lumpSum": lump_units * price,
                    "timed": timed_units * price + timed_cash if timing is not None else None,
                }
            )

    end_price = closes[last]
    end_day = _date(timestamps[last])
    flows = [(_date(timestamps[i]), -a) for i, a in zip(months, amounts, strict=True)]
    sip_value = sip_units * end_price
    lump_value = lump_units * end_price
    timed_value = timed_units * end_price + timed_cash

    def redeem_tax(purchases: list[dict[str, Any]]) -> float:
        trades = [
            {
                "entryDate": lot["date"],
                "exitDate": timestamps[last],
                "side": "long",
                "gain": lot["units"] * end_price * (1 - sell_cost) - lot["cost"],
            }
            for lot in purchases
        ]
        return tax.tax_report(trades, 1.0, 1.0, region=region, crypto=crypto, currency=currency)["totalTax"]

    years = max((end_day - first_day).days / 365, 1 / 12)
    result = {
        "months": len(months),
        "years": years,
        "invested": total,
        "currency": currency,
        "sip": {
            "value": sip_value,
            "gain": sip_value - total,
            "xirr": xirr([*flows, (end_day, sip_value)]),
            "taxIfRedeemed": redeem_tax(lots),
        },
        "lumpSum": {
            "value": lump_value,
            "gain": lump_value - total,
            "cagr": (lump_value / total) ** (1 / years) - 1 if total and lump_value > 0 else None,
            "taxIfRedeemed": redeem_tax([{"date": timestamps[months[0]], "units": lump_units, "cost": total}]),
        },
        "series": series,
        "period": {"start": timestamps[months[0]], "end": timestamps[last]},
        "lastInstalment": amounts[-1],
    }
    if timing is not None:
        result["timed"] = {
            "value": timed_value,
            "gain": timed_value - total,
            "xirr": xirr([*flows, (end_day, timed_value)]),
            "cashWaiting": timed_cash,
            "taxIfRedeemed": redeem_tax(timed_lots),
        }
    return result
