"""Option-income strategies: covered calls and cash-secured puts, with modelled premiums.

Historical option prices aren't freely available, so premiums are priced with Black-Scholes using the stock's
recent realised volatility times a volatility premium (implied volatility usually trades above realised;
1.1 by default). Results are therefore an approximation of what the strategy earns, not a replay of real
quotes. Every option is held to expiry and settled in cash, and each new option pays a trading cost.

* Covered call: own the stock and sell a call `otm` above the price every `days` trading days. You keep the
  premium but give up gains above the strike.
* Cash-secured put: keep cash (earning the risk-free rate) and sell a put `otm` below the price. You keep the
  premium but pay the shortfall when the stock ends below the strike.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

from backend import risk

_N = NormalDist()
STRATEGIES = ("covered-call", "cash-secured-put")


def black_scholes(kind: str, spot: float, strike: float, years: float, rate: float, vol: float) -> float:
    """European option price (kind "call" or "put")."""
    if years <= 0 or vol <= 0:
        return max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + (rate + vol * vol / 2) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    if kind == "call":
        return spot * _N.cdf(d1) - strike * math.exp(-rate * years) * _N.cdf(d2)
    return strike * math.exp(-rate * years) * _N.cdf(-d2) - spot * _N.cdf(-d1)


def _realised_vol(closes: list[float], t: int, window: int = 20) -> float:
    returns = [closes[i] / closes[i - 1] - 1 for i in range(max(1, t - window + 1), t + 1)]
    return risk.stdev(returns) * math.sqrt(risk.TRADING_DAYS) if len(returns) > 2 else 0.2


def backtest(
    closes: list[float],
    timestamps: list[str],
    start: int,
    *,
    strategy: str = "covered-call",
    otm: float = 0.05,
    days: int = 21,
    vol_premium: float = 1.1,
    risk_free: float = 0.04,
    cost_bps: float = 5.0,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
    last = len(closes) - 1
    if last - start < days * 2:
        raise ValueError("Not enough price history for this option strategy")
    daily_rate = risk_free / risk.TRADING_DAYS
    cost = cost_bps / 10_000
    call = strategy == "covered-call"
    kind = "call" if call else "put"

    equity, returns = [1.0], []
    shares = 0.0  # covered call: stock owned
    cash = 0.0 if call else 1.0
    sold = strike = vol = 0.0  # number of options written, their strike and the volatility they were priced at
    expiry = start
    premiums = 0.0
    assigned = rolls = 0
    prev_value = 1.0
    if call:
        shares = (1 - cost) / closes[start]  # buy the stock on day one
    for t in range(start, last + 1):
        price = closes[t]
        if t > start:
            cash *= 1 + daily_rate  # idle cash earns the risk-free rate
        if t == expiry and t > start and sold:
            payout = max(price - strike, 0.0) if call else max(strike - price, 0.0)
            cash -= sold * payout  # settled in cash at expiry
            assigned += int(payout > 0)
            sold = 0.0
        if t == expiry and t + days <= last:
            # Write the next option on the whole account.
            account = shares * price + cash
            vol = _realised_vol(closes, t) * vol_premium
            strike = price * (1 + otm) if call else price * (1 - otm)
            sold = shares if call else account / strike  # calls covered by shares, puts by cash at the strike
            premium = black_scholes(kind, price, strike, days / risk.TRADING_DAYS, risk_free, vol)
            cash += sold * premium - sold * price * cost
            premiums += sold * premium / account if account else 0.0
            expiry = t + days
            rolls += 1
        remaining = max(expiry - t, 0) / risk.TRADING_DAYS
        option = black_scholes(kind, price, strike, remaining, risk_free, vol) if sold else 0.0
        value = shares * price + cash - sold * option
        if t > start:
            returns.append(value / prev_value - 1 if prev_value else 0.0)
            equity.append(value)
        prev_value = value

    hold = [closes[t] / closes[start] for t in range(start, last + 1)]
    stats = risk.summary(returns, equity, risk_free)
    hold_stats = risk.summary([hold[k] / hold[k - 1] - 1 for k in range(1, len(hold))], hold, risk_free)
    years = (last - start) / risk.TRADING_DAYS
    drawdown = risk.drawdowns(equity)
    return {
        "strategy": strategy,
        "otm": otm,
        "days": days,
        "volPremium": vol_premium,
        "metrics": {
            **stats,
            "buyHoldReturn": hold_stats["totalReturn"],
            "excessReturn": stats["totalReturn"] - hold_stats["totalReturn"],
            "premiumYield": premiums / years if years else 0.0,
            "rolls": rolls,
            "assigned": assigned,
            "assignedShare": assigned / rolls if rolls else 0.0,
        },
        "buyHold": hold_stats,
        "curve": [
            {"timestamp": timestamps[t], "equity": equity[k], "buyHold": hold[k], "drawdown": drawdown[k]}
            for k, t in enumerate(range(start, last + 1))
        ],
        "period": {"start": timestamps[start], "end": timestamps[last], "days": last - start + 1},
    }
