"""Portfolio backtests: one strategy run on a basket of stocks at once.

On each rebalance day (the first trading day of the week, month or quarter) the basket picks which stocks to
hold (all of them, or the top N by 6-month momentum) and how much of the portfolio each gets:

* equal: the same weight each.
* inverse-vol: weight proportional to 1 / volatility over the last 63 days, so calm stocks get more money.
* risk-parity: equal risk contribution from each stock, using the 63-day covariance matrix, so correlated
  stocks share one risk budget.

Each stock's strategy signal then decides whether its slice is invested (long, or short when allowed) or
left in cash; the cash is not redistributed. Between rebalances weights drift with prices, and a stock is
only traded when its signal changes. Decisions are made at a day's close and filled at the next day's close,
and every buy and sell pays that stock's trading costs.

Only days on which every stock in the basket traded are used, so mixing exchanges with different holidays
drops those days.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from backend import risk, strategies

WEIGHTINGS = ("equal", "inverse-vol", "risk-parity")
REBALANCES = ("weekly", "monthly", "quarterly")
VOL_WINDOW = 63
MOMENTUM_WINDOW = 126


def align(data: dict[str, tuple[list[float], list[str]]]) -> tuple[list[str], dict[str, list[float]]]:
    """Closes on the days every symbol traded, as (timestamps, {symbol: closes})."""
    by_day = {
        symbol: {ts[:10]: (ts, close) for close, ts in zip(closes, timestamps, strict=True)}
        for symbol, (closes, timestamps) in data.items()
    }
    common = sorted(set.intersection(*(set(days) for days in by_day.values()))) if by_day else []
    first = next(iter(by_day.values()), {})
    timestamps = [first[day][0] for day in common]
    return timestamps, {symbol: [days[day][1] for day in common] for symbol, days in by_day.items()}


BARS_PER_PERIOD = {"weekly": 5, "monthly": 21, "quarterly": 63}


def _period(day: str, rebalance: str, index: int) -> tuple[int, ...]:
    try:
        d = date.fromisoformat(day[:10])
    except ValueError:  # undated data: count trading days instead
        return (index // BARS_PER_PERIOD[rebalance],)
    if rebalance == "weekly":
        year, week, _ = d.isocalendar()
        return (year, week)
    if rebalance == "quarterly":
        return (d.year, (d.month - 1) // 3)
    return (d.year, d.month)


def _returns(closes: list[float], end: int, window: int) -> list[float]:
    begin = max(1, end - window + 1)
    return [closes[i] / closes[i - 1] - 1 for i in range(begin, end + 1)]


def risk_parity(cov: list[list[float]], sweeps: int = 100) -> list[float]:
    """Equal-risk-contribution weights by cyclical coordinate descent (each stock's risk share = 1/n)."""
    n = len(cov)
    w = [1.0 / n] * n
    budget = 1.0 / n
    for _ in range(sweeps):
        for i in range(n):
            if cov[i][i] <= 0:
                continue
            c = sum(cov[i][j] * w[j] for j in range(n) if j != i)
            w[i] = (-c + math.sqrt(c * c + 4 * cov[i][i] * budget)) / (2 * cov[i][i])
    total = sum(w)
    return [x / total for x in w] if total > 0 else [1.0 / n] * n


def target_weights(closes: dict[str, list[float]], t: int, weighting: str, top_n: int) -> dict[str, float]:
    symbols = list(closes)
    if top_n and top_n < len(symbols):
        momentum = {s: closes[s][t] / closes[s][max(0, t - MOMENTUM_WINDOW)] - 1 for s in symbols}
        symbols = sorted(symbols, key=lambda s: momentum[s], reverse=True)[:top_n]
    weights = {s: 0.0 for s in closes}
    if not symbols:
        return weights
    if weighting == "equal":
        for s in symbols:
            weights[s] = 1 / len(symbols)
        return weights
    series = {s: _returns(closes[s], t, VOL_WINDOW) for s in symbols}
    if weighting == "inverse-vol":
        inverse = {s: 1 / (risk.stdev(r) or 1e-4) for s, r in series.items()}
        total = sum(inverse.values())
        for s in symbols:
            weights[s] = inverse[s] / total
        return weights
    length = min(len(r) for r in series.values())
    rows = [series[s][-length:] for s in symbols]
    means = [sum(r) / length for r in rows]
    cov = [
        [
            sum((a - means[i]) * (b - means[j]) for a, b in zip(rows[i], rows[j], strict=True)) / max(length - 1, 1)
            for j in range(len(rows))
        ]
        for i in range(len(rows))
    ]
    for s, w in zip(symbols, risk_parity(cov), strict=True):
        weights[s] = w
    return weights


def run(
    data: dict[str, tuple[list[float], list[str]]],
    strategy_id: str,
    params: dict[str, float],
    rules: dict[str, Any] | None = None,
    *,
    weighting: str = "equal",
    rebalance: str = "monthly",
    top_n: int = 0,
    costs: dict[str, tuple[float, float]] | None = None,
    allow_short: bool = False,
    eval_days: int = 730,
    benchmark: tuple[list[str], list[float]] | None = None,
    benchmark_info: tuple[str, str] | None = None,
    risk_free: float = 0.0,
) -> dict[str, Any]:
    timestamps, closes = align(data)
    symbols = list(closes)
    warm = max(strategies.warmup(strategy_id, params, rules), VOL_WINDOW, MOMENTUM_WINDOW if top_n else 0)
    if len(timestamps) < warm + 60:
        raise ValueError("Not enough shared price history for this basket")
    start = max(strategies.evaluation_start(timestamps, eval_days), warm)
    last = len(timestamps) - 1
    signals = {
        s: [float(x) for x in strategies.signals(strategy_id, closes[s], params, rules, allow_short=allow_short)[0]]
        for s in symbols
    }
    costs = costs or {}

    holdings = {s: 0.0 for s in symbols}
    cash, value = 1.0, 1.0
    base = {s: 0.0 for s in symbols}
    pending: dict[str, float] = {}
    equity: list[float] = []
    daily: list[float] = []
    contribution = {s: 0.0 for s in symbols}
    weight_sum = {s: 0.0 for s in symbols}
    days_held = {s: 0 for s in symbols}
    turnover = total_cost = 0.0
    rebalances = 0
    for t in range(start, last + 1):
        before = value
        if t > start:
            for s in symbols:
                move = closes[s][t] / closes[s][t - 1] - 1
                contribution[s] += holdings[s] * move / before if before else 0.0
                holdings[s] *= 1 + move
            value = cash + sum(holdings.values())
            for s, weight in pending.items():
                trade = weight * value - holdings[s]
                buy, sell = costs.get(s, (10.0, 10.0))
                cost = abs(trade) * (buy if trade > 0 else sell) / 10_000
                cash -= trade + cost
                holdings[s] += trade
                turnover += abs(trade) / value if value else 0.0
                total_cost += cost
            pending = {}
            value = max(cash + sum(holdings.values()), 0.0)
            daily.append(value / before - 1 if before else 0.0)
            for s in symbols:
                weight_sum[s] += holdings[s] / value if value else 0.0
                days_held[s] += int(abs(holdings[s]) > 1e-12)
        equity.append(value)
        # Decide at this close; the orders fill at the next close.
        new_period = t == start or _period(timestamps[t], rebalance, t) != _period(timestamps[t - 1], rebalance, t - 1)
        if new_period:
            base = target_weights({s: closes[s][: t + 1] for s in symbols}, t, weighting, top_n)
            rebalances += 1
            pending = {s: base[s] * signals[s][t] for s in symbols}
        else:
            pending = {s: base[s] * signals[s][t] for s in symbols if signals[s][t] != signals[s][t - 1]}

    stats = risk.summary(daily, [1.0, *equity], risk_free)
    stats["annualizedReturn"] = risk.annualise(stats["totalReturn"], max(last - start, 1))
    # Equal-weight buy & hold of the same stocks, never rebalanced.
    hold = [sum(closes[s][t] / closes[s][start] for s in symbols) / len(symbols) for t in range(start, last + 1)]
    hold_stats = risk.summary([hold[k] / hold[k - 1] - 1 for k in range(1, len(hold))], hold, risk_free)
    bench_symbol, bench_name = benchmark_info or (risk.BENCHMARK_SYMBOL, risk.BENCHMARK_NAME)
    index = (
        risk.benchmark(
            timestamps[start:], equity, *benchmark, symbol=bench_symbol, name=bench_name, risk_free=risk_free
        )
        if benchmark
        else None
    )
    drawdown = risk.drawdowns(equity)
    days = max(last - start, 1)
    return {
        "symbols": symbols,
        "strategyId": strategy_id,
        "parameters": params,
        "weighting": weighting,
        "rebalance": rebalance,
        "topN": top_n,
        "metrics": {
            **stats,
            **risk.tail_risk(daily),
            "equalWeightReturn": hold_stats["totalReturn"],
            "excessReturn": stats["totalReturn"] - hold_stats["totalReturn"],
            "turnover": turnover / days * risk.TRADING_DAYS,
            "costPaid": total_cost,
            "rebalances": rebalances,
        },
        "equalWeight": hold_stats,
        "benchmark": index,
        "holdings": [
            {
                "symbol": s,
                "avgWeight": weight_sum[s] / days,
                "finalWeight": holdings[s] / value if value else 0.0,
                "contribution": contribution[s],
                "timeInMarket": days_held[s] / days,
                "buyHoldReturn": closes[s][last] / closes[s][start] - 1,
            }
            for s in sorted(symbols, key=lambda s: contribution[s], reverse=True)
        ],
        "curve": [
            {"timestamp": timestamps[t], "equity": equity[k], "equalWeight": hold[k], "drawdown": drawdown[k]}
            for k, t in enumerate(range(start, last + 1))
        ],
        "monthly": risk.monthly_returns(timestamps[start:], equity, 1.0),
        "period": {"start": timestamps[start], "end": timestamps[last], "days": last - start + 1},
    }


def cross_section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How consistent a strategy is across stocks: a real edge should show up on most of them."""
    if not rows:
        return {"count": 0}
    sharpes = sorted(r["sharpe"] for r in rows)

    def quantile(q: float) -> float:
        return sharpes[min(len(sharpes) - 1, int(q * (len(sharpes) - 1) + 0.5))]

    return {
        "count": len(rows),
        "medianSharpe": quantile(0.5),
        "sharpeP25": quantile(0.25),
        "sharpeP75": quantile(0.75),
        "shareProfitable": sum(1 for r in rows if r["totalReturn"] > 0) / len(rows),
        "shareBeatBuyHold": sum(1 for r in rows if r["excessReturn"] > 0) / len(rows),
        "shareSharpeAboveBuyHold": sum(1 for r in rows if r["sharpe"] > r["buyHoldSharpe"]) / len(rows),
    }
