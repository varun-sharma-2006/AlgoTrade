"""Risk statistics for daily return series, and comparison against a benchmark index.

All ratios are annualised with 252 trading days and a risk-free rate of 0.
"""

from __future__ import annotations

import math
from typing import Any

TRADING_DAYS = 252
BENCHMARK_SYMBOL = "^GSPC"
BENCHMARK_NAME = "S&P 500"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    """Sample standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def drawdowns(equity: list[float]) -> list[float]:
    """Distance below the running peak for each point, as a fraction <= 0."""
    peak = equity[0] if equity else 0.0
    result = []
    for value in equity:
        peak = max(peak, value)
        result.append(value / peak - 1 if peak else 0.0)
    return result


def annualise(total_return: float, periods: int) -> float:
    if total_return <= -1:
        return -1.0
    return (1 + total_return) ** (TRADING_DAYS / max(periods, 1)) - 1


def summary(daily_returns: list[float], equity: list[float]) -> dict[str, float]:
    """Return, volatility, Sharpe, Sortino, drawdown and Calmar for one equity curve."""
    m = mean(daily_returns)
    sd = stdev(daily_returns)
    # Downside deviation: only losing days count as risk (target return 0).
    downside = math.sqrt(sum(min(r, 0.0) ** 2 for r in daily_returns) / len(daily_returns)) if daily_returns else 0.0
    total = equity[-1] / equity[0] - 1 if equity and equity[0] else 0.0
    annual = annualise(total, len(daily_returns))
    max_dd = abs(min(drawdowns(equity), default=0.0))
    return {
        "totalReturn": total,
        "annualizedReturn": annual,
        "volatility": sd * math.sqrt(TRADING_DAYS),
        "sharpe": m / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0,
        "sortino": m / downside * math.sqrt(TRADING_DAYS) if downside > 0 else 0.0,
        "maxDrawdown": max_dd,
        "calmar": annual / max_dd if max_dd > 0 else 0.0,
    }


def _day(timestamp: str) -> str:
    return timestamp[:10]


def benchmark(
    timestamps: list[str], equity: list[float], bench_timestamps: list[str], bench_closes: list[float]
) -> dict[str, Any] | None:
    """Compare a strategy's equity curve with an index over the dates both traded.

    Returns the index's own statistics, plus the strategy's beta, alpha and correlation to it.
    """
    by_day = {_day(ts): close for ts, close in zip(bench_timestamps, bench_closes, strict=True)}
    common = [i for i, ts in enumerate(timestamps) if _day(ts) in by_day]
    if len(common) < 20:
        return None
    strat: list[float] = []
    index: list[float] = []
    for prev, cur in zip(common[:-1], common[1:], strict=True):
        strat.append(equity[cur] / equity[prev] - 1)
        index.append(by_day[_day(timestamps[cur])] / by_day[_day(timestamps[prev])] - 1)
    index_equity = [by_day[_day(timestamps[i])] for i in common]
    stats = summary(index, index_equity)

    ms, mi = mean(strat), mean(index)
    cov = sum((s - ms) * (b - mi) for s, b in zip(strat, index, strict=True)) / (len(strat) - 1)
    var_i = stdev(index) ** 2
    sd_s = stdev(strat)
    beta = cov / var_i if var_i > 0 else 0.0
    return {
        "symbol": BENCHMARK_SYMBOL,
        "name": BENCHMARK_NAME,
        **stats,
        "beta": beta,
        # Annualised return the strategy earned beyond what its market exposure explains.
        "alpha": (ms - beta * mi) * TRADING_DAYS,
        "correlation": cov / (sd_s * math.sqrt(var_i)) if sd_s > 0 and var_i > 0 else 0.0,
    }
