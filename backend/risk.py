"""Risk statistics for daily return series, and comparison against a benchmark index.

All ratios are annualised with 252 trading days. Sharpe, Sortino and alpha subtract a risk-free rate
(annual, 0 by default), spread evenly over the trading days.
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


def summary(daily_returns: list[float], equity: list[float], risk_free: float = 0.0) -> dict[str, float]:
    """Return, volatility, Sharpe, Sortino, drawdown and Calmar for one equity curve."""
    rf = risk_free / TRADING_DAYS
    m = mean(daily_returns) - rf
    sd = stdev(daily_returns)
    # Downside deviation: only days below the risk-free return count as risk.
    downside = (
        math.sqrt(sum(min(r - rf, 0.0) ** 2 for r in daily_returns) / len(daily_returns)) if daily_returns else 0.0
    )
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
    timestamps: list[str],
    equity: list[float],
    bench_timestamps: list[str],
    bench_closes: list[float],
    *,
    symbol: str = BENCHMARK_SYMBOL,
    name: str = BENCHMARK_NAME,
    risk_free: float = 0.0,
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
    stats = summary(index, index_equity, risk_free)

    ms, mi = mean(strat), mean(index)
    cov = sum((s - ms) * (b - mi) for s, b in zip(strat, index, strict=True)) / (len(strat) - 1)
    var_i = stdev(index) ** 2
    sd_s = stdev(strat)
    beta = cov / var_i if var_i > 0 else 0.0
    rf = risk_free / TRADING_DAYS
    return {
        "symbol": symbol,
        "name": name,
        **stats,
        "beta": beta,
        # Annualised return the strategy earned beyond what its market exposure explains (Jensen's alpha).
        "alpha": ((ms - rf) - beta * (mi - rf)) * TRADING_DAYS,
        "correlation": cov / (sd_s * math.sqrt(var_i)) if sd_s > 0 and var_i > 0 else 0.0,
    }


def aligned_returns(
    timestamps: list[str], bench_timestamps: list[str], bench_closes: list[float]
) -> list[float | None]:
    """The index's daily return on each of `timestamps` (None where the index didn't trade both days)."""
    by_day = {_day(ts): close for ts, close in zip(bench_timestamps, bench_closes, strict=True)}
    out: list[float | None] = [None]
    for prev, cur in zip(timestamps[:-1], timestamps[1:], strict=True):
        a, b = by_day.get(_day(prev)), by_day.get(_day(cur))
        out.append(b / a - 1 if a and b else None)
    return out


def tail_risk(daily_returns: list[float], level: float = 0.95) -> dict[str, float]:
    """Historical one-day Value at Risk and Conditional VaR (expected shortfall), as positive losses."""
    if not daily_returns:
        return {"var95": 0.0, "cvar95": 0.0}
    ordered = sorted(daily_returns)
    cut = max(1, int(len(ordered) * (1 - level)))
    tail = ordered[:cut]
    return {"var95": max(0.0, -ordered[cut - 1]), "cvar95": max(0.0, -mean(tail))}


def profit_factor(trade_returns: list[float]) -> float | None:
    """Gross gains of winning trades divided by gross losses of losing trades (None without losing trades)."""
    gains = sum(r for r in trade_returns if r > 0)
    losses = -sum(r for r in trade_returns if r < 0)
    return gains / losses if losses > 0 else None


def monthly_returns(timestamps: list[str], equity: list[float], initial: float | None = None) -> list[dict[str, Any]]:
    """Return for each calendar month, measured from the previous month's last value (or `initial`)."""
    months: list[dict[str, Any]] = []
    previous = initial if initial is not None else (equity[0] if equity else 1.0)
    for k, ts in enumerate(timestamps):
        key = ts[:7]
        if not months or months[-1]["month"] != key:
            if months:
                previous = equity[k - 1]
            months.append({"month": key, "year": int(key[:4]) if key[:4].isdigit() else 0, "start": previous})
        months[-1]["end"] = equity[k]
    return [
        {"month": m["month"], "year": m["year"], "return": m["end"] / m["start"] - 1 if m["start"] else 0.0}
        for m in months
    ]


def rolling_sharpe(daily_returns: list[float], window: int = 126, risk_free: float = 0.0) -> list[float | None]:
    """Annualised Sharpe ratio over the trailing `window` days, aligned with an equity curve (first value None)."""
    rf = risk_free / TRADING_DAYS
    out: list[float | None] = [None] * (len(daily_returns) + 1)
    s = s2 = 0.0
    for i, r in enumerate(daily_returns):
        s += r
        s2 += r * r
        if i >= window:
            old = daily_returns[i - window]
            s -= old
            s2 -= old * old
        if i + 1 >= window:
            m = s / window
            var = max((s2 - window * m * m) / (window - 1), 0.0)
            out[i + 1] = (m - rf) / math.sqrt(var) * math.sqrt(TRADING_DAYS) if var > 0 else 0.0
    return out


def rolling_beta(
    daily_returns: list[float], index_returns: list[float | None], window: int = 126
) -> list[float | None]:
    """Trailing beta to the index, aligned with an equity curve. `index_returns` is aligned the same way."""
    out: list[float | None] = [None] * (len(daily_returns) + 1)
    pairs = [(r, index_returns[k + 1]) for k, r in enumerate(daily_returns)]
    for i in range(window - 1, len(pairs)):
        chunk = [(a, b) for a, b in pairs[i + 1 - window : i + 1] if b is not None]
        if len(chunk) < window // 2:
            continue
        ma = mean([a for a, _ in chunk])
        mb = mean([b for _, b in chunk])
        cov = sum((a - ma) * (b - mb) for a, b in chunk)
        var = sum((b - mb) ** 2 for _, b in chunk)
        out[i + 1] = cov / var if var > 0 else None
    return out
