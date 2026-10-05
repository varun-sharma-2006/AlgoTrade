"""Is a backtest result skill, luck or overfitting? Four checks:

* Monte Carlo (block bootstrap): resample the strategy's daily returns in 20-day blocks (keeping short-term
  autocorrelation) many times, giving a range of outcomes for return, drawdown and Sharpe instead of the
  single path history happened to take, plus the chance of losing money or trailing buy & hold.
* Parameter sensitivity: the same strategy over a grid of settings. A broad region of good results means a
  robust idea; one bright cell surrounded by poor ones means the settings were fitted to noise.
* Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014): the probability that the Sharpe ratio is above
  what the best of N random strategies would show by luck, given how many settings were tried and the
  returns' skew and fat tails.
* Probability of backtest overfitting (combinatorially symmetric cross-validation): split the history into
  8 blocks; for each way of choosing 4 of them as "in sample", pick the best settings there and see where
  they rank on the other 4. PBO is how often the in-sample winner lands in the bottom half out of sample.
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from statistics import NormalDist
from typing import Any

from backend import risk, strategies

SENSITIVITY_GRIDS: dict[str, dict[str, tuple[str, list[float]]]] = {
    "sma-crossover": {
        "x": ("shortWindow", [5, 10, 15, 20, 30, 40, 50]),
        "y": ("longWindow", [40, 60, 80, 100, 150, 200]),
    },
    "mean-reversion": {"x": ("lookback", [10, 15, 20, 25, 30, 40]), "y": ("deviation", [1.0, 1.5, 2.0, 2.5, 3.0])},
    "trend-follow": {"x": ("channel", [10, 15, 20, 30, 40, 55, 70, 100])},
    "regime-switch": {"x": ("erWindow", [10, 20, 30, 40]), "y": ("erThreshold", [0.15, 0.2, 0.3, 0.4, 0.5])},
    "ml-logistic": {"x": ("threshold", [0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6])},
}
EULER_GAMMA = 0.5772156649
_normal = NormalDist()


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def at(q: float) -> float:
        if not ordered:
            return 0.0
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"p5": at(0.05), "p25": at(0.25), "p50": at(0.5), "p75": at(0.75), "p95": at(0.95)}


def monte_carlo(
    returns: list[float],
    benchmark_returns: list[float] | None = None,
    *,
    paths: int = 500,
    block: int = 20,
    seed: int = 7,
    checkpoints: int = 60,
    periods: float = risk.TRADING_DAYS,
) -> dict[str, Any] | None:
    """Stationary-block bootstrap of daily returns. Strategy and buy & hold are resampled on the same days."""
    total = len(returns)
    if total < 2 * block:
        return None
    rng = random.Random(seed)
    steps = sorted({round(k * total / checkpoints) for k in range(checkpoints + 1)})
    at_step = {s: k for k, s in enumerate(steps)}
    fan: list[list[float]] = [[] for _ in steps]
    finals, drawdowns, sharpes = [], [], []
    beat = 0
    for _ in range(paths):
        equity = peak = bench = 1.0
        worst = 0.0
        s = s2 = 0.0
        done = 0
        fan[0].append(1.0)
        while done < total:
            begin = rng.randrange(0, total - block + 1)
            for j in range(begin, begin + min(block, total - done)):
                r = returns[j]
                equity *= 1 + r
                peak = max(peak, equity)
                worst = min(worst, equity / peak - 1)
                s += r
                s2 += r * r
                if benchmark_returns is not None:
                    bench *= 1 + benchmark_returns[j]
                done += 1
                if done in at_step:
                    fan[at_step[done]].append(equity)
        finals.append(equity - 1)
        drawdowns.append(-worst)
        mean = s / total
        var = (s2 - total * mean * mean) / (total - 1)
        sharpes.append(mean / math.sqrt(var) * math.sqrt(periods) if var > 0 else 0.0)
        beat += int(benchmark_returns is not None and equity > bench)
    return {
        "paths": paths,
        "block": block,
        "finalReturn": _percentiles(finals),
        "maxDrawdown": _percentiles(drawdowns),
        "sharpe": _percentiles(sharpes),
        "probLoss": sum(1 for f in finals if f < 0) / paths,
        "probBeatBuyHold": beat / paths if benchmark_returns is not None else None,
        "fan": [{"step": step, **_percentiles(values)} for step, values in zip(steps, fan, strict=True)],
    }


def _moments(returns: list[float]) -> tuple[float, float, float, float]:
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    sd = math.sqrt(var)
    if sd == 0:
        return mean, 0.0, 0.0, 3.0
    skew = sum((r - mean) ** 3 for r in returns) / n / sd**3
    kurt = sum((r - mean) ** 4 for r in returns) / n / sd**4
    return mean, sd, skew, kurt


def deflated_sharpe(
    returns: list[float], trial_sharpes: list[float], risk_free: float = 0.0, periods: float = risk.TRADING_DAYS
) -> dict[str, Any] | None:
    """Deflated and probabilistic Sharpe ratio. `trial_sharpes` are the daily (not annualised) Sharpe ratios
    of every setting that was tried, including this one. Returns are measured above the risk-free rate."""
    if len(returns) < 30:
        return None
    rf = risk_free / periods
    returns = [r - rf for r in returns]
    mean, sd, skew, kurt = _moments(returns)
    sr = mean / sd if sd else 0.0
    trials = len(trial_sharpes)
    if trials > 1:
        m = sum(trial_sharpes) / trials
        var_sr = sum((x - m) ** 2 for x in trial_sharpes) / (trials - 1)
        expected_max = math.sqrt(var_sr) * (
            (1 - EULER_GAMMA) * _normal.inv_cdf(1 - 1 / trials)
            + EULER_GAMMA * _normal.inv_cdf(1 - 1 / (trials * math.e))
        )
    else:
        expected_max = 0.0
    denominator = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    scale = math.sqrt(len(returns) - 1) / denominator
    annual = math.sqrt(periods)
    return {
        "trials": trials,
        "sharpe": sr * annual,
        "expectedMaxSharpe": expected_max * annual,
        "deflatedSharpe": _normal.cdf((sr - expected_max) * scale),
        "probabilisticSharpe": _normal.cdf(sr * scale),
        "skew": skew,
        "kurtosis": kurt,
    }


def pbo(trial_returns: list[list[float]], slices: int = 8, periods: float = risk.TRADING_DAYS) -> dict[str, Any] | None:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation."""
    trials = len(trial_returns)
    length = min((len(r) for r in trial_returns), default=0)
    if trials < 3 or length < slices * 10:
        return None
    size = length // slices
    # Sums per trial per block, so the Sharpe of any union of blocks is cheap.
    stats = [
        [(sum(r[b * size : (b + 1) * size]), sum(x * x for x in r[b * size : (b + 1) * size])) for b in range(slices)]
        for r in trial_returns
    ]

    def sharpe(n: int, blocks: tuple[int, ...]) -> float:
        s = sum(stats[n][b][0] for b in blocks)
        s2 = sum(stats[n][b][1] for b in blocks)
        count = size * len(blocks)
        mean = s / count
        var = (s2 - count * mean * mean) / (count - 1)
        return mean / math.sqrt(var) if var > 1e-18 else 0.0

    logits = []
    oos_of_winner = []
    everything = set(range(slices))
    for chosen in combinations(range(slices), slices // 2):
        rest = tuple(sorted(everything - set(chosen)))
        in_sample = [sharpe(n, chosen) for n in range(trials)]
        out_sample = [sharpe(n, rest) for n in range(trials)]
        winner = max(range(trials), key=lambda n: in_sample[n])
        score = out_sample[winner]
        below = sum(1 for v in out_sample if v < score)
        ties = sum(1 for v in out_sample if v == score) - 1
        rank = below + 1 + ties / 2  # 1 = worst, trials = best
        omega = rank / (trials + 1)
        logits.append(math.log(omega / (1 - omega)))
        oos_of_winner.append(score * math.sqrt(periods))
    return {
        "pbo": sum(1 for x in logits if x <= 0) / len(logits),
        "combinations": len(logits),
        "trials": trials,
        "slices": slices,
        "medianLogit": sorted(logits)[len(logits) // 2],
        "winnerOutOfSampleSharpe": _percentiles(oos_of_winner)["p50"],
    }


def sensitivity(
    closes: list[float],
    start: int,
    strategy_id: str,
    base_params: dict[str, float],
    buy_cost_bps: float,
    sell_cost_bps: float,
    *,
    opens: list[float] | None = None,
    execution: str = "close",
    allow_short: bool = False,
    risk_free: float = 0.0,
    bars: strategies.Bars | None = None,
    market_closes: list[float | None] | None = None,
    periods: float = risk.TRADING_DAYS,
) -> tuple[dict[str, Any] | None, list[list[float]]]:
    """Sharpe, return and drawdown over a grid of settings (full-size positions), plus each setting's returns."""
    spec = SENSITIVITY_GRIDS.get(strategy_id)
    if not spec:
        return None, []
    x_name, xs = spec["x"]
    y_name, ys = spec.get("y", (None, [None]))
    last = len(closes) - 1
    cells = []
    trial_returns: list[list[float]] = []
    for x in xs:
        for y in ys:
            params = {**strategies.DEFAULT_PARAMS[strategy_id], **base_params, x_name: x}
            if y_name:
                params[y_name] = y
            cell: dict[str, Any] = {"x": x, "y": y}
            if strategies.validate(strategy_id, params) or strategies.warmup(strategy_id, params) >= start:
                cells.append(cell | {"valid": False})
                continue
            target = strategies.plan(
                strategy_id, closes, params, allow_short=allow_short, bars=bars, market_closes=market_closes
            )["target"]
            run = strategies.simulate(
                closes,
                target,
                start,
                last,
                buy_cost_bps,
                sell_cost_bps=sell_cost_bps,
                opens=opens,
                execution=execution,
                periods=periods,
            )
            stats = risk.summary(run["dailyReturns"], [1.0, *run["equity"]], risk_free, periods)
            cells.append(
                cell
                | {
                    "valid": True,
                    "sharpe": stats["sharpe"],
                    "totalReturn": stats["totalReturn"],
                    "maxDrawdown": stats["maxDrawdown"],
                    "trades": len(run["trades"]),
                }
            )
            trial_returns.append(run["dailyReturns"])
    valid = [c for c in cells if c["valid"]]
    sharpes = sorted(c["sharpe"] for c in valid)
    return (
        {
            "xParam": x_name,
            "xValues": xs,
            "yParam": y_name,
            "yValues": ys if y_name else [],
            "cells": cells,
            "current": {x_name: base_params.get(x_name), **({y_name: base_params.get(y_name)} if y_name else {})},
            "positiveShare": sum(1 for s in sharpes if s > 0) / len(sharpes) if sharpes else 0.0,
            "medianSharpe": sharpes[len(sharpes) // 2] if sharpes else 0.0,
            "bestSharpe": sharpes[-1] if sharpes else 0.0,
        },
        trial_returns,
    )


def daily_sharpe(returns: list[float], risk_free: float = 0.0, periods: float = risk.TRADING_DAYS) -> float:
    """Per-bar (not annualised) Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    sd = risk.stdev(returns)
    return (risk.mean(returns) - risk_free / periods) / sd if sd else 0.0


def random_entries(
    closes: list[float],
    start: int,
    trade_bars: list[int],
    actual_return: float,
    cost_bps: float,
    *,
    paths: int = 300,
    seed: int = 11,
) -> dict[str, Any] | None:
    """Would random timing have done as well? Places the same number of trades, with the same holding
    periods, at random non-overlapping times in the window, and ranks the strategy's return among them."""
    last = len(closes) - 1
    span = last - start
    durations = [max(1, b) for b in trade_bars]
    if not durations or sum(durations) >= span:
        return None
    rng = random.Random(seed)
    cost = cost_bps / 10_000
    results = []
    for _ in range(paths):
        order = durations[:]
        rng.shuffle(order)
        free = span - sum(order)
        # Random gaps that add up to the free bars: stars and bars.
        cuts = sorted(rng.randint(0, free) for _ in order)
        gaps = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))]
        t, value = start, 1.0
        for gap, held in zip(gaps, order, strict=True):
            t += gap
            value *= (1 - cost) * closes[min(t + held, last)] / closes[t] * (1 - cost)
            t += held
        results.append(value - 1)
    results.sort()
    beaten = sum(1 for r in results if r < actual_return)
    return {
        "paths": paths,
        "trades": len(durations),
        "percentile": beaten / paths,
        "median": results[len(results) // 2],
        "p95": results[int(0.95 * (paths - 1))],
    }


def sixty_forty(
    stock: tuple[list[str], list[float]],
    bond: tuple[list[str], list[float]],
    start_day: str,
    risk_free: float = 0.0,
) -> dict[str, Any] | None:
    """A 60% stocks / 40% bonds portfolio rebalanced monthly, from `start_day` (a classic passive benchmark)."""
    s_by = {ts[:10]: c for ts, c in zip(*stock, strict=True)}
    b_by = {ts[:10]: c for ts, c in zip(*bond, strict=True)}
    days = sorted(d for d in s_by if d in b_by and d >= start_day[:10])
    if len(days) < 60:
        return None
    value, w_s, w_b = 1.0, 0.6, 0.4
    equity, returns = [1.0], []
    for prev, day in zip(days[:-1], days[1:], strict=True):
        if day[:7] != prev[:7]:
            w_s, w_b = 0.6, 0.4  # rebalance on the first day of each month
        rs, rb = s_by[day] / s_by[prev] - 1, b_by[day] / b_by[prev] - 1
        growth = w_s * (1 + rs) + w_b * (1 + rb)
        w_s, w_b = w_s * (1 + rs) / growth, w_b * (1 + rb) / growth
        returns.append(growth - 1)
        value *= growth
        equity.append(value)
    return risk.summary(returns, equity, risk_free)
