"""Strategy catalogue and an honest backtester.

Each strategy turns daily data into a target position (1 = long, 0 = flat, -1 = short) using only data up to
that day. Position sizing then scales the target (a fixed fraction, or volatility targeting), and the
backtester trades it:

* Execution: a decision made at a day's close fills either at that close ("close") or, more realistically,
  at the next day's open ("next_open"), so overnight gaps are paid like a real trader would pay them.
* Stops: rule-based stop-losses, take-profits and trailing stops are checked against each day's high and low
  when OHLC data is available, and fill at the stop level (or at the open when the price gaps through it).
* Costs: every buy and sell pays a fee plus slippage on the traded value (buys and sells can differ, as with
  Indian STT and stamp duty), and short positions pay a borrow fee for every day they are held.

Each round-trip trade is recorded, so win rate, profit factor, Sharpe and drawdown come from simulated
trades rather than from the underlying stock.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from typing import Any

STRATEGIES: list[dict[str, Any]] = [
    {
        "id": "sma-crossover",
        "name": "Simple moving average crossover",
        "description": "Long while the short SMA is above the long SMA; flat (or short) otherwise. Rides big trends, gets whipsawed in ranges.",
        "recommendedFor": ["momentum", "swing"],
        "parameters": [{"name": "shortWindow", "value": "20"}, {"name": "longWindow", "value": "60"}],
    },
    {
        "id": "mean-reversion",
        "name": "Mean reversion (Bollinger)",
        "description": "Buy when price closes below the lower Bollinger band, sell when it recovers to the middle band. With shorting on, also sells rallies above the upper band.",
        "recommendedFor": ["range-bound", "volatility"],
        "parameters": [{"name": "lookback", "value": "20"}, {"name": "deviation", "value": "2"}],
    },
    {
        "id": "trend-follow",
        "name": "Trend following breakout (Donchian)",
        "description": "Buy a close above the prior N-day high, exit on a close below the prior N/2-day low (mirrored for shorts).",
        "recommendedFor": ["breakout", "trend"],
        "parameters": [{"name": "channel", "value": "20"}],
    },
    {
        "id": "regime-switch",
        "name": "Regime switching (trend vs range)",
        "description": "Measures how trending the market is with Kaufman's efficiency ratio. In trends it follows the 20/50-day SMA crossover; in choppy ranges it trades Bollinger mean reversion.",
        "recommendedFor": ["adaptive", "regime"],
        "parameters": [{"name": "erWindow", "value": "20"}, {"name": "erThreshold", "value": "0.3"}],
    },
    {
        "id": "ml-logistic",
        "name": "Machine learning (logistic regression or boosted trees)",
        "description": "Predicts whether the price will be higher after the horizon from up to 12 price, volume, range and market features, retrained on past data only. Long while the predicted chance of a rise is at least the threshold.",
        "recommendedFor": ["machine learning", "research"],
        "parameters": [
            {"name": "threshold", "value": "0.52"},
            {"name": "trainWindow", "value": "504"},
            {"name": "horizon", "value": "1"},
            {"name": "modelType", "value": "0"},
        ],
    },
    {
        "id": "buy-hold",
        "name": "Buy & hold",
        "description": "Buy on day one and never sell. The baseline every other strategy has to beat.",
        "recommendedFor": ["baseline", "long-term"],
        "parameters": [],
    },
]
# "custom" strategies come from the Strategy Builder and carry their own rules (see backend/rules.py).
STRATEGY_IDS = {s["id"] for s in STRATEGIES} | {"custom"}
DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "sma-crossover": {"shortWindow": 20, "longWindow": 60},
    "mean-reversion": {"lookback": 20, "deviation": 2.0},
    "trend-follow": {"channel": 20},
    "regime-switch": {"erWindow": 20, "erThreshold": 0.3},
    "ml-logistic": {"threshold": 0.52, "trainWindow": 504, "horizon": 1, "modelType": 0},
    "buy-hold": {},
    "custom": {},
}
# Added later; older saved strategies may not have them, so they fall back to the defaults.
OPTIONAL_PARAMS = {"horizon", "modelType"}

EXECUTIONS = ("close", "next_open")
SIZINGS = ("full", "fixed", "vol-target")

Series = list[float | None]
Bars = dict[str, list[Any]]  # optional "open", "high", "low", "volume" lists aligned with the closes


def moving_average(values: list[float], window: int) -> Series:
    result: Series = []
    accumulator = 0.0
    for index, value in enumerate(values):
        accumulator += value
        if index >= window:
            accumulator -= values[index - window]
        result.append(accumulator / window if index + 1 >= window else None)
    return result


def rolling_std(values: list[float], window: int) -> Series:
    """Population standard deviation over a sliding window, in one pass."""
    result: Series = []
    total = total_sq = 0.0
    for index, value in enumerate(values):
        total += value
        total_sq += value * value
        if index >= window:
            old = values[index - window]
            total -= old
            total_sq -= old * old
        if index + 1 < window:
            result.append(None)
            continue
        mean = total / window
        var = total_sq / window - mean * mean
        # Rounding in the running sums can leave a tiny residue for a flat window.
        result.append(math.sqrt(var) if var > 1e-12 * max(mean * mean, 1.0) else 0.0)
    return result


def compute_drawdown(values: list[float]) -> float:
    """Largest peak-to-trough fall, as a positive fraction."""
    peak = values[0] if values else 0.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            worst = min(worst, (value - peak) / peak)
    return abs(worst)


def validate(strategy_id: str, params: dict[str, float], rules: dict[str, Any] | None = None) -> str | None:
    if strategy_id not in STRATEGY_IDS:
        return f"Unknown strategy '{strategy_id}'. Choose one of: {', '.join(sorted(STRATEGY_IDS))}"
    missing = set(DEFAULT_PARAMS[strategy_id]) - set(params) - OPTIONAL_PARAMS
    if missing:
        return f"Missing strategy parameters: {', '.join(sorted(missing))}"
    if strategy_id == "sma-crossover" and params["shortWindow"] >= params["longWindow"]:
        return "shortWindow must be less than longWindow"
    if strategy_id == "custom" and not (rules and rules.get("entry")):
        return "A custom strategy needs at least one entry rule"
    if strategy_id == "ml-logistic":
        if not 0.3 <= params["threshold"] <= 0.8:
            return "threshold must be between 0.3 and 0.8"
        if not 126 <= params["trainWindow"] <= 1000:
            return "trainWindow must be between 126 and 1000 days"
        if not 1 <= params.get("horizon", 1) <= 20:
            return "horizon must be between 1 and 20 days"
        if params.get("modelType", 0) not in (0, 1):
            return "modelType must be 0 (logistic regression) or 1 (boosted trees)"
    if strategy_id == "regime-switch":
        if not 5 <= params["erWindow"] <= 120:
            return "erWindow must be between 5 and 120 days"
        if not 0.05 <= params["erThreshold"] <= 0.95:
            return "erThreshold must be between 0.05 and 0.95"
    return None


WARMUP_PARAM = {"sma-crossover": "longWindow", "mean-reversion": "lookback", "trend-follow": "channel"}


def warmup(strategy_id: str, params: dict[str, float], rules: dict[str, Any] | None = None) -> int:
    """Days of history a strategy needs before it can produce a signal."""
    if strategy_id == "custom":
        from backend import rules as rule_engine

        return rule_engine.warmup(rules or {"entry": []})
    if strategy_id == "buy-hold":
        return 1
    if strategy_id == "ml-logistic":
        from backend import ml

        return ml.warmup()
    if strategy_id == "regime-switch":
        return max(50, int(params["erWindow"])) + 1
    return int(params[WARMUP_PARAM[strategy_id]])


def evaluation_start(timestamps: list[str], days: int) -> int:
    """Index of the first bar within the last `days` calendar days; earlier bars only warm up indicators."""
    try:
        last = date.fromisoformat(timestamps[-1][:10])
    except (IndexError, ValueError):
        return 0
    cutoff = (last - timedelta(days=days)).isoformat()
    return next((i for i, ts in enumerate(timestamps) if ts[:10] >= cutoff), 0)


def period_days(period: str) -> int:
    """Calendar days in a Yahoo-style period such as "2y", "6mo" or "90d"."""
    match = re.fullmatch(r"(\d+)(y|mo|d)", period.strip())
    if not match:
        return 730
    return int(match.group(1)) * {"y": 365, "mo": 30, "d": 1}[match.group(2)]


def efficiency_ratio(closes: list[float], window: int) -> Series:
    """Kaufman's efficiency ratio: net move over the window divided by the total distance travelled (0-1)."""
    n = len(closes)
    out: Series = [None] * n
    path = 0.0
    for i in range(1, n):
        path += abs(closes[i] - closes[i - 1])
        if i > window:
            path -= abs(closes[i - window] - closes[i - window - 1])
        if i >= window:
            out[i] = abs(closes[i] - closes[i - window]) / path if path > 1e-12 else 0.0
    return out


def _bollinger_states(closes: list[float], mid: Series, lower: Series, upper: Series, allow_short: bool) -> list[int]:
    held = 0
    states = [0] * len(closes)
    for i, close in enumerate(closes):
        if mid[i] is not None:
            if held == 0:
                if close < lower[i]:
                    held = 1
                elif allow_short and close > upper[i]:
                    held = -1
            elif held == 1 and close >= mid[i]:
                held = 0
            elif held == -1 and close <= mid[i]:
                held = 0
        states[i] = held
    return states


def _bands(closes: list[float], lookback: int, dev: float) -> tuple[Series, Series, Series]:
    mid = moving_average(closes, lookback)
    std = rolling_std(closes, lookback)
    lower: Series = [m - dev * s if m is not None and s is not None else None for m, s in zip(mid, std, strict=True)]
    upper: Series = [m + dev * s if m is not None and s is not None else None for m, s in zip(mid, std, strict=True)]
    return mid, lower, upper


def plan(
    strategy_id: str,
    closes: list[float],
    params: dict[str, float],
    rules: dict[str, Any] | None = None,
    *,
    allow_short: bool = False,
    bars: Bars | None = None,
    execution: str = "close",
    market_closes: Series | None = None,
) -> dict[str, Any]:
    """Target position per day (1/0/-1), indicator lines for charting, and intraday stop fills {bar: price}."""
    n = len(closes)
    target = [0] * n
    fills: dict[int, float] = {}
    if strategy_id == "buy-hold":
        # Flat on the first day so the entry (and its fee) is recorded like any other trade.
        return {"target": [0] + [1] * (n - 1), "indicators": {}, "fills": fills}

    if strategy_id == "custom":
        from backend import rules as rule_engine

        target, indicators, fills = rule_engine.plan(closes, rules or {"entry": []}, bars=bars, execution=execution)
        return {"target": target, "indicators": indicators, "fills": fills}

    if strategy_id == "ml-logistic":
        from backend import ml

        target, indicators = ml.signals(closes, params, allow_short=allow_short, bars=bars, market=market_closes)
        return {"target": target, "indicators": indicators, "fills": fills}

    if strategy_id == "sma-crossover":
        short = moving_average(closes, int(params["shortWindow"]))
        long_ = moving_average(closes, int(params["longWindow"]))
        for i in range(n):
            if short[i] is not None and long_[i] is not None:
                target[i] = 1 if short[i] > long_[i] else (-1 if allow_short and short[i] < long_[i] else 0)
        return {"target": target, "indicators": {"shortSma": short, "longSma": long_}, "fills": fills}

    if strategy_id == "mean-reversion":
        mid, lower, upper = _bands(closes, int(params["lookback"]), float(params["deviation"]))
        target = _bollinger_states(closes, mid, lower, upper, allow_short)
        return {
            "target": target,
            "indicators": {"middleBand": mid, "lowerBand": lower, "upperBand": upper},
            "fills": fills,
        }

    if strategy_id == "trend-follow":
        channel = int(params["channel"])
        exit_len = max(channel // 2, 2)
        upper_line: Series = [None] * n
        lower_line: Series = [None] * n
        held = 0
        for i in range(n):
            if i >= channel:
                high, low = max(closes[i - channel : i]), min(closes[i - channel : i])
                exit_low, exit_high = min(closes[i - exit_len : i]), max(closes[i - exit_len : i])
                upper_line[i], lower_line[i] = high, exit_low
                if held == 0:
                    if closes[i] > high:
                        held = 1
                    elif allow_short and closes[i] < low:
                        held = -1
                elif held == 1 and closes[i] < exit_low:
                    held = 0
                elif held == -1 and closes[i] > exit_high:
                    held = 0
            target[i] = held
        return {"target": target, "indicators": {"channelHigh": upper_line, "channelLow": lower_line}, "fills": fills}

    if strategy_id == "regime-switch":
        er = efficiency_ratio(closes, int(params["erWindow"]))
        threshold = float(params["erThreshold"])
        short, long_ = moving_average(closes, 20), moving_average(closes, 50)
        mid, lower, upper = _bands(closes, 20, 2.0)
        ranging = _bollinger_states(closes, mid, lower, upper, allow_short)
        for i in range(n):
            if er[i] is None or short[i] is None or long_[i] is None:
                continue
            if er[i] >= threshold:
                target[i] = 1 if short[i] > long_[i] else (-1 if allow_short and short[i] < long_[i] else 0)
            else:
                target[i] = ranging[i]
        return {
            "target": target,
            "indicators": {"shortSma": short, "longSma": long_, "middleBand": mid},
            "fills": fills,
            "regime": er,
        }

    raise ValueError(f"Unknown strategy {strategy_id}")


def signals(
    strategy_id: str,
    closes: list[float],
    params: dict[str, float],
    rules: dict[str, Any] | None = None,
    *,
    allow_short: bool = False,
    bars: Bars | None = None,
) -> tuple[list[int], dict[str, Series]]:
    """Target position for each day, plus indicator lines for charting."""
    planned = plan(strategy_id, closes, params, rules, allow_short=allow_short, bars=bars)
    return planned["target"], planned["indicators"]


def size_positions(
    target: list[int],
    closes: list[float],
    mode: str = "full",
    *,
    fraction: float = 1.0,
    target_vol: float = 0.15,
    vol_window: int = 20,
    max_leverage: float = 1.0,
    band: float = 0.1,
) -> list[float]:
    """Turn a 1/0/-1 target into position weights.

    full: 100% of equity. fixed: `fraction` of equity. vol-target: size so the position's expected volatility
    is `target_vol` a year, from the last `vol_window` days of returns (known at the close), capped at
    `max_leverage`. Changes smaller than `band` are skipped, so the position isn't rebalanced every day.
    """
    if mode == "full":
        return [float(t) for t in target]
    if mode == "fixed":
        return [t * fraction for t in target]
    returns = [0.0] + [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    vols = rolling_std(returns, vol_window)
    out: list[float] = []
    previous = 0.0
    for t, side in enumerate(target):
        if not side:
            previous = 0.0
            out.append(0.0)
            continue
        vol = vols[t] * math.sqrt(252) if t >= vol_window and vols[t] else None
        weight = max_leverage if not vol else min(max_leverage, target_vol / vol)
        new = round(side * weight, 4)
        if previous and (previous > 0) == (new > 0) and abs(new - previous) < band:
            new = previous
        out.append(new)
        previous = new
    return out


def simulate(
    closes: list[float],
    target: list[float],
    start: int,
    end: int,
    cost_bps: float,
    *,
    held: float = 0.0,
    value: float = 1.0,
    opens: list[float] | None = None,
    execution: str = "close",
    fills: dict[int, float] | None = None,
    sell_cost_bps: float | None = None,
    borrow_bps: float = 0.0,
) -> dict[str, Any]:
    """Trade `target` weights over bars start..end (inclusive), starting with `value` and position `held`.

    With execution="close", the position decided at a bar's close is taken at that close. With "next_open"
    (and `opens` given) it is taken at the next bar's open. `fills` are intraday stop exits {bar: price}.
    Buys pay `cost_bps` and sells `sell_cost_bps` (default: the same) on the traded value; shorts pay
    `borrow_bps` a year while held.
    """
    buy_rate = cost_bps / 10_000
    sell_rate = (cost_bps if sell_cost_bps is None else sell_cost_bps) / 10_000
    borrow = borrow_bps / 10_000 / 252
    use_open = execution == "next_open" and opens is not None
    held = float(held)
    equity: list[float] = []
    daily_returns: list[float] = []
    trades: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    entry: dict[str, Any] | None = (
        {"entryIndex": start, "entryValue": value, "entryPrice": closes[start], "side": 1 if held > 0 else -1}
        if held
        else None
    )
    days_in_market = 0
    turnover = 0.0

    def trade(t: int, new: float, price: float) -> None:
        nonlocal value, held, entry, turnover
        new = float(new)
        if new == held:
            return
        before, was = value, held
        if held and (new == 0 or (new > 0) != (held > 0)):  # close the open trade (or the first leg of a flip)
            value *= 1 - abs(held) * (sell_rate if held > 0 else buy_rate)
            if entry is not None:
                trades.append({**entry, "exitIndex": t, "exitValue": value, "exitPrice": price})
            entry, held = None, 0.0
        if new != held:
            if held == 0:
                # Equity before the entry cost, so trade returns include both sides.
                entry = {"entryIndex": t, "entryValue": value, "entryPrice": price, "side": 1 if new > 0 else -1}
                value *= 1 - abs(new) * (buy_rate if new > 0 else sell_rate)
            else:  # resize in the same direction
                growing = abs(new) > abs(held)
                rate = (buy_rate if new > 0 else sell_rate) if growing else (sell_rate if new > 0 else buy_rate)
                value *= 1 - abs(new - held) * rate
            held = new
        turnover += abs(new - was)
        log.append({"index": t, "from": was, "to": new, "price": price, "valueBefore": before, "cost": before - value})

    for t in range(start, end + 1):
        before = value
        if t > start:
            if use_open:
                if held:
                    value *= 1 + held * (opens[t] / closes[t - 1] - 1)
                trade(t, target[t - 1], opens[t])
                reference = opens[t]
            else:
                reference = closes[t - 1]
            if held:
                days_in_market += 1
                short = held < 0
                stop = fills.get(t) if fills else None
                if stop is not None:
                    value *= 1 + held * (stop / reference - 1)
                    trade(t, 0.0, stop)
                else:
                    value *= 1 + held * (closes[t] / reference - 1)
                if short and borrow:
                    value *= 1 - borrow
            value = max(value, 0.0)
        if not use_open:
            trade(t, target[t], closes[t])
        if t > start:
            daily_returns.append(value / before - 1 if before else 0.0)
        equity.append(value)
    return {
        "equity": equity,
        "dailyReturns": daily_returns,
        "trades": trades,
        "entry": entry,
        "held": held,
        "daysInMarket": days_in_market,
        "turnover": turnover,
        "fills": log,
    }


def backtest(
    closes: list[float],
    timestamps: list[str],
    strategy_id: str,
    params: dict[str, float],
    fee_bps: float = 10.0,
    rules: dict[str, Any] | None = None,
    *,
    slippage_bps: float = 0.0,
    start: int = 0,
    benchmark: tuple[list[str], list[float]] | None = None,
    benchmark_info: tuple[str, str] | None = None,
    bars: Bars | None = None,
    execution: str = "close",
    sizing: dict[str, Any] | None = None,
    allow_short: bool = False,
    borrow_bps: float = 0.0,
    sell_fee_bps: float | None = None,
    risk_free: float = 0.0,
    market_closes: Series | None = None,
) -> dict[str, Any]:
    """Backtest bars start..end. Bars before `start` only warm up indicators; the money starts in cash."""
    from backend import risk

    planned = plan(
        strategy_id,
        closes,
        params,
        rules,
        allow_short=allow_short,
        bars=bars,
        execution=execution,
        market_closes=market_closes,
    )
    sizing = sizing or {}
    sized = size_positions(
        planned["target"],
        closes,
        sizing.get("mode", "full"),
        fraction=float(sizing.get("fraction", 1.0)),
        target_vol=float(sizing.get("targetVol", 0.15)),
        max_leverage=float(sizing.get("maxLeverage", 1.0)),
    )
    opens = (bars or {}).get("open")
    if opens is not None and any(o is None for o in opens):
        opens = None
    last = len(closes) - 1
    buy_cost = fee_bps + slippage_bps
    sell_cost = (fee_bps if sell_fee_bps is None else sell_fee_bps) + slippage_bps
    run = simulate(
        closes,
        sized,
        start,
        last,
        buy_cost,
        sell_cost_bps=sell_cost,
        opens=opens,
        execution=execution,
        fills=planned["fills"],
        borrow_bps=borrow_bps,
    )
    equity = run["equity"]
    trades = [_trade(tr, timestamps) for tr in run["trades"]]
    open_trade = (
        _trade({**run["entry"], "exitIndex": last, "exitValue": equity[-1], "exitPrice": closes[last]}, timestamps)
        if run["entry"]
        else None
    )

    # Measured from the starting capital (1.0), so a cost paid on the first bar counts.
    stats = risk.summary(run["dailyReturns"], [1.0, *equity], risk_free)
    stats["annualizedReturn"] = risk.annualise(stats["totalReturn"], max(last - start, 1))
    buy_hold_equity = [closes[t] / closes[start] for t in range(start, last + 1)]
    buy_hold = risk.summary(
        [buy_hold_equity[k] / buy_hold_equity[k - 1] - 1 for k in range(1, len(buy_hold_equity))],
        buy_hold_equity,
        risk_free,
    )
    trade_returns = [tr["return"] for tr in trades]
    wins = [r for r in trade_returns if r > 0]
    days = max(last - start, 1)
    window = sized[start : last + 1]
    metrics = {
        **stats,
        **risk.tail_risk(run["dailyReturns"]),
        "buyHoldReturn": buy_hold["totalReturn"],
        "excessReturn": stats["totalReturn"] - buy_hold["totalReturn"],
        "winRate": len(wins) / len(trades) if trades else 0.0,
        "trades": len(trades) + (1 if open_trade else 0),
        "closedTrades": len(trades),
        "longTrades": sum(1 for tr in trades if tr["side"] == "long"),
        "shortTrades": sum(1 for tr in trades if tr["side"] == "short"),
        "avgTradeReturn": sum(trade_returns) / len(trades) if trades else 0.0,
        "profitFactor": risk.profit_factor(trade_returns),
        "exposure": run["daysInMarket"] / days,
        "avgGrossExposure": sum(abs(w) for w in window) / len(window) if window else 0.0,
        # Traded value per year as a multiple of equity (1.0 = the whole portfolio turned over once a year).
        "turnover": run["turnover"] / days * risk.TRADING_DAYS,
        "feeBps": fee_bps,
        "sellFeeBps": fee_bps if sell_fee_bps is None else sell_fee_bps,
        "slippageBps": slippage_bps,
        "borrowBps": borrow_bps,
        "riskFreeRate": risk_free,
        "execution": execution if execution in EXECUTIONS else "close",
        "sizing": sizing.get("mode", "full"),
        "allowShort": allow_short,
    }

    bench_symbol, bench_name = benchmark_info or (risk.BENCHMARK_SYMBOL, risk.BENCHMARK_NAME)
    index = (
        risk.benchmark(
            timestamps[start:], equity, *benchmark, symbol=bench_symbol, name=bench_name, risk_free=risk_free
        )
        if benchmark
        else None
    )
    drawdown = risk.drawdowns(equity)
    buy_hold_drawdown = risk.drawdowns(buy_hold_equity)
    rolling_sharpe = risk.rolling_sharpe(run["dailyReturns"], 126, risk_free)
    rolling_beta: list[float | None] = [None] * len(equity)
    if index:
        rolling_beta = risk.rolling_beta(run["dailyReturns"], risk.aligned_returns(timestamps[start:], *benchmark))
    regime = planned.get("regime")
    sample = []
    for k, i in enumerate(range(start, last + 1)):
        point: dict[str, Any] = {
            "timestamp": timestamps[i],
            "close": closes[i],
            "equity": equity[k],
            "buyHold": buy_hold_equity[k],
            "drawdown": drawdown[k],
            "buyHoldDrawdown": buy_hold_drawdown[k],
            "position": sized[i],
            "rollingSharpe": rolling_sharpe[k],
            "rollingBeta": rolling_beta[k],
        }
        if regime is not None:
            point["efficiencyRatio"] = regime[i]
        for name, line in planned["indicators"].items():
            point[name] = line[i] if line[i] is not None else (None if name == "probUp" else closes[i])
        sample.append(point)
    return {
        "metrics": metrics,
        "buyHold": buy_hold,
        "benchmark": index,
        "trades": trades[-50:],
        "openTrade": open_trade,
        "sample": sample,
        "monthly": risk.monthly_returns(timestamps[start:], equity, 1.0),
        "dailyReturns": run["dailyReturns"],
        "period": {"start": timestamps[start], "end": timestamps[last], "days": last - start + 1},
    }


def _trade(record: dict[str, Any], timestamps: list[str]) -> dict[str, Any]:
    entry_value = record["entryValue"]
    return {
        "entryDate": timestamps[record["entryIndex"]],
        "entryPrice": record["entryPrice"],
        "exitDate": timestamps[record["exitIndex"]],
        "exitPrice": record["exitPrice"],
        "side": "long" if record["side"] > 0 else "short",
        "bars": record["exitIndex"] - record["entryIndex"],
        "return": record["exitValue"] / entry_value - 1 if entry_value else 0.0,
    }


SIGNALS = {
    (1, 0): "buy",
    (1, 1): "hold",
    (0, 1): "sell",
    (0, 0): "wait",
    (-1, 0): "short",
    (-1, -1): "hold",
    (0, -1): "cover",
    (1, -1): "buy",
    (-1, 1): "short",
}
EXPLANATIONS = {
    "buy": "The entry rule triggered at today's close: buy (at the next open when trading on next-day opens).",
    "hold": "The strategy already has a position and its exit rule has not triggered.",
    "sell": "The exit rule triggered at today's close: sell the long position.",
    "wait": "The strategy is flat and waiting for its entry rule.",
    "short": "The short-entry rule triggered at today's close: sell short.",
    "cover": "The short position's exit rule triggered at today's close: buy it back.",
}


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def current_signal(
    strategy_id: str,
    closes: list[float],
    params: dict[str, float],
    rules: dict[str, Any] | None = None,
    *,
    allow_short: bool = False,
    bars: Bars | None = None,
    market_closes: Series | None = None,
) -> dict[str, Any]:
    """Today's action for a trained strategy: buy / hold / sell / wait (and short / cover with shorting)."""
    planned = plan(strategy_id, closes, params, rules, allow_short=allow_short, bars=bars, market_closes=market_closes)
    target, indicators = planned["target"], planned["indicators"]
    today, yesterday = _sign(target[-1]), _sign(target[-2]) if len(target) > 1 else 0
    signal = SIGNALS[(today, yesterday)]
    price = closes[-1]
    if strategy_id == "buy-hold":
        strength = 1.0
        detail = f"Buy & hold stays invested; the last close was {price:.2f}."
    elif strategy_id == "ml-logistic":
        probability = indicators["probUp"][-1]
        threshold = float(params["threshold"])
        if probability is None:
            strength, detail = 0.0, "Not enough history to train the model yet."
        else:
            strength = abs(probability - threshold) * 10
            horizon = int(params.get("horizon", 1))
            when = "tomorrow" if horizon == 1 else f"in {horizon} trading days"
            detail = (
                f"The model puts the chance of a higher close {when} at {probability * 100:.1f}% "
                f"(it buys at {threshold * 100:.0f}% or more)."
            )
    elif strategy_id == "custom":
        from backend import rules as rule_engine

        strength = 1.0 if today else 0.5
        detail = f"Rules: {rule_engine.describe(rules or {'entry': []})}"
    elif strategy_id == "sma-crossover":
        short, long_ = indicators["shortSma"][-1], indicators["longSma"][-1]
        spread = (short - long_) / long_ if long_ else 0.0
        strength = abs(spread) * 20
        detail = f"Short/long SMA spread is {spread * 100:+.2f}% (short {short:.2f} vs long {long_:.2f})."
    elif strategy_id == "mean-reversion":
        mid, low = indicators["middleBand"][-1], indicators["lowerBand"][-1]
        band = (mid - low) or 1.0
        strength = abs(price - mid) / band
        detail = f"Price {price:.2f} vs middle band {mid:.2f} and lower band {low:.2f}."
    elif strategy_id == "regime-switch":
        er = planned["regime"][-1] or 0.0
        trending = er >= float(params["erThreshold"])
        strength = abs(er - float(params["erThreshold"])) * 3
        detail = (
            f"Efficiency ratio {er:.2f} vs threshold {float(params['erThreshold']):.2f}: the market is "
            f"{'trending, so the SMA 20/50 crossover is in charge' if trending else 'ranging, so Bollinger mean reversion is in charge'}."
        )
    else:
        high, low = indicators["channelHigh"][-1], indicators["channelLow"][-1]
        width = (high - low) or 1.0
        strength = (price - low) / width if today else (high - price) / width
        detail = f"Price {price:.2f} vs breakout level {high:.2f} and exit level {low:.2f}."
    return {
        "signal": signal,
        "position": target[-1],
        "confidence": max(0.0, min(strength, 1.0)),
        "summary": f"{EXPLANATIONS[signal]} {detail}",
    }
