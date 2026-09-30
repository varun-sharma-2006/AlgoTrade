"""Strategy catalogue and an honest long-only backtester.

Each strategy turns daily closes into a target position (1 = long, 0 = flat) using only data up to
that day. The backtester holds that position over the next day, charges a fee on every entry and
exit, and records each round-trip trade, so win rate, Sharpe and drawdown come from simulated trades
rather than from the underlying stock.
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
        "description": "Long while the short SMA is above the long SMA; flat otherwise. Rides big trends, gets whipsawed in ranges.",
        "recommendedFor": ["momentum", "swing"],
        "parameters": [{"name": "shortWindow", "value": "20"}, {"name": "longWindow", "value": "60"}],
    },
    {
        "id": "mean-reversion",
        "name": "Mean reversion (Bollinger)",
        "description": "Buy when price closes below the lower Bollinger band, sell when it recovers to the middle band.",
        "recommendedFor": ["range-bound", "volatility"],
        "parameters": [{"name": "lookback", "value": "20"}, {"name": "deviation", "value": "2"}],
    },
    {
        "id": "trend-follow",
        "name": "Trend following breakout (Donchian)",
        "description": "Buy a close above the prior N-day high, exit on a close below the prior N/2-day low.",
        "recommendedFor": ["breakout", "trend"],
        "parameters": [{"name": "channel", "value": "20"}],
    },
    {
        "id": "ml-logistic",
        "name": "Machine learning (logistic regression)",
        "description": "Predicts tomorrow's direction from 8 price features, retrained monthly on past data only. Long while the predicted chance of a rise is at least the threshold.",
        "recommendedFor": ["machine learning", "research"],
        "parameters": [{"name": "threshold", "value": "0.52"}, {"name": "trainWindow", "value": "504"}],
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
    "ml-logistic": {"threshold": 0.52, "trainWindow": 504},
    "buy-hold": {},
    "custom": {},
}

Series = list[float | None]


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
    result: Series = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        chunk = values[index + 1 - window : index + 1]
        mean = sum(chunk) / window
        result.append(math.sqrt(sum((v - mean) ** 2 for v in chunk) / window))
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
    missing = set(DEFAULT_PARAMS[strategy_id]) - set(params)
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


def signals(
    strategy_id: str, closes: list[float], params: dict[str, float], rules: dict[str, Any] | None = None
) -> tuple[list[int], dict[str, Series]]:
    """Target position for each day, plus indicator lines for charting."""
    n = len(closes)
    target = [0] * n
    if strategy_id == "buy-hold":
        # Flat on the first day so the entry (and its fee) is recorded like any other trade.
        return [0] + [1] * (n - 1), {}

    if strategy_id == "custom":
        from backend import rules as rule_engine

        return rule_engine.signals(closes, rules or {"entry": []})

    if strategy_id == "ml-logistic":
        from backend import ml

        return ml.signals(closes, params)

    if strategy_id == "sma-crossover":
        short = moving_average(closes, int(params["shortWindow"]))
        long_ = moving_average(closes, int(params["longWindow"]))
        for i in range(n):
            target[i] = int(short[i] is not None and long_[i] is not None and short[i] > long_[i])
        return target, {"shortSma": short, "longSma": long_}

    if strategy_id == "mean-reversion":
        lookback, dev = int(params["lookback"]), float(params["deviation"])
        mid = moving_average(closes, lookback)
        std = rolling_std(closes, lookback)
        lower: Series = [
            m - dev * s if m is not None and s is not None else None for m, s in zip(mid, std, strict=True)
        ]
        upper: Series = [
            m + dev * s if m is not None and s is not None else None for m, s in zip(mid, std, strict=True)
        ]
        held = 0
        for i in range(n):
            if mid[i] is not None:
                if not held and closes[i] < lower[i]:
                    held = 1
                elif held and closes[i] >= mid[i]:
                    held = 0
            target[i] = held
        return target, {"middleBand": mid, "lowerBand": lower, "upperBand": upper}

    if strategy_id == "trend-follow":
        channel = int(params["channel"])
        exit_len = max(channel // 2, 2)
        upper_line: Series = [None] * n
        lower_line: Series = [None] * n
        held = 0
        for i in range(n):
            if i >= channel:
                upper_line[i] = max(closes[i - channel : i])
                lower_line[i] = min(closes[i - exit_len : i])
                if not held and closes[i] > upper_line[i]:
                    held = 1
                elif held and closes[i] < lower_line[i]:
                    held = 0
            target[i] = held
        return target, {"channelHigh": upper_line, "channelLow": lower_line}

    raise ValueError(f"Unknown strategy {strategy_id}")


def simulate(
    closes: list[float],
    target: list[int],
    start: int,
    end: int,
    cost_bps: float,
    *,
    held: int = 0,
    value: float = 1.0,
) -> dict[str, Any]:
    """Trade `target` over bars start..end (inclusive), starting with `value` and position `held`.

    The position decided at a bar's close is held through the next bar, and every switch pays `cost_bps`
    (fee plus slippage) on the traded value.
    """
    cost = cost_bps / 10_000
    equity: list[float] = []
    daily_returns: list[float] = []
    trades: list[tuple[int, float, int, float]] = []  # (entry index, value before entry, exit index, exit value)
    entry: tuple[int, float] | None = (start, value) if held else None
    days_in_market = 0
    for t in range(start, end + 1):
        before = value
        if t > start and held:
            value *= closes[t] / closes[t - 1]
            days_in_market += 1
        if target[t] != held:
            if target[t]:
                entry = (t, value)  # equity before the entry cost, so trade returns include both sides
            value *= 1 - cost
            if not target[t] and entry is not None:
                trades.append((entry[0], entry[1], t, value))
                entry = None
            held = target[t]
        if t > start:
            daily_returns.append(value / before - 1)
        equity.append(value)
    return {
        "equity": equity,
        "dailyReturns": daily_returns,
        "trades": trades,
        "entry": entry,
        "held": held,
        "daysInMarket": days_in_market,
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
) -> dict[str, Any]:
    """Backtest bars start..end. Bars before `start` only warm up indicators; the money starts in cash."""
    from backend import risk

    target, indicators = signals(strategy_id, closes, params, rules)
    last = len(closes) - 1
    run = simulate(closes, target, start, last, fee_bps + slippage_bps)
    equity = run["equity"]
    trades = [_trade(i, v, j, w, closes, timestamps) for i, v, j, w in run["trades"]]
    open_trade = _trade(*run["entry"], last, equity[-1], closes, timestamps) if run["entry"] else None

    # Measured from the starting capital (1.0), so a cost paid on the first bar counts.
    stats = risk.summary(run["dailyReturns"], [1.0, *equity])
    stats["annualizedReturn"] = risk.annualise(stats["totalReturn"], max(last - start, 1))
    buy_hold_equity = [closes[t] / closes[start] for t in range(start, last + 1)]
    buy_hold = risk.summary(
        [buy_hold_equity[k] / buy_hold_equity[k - 1] - 1 for k in range(1, len(buy_hold_equity))], buy_hold_equity
    )
    wins = [tr for tr in trades if tr["return"] > 0]
    metrics = {
        **stats,
        "buyHoldReturn": buy_hold["totalReturn"],
        "excessReturn": stats["totalReturn"] - buy_hold["totalReturn"],
        "winRate": len(wins) / len(trades) if trades else 0.0,
        "trades": len(trades) + (1 if open_trade else 0),
        "closedTrades": len(trades),
        "avgTradeReturn": sum(tr["return"] for tr in trades) / len(trades) if trades else 0.0,
        "exposure": run["daysInMarket"] / max(last - start, 1),
        "feeBps": fee_bps,
        "slippageBps": slippage_bps,
    }

    index = risk.benchmark(timestamps[start:], equity, *benchmark) if benchmark else None
    drawdown = risk.drawdowns(equity)
    buy_hold_drawdown = risk.drawdowns(buy_hold_equity)
    sample = []
    for k, i in enumerate(range(start, last + 1)):
        point: dict[str, Any] = {
            "timestamp": timestamps[i],
            "close": closes[i],
            "equity": equity[k],
            "buyHold": buy_hold_equity[k],
            "drawdown": drawdown[k],
            "buyHoldDrawdown": buy_hold_drawdown[k],
            "position": target[i],
        }
        for name, line in indicators.items():
            point[name] = line[i] if line[i] is not None else (None if name == "probUp" else closes[i])
        sample.append(point)
    return {
        "metrics": metrics,
        "buyHold": buy_hold,
        "benchmark": index,
        "trades": trades[-10:],
        "openTrade": open_trade,
        "sample": sample,
        "period": {"start": timestamps[start], "end": timestamps[last], "days": last - start + 1},
    }


def _trade(
    entry_index: int, entry_value: float, exit_index: int, exit_value: float, closes: list[float], timestamps: list[str]
) -> dict[str, Any]:
    return {
        "entryDate": timestamps[entry_index],
        "entryPrice": closes[entry_index],
        "exitDate": timestamps[exit_index],
        "exitPrice": closes[exit_index],
        "return": exit_value / entry_value - 1 if entry_value else 0.0,
    }


def current_signal(
    strategy_id: str, closes: list[float], params: dict[str, float], rules: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Today's action for a trained strategy: buy / hold / sell / wait."""
    target, indicators = signals(strategy_id, closes, params, rules)
    today, yesterday = target[-1], target[-2] if len(target) > 1 else 0
    signal = {(1, 0): "buy", (1, 1): "hold", (0, 1): "sell", (0, 0): "wait"}[(today, yesterday)]
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
            detail = (
                f"The model puts the chance of a higher close tomorrow at {probability * 100:.1f}% "
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
    else:
        high, low = indicators["channelHigh"][-1], indicators["channelLow"][-1]
        width = (high - low) or 1.0
        strength = (price - low) / width if today else (high - price) / width
        detail = f"Price {price:.2f} vs breakout level {high:.2f} and exit level {low:.2f}."
    explanations = {
        "buy": "The strategy entered a long position at today's close.",
        "hold": "The strategy is already long and its exit rule has not triggered.",
        "sell": "The exit rule triggered today; the strategy closed its position.",
        "wait": "The strategy is flat and waiting for its entry rule.",
    }
    return {"signal": signal, "confidence": max(0.0, min(strength, 1.0)), "summary": f"{explanations[signal]} {detail}"}
