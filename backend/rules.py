"""Rule engine for user-built strategies (the Strategy Builder).

A strategy is a set of plain rules evaluated on daily closes:

    enter when ALL entry conditions are true (while flat)
    exit  when ANY exit condition is true, or the stop-loss / take-profit is hit (while long)

Each condition compares two operands: the price, an indicator (SMA, EMA, RSI) or a fixed value, with
">", "<", "crosses above" or "crosses below". Like the built-in strategies, a decision on day t only
uses data up to day t, and the backtester holds that position from the next day.
"""

from __future__ import annotations

from typing import Any

from backend.strategies import Series, moving_average

INDICATORS = {"sma", "ema", "rsi"}


def ema(values: list[float], period: int) -> Series:
    alpha = 2 / (period + 1)
    result: Series = [None] * len(values)
    if len(values) < period:
        return result
    current = sum(values[:period]) / period  # seed with the simple average
    result[period - 1] = current
    for i in range(period, len(values)):
        current = alpha * values[i] + (1 - alpha) * current
        result[i] = current
    return result


def rsi(values: list[float], period: int) -> Series:
    """Wilder's RSI (0-100)."""
    result: Series = [None] * len(values)
    if len(values) <= period:
        return result
    gains = [max(b - a, 0.0) for a, b in zip(values[:-1], values[1:], strict=True)]
    losses = [max(a - b, 0.0) for a, b in zip(values[:-1], values[1:], strict=True)]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(values)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        result[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return result


def operand_series(closes: list[float], operand: dict[str, Any], cache: dict[tuple, Series]) -> Series:
    kind = operand["kind"]
    if kind == "price":
        return list(closes)
    if kind == "value":
        return [float(operand["value"])] * len(closes)
    key = (kind, int(operand["period"]))
    if key not in cache:
        builder = {"sma": moving_average, "ema": ema, "rsi": rsi}[kind]
        cache[key] = builder(closes, key[1])
    return cache[key]


def label(operand: dict[str, Any]) -> str:
    kind = operand["kind"]
    if kind == "price":
        return "Price"
    if kind == "value":
        value = float(operand["value"])
        return f"{value:g}"
    return f"{kind.upper()}({int(operand['period'])})"


OP_WORDS = {">": "is above", "<": "is below", "crosses_above": "crosses above", "crosses_below": "crosses below"}


def describe_condition(condition: dict[str, Any]) -> str:
    return f"{label(condition['left'])} {OP_WORDS[condition['op']]} {label(condition['right'])}"


def describe(rules: dict[str, Any]) -> str:
    parts = ["Buy when " + " and ".join(describe_condition(c) for c in rules["entry"])]
    exits = [describe_condition(c) for c in rules.get("exit") or []]
    if rules.get("stopLoss"):
        exits.append(f"price falls {rules['stopLoss'] * 100:g}% below entry")
    if rules.get("takeProfit"):
        exits.append(f"price rises {rules['takeProfit'] * 100:g}% above entry")
    parts.append("sell when " + " or ".join(exits) if exits else "hold once bought")
    return "; ".join(parts) + "."


def warmup(rules: dict[str, Any]) -> int:
    periods = [
        int(operand["period"])
        for condition in [*rules["entry"], *(rules.get("exit") or [])]
        for operand in (condition["left"], condition["right"])
        if operand["kind"] in INDICATORS
    ]
    return max(periods, default=1) + 1


def _holds(condition: dict[str, Any], i: int, left: Series, right: Series) -> bool:
    a, b = left[i], right[i]
    if a is None or b is None:
        return False
    op = condition["op"]
    if op == ">":
        return a > b
    if op == "<":
        return a < b
    if i == 0 or left[i - 1] is None or right[i - 1] is None:
        return False
    if op == "crosses_above":
        return a > b and left[i - 1] <= right[i - 1]
    return a < b and left[i - 1] >= right[i - 1]  # crosses_below


def signals(closes: list[float], rules: dict[str, Any]) -> tuple[list[int], dict[str, Series]]:
    cache: dict[tuple, Series] = {}

    def prepared(conditions: list[dict[str, Any]]) -> list[tuple[dict[str, Any], Series, Series]]:
        return [
            (c, operand_series(closes, c["left"], cache), operand_series(closes, c["right"], cache)) for c in conditions
        ]

    entry = prepared(rules["entry"])
    exits = prepared(rules.get("exit") or [])
    stop, take = rules.get("stopLoss"), rules.get("takeProfit")
    target = [0] * len(closes)
    held, entry_price = 0, 0.0
    for i in range(len(closes)):
        if not held:
            if all(_holds(c, i, left, right) for c, left, right in entry):
                held, entry_price = 1, closes[i]
        else:
            hit_stop = stop is not None and closes[i] <= entry_price * (1 - stop)
            hit_take = take is not None and closes[i] >= entry_price * (1 + take)
            if hit_stop or hit_take or any(_holds(c, i, left, right) for c, left, right in exits):
                held = 0
        target[i] = held
    indicators = {f"{kind.upper()}{period}": line for (kind, period), line in cache.items() if kind != "rsi"}
    return target, indicators
