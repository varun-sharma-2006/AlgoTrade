"""Rule engine for user-built strategies (the Strategy Builder).

A strategy is a set of plain rules evaluated on daily data:

    enter when ALL (or ANY) entry conditions are true (while flat)
    exit  when ANY exit condition is true, the stop-loss / take-profit / trailing stop is hit,
          or the position has been held for `maxHoldDays` (while in a position)

Each condition compares two operands: the price, an indicator (SMA, EMA, RSI, MACD, ATR, volume, the highest
or lowest close of the previous N days, rate of change) or a fixed value, with ">", "<", "crosses above" or
"crosses below". `side` makes the rules open short positions instead of long ones.

Like the built-in strategies, a decision on day t only uses data up to day t. When daily highs and lows are
available, stops are checked during the day: a position is closed at the stop level the first day the low
(for a long) touches it, or at the open if the price gaps through it. Without them, stops are checked on
closing prices.
"""

from __future__ import annotations

from typing import Any

from backend.strategies import Bars, Series, moving_average

# Operands that take a period.
INDICATORS = {"sma", "ema", "rsi", "atr", "volume_sma", "highest", "lowest", "roc"}
# Operands with fixed settings (MACD uses the standard 12/26/9).
FIXED = {"macd", "macd_signal", "macd_hist", "volume"}
MACD_WARMUP = 26 + 9


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


def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Series, Series, Series]:
    """MACD line, its signal line and the histogram."""
    fast_line, slow_line = ema(values, fast), ema(values, slow)
    line: Series = [
        f - s if f is not None and s is not None else None for f, s in zip(fast_line, slow_line, strict=True)
    ]
    first = next((i for i, v in enumerate(line) if v is not None), len(values))
    tail = ema([v for v in line[first:] if v is not None], signal)
    signal_line: Series = [None] * first + tail
    hist: Series = [a - b if a is not None and b is not None else None for a, b in zip(line, signal_line, strict=True)]
    return line, signal_line, hist


def atr(closes: list[float], period: int, highs: list[float] | None = None, lows: list[float] | None = None) -> Series:
    """Wilder's average true range. Without highs and lows, the true range is the close-to-close move."""
    n = len(closes)
    result: Series = [None] * n
    ranges = []
    for i in range(1, n):
        if highs is not None and lows is not None:
            ranges.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
        else:
            ranges.append(abs(closes[i] - closes[i - 1]))
    if len(ranges) < period:
        return result
    current = sum(ranges[:period]) / period
    result[period] = current
    for i in range(period + 1, n):
        current = (current * (period - 1) + ranges[i - 1]) / period
        result[i] = current
    return result


def _extreme(closes: list[float], period: int, pick: Any) -> Series:
    """Highest or lowest close of the previous `period` days (today excluded, so "price crosses above" works)."""
    return [pick(closes[i - period : i]) if i >= period else None for i in range(len(closes))]


def operand_series(
    closes: list[float], operand: dict[str, Any], cache: dict[tuple, Series], bars: Bars | None = None
) -> Series:
    kind = operand["kind"]
    n = len(closes)
    if kind == "price":
        return list(closes)
    if kind == "value":
        return [float(operand["value"])] * n
    period = int(operand.get("period") or 0)
    key = (kind, period)
    if key in cache:
        return cache[key]
    bars = bars or {}
    volume = bars.get("volume")
    has_volume = volume is not None and all(v is not None for v in volume)
    if kind in {"macd", "macd_signal", "macd_hist"}:
        line, signal_line, hist = macd(closes)
        cache[("macd", 0)], cache[("macd_signal", 0)], cache[("macd_hist", 0)] = line, signal_line, hist
        return cache[key]
    if kind == "sma":
        series = moving_average(closes, period)
    elif kind == "ema":
        series = ema(closes, period)
    elif kind == "rsi":
        series = rsi(closes, period)
    elif kind == "atr":
        highs, lows = bars.get("high"), bars.get("low")
        usable = highs is not None and lows is not None and None not in highs and None not in lows
        series = atr(closes, period, highs if usable else None, lows if usable else None)
    elif kind == "volume":
        series = [float(v) for v in volume] if has_volume else [None] * n
    elif kind == "volume_sma":
        series = moving_average([float(v) for v in volume], period) if has_volume else [None] * n
    elif kind == "highest":
        series = _extreme(closes, period, max)
    elif kind == "lowest":
        series = _extreme(closes, period, min)
    elif kind == "roc":
        series = [(closes[i] / closes[i - period] - 1) * 100 if i >= period else None for i in range(n)]
    else:
        raise ValueError(f"Unknown operand {kind}")
    cache[key] = series
    return series


LABELS = {
    "macd": "MACD",
    "macd_signal": "MACD signal",
    "macd_hist": "MACD histogram",
    "volume": "Volume",
    "volume_sma": "Volume SMA",
    "highest": "Highest close",
    "lowest": "Lowest close",
    "roc": "ROC%",
}


def label(operand: dict[str, Any]) -> str:
    kind = operand["kind"]
    if kind == "price":
        return "Price"
    if kind == "value":
        value = float(operand["value"])
        return f"{value:g}"
    if kind in FIXED:
        return LABELS[kind]
    name = LABELS.get(kind, kind.upper())
    return f"{name}({int(operand['period'])})"


OP_WORDS = {">": "is above", "<": "is below", "crosses_above": "crosses above", "crosses_below": "crosses below"}


def describe_condition(condition: dict[str, Any]) -> str:
    return f"{label(condition['left'])} {OP_WORDS[condition['op']]} {label(condition['right'])}"


def describe(rules: dict[str, Any]) -> str:
    short = rules.get("side") == "short"
    joiner = " or " if rules.get("entryMode") == "any" else " and "
    parts = [
        ("Sell short when " if short else "Buy when ") + joiner.join(describe_condition(c) for c in rules["entry"])
    ]
    exits = [describe_condition(c) for c in rules.get("exit") or []]
    if rules.get("stopLoss"):
        exits.append(
            f"price {'rises' if short else 'falls'} {rules['stopLoss'] * 100:g}% {'above' if short else 'below'} entry"
        )
    if rules.get("takeProfit"):
        exits.append(
            f"price {'falls' if short else 'rises'} {rules['takeProfit'] * 100:g}% {'below' if short else 'above'} entry"
        )
    if rules.get("trailingStop"):
        exits.append(
            f"price {'rises' if short else 'falls'} {rules['trailingStop'] * 100:g}% from its "
            f"{'lowest' if short else 'highest'} close since entry"
        )
    if rules.get("maxHoldDays"):
        exits.append(f"{int(rules['maxHoldDays'])} trading days have passed")
    close_word = "cover" if short else "sell"
    parts.append(
        f"{close_word} when " + " or ".join(exits) if exits else ("hold once shorted" if short else "hold once bought")
    )
    return "; ".join(parts) + "."


def warmup(rules: dict[str, Any]) -> int:
    periods = []
    for condition in [*rules["entry"], *(rules.get("exit") or [])]:
        for operand in (condition["left"], condition["right"]):
            if operand["kind"] in INDICATORS:
                periods.append(int(operand["period"]))
            elif operand["kind"] in {"macd", "macd_signal", "macd_hist"}:
                periods.append(MACD_WARMUP)
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


def _intraday_exit(
    side: int, open_: float, high: float, low: float, stop: float | None, take: float | None, gap_check: bool
) -> float | None:
    """Fill price if a stop or target was hit during the bar. A stop is assumed to fill before a target."""
    if side > 0:
        if gap_check and stop is not None and open_ <= stop:
            return open_
        if gap_check and take is not None and open_ >= take:
            return open_
        if stop is not None and low <= stop:
            return stop
        if take is not None and high >= take:
            return take
    else:
        if gap_check and stop is not None and open_ >= stop:
            return open_
        if gap_check and take is not None and open_ <= take:
            return open_
        if stop is not None and high >= stop:
            return stop
        if take is not None and low <= take:
            return take
    return None


def plan(
    closes: list[float], rules: dict[str, Any], *, bars: Bars | None = None, execution: str = "close"
) -> tuple[list[int], dict[str, Series], dict[int, float]]:
    """Target positions (1/0/-1), indicator lines, and intraday stop fills {bar: price}."""
    cache: dict[tuple, Series] = {}

    def prepared(conditions: list[dict[str, Any]]) -> list[tuple[dict[str, Any], Series, Series]]:
        return [
            (c, operand_series(closes, c["left"], cache, bars), operand_series(closes, c["right"], cache, bars))
            for c in conditions
        ]

    entry = prepared(rules["entry"])
    exits = prepared(rules.get("exit") or [])
    combine = any if rules.get("entryMode") == "any" else all
    side = -1 if rules.get("side") == "short" else 1
    stop_pct, take_pct = rules.get("stopLoss"), rules.get("takeProfit")
    trail_pct, max_hold = rules.get("trailingStop"), rules.get("maxHoldDays")
    bars = bars or {}
    opens, highs, lows = bars.get("open"), bars.get("high"), bars.get("low")
    intraday = all(series is not None and None not in series for series in (opens, highs, lows))
    next_open = execution == "next_open" and intraday

    n = len(closes)
    target = [0] * n
    fills: dict[int, float] = {}
    held, entered, entry_price, extreme = 0, -1, 0.0, 0.0
    for i in range(n):
        if held and i > entered:
            if next_open and i == entered + 1:
                entry_price = extreme = opens[i]  # the order placed at the entry close fills at this open
            stop = take = None
            if stop_pct:
                stop = entry_price * (1 - side * stop_pct)
            if trail_pct:
                trailing = extreme * (1 - side * trail_pct)
                stop = trailing if stop is None else (max(stop, trailing) if side > 0 else min(stop, trailing))
            if take_pct:
                take = entry_price * (1 + side * take_pct)
            exited = False
            if intraday:
                price = _intraday_exit(
                    side, opens[i], highs[i], lows[i], stop, take, gap_check=not (next_open and i == entered + 1)
                )
                if price is not None:
                    fills[i] = price
                    exited = True
            else:
                close = closes[i]
                hit_stop = stop is not None and (close <= stop if side > 0 else close >= stop)
                hit_take = take is not None and (close >= take if side > 0 else close <= take)
                exited = hit_stop or hit_take
            # The trailing reference moves after today's check, so a stop never uses today's own extreme.
            if side > 0:
                extreme = max(extreme, highs[i] if intraday else closes[i])
            else:
                extreme = min(extreme, lows[i] if intraday else closes[i])
            if not exited and max_hold and i - entered >= max_hold:
                exited = True
            if not exited and any(_holds(c, i, left, right) for c, left, right in exits):
                exited = True
            if exited:
                held = 0
        elif not held:
            if entry and combine(_holds(c, i, left, right) for c, left, right in entry):
                held, entered = 1, i
                entry_price = extreme = closes[i]
        target[i] = held * side
    indicators = {f"{kind.upper()}{period}": line for (kind, period), line in cache.items() if kind in {"sma", "ema"}}
    return target, indicators, fills


def signals(
    closes: list[float], rules: dict[str, Any], bars: Bars | None = None
) -> tuple[list[int], dict[str, Series]]:
    target, indicators, _ = plan(closes, rules, bars=bars)
    return target, indicators


# ---------- Export to TradingView Pine Script ----------


def _pine_operand(operand: dict[str, Any]) -> str:
    kind = operand["kind"]
    period = int(operand.get("period") or 0)
    return {
        "price": lambda: "close",
        "value": lambda: f"{float(operand['value']):g}",
        "sma": lambda: f"ta.sma(close, {period})",
        "ema": lambda: f"ta.ema(close, {period})",
        "rsi": lambda: f"ta.rsi(close, {period})",
        "atr": lambda: f"ta.atr({period})",
        "macd": lambda: "macdLine",
        "macd_signal": lambda: "macdSignal",
        "macd_hist": lambda: "macdHist",
        "volume": lambda: "volume",
        "volume_sma": lambda: f"ta.sma(volume, {period})",
        "highest": lambda: f"ta.highest(close, {period})[1]",
        "lowest": lambda: f"ta.lowest(close, {period})[1]",
        "roc": lambda: f"ta.roc(close, {period})",
    }[kind]()


def _pine_condition(condition: dict[str, Any]) -> str:
    left, right = _pine_operand(condition["left"]), _pine_operand(condition["right"])
    op = condition["op"]
    if op == "crosses_above":
        return f"ta.crossover({left}, {right})"
    if op == "crosses_below":
        return f"ta.crossunder({left}, {right})"
    return f"{left} {op} {right}"


def to_pine(rules: dict[str, Any], name: str = "Custom strategy", cost_pct: float = 0.15) -> str:
    """A TradingView Pine Script v5 strategy equivalent to the rules."""
    short = rules.get("side") == "short"
    direction = "strategy.short" if short else "strategy.long"
    joiner = " or " if rules.get("entryMode") == "any" else " and "
    uses_macd = any(
        operand["kind"] in {"macd", "macd_signal", "macd_hist"}
        for condition in [*rules["entry"], *(rules.get("exit") or [])]
        for operand in (condition["left"], condition["right"])
    )
    safe_name = name.replace('"', "'")[:60]
    lines = [
        "//@version=5",
        f'strategy("{safe_name}", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100,',
        f"     commission_type=strategy.commission.percent, commission_value={cost_pct:g}, process_orders_on_close=false)",
        "",
        f"// {describe(rules)}",
    ]
    if uses_macd:
        lines.append("[macdLine, macdSignal, macdHist] = ta.macd(close, 12, 26, 9)")
    lines.append("entryCond = " + joiner.join(f"({_pine_condition(c)})" for c in rules["entry"]))
    exits = rules.get("exit") or []
    lines.append("exitCond = " + (" or ".join(f"({_pine_condition(c)})" for c in exits) if exits else "false"))
    lines += [
        "",
        "if entryCond and strategy.position_size == 0",
        f'    strategy.entry("Rules", {direction})',
    ]
    if rules.get("maxHoldDays"):
        lines += [
            "var int entryBar = na",
            "if strategy.position_size != 0 and strategy.position_size[1] == 0",
            "    entryBar := bar_index",
            f"timeExit = strategy.position_size != 0 and bar_index - entryBar >= {int(rules['maxHoldDays'])}",
        ]
    else:
        lines.append("timeExit = false")
    lines += ["if exitCond or timeExit", '    strategy.close("Rules")']
    sign = -1 if short else 1
    stop_parts = []
    if rules.get("stopLoss"):
        stop_parts.append(f"strategy.position_avg_price * {1 - sign * rules['stopLoss']:g}")
    if rules.get("trailingStop"):
        pick = "ta.lowest" if short else "ta.highest"
        lines.append(
            f"trailRef = {pick}({'low' if short else 'high'}, math.max(1, bar_index - nz(strategy.opentrades.entry_bar_index(0), bar_index) + 1))"
        )
        stop_parts.append(f"trailRef * {1 - sign * rules['trailingStop']:g}")
    take = f"strategy.position_avg_price * {1 + sign * rules['takeProfit']:g}" if rules.get("takeProfit") else "na"
    if stop_parts or rules.get("takeProfit"):
        if len(stop_parts) == 2:
            stop = f"math.{'min' if short else 'max'}({stop_parts[0]}, {stop_parts[1]})"
        else:
            stop = stop_parts[0] if stop_parts else "na"
        lines += [
            "if strategy.position_size != 0",
            f'    strategy.exit("Stops", "Rules", stop={stop}, limit={take})',
        ]
    return "\n".join(lines) + "\n"
