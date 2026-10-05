"""Export a backtest as a Jupyter notebook that reproduces it independently with pandas and yfinance.

The notebook re-implements the strategy's signals and the same trading rules (decision at the close, fill at
the close or the next open, costs on every buy and sell), so anyone can audit or extend the result outside
the app. The machine-learning strategy isn't exported.
"""

from __future__ import annotations

import json
from typing import Any

SIGNAL_CODE = {
    "buy-hold": "signal = pd.Series(1, index=df.index)\nsignal.iloc[0] = 0",
    "sma-crossover": (
        "short = df.Close.rolling({shortWindow}).mean()\nlong = df.Close.rolling({longWindow}).mean()\n"
        "signal = (short > long).astype(int)\nif ALLOW_SHORT:\n    signal[short < long] = -1"
    ),
    "mean-reversion": (
        "mid = df.Close.rolling({lookback}).mean()\nstd = df.Close.rolling({lookback}).std(ddof=0)\n"
        "lower, upper = mid - {deviation} * std, mid + {deviation} * std\n"
        "state, held = [], 0\n"
        "for close, m, lo, up in zip(df.Close, mid, lower, upper):\n"
        "    if not np.isnan(m):\n"
        "        if held == 0 and close < lo: held = 1\n"
        "        elif held == 0 and ALLOW_SHORT and close > up: held = -1\n"
        "        elif held == 1 and close >= m: held = 0\n"
        "        elif held == -1 and close <= m: held = 0\n"
        "    state.append(held)\n"
        "signal = pd.Series(state, index=df.index)"
    ),
    "trend-follow": (
        "high = df.Close.shift(1).rolling({channel}).max()\nlow = df.Close.shift(1).rolling({channel}).min()\n"
        "exit_low = df.Close.shift(1).rolling(max({channel} // 2, 2)).min()\n"
        "exit_high = df.Close.shift(1).rolling(max({channel} // 2, 2)).max()\n"
        "state, held = [], 0\n"
        "for close, h, l, xl, xh in zip(df.Close, high, low, exit_low, exit_high):\n"
        "    if not np.isnan(h):\n"
        "        if held == 0 and close > h: held = 1\n"
        "        elif held == 0 and ALLOW_SHORT and close < l: held = -1\n"
        "        elif held == 1 and close < xl: held = 0\n"
        "        elif held == -1 and close > xh: held = 0\n"
        "    state.append(held)\n"
        "signal = pd.Series(state, index=df.index)"
    ),
    "regime-switch": (
        "change = df.Close.diff().abs()\n"
        "er = (df.Close - df.Close.shift({erWindow})).abs() / change.rolling({erWindow}).sum()\n"
        "short, long = df.Close.rolling(20).mean(), df.Close.rolling(50).mean()\n"
        "mid = df.Close.rolling(20).mean(); std = df.Close.rolling(20).std(ddof=0)\n"
        "lower, upper = mid - 2 * std, mid + 2 * std\n"
        "state, held = [], 0\n"
        "for close, m, lo, up in zip(df.Close, mid, lower, upper):\n"
        "    if not np.isnan(m):\n"
        "        if held == 0 and close < lo: held = 1\n"
        "        elif held == 0 and ALLOW_SHORT and close > up: held = -1\n"
        "        elif held == 1 and close >= m: held = 0\n"
        "        elif held == -1 and close <= m: held = 0\n"
        "    state.append(held)\n"
        "ranging = pd.Series(state, index=df.index)\n"
        "trend = (short > long).astype(int) - ((short < long) & ALLOW_SHORT).astype(int)\n"
        "signal = trend.where(er >= {erThreshold}, ranging).where(long.notna() & er.notna(), 0)"
    ),
}

RULES_CODE = """import operator

def operand(spec):
    kind, n = spec["kind"], spec.get("period")
    c = df.Close
    if kind == "price": return c
    if kind == "value": return pd.Series(spec["value"], index=df.index)
    if kind == "sma": return c.rolling(n).mean()
    if kind == "ema": return c.ewm(span=n, adjust=False, min_periods=n).mean()
    if kind == "rsi":
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        return 100 - 100 / (1 + gain / loss)
    if kind in ("macd", "macd_signal", "macd_hist"):
        line = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        sig = line.ewm(span=9, adjust=False).mean()
        return {"macd": line, "macd_signal": sig, "macd_hist": line - sig}[kind]
    if kind == "atr":
        tr = pd.concat([df.High - df.Low, (df.High - c.shift()).abs(), (df.Low - c.shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    if kind == "volume": return df.Volume.astype(float)
    if kind == "volume_sma": return df.Volume.astype(float).rolling(n).mean()
    if kind == "highest": return c.shift(1).rolling(n).max()
    if kind == "lowest": return c.shift(1).rolling(n).min()
    if kind == "roc": return (c / c.shift(n) - 1) * 100
    raise ValueError(kind)

def holds(cond):
    a, b = operand(cond["left"]), operand(cond["right"])
    if cond["op"] == ">": return a > b
    if cond["op"] == "<": return a < b
    if cond["op"] == "crosses_above": return (a > b) & (a.shift() <= b.shift())
    return (a < b) & (a.shift() >= b.shift())

entry = [holds(c) for c in RULES["entry"]]
entry = pd.concat(entry, axis=1).any(axis=1) if RULES.get("entryMode") == "any" else pd.concat(entry, axis=1).all(axis=1)
exits = pd.concat([holds(c) for c in RULES.get("exit") or []], axis=1).any(axis=1) if RULES.get("exit") else pd.Series(False, index=df.index)
side = -1 if RULES.get("side") == "short" else 1
state, held, entry_i, entry_price, extreme = [], 0, 0, 0.0, 0.0
for i, close in enumerate(df.Close):
    if held and i > entry_i:
        stop = RULES.get("stopLoss"); take = RULES.get("takeProfit"); trail = RULES.get("trailingStop")
        hit = (stop and (close <= entry_price * (1 - stop) if side > 0 else close >= entry_price * (1 + stop))) \\
            or (take and (close >= entry_price * (1 + take) if side > 0 else close <= entry_price * (1 - take))) \\
            or (trail and (close <= extreme * (1 - trail) if side > 0 else close >= extreme * (1 + trail)))
        extreme = max(extreme, close) if side > 0 else min(extreme, close)
        if hit or (RULES.get("maxHoldDays") and i - entry_i >= RULES["maxHoldDays"]) or exits.iloc[i]:
            held = 0
    elif not held and entry.iloc[i]:
        held, entry_i, entry_price, extreme = 1, i, close, close
    state.append(held * side)
signal = pd.Series(state, index=df.index)
# Note: stops here are checked on closing prices; the app checks them against daily highs and lows."""

BACKTEST_CODE = """# Trade the signal: decided at each close, filled at the next open (or the same close), costs on every trade.
position = signal.shift(1).fillna(0) if EXECUTION == "next_open" else signal
window = df.index >= df.index[-1] - pd.Timedelta(days=BACKTEST_DAYS)
d = df[window].copy()
pos = position[window].copy()
pos.iloc[0] = 0  # start in cash
prev_close = df.Close.shift(1)[window]
if EXECUTION == "next_open":
    gap = d.Open / prev_close - 1        # overnight move, earned by yesterday's position
    session = d.Close / d.Open - 1       # today's move, earned by the position filled at the open
    gross = pos.shift(1).fillna(0) * gap + pos * session
else:
    gross = pos.shift(1).fillna(0) * (d.Close / prev_close - 1)
traded = pos.diff().fillna(pos)
cost = traded.clip(lower=0) * BUY_COST + (-traded).clip(lower=0) * SELL_COST
returns = (1 + gross) * (1 - cost) - 1
equity = (1 + returns).cumprod()
hold = d.Close / d.Close.iloc[0]

years = len(returns) / 252
sharpe = (returns.mean() - RISK_FREE / 252) / returns.std() * np.sqrt(252)
drawdown = (equity / equity.cummax() - 1).min()
print(f"Strategy return {equity.iloc[-1] - 1:+.2%}  (buy & hold {hold.iloc[-1] - 1:+.2%})")
print(f"Annualised {equity.iloc[-1] ** (1 / years) - 1:+.2%}, Sharpe {sharpe:.2f}, max drawdown {drawdown:.2%}")
print(f"Trades (entries): {int((traded > 0).sum())}")"""

PLOT_CODE = """ax = equity.plot(label="Strategy", figsize=(10, 4))
hold.plot(ax=ax, label="Buy & hold", linestyle="--")
ax.set_title(f"{SYMBOL}: growth of $1"); ax.legend(); plt.show()"""


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def build(
    symbol: str,
    strategy_id: str,
    params: dict[str, float],
    rules: dict[str, Any] | None,
    *,
    title: str,
    execution: str,
    buy_cost_bps: float,
    sell_cost_bps: float,
    risk_free: float,
    allow_short: bool,
    backtest_days: int,
    summary: str,
) -> dict[str, Any]:
    if strategy_id == "custom":
        signal = RULES_CODE
    elif strategy_id in SIGNAL_CODE:
        values = {k: (int(v) if float(v).is_integer() else v) for k, v in params.items()}
        signal = SIGNAL_CODE[strategy_id].format(**values)
    else:
        raise ValueError("This strategy can't be exported to a notebook")
    settings = (
        f'SYMBOL = "{symbol}"\nEXECUTION = "{execution}"  # "next_open" or "close"\n'
        f"BUY_COST = {buy_cost_bps / 10_000:.6f}  # fee + slippage per buy, as a fraction\n"
        f"SELL_COST = {sell_cost_bps / 10_000:.6f}\nRISK_FREE = {risk_free}\nALLOW_SHORT = {allow_short}\n"
        f"BACKTEST_DAYS = {backtest_days}\n"
    )
    if rules is not None:
        settings += f"RULES = {json.dumps(rules, indent=2)}\n"
    cells = [
        _markdown(
            f"# {title}\n\n{summary}\n\nExported from Algo Trade Simulator. This notebook re-implements the strategy "
            "with pandas so you can audit or extend it. Small differences from the app are expected (data "
            "adjustments, and the app checks stops against daily highs and lows)."
        ),
        _code("%pip install --quiet yfinance pandas numpy matplotlib"),
        _code("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport yfinance as yf"),
        _code(settings),
        _code(
            'df = yf.download(SYMBOL, period="5y", interval="1d", auto_adjust=True, progress=False)\n'
            "if isinstance(df.columns, pd.MultiIndex):\n    df.columns = df.columns.get_level_values(0)\n"
            "df = df.dropna()\ndf.tail()"
        ),
        _markdown(
            "## Signal\nThe target position for each day (1 = long, 0 = flat, -1 = short), using only data up to that day."
        ),
        _code(signal),
        _markdown("## Backtest"),
        _code(BACKTEST_CODE),
        _code(PLOT_CODE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
