"""Copilot actions: backend functions the chatbot can run, for Gemini's function calling and for the offline fallback.

Gemini decides when to call a tool ("Backtest AAPL with SMA 20/60" -> run_backtest), the Toolbox runs the
same code as the Strategy lab, and the numbers go back to Gemini to explain. Each successful call also
becomes an action card in the chat. When Gemini is unavailable, `local_action` recognises the common
requests with plain pattern matching, so "backtest NVDA" still works.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date
from typing import Any

from fastapi import HTTPException

from backend import advisor, strategies
from backend.config import logger
from backend.schemas import SimulationInput

STRATEGY_NAMES = {
    "sma-crossover": "SMA crossover",
    "mean-reversion": "Mean reversion (Bollinger)",
    "trend-follow": "Breakout (Donchian)",
    "ml-logistic": "Machine learning (logistic regression)",
    "buy-hold": "Buy & hold",
}
BUILT_IN = list(STRATEGY_NAMES)
TUNABLE = ["sma-crossover", "mean-reversion", "trend-follow", "ml-logistic"]
PARAMETER_SCHEMA = {
    "shortWindow": {"type": "integer", "description": "SMA crossover: short moving average, days (default 20)"},
    "longWindow": {"type": "integer", "description": "SMA crossover: long moving average, days (default 60)"},
    "lookback": {"type": "integer", "description": "Mean reversion: Bollinger lookback, days (default 20)"},
    "deviation": {"type": "number", "description": "Mean reversion: band width in standard deviations (default 2)"},
    "channel": {"type": "integer", "description": "Breakout: Donchian channel length, days (default 20)"},
    "threshold": {"type": "number", "description": "ML: probability of a rise needed to buy, 0.3-0.8 (default 0.52)"},
}
SYMBOL = {"type": "string", "description": "Yahoo Finance ticker, e.g. AAPL, RELIANCE.NS, BTC-USD"}

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "run_backtest",
        "description": (
            "Backtest a trading strategy on the last 2 years of real daily prices, with trading fees and slippage, "
            "and compare it with buy & hold and the S&P 500. Use this whenever the user asks how a strategy would "
            "have done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": SYMBOL,
                "strategy": {"type": "string", "enum": BUILT_IN},
                **PARAMETER_SCHEMA,
                "slippageBps": {"type": "number", "description": "Slippage per trade in basis points (default 5)"},
            },
            "required": ["symbol", "strategy"],
        },
    },
    {
        "name": "compare_strategies",
        "description": "Backtest every built-in strategy with default settings on one symbol and rank them by Sharpe ratio.",
        "parameters": {"type": "object", "properties": {"symbol": SYMBOL}, "required": ["symbol"]},
    },
    {
        "name": "walk_forward_test",
        "description": (
            "Walk-forward test: tune the strategy on each past year, trade the chosen settings on the next quarter, "
            "and roll forward over 5 years. Shows whether a strategy's backtest edge holds up out of sample."
        ),
        "parameters": {
            "type": "object",
            "properties": {"symbol": SYMBOL, "strategy": {"type": "string", "enum": TUNABLE}},
            "required": ["symbol", "strategy"],
        },
    },
    {
        "name": "create_simulation",
        "description": (
            "Create a paper-trading simulation in the user's portfolio. Only call this when the user explicitly asks "
            "to create, start or invest in a simulation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": SYMBOL,
                "strategy": {"type": "string", "enum": BUILT_IN},
                "startingCapital": {"type": "number", "description": "Paper money to invest, in USD"},
                "startDate": {
                    "type": "string",
                    "description": "Optional YYYY-MM-DD start date, today or up to one year ago (backdating)",
                },
                **PARAMETER_SCHEMA,
            },
            "required": ["symbol", "strategy", "startingCapital"],
        },
    },
]


def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _params(strategy_id: str, args: dict[str, Any]) -> dict[str, float]:
    defaults = strategies.DEFAULT_PARAMS.get(strategy_id, {})
    return {name: float(args.get(name, default)) for name, default in defaults.items()}


def _strategy(args: dict[str, Any], allowed: list[str]) -> str:
    strategy_id = str(args.get("strategy") or "sma-crossover")
    if strategy_id not in allowed:
        raise HTTPException(status_code=422, detail=f"Unknown strategy '{strategy_id}'")
    return strategy_id


def label(strategy_id: str, params: dict[str, float]) -> str:
    if strategy_id == "sma-crossover":
        return f"SMA {int(params['shortWindow'])}/{int(params['longWindow'])}"
    if strategy_id == "mean-reversion":
        return f"Bollinger {int(params['lookback'])}, {params['deviation']:g}σ"
    if strategy_id == "trend-follow":
        return f"Breakout {int(params['channel'])}-day"
    if strategy_id == "ml-logistic":
        return f"ML (P(up) ≥ {params['threshold']:.2f})"
    return STRATEGY_NAMES.get(strategy_id, strategy_id)


def summarise_backtest(result: dict[str, Any]) -> dict[str, Any]:
    """The numbers the LLM needs (and the chat card shows), without the daily series."""
    m = result["metrics"]
    bench = result.get("benchmark") or {}
    out = {
        "symbol": result["symbol"],
        "strategyId": result["strategyId"],
        "strategy": label(result["strategyId"], result["parameters"]),
        "parameters": result["parameters"],
        "periodStart": result["period"]["start"][:10],
        "periodEnd": result["period"]["end"][:10],
        "strategyReturn": m["totalReturn"],
        "buyHoldReturn": m["buyHoldReturn"],
        "sp500Return": bench.get("totalReturn"),
        "annualizedReturn": m["annualizedReturn"],
        "volatility": m["volatility"],
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "maxDrawdown": m["maxDrawdown"],
        "buyHoldSharpe": result["buyHold"]["sharpe"],
        "buyHoldMaxDrawdown": result["buyHold"]["maxDrawdown"],
        "betaToSp500": bench.get("beta"),
        "trades": m["trades"],
        "winRate": m["winRate"],
        "timeInMarket": m["exposure"],
        "costPerTradeBps": m["feeBps"] + m["slippageBps"],
    }
    model = result.get("model")
    if model:
        out |= {
            "modelAccuracy": model["accuracy"],
            "alwaysUpAccuracy": model["baselineAccuracy"],
            "modelAuc": model["auc"],
        }
    return {key: _round(value) for key, value in out.items()}


def summarise_walk_forward(result: dict[str, Any]) -> dict[str, Any]:
    m = result["metrics"]
    out = {
        "symbol": result["symbol"],
        "strategyId": result["strategyId"],
        "strategy": STRATEGY_NAMES[result["strategyId"]],
        "periodStart": result["period"]["start"][:10],
        "periodEnd": result["period"]["end"][:10],
        "folds": len(result["folds"]),
        "tunedInSampleAnnualized": m["inSampleAnnualized"],
        "outOfSampleAnnualized": m["outOfSampleAnnualized"],
        "outOfSampleReturn": m["outOfSampleReturn"],
        "outOfSampleSharpe": m["outOfSampleSharpe"],
        "outOfSampleMaxDrawdown": m["outOfSampleMaxDrawdown"],
        "buyHoldReturn": m["buyHoldReturn"],
        "buyHoldSharpe": m["buyHoldSharpe"],
        "foldsBeatBuyHold": m["foldsBeatBuyHold"],
        "mostChosenParams": m["mostChosenParams"],
        "mostChosenCount": m["mostChosenCount"],
    }
    return {key: _round(value) for key, value in out.items()}


class Toolbox:
    """Runs copilot tools for one chat request and collects the action cards to show."""

    def __init__(self, create_simulation: Callable[[SimulationInput], dict[str, Any]] | None = None) -> None:
        self.actions: list[dict[str, Any]] = []
        self.create_simulation = create_simulation
        self._memo: dict[tuple[str, str], dict[str, Any]] = {}

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        # Retries with another Gemini model can repeat a call; answer from memory so nothing runs twice.
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key not in self._memo:
            self._memo[key] = self._run(name, args or {})
        return self._memo[key]

    def _run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        from backend.routes import analytics
        from backend.routes.simulations import prepare_simulation

        try:
            symbol = str(args.get("symbol") or "").strip().upper()
            if not symbol:
                return {"error": "A ticker symbol is required"}
            if name == "run_backtest":
                strategy_id = _strategy(args, BUILT_IN)
                slippage = args.get("slippageBps")
                result = analytics.run_backtest(
                    symbol,
                    strategy_id,
                    _params(strategy_id, args),
                    None,
                    float(slippage) if slippage is not None else None,
                )
                summary = summarise_backtest(result)
                self.actions.append(
                    {"type": "backtest", "label": f"Backtest · {symbol} · {summary['strategy']}", "data": summary}
                )
                return summary
            if name == "compare_strategies":
                rows = [summarise_backtest(r) for r in analytics.compare_strategies(symbol)]
                if not rows:
                    return {"error": f"Not enough price history for {symbol}"}
                self.actions.append(
                    {
                        "type": "comparison",
                        "label": f"Compared {len(rows)} strategies on {symbol}",
                        "data": {"rows": rows},
                    }
                )
                return {"symbol": symbol, "rankedBySharpe": rows}
            if name == "walk_forward_test":
                strategy_id = _strategy(args, TUNABLE)
                summary = summarise_walk_forward(analytics.run_walk_forward(symbol, strategy_id))
                self.actions.append(
                    {
                        "type": "walkforward",
                        "label": f"Walk-forward · {symbol} · {summary['strategy']}",
                        "data": summary,
                    }
                )
                return summary
            if name == "create_simulation":
                if self.create_simulation is None:
                    return {"error": "Creating simulations is not available here"}
                strategy_id = _strategy(args, BUILT_IN)
                start = args.get("startDate")
                payload = prepare_simulation(
                    SimulationInput(
                        symbol=symbol,
                        strategy=STRATEGY_NAMES[strategy_id],
                        strategyId=strategy_id,
                        parameters=_params(strategy_id, args),
                        startingCapital=float(args.get("startingCapital") or 0),
                        startDate=date.fromisoformat(str(start)[:10]) if start else None,
                        notes="Created by the trading copilot",
                    )
                )
                simulation = self.create_simulation(payload)
                self.actions.append(
                    {
                        "type": "simulation",
                        "label": f"Created simulation · {symbol} · {payload.strategy}",
                        "data": simulation,
                    }
                )
                return {
                    "created": True,
                    "symbol": symbol,
                    "strategy": payload.strategy,
                    "startingCapital": payload.startingCapital,
                    "startDate": simulation.get("startDate"),
                }
            return {"error": f"Unknown tool {name}"}
        except HTTPException as exc:
            return {"error": str(exc.detail)}
        except (ValueError, TypeError) as exc:
            return {"error": f"Invalid arguments: {str(exc)[:200]}"}
        except Exception as exc:  # a tool must never break the chat
            logger.warning("Copilot tool %s failed: %s", name, exc)
            return {"error": "The tool failed; try again in a moment"}


# ---------- Offline fallback: recognise common requests without the LLM ----------

STRATEGY_WORDS = [
    ("ml-logistic", r"machine[- ]learning|\bml\b|logistic|\bai model|predictive model"),
    ("mean-reversion", r"mean[- ]reversion|bollinger"),
    ("trend-follow", r"breakout|donchian|trend[- ]follow"),
    ("buy-hold", r"buy[- ](?:and|&|n)[- ]hold"),
    ("sma-crossover", r"\bsma\b|moving[- ]average|crossover|golden cross"),
]
NOT_SYMBOLS = {"ML", "SMA", "EMA", "RSI", "AI", "LR", "SP", "BH"}


def detect_strategy(message: str) -> str | None:
    lowered = message.lower()
    return next((sid for sid, pattern in STRATEGY_WORDS if re.search(pattern, lowered)), None)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def describe_backtest(s: dict[str, Any]) -> str:
    beat = s["strategyReturn"] - s["buyHoldReturn"]
    lines = [
        f"Backtest: {s['symbol']} · {s['strategy']} · {s['periodStart']} to {s['periodEnd']}, "
        f"{s['costPerTradeBps']:g} bps cost per trade.",
        f"- Strategy return {_pct(s['strategyReturn'])} vs buy & hold {_pct(s['buyHoldReturn'])} "
        f"({'beat' if beat >= 0 else 'lagged'} it by {abs(beat) * 100:.1f} points); S&P 500 {_pct(s['sp500Return'])}.",
        f"- Risk: Sharpe {s['sharpe']:.2f} (buy & hold {s['buyHoldSharpe']:.2f}), Sortino {s['sortino']:.2f}, "
        f"volatility {s['volatility'] * 100:.1f}%, max drawdown {s['maxDrawdown'] * 100:.1f}% "
        f"(buy & hold {s['buyHoldMaxDrawdown'] * 100:.1f}%).",
        f"- Trading: {s['trades']} trades, win rate {s['winRate'] * 100:.0f}%, "
        f"in the market {s['timeInMarket'] * 100:.0f}% of the time"
        + (f", beta to the S&P 500 {s['betaToSp500']:.2f}." if s.get("betaToSp500") is not None else "."),
    ]
    if "modelAccuracy" in s:
        lines.append(
            f"- Model: {s['modelAccuracy'] * 100:.1f}% direction accuracy out of sample vs "
            f"{s['alwaysUpAccuracy'] * 100:.1f}% for always guessing the majority direction."
        )
    return "\n".join(lines)


def describe_comparison(symbol: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"All built-in strategies on {symbol}, last 2 years, ranked by Sharpe ratio:"]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"{i}. {row['strategy']}: return {_pct(row['strategyReturn'])}, Sharpe {row['sharpe']:.2f}, "
            f"max drawdown {row['maxDrawdown'] * 100:.1f}%"
        )
    lines.append(
        f"Returns include {rows[0]['costPerTradeBps']:g} bps of fees and slippage per trade; "
        f"the S&P 500 returned {_pct(rows[0]['sp500Return'])} over the same period."
    )
    return "\n".join(lines)


def describe_walk_forward(s: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Walk-forward test: {s['symbol']} · {s['strategy']} · {s['folds']} quarterly folds, "
            f"{s['periodStart']} to {s['periodEnd']}.",
            f"- Tuned on each past year it looked like {_pct(s['tunedInSampleAnnualized'])} a year; "
            f"out of sample it made {_pct(s['outOfSampleAnnualized'])} a year.",
            f"- Out-of-sample total {_pct(s['outOfSampleReturn'])} vs buy & hold {_pct(s['buyHoldReturn'])}; "
            f"Sharpe {s['outOfSampleSharpe']:.2f} vs {s['buyHoldSharpe']:.2f}.",
            f"- It beat buy & hold in {s['foldsBeatBuyHold']} of {s['folds']} quarters. The most common settings "
            f"({', '.join(f'{k} {v:g}' for k, v in s['mostChosenParams'].items())}) were picked "
            f"{s['mostChosenCount']} times.",
        ]
    )


def local_action(message: str, toolbox: Toolbox) -> str | None:
    """Run a backtest, comparison or walk-forward test when the message clearly asks for one."""
    lowered = message.lower()
    symbols = [s for s in advisor.extract_symbols(message) if s not in NOT_SYMBOLS]
    if not symbols:
        return None
    symbol = symbols[0]
    strategy_id = detect_strategy(message)
    if re.search(r"walk[- ]?forward|out[- ]of[- ]sample", lowered):
        chosen = strategy_id if strategy_id in TUNABLE else "sma-crossover"
        result = toolbox.call("walk_forward_test", {"symbol": symbol, "strategy": chosen})
        return result.get("error") or describe_walk_forward(result)
    wants_compare = re.search(r"\bcompare\b|best strateg|which strateg|all (?:the )?strateg", lowered)
    wants_backtest = re.search(r"back[- ]?test", lowered) or (strategy_id and re.search(r"\btest\b", lowered))
    if wants_compare or (wants_backtest and not strategy_id):
        result = toolbox.call("compare_strategies", {"symbol": symbol})
        return result.get("error") or describe_comparison(symbol, result["rankedBySharpe"])
    if wants_backtest and strategy_id:
        args: dict[str, Any] = {"symbol": symbol, "strategy": strategy_id}
        windows = re.search(r"(\d{1,3})\s*(?:/|and|,|-)\s*(\d{1,3})", message)
        if strategy_id == "sma-crossover" and windows:
            args |= {"shortWindow": int(windows.group(1)), "longWindow": int(windows.group(2))}
        result = toolbox.call("run_backtest", args)
        return result.get("error") or describe_backtest(result)
    return None
