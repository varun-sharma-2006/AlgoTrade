from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend import ml, risk, strategies, walkforward
from backend.config import logger, settings
from backend.deps import get_current_user, get_db
from backend.market import WATCHLIST_SYMBOLS, fetch_chart
from backend.routes.market import parse_symbols
from backend.schemas import PredictionPayload, TrainingPayload, WalkForwardPayload
from backend.stores import Store, now, serialize_mongo_doc

router = APIRouter(tags=["analytics"])


def strategy_params(payload: TrainingPayload) -> dict[str, float]:
    names = strategies.DEFAULT_PARAMS.get(payload.strategyId, {})
    return {name: getattr(payload, name) for name in names}


@router.get("/analytics/strategies")
async def get_strategies() -> list[dict[str, Any]]:
    return strategies.STRATEGIES


@router.get("/analytics/overview")
async def get_overview(
    user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    simulations = await store.list_simulations(user["id"])
    trained = await store.list_trained(user["id"])
    total_capital = sum(sim["startingCapital"] for sim in simulations)
    totals = {
        "totalSimulations": len(simulations),
        "activeSimulations": sum(1 for sim in simulations if sim["status"].lower() == "active"),
        "completedSimulations": sum(1 for sim in simulations if sim["status"].lower() == "completed"),
        "totalStartingCapital": total_capital,
        "averageStartingCapital": total_capital / len(simulations) if simulations else 0.0,
        "trainedModels": len(trained),
    }
    recent = sorted(simulations, key=lambda item: item["createdAt"], reverse=True)[:5]
    trained_symbols = [
        f"{entry['symbol']} ({(entry.get('payload') or {}).get('strategyId') or entry.get('strategyId') or entry.get('strategy_id')})"
        for entry in trained
    ]
    return {
        "totals": totals,
        "watchlist": WATCHLIST_SYMBOLS,
        "recentSimulations": serialize_mongo_doc(recent),
        "strategiesTrained": trained_symbols,
    }


@router.get("/analytics/sparkline", dependencies=[Depends(get_current_user)])
async def get_sparkline(symbols: str | None = None) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for symbol in parse_symbols(symbols):
        chart = fetch_chart(symbol, range_value="1mo", interval="1d")
        points = [{"timestamp": p["timestamp"], "close": p["close"]} for p in chart["points"][-40:]]
        series.append({"symbol": symbol, "points": points})
    return series


BENCHMARK_CACHE_SECONDS = 600
_benchmark_cache: dict[str, tuple[float, tuple[list[str], list[float]] | None]] = {}


def history(symbol: str) -> tuple[list[float], list[str]]:
    """Daily closes and timestamps over HISTORY_PERIOD (the backtest window plus warm-up and training data)."""
    chart = fetch_chart(symbol, range_value=settings.history_period, interval="1d")
    return [p["close"] for p in chart["points"]], [p["timestamp"] for p in chart["points"]]


def benchmark_history() -> tuple[list[str], list[float]] | None:
    cached = _benchmark_cache.get("index")
    if cached and time.time() - cached[0] < BENCHMARK_CACHE_SECONDS:
        return cached[1]
    try:
        closes, timestamps = history(risk.BENCHMARK_SYMBOL)
        data: tuple[list[str], list[float]] | None = (timestamps, closes)
    except HTTPException:
        logger.warning("Benchmark %s unavailable", risk.BENCHMARK_SYMBOL)
        data = None
    _benchmark_cache["index"] = (time.time(), data)
    return data


def backtest_on(
    symbol: str,
    closes: list[float],
    timestamps: list[str],
    strategy_id: str,
    params: dict[str, float],
    rules: dict[str, Any] | None = None,
    slippage_bps: float | None = None,
    bench: tuple[list[str], list[float]] | None = None,
) -> dict[str, Any]:
    problem = strategies.validate(strategy_id, params, rules)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    if len(closes) < strategies.warmup(strategy_id, params, rules) + 30:
        raise HTTPException(status_code=422, detail="Not enough price history for these parameters")
    start = strategies.evaluation_start(timestamps, strategies.period_days(settings.backtest_period))
    slippage = settings.slippage_bps if slippage_bps is None else slippage_bps
    report = strategies.backtest(
        closes,
        timestamps,
        strategy_id,
        params,
        settings.trading_fee_bps,
        rules,
        slippage_bps=slippage,
        start=start,
        benchmark=bench,
    )
    if strategy_id == "ml-logistic":
        report["model"] = ml.report(closes, params, start)
    return {"symbol": symbol.upper(), "strategyId": strategy_id, "parameters": params, "rules": rules, **report}


def run_backtest(
    symbol: str,
    strategy_id: str,
    params: dict[str, float],
    rules: dict[str, Any] | None = None,
    slippage_bps: float | None = None,
) -> dict[str, Any]:
    closes, timestamps = history(symbol)
    return backtest_on(symbol, closes, timestamps, strategy_id, params, rules, slippage_bps, benchmark_history())


def compare_strategies(symbol: str, slippage_bps: float | None = None) -> list[dict[str, Any]]:
    """Every built-in strategy with default settings on one symbol, best Sharpe first."""
    closes, timestamps = history(symbol)
    bench = benchmark_history()
    results = []
    for strategy_id, params in strategies.DEFAULT_PARAMS.items():
        if strategy_id == "custom":
            continue
        try:
            results.append(backtest_on(symbol, closes, timestamps, strategy_id, params, None, slippage_bps, bench))
        except HTTPException as exc:
            logger.info("Skipping %s for %s: %s", strategy_id, symbol, exc.detail)
    results.sort(key=lambda r: r["metrics"]["sharpe"], reverse=True)
    return results


def run_walk_forward(symbol: str, strategy_id: str, slippage_bps: float | None = None) -> dict[str, Any]:
    if strategy_id not in walkforward.GRIDS:
        raise HTTPException(status_code=422, detail="Walk-forward testing needs a strategy with parameters to tune")
    closes, timestamps = history(symbol)
    slippage = settings.slippage_bps if slippage_bps is None else slippage_bps
    try:
        result = walkforward.run(closes, timestamps, strategy_id, settings.trading_fee_bps + slippage)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"symbol": symbol.upper(), **result, "feeBps": settings.trading_fee_bps, "slippageBps": slippage}


@router.post("/analytics/train")
async def train_strategy(
    payload: TrainingPayload, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    params = strategy_params(payload)
    rules = payload.rules.model_dump() if payload.rules else None
    report = await asyncio.to_thread(
        run_backtest, payload.symbol, payload.strategyId, params, rules, payload.slippageBps
    )
    result = {
        **report,
        # Kept for clients that read the SMA windows directly.
        "shortWindow": payload.shortWindow,
        "longWindow": payload.longWindow,
        "trainedAt": now().isoformat(),
    }
    # The daily series is only needed for the response, not to remember which strategy was trained.
    stored = {key: value for key, value in result.items() if key != "sample"}
    await store.record_training(user["id"], payload.symbol, payload.strategyId, stored)
    return result


@router.post("/analytics/walk-forward", dependencies=[Depends(get_current_user)])
async def walk_forward(payload: WalkForwardPayload) -> dict[str, Any]:
    return await asyncio.to_thread(run_walk_forward, payload.symbol, payload.strategyId, payload.slippageBps)


@router.post("/analytics/predict")
async def predict(
    payload: PredictionPayload, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    rules = payload.rules.model_dump() if payload.rules else None
    if payload.strategyId is not None and payload.parameters is not None:
        # The client says which strategy it trained, so this works on any server instance.
        strategy_id, params = payload.strategyId, payload.parameters
    else:
        training = await store.get_training(user["id"], payload.symbol)
        if not training:
            raise HTTPException(status_code=404, detail="Train the strategy first")
        trained = training.get("payload") or {}
        strategy_id = trained.get("strategyId") or training.get("strategyId") or "sma-crossover"
        params = trained.get("parameters")
        if params is None:
            params = {"shortWindow": trained.get("shortWindow", 20), "longWindow": trained.get("longWindow", 60)}
        rules = rules or trained.get("rules")
    problem = strategies.validate(strategy_id, params, rules)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    closes, _ = history(payload.symbol)
    if len(closes) < strategies.warmup(strategy_id, params, rules) + 2:
        raise HTTPException(status_code=422, detail="Not enough data for prediction")
    outlook = strategies.current_signal(strategy_id, closes, params, rules)
    return {
        "symbol": payload.symbol.upper(),
        "strategyId": strategy_id,
        **outlook,
        "metadata": {"recent": closes[-5:], "parameters": params},
        "generatedAt": now().isoformat(),
    }
