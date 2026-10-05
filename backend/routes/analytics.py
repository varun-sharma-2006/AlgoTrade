from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend import basket, costs, ml, risk, robustness, strategies, walkforward
from backend import rules as rule_engine
from backend.config import logger, settings
from backend.deps import get_current_user, get_db
from backend.market import WATCHLIST_SYMBOLS, fetch_chart
from backend.routes.market import parse_symbols
from backend.schemas import (
    BasketPayload,
    PineExportPayload,
    PredictionPayload,
    RobustnessPayload,
    TrainingPayload,
    WalkForwardPayload,
)
from backend.stores import Store, now, serialize_mongo_doc

router = APIRouter(tags=["analytics"])

MAX_ML_BASKET = 8  # the ML strategy retrains per stock, so big ML baskets would be slow


def strategy_params(payload: TrainingPayload) -> dict[str, float]:
    names = strategies.DEFAULT_PARAMS.get(payload.strategyId, {})
    return {name: getattr(payload, name) for name in names}


def trading_options(payload: TrainingPayload | None = None) -> dict[str, Any]:
    """Execution, sizing, shorting and cost settings for a backtest, with server defaults for unset fields."""
    if payload is None:
        return {
            "execution": settings.execution,
            "sizing": {"mode": "full"},
            "allowShort": False,
            "borrowBps": settings.borrow_bps,
            "riskFree": settings.risk_free_rate,
            "slippage": settings.slippage_bps,
        }
    return {
        "execution": payload.execution or settings.execution,
        "sizing": {
            "mode": payload.sizing,
            "fraction": payload.sizeFraction,
            "targetVol": payload.targetVol,
            "maxLeverage": payload.maxLeverage,
        },
        "allowShort": payload.allowShort,
        "borrowBps": settings.borrow_bps if payload.borrowBps is None else payload.borrowBps,
        "riskFree": settings.risk_free_rate if payload.riskFreeRate is None else payload.riskFreeRate,
        "slippage": settings.slippage_bps if payload.slippageBps is None else payload.slippageBps,
    }


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


def load(symbol: str) -> dict[str, Any]:
    """Daily closes, timestamps, OHLCV bars and currency over HISTORY_PERIOD (backtest window plus warm-up)."""
    chart = fetch_chart(symbol, range_value=settings.history_period, interval="1d")
    points = chart["points"]
    bars = {key: [p.get(key) for p in points] for key in ("open", "high", "low", "volume")}
    return {
        "closes": [p["close"] for p in points],
        "timestamps": [p["timestamp"] for p in points],
        "bars": bars,
        "currency": chart.get("currency") or "USD",
    }


def history(symbol: str) -> tuple[list[float], list[str]]:
    """Daily closes and timestamps over HISTORY_PERIOD."""
    data = load(symbol)
    return data["closes"], data["timestamps"]


def benchmark_history(index: str = risk.BENCHMARK_SYMBOL) -> tuple[list[str], list[float]] | None:
    cached = _benchmark_cache.get(index)
    if cached and time.time() - cached[0] < BENCHMARK_CACHE_SECONDS:
        return cached[1]
    try:
        closes, timestamps = history(index)
        data: tuple[list[str], list[float]] | None = (timestamps, closes)
    except HTTPException:
        logger.warning("Benchmark %s unavailable", index)
        data = None
    _benchmark_cache[index] = (time.time(), data)
    return data


def market_series(timestamps: list[str], bench: tuple[list[str], list[float]] | None) -> list[float | None] | None:
    """The index's close on each of `timestamps` (None where it didn't trade), for the ML market features."""
    if not bench:
        return None
    by_day = {ts[:10]: close for ts, close in zip(*bench, strict=True)}
    return [by_day.get(ts[:10]) for ts in timestamps]


def backtest_on(
    symbol: str,
    data: dict[str, Any] | tuple[list[float], list[str]],
    strategy_id: str,
    params: dict[str, float],
    rules: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    bench: tuple[list[str], list[float]] | None = None,
) -> dict[str, Any]:
    if isinstance(data, tuple):  # (closes, timestamps) without OHLC bars
        data = {"closes": data[0], "timestamps": data[1], "bars": None, "currency": "USD"}
    closes, timestamps, bars = data["closes"], data["timestamps"], data["bars"]
    options = options or trading_options()
    problem = strategies.validate(strategy_id, params, rules)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    if len(closes) < strategies.warmup(strategy_id, params, rules) + 30:
        raise HTTPException(status_code=422, detail="Not enough price history for these parameters")
    start = strategies.evaluation_start(timestamps, strategies.period_days(settings.backtest_period))
    cost = costs.cost_model(symbol, settings.trading_fee_bps, options["slippage"], settings.india_brokerage_bps)
    market = market_series(timestamps, bench) if strategy_id == "ml-logistic" else None
    report = strategies.backtest(
        closes,
        timestamps,
        strategy_id,
        params,
        cost["buyFeeBps"],
        rules,
        slippage_bps=options["slippage"],
        sell_fee_bps=cost["sellFeeBps"],
        start=start,
        benchmark=bench,
        benchmark_info=costs.benchmark_for(symbol),
        bars=bars,
        execution=options["execution"],
        sizing=options["sizing"],
        allow_short=options["allowShort"],
        borrow_bps=options["borrowBps"],
        risk_free=options["riskFree"],
        market_closes=market,
    )
    if strategy_id == "ml-logistic":
        report["model"] = ml.report(closes, params, start, cost_bps=cost["buyBps"], bars=bars, market=market)
    return {
        "symbol": symbol.upper(),
        "strategyId": strategy_id,
        "parameters": params,
        "rules": rules,
        "currency": data.get("currency") or "USD",
        "costs": cost,
        **report,
    }


def run_backtest(
    symbol: str,
    strategy_id: str,
    params: dict[str, float],
    rules: dict[str, Any] | None = None,
    slippage_bps: float | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or trading_options()
    if slippage_bps is not None:
        options = options | {"slippage": slippage_bps}
    data = load(symbol)
    bench = benchmark_history(costs.benchmark_for(symbol)[0])
    return backtest_on(symbol, data, strategy_id, params, rules, options, bench)


def compare_strategies(symbol: str, slippage_bps: float | None = None) -> list[dict[str, Any]]:
    """Every built-in strategy with default settings on one symbol, best Sharpe first."""
    data = load(symbol)
    bench = benchmark_history(costs.benchmark_for(symbol)[0])
    options = trading_options()
    if slippage_bps is not None:
        options["slippage"] = slippage_bps
    results = []
    for strategy_id, params in strategies.DEFAULT_PARAMS.items():
        if strategy_id == "custom":
            continue
        try:
            results.append(backtest_on(symbol, data, strategy_id, params, None, options, bench))
        except HTTPException as exc:
            logger.info("Skipping %s for %s: %s", strategy_id, symbol, exc.detail)
    results.sort(key=lambda r: r["metrics"]["sharpe"], reverse=True)
    return results


def run_walk_forward(
    symbol: str,
    strategy_id: str,
    slippage_bps: float | None = None,
    *,
    execution: str | None = None,
    allow_short: bool = False,
) -> dict[str, Any]:
    if strategy_id not in walkforward.GRIDS:
        raise HTTPException(status_code=422, detail="Walk-forward testing needs a strategy with parameters to tune")
    data = load(symbol)
    slippage = settings.slippage_bps if slippage_bps is None else slippage_bps
    cost = costs.cost_model(symbol, settings.trading_fee_bps, slippage, settings.india_brokerage_bps)
    opens = data["bars"].get("open") if data["bars"] else None
    try:
        result = walkforward.run(
            data["closes"],
            data["timestamps"],
            strategy_id,
            cost["buyBps"],
            sell_cost_bps=cost["sellBps"],
            opens=None if opens is None or None in opens else opens,
            execution=execution or settings.execution,
            allow_short=allow_short,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "symbol": symbol.upper(),
        **result,
        "feeBps": cost["buyFeeBps"],
        "slippageBps": slippage,
        "costs": cost,
    }


def _public(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "dailyReturns"}


@router.post("/analytics/train")
async def train_strategy(
    payload: TrainingPayload, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    params = strategy_params(payload)
    rules = payload.rules.model_dump() if payload.rules else None
    report = await asyncio.to_thread(
        run_backtest, payload.symbol, payload.strategyId, params, rules, None, trading_options(payload)
    )
    result = {
        **_public(report),
        # Kept for clients that read the SMA windows directly.
        "shortWindow": payload.shortWindow,
        "longWindow": payload.longWindow,
        "trainedAt": now().isoformat(),
    }
    # The daily series are only needed for the response, not to remember which strategy was trained.
    stored = {key: value for key, value in result.items() if key not in {"sample", "monthly"}}
    await store.record_training(user["id"], payload.symbol, payload.strategyId, stored)
    return result


@router.post("/analytics/walk-forward", dependencies=[Depends(get_current_user)])
async def walk_forward(payload: WalkForwardPayload) -> dict[str, Any]:
    return await asyncio.to_thread(
        run_walk_forward,
        payload.symbol,
        payload.strategyId,
        payload.slippageBps,
        execution=payload.execution,
        allow_short=payload.allowShort,
    )


def run_robustness(payload: RobustnessPayload) -> dict[str, Any]:
    params = strategy_params(payload)
    rules = payload.rules.model_dump() if payload.rules else None
    options = trading_options(payload)
    data = load(payload.symbol)
    bench = benchmark_history(costs.benchmark_for(payload.symbol)[0])
    report = backtest_on(payload.symbol, data, payload.strategyId, params, rules, options, bench)
    daily = report["dailyReturns"]
    sample = report["sample"]
    hold = [sample[k]["buyHold"] / sample[k - 1]["buyHold"] - 1 for k in range(1, len(sample))]
    monte = robustness.monte_carlo(daily, hold, paths=payload.paths)
    if monte:
        for point in monte["fan"]:
            point["timestamp"] = sample[min(point["step"], len(sample) - 1)]["timestamp"]
    start = strategies.evaluation_start(data["timestamps"], strategies.period_days(settings.backtest_period))
    opens = data["bars"].get("open")
    grid, trials = (None, [])
    if payload.strategyId != "custom":
        grid, trials = robustness.sensitivity(
            data["closes"],
            start,
            payload.strategyId,
            params,
            report["costs"]["buyBps"],
            report["costs"]["sellBps"],
            opens=None if opens is None or None in opens else opens,
            execution=options["execution"],
            allow_short=options["allowShort"],
            risk_free=options["riskFree"],
            bars=data["bars"],
            market_closes=market_series(data["timestamps"], bench) if payload.strategyId == "ml-logistic" else None,
        )
    rf = options["riskFree"]
    trial_sharpes = [robustness.daily_sharpe(r, rf) for r in trials] or [robustness.daily_sharpe(daily, rf)]
    return {
        "symbol": payload.symbol.upper(),
        "strategyId": payload.strategyId,
        "parameters": params,
        "metrics": {
            key: report["metrics"][key] for key in ("totalReturn", "sharpe", "maxDrawdown", "buyHoldReturn", "trades")
        },
        "monteCarlo": monte,
        "sensitivity": grid,
        "deflatedSharpe": robustness.deflated_sharpe(daily, trial_sharpes, rf),
        "pbo": robustness.pbo(trials) if trials else None,
        "period": report["period"],
    }


@router.post("/analytics/robustness", dependencies=[Depends(get_current_user)])
async def robustness_check(payload: RobustnessPayload) -> dict[str, Any]:
    return await asyncio.to_thread(run_robustness, payload)


def _load_many(symbols: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    def one(symbol: str) -> dict[str, Any] | None:
        try:
            return load(symbol)
        except HTTPException:
            return None

    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
        loaded = dict(zip(symbols, pool.map(one, symbols), strict=True))
    return {s: d for s, d in loaded.items() if d}, [s for s, d in loaded.items() if not d]


def run_basket(payload: BasketPayload) -> dict[str, Any]:
    strategy_id = payload.strategyId
    if strategy_id == "ml-logistic" and len(payload.symbols) > MAX_ML_BASKET:
        raise HTTPException(status_code=422, detail=f"The ML strategy supports up to {MAX_ML_BASKET} stocks per basket")
    params = strategies.DEFAULT_PARAMS.get(strategy_id, {}) | payload.parameters
    rules = payload.rules.model_dump() if payload.rules else None
    problem = strategies.validate(strategy_id, params, rules)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    loaded, missing = _load_many(payload.symbols)
    if len(loaded) < 2:
        raise HTTPException(status_code=502, detail=f"Not enough price data (missing: {', '.join(missing) or 'all'})")
    slippage = settings.slippage_bps if payload.slippageBps is None else payload.slippageBps
    cost_models = {
        s: costs.cost_model(s, settings.trading_fee_bps, slippage, settings.india_brokerage_bps) for s in loaded
    }
    all_indian = all(costs.market(s) == "IN" for s in loaded)
    bench_info = costs.BENCHMARKS["IN" if all_indian else "US"]
    bench = benchmark_history(bench_info[0])
    risk_free = settings.risk_free_rate if payload.riskFreeRate is None else payload.riskFreeRate
    try:
        result = basket.run(
            {s: (d["closes"], d["timestamps"]) for s, d in loaded.items()},
            strategy_id,
            params,
            rules,
            weighting=payload.weighting,
            rebalance=payload.rebalance,
            top_n=payload.topN,
            costs={s: (m["buyBps"], m["sellBps"]) for s, m in cost_models.items()},
            allow_short=payload.allowShort,
            eval_days=strategies.period_days(settings.backtest_period),
            benchmark=bench,
            benchmark_info=bench_info,
            risk_free=risk_free,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The same strategy on each stock alone: does the edge show up broadly, or on one lucky ticker?
    options = trading_options() | {"slippage": slippage, "allowShort": payload.allowShort, "riskFree": risk_free}
    rows = []
    for symbol, data in loaded.items():
        try:
            single = backtest_on(symbol, data, strategy_id, params, rules, options, None)
        except HTTPException:
            continue
        m = single["metrics"]
        rows.append(
            {
                "symbol": symbol,
                "sharpe": m["sharpe"],
                "totalReturn": m["totalReturn"],
                "buyHoldReturn": m["buyHoldReturn"],
                "excessReturn": m["excessReturn"],
                "buyHoldSharpe": single["buyHold"]["sharpe"],
                "maxDrawdown": m["maxDrawdown"],
                "trades": m["trades"],
            }
        )
    rows.sort(key=lambda r: r["sharpe"], reverse=True)
    return result | {
        "missing": missing,
        "crossSection": {"rows": rows, "summary": basket.cross_section(rows)},
        "riskFreeRate": risk_free,
    }


@router.post("/analytics/basket", dependencies=[Depends(get_current_user)])
async def basket_backtest(payload: BasketPayload) -> dict[str, Any]:
    return await asyncio.to_thread(run_basket, payload)


@router.post("/strategies/pine", dependencies=[Depends(get_current_user)])
async def export_pine(payload: PineExportPayload) -> dict[str, str]:
    cost_pct = (settings.trading_fee_bps + settings.slippage_bps) / 100
    return {"script": rule_engine.to_pine(payload.rules.model_dump(), payload.name, cost_pct)}


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
    data = await asyncio.to_thread(load, payload.symbol)
    closes = data["closes"]
    if len(closes) < strategies.warmup(strategy_id, params, rules) + 2:
        raise HTTPException(status_code=422, detail="Not enough data for prediction")
    market = None
    if strategy_id == "ml-logistic":
        market = market_series(data["timestamps"], benchmark_history(costs.benchmark_for(payload.symbol)[0]))
    outlook = strategies.current_signal(
        strategy_id,
        closes,
        params,
        rules,
        allow_short=payload.allowShort,
        bars=data["bars"],
        market_closes=market,
    )
    return {
        "symbol": payload.symbol.upper(),
        "strategyId": strategy_id,
        **outlook,
        "metadata": {"recent": closes[-5:], "parameters": params},
        "generatedAt": now().isoformat(),
    }
