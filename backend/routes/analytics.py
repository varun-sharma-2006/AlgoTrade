from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend import basket, costs, earnings, factors, ml, news, risk, robustness, strategies, tax, walkforward
from backend import rules as rule_engine
from backend.config import logger, settings
from backend.deps import get_current_user, get_db, heavy_limit
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
HOURLY_RANGE = "730d"  # Yahoo keeps about two years of hourly bars
HOURLY_BACKTEST_DAYS = 365  # hourly backtests report on the last year; the first year warms up and trains
MAX_SAMPLE_POINTS = 2500  # thin long hourly series before sending them to the browser


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
            "interval": "1d",
            "capital": None,
            "impact": False,
            "avoidEarnings": False,
            "news": False,
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
        "interval": payload.interval,
        "capital": payload.capital,
        "impact": payload.marketImpact,
        "avoidEarnings": payload.avoidEarnings,
        "news": payload.newsFeatures,
    }


def default_capital(currency: str) -> float:
    """Starting capital used for after-tax returns and market impact when none is given."""
    return 1_000_000.0 if currency.upper() == "INR" else 100_000.0


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


def load(symbol: str, interval: str = "1d") -> dict[str, Any]:
    """Closes, timestamps, OHLCV bars and currency: daily over HISTORY_PERIOD (backtest window plus warm-up),
    or hourly over the last two years."""
    range_value = HOURLY_RANGE if interval == "1h" else settings.history_period
    chart = fetch_chart(symbol, range_value=range_value, interval=interval)
    points = chart["points"]
    bars = {key: [p.get(key) for p in points] for key in ("open", "high", "low", "volume")}
    return {
        "closes": [p["close"] for p in points],
        "timestamps": [p["timestamp"] for p in points],
        "bars": bars,
        "currency": chart.get("currency") or "USD",
        "interval": interval,
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
    start = evaluation_start(timestamps, data.get("interval", "1d"))
    cost = costs.cost_model(symbol, settings.trading_fee_bps, options["slippage"], settings.india_brokerage_bps)
    market = market_series(timestamps, bench) if strategy_id == "ml-logistic" else None
    currency = data.get("currency") or "USD"
    capital = options.get("capital") or default_capital(currency)
    periods = strategies.bars_per_year(timestamps, symbol)
    blackout = reaction = None
    earnings_info: dict[str, Any] = {"available": False}
    if costs.market(symbol) == "US" and data.get("interval", "1d") == "1d":
        dates = earnings.earnings_dates(symbol)
        if dates:
            blackout, reaction = earnings.windows(timestamps, dates, options.get("execution") == "next_open")
            earnings_info = {"available": True, "recent": [d for d in dates if d >= timestamps[start][:10]][-12:]}
        elif not earnings.enabled():
            earnings_info["reason"] = "Earnings dates need SEC_CONTACT_EMAIL on the server"
    if options.get("avoidEarnings") and not blackout:
        earnings_info["reason"] = earnings_info.get("reason") or "No earnings dates for this symbol (US stocks only)"
    news_info: dict[str, Any] | None = None
    if strategy_id == "ml-logistic" and options.get("news") and bars is not None:
        series, detail = news.tone_feature(symbol, timestamps)
        news_info = {"used": series is not None, "detail": detail}
        if series is not None:
            bars = {**bars, "news": series}
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
        periods=periods,
        impact={"capital": capital} if options.get("impact") else None,
        blackout=blackout if options.get("avoidEarnings") else None,
        reaction_bars=reaction,
    )
    report["earnings"] = earnings_info
    if news_info is not None:
        report["news"] = news_info
    if strategy_id == "ml-logistic":
        report["model"] = ml.report(closes, params, start, cost_bps=cost["buyBps"], bars=bars, market=market)
    region, crypto = costs.tax_region(symbol, settings.base_currency)
    report["tax"] = tax.tax_report(
        report["_trades"],
        capital,
        report["_finalEquity"],
        region=region,
        crypto=crypto,
        currency=currency,
        us_short_rate=settings.us_short_term_tax,
        us_long_rate=settings.us_long_term_tax,
    )
    return {
        "symbol": symbol.upper(),
        "strategyId": strategy_id,
        "parameters": params,
        "rules": rules,
        "currency": currency,
        "interval": data.get("interval", "1d"),
        "costs": cost,
        **report,
    }


def evaluation_start(timestamps: list[str], interval: str = "1d") -> int:
    days = HOURLY_BACKTEST_DAYS if interval == "1h" else strategies.period_days(settings.backtest_period)
    return strategies.evaluation_start(timestamps, days)


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
    data = load(symbol, options.get("interval", "1d"))
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
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if strategy_id not in walkforward.GRIDS and strategy_id != "custom":
        raise HTTPException(status_code=422, detail="Walk-forward testing needs a strategy with parameters to tune")
    if strategy_id == "custom" and not (rules and rules.get("entry")):
        raise HTTPException(status_code=422, detail="Send the custom strategy's rules to walk-forward test them")
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
            rules=rules,
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
    """The report without internal fields, with long (hourly) daily series thinned for the browser."""
    out = {key: value for key, value in report.items() if key != "dailyReturns" and not key.startswith("_")}
    sample = out.get("sample") or []
    if len(sample) > MAX_SAMPLE_POINTS:
        step = -(-len(sample) // MAX_SAMPLE_POINTS)
        out["sample"] = sample[::step] + ([sample[-1]] if (len(sample) - 1) % step else [])
    return out


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


@router.post("/analytics/walk-forward", dependencies=[Depends(heavy_limit)])
async def walk_forward(payload: WalkForwardPayload) -> dict[str, Any]:
    return await asyncio.to_thread(
        run_walk_forward,
        payload.symbol,
        payload.strategyId,
        payload.slippageBps,
        execution=payload.execution,
        allow_short=payload.allowShort,
        rules=payload.rules.model_dump() if payload.rules else None,
    )


def run_robustness(payload: RobustnessPayload, report: dict[str, Any] | None = None) -> dict[str, Any]:
    params = strategy_params(payload)
    rules = payload.rules.model_dump() if payload.rules else None
    options = trading_options(payload)
    data = load(payload.symbol, payload.interval)
    bench = benchmark_history(costs.benchmark_for(payload.symbol)[0])
    if report is None:
        report = backtest_on(payload.symbol, data, payload.strategyId, params, rules, options, bench)
    periods = report["metrics"].get("periodsPerYear") or risk.TRADING_DAYS
    daily = report["dailyReturns"]
    sample = report["sample"]
    hold = [sample[k]["buyHold"] / sample[k - 1]["buyHold"] - 1 for k in range(1, len(sample))]
    monte = robustness.monte_carlo(daily, hold, paths=payload.paths, periods=periods)
    if monte:
        for point in monte["fan"]:
            point["timestamp"] = sample[min(point["step"], len(sample) - 1)]["timestamp"]
    start = evaluation_start(data["timestamps"], payload.interval)
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
            periods=periods,
        )
    rf = options["riskFree"]
    trial_sharpes = [robustness.daily_sharpe(r, rf, periods) for r in trials] or [
        robustness.daily_sharpe(daily, rf, periods)
    ]
    random_timing = None
    if report["metrics"].get("shortTrades", 0) == 0 and report.get("_trades"):
        random_timing = robustness.random_entries(
            data["closes"],
            start,
            [t["bars"] for t in report["_trades"]],
            report["metrics"]["totalReturn"],
            report["costs"]["buyBps"],
        )
    us_stock = costs.market(payload.symbol) == "US" and payload.interval == "1d"
    return {
        "symbol": payload.symbol.upper(),
        "strategyId": payload.strategyId,
        "parameters": params,
        "metrics": {
            key: report["metrics"][key] for key in ("totalReturn", "sharpe", "maxDrawdown", "buyHoldReturn", "trades")
        },
        "monteCarlo": monte,
        "sensitivity": grid,
        "deflatedSharpe": robustness.deflated_sharpe(daily, trial_sharpes, rf, periods),
        "pbo": robustness.pbo(trials, periods=periods) if trials else None,
        "randomEntries": random_timing,
        "sixtyForty": sixty_forty(report["period"]["start"], rf) if us_stock else None,
        "factors": factors.attribution(data["timestamps"][start:], daily) if us_stock else None,
        "period": report["period"],
    }


def sixty_forty(start_day: str, risk_free: float) -> dict[str, Any] | None:
    """60% S&P 500 ETF (SPY) / 40% US bond ETF (AGG), rebalanced monthly, over the same window."""
    try:
        spy, agg = history("SPY"), history("AGG")
    except HTTPException:
        return None
    return robustness.sixty_forty((spy[1], spy[0]), (agg[1], agg[0]), start_day, risk_free)


@router.post("/analytics/robustness", dependencies=[Depends(heavy_limit)])
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


@router.post("/analytics/basket", dependencies=[Depends(heavy_limit)])
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
