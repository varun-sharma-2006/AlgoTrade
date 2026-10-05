"""Research tools: optimiser with overfitting guardrails, AI review, leaderboard, SIP planner, options lab,
plain-English strategies and notebook export."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend import costs, nlrules, notebook, options, review, robustness, sip, strategies, survivorship, walkforward
from backend import rules as rule_engine
from backend.config import settings
from backend.deps import get_current_user, heavy_limit
from backend.routes.analytics import (
    _load_many,
    backtest_on,
    benchmark_history,
    evaluation_start,
    load,
    market_series,
    run_robustness,
    run_walk_forward,
    strategy_params,
    trading_options,
)
from backend.schemas import (
    LeaderboardPayload,
    OptionsPayload,
    RobustnessPayload,
    SipPayload,
    SurvivorshipPayload,
    TextRulesPayload,
    TrainingPayload,
)

router = APIRouter(tags=["research"])

STRATEGY_LABELS = {s["id"]: s["name"] for s in strategies.STRATEGIES} | {"custom": "Custom rules"}


def _rules(payload: TrainingPayload) -> dict[str, Any] | None:
    return payload.rules.model_dump() if payload.rules else None


# ---------- Optimiser with guardrails ----------


def run_optimize(payload: TrainingPayload) -> dict[str, Any]:
    params = strategy_params(payload)
    opts = trading_options(payload)
    data = load(payload.symbol, payload.interval)
    bench = benchmark_history(costs.benchmark_for(payload.symbol)[0])
    current = backtest_on(payload.symbol, data, payload.strategyId, params, _rules(payload), opts, bench)
    periods = current["metrics"]["periodsPerYear"]
    opens = data["bars"].get("open")
    grid, trials = robustness.sensitivity(
        data["closes"],
        evaluation_start(data["timestamps"], payload.interval),
        payload.strategyId,
        params,
        current["costs"]["buyBps"],
        current["costs"]["sellBps"],
        opens=None if opens is None or None in opens else opens,
        execution=opts["execution"],
        allow_short=opts["allowShort"],
        risk_free=opts["riskFree"],
        bars=data["bars"],
        market_closes=market_series(data["timestamps"], bench) if payload.strategyId == "ml-logistic" else None,
        periods=periods,
    )
    if not grid or not trials:
        raise HTTPException(status_code=422, detail="This strategy has no settings to optimise")
    valid = [cell for cell in grid["cells"] if cell["valid"]]
    best_index = max(range(len(valid)), key=lambda i: valid[i]["sharpe"])
    best = valid[best_index]
    best_params = params | {grid["xParam"]: best["x"]} | ({grid["yParam"]: best["y"]} if grid["yParam"] else {})
    rf = opts["riskFree"]
    sharpes = [robustness.daily_sharpe(r, rf, periods) for r in trials]
    dsr = robustness.deflated_sharpe(trials[best_index], sharpes, rf, periods)
    overfit = robustness.pbo(trials, periods=periods)
    wf = None
    if payload.strategyId in walkforward.GRIDS and payload.interval == "1d":
        try:
            result = run_walk_forward(
                payload.symbol,
                payload.strategyId,
                opts["slippage"],
                execution=opts["execution"],
                allow_short=opts["allowShort"],
            )
            wf = {
                key: result["metrics"][key]
                for key in ("inSampleAnnualized", "outOfSampleAnnualized", "buyHoldAnnualized")
            }
        except HTTPException:
            wf = None
    warnings = []
    if dsr and dsr["deflatedSharpe"] < 0.95:
        warnings.append(
            f"After trying {len(trials)} settings, there is only {dsr['deflatedSharpe'] * 100:.0f}% confidence the best "
            "one's Sharpe ratio isn't luck (95% is the usual bar)."
        )
    if overfit and overfit["pbo"] > 0.5:
        warnings.append(
            f"The in-sample winner usually did worse than average out of sample (probability of overfitting "
            f"{overfit['pbo'] * 100:.0f}%)."
        )
    if wf and wf["inSampleAnnualized"] - wf["outOfSampleAnnualized"] > 0.05:
        warnings.append(
            f"In walk-forward testing, tuned settings made {wf['inSampleAnnualized'] * 100:.1f}% a year in sample but "
            f"{wf['outOfSampleAnnualized'] * 100:.1f}% on unseen data."
        )
    return {
        "symbol": payload.symbol.upper(),
        "strategyId": payload.strategyId,
        "current": {
            "parameters": params,
            "sharpe": current["metrics"]["sharpe"],
            "totalReturn": current["metrics"]["totalReturn"],
        },
        "best": {
            "parameters": best_params,
            "sharpe": best["sharpe"],
            "totalReturn": best["totalReturn"],
            "maxDrawdown": best["maxDrawdown"],
            "trades": best["trades"],
        },
        "trials": len(trials),
        "deflatedSharpe": dsr,
        "pbo": overfit,
        "walkForward": wf,
        "warnings": warnings,
        "trustworthy": not warnings,
    }


@router.post("/analytics/optimize", dependencies=[Depends(heavy_limit)])
async def optimize(payload: TrainingPayload) -> dict[str, Any]:
    return await asyncio.to_thread(run_optimize, payload)


# ---------- AI backtest review ----------


def run_review(payload: RobustnessPayload, with_summary: bool = True) -> dict[str, Any]:
    params = strategy_params(payload)
    opts = trading_options(payload)
    data = load(payload.symbol, payload.interval)
    bench = benchmark_history(costs.benchmark_for(payload.symbol)[0])
    report = backtest_on(payload.symbol, data, payload.strategyId, params, _rules(payload), opts, bench)
    close_report = None
    if opts["execution"] == "next_open":
        close_report = backtest_on(
            payload.symbol, data, payload.strategyId, params, _rules(payload), opts | {"execution": "close"}, bench
        )
    robust = run_robustness(payload, report)
    wf = None
    if payload.interval == "1d" and (payload.strategyId in walkforward.GRIDS or payload.strategyId == "custom"):
        try:
            wf = run_walk_forward(
                payload.symbol,
                payload.strategyId,
                opts["slippage"],
                execution=opts["execution"],
                allow_short=opts["allowShort"],
                rules=_rules(payload),
            )
        except HTTPException:
            wf = None
    items = review.findings(report, close_report=close_report, robustness=robust, walk_forward=wf)
    overall = review.verdict(items)
    label = STRATEGY_LABELS.get(payload.strategyId, payload.strategyId)
    return {
        "symbol": payload.symbol.upper(),
        "strategyId": payload.strategyId,
        "findings": items,
        "verdict": overall,
        "summary": review.summarise(payload.symbol.upper(), label, items, overall) if with_summary else None,
    }


@router.post("/analytics/review", dependencies=[Depends(heavy_limit)])
async def backtest_review(payload: RobustnessPayload) -> dict[str, Any]:
    return await asyncio.to_thread(run_review, payload)


# ---------- Strategy leaderboard ----------


def run_leaderboard(payload: LeaderboardPayload) -> dict[str, Any]:
    loaded, missing = _load_many(payload.symbols)
    if not loaded:
        raise HTTPException(status_code=502, detail="No price data for these symbols")
    entries: list[tuple[str, str, dict[str, float], dict[str, Any] | None]] = [
        (sid, STRATEGY_LABELS[sid], dict(params), None)
        for sid, params in strategies.DEFAULT_PARAMS.items()
        if sid != "custom"
    ] + [("custom", entry.name, {}, entry.rules.model_dump()) for entry in payload.custom]
    opts = trading_options()
    rows = []
    for strategy_id, name, params, rules in entries:
        results = []
        for symbol, data in loaded.items():
            try:
                m = backtest_on(symbol, data, strategy_id, params, rules, opts, None)
            except HTTPException:
                continue
            results.append(
                {
                    "symbol": symbol,
                    "sharpe": m["metrics"]["sharpe"],
                    "totalReturn": m["metrics"]["totalReturn"],
                    "excessReturn": m["metrics"]["excessReturn"],
                    "maxDrawdown": m["metrics"]["maxDrawdown"],
                    "beatSharpe": m["metrics"]["sharpe"] > m["buyHold"]["sharpe"],
                }
            )
        if not results:
            continue
        sharpes = sorted(r["sharpe"] for r in results)
        drawdowns = sorted(r["maxDrawdown"] for r in results)
        beat = sum(1 for r in results if r["beatSharpe"]) / len(results)
        median_sharpe = sharpes[len(sharpes) // 2]
        rows.append(
            {
                "strategyId": strategy_id,
                "name": name,
                "symbols": len(results),
                "medianSharpe": median_sharpe,
                "worstSharpe": sharpes[0],
                "medianReturn": sorted(r["totalReturn"] for r in results)[len(results) // 2],
                "medianMaxDrawdown": drawdowns[len(drawdowns) // 2],
                "shareBeatBuyHold": sum(1 for r in results if r["excessReturn"] > 0) / len(results),
                "shareBetterSharpe": beat,
                # Robustness score: typical risk-adjusted return, weighted by how consistently it beats holding.
                "score": median_sharpe * (0.5 + 0.5 * beat),
                "results": results,
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"symbols": list(loaded), "missing": missing, "rows": rows}


@router.post("/analytics/leaderboard", dependencies=[Depends(heavy_limit)])
async def leaderboard(payload: LeaderboardPayload) -> dict[str, Any]:
    return await asyncio.to_thread(run_leaderboard, payload)


# ---------- SIP planner ----------


def run_sip(payload: SipPayload) -> dict[str, Any]:
    data = load(payload.symbol)
    closes, timestamps = data["closes"], data["timestamps"]
    start = strategies.evaluation_start(timestamps, payload.years * 365)
    cost = costs.cost_model(
        payload.symbol, settings.trading_fee_bps, settings.slippage_bps, settings.india_brokerage_bps
    )
    timing = None
    if payload.timingStrategy:
        strategy_id = payload.timingStrategy
        params = dict(strategies.DEFAULT_PARAMS.get(strategy_id, {}))
        rules = payload.timingRules.model_dump() if payload.timingRules else None
        problem = strategies.validate(strategy_id, params, rules)
        if problem:
            raise HTTPException(status_code=422, detail=problem)
        timing = strategies.signals(strategy_id, closes, params, rules, bars=data["bars"])[0]
    region, crypto = costs.tax_region(payload.symbol, settings.base_currency)
    try:
        result = sip.run(
            closes,
            timestamps,
            monthly=payload.monthly,
            start=start,
            step_up=payload.stepUp,
            fee_bps=cost["buyBps"],
            sell_fee_bps=cost["sellBps"],
            timing=timing,
            region=region,
            crypto=crypto,
            currency=data["currency"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    label = STRATEGY_LABELS.get(payload.timingStrategy or "", payload.timingStrategy)
    return {"symbol": payload.symbol.upper(), "timingStrategy": label, "region": region, **result}


@router.post("/analytics/sip", dependencies=[Depends(heavy_limit)])
async def sip_plan(payload: SipPayload) -> dict[str, Any]:
    return await asyncio.to_thread(run_sip, payload)


# ---------- Options lab ----------


def run_options(payload: OptionsPayload) -> dict[str, Any]:
    data = load(payload.symbol)
    start = evaluation_start(data["timestamps"])
    try:
        result = options.backtest(
            data["closes"],
            data["timestamps"],
            start,
            strategy=payload.strategy,
            otm=payload.otm,
            days=payload.days,
            vol_premium=payload.volPremium,
            risk_free=settings.risk_free_rate if payload.riskFreeRate is None else payload.riskFreeRate,
            cost_bps=payload.costBps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"symbol": payload.symbol.upper(), "currency": data["currency"], "lastPrice": data["closes"][-1], **result}


@router.post("/analytics/options", dependencies=[Depends(heavy_limit)])
async def options_backtest(payload: OptionsPayload) -> dict[str, Any]:
    return await asyncio.to_thread(run_options, payload)


# ---------- Plain English -> rules, and notebook export ----------


@router.post("/strategies/from-text", dependencies=[Depends(heavy_limit)])
async def rules_from_text(payload: TextRulesPayload) -> dict[str, Any]:
    result = await asyncio.to_thread(nlrules.to_rules, payload.text)
    if not result["rules"]:
        detail = "Couldn't find a buy rule in that description. Try e.g. 'buy when RSI is below 30, sell above 60'."
        raise HTTPException(status_code=422, detail=f"{detail} {result['notes']}".strip())
    return result


@router.post("/strategies/notebook", dependencies=[Depends(get_current_user)])
async def export_notebook(payload: TrainingPayload) -> dict[str, Any]:
    if payload.strategyId == "ml-logistic":
        raise HTTPException(status_code=422, detail="The machine-learning strategy can't be exported to a notebook")
    params = strategy_params(payload)
    rules = _rules(payload)
    problem = strategies.validate(payload.strategyId, params, rules)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    opts = trading_options(payload)
    cost = costs.cost_model(payload.symbol, settings.trading_fee_bps, opts["slippage"], settings.india_brokerage_bps)
    label = STRATEGY_LABELS.get(payload.strategyId, payload.strategyId)
    summary = ", ".join(f"{k} {v:g}" for k, v in params.items()) or (
        rule_engine.describe(rules) if rules else "Buy and hold"
    )
    return notebook.build(
        payload.symbol.upper(),
        payload.strategyId,
        params,
        rules,
        title=f"{label} on {payload.symbol.upper()}",
        execution=opts["execution"],
        buy_cost_bps=cost["buyBps"],
        sell_cost_bps=cost["sellBps"],
        risk_free=opts["riskFree"],
        allow_short=opts["allowShort"],
        backtest_days=strategies.period_days(settings.backtest_period),
        summary=summary,
    )


# ---------- Survivorship bias ----------


def _years_ago(years: int) -> str:
    from datetime import date, timedelta

    return (date.today() - timedelta(days=round(365.25 * years))).isoformat()


@router.post("/analytics/survivorship", dependencies=[Depends(get_current_user)])
async def survivorship_check(payload: SurvivorshipPayload) -> dict[str, Any]:
    """Which of these symbols were S&P 500 members when the backtest starts, and who has left since."""
    symbols = [s.strip().upper() for s in payload.symbols if s.strip()]
    result = await asyncio.to_thread(survivorship.check, symbols, _years_ago(payload.years))
    if result is None:
        raise HTTPException(status_code=502, detail="S&P 500 history is unavailable right now")
    return result


@router.get("/analytics/sp500-sample", dependencies=[Depends(get_current_user)])
async def sp500_sample(size: int = 20, years: int = 2, seed: int | None = None) -> dict[str, Any]:
    """A random sample of the S&P 500 as it was `years` ago, including companies that have since left it."""
    result = await asyncio.to_thread(
        survivorship.sample, _years_ago(max(1, min(years, 10))), max(2, min(size, 30)), seed
    )
    if result is None:
        raise HTTPException(status_code=502, detail="S&P 500 history is unavailable right now")
    return result
