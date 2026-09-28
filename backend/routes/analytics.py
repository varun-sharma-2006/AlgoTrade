from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend import strategies
from backend.config import settings
from backend.deps import get_current_user, get_db
from backend.market import WATCHLIST_SYMBOLS, fetch_chart
from backend.routes.market import parse_symbols
from backend.schemas import PredictionPayload, TrainingPayload
from backend.stores import Store, now, serialize_mongo_doc

router = APIRouter(tags=["analytics"])

PARAMS_BY_STRATEGY = {
    "sma-crossover": ("shortWindow", "longWindow"),
    "mean-reversion": ("lookback", "deviation"),
    "trend-follow": ("channel",),
}


def strategy_params(payload: TrainingPayload) -> dict[str, float]:
    names = PARAMS_BY_STRATEGY.get(payload.strategyId, ())
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


@router.post("/analytics/train")
async def train_strategy(
    payload: TrainingPayload, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    params = strategy_params(payload)
    problem = strategies.validate(payload.strategyId, params)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    chart = fetch_chart(payload.symbol, range_value=settings.backtest_period, interval="1d")
    closes = [point["close"] for point in chart["points"]]
    if len(closes) < strategies.warmup(payload.strategyId, params) + 30:
        raise HTTPException(status_code=422, detail="Not enough price history for these parameters")
    timestamps = [point["timestamp"] for point in chart["points"]]
    report = strategies.backtest(closes, timestamps, payload.strategyId, params, settings.trading_fee_bps)
    result = {
        "symbol": payload.symbol.upper(),
        "strategyId": payload.strategyId,
        "parameters": params,
        # Kept for clients that read the SMA windows directly.
        "shortWindow": payload.shortWindow,
        "longWindow": payload.longWindow,
        **report,
        "trainedAt": now().isoformat(),
    }
    await store.record_training(user["id"], payload.symbol, payload.strategyId, result)
    return result


@router.post("/analytics/predict")
async def predict(
    payload: PredictionPayload, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    training = await store.get_training(user["id"], payload.symbol)
    if not training:
        raise HTTPException(status_code=404, detail="Train the strategy first")
    trained = training.get("payload") or {}
    strategy_id = trained.get("strategyId") or training.get("strategyId") or "sma-crossover"
    params = trained.get("parameters") or {
        "shortWindow": trained.get("shortWindow", 20),
        "longWindow": trained.get("longWindow", 60),
    }
    chart = fetch_chart(payload.symbol, range_value="1y", interval="1d")
    closes = [point["close"] for point in chart["points"]]
    if len(closes) < strategies.warmup(strategy_id, params) + 2:
        raise HTTPException(status_code=422, detail="Not enough data for prediction")
    outlook = strategies.current_signal(strategy_id, closes, params)
    return {
        "symbol": payload.symbol.upper(),
        "strategyId": strategy_id,
        **outlook,
        "metadata": {"recent": closes[-5:], "parameters": params},
        "generatedAt": now().isoformat(),
    }
