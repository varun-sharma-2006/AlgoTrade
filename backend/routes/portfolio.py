from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend import portfolio, strategies
from backend.config import logger, settings
from backend.deps import get_current_user, get_db
from backend.market import fetch_chart
from backend.schemas import CustomStrategyInput
from backend.stores import Store

router = APIRouter(tags=["portfolio"])


def _chart(symbol: str) -> dict[str, Any] | None:
    try:
        return fetch_chart(symbol, range_value=settings.history_period, interval="1d")
    except HTTPException:
        logger.warning("No price history for %s", symbol)
        return None


def _position(sim: dict[str, Any], chart: dict[str, Any] | None) -> dict[str, Any]:
    strategy_id = sim.get("strategyId") or "buy-hold"  # simulations created before strategies were tracked
    base = {
        "id": sim["id"],
        "symbol": sim["symbol"],
        "strategy": sim.get("strategy") or "Buy & hold",
        "strategyId": strategy_id,
        "status": sim.get("status", "active"),
        "startingCapital": float(sim["startingCapital"]),
        "currency": (chart or {}).get("currency") or "USD",
    }
    if not chart or not chart.get("points"):
        return base | {"value": base["startingCapital"], "pnl": 0.0, "pnlPct": 0.0, "error": "No price data"}
    closes = [p["close"] for p in chart["points"]]
    timestamps = [p["timestamp"] for p in chart["points"]]
    params = strategies.DEFAULT_PARAMS.get(strategy_id, {}) | (sim.get("parameters") or {})
    valued = portfolio.value_simulation(
        closes,
        timestamps,
        strategy_id=strategy_id,
        params=params,
        rules=sim.get("rules"),
        start_date=sim.get("startDate") or str(sim["createdAt"])[:10],
        capital=base["startingCapital"],
        fee_bps=settings.trading_fee_bps + settings.slippage_bps,
    )
    return base | valued


@router.get("/portfolio")
async def get_portfolio(
    user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    """Every simulation valued at today's prices, plus portfolio totals and history."""
    sims = await store.list_simulations(user["id"])
    symbols = sorted({sim["symbol"] for sim in sims})
    with ThreadPoolExecutor(max_workers=min(8, max(len(symbols), 1))) as pool:
        charts = dict(zip(symbols, pool.map(_chart, symbols), strict=True))
    positions = [_position(sim, charts.get(sim["symbol"])) for sim in sims]
    positions.sort(key=lambda p: p["value"], reverse=True)
    return portfolio.combine(positions) | {"positions": positions}


@router.get("/strategies/custom")
async def list_custom(user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)) -> list[dict]:
    return await store.list_custom_strategies(user["id"])


@router.post("/strategies/custom")
async def create_custom(
    payload: CustomStrategyInput, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    try:
        return await store.add_custom_strategy(user["id"], payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/strategies/custom/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_custom(
    strategy_id: str, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> Response:
    try:
        await store.delete_custom_strategy(user["id"], strategy_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Strategy not found") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
