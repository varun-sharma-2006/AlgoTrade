from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend import costs, portfolio, strategies
from backend.config import logger, settings
from backend.deps import get_current_user, get_db
from backend.market import fetch_chart, fx_rates
from backend.schemas import CustomStrategyInput
from backend.stores import Store

router = APIRouter(tags=["portfolio"])


def _chart(symbol: str) -> dict[str, Any] | None:
    try:
        return fetch_chart(symbol, range_value=settings.history_period, interval="1d")
    except HTTPException:
        logger.warning("No price history for %s", symbol)
        return None


def _fx(currency: str | None) -> dict[str, float] | None:
    try:
        return fx_rates(currency, settings.base_currency, settings.history_period)
    except HTTPException:
        logger.warning("No exchange rate for %s -> %s", currency, settings.base_currency)
        return None


def _position(
    sim: dict[str, Any], chart: dict[str, Any] | None, rates: dict[str, float] | None = None
) -> dict[str, Any]:
    strategy_id = sim.get("strategyId") or "buy-hold"  # simulations created before strategies were tracked
    currency = (chart or {}).get("currency") or settings.base_currency
    base = {
        "id": sim["id"],
        "symbol": sim["symbol"],
        "strategy": sim.get("strategy") or "Buy & hold",
        "strategyId": strategy_id,
        "status": sim.get("status", "active"),
        "startingCapital": float(sim["startingCapital"]),
        "currency": currency,
        "baseCurrency": settings.base_currency,
        "brokerMirror": bool(sim.get("brokerMirror")),
    }
    if not chart or not chart.get("points"):
        return base | {"value": base["startingCapital"], "pnl": 0.0, "pnlPct": 0.0, "error": "No price data"}
    points = chart["points"]
    closes = [p["close"] for p in points]
    timestamps = [p["timestamp"] for p in points]
    bars = {key: [p.get(key) for p in points] for key in ("open", "high", "low", "volume")}
    params = strategies.DEFAULT_PARAMS.get(strategy_id, {}) | (sim.get("parameters") or {})
    cost = costs.cost_model(
        sim["symbol"], settings.trading_fee_bps, settings.slippage_bps, settings.india_brokerage_bps
    )
    fx = portfolio.align_fx([ts[:10] for ts in timestamps], rates)
    valued = portfolio.value_simulation(
        closes,
        timestamps,
        strategy_id=strategy_id,
        params=params,
        rules=sim.get("rules"),
        start_date=sim.get("startDate") or str(sim["createdAt"])[:10],
        capital=base["startingCapital"],
        fee_bps=cost["buyFeeBps"],
        sell_fee_bps=cost["sellFeeBps"],
        slippage_bps=settings.slippage_bps,
        bars=bars,
        execution=settings.execution,
        fx=fx,
    )
    converted = currency.upper() != settings.base_currency or currency in {"GBp", "ZAc", "ILA"}
    return base | valued | {"fxConverted": bool(fx), "fxMissing": converted and not fx}


@router.get("/portfolio")
async def get_portfolio(
    user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    """Every simulation valued at today's prices, plus portfolio totals and history."""
    sims = await store.list_simulations(user["id"])
    symbols = sorted({sim["symbol"] for sim in sims})
    with ThreadPoolExecutor(max_workers=min(8, max(len(symbols), 1))) as pool:
        charts = dict(zip(symbols, pool.map(_chart, symbols), strict=True))
        currencies = sorted({(chart or {}).get("currency") or settings.base_currency for chart in charts.values()})
        rates = dict(zip(currencies, pool.map(_fx, currencies), strict=True))
    positions = [
        _position(sim, charts.get(sim["symbol"]), rates.get((charts.get(sim["symbol"]) or {}).get("currency")))
        for sim in sims
    ]
    positions.sort(key=lambda p: p["value"], reverse=True)
    combined = portfolio.combine(positions)
    combined["summary"]["baseCurrency"] = settings.base_currency
    combined["summary"]["execution"] = settings.execution
    return combined | {"positions": positions}


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
