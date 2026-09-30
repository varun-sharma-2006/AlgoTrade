from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend import strategies
from backend.deps import get_current_user, get_db
from backend.schemas import SimulationInput, SimulationUpdate
from backend.stores import Store, now

router = APIRouter(tags=["simulations"])


@router.get("/simulations")
async def list_simulations(
    user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> list[dict[str, Any]]:
    return await store.list_simulations(user["id"])


MAX_BACKDATE_DAYS = 365  # portfolio valuation fetches HISTORY_PERIOD (5 years), leaving warm-up for indicators


def prepare_simulation(payload: SimulationInput) -> SimulationInput:
    """Validate a new simulation and fill in the strategy's default parameters."""
    params = strategies.DEFAULT_PARAMS.get(payload.strategyId, {}) | payload.parameters
    rules = payload.rules.model_dump() if payload.rules else None
    problem = strategies.validate(payload.strategyId, params, rules)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    today = now().date()
    if payload.startDate and payload.startDate > today:
        raise HTTPException(status_code=422, detail="The start date can't be in the future")
    if payload.startDate and payload.startDate < today - timedelta(days=MAX_BACKDATE_DAYS):
        raise HTTPException(status_code=422, detail="The start date can be at most one year ago")
    return payload.model_copy(update={"parameters": params})


@router.post("/simulations")
async def create_simulation(
    payload: SimulationInput, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    return await store.add_simulation(user["id"], prepare_simulation(payload))


@router.patch("/simulations/{sim_id}")
async def patch_simulation(
    sim_id: str,
    payload: SimulationUpdate,
    user: dict[str, Any] = Depends(get_current_user),
    store: Store = Depends(get_db),
) -> dict[str, Any]:
    try:
        return await store.update_simulation(user["id"], sim_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Simulation not found") from exc


@router.delete("/simulations/{sim_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def remove_simulation(
    sim_id: str, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> Response:
    try:
        await store.delete_simulation(user["id"], sim_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Simulation not found") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
