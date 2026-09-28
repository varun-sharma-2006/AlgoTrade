from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend.deps import get_current_user, get_db
from backend.schemas import SimulationInput, SimulationUpdate
from backend.stores import Store

router = APIRouter(tags=["simulations"])


@router.get("/simulations")
async def list_simulations(
    user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> list[dict[str, Any]]:
    return await store.list_simulations(user["id"])


@router.post("/simulations")
async def create_simulation(
    payload: SimulationInput, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    return await store.add_simulation(user["id"], payload)


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
