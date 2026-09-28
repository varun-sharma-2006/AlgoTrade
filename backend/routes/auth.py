from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from backend.config import settings
from backend.deps import get_db
from backend.schemas import DevAuthBypassRequest, LoginRequest, SignupRequest
from backend.stores import Store

router = APIRouter(tags=["auth"])


async def _session_response(store: Store, user: dict[str, Any]) -> dict[str, Any]:
    session = await store.create_session(user["id"])
    return {"token": session["token"], "user": user}


@router.post("/auth/signup")
async def signup(payload: SignupRequest, store: Store = Depends(get_db)) -> dict[str, Any]:
    try:
        user = await store.create_user(payload.email, payload.name, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return await _session_response(store, user)


@router.post("/auth/login")
async def login(payload: LoginRequest, store: Store = Depends(get_db)) -> dict[str, Any]:
    user = await store.get_user_by_credentials(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return await _session_response(store, user)


@router.post("/dev/auth/bypass")
async def dev_auth_bypass(
    payload: DevAuthBypassRequest | None = None, store: Store = Depends(get_db)
) -> dict[str, Any]:
    if not settings.enable_dev_endpoints:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dev endpoints are disabled")
    email = (payload.email if payload and payload.email else "dev@example.com").lower()
    name = payload.name if payload and payload.name else "Dev User"
    return await _session_response(store, await store.ensure_user(email, name))
