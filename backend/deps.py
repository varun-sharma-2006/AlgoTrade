"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status

from backend import stores
from backend.config import settings
from backend.stores import Store


async def get_db(request: Request) -> Store:
    store = getattr(request.app.state, "store", None)
    # Serverless hosts (Vercel) and mounted sub-apps may never run the lifespan hook, and may serve
    # requests on a new event loop, which a Mongo client can't be reused across. Create it as needed.
    stale_loop = isinstance(store, stores.MongoStore) and store.loop not in (None, asyncio.get_running_loop())
    if store is None or stale_loop:
        store = request.app.state.store = stores.create_store()
    return store


async def get_current_user(authorization: str = Header(""), store: Store = Depends(get_db)) -> dict[str, Any]:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    user = await store.resolve_token(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session")
    return user | {"token": token}


def is_admin(user: dict[str, Any]) -> bool:
    return user.get("email", "").lower() in settings.admin_emails


async def require_admin(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if not is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admins only")
    return user


_heavy_hits: dict[str, list[float]] = {}


async def heavy_limit(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    """Rate limit for CPU-heavy research endpoints: HEAVY_REQUESTS_PER_MINUTE per user (per server instance)."""
    stamp = time.time()
    recent = [t for t in _heavy_hits.get(user["id"], []) if stamp - t < 60]
    if len(recent) >= settings.heavy_requests_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many research runs in the last minute; please wait a moment and try again",
        )
    recent.append(stamp)
    _heavy_hits[user["id"]] = recent
    return user
