"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status

from backend import stores
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
