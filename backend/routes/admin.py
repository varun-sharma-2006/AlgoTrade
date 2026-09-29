from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from backend.deps import get_db, require_admin
from backend.stores import Store

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/visitors")
async def visitors(store: Store = Depends(get_db)) -> dict[str, Any]:
    """Everyone who has signed in, and the most recent sign-ins. Only for ADMIN_EMAILS."""
    users = await store.list_users()
    return {"users": users, "logins": await store.list_logins(100), "totalUsers": len(users)}
