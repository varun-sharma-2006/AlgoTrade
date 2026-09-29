"""FastAPI application entry point: `uvicorn backend.main:app`."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend import stores
from backend.config import settings
from backend.deps import get_current_user, get_db
from backend.routes import analytics, auth, chat, market, simulations
from backend.stores import InMemoryStore, MongoStore, now

__all__ = ["app", "get_current_user", "get_db", "MongoStore", "InMemoryStore", "now"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.store = stores.create_store()
    try:
        yield
    finally:
        await app.state.store.close()


app = FastAPI(title="Algo Trade Simulator API", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list({settings.frontend_origin, "http://localhost:5173"}),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
for module in (auth, market, simulations, analytics, chat):
    app.include_router(module.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, Any]:
    return {"status": "ok", "timestamp": now().isoformat()}


def mount_frontend(target: FastAPI, static_dir: str) -> None:
    """Serve the built single-page app; unknown paths fall back to index.html. Registered after the API routes."""
    root = Path(static_dir).resolve()
    index = root / "index.html"

    @target.get("/{full_path:path}", include_in_schema=False)
    async def frontend(full_path: str) -> FileResponse:
        candidate = (root / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        return FileResponse(index)


if settings.static_dir and Path(settings.static_dir, "index.html").is_file():
    mount_frontend(app, settings.static_dir)
