"""FastAPI application entry point: `uvicorn backend.main:app`."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import logger, settings
from backend.deps import get_current_user, get_db
from backend.routes import analytics, auth, chat, market, simulations
from backend.stores import InMemoryStore, MongoStore, Store, now

__all__ = ["app", "create_store", "get_current_user", "get_db", "MongoStore", "InMemoryStore", "now"]


def create_store() -> Store:
    if settings.use_in_memory_db:
        logger.info("Using in-memory store")
        return InMemoryStore()
    logger.info("Connecting to MongoDB at %s", settings.mongo_uri)
    return MongoStore(settings.mongo_uri, settings.mongo_db_name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.store = create_store()
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
