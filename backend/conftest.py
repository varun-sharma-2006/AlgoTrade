"""Shared fixtures: the API against both storage backends, with market data stubbed out."""

import pytest
from fastapi.testclient import TestClient
from mongomock_motor import AsyncMongoMockClient

from backend import deps, main, stores
from backend.config import settings
from backend.routes import analytics
from backend.stores import InMemoryStore, MongoStore


def fake_chart(symbol, range_value="1mo", interval="1d"):
    closes = [100 - i * 0.5 for i in range(60)] + [70 + i for i in range(80)]
    points = [
        {"timestamp": f"2026-{i:04d}", "open": c, "high": c, "low": c, "close": c, "volume": 1000}
        for i, c in enumerate(closes)
    ]
    return {"symbol": symbol.upper(), "points": points}


@pytest.fixture(params=["memory", "mongo"])
def client(request, monkeypatch):
    if request.param == "memory":
        store = InMemoryStore()
    else:
        store = MongoStore("mongodb://test", "test-db", client=AsyncMongoMockClient())
    monkeypatch.setattr(stores, "create_store", lambda: store)
    monkeypatch.setattr(analytics, "fetch_chart", fake_chart)
    monkeypatch.setattr(settings, "enable_dev_endpoints", True)
    deps._heavy_hits.clear()  # the research rate limit is per user and would carry over between tests
    with TestClient(main.app) as test_client:
        yield test_client
