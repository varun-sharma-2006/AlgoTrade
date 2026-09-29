"""Signed in-memory sessions must work across server instances (e.g. Vercel's parallel functions)."""

import asyncio
from datetime import timedelta

from fastapi.testclient import TestClient

from backend import main, stores
from backend.config import settings
from backend.routes import analytics
from backend.stores import InMemoryStore, now, sign_token, verify_token
from backend.test_api import fake_chart


def test_token_from_one_instance_is_accepted_by_another():
    async def scenario():
        first, second = InMemoryStore(), InMemoryStore()
        user = await first.ensure_user("user@example.com", "User")
        session = await first.create_session(user["id"])
        resolved = await second.resolve_token(session["token"])
        assert resolved == user
        # The same email maps to the same id everywhere, so data keys line up across instances.
        assert (await second.ensure_user("user@example.com", "User"))["id"] == user["id"]

    asyncio.run(scenario())


def test_tampered_and_expired_tokens_are_rejected():
    claims = {"sub": "abc", "email": "a@example.com", "name": "A", "exp": int((now() + timedelta(hours=1)).timestamp())}
    token = sign_token(claims)
    assert verify_token(token) == claims
    body, signature = token.split(".")
    assert verify_token(f"{body}x.{signature}") is None
    assert verify_token(f"{body}.{signature[:-2]}AA") is None
    assert verify_token("not-a-token") is None
    expired = sign_token(claims | {"exp": int((now() - timedelta(seconds=1)).timestamp())})
    assert verify_token(expired) is None


def test_predict_works_on_an_instance_that_never_saw_the_training(monkeypatch):
    instances = iter([InMemoryStore(), InMemoryStore()])
    monkeypatch.setattr(analytics, "fetch_chart", fake_chart)
    monkeypatch.setattr(settings, "enable_dev_endpoints", True)

    monkeypatch.setattr(stores, "create_store", lambda: next(instances))
    with TestClient(main.app) as client_a:
        token = client_a.post("/dev/auth/bypass").json()["token"]
        body = {"symbol": "AAPL", "strategyId": "trend-follow", "channel": 20}
        assert (
            client_a.post("/analytics/train", headers={"Authorization": f"Bearer {token}"}, json=body).status_code
            == 200
        )

    with TestClient(main.app) as client_b:  # a fresh instance with empty memory
        headers = {"Authorization": f"Bearer {token}"}
        assert client_b.get("/simulations", headers=headers).status_code == 200
        assert client_b.post("/analytics/predict", headers=headers, json={"symbol": "AAPL"}).status_code == 404
        explicit = {"symbol": "AAPL", "strategyId": "trend-follow", "parameters": {"channel": 20}}
        response = client_b.post("/analytics/predict", headers=headers, json=explicit)
        assert response.status_code == 200 and response.json()["signal"] in {"buy", "hold", "sell", "wait"}
        bad = {"symbol": "AAPL", "strategyId": "sma-crossover", "parameters": {"shortWindow": 50}}
        assert client_b.post("/analytics/predict", headers=headers, json=bad).status_code == 422


def test_mongo_store_is_recreated_on_a_new_event_loop(monkeypatch):
    from types import SimpleNamespace

    from mongomock_motor import AsyncMongoMockClient

    from backend.deps import get_db

    made = []

    def fake_create():
        made.append(stores.MongoStore("mongodb://test", "db", client=AsyncMongoMockClient()))
        return made[-1]

    monkeypatch.setattr(stores, "create_store", fake_create)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    async def fetch():
        return await get_db(request)

    first = asyncio.run(fetch())
    second = asyncio.run(fetch())  # asyncio.run always creates a new loop
    assert first is not second and len(made) == 2

    async def twice():
        return await get_db(request), await get_db(request)

    a, b = asyncio.run(twice())
    assert a is b  # same loop: reused
