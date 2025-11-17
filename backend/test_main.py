import pytest
import pytest_asyncio
from backend.main import MongoStore, now
import secrets
from datetime import timedelta
import asyncio
from unittest.mock import AsyncMock
from bson import ObjectId


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
def store():
    store = MongoStore("mongodb://localhost:27017", "algo-trade-simulator-test")
    store.db = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_resolve_token_mocked(store: MongoStore):
    user_id = ObjectId()
    user_email = "test@example.com"
    user_name = "Test User"
    token = secrets.token_urlsafe(32)
    expires_at = now() + timedelta(days=1)
    expired_expires_at = now() - timedelta(days=1)

    # 1. Test with a valid token
    store.db.sessions.find_one.return_value = {
        "_id": token,
        "user_id": user_id,
        "expires_at": expires_at,
    }
    store.db.users.find_one.return_value = {
        "_id": user_id,
        "email": user_email,
        "name": user_name,
    }
    user = await store.resolve_token(token)
    assert user is not None
    assert user["id"] == str(user_id)
    assert user["email"] == user_email
    assert user["name"] == user_name

    # 2. Test with an invalid token
    store.db.sessions.find_one.return_value = None
    user = await store.resolve_token("invalid_token")
    assert user is None

    # 3. Test with an expired token
    store.db.sessions.find_one.return_value = {
        "_id": token,
        "user_id": user_id,
        "expires_at": expired_expires_at,
    }
    user = await store.resolve_token(token)
    assert user is None


@pytest.mark.asyncio
async def test_get_overview_endpoint(store: MongoStore):
    user_id = "some_user_id"
    store.list_simulations = AsyncMock(
        return_value=[
            {
                "id": "sim1",
                "userId": user_id,
                "startingCapital": 10000,
                "status": "active",
                "createdAt": "2023-01-01T12:00:00Z",
            }
        ]
    )
    store.list_trained = AsyncMock(
        return_value=[
            {
                "id": "train1",
                "userId": user_id,
                "symbol": "AAPL",
                "strategy_id": "sma-crossover",
                "payload": {},
            }
        ]
    )

    from fastapi.testclient import TestClient
    from backend.main import app, get_current_user, get_db

    async def override_get_current_user():
        return {"id": user_id}

    async def override_get_db():
        return store

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)
    response = client.get("/analytics/overview")

    assert response.status_code == 200
    data = response.json()
    assert data["totals"]["totalSimulations"] == 1
    assert data["totals"]["trainedModels"] == 1
