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
