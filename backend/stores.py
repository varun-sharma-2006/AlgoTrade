"""Persistence: a MongoDB store and an in-memory store with the same async interface.

Routes only talk to this interface, so they don't need to know which backend is active.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from passlib.context import CryptContext

from backend.config import logger, settings
from backend.schemas import SimulationInput, SimulationUpdate

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def now() -> datetime:
    return datetime.now(UTC)


def serialize_mongo_doc(doc: Any) -> Any:
    if isinstance(doc, dict):
        serialized = {key: serialize_mongo_doc(value) for key, value in doc.items()}
        if "_id" in serialized:
            serialized["id"] = serialized.pop("_id")
        return serialized
    if isinstance(doc, list):
        return [serialize_mongo_doc(item) for item in doc]
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, datetime):
        return doc.isoformat()
    return doc


def _public_user(record: dict[str, Any]) -> dict[str, Any]:
    user = {"id": str(record.get("id") or record.get("_id")), "email": record["email"], "name": record["name"]}
    if record.get("picture"):
        user["picture"] = record["picture"]
    return user


def _visitor(record: dict[str, Any]) -> dict[str, Any]:
    """User fields shown on the admin Visitors page (never the password hash)."""
    return _public_user(record) | {
        "createdAt": record.get("createdAt"),
        "lastLoginAt": record.get("lastLoginAt"),
        "loginCount": record.get("loginCount", 0),
    }


def _session_expiry() -> datetime:
    return now() + timedelta(days=settings.session_duration_days)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_token(claims: dict[str, Any]) -> str:
    body = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
    signature = hmac.new(settings.session_secret.encode(), body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64(signature)}"


def verify_token(token: str) -> dict[str, Any] | None:
    """Claims of a valid, unexpired signed token; None for anything else."""
    body, _, signature = token.partition(".")
    expected = hmac.new(settings.session_secret.encode(), body.encode(), hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(_unb64(signature), expected):
            return None
        claims = json.loads(_unb64(body))
    except (ValueError, TypeError):
        return None
    if not isinstance(claims, dict) or claims.get("exp", 0) <= now().timestamp():
        return None
    return claims


def session_allowed(provider: str | None) -> bool:
    """Once Sign in with Google is on, only Google sessions count: sessions from the old shared demo
    login or password accounts (including ones created before providers were recorded) are rejected."""
    return not settings.google_client_id or provider == "google"


def _object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError) as exc:
        raise KeyError("Simulation not found") from exc


class MongoStore:
    def __init__(self, uri: str, database_name: str, client: AsyncIOMotorClient | None = None) -> None:
        self.client = client or AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[database_name]
        # Motor clients are tied to the event loop they were created on (see backend.deps.get_db).
        self.loop = _running_loop()

    async def close(self) -> None:
        self.client.close()

    @property
    def users(self) -> AsyncIOMotorCollection:
        return self.db.users

    @property
    def sessions(self) -> AsyncIOMotorCollection:
        return self.db.sessions

    @property
    def simulations(self) -> AsyncIOMotorCollection:
        return self.db.simulations

    @property
    def trained(self) -> AsyncIOMotorCollection:
        return self.db.trained

    # Users and sessions
    async def create_user(self, email: str, name: str, password: str) -> dict[str, Any]:
        if await self.users.find_one({"email": email.lower()}):
            raise ValueError("Email already registered")
        doc = {
            "_id": ObjectId(),
            "email": email.lower(),
            "name": name,
            "password_hash": pwd_context.hash(password),
            "createdAt": now(),
        }
        await self.users.insert_one(doc)
        return _public_user(doc)

    async def get_user_by_credentials(self, email: str, password: str) -> dict[str, Any] | None:
        doc = await self.users.find_one({"email": email.lower()})
        if not doc or not pwd_context.verify(password, doc["password_hash"]):
            return None
        return _public_user(doc)

    async def ensure_user(self, email: str, name: str) -> dict[str, Any]:
        doc = await self.users.find_one({"email": email.lower()})
        if doc:
            return _public_user(doc)
        return await self.create_user(email, name, secrets.token_urlsafe(12))

    async def create_session(self, user_id: str, provider: str = "password") -> dict[str, Any]:
        token = secrets.token_urlsafe(32)
        expiry = _session_expiry()
        await self.sessions.insert_one(
            {"_id": token, "user_id": ObjectId(user_id), "expires_at": expiry, "provider": provider}
        )
        return {"token": token, "expires_at": expiry}

    async def resolve_token(self, token: str) -> dict[str, Any] | None:
        session = await self.db.sessions.find_one({"_id": token})
        if not session:
            return None
        expires_at = session["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at <= now() or not session_allowed(session.get("provider")):
            return None
        user = await self.db.users.find_one({"_id": session["user_id"]})
        return _public_user(user) if user else None

    # Simulations
    async def list_simulations(self, user_id: str) -> list[dict[str, Any]]:
        cursor = self.simulations.find({"userId": ObjectId(user_id)})
        return serialize_mongo_doc(await cursor.to_list(length=100))

    async def add_simulation(self, user_id: str, payload: SimulationInput) -> dict[str, Any]:
        doc = {
            "userId": ObjectId(user_id),
            "symbol": payload.symbol.upper(),
            "strategy": payload.strategy,
            "startingCapital": float(payload.startingCapital),
            "status": "active",
            "notes": payload.notes,
            "createdAt": now(),
        }
        result = await self.simulations.insert_one(doc)
        return serialize_mongo_doc(doc | {"_id": result.inserted_id})

    async def update_simulation(self, user_id: str, sim_id: str, payload: SimulationUpdate) -> dict[str, Any]:
        update = payload.model_dump(exclude_none=True)
        query = {"_id": _object_id(sim_id), "userId": ObjectId(user_id)}
        if not update:
            doc = await self.simulations.find_one(query)
        else:
            doc = await self.simulations.find_one_and_update(query, {"$set": update}, return_document=True)
        if not doc:
            raise KeyError("Simulation not found")
        return serialize_mongo_doc(doc)

    async def delete_simulation(self, user_id: str, sim_id: str) -> None:
        result = await self.simulations.delete_one({"_id": _object_id(sim_id), "userId": ObjectId(user_id)})
        if result.deleted_count == 0:
            raise KeyError("Simulation not found")

    # Trained strategies
    async def record_training(self, user_id: str, symbol: str, strategy_id: str, payload: dict[str, Any]) -> None:
        key = {"userId": ObjectId(user_id), "symbol": symbol.upper()}
        doc = key | {"strategyId": strategy_id, "payload": payload, "trainedAt": now()}
        await self.trained.update_one(key, {"$set": doc}, upsert=True)

    async def get_training(self, user_id: str, symbol: str) -> dict[str, Any] | None:
        doc = await self.trained.find_one({"userId": ObjectId(user_id), "symbol": symbol.upper()})
        return serialize_mongo_doc(doc) if doc else None

    async def list_trained(self, user_id: str) -> list[dict[str, Any]]:
        cursor = self.trained.find({"userId": ObjectId(user_id)})
        return serialize_mongo_doc(await cursor.to_list(length=100))

    # Sign-in tracking
    async def record_login(
        self, user: dict[str, Any], picture: str | None, provider: str, user_agent: str | None
    ) -> None:
        stamp = now()
        update: dict[str, Any] = {"$set": {"lastLoginAt": stamp, "name": user["name"]}, "$inc": {"loginCount": 1}}
        if picture:
            update["$set"]["picture"] = picture
        await self.users.update_one({"_id": ObjectId(user["id"])}, update)
        await self.db.logins.insert_one(
            {
                "userId": ObjectId(user["id"]),
                "email": user["email"],
                "name": user["name"],
                "picture": picture,
                "provider": provider,
                "userAgent": user_agent,
                "at": stamp,
            }
        )

    async def list_users(self) -> list[dict[str, Any]]:
        cursor = self.users.find({}, {"password_hash": 0}).sort("lastLoginAt", -1)
        return [_visitor(doc) for doc in serialize_mongo_doc(await cursor.to_list(length=1000))]

    async def list_logins(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.db.logins.find({}).sort("at", -1).limit(limit)
        return serialize_mongo_doc(await cursor.to_list(length=limit))


class InMemoryStore:
    """Ephemeral store for local development and demos (USE_IN_MEMORY_DB=true). Data resets on restart.

    Session tokens are signed rather than stored, so they stay valid across processes: serverless hosts
    like Vercel run several instances that don't share memory, and a stored session would only be known
    to the instance that created it.
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users_by_email: dict[str, dict[str, Any]] = {}
        self.users_by_id: dict[str, dict[str, Any]] = {}
        self.simulations: dict[str, dict[str, Any]] = {}
        self.trained: dict[str, dict[str, Any]] = {}
        self.logins: list[dict[str, Any]] = []

    async def close(self) -> None:
        return None

    def _add_user(self, email: str, name: str, password: str) -> dict[str, Any]:
        record = {
            # Deterministic, so the same account has the same id on every instance.
            "id": uuid.uuid5(uuid.NAMESPACE_URL, f"algo-trade:{email.lower()}").hex,
            "email": email.lower(),
            "name": name,
            "password_hash": pwd_context.hash(password),
            "createdAt": now().isoformat(),
        }
        self.users_by_email[record["email"]] = record
        self.users_by_id[record["id"]] = record
        return _public_user(record)

    async def create_user(self, email: str, name: str, password: str) -> dict[str, Any]:
        async with self.lock:
            if email.lower() in self.users_by_email:
                raise ValueError("Email already registered")
            return self._add_user(email, name, password)

    async def get_user_by_credentials(self, email: str, password: str) -> dict[str, Any] | None:
        async with self.lock:
            record = self.users_by_email.get(email.lower())
            if not record or not pwd_context.verify(password, record["password_hash"]):
                return None
            return _public_user(record)

    async def ensure_user(self, email: str, name: str) -> dict[str, Any]:
        async with self.lock:
            record = self.users_by_email.get(email.lower())
            if record:
                return _public_user(record)
            return self._add_user(email, name, secrets.token_urlsafe(12))

    async def create_session(self, user_id: str, provider: str = "password") -> dict[str, Any]:
        async with self.lock:
            user = self.users_by_id[user_id]
        expiry = _session_expiry()
        claims = {
            "sub": user_id,
            "email": user["email"],
            "name": user["name"],
            "exp": int(expiry.timestamp()),
            "prv": provider,
        }
        return {"token": sign_token(claims), "expires_at": expiry}

    async def resolve_token(self, token: str) -> dict[str, Any] | None:
        claims = verify_token(token)
        if not claims or not session_allowed(claims.get("prv")):
            return None
        async with self.lock:
            user = self.users_by_id.get(claims["sub"])
        return _public_user(user) if user else {"id": claims["sub"], "email": claims["email"], "name": claims["name"]}

    async def list_simulations(self, user_id: str) -> list[dict[str, Any]]:
        async with self.lock:
            return [dict(record) for record in self.simulations.values() if record["userId"] == user_id]

    async def add_simulation(self, user_id: str, payload: SimulationInput) -> dict[str, Any]:
        async with self.lock:
            record = {
                "id": uuid.uuid4().hex,
                "userId": user_id,
                "symbol": payload.symbol.upper(),
                "strategy": payload.strategy,
                "startingCapital": float(payload.startingCapital),
                "status": "active",
                "notes": payload.notes,
                "createdAt": now().isoformat(),
            }
            self.simulations[record["id"]] = record
            return dict(record)

    async def update_simulation(self, user_id: str, sim_id: str, payload: SimulationUpdate) -> dict[str, Any]:
        async with self.lock:
            record = self.simulations.get(sim_id)
            if not record or record["userId"] != user_id:
                raise KeyError("Simulation not found")
            record.update(payload.model_dump(exclude_none=True))
            return dict(record)

    async def delete_simulation(self, user_id: str, sim_id: str) -> None:
        async with self.lock:
            record = self.simulations.get(sim_id)
            if not record or record["userId"] != user_id:
                raise KeyError("Simulation not found")
            self.simulations.pop(sim_id, None)

    async def record_training(self, user_id: str, symbol: str, strategy_id: str, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.trained[f"{user_id}:{symbol.upper()}"] = {
                "symbol": symbol.upper(),
                "strategyId": strategy_id,
                "userId": user_id,
                "payload": payload,
                "trainedAt": now().isoformat(),
            }

    async def get_training(self, user_id: str, symbol: str) -> dict[str, Any] | None:
        async with self.lock:
            return self.trained.get(f"{user_id}:{symbol.upper()}")

    async def list_trained(self, user_id: str) -> list[dict[str, Any]]:
        async with self.lock:
            return [item for item in self.trained.values() if item["userId"] == user_id]

    async def record_login(
        self, user: dict[str, Any], picture: str | None, provider: str, user_agent: str | None
    ) -> None:
        stamp = now().isoformat()
        async with self.lock:
            record = self.users_by_id.get(user["id"])
            if record:
                record.update(name=user["name"], lastLoginAt=stamp, loginCount=record.get("loginCount", 0) + 1)
                if picture:
                    record["picture"] = picture
            self.logins.append(
                {
                    "id": uuid.uuid4().hex,
                    "userId": user["id"],
                    "email": user["email"],
                    "name": user["name"],
                    "picture": picture,
                    "provider": provider,
                    "userAgent": user_agent,
                    "at": stamp,
                }
            )

    async def list_users(self) -> list[dict[str, Any]]:
        async with self.lock:
            users = [_visitor(record) for record in self.users_by_id.values()]
        return sorted(users, key=lambda u: u["lastLoginAt"] or "", reverse=True)

    async def list_logins(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.lock:
            return list(reversed(self.logins[-limit:]))


Store = MongoStore | InMemoryStore


def create_store() -> Store:
    if settings.use_in_memory_db:
        logger.info("Using in-memory store")
        return InMemoryStore()
    logger.info("Connecting to MongoDB at %s", settings.mongo_uri)
    return MongoStore(settings.mongo_uri, settings.mongo_db_name)
