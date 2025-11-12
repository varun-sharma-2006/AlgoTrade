import asyncio
import os
import sys

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in TRUTHY_ENV_VALUES


def resolve_mongo_uri() -> str:
    return (
        os.getenv("MONGO_URL")
        or os.getenv("MONGODB_URI")
        or os.getenv("MONGO_URI")
        or "mongodb://localhost:27017"
    )


def get_database_name() -> str:
    return os.getenv("MONGODB_DB", "algo-trade-simulator")


async def ping_mongo(uri: str) -> None:
    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    try:
        result = await client.admin.command("ping")
    finally:
        client.close()
    print(f"Ping successful: {result}")


async def main() -> int:
    if env_flag("USE_IN_MEMORY_DB"):
        print("USE_IN_MEMORY_DB is enabled; skipping MongoDB connection test.")
        return 0

    uri = resolve_mongo_uri()
    database = get_database_name()
    print(f"Testing MongoDB connection at {uri} (database: {database})")

    try:
        await ping_mongo(uri)
    except ServerSelectionTimeoutError as exc:
        print(f"Unable to connect to MongoDB: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error while connecting to MongoDB: {exc}")
        return 1

    print("Connection check completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
