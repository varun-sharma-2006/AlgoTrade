"""Check local configuration: `python -m backend.scripts.check_setup`."""

from __future__ import annotations

import asyncio
import sys

from motor.motor_asyncio import AsyncIOMotorClient

from backend.config import settings


async def ping_mongo() -> None:
    client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        await client.admin.command("ping")
    finally:
        client.close()


def main() -> int:
    print("GOOGLE_API_KEY:", "configured" if settings.google_api_key else "not set (chatbot uses the built-in analyst)")
    if settings.use_in_memory_db:
        print("USE_IN_MEMORY_DB is enabled; skipping MongoDB check.")
        return 0
    print(f"Pinging MongoDB at {settings.mongo_uri} ...")
    try:
        asyncio.run(ping_mongo())
    except Exception as exc:
        print(f"Unable to connect to MongoDB: {exc}")
        return 1
    print("MongoDB connection OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
