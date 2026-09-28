"""Environment-driven settings shared by the whole backend."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# backend/.env wins over stale machine-level variables so local config is predictable.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
DEFAULT_GEMINI_MODELS = "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-flash-latest"

logger = logging.getLogger("algo_trade_backend")


def env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in TRUTHY_ENV_VALUES


def resolve_mongo_uri() -> str:
    return os.getenv("MONGO_URL") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URI") or "mongodb://localhost:27017"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseModel):
    frontend_origin: str = Field(default_factory=lambda: os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"))
    session_duration_days: int = Field(default_factory=lambda: int(os.getenv("SESSION_DURATION_DAYS", "7")))
    enable_dev_endpoints: bool = Field(default_factory=lambda: env_flag("ENABLE_DEV_ENDPOINTS"))
    use_in_memory_db: bool = Field(default_factory=lambda: env_flag("USE_IN_MEMORY_DB"))
    google_api_key: str | None = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    mongo_uri: str = Field(default_factory=resolve_mongo_uri)
    mongo_db_name: str = Field(default_factory=lambda: os.getenv("MONGODB_DB", "algo-trade-simulator"))
    gemini_models: list[str] = Field(default_factory=lambda: _csv(os.getenv("GEMINI_MODELS", DEFAULT_GEMINI_MODELS)))
    gemini_budget_seconds: float = Field(default_factory=lambda: float(os.getenv("GEMINI_BUDGET_SECONDS", "12")))
    backtest_period: str = Field(default_factory=lambda: os.getenv("BACKTEST_PERIOD", "2y"))
    trading_fee_bps: float = Field(default_factory=lambda: float(os.getenv("TRADING_FEE_BPS", "10")))
    yahoo_user_agent: str = Field(
        default_factory=lambda: os.getenv(
            "YAHOO_USER_AGENT", "Mozilla/5.0 (compatible; AlgoTradeSimulator/1.0; +https://example.com)"
        ),
    )


settings = Settings()
