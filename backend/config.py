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
    # Signs in-memory-store session tokens. Set a long random value in any shared deployment.
    session_secret: str = Field(default_factory=lambda: os.getenv("SESSION_SECRET", "dev-only-insecure-session-secret"))
    enable_dev_endpoints: bool = Field(default_factory=lambda: env_flag("ENABLE_DEV_ENDPOINTS"))
    use_in_memory_db: bool = Field(default_factory=lambda: env_flag("USE_IN_MEMORY_DB"))
    google_api_key: str | None = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    # "Sign in with Google" OAuth client ID (public, not a secret). When set, visitors must sign in with
    # Google and email/password sign-up is turned off.
    google_client_id: str = Field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_ID", "").strip())
    # Emails allowed to open the Visitors (admin) page.
    admin_emails: list[str] = Field(default_factory=lambda: [e.lower() for e in _csv(os.getenv("ADMIN_EMAILS", ""))])
    mongo_uri: str = Field(default_factory=resolve_mongo_uri)
    mongo_db_name: str = Field(default_factory=lambda: os.getenv("MONGODB_DB", "algo-trade-simulator"))
    gemini_models: list[str] = Field(default_factory=lambda: _csv(os.getenv("GEMINI_MODELS", DEFAULT_GEMINI_MODELS)))
    gemini_budget_seconds: float = Field(default_factory=lambda: float(os.getenv("GEMINI_BUDGET_SECONDS", "12")))
    # Backtests report on the last BACKTEST_PERIOD; the extra HISTORY_PERIOD before it warms up indicators and
    # gives the machine-learning strategy past data to train on.
    backtest_period: str = Field(default_factory=lambda: os.getenv("BACKTEST_PERIOD", "2y"))
    history_period: str = Field(default_factory=lambda: os.getenv("HISTORY_PERIOD", "5y"))
    trading_fee_bps: float = Field(default_factory=lambda: float(os.getenv("TRADING_FEE_BPS", "10")))
    # Default slippage: how much worse than the close each fill is assumed to be.
    slippage_bps: float = Field(default_factory=lambda: float(os.getenv("SLIPPAGE_BPS", "5")))
    # When a decision made at a day's close is filled: "next_open" (realistic) or "close" (the same close).
    execution: str = Field(default_factory=lambda: os.getenv("EXECUTION", "next_open"))
    # Annual risk-free rate used by Sharpe, Sortino and alpha (e.g. 0.04 = 4%).
    risk_free_rate: float = Field(default_factory=lambda: float(os.getenv("RISK_FREE_RATE", "0.04")))
    # Annual fee for borrowing shares to short, in basis points.
    borrow_bps: float = Field(default_factory=lambda: float(os.getenv("BORROW_BPS", "50")))
    # Brokerage on Indian (NSE/BSE) delivery trades, in basis points; most discount brokers charge 0.
    india_brokerage_bps: float = Field(default_factory=lambda: float(os.getenv("INDIA_BROKERAGE_BPS", "0")))
    # Currency the portfolio is valued in; positions in other currencies are converted at daily FX rates.
    base_currency: str = Field(default_factory=lambda: os.getenv("BASE_CURRENCY", "USD").upper())
    # Daily signal job (/cron/daily): the bearer token schedulers must send (Vercel Cron sends CRON_SECRET).
    cron_secret: str = Field(default_factory=lambda: os.getenv("CRON_SECRET", ""))
    # Alerts. Email needs SMTP settings; Telegram needs a bot token (each user adds their own chat id).
    smtp_host: str = Field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    smtp_port: int = Field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: str = Field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_password: str = Field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    smtp_from: str = Field(default_factory=lambda: os.getenv("SMTP_FROM", ""))
    telegram_bot_token: str = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    # After-tax returns for US-taxed assets: short-term (ordinary income) and long-term capital-gains rates.
    us_short_term_tax: float = Field(default_factory=lambda: float(os.getenv("US_SHORT_TERM_TAX", "0.24")))
    us_long_term_tax: float = Field(default_factory=lambda: float(os.getenv("US_LONG_TERM_TAX", "0.15")))
    # Heavy research endpoints (robustness, baskets, optimiser...) allowed per user per minute, per server instance.
    heavy_requests_per_minute: int = Field(default_factory=lambda: int(os.getenv("HEAVY_REQUESTS_PER_MINUTE", "20")))
    # Error reporting (optional): a Sentry DSN.
    sentry_dsn: str = Field(default_factory=lambda: os.getenv("SENTRY_DSN", ""))
    # Alpaca paper trading (optional, admins only): mirror simulations' signals as paper orders.
    alpaca_key_id: str = Field(default_factory=lambda: os.getenv("ALPACA_KEY_ID", ""))
    alpaca_secret_key: str = Field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_base_url: str = Field(
        default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    )
    # Cache daily price history in MongoDB (when MongoDB is used), refreshing only the latest days.
    cache_prices_in_db: bool = Field(default_factory=lambda: env_flag("CACHE_PRICES_IN_DB", "true"))
    # Contact email the SEC requires in requests for EDGAR data (earnings dates). Without it, earnings features are off.
    sec_contact_email: str = Field(default_factory=lambda: os.getenv("SEC_CONTACT_EMAIL", "").strip())
    # Web Push notifications (VAPID keys; generate with backend/scripts/vapid_keys.py).
    vapid_public_key: str = Field(default_factory=lambda: os.getenv("VAPID_PUBLIC_KEY", "").strip())
    vapid_private_key: str = Field(default_factory=lambda: os.getenv("VAPID_PRIVATE_KEY", "").strip())
    vapid_subject: str = Field(default_factory=lambda: os.getenv("VAPID_SUBJECT", "https://algo-trade-mu.vercel.app"))
    # The site's public address (e.g. https://algo-trade-mu.vercel.app), for share-link previews.
    public_url: str = Field(default_factory=lambda: os.getenv("PUBLIC_URL", "").strip())
    # Built frontend (npm run build -> dist/) to serve from the API, for single-container deploys.
    static_dir: str = Field(default_factory=lambda: os.getenv("STATIC_DIR", ""))
    yahoo_user_agent: str = Field(
        default_factory=lambda: os.getenv(
            "YAHOO_USER_AGENT", "Mozilla/5.0 (compatible; AlgoTradeSimulator/1.0; +https://example.com)"
        ),
    )


settings = Settings()
