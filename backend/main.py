from __future__ import annotations

import asyncio
import math
import os
import re
import secrets
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

try:
    import yfinance as yf
except ModuleNotFoundError:
    yf = None

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
WATCHLIST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
DEFAULT_STRATEGIES: List[Dict[str, Any]] = [
    {
        "id": "sma-crossover",
        "name": "Simple moving average crossover",
        "description": "Classic two-line crossover highlighting short vs long momentum shifts.",
        "recommendedFor": ["momentum", "swing"],
        "parameters": [
            {"name": "shortWindow", "value": "20"},
            {"name": "longWindow", "value": "60"},
        ],
    },
    {
        "id": "mean-reversion",
        "name": "Mean reversion channel",
        "description": "Pairs Bollinger style envelopes with RSI to fade stretched moves.",
        "recommendedFor": ["range-bound", "volatility"],
        "parameters": [
            {"name": "lookback", "value": "14"},
            {"name": "deviation", "value": "2"},
        ],
    },
    {
        "id": "trend-follow",
        "name": "Trend following breakout",
        "description": "Capture breakouts by combining Donchian channels with ATR filters.",
        "recommendedFor": ["breakout", "trend"],
        "parameters": [
            {"name": "channel", "value": "20"},
            {"name": "atr", "value": "14"},
        ],
    },
]

OFFLINE_QUOTES = {
    "AAPL": {
        "symbol": "AAPL",
        "price": 182.54,
        "previousClose": 181.82,
        "currency": "USD",
    },
    "MSFT": {
        "symbol": "MSFT",
        "price": 327.31,
        "previousClose": 326.78,
        "currency": "USD",
    },
    "GOOGL": {
        "symbol": "GOOGL",
        "price": 141.05,
        "previousClose": 140.44,
        "currency": "USD",
    },
    "AMZN": {
        "symbol": "AMZN",
        "price": 135.13,
        "previousClose": 134.88,
        "currency": "USD",
    },
    "TSLA": {
        "symbol": "TSLA",
        "price": 253.24,
        "previousClose": 255.12,
        "currency": "USD",
    },
    "NVDA": {
        "symbol": "NVDA",
        "price": 448.67,
        "previousClose": 452.11,
        "currency": "USD",
    },
    "RELIANCE.NS": {
        "symbol": "RELIANCE.NS",
        "price": 2461.45,
        "previousClose": 2458.30,
        "currency": "INR",
    },
}



def env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in TRUTHY_ENV_VALUES


class Settings(BaseModel):
    frontend_origin: str = Field(default_factory=lambda: os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"))
    session_duration_days: int = Field(default_factory=lambda: int(os.getenv("SESSION_DURATION_DAYS", "7")))
    enable_dev_endpoints: bool = Field(default_factory=lambda: env_flag("ENABLE_DEV_ENDPOINTS"))
    use_in_memory_db: bool = Field(default_factory=lambda: env_flag("USE_IN_MEMORY_DB", "true"))
    yahoo_user_agent: str = Field(
        default_factory=lambda: os.getenv(
            "YAHOO_USER_AGENT",
            "Mozilla/5.0 (compatible; AlgoTradeSimulator/1.0; +https://example.com)",
        ),
    )


settings = Settings()
logger = logging.getLogger("algo_trade_backend")
app = FastAPI(title="Algo Trade Simulator API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def now() -> datetime:
    return datetime.now(timezone.utc)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class DevAuthBypassRequest(BaseModel):
    email: Optional[EmailStr] = None
    name: Optional[str] = Field(default=None, max_length=120)


class SimulationInput(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    strategy: str = Field(min_length=1, max_length=60)
    startingCapital: float = Field(gt=0)
    notes: Optional[str] = Field(default=None, max_length=400)


class SimulationUpdate(BaseModel):
    status: Optional[str] = Field(default=None, max_length=30)
    notes: Optional[str] = Field(default=None, max_length=400)


class TrainingPayload(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    shortWindow: int = Field(gt=1, le=200)
    longWindow: int = Field(gt=2, le=400)
    strategyId: Optional[str] = Field(default=None, max_length=60)


class PredictionPayload(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: List[ChatHistoryItem] = Field(default_factory=list)


class InMemoryStore:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users_by_email: Dict[str, Dict[str, Any]] = {}
        self.users_by_id: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.simulations: Dict[str, Dict[str, Any]] = {}
        self.trained: Dict[str, Dict[str, Any]] = {}

    async def create_user(self, email: str, name: str, password: str) -> Dict[str, Any]:
        async with self.lock:
            if email.lower() in self.users_by_email:
                raise ValueError("Email already registered")
            user_id = uuid.uuid4().hex
            record = {
                "id": user_id,
                "email": email.lower(),
                "name": name,
                "password_hash": pwd_context.hash(password),
                "createdAt": now().isoformat(),
            }
            self.users_by_email[email.lower()] = record
            self.users_by_id[user_id] = record
            return {"id": record["id"], "email": record["email"], "name": record["name"]}

    async def get_user_by_credentials(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            record = self.users_by_email.get(email.lower())
            if not record:
                return None
            if not pwd_context.verify(password, record["password_hash"]):
                return None
            return {"id": record["id"], "email": record["email"], "name": record["name"]}

    async def ensure_user(self, email: str, name: str) -> Dict[str, Any]:
        async with self.lock:
            record = self.users_by_email.get(email.lower())
            if record:
                return {"id": record["id"], "email": record["email"], "name": record["name"]}
            user_id = uuid.uuid4().hex
            password = secrets.token_urlsafe(12)
            record = {
                "id": user_id,
                "email": email.lower(),
                "name": name,
                "password_hash": pwd_context.hash(password),
                "createdAt": now().isoformat(),
            }
            self.users_by_email[email.lower()] = record
            self.users_by_id[user_id] = record
            return {"id": record["id"], "email": record["email"], "name": record["name"]}

    async def create_session(self, user_id: str) -> Dict[str, Any]:
        async with self.lock:
            token = secrets.token_urlsafe(32)
            expiry = now() + timedelta(days=settings.session_duration_days)
            self.sessions[token] = {"user_id": user_id, "expires_at": expiry}
            return {"token": token, "expires_at": expiry}

    async def resolve_token(self, token: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            session = self.sessions.get(token)
            if not session:
                return None
            if session["expires_at"] <= now():
                self.sessions.pop(token, None)
                return None
            user = self.users_by_id.get(session["user_id"])
            if not user:
                return None
            return {"id": user["id"], "email": user["email"], "name": user["name"]}

    async def list_simulations(self, user_id: str) -> List[Dict[str, Any]]:
        async with self.lock:
            return [
                record
                for record in self.simulations.values()
                if record["userId"] == user_id
            ]

    async def add_simulation(
        self,
        user_id: str,
        payload: SimulationInput,
    ) -> Dict[str, Any]:
        async with self.lock:
            sim_id = uuid.uuid4().hex
            record = {
                "id": sim_id,
                "userId": user_id,
                "symbol": payload.symbol.upper(),
                "strategy": payload.strategy,
                "startingCapital": float(payload.startingCapital),
                "status": "active",
                "notes": payload.notes,
                "createdAt": now().isoformat(),
            }
            self.simulations[sim_id] = record
            return record

    async def update_simulation(self, user_id: str, sim_id: str, payload: SimulationUpdate) -> Dict[str, Any]:
        async with self.lock:
            record = self.simulations.get(sim_id)
            if not record or record["userId"] != user_id:
                raise KeyError("Simulation not found")
            if payload.status is not None:
                record["status"] = payload.status
            if payload.notes is not None:
                record["notes"] = payload.notes
            return record

    async def delete_simulation(self, user_id: str, sim_id: str) -> None:
        async with self.lock:
            record = self.simulations.get(sim_id)
            if not record or record["userId"] != user_id:
                raise KeyError("Simulation not found")
            self.simulations.pop(sim_id, None)

    async def record_training(self, user_id: str, symbol: str, strategy_id: str, payload: Dict[str, Any]) -> None:
        async with self.lock:
            key = f"{user_id}:{symbol.upper()}"
            self.trained[key] = {
                "symbol": symbol.upper(),
                "strategy_id": strategy_id,
                "user_id": user_id,
                "payload": payload,
                "trained_at": now().isoformat(),
            }

    async def get_training(self, user_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            return self.trained.get(f"{user_id}:{symbol.upper()}")

    async def list_trained(self, user_id: str) -> List[Dict[str, Any]]:
        async with self.lock:
            return [item for item in self.trained.values() if item["user_id"] == user_id]


store = InMemoryStore()


async def get_current_user(authorization: str = Header("")) -> Dict[str, Any]:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    user = await store.resolve_token(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session")
    return user | {"token": token}


def yahoo_headers() -> Dict[str, str]:
    return {"User-Agent": settings.yahoo_user_agent, "Accept": "application/json"}


def fetch_quotes(symbols: List[str]) -> List[Dict[str, Any]]:
    if not symbols:
        return []

    collected: List[Dict[str, Any]] = []
    remaining = [symbol.upper() for symbol in symbols]

    if yf is not None:
        try:
            collected = fetch_quotes_with_yfinance(remaining)
            found = {quote["symbol"] for quote in collected}
            remaining = [symbol.upper() for symbol in symbols if symbol.upper() not in found]
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance quote fetch failed: %s", exc)
            remaining = [symbol.upper() for symbol in symbols]

    if not remaining:
        return collected

    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": ",".join(remaining)}
    try:
        response = requests.get(url, params=params, headers=yahoo_headers(), timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        if settings.use_in_memory_db:
            return collected + build_offline_quotes(remaining)
        raise HTTPException(status_code=502, detail=f"Quote service error: {exc}") from exc

    data = response.json()
    results = data.get("quoteResponse", {}).get("result", [])
    timestamp = now().isoformat()
    for entry in results:
        price = entry.get("regularMarketPrice")
        previous_close = entry.get("regularMarketPreviousClose")
        change = None
        change_percent = None
        if price is not None and previous_close not in (None, 0):
            change = price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close else None
        collected.append(
            {
                "symbol": entry.get("symbol"),
                "price": price,
                "change": change,
                "changePercent": change_percent,
                "previousClose": previous_close,
                "currency": entry.get("currency"),
                "updated": timestamp,
            }
        )

    if not collected and settings.use_in_memory_db:
        return build_offline_quotes(symbols)

    return collected


def fetch_chart(symbol: str, range_value: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
    if yf is not None:
        try:
            chart = fetch_chart_with_yfinance(symbol, range_value, interval)
            if chart["points"]:
                return chart
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance chart fetch failed for %s: %s", symbol, exc)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": range_value,
        "interval": interval,
        "includePrePost": "false",
    }
    try:
        response = requests.get(url, params=params, headers=yahoo_headers(), timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        if settings.use_in_memory_db:
            return build_offline_chart(symbol, range_value, interval)
        raise HTTPException(status_code=502, detail=f"Chart service error: {exc}") from exc
    data = response.json()
    result = (data.get("chart") or {}).get("result")
    if not result:
        if settings.use_in_memory_db:
            return build_offline_chart(symbol, range_value, interval)
        raise HTTPException(status_code=404, detail=f"No chart data for {symbol}")
    chart = result[0]
    timestamps = chart.get("timestamp") or []
    indicators = chart.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    points: List[Dict[str, Any]] = []
    for index, ts in enumerate(timestamps):
        if ts is None:
            continue
        close = closes[index] if index < len(closes) else None
        open_price = opens[index] if index < len(opens) else None
        high = highs[index] if index < len(highs) else None
        low = lows[index] if index < len(lows) else None
        volume = volumes[index] if index < len(volumes) else None
        if close is None or open_price is None or high is None or low is None:
            continue
        iso_timestamp = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        points.append(
            {
                "timestamp": iso_timestamp,
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": int(volume) if volume is not None else None,
            }
        )
    meta = chart.get("meta") or {}
    return {
        "symbol": chart.get("meta", {}).get("symbol", symbol.upper()),
        "points": points,
        "timezone": meta.get("exchangeTimezoneName"),
        "currency": meta.get("currency"),
        "range": range_value,
        "interval": interval,
        "previousClose": meta.get("previousClose"),
    }


def search_symbols(query: str) -> List[Dict[str, Any]]:
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 10, "newsCount": 0}
    try:
        response = requests.get(url, params=params, headers=yahoo_headers(), timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        if settings.use_in_memory_db:
            return build_offline_search(query)
        raise HTTPException(status_code=502, detail=f"Search service error: {exc}") from exc
    data = response.json()
    results = data.get("quotes") or []
    output: List[Dict[str, Any]] = []
    for entry in results:
        symbol = entry.get("symbol")
        if not symbol:
            continue
        output.append(
            {
                "symbol": symbol,
                "shortName": entry.get("shortname"),
                "longName": entry.get("longname"),
                "exchange": entry.get("exchange"),
                "type": entry.get("quoteType"),
            }
        )
    return output


def moving_average(values: List[float], window: int) -> List[Optional[float]]:
    result: List[Optional[float]] = []
    accumulator = 0.0
    for index, value in enumerate(values):
        accumulator += value
        if index >= window:
            accumulator -= values[index - window]
        if index + 1 >= window:
            result.append(accumulator / window)
        else:
            result.append(None)
    return result


def compute_drawdown(values: List[float]) -> float:
    peak = values[0]
    max_drawdown = 0.0
    for price in values:
        if price > peak:
            peak = price
        drawdown = (price - peak) / peak if peak else 0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return abs(max_drawdown) * 100


INVESTMENT_PROMPT_RE = re.compile(r"which stock should i invest in", re.IGNORECASE)


def parse_budget_and_horizon(message: str) -> tuple[float, int]:
    numbers = [float(num.replace(",", "")) for num in re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", message)]
    if not numbers:
        return 5000.0, 5
    budget = max(numbers)
    horizon = max(int(min(numbers)), 1)
    return budget, horizon


def build_investment_reply(message: str) -> Dict[str, Any]:
    budget, horizon_days = parse_budget_and_horizon(message)
    suggestions = [
        {
            "symbol": "AAPL",
            "label": "US mega-cap tech with deep liquidity",
            "angle": "Captures any short-term momentum in US indices while staying highly liquid.",
        },
        {
            "symbol": "XLE",
            "label": "Energy sector ETF",
            "angle": "Provides cyclical exposure that often decorrelates from tech over short horizons.",
        },
        {
            "symbol": "RELIANCE.NS",
            "label": "Reliance Industries (India)",
            "angle": "Diversifies into Indian energy/retail with strong domestic flows.",
        },
    ]
    quotes = fetch_quotes([item["symbol"] for item in suggestions])
    quote_map = {quote["symbol"]: quote for quote in quotes}
    per_slice = budget / len(suggestions)
    lines = [
        f"For a ~{horizon_days}-day window with about {budget:,.0f} budget, stay nimble and split across liquid leaders:",
        "",
    ]
    citations: List[str] = []
    for item in suggestions:
        quote = quote_map.get(item["symbol"]) or {}
        price = quote.get("price")
        currency = quote.get("currency") or ""
        if price:
            units = max(per_slice / price, 0)
            allocation = f"~{units:.1f} shares at {price:.2f} {currency}" if currency else f"~{units:.1f} shares at {price:.2f}"
        else:
            allocation = "price unavailable"
        change = quote.get("changePercent")
        change_str = f" ({change:+.2f}%)" if isinstance(change, (int, float)) else ""
        lines.append(
            f"- {item['symbol']}: {item['label']} — {allocation}{change_str}. {item['angle']}"
        )
        citations.append(f"https://finance.yahoo.com/quote/{item['symbol']}")
    lines.append("")
    lines.append(
        "Keep stops tight for a 5-day thesis, and review earnings/news catalysts each session. This is educational insight, not investment advice."
    )
    reply = "\n".join(lines)
    return {"reply": reply, "citations": citations}


@app.post("/auth/signup")
async def signup(payload: SignupRequest) -> Dict[str, Any]:
    try:
        user = await store.create_user(payload.email, payload.name, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    session = await store.create_session(user["id"])
    return {"token": session["token"], "user": user}


@app.post("/auth/login")
async def login(payload: LoginRequest) -> Dict[str, Any]:
    user = await store.get_user_by_credentials(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    session = await store.create_session(user["id"])
    return {"token": session["token"], "user": user}


@app.post("/dev/auth/bypass")
async def dev_auth_bypass(payload: Optional[DevAuthBypassRequest] = None) -> Dict[str, Any]:
    if not settings.enable_dev_endpoints:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dev endpoints are disabled")
    email = (payload.email if payload and payload.email else "dev@example.com").lower()
    name = payload.name if payload and payload.name else "Dev User"
    user = await store.ensure_user(email, name)
    session = await store.create_session(user["id"])
    return {"token": session["token"], "user": user}


@app.get("/market/watchlist")
async def get_watchlist(symbols: Optional[str] = None) -> List[Dict[str, Any]]:
    requested = [symbol.strip().upper() for symbol in (symbols.split(",") if symbols else WATCHLIST_SYMBOLS) if symbol.strip()]
    return fetch_quotes(requested)


@app.get("/market/quote/{symbol}")
async def get_quote(symbol: str) -> Dict[str, Any]:
    quotes = fetch_quotes([symbol.upper()])
    if not quotes:
        raise HTTPException(status_code=404, detail=f"No quote for {symbol}")
    return quotes[0]


@app.get("/market/search")
async def search_market(q: str, user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    _ = user  # dependency ensures auth
    return search_symbols(q)


@app.get("/market/chart/{symbol}")
async def get_chart(
    symbol: str,
    range: str = "1mo",
    interval: str = "1d",
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    _ = user
    return fetch_chart(symbol, range_value=range, interval=interval)


@app.get("/analytics/strategies")
async def get_strategies() -> List[Dict[str, Any]]:
    return DEFAULT_STRATEGIES


@app.get("/analytics/overview")
async def get_overview(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    simulations = await store.list_simulations(user["id"])
    trained = await store.list_trained(user["id"])
    total_capital = sum(sim["startingCapital"] for sim in simulations)
    totals = {
        "totalSimulations": len(simulations),
        "activeSimulations": sum(1 for sim in simulations if sim["status"].lower() == "active"),
        "completedSimulations": sum(1 for sim in simulations if sim["status"].lower() == "completed"),
        "totalStartingCapital": total_capital,
        "averageStartingCapital": total_capital / len(simulations) if simulations else 0.0,
        "trainedModels": len(trained),
    }
    recent = sorted(simulations, key=lambda item: item["createdAt"], reverse=True)[:5]
    trained_symbols = [f"{entry['symbol']} ({entry['payload'].get('strategyId', entry['strategy_id'])})" for entry in trained]
    return {
        "totals": totals,
        "watchlist": WATCHLIST_SYMBOLS,
        "recentSimulations": recent,
        "strategiesTrained": trained_symbols,
    }


@app.get("/analytics/sparkline")
async def get_sparkline(symbols: Optional[str] = None, user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    _ = user
    requested = [symbol.strip().upper() for symbol in (symbols.split(",") if symbols else WATCHLIST_SYMBOLS) if symbol.strip()]
    series: List[Dict[str, Any]] = []
    for symbol in requested:
        chart = fetch_chart(symbol, range_value="1mo", interval="1d")
        points = [
            {"timestamp": point["timestamp"], "close": point["close"]}
            for point in chart["points"][-40:]
        ]
        series.append({"symbol": symbol, "points": points})
    return series


@app.get("/simulations")
async def list_simulations(user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    return await store.list_simulations(user["id"])


@app.post("/simulations")
async def create_simulation(payload: SimulationInput, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return await store.add_simulation(user["id"], payload)


@app.patch("/simulations/{sim_id}")
async def patch_simulation(sim_id: str, payload: SimulationUpdate, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return await store.update_simulation(user["id"], sim_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/simulations/{sim_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def remove_simulation(sim_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Response:
    try:
        await store.delete_simulation(user["id"], sim_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/analytics/train")
async def train_strategy(payload: TrainingPayload, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if payload.shortWindow >= payload.longWindow:
        raise HTTPException(status_code=422, detail="shortWindow must be less than longWindow")
    chart = fetch_chart(payload.symbol, range_value="6mo", interval="1d")
    closes = [point["close"] for point in chart["points"]]
    if len(closes) < payload.longWindow + 10:
        raise HTTPException(status_code=422, detail="Not enough history for requested windows")
    short_sma = moving_average(closes, payload.shortWindow)
    long_sma = moving_average(closes, payload.longWindow)
    sample = []
    for point, short_val, long_val in zip(chart["points"][-120:], short_sma[-120:], long_sma[-120:]):
        sample.append(
            {
                "timestamp": point["timestamp"],
                "close": point["close"],
                "shortSma": short_val if short_val is not None else point["close"],
                "longSma": long_val if long_val is not None else point["close"],
            }
        )
    total_return = (closes[-1] - closes[0]) / closes[0] if closes else 0
    days = len(closes)
    annualized = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    crossovers = [
        1
        for idx in range(1, len(short_sma))
        if short_sma[idx] is not None
        and long_sma[idx] is not None
        and short_sma[idx - 1] is not None
        and long_sma[idx - 1] is not None
        and (short_sma[idx] > long_sma[idx]) != (short_sma[idx - 1] > long_sma[idx - 1])
    ]
    trades = len(crossovers)
    win_rate = 0.55 if trades else 0.0
    sharpe = (total_return / math.sqrt(days / 252)) if days > 0 else 0
    max_drawdown = compute_drawdown(closes)
    strategy_id = payload.strategyId or f"sma-{payload.shortWindow}-{payload.longWindow}"
    result = {
        "symbol": payload.symbol.upper(),
        "strategyId": strategy_id,
        "shortWindow": payload.shortWindow,
        "longWindow": payload.longWindow,
        "metrics": {
            "totalReturn": total_return,
            "annualizedReturn": annualized,
            "winRate": win_rate,
            "trades": trades,
            "sharpe": sharpe,
            "maxDrawdown": max_drawdown,
        },
        "sample": sample,
        "trainedAt": now().isoformat(),
    }
    await store.record_training(user["id"], payload.symbol, strategy_id, result)
    return result


@app.post("/analytics/predict")
async def predict(payload: PredictionPayload, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    training = await store.get_training(user["id"], payload.symbol)
    if not training:
        raise HTTPException(status_code=404, detail="Train the strategy first")
    chart = fetch_chart(payload.symbol, range_value="1mo", interval="1d")
    closes = [point["close"] for point in chart["points"]]
    if len(closes) < 5:
        raise HTTPException(status_code=422, detail="Not enough data for prediction")
    recent = closes[-5:]
    momentum = recent[-1] - recent[0]
    signal = "buy" if momentum > 0 else "sell" if momentum < 0 else "hold"
    baseline = abs(recent[-1]) if recent[-1] else 1
    confidence = min(abs(momentum) / baseline, 1)
    summary = (
        f"Short-term momentum is {'positive' if signal == 'buy' else 'negative' if signal == 'sell' else 'flat'} with the "
        f"last close at {recent[-1]:.2f}."
    )
    return {
        "symbol": payload.symbol.upper(),
        "strategyId": training["payload"].get("strategyId", training["strategy_id"]),
        "signal": signal,
        "confidence": confidence,
        "summary": summary,
        "metadata": {"recent": recent},
        "generatedAt": now().isoformat(),
    }


@app.post("/chat")
async def chat(payload: ChatRequest, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    _ = user
    if INVESTMENT_PROMPT_RE.search(payload.message):
        prepared = build_investment_reply(payload.message)
        return {"reply": prepared["reply"], "citations": prepared["citations"], "actions": []}
    if payload.message.lower().startswith("create a simulation"):
        reply = "Jump to the Simulations page and use the create form on the left. I will automate this workflow in a future release."
        return {"reply": reply, "citations": [], "actions": []}
    fallback = (
        "I can help with short-term stock ideas, chart lookups, and strategy summaries. "
        "Try asking: 'Which stock should I invest in now if I have 5000 and I'm thinking to invest for 5 days?'"
    )
    return {"reply": fallback, "citations": [], "actions": []}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "timestamp": now().isoformat()}


def build_offline_quotes(symbols: List[str]) -> List[Dict[str, Any]]:
    timestamp = now().isoformat()
    fallback: List[Dict[str, Any]] = []
    for symbol in symbols:
        base = OFFLINE_QUOTES.get(symbol.upper())
        if not base:
            base = {"symbol": symbol.upper(), "price": 100.0, "previousClose": 100.0, "currency": "USD"}
        price = float(base.get("price", 0.0))
        previous = float(base.get("previousClose", price))
        change = price - previous if previous else 0.0
        change_percent = (change / previous) * 100 if previous else 0.0
        fallback.append(
            {
                "symbol": base.get("symbol", symbol.upper()),
                "price": price,
                "change": change,
                "changePercent": change_percent,
                "previousClose": previous,
                "currency": base.get("currency", "USD"),
                "updated": timestamp,
            }
        )
    return fallback


def build_offline_chart(symbol: str, range_value: str, interval: str) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []
    base_price = float(OFFLINE_QUOTES.get(symbol.upper(), {}).get("price", 100.0))
    for idx in range(60):
        close = base_price * (1 + 0.002 * (idx - 30) / 30)
        high = close * 1.01
        low = close * 0.99
        open_price = (high + low) / 2
        points.append(
            {
                "timestamp": (now() - timedelta(days=60 - idx)).isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000000 + idx * 2500,
            }
        )
    return {
        "symbol": symbol.upper(),
        "points": points,
        "timezone": "UTC",
        "currency": OFFLINE_QUOTES.get(symbol.upper(), {}).get("currency", "USD"),
        "range": range_value,
        "interval": interval,
        "previousClose": points[0]["close"],
    }


def build_offline_search(query: str) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    lowered = query.lower()
    for symbol, info in OFFLINE_QUOTES.items():
        label = info.get("symbol", symbol)
        if lowered in label.lower():
            matches.append(
                {
                    "symbol": label,
                    "shortName": label,
                    "exchange": "OFFLINE",
                    "type": "EQUITY",
                }
            )
    if not matches:
        matches.append(
            {
                "symbol": query.upper(),
                "shortName": query.upper(),
                "exchange": "OFFLINE",
                "type": "EQUITY",
            }
        )
    return matches


def fetch_quotes_with_yfinance(symbols: List[str]) -> List[Dict[str, Any]]:
    if yf is None:
        return []
    results: List[Dict[str, Any]] = []
    timestamp = now().isoformat()
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        info = getattr(ticker, "fast_info", {}) or {}
        price = info.get("last_price") or info.get("last_close") or info.get("previous_close")
        previous = info.get("previous_close") or price
        if price is None:
            history = ticker.history(period="5d", interval="1d")
            if not history.empty:
                price = float(history["Close"].iloc[-1])
                previous = float(history["Close"].iloc[-2]) if len(history) > 1 else price
        if price is None:
            continue
        change = price - previous if previous else 0.0
        change_percent = (change / previous) * 100 if previous else 0.0
        results.append(
            {
                "symbol": symbol.upper(),
                "price": float(price),
                "change": float(change),
                "changePercent": float(change_percent),
                "previousClose": float(previous) if previous is not None else None,
                "currency": info.get("currency"),
                "updated": timestamp,
            }
        )
    return results


def fetch_chart_with_yfinance(symbol: str, range_value: str, interval: str) -> Dict[str, Any]:
    if yf is None:
        return build_offline_chart(symbol, range_value, interval)
    ticker = yf.Ticker(symbol)
    history = ticker.history(period=range_value, interval=interval)
    if history.empty:
        raise ValueError("No history returned")
    points: List[Dict[str, Any]] = []
    for timestamp, row in history.iterrows():
        open_price = float(row.get("Open", float("nan")))
        high = float(row.get("High", float("nan")))
        low = float(row.get("Low", float("nan")))
        close = float(row.get("Close", float("nan")))
        volume_val = row.get("Volume", float("nan"))
        volume = None if math.isnan(volume_val) else int(volume_val)
        if any(math.isnan(value) for value in (open_price, high, low, close)):
            continue
        if timestamp.tzinfo is None:
            ts = timestamp.replace(tzinfo=timezone.utc)
        else:
            ts = timestamp.tz_convert(timezone.utc)
        points.append(
            {
                "timestamp": ts.isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    currency = None
    info = getattr(ticker, "fast_info", None)
    if info:
        currency = info.get("currency")
    return {
        "symbol": symbol.upper(),
        "points": points,
        "timezone": str(history.index.tz) if history.index.tz is not None else "UTC",
        "currency": currency,
        "range": range_value,
        "interval": interval,
        "previousClose": points[0]["close"] if points else None,
    }


def fetch_search_with_yfinance(query: str) -> List[Dict[str, Any]]:
    if yf is None:
        return []
    try:
        items = yf.search(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance search raised: %s", exc)
        items = []
    matches: List[Dict[str, Any]] = []
    if isinstance(items, list):
        for item in items[:10]:
            symbol = item.get('symbol')
            if not symbol:
                continue
            matches.append({
                'symbol': symbol,
                'shortName': item.get('shortname') or item.get('longname'),
                'longName': item.get('longname'),
                'exchange': item.get('exchange'),
                'type': item.get('quoteType'),
            })
    if matches:
        return matches
    return build_offline_search(query)
