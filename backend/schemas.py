"""Request bodies accepted by the API."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class DevAuthBypassRequest(BaseModel):
    email: EmailStr | None = None
    name: str | None = Field(default=None, max_length=120)


class SimulationInput(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    strategy: str = Field(min_length=1, max_length=60)
    startingCapital: float = Field(gt=0)
    notes: str | None = Field(default=None, max_length=400)


class SimulationUpdate(BaseModel):
    status: str | None = Field(default=None, max_length=30)
    notes: str | None = Field(default=None, max_length=400)


class TrainingPayload(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    strategyId: str = Field(default="sma-crossover", max_length=60)
    # SMA crossover
    shortWindow: int = Field(default=20, gt=1, le=200)
    longWindow: int = Field(default=60, gt=2, le=400)
    # Mean reversion
    lookback: int = Field(default=20, ge=5, le=200)
    deviation: float = Field(default=2.0, gt=0, le=5)
    # Trend-following breakout
    channel: int = Field(default=20, ge=5, le=200)


class PredictionPayload(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    # Optional: the strategy to evaluate. Without it, the user's last trained strategy for the symbol is used.
    strategyId: str | None = Field(default=None, max_length=60)
    parameters: dict[str, float] | None = None


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatHistoryItem] = Field(default_factory=list)
