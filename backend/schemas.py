"""Request bodies accepted by the API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, model_validator


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class GoogleLoginRequest(BaseModel):
    credential: str = Field(min_length=20, max_length=4096)  # the ID token from Google Identity Services


class DevAuthBypassRequest(BaseModel):
    email: EmailStr | None = None
    name: str | None = Field(default=None, max_length=120)


class Operand(BaseModel):
    """One side of a rule: the price, an indicator over `period` days, or a fixed `value`."""

    kind: Literal["price", "sma", "ema", "rsi", "value"]
    period: int | None = Field(default=None, ge=2, le=250)
    value: float | None = Field(default=None, ge=-1e9, le=1e9)

    @model_validator(mode="after")
    def check(self) -> Operand:
        if self.kind in {"sma", "ema", "rsi"} and self.period is None:
            raise ValueError(f"{self.kind.upper()} needs a period")
        if self.kind == "value" and self.value is None:
            raise ValueError("A fixed value needs a number")
        return self


class Condition(BaseModel):
    left: Operand
    op: Literal[">", "<", "crosses_above", "crosses_below"]
    right: Operand


class StrategyRules(BaseModel):
    entry: list[Condition] = Field(min_length=1, max_length=5)  # all must hold to buy
    exit: list[Condition] = Field(default_factory=list, max_length=5)  # any one sells
    stopLoss: float | None = Field(default=None, gt=0, lt=1)  # fraction below the entry price
    takeProfit: float | None = Field(default=None, gt=0, le=10)  # fraction above the entry price


class CustomStrategyInput(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    description: str | None = Field(default=None, max_length=300)
    rules: StrategyRules


class SimulationInput(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    strategy: str = Field(min_length=1, max_length=60)  # display name
    startingCapital: float = Field(gt=0, le=1e9)
    notes: str | None = Field(default=None, max_length=400)
    # How the simulation trades. Older simulations without these are valued as buy & hold.
    strategyId: str = Field(default="buy-hold", max_length=60)
    parameters: dict[str, float] = Field(default_factory=dict)
    rules: StrategyRules | None = None
    # When the simulated money was invested; defaults to today. Backdating shows a real track record.
    startDate: date | None = None


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
    # Strategy Builder ("custom")
    rules: StrategyRules | None = None


class PredictionPayload(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    # Optional: the strategy to evaluate. Without it, the user's last trained strategy for the symbol is used.
    strategyId: str | None = Field(default=None, max_length=60)
    parameters: dict[str, float] | None = None
    rules: StrategyRules | None = None


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatHistoryItem] = Field(default_factory=list)
