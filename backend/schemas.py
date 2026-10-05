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


PERIOD_OPERANDS = {"sma", "ema", "rsi", "atr", "volume_sma", "highest", "lowest", "roc"}


class Operand(BaseModel):
    """One side of a rule: the price, an indicator over `period` days, or a fixed `value`."""

    kind: Literal[
        "price",
        "sma",
        "ema",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "volume",
        "volume_sma",
        "highest",
        "lowest",
        "roc",
        "value",
    ]
    period: int | None = Field(default=None, ge=2, le=250)
    value: float | None = Field(default=None, ge=-1e12, le=1e12)

    @model_validator(mode="after")
    def check(self) -> Operand:
        if self.kind in PERIOD_OPERANDS and self.period is None:
            raise ValueError(f"{self.kind.upper()} needs a period")
        if self.kind == "value" and self.value is None:
            raise ValueError("A fixed value needs a number")
        return self


class Condition(BaseModel):
    left: Operand
    op: Literal[">", "<", "crosses_above", "crosses_below"]
    right: Operand


class StrategyRules(BaseModel):
    entry: list[Condition] = Field(min_length=1, max_length=5)  # all (or any, see entryMode) must hold to enter
    exit: list[Condition] = Field(default_factory=list, max_length=5)  # any one closes the position
    entryMode: Literal["all", "any"] = "all"
    side: Literal["long", "short"] = "long"
    stopLoss: float | None = Field(default=None, gt=0, lt=1)  # fraction against the entry price
    takeProfit: float | None = Field(default=None, gt=0, le=10)  # fraction in favour of the entry price
    trailingStop: float | None = Field(default=None, gt=0, lt=1)  # fraction from the best close since entry
    maxHoldDays: int | None = Field(default=None, ge=1, le=1000)  # exit after this many trading days


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
    # Machine learning
    threshold: float = Field(default=0.52, ge=0.3, le=0.8)
    trainWindow: int = Field(default=504, ge=126, le=1000)
    horizon: int = Field(default=1, ge=1, le=20)  # predict the direction this many days ahead
    modelType: Literal[0, 1] = 0  # 0 = logistic regression, 1 = gradient-boosted trees
    # Regime switching
    erWindow: int = Field(default=20, ge=5, le=120)
    erThreshold: float = Field(default=0.3, ge=0.05, le=0.95)
    # Strategy Builder ("custom")
    rules: StrategyRules | None = None
    # Extra cost per trade on top of the fee; defaults to the server's SLIPPAGE_BPS.
    slippageBps: float | None = Field(default=None, ge=0, le=100)
    # How trades are executed and sized. Unset fields use the server's defaults.
    execution: Literal["close", "next_open"] | None = None
    sizing: Literal["full", "fixed", "vol-target"] = "full"
    sizeFraction: float = Field(default=0.5, gt=0, le=1)  # "fixed": share of equity per position
    targetVol: float = Field(default=0.15, gt=0.01, le=1)  # "vol-target": annual volatility to aim for
    maxLeverage: float = Field(default=1.0, gt=0, le=3)  # "vol-target": cap on position size
    allowShort: bool = False
    borrowBps: float | None = Field(default=None, ge=0, le=5000)  # annual cost of borrowing shares to short
    riskFreeRate: float | None = Field(default=None, ge=0, le=0.25)


TUNABLE_STRATEGY = Literal["sma-crossover", "mean-reversion", "trend-follow", "regime-switch", "ml-logistic"]


class WalkForwardPayload(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    strategyId: TUNABLE_STRATEGY = "sma-crossover"
    slippageBps: float | None = Field(default=None, ge=0, le=100)
    execution: Literal["close", "next_open"] | None = None
    allowShort: bool = False


class RobustnessPayload(TrainingPayload):
    paths: int = Field(default=500, ge=100, le=2000)  # Monte Carlo resamples


class BasketPayload(BaseModel):
    symbols: list[str] = Field(min_length=2, max_length=30)
    strategyId: str = Field(default="sma-crossover", max_length=60)
    parameters: dict[str, float] = Field(default_factory=dict)
    rules: StrategyRules | None = None
    weighting: Literal["equal", "inverse-vol", "risk-parity"] = "equal"
    rebalance: Literal["weekly", "monthly", "quarterly"] = "monthly"
    topN: int = Field(default=0, ge=0, le=30)  # 0 = hold every stock; otherwise the top N by 6-month momentum
    allowShort: bool = False
    slippageBps: float | None = Field(default=None, ge=0, le=100)
    riskFreeRate: float | None = Field(default=None, ge=0, le=0.25)

    @model_validator(mode="after")
    def check(self) -> BasketPayload:
        cleaned = list(dict.fromkeys(s.strip().upper() for s in self.symbols if s.strip()))
        if len(cleaned) < 2:
            raise ValueError("A basket needs at least two different symbols")
        if any(len(s) > 20 for s in cleaned):
            raise ValueError("Symbols are at most 20 characters")
        self.symbols = cleaned
        return self


class PineExportPayload(BaseModel):
    name: str = Field(default="Custom strategy", max_length=60)
    rules: StrategyRules


class AlertSettings(BaseModel):
    email: bool = False  # email the signed-in address when a simulation's strategy signals a trade
    telegramChatId: str | None = Field(default=None, max_length=40, pattern=r"^-?\d+$")


class PredictionPayload(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    # Optional: the strategy to evaluate. Without it, the user's last trained strategy for the symbol is used.
    strategyId: str | None = Field(default=None, max_length=60)
    parameters: dict[str, float] | None = None
    rules: StrategyRules | None = None
    allowShort: bool = False


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatHistoryItem] = Field(default_factory=list)
