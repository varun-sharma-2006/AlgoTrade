"""Rule-based trading analyst used by the chatbot.

It answers from real price history, so the chatbot still gives a useful, specific reply
when the Gemini API is rate-limited or overloaded. The same analysis is also handed to
Gemini as context so AI answers are grounded in current numbers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

CloseLoader = Callable[[Sequence[str]], dict[str, list[float]]]

US_UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM", "LLY", "COST"]
INDIA_UNIVERSE = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS"]
US_BENCHMARK = "SPY"
INDIA_BENCHMARK = "^NSEI"

COMPANY_TICKERS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "jpmorgan": "JPM",
    "visa": "V",
    "exxon": "XOM",
    "walmart": "WMT",
    "costco": "COST",
    "amd": "AMD",
    "intel": "INTC",
    "coca cola": "KO",
    "coca-cola": "KO",
    "disney": "DIS",
    "nike": "NKE",
    "boeing": "BA",
    "berkshire": "BRK-B",
    "eli lilly": "LLY",
    "palantir": "PLTR",
    "uber": "UBER",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "adobe": "ADBE",
    "broadcom": "AVGO",
    "qualcomm": "QCOM",
    "paypal": "PYPL",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "airtel": "BHARTIARTL.NS",
    "itc": "ITC.NS",
    "wipro": "WIPRO.NS",
    "tata motors": "TATAMOTORS.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "larsen": "LT.NS",
    "adani": "ADANIENT.NS",
    "zomato": "ZOMATO.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "sp500": "SPY",
    "s&p 500": "SPY",
    "s&p": "SPY",
    "nasdaq": "QQQ",
    "nifty": "^NSEI",
    "sensex": "^BSESN",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "gold": "GLD",
}
TICKER_RE = re.compile(r"(?<![\w$])\$?([A-Z]{1,5}(?:[.-][A-Z]{1,2})?)(?![\w])")
NOT_TICKERS = {
    "I",
    "A",
    "AI",
    "SMA",
    "EMA",
    "RSI",
    "MACD",
    "ETF",
    "ETFS",
    "USD",
    "INR",
    "OK",
    "US",
    "USA",
    "IPO",
    "CEO",
    "PE",
    "EPS",
    "ATR",
    "ROI",
    "YTD",
    "IT",
    "AM",
    "PM",
    "UK",
    "EU",
    "GDP",
    "FD",
    "SIP",
    "NSE",
    "BSE",
    "NYSE",
    "HI",
    "VS",
    "OR",
    "AND",
    "THE",
    "TO",
    "ME",
    "MY",
    "IS",
    "BUY",
    "SELL",
    "HOLD",
    "WHY",
    "HOW",
    "WHAT",
}

RECOMMEND_RE = re.compile(
    r"\b(buy|invest|investing|recommend|suggest|pick|best|top|should i|which (stock|share|one)|portfolio|"
    r"good stock|strategy|opportunit|where to put|what to trade)\b",
    re.IGNORECASE,
)
INDIA_RE = re.compile(r"\b(india|indian|nse|bse|nifty|sensex|rupee|rupees|inr|rs\.?|₹)", re.IGNORECASE)
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|yo|namaste|good (morning|afternoon|evening))\b[\s!.?]*$", re.IGNORECASE)


@dataclass
class Snapshot:
    symbol: str
    price: float
    change_1d: float
    return_1m: float
    return_3m: float
    return_6m: float
    sma20: float
    sma50: float
    rsi14: float
    volatility: float  # annualised
    drawdown: float  # from 6-month high, as a fraction
    score: float = 0.0

    @property
    def uptrend(self) -> bool:
        return self.price > self.sma20 > self.sma50

    @property
    def downtrend(self) -> bool:
        return self.price < self.sma20 < self.sma50

    @property
    def trend_label(self) -> str:
        if self.uptrend:
            return "uptrend (price > 20-day SMA > 50-day SMA)"
        if self.downtrend:
            return "downtrend (price < 20-day SMA < 50-day SMA)"
        if self.price > self.sma50:
            return "mixed, holding above the 50-day SMA"
        return "mixed, below the 50-day SMA"

    @property
    def signal(self) -> str:
        if self.uptrend and self.rsi14 < 70 and self.return_3m > 0:
            return "BUY candidate"
        if self.uptrend and self.rsi14 >= 70:
            return "WAIT for a pullback (overbought)"
        if self.downtrend:
            return "AVOID for now"
        if self.rsi14 < 30:
            return "WATCH for a reversal (oversold)"
        return "HOLD / neutral"


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _ret(closes: list[float], days: int) -> float:
    if len(closes) <= days:
        days = len(closes) - 1
    if days <= 0 or not closes[-days - 1]:
        return 0.0
    return closes[-1] / closes[-days - 1] - 1


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains, losses = 0.0, 0.0
    for prev, cur in zip(closes[-period - 1 : -1], closes[-period:], strict=True):
        delta = cur - prev
        gains += max(delta, 0)
        losses += max(-delta, 0)
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100 - 100 / (1 + rs)


def build_snapshot(symbol: str, closes: list[float]) -> Snapshot | None:
    closes = [c for c in closes if c and not math.isnan(c)]
    if len(closes) < 30:
        return None
    returns = [b / a - 1 for a, b in zip(closes[:-1], closes[1:], strict=True) if a]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
    window50 = closes[-50:]
    snap = Snapshot(
        symbol=symbol,
        price=closes[-1],
        change_1d=_ret(closes, 1),
        return_1m=_ret(closes, 21),
        return_3m=_ret(closes, 63),
        return_6m=_ret(closes, len(closes) - 1),
        sma20=sum(closes[-20:]) / 20,
        sma50=sum(window50) / len(window50),
        rsi14=_rsi(closes),
        volatility=math.sqrt(variance) * math.sqrt(252),
        drawdown=closes[-1] / max(closes) - 1,
    )
    # Momentum + trend, penalise overbought and very volatile names.
    score = 0.5 * snap.return_3m + 0.3 * snap.return_1m + 0.2 * snap.return_6m
    score += 0.05 if snap.uptrend else (-0.05 if snap.downtrend else 0)
    if snap.rsi14 > 70:
        score -= (snap.rsi14 - 70) / 200
    score -= max(snap.volatility - 0.35, 0) * 0.15
    snap.score = score
    return snap


def extract_symbols(message: str) -> list[str]:
    found: list[str] = []
    lowered = message.lower()
    for name, ticker in sorted(COMPANY_TICKERS.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", lowered) and ticker not in found:
            found.append(ticker)
    for match in TICKER_RE.findall(message):
        if match not in NOT_TICKERS and match not in found:
            found.append(match)
    return found[:8]


def parse_budget(message: str) -> float | None:
    candidates = []
    for m in re.finditer(
        r"(\$|₹|rs\.?\s*|inr\s*|usd\s*)?(\d[\d,]*(?:\.\d+)?)\s*(k|lakhs?|lacs?|crores?|thousand|million)?\s*"
        r"(dollars?|rupees?|rs\b|inr\b|usd\b)?(\s*(days?|weeks?|months?|years?|yrs?|%))?",
        message,
        re.IGNORECASE,
    ):
        if m.group(5):  # a duration or percentage, not money
            continue
        value = float(m.group(2).replace(",", ""))
        unit = (m.group(3) or "").lower()
        value *= {"k": 1e3, "thousand": 1e3, "million": 1e6}.get(unit, 1)
        if unit.startswith("lakh") or unit.startswith("lac"):
            value *= 1e5
        if unit.startswith("crore"):
            value *= 1e7
        if m.group(1) or m.group(4) or unit or value >= 100:
            candidates.append(value)
    return max(candidates) if candidates else None


def parse_horizon_days(message: str) -> int | None:
    m = re.search(r"(\d+)\s*(day|week|month|year|yr)s?", message, re.IGNORECASE)
    if not m:
        return None
    n = int(m.group(1))
    return n * {"day": 1, "week": 7, "month": 30, "year": 365, "yr": 365}[m.group(2).lower()]


def describe(snap: Snapshot) -> list[str]:
    reasons = []
    reasons.append(f"Trend: {snap.trend_label}.")
    reasons.append(
        f"Momentum: {_pct(snap.return_1m)} over 1 month, {_pct(snap.return_3m)} over 3 months, "
        f"{_pct(snap.return_6m)} over 6 months."
    )
    if snap.rsi14 >= 70:
        rsi_note = "overbought, so chasing here carries pullback risk"
    elif snap.rsi14 <= 30:
        rsi_note = "oversold, so a bounce is possible but the trend is weak"
    else:
        rsi_note = "not stretched in either direction"
    reasons.append(f"RSI(14) {snap.rsi14:.0f}: {rsi_note}.")
    reasons.append(
        f"Risk: annualised volatility {snap.volatility * 100:.0f}%, currently {abs(snap.drawdown) * 100:.1f}% below its 6-month high."
    )
    return reasons


def trade_plan(snap: Snapshot, budget: float | None, allocation: float | None = None) -> list[str]:
    # Stop just under the 50-day SMA, but never more than 8% or less than 3% below entry.
    stop = snap.sma50 * 0.99 if snap.sma50 < snap.price else snap.price * 0.92
    stop = min(max(stop, snap.price * 0.92), snap.price * 0.97)
    risk = snap.price - stop
    target = snap.price + 2 * risk
    entry = (
        f"Entry: scale in near {snap.price:.2f}, or wait for a dip toward the 20-day SMA ({snap.sma20:.2f})"
        if snap.rsi14 < 70
        else f"Entry: wait for a pullback toward the 20-day SMA ({snap.sma20:.2f}) rather than buying at {snap.price:.2f}"
    )
    lines = [
        entry + ".",
        f"Stop-loss: about {stop:.2f} ({(stop / snap.price - 1) * 100:.1f}%). Exit if it closes below that.",
        f"Target: about {target:.2f} (2:1 reward-to-risk). Trail the stop up as the 20-day SMA rises.",
    ]
    if budget:
        allocation = allocation or budget
        max_loss = budget * 0.01
        qty = min(max_loss / risk if risk > 0 else 0, allocation / snap.price)
        lines.append(
            f"Sizing for your {budget:,.0f}: put about {qty * snap.price:,.0f} here (~{qty:.2f} shares). "
            f"If the stop is hit you lose ~{qty * risk:,.0f}, about 1% of your capital. Keep the rest in cash "
            "or a broad index fund until another setup appears."
        )
    return lines


def recommend(message: str, load: CloseLoader) -> dict[str, object]:
    mentioned = extract_symbols(message)
    india = bool(INDIA_RE.search(message)) or any(s.endswith(".NS") for s in mentioned)
    universe = mentioned if len(mentioned) >= 2 else (INDIA_UNIVERSE if india else US_UNIVERSE)
    benchmark = INDIA_BENCHMARK if india else US_BENCHMARK
    data = load(list(universe) + [benchmark])
    snaps = [s for s in (build_snapshot(sym, data.get(sym, [])) for sym in universe) if s]
    if not snaps:
        return general_reply(message)
    snaps.sort(key=lambda s: s.score, reverse=True)
    bench = build_snapshot(benchmark, data.get(benchmark, []))
    budget = parse_budget(message)
    horizon = parse_horizon_days(message)

    lines = [
        f"Here is a momentum + trend screen of {len(snaps)} {'Indian' if india else 'US'} large caps, "
        f"using live prices and the last 6 months of daily data."
    ]
    if bench:
        lines.append(f"Market backdrop ({benchmark}): {bench.trend_label}, {_pct(bench.return_3m)} over 3 months.")
    lines.append("")
    lines.append("Ranking (best setup first):")
    for rank, s in enumerate(snaps, 1):
        lines.append(f"{rank}. {s.symbol} at {s.price:.2f}: 3M {_pct(s.return_3m)}, RSI {s.rsi14:.0f}, {s.signal}")

    picks = [s for s in snaps if s.signal == "BUY candidate"][:2]
    for pick in picks:
        lines.append("")
        lines.append(f"Top pick: {pick.symbol}. Why:")
        lines.extend(f"- {r}" for r in describe(pick))
        lines.append("Strategy (SMA trend-following swing trade):")
        lines.extend(f"- {p}" for p in trade_plan(pick, budget, budget / len(picks) if budget else None))

    if not picks:
        best = snaps[0]
        lines.append("")
        lines.append(
            "No stock passes the buy rules right now (price above rising 20- and 50-day SMAs, positive 3-month "
            "momentum, RSI under 70). The disciplined move is to wait rather than force a trade."
        )
        lines.append(f"Closest to a setup: {best.symbol}.")
        lines.extend(f"- {r}" for r in describe(best))
        lines.append(
            f"- Buy trigger to watch: a close above both the 20-day SMA ({best.sma20:.2f}) and the 50-day SMA "
            f"({best.sma50:.2f}). Until then, keep capital in cash or an index fund."
        )

    if bench and bench.downtrend:
        lines.append("")
        lines.append(
            f"Caution: the broad market ({benchmark}) is in a downtrend, so most stocks fall with it. "
            "Use smaller positions and tighter stops."
        )

    avoid = [s for s in snaps if s.downtrend][:3]
    if avoid:
        lines.append("")
        lines.append(
            "Avoid for now: "
            + ", ".join(f"{s.symbol} (below its 20- and 50-day SMAs, RSI {s.rsi14:.0f})" for s in avoid)
            + ". Wait until the price reclaims its 50-day SMA."
        )

    lines.append("")
    if horizon is not None and horizon <= 10:
        lines.append(
            f"For a {horizon}-day horizon, keep stops tight and size small: short-term moves are mostly noise."
        )
    lines.append(
        "How to use this in the app: open Simulations, run a backtest on the top pick in the Strategy lab to see how "
        "the strategy compares with buy & hold, then press Today's signal before committing capital."
    )
    return {"reply": "\n".join(lines), "citations": [_yahoo(s.symbol) for s in picks]}


def analyse(symbols: list[str], message: str, load: CloseLoader) -> dict[str, object]:
    data = load(symbols)
    snaps = [s for s in (build_snapshot(sym, data.get(sym, [])) for sym in symbols) if s]
    missing = [sym for sym in symbols if sym not in {s.symbol for s in snaps}]
    if not snaps:
        return general_reply(message, note=f"I couldn't find price history for {', '.join(symbols)}.")
    budget = parse_budget(message)
    lines: list[str] = []
    for s in snaps:
        if lines:
            lines.append("")
        lines.append(f"{s.symbol}: {s.price:.2f} ({_pct(s.change_1d)} today). Signal: {s.signal}.")
        lines.extend(f"- {r}" for r in describe(s))
        if s.signal in {"BUY candidate", "WAIT for a pullback (overbought)"}:
            lines.append("Possible plan:")
            lines.extend(f"- {p}" for p in trade_plan(s, budget))
        elif s.downtrend:
            lines.append(
                f"- Plan: stay out until it closes back above the 50-day SMA ({s.sma50:.2f}). "
                "Buying a downtrend is catching a falling knife."
            )
        else:
            lines.append(
                f"- Plan: a close above the 20-day SMA ({s.sma20:.2f}) with the 20-day SMA turning up would be "
                "the trigger to consider buying."
            )
    if len(snaps) > 1:
        best = max(snaps, key=lambda s: s.score)
        lines.append("")
        lines.append(f"Of these, {best.symbol} has the strongest trend and momentum setup right now.")
    if missing:
        lines.append("")
        lines.append(f"No data found for: {', '.join(missing)} (check the ticker; NSE stocks end in .NS, e.g. TCS.NS).")
    return {"reply": "\n".join(lines), "citations": [_yahoo(s.symbol) for s in snaps]}


CONCEPTS: list[tuple[tuple[str, ...], str]] = [
    (
        ("golden cross", "death cross"),
        "A golden cross is when a short-term moving average (commonly the 50-day) crosses above a long-term one "
        "(commonly the 200-day). It signals that recent momentum is turning bullish. A death cross is the opposite: "
        "the short average falls below the long one, signalling weakness. Both lag price, so traders confirm them "
        "with volume, the broader market trend, or RSI, and always use a stop-loss.",
    ),
    (
        ("sma", "moving average", "crossover", "ema"),
        "A simple moving average (SMA) is the average closing price over N days. It smooths out noise so you can see "
        "the trend. An SMA crossover strategy buys when a short SMA (e.g. 20-day) crosses above a long SMA (e.g. 60-day) "
        "and sells when it crosses back below. It captures big trends but gets whipsawed in sideways markets. An EMA "
        "weights recent prices more heavily, so it reacts faster. You can backtest this in the Strategy lab on the "
        "Simulations page.",
    ),
    (
        ("rsi", "relative strength", "overbought", "oversold"),
        "RSI (Relative Strength Index) measures the speed of recent gains vs losses on a 0-100 scale, usually over "
        "14 days. Above 70 is 'overbought' (the move may be stretched) and below 30 is 'oversold'. In strong trends RSI "
        "can stay overbought for weeks, so use it to time entries within a trend (buy dips when RSI cools to 40-50 in "
        "an uptrend) rather than as a standalone sell signal.",
    ),
    (
        ("macd",),
        "MACD is the 12-day EMA minus the 26-day EMA, with a 9-day EMA of that line as the 'signal line'. A MACD cross "
        "above its signal line suggests building upside momentum, and a cross below suggests fading momentum. The "
        "histogram shows the gap between the two. It works best as confirmation of a trend you already see on price.",
    ),
    (
        ("bollinger",),
        "Bollinger Bands are a 20-day SMA with bands 2 standard deviations above and below. Price touching the lower "
        "band in a range-bound market often mean-reverts toward the middle. A 'squeeze' (bands narrowing) often comes "
        "before a big move. In strong trends, price can 'walk the band', so don't fade it blindly.",
    ),
    (
        ("mean reversion",),
        "Mean reversion bets that price returns toward its average after an extreme move. Typical rules: buy when price "
        "is far below its 20-day SMA or RSI < 30 in a range-bound stock, and exit at the average. It has a high win rate "
        "but suffers large losses when a stock keeps trending, so position size small and use hard stops.",
    ),
    (
        ("momentum", "trend following", "breakout", "donchian"),
        "Trend following / momentum buys strength: stocks making new highs or trading above rising moving averages. A "
        "classic breakout rule is buying a close above the 20-day high (Donchian channel) and exiting on a close below "
        "the 10-day low. Win rates are often only 35-45%, but winners are much bigger than losers. Discipline with stops "
        "is what makes it work.",
    ),
    (
        ("stop loss", "stop-loss", "stoploss", "trailing stop"),
        "A stop-loss is a pre-set exit price that caps your loss if a trade goes wrong. Common placements: below a "
        "recent swing low, below the 50-day SMA, or 1.5-2x ATR under your entry. A trailing stop moves up as price rises "
        "and locks in gains. Decide the stop before you enter, and size the position so hitting it costs at most 1-2% "
        "of your capital.",
    ),
    (
        (
            "position size",
            "position sizing",
            "how much to invest",
            "how many shares",
            "risk per trade",
            "money management",
        ),
        "Position sizing rule of thumb: risk 1-2% of your account per trade. Shares = (account x risk %) / (entry - stop). "
        "For example, a 10,000 account risking 1% (100) with a 5 stop distance gives 20 shares. This keeps any single "
        "loss small, so a losing streak can't wipe you out.",
    ),
    (
        ("diversif",),
        "Diversification means spreading money across assets that don't move together: different sectors, geographies, "
        "and asset classes (stocks, bonds, gold). It reduces the damage any one position can do. For most people a "
        "low-cost index fund/ETF (e.g. SPY or a Nifty 50 fund) is the simplest diversified core, with individual stock "
        "trades as a smaller satellite.",
    ),
    (
        ("sharpe",),
        "The Sharpe ratio is (return - risk-free rate) / volatility. It tells you how much return you earned per unit of "
        "risk. Above 1 is good, above 2 is excellent. Compare strategies by Sharpe rather than raw return, because a high "
        "return with wild swings is often worse than a steadier, smaller one.",
    ),
    (
        ("drawdown",),
        "Max drawdown is the largest peak-to-trough fall in portfolio value. A 50% drawdown needs a 100% gain to recover, "
        "which is why limiting drawdowns matters more than maximising returns. The Strategy lab reports max drawdown for "
        "each backtest.",
    ),
    (
        ("backtest",),
        "Backtesting runs a strategy's rules on historical data to see how it would have performed. Watch out for "
        "overfitting (tuning parameters until the past looks perfect), look-ahead bias, and ignoring fees/slippage. "
        "Validate on a different period than the one you tuned on, and always compare with buy & hold. In this app, "
        "the Strategy lab on the Simulations page backtests SMA crossover, mean reversion and breakout strategies "
        "on 2 years of data with fees.",
    ),
    (
        ("volatility", "beta", "atr"),
        "Volatility measures how much a price swings, usually as the annualised standard deviation of daily returns. "
        "Beta measures sensitivity to the overall market (beta 1.5 means it tends to move 1.5x the market). ATR "
        "(Average True Range) is the average daily range, and is useful for placing stops outside normal noise.",
    ),
    (
        ("p/e", "pe ratio", "price to earnings", "valuation", "fundamental"),
        "The P/E ratio is share price divided by earnings per share. It shows how much you pay for 1 unit of profit. "
        "Compare it with the company's own history and its sector rather than across industries. Fast growers "
        "justify higher P/Es. Pair fundamentals (earnings growth, debt, margins) with technicals for timing.",
    ),
    (
        ("etf", "index fund", "mutual fund", "sip"),
        "An ETF or index fund holds a basket of stocks (e.g. the S&P 500 or Nifty 50) at very low cost. It gives you "
        "instant diversification. Investing a fixed amount monthly (an SIP / dollar-cost averaging) removes timing risk "
        "and is historically one of the most reliable ways to build long-term wealth.",
    ),
    (
        ("support", "resistance"),
        "Support is a price level where buying has repeatedly stopped declines. Resistance is where selling has capped "
        "rallies. A break above resistance on strong volume often turns it into new support. Traders place stops just "
        "below support and take profits near resistance.",
    ),
    (
        ("candlestick", "candle", "doji", "hammer", "engulfing"),
        "Each candlestick shows the open, high, low and close for a period. Green/red bodies show whether it closed up or "
        "down, and the wicks show the extremes. Patterns like a hammer (long lower wick after a decline) or bullish "
        "engulfing hint at reversals, but they are far more reliable at key support/resistance levels. See the Live "
        "data page for candle charts.",
    ),
    (
        ("day trading", "intraday", "swing trad", "scalp"),
        "Day trading opens and closes positions within a day, which demands speed, low fees and strict discipline, and "
        "most retail day traders lose money. Swing trading holds for days to weeks to capture a trend leg, which suits "
        "moving-average and RSI setups and needs less screen time. Long-term investing holds for years and relies on "
        "business growth.",
    ),
    (
        ("short sell", "shorting", "short selling"),
        "Short selling means borrowing shares to sell now and buying them back later, profiting if the price falls. "
        "Losses are theoretically unlimited, because price can keep rising, so shorts need tight stops and are best "
        "used in clear downtrends.",
    ),
    (
        ("option", "call", "put"),
        "Options give the right (not the obligation) to buy (call) or sell (put) at a set price before expiry. Buyers "
        "risk only the premium but lose value to time decay. Sellers collect premium but can face large losses. Beginners "
        "should learn on paper first. This simulator focuses on stocks.",
    ),
    (
        ("simulation", "this app", "how do i use", "how to use", "train", "predict", "portfolio", "builder"),
        "Using the simulator: 1) Simulations: invest paper money in a stock with a strategy, optionally backdated up "
        "to a year. 2) Portfolio: see every simulation's live value, profit and loss, allocation and how it compares "
        "with buy & hold. 3) Strategy lab (Simulations page): backtest SMA crossover, mean reversion, breakout or "
        "buy & hold, then press Today's signal. 4) Strategy builder: combine price, SMA, EMA and RSI rules with a "
        "stop-loss or take-profit, backtest them and save your own strategies. 5) Live markets: candle charts for any "
        "ticker. 6) Ask me about any stock (e.g. 'analyse NVDA') or 'which stock should I buy' for a ranked screen.",
    ),
]


def explain_concept(message: str) -> str | None:
    lowered = message.lower()
    hits = [text for keys, text in CONCEPTS if any(k in lowered for k in keys)]
    if not hits:
        return None
    return "\n\n".join(hits[:2])


def general_reply(message: str, note: str | None = None) -> dict[str, object]:
    lines = [note] if note else []
    if GREETING_RE.match(message):
        lines.append("Hi! I'm your trading copilot.")
    elif not note:
        lines.append(
            "The AI model is overloaded at this moment, so I can't give a free-form answer to that one. "
            "Please ask again in a minute. Meanwhile I can answer anything trading-related from live data."
        )
        lines.append("")
    lines += [
        "Here's what I can do right now:",
        "- 'Which stock should I buy?' gives a live ranked screen of large caps with reasons and a trade plan (add 'India' for NSE stocks).",
        "- 'Analyse AAPL' or 'compare TSLA and NVDA' gives trend, momentum, RSI, risk and a suggested entry/stop/target.",
        "- Concepts: SMA crossovers, RSI, MACD, stop-losses, position sizing, Sharpe, drawdown, ETFs, and more.",
        "- 'I have 5000 for 1 month, what should I buy?' does the same screen with position sizing for your budget.",
    ]
    return {"reply": "\n".join(lines), "citations": []}


def local_answer(message: str, load: CloseLoader) -> dict[str, object]:
    symbols = extract_symbols(message)
    wants_pick = bool(RECOMMEND_RE.search(message))
    concept = explain_concept(message)
    if (
        wants_pick
        and len(symbols) != 1
        and not (concept and not re.search(r"\b(stock|share|buy|invest)", message, re.I))
    ):
        result = recommend(message, load)
    elif symbols:
        result = analyse(symbols, message, load)
    elif concept:
        result = {"reply": concept, "citations": []}
    else:
        result = general_reply(message)
    if concept and result.get("reply") != concept and symbols:
        result["reply"] = f"{result['reply']}\n\nBackground: {concept}"
    return result


def market_context(message: str, load: CloseLoader) -> str | None:
    """Compact live-data summary handed to the LLM so its answer uses real numbers."""
    symbols = extract_symbols(message)
    if not symbols and not RECOMMEND_RE.search(message):
        return None
    india = bool(INDIA_RE.search(message)) or any(s.endswith(".NS") for s in symbols)
    universe = symbols or (INDIA_UNIVERSE if india else US_UNIVERSE)
    data = load(universe)
    snaps = [s for s in (build_snapshot(sym, data.get(sym, [])) for sym in universe) if s]
    if not snaps:
        return None
    snaps.sort(key=lambda s: s.score, reverse=True)
    rows = [
        f"{s.symbol}: price {s.price:.2f}, 1D {_pct(s.change_1d)}, 1M {_pct(s.return_1m)}, 3M {_pct(s.return_3m)}, "
        f"6M {_pct(s.return_6m)}, SMA20 {s.sma20:.2f}, SMA50 {s.sma50:.2f}, RSI14 {s.rsi14:.0f}, "
        f"vol {s.volatility * 100:.0f}%, {abs(s.drawdown) * 100:.1f}% off 6M high, rule-based signal: {s.signal}"
        for s in snaps
    ]
    return "Live market data (daily closes, last 6 months), ranked by trend/momentum score:\n" + "\n".join(rows)


def _yahoo(symbol: str) -> str:
    return f"https://finance.yahoo.com/quote/{symbol}"
