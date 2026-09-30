# Algo Trade Simulator: Project Explanation

Built by **Varun Sharma** and **Yashika Garg**.
Live demo: https://algo-trade-mu.vercel.app · Code: https://github.com/varun-sharma-2006/AlgoTrade

> Everything below describes the app as it is deployed today. If you change a feature, update this file too.

---

## 1. The elevator pitch

Algo Trade Simulator is a full-stack web app for testing trading ideas honestly. You pick a stock and a strategy
(or design your own), and it replays two years of real daily prices trade by trade, with fees and without
look-ahead bias, and always shows the result next to simply buying and holding. You can also run backdated paper
portfolios, study live candlestick charts, and ask an AI copilot about any stock.

**The one-line story:** our first version reported the stock's buy-and-hold return as the strategy's return and
used a hard-coded 55% win rate. It looked impressive and meant nothing. Rebuilding it to be honest is what the
project is really about.

---

## 2. Features (what a user can do)

| Area | What it does |
| --- | --- |
| **Sign in with Google** | One-click sign-in. The server verifies Google's ID token (signature, audience, verified email). Each user has a private workspace. |
| **Overview** | Portfolio stats, watchlist sparklines, recent simulations. |
| **Strategy lab** | Backtest 4 built-in strategies on 2 years of data: SMA crossover, Bollinger mean reversion, Donchian breakout, and buy & hold (the baseline). Shows strategy vs buy & hold return, annualised return, Sharpe ratio, max drawdown, win rate, trade count, average trade, time in market, an equity curve and a trade log. "Today's signal" says buy / hold / sell / wait for today. |
| **Strategy Builder** | Design your own strategy from rules: price, SMA, EMA, RSI or a number, compared with *is above*, *is below*, *crosses above*, *crosses below*. Up to 5 buy rules (all must be true) and 5 sell rules (any one sells), plus optional stop-loss and take-profit. The rules are shown back in plain English. Backtest, save up to 20, and use them in simulations. 4 presets included. |
| **Simulations** | Invest paper money in a stock with any strategy, optionally backdated up to a year. |
| **Portfolio** | Replays every simulation on real daily prices with its strategy's buy/sell rules and fees: current value, P&L, today's change, invested or in cash, allocation donut, value-over-time chart, and how each did against buy & hold. |
| **Trading copilot** | Google Gemini answers with live market data (trend vs moving averages, momentum, RSI, volatility) passed in as context. If Gemini is rate-limited or overloaded, a built-in rule-based analyst answers instead: a ranked stock screen, per-ticker analysis with entry/stop/target, or explanations of ~20 trading concepts. |
| **Live markets** | Quotes and candlestick charts for any stock, index or crypto, with ticker search. |
| **Visitors (admin only)** | Who signed in, when, how often and on which device, with CSV export. Only emails in `ADMIN_EMAILS` can open it; the server enforces this. |

What it does **not** do (don't claim these): real-money trading, intraday or tick-level backtests, short selling,
or letting the chatbot create simulations for you.

---

## 3. Architecture

```
React + TypeScript (Vite)  ──REST + bearer token──▶  FastAPI (Python)
                                                        ├── routes/      auth · market · simulations · analytics · portfolio · chat · admin
                                                        ├── strategies.py   built-in strategies + backtester
                                                        ├── rules.py        Strategy Builder rule engine
                                                        ├── portfolio.py    replays simulations, totals the portfolio
                                                        ├── advisor.py      rule-based analyst (chat fallback)
                                                        ├── gemini.py       Gemini client with model fallback
                                                        ├── market.py       Yahoo Finance chart/quote/search API
                                                        └── stores.py       MongoStore | InMemoryStore (same interface)
                                                                 │
                                        MongoDB Atlas ◀──────────┘       Yahoo Finance · Google Gemini · Google Sign-In
```

- **Frontend:** React 18 + TypeScript, built with Vite. A hand-written design system in CSS (no UI library).
  Types for API responses live in `client/src/types.ts` and mirror the backend's Pydantic models; they are
  maintained by hand, not generated.
- **Backend:** FastAPI split into routers, with Pydantic models validating every request (for example, a rule
  needs a period for SMA/EMA/RSI; a simulation can't start in the future or more than a year ago).
- **Storage:** MongoDB Atlas via the async `motor` driver in production; an in-memory store for local development.
  Both implement the same methods, so routes never branch on which one is active. Every query is scoped to the
  signed-in user.
- **Deployment:** Vercel. The React build is served as static files and the API runs as a Python serverless
  function under `/api`. Every push to `master` redeploys. GitHub Actions runs linting and all tests on every push.

---

## 4. How the backtester works (be ready to explain this)

1. Fetch 2 years of daily closes (split- and dividend-adjusted) from Yahoo Finance.
2. The strategy turns the price series into a **target position for each day**: 1 (invested) or 0 (cash),
   using only data up to that day's close.
3. The position decided at day *t*'s close is **held through day *t+1***. This is what prevents look-ahead bias:
   you can't trade on a close you haven't seen yet.
4. Every change of position pays a **10 basis-point fee** (0.1%), on both the entry and the exit.
5. Each round trip is recorded as a trade; metrics come from the simulated equity curve:
   - **Sharpe ratio:** mean daily return / standard deviation × √252 (risk-free rate 0).
   - **Max drawdown:** largest peak-to-trough fall of the equity curve.
   - **Win rate:** share of closed trades that made money.
   - **Exposure:** share of days the strategy was invested.
6. The result is always compared with **buy & hold** over the same period.

The **portfolio** uses the same rules, starting from each simulation's start date with the money in cash, so a
strategy that is already "long" buys at that day's close.

**Built-in strategies:**
- **SMA crossover:** invested while the short moving average is above the long one (default 20/60).
- **Mean reversion (Bollinger):** buy when the close drops below the lower band (20-day, 2 std devs), sell when it
  recovers to the middle band.
- **Breakout (Donchian):** buy a close above the prior 20-day high, sell on a close below the prior 10-day low.
- **Buy & hold:** the baseline.

---

## 5. Engineering decisions worth talking about

| Decision | Why |
| --- | --- |
| Honest backtester instead of headline numbers | The original returns and win rate were not real. Simulating trades with fees and next-day execution makes the numbers defensible. |
| One store interface, two implementations | Lets us develop and test without a database, and the same API tests run against both stores. |
| Signed session tokens for the in-memory store | Vercel runs several serverless instances that don't share memory; a stored session was unknown to other instances, which caused a login loop. Signed (HMAC) tokens can be verified anywhere. |
| Rebuild the MongoDB client when the event loop changes | Serverless runtimes can serve requests on a new event loop, and an async Mongo client can't be reused across loops. |
| Direct Yahoo Finance API calls instead of `yfinance` | Removing `yfinance` (and its pandas/numpy dependencies) shrank the serverless bundle from ~240 MB, at Vercel's limit, to ~49 MB. |
| Gemini with fallback | The free Gemini tier is often rate-limited. The app tries several models within a time budget, then answers from the rule-based analyst instead of showing an error. |
| Google sign-in verified on the server | The browser can't be trusted; the backend checks the token's signature, audience (our client ID), expiry and that the email is verified. Old non-Google sessions are rejected once Google sign-in is on. |
| Tests and CI | 86 automated tests (65 backend with pytest, 21 frontend with Vitest) run on every push via GitHub Actions. |

---

## 6. Limitations and next steps

- Daily data only; no intraday strategies.
- Long-only; no short selling or position sizing inside strategies.
- Fees are a flat 10 bps; no slippage or spread modelling.
- Backtests use a single 2-year window; walk-forward testing (tune on one period, test on the next) would make
  results more robust.
- Possible next steps: more indicators (MACD, ATR, volume), position sizing rules, walk-forward testing, and
  exporting backtest reports.
