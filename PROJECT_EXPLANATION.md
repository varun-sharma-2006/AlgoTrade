# Algo Trade Simulator: Project Explanation

Built by **Varun Sharma** and **Yashika Garg**.
Live demo: https://algo-trade-mu.vercel.app · Code: https://github.com/varun-sharma-2006/AlgoTrade

> Everything below describes the app as it is deployed today. If you change a feature, update this file too.

---

## 1. The elevator pitch

Algo Trade Simulator is a full-stack web app for testing trading ideas honestly. You pick a stock and a strategy
(a classic rule, a machine-learning model, or your own rules), and it replays two years of real daily prices trade
by trade, with fees, slippage and no look-ahead bias. Every result is compared with buy & hold and the S&P 500 on
return and risk. Walk-forward testing shows whether tuned settings survive on data they have never seen. You can
also run backdated paper portfolios, study live charts, and ask an AI copilot that can run all of this for you.

**The one-line story:** our first version reported the stock's buy-and-hold return as the strategy's return and
used a hard-coded 55% win rate. It looked impressive and meant nothing. Rebuilding it to be honest is what the
project is really about, and the honest answer is often that the strategy lost to buy & hold.

---

## 2. Features (what a user can do)

| Area | What it does |
| --- | --- |
| **Sign in with Google** | One-click sign-in. The server verifies Google's ID token (signature, audience, verified email). Each user has a private workspace. |
| **Overview** | Portfolio stats, watchlist sparklines, recent simulations. |
| **Strategy lab** | Backtest 5 built-in strategies on the last 2 years: SMA crossover, Bollinger mean reversion, Donchian breakout, a machine-learning model, and buy & hold (the baseline). Adjustable slippage on top of the fee. Shows a growth-of-$1 chart against buy & hold, a drawdown chart, a risk table (return, annualised return, volatility, Sharpe, Sortino, max drawdown, Calmar) for the strategy, buy & hold and the S&P 500, beta/alpha/correlation to the index, win rate, trades, time in market and the trade log. "Today's signal" says buy / hold / sell / wait for today. |
| **Walk-forward test** | For the tunable strategies: each quarter, every parameter set in a grid is backtested on the previous year and the best one (by Sharpe) trades the next quarter. Rolls forward over 5 years and shows tuned (in-sample) vs out-of-sample returns, a stitched out-of-sample equity curve and a fold-by-fold table. |
| **Machine-learning strategy** | Logistic regression predicting whether tomorrow closes higher, from 8 price features. Refitted monthly on past data only. Shows out-of-sample accuracy vs an always-up baseline, ROC-AUC, precision when long, and the model's feature weights. |
| **Strategy Builder** | Design your own strategy from rules: price, SMA, EMA, RSI or a number, compared with *is above*, *is below*, *crosses above*, *crosses below*. Up to 5 buy rules (all must be true) and 5 sell rules (any one sells), plus optional stop-loss and take-profit. Rules are shown back in plain English. Backtest, save up to 20, and use them in simulations. 4 presets included. |
| **Simulations** | Invest paper money in a stock with any strategy, optionally backdated up to a year. |
| **Portfolio** | Replays every simulation on real daily prices with its strategy's rules, fees and slippage: current value, P&L, today's change, invested or in cash, allocation donut, value-over-time chart, and how each did against buy & hold. |
| **Trading copilot** | Google Gemini with function calling. It can run the backtester, compare all strategies on a stock, run a walk-forward test, or create a simulation when asked, then explain the real numbers; results appear as cards in the chat. It also gets live market data (trend, momentum, RSI, volatility) as context. If Gemini is rate-limited, the same backtest/compare/walk-forward requests are recognised with pattern matching, and a rule-based analyst answers everything else. |
| **Live markets** | Quotes and candlestick charts for any stock, index or crypto, with ticker search. |
| **Visitors (admin only)** | Who signed in, when, how often and on which device, with CSV export. Only emails in `ADMIN_EMAILS` can open it; the server enforces this. |

What it does **not** do (don't claim these): real-money trading, intraday or tick-level backtests, short selling,
or predicting the market reliably (the ML model is an honest experiment, and usually shows about 50% accuracy).

---

## 3. Architecture

```
React + TypeScript (Vite)  ──REST + bearer token──▶  FastAPI (Python)
                                                        ├── routes/        auth · market · simulations · analytics · portfolio · chat · admin
                                                        ├── strategies.py  built-in strategies + backtester (shared simulate loop)
                                                        ├── risk.py        volatility, Sharpe, Sortino, drawdown, Calmar, beta/alpha
                                                        ├── ml.py          logistic regression strategy, time-aware retraining
                                                        ├── walkforward.py tune on a year, test on the next quarter
                                                        ├── rules.py       Strategy Builder rule engine
                                                        ├── portfolio.py   replays simulations, totals the portfolio
                                                        ├── copilot.py     chat tools (function calling) + offline parser
                                                        ├── advisor.py     rule-based analyst (chat fallback)
                                                        ├── gemini.py      Gemini client: model fallback + tool-call loop
                                                        ├── market.py      Yahoo Finance chart/quote/search API
                                                        └── stores.py      MongoStore | InMemoryStore (same interface)
                                                                 │
                                        MongoDB Atlas ◀──────────┘       Yahoo Finance · Google Gemini · Google Sign-In
```

- **Frontend:** React 18 + TypeScript, built with Vite. A hand-written design system in CSS and hand-written SVG
  charts (no UI or chart library). Types for API responses live in `client/src/types.ts` and mirror the backend's
  responses; they are maintained by hand, not generated.
- **Backend:** FastAPI split into routers, with Pydantic models validating every request (for example, a rule
  needs a period for SMA/EMA/RSI; a simulation can't start in the future or more than a year ago). The backtest,
  comparison and walk-forward functions are plain Python functions shared by the API routes and the copilot tools.
- **Storage:** MongoDB Atlas via the async `motor` driver in production; an in-memory store for local development.
  Both implement the same methods, so routes never branch on which one is active. Every query is scoped to the
  signed-in user.
- **Deployment:** Vercel. The React build is served as static files and the API runs as a Python serverless
  function under `/api`. Every push to `master` redeploys. GitHub Actions runs linting and all tests on every push.

---

## 4. How the backtester works (be ready to explain this)

1. Fetch 5 years of daily closes (split- and dividend-adjusted) from Yahoo Finance. Results are reported for the
   last 2 years; the earlier data only warms up indicators and trains the ML model.
2. The strategy turns the price series into a **target position for each day**: 1 (invested) or 0 (cash),
   using only data up to that day's close.
3. The money starts in cash at the beginning of the 2-year window. The position decided at day *t*'s close is
   **held through day *t+1***. This prevents look-ahead bias: you can't trade on a close you haven't seen yet.
4. Every change of position pays a **10 bps fee plus slippage** (5 bps by default, adjustable), on both the entry
   and the exit.
5. Metrics come from the simulated equity curve (all annualised with 252 trading days, risk-free rate 0):
   - **Volatility:** standard deviation of daily returns × √252.
   - **Sharpe ratio:** mean daily return / standard deviation × √252.
   - **Sortino ratio:** like Sharpe, but divides by downside deviation, so only losing days count as risk.
   - **Max drawdown:** largest peak-to-trough fall of the equity curve. **Calmar:** annualised return / max drawdown.
   - **Beta, alpha, correlation:** regression of the strategy's daily returns on the S&P 500's, over the dates both
     traded. Beta is market sensitivity; alpha is the annual return not explained by it.
   - **Win rate** (closed trades that made money), **average trade**, **time in market**.
6. The same statistics are computed for **buy & hold** of the stock and for the **S&P 500** over the same window.

The **portfolio** uses the same simulation loop from each simulation's start date, so a strategy that is already
"long" buys at that day's close.

**Built-in strategies:**
- **SMA crossover:** invested while the short moving average is above the long one (default 20/60).
- **Mean reversion (Bollinger):** buy when the close drops below the lower band (20-day, 2 std devs), sell when it
  recovers to the middle band.
- **Breakout (Donchian):** buy a close above the prior 20-day high, sell on a close below the prior 10-day low.
- **Machine learning (logistic regression):** see below.
- **Buy & hold:** the baseline.

---

## 5. The machine-learning strategy

- **Features** (for day *t*, from closes up to *t*): 1-, 5- and 20-day returns; price vs its 10- and 50-day
  averages; RSI(14); 20-day volatility; Bollinger z-score.
- **Label:** did the next day close higher? That label is only known one day later, so the model used on day *t*
  is trained on days up to *t−1*.
- **Training:** refitted every 21 trading days on a rolling window (default 504 days) with at least a year of
  labelled data. Features are standardised using the training window only. L2-regularised logistic regression is
  fitted with Newton's method: an 8-feature model, solved with a small Gaussian elimination.
- **Trading rule:** long while the predicted chance of a rise is at least the threshold (default 0.52).
- **Evaluation:** out-of-sample accuracy vs the always-guess-the-majority baseline, ROC-AUC (Mann-Whitney), share of
  up days while long, and, most importantly, the backtest against buy & hold after costs.
- **Typical result:** about 50% accuracy and AUC near 0.5, i.e. no reliable edge from price data alone. The app
  says so plainly, which is the honest and expected outcome, and a good interview talking point.
- **No numpy or scikit-learn:** pure Python keeps the serverless bundle small, and runs in about half a second
  on 5 years of data. A test checks that changing future prices never changes past predictions.

---

## 6. Walk-forward testing

- Grids: 15 SMA window pairs, 9 Bollinger settings, 4 breakout channels, 4 ML thresholds (the ML thresholds share
  one model run, since the probabilities don't depend on the threshold).
- Each fold: backtest every grid entry on the previous 252 trading days, pick the best Sharpe, trade it on the
  next 63 days. The position carries over between folds, like a trader who re-tunes every quarter.
- Output: tuned (in-sample) annualised return vs out-of-sample annualised return, out-of-sample Sharpe and
  drawdown vs buy & hold, quarters that beat buy & hold, and how often the most common settings were chosen.
- Example (AAPL, SMA crossover): +23.9% a year when tuned, +10.4% a year out of sample, vs +18.8% for buy & hold.
  That gap is curve fitting made visible.

---

## 7. Engineering decisions worth talking about

| Decision | Why |
| --- | --- |
| Honest backtester instead of headline numbers | The original returns and win rate were not real. Simulating trades with costs and next-day execution makes the numbers defensible. |
| One simulation loop for everything | The backtester, the walk-forward folds and the portfolio all use the same `simulate` function, so costs and timing are identical everywhere. |
| Report on 2 years, fetch 5 | Indicators are warmed up before the window starts, and the ML model has past data to learn from, without changing the period results are reported on. |
| One store interface, two implementations | Lets us develop and test without a database, and the same API tests run against both stores. |
| Signed session tokens for the in-memory store | Vercel runs several serverless instances that don't share memory; a stored session was unknown to other instances, which caused a login loop. Signed (HMAC) tokens can be verified anywhere. |
| Rebuild the MongoDB client when the event loop changes | Serverless runtimes can serve requests on a new event loop, and an async Mongo client can't be reused across loops. |
| Direct Yahoo Finance API calls instead of `yfinance` | Removing `yfinance` (and its pandas/numpy dependencies) shrank the serverless bundle from ~240 MB, at Vercel's limit, to ~49 MB. The ML model is pure Python for the same reason. |
| Gemini function calling instead of an agent framework | The tools are declared as JSON schemas and the loop is ~40 lines, so there is no LangChain dependency. Tool results are memoised per request, so a retry with another model never creates a simulation twice. |
| Offline fallback for the copilot | The free Gemini tier is often rate-limited. The app tries several models within a time budget, then recognises backtest/compare/walk-forward requests itself, and uses the rule-based analyst for everything else. |
| Google sign-in verified on the server | The browser can't be trusted; the backend checks the token's signature, audience (our client ID), expiry and that the email is verified. Old non-Google sessions are rejected once Google sign-in is on. |
| Tests and CI | 107 automated tests (80 backend with pytest, 27 frontend with Vitest) run on every push via GitHub Actions. They include checks that the ML model and walk-forward never use future data. |

---

## 8. Limitations and next steps

- Daily data only; no intraday strategies.
- Long-only; no short selling or position sizing inside strategies.
- Costs are a flat fee plus flat slippage in basis points; no spread or market-impact model.
- The ML model uses price features only, and one model type (logistic regression).
- Possible next steps: more indicators (MACD, ATR, volume) and ML features, other models (e.g. gradient boosting),
  position sizing, and exporting backtest reports.
