# Interview Guide: Algo Trade Simulator (10–15 minutes)

A structured way to present the project. For the full technical detail, see
[PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).

> **Team project.** Algo Trade Simulator was built by Varun Sharma and Yashika Garg. In interviews, say "we"
> for the project and be specific about **your own** part. Fill in the section below before you use this guide.

## 0. Your part (fill this in)

- What I personally built: `...`
- A decision I made and why: `...`
- A bug I fixed myself: `...`

Interviewers almost always ask "what did *you* do?" on team projects. A clear, honest answer here matters more
than anything else in this guide.

---

## 1. Introduction (0:00 – 2:00)

- **What it is:** "Algo Trade Simulator is a full-stack web app for testing trading strategies honestly. You pick
  a stock and a strategy, a classic rule, a machine-learning model or your own rules, and it replays two years of
  real prices trade by trade, with fees and slippage, against buy & hold and the S&P 500."
- **The problem:** "Most backtests you see online flatter the strategy. Our own first version did too: it
  reported the stock's buy-and-hold return as the strategy's return and used a hard-coded win rate. We rebuilt it
  so the numbers are real, added walk-forward testing to expose curve fitting, and very often the honest answer
  is that the strategy lost to buy & hold."
- **Stack:** "Python and FastAPI on the backend, React and TypeScript on the frontend, MongoDB Atlas for storage,
  Gemini with function calling for the copilot, deployed on Vercel, with 107 automated tests in GitHub Actions."

## 2. Demo (2:00 – 6:00)

Open https://algo-trade-mu.vercel.app and sign in with Google. Suggested order:

1. **Strategy lab:** backtest the SMA crossover on AAPL. Point at the growth-of-$1 and drawdown charts, then the
   risk table: strategy vs buy & hold vs the S&P 500 on Sharpe, Sortino and drawdown.
2. **Walk-forward test:** press *Walk-forward test*. Show the tuned vs out-of-sample returns and the fold table.
   "Tuned on the past it looked like 24% a year; on data it had never seen it made about 10%."
3. **Machine learning:** switch to the ML strategy on NVDA. Show accuracy vs the always-up baseline, the AUC and
   the feature weights, and be upfront that it has no real edge.
4. **Trading copilot:** ask "Compare strategies on NVDA" and show the card with the real numbers, then "Which stock
   should I buy and why?".
5. **Strategy Builder → Portfolio:** load a preset, backtest it, then show a backdated simulation's value and
   the "vs buy & hold" column on the Portfolio page.

Tip: open the site once before the interview so the serverless API is warm.

## 3. Technical deep dive (6:00 – 10:00)

- **No look-ahead bias:** "Each strategy decides its position at a day's close using only data up to that close,
  and the position is held from the next day. Every entry and exit pays a 10 basis-point fee plus slippage."
- **Risk metrics:** "Sharpe is mean daily return over its standard deviation, annualised with √252. Sortino only
  counts downside volatility. Beta is the regression slope of the strategy's daily returns on the S&P 500's."
- **Walk-forward:** "Each quarter we backtest a parameter grid on the previous year, pick the best Sharpe, and
  trade it on the next quarter, which it has never seen. The gap between tuned and out-of-sample returns is
  curve fitting."
- **Machine learning:** "Logistic regression on 8 price features, refitted monthly on a rolling window. The label
  for day *i* is only known on day *i+1*, so the model used on day *t* is trained on days up to *t−1*, and scaling
  is fitted on the training window only. We wrote it with Newton's method in plain Python to keep the serverless
  bundle small. A test changes future prices and checks that past predictions don't move."
- **Copilot actions:** "Gemini gets JSON-schema declarations of our backtest, compare, walk-forward and
  create-simulation functions. When it returns a function call we run it, send the result back, and it explains
  the numbers. Results are memoised per request, so a retry never creates a simulation twice."
- **Storage abstraction:** "MongoDB and an in-memory store implement the same interface, so routes don't care
  which is active, and our API tests run against both."

## 4. Challenges and how they were solved (10:00 – 13:00)

Pick one or two:

- **Fake metrics → honest backtester.** The original numbers weren't real. We replaced them with a trade-by-trade
  simulation and a buy-and-hold comparison.
- **Login loop on Vercel.** Parallel serverless instances don't share memory, so a session created on one
  instance was unknown to the next. We switched to signed session tokens that any instance can verify.
- **Bundle too large to deploy.** `yfinance` pulled in pandas and numpy, putting the bundle around 240 MB, right
  at Vercel's limit. We replaced it with direct calls to Yahoo's JSON API, about 49 MB. That is also why the ML
  model is written without numpy or scikit-learn.
- **AI rate limits.** The free Gemini tier frequently returned "quota exceeded" or "high demand". The app tries
  several models within a time budget, then handles backtest requests itself and falls back to a rule-based
  analyst built on live data.
- **A test caught a walk-forward bug.** The stitched out-of-sample curve was shifted by one day inside the first
  fold. A test asserting that every date appears exactly once caught it before release.
- **Security gap after adding Google sign-in.** Browsers with an old demo session could skip the new login. We
  made the server reject any non-Google session once Google sign-in is enabled.

## 5. Wrap-up (13:00 – 15:00)

- **What we learned:** measure against a baseline, test out of sample, design for external services failing, and
  treat deployment, auth and tests as part of the product.
- **Next steps:** more indicators and ML features, other model types, position sizing, and slippage models based
  on volume.

---

## Likely questions (have an answer ready)

- *Why compare with buy & hold?* It's the free alternative. A strategy that can't beat it after costs isn't adding
  value.
- *How do you avoid look-ahead bias?* Decide at the close with data up to that close; trade from the next day. The
  ML model and walk-forward folds only ever train on the past, and tests check this.
- *Your ML model is about 50% accurate. Isn't that bad?* It's the honest result: daily direction is close to
  random from price data alone. The point is the pipeline, time-aware training, a baseline and a real backtest,
  which shows the model has no edge instead of hiding it.
- *Why logistic regression and not a neural network?* It's interpretable (we show the feature weights), fast to
  retrain monthly, and a fair first baseline. A more complex model would need more features to have anything to
  learn.
- *Sharpe vs Sortino?* Sharpe penalises all volatility; Sortino only the downside, which is what investors mind.
- *What does a beta of 0.2 mean?* The strategy moves about 0.2% for each 1% move in the S&P 500, usually because
  it's in cash much of the time.
- *Why is Sharpe annualised with √252?* There are about 252 trading days a year, and volatility scales with the
  square root of time.
- *Why MongoDB?* Simulations and saved strategies have varying shapes (different parameters, nested rules), which
  suits documents; Atlas also has a free tier with no card.
- *How is sign-in secure?* The server verifies Google's ID token signature, audience, expiry and verified email;
  the browser is never trusted.

**Don't claim:** 5 years of backtest results (it reports 2 years; 5 are fetched for warm-up and training), that
the ML model predicts the market, shared frontend/backend types, or real trading. None of these are true.
