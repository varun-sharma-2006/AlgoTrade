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
  a stock and a strategy, or design your own, and it replays two years of real prices trade by trade, with fees,
  and always shows the result next to simply buying and holding."
- **The problem:** "Most backtests you see online flatter the strategy. Our own first version did too: it
  reported the stock's buy-and-hold return as the strategy's return and used a hard-coded win rate. We rebuilt it
  so the numbers are real, and very often the honest answer is that the strategy lost to buy & hold."
- **Stack:** "Python and FastAPI on the backend, React and TypeScript on the frontend, MongoDB Atlas for storage,
  deployed on Vercel, with 86 automated tests running in GitHub Actions."

## 2. Demo (2:00 – 6:00)

Open https://algo-trade-mu.vercel.app and sign in with Google. Suggested order:

1. **Strategy lab:** backtest the SMA crossover on AAPL. Point at *strategy return vs buy & hold*, the equity curve
   and the trade log. Press *Today's signal*.
2. **Strategy Builder:** load the "RSI dip buyer" preset, show the plain-English summary, change a number, run the
   backtest, save it.
3. **Simulations → Portfolio:** create a simulation backdated a few months using the saved strategy, then open
   Portfolio and show its value, P&L and the "vs buy & hold" column.
4. **Trading copilot:** ask "Which stock should I buy and why?" and show that the answer cites live numbers.
5. **Live markets:** search a ticker and switch the chart range.

Tip: open the site once before the interview so the serverless API is warm.

## 3. Technical deep dive (6:00 – 10:00)

- **No look-ahead bias:** "Each strategy decides its position at a day's close using only data up to that close,
  and the position is held from the next day. Every entry and exit pays a 10 basis-point fee."
- **Metrics:** "Sharpe is the mean daily return over its standard deviation, annualised with √252. Drawdown is
  the worst peak-to-trough fall of the equity curve. Win rate counts only closed trades."
- **Rule engine:** "The Strategy Builder turns rules like *RSI(14) is below 30* or *SMA(50) crosses above
  SMA(200)* into daily positions: all buy rules must hold, any sell rule or the stop-loss/take-profit exits.
  Requests are validated with Pydantic, so a rule can't be missing a period."
- **Storage abstraction:** "MongoDB and an in-memory store implement the same interface, so routes don't care
  which is active, and our API tests run against both."
- **Serverless:** "On Vercel the API runs as serverless functions, which forced a few real fixes; see the
  challenges below."

## 4. Challenges and how they were solved (10:00 – 13:00)

Pick one or two:

- **Fake metrics → honest backtester.** The original numbers weren't real. We replaced them with a trade-by-trade
  simulation and a buy-and-hold comparison.
- **Login loop on Vercel.** Parallel serverless instances don't share memory, so a session created on one
  instance was unknown to the next. We switched to signed session tokens that any instance can verify.
- **Bundle too large to deploy.** `yfinance` pulled in pandas and numpy, putting the bundle around 240 MB, right
  at Vercel's limit. We replaced it with direct calls to Yahoo's JSON API: about 49 MB.
- **AI rate limits.** The free Gemini tier frequently returned "quota exceeded" or "high demand". The app now
  tries several models within a time budget and then answers from a rule-based analyst built on live data.
- **Security gap after adding Google sign-in.** Browsers with an old demo session could skip the new login. We
  made the server reject any non-Google session once Google sign-in is enabled.

## 5. Wrap-up (13:00 – 15:00)

- **What we learned:** measure against a baseline, design for external services failing, and treat deployment,
  auth and tests as part of the product.
- **Next steps:** walk-forward testing, slippage modelling, more indicators (MACD, ATR, volume) and position
  sizing.

---

## Likely questions (have an answer ready)

- *Why compare with buy & hold?* It's the free alternative. A strategy that can't beat it after fees isn't adding
  value.
- *How do you avoid look-ahead bias?* Decide at the close with data up to that close; trade from the next day.
- *Why is your Sharpe ratio annualised with √252?* There are about 252 trading days a year, and volatility
  scales with the square root of time.
- *Why MongoDB?* Simulations and saved strategies have varying shapes (different parameters, nested rules), which
  suits documents; Atlas also has a free tier with no card.
- *How is sign-in secure?* The server verifies Google's ID token signature, audience, expiry and verified email;
  the browser is never trusted.
- *What would you change with more time?* See "Next steps" above, and be honest about the limitations listed in
  PROJECT_EXPLANATION.md.

**Don't claim:** that the chatbot creates simulations, 5 years of data, shared frontend/backend types, or real
trading. None of these are true.
