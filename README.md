# Algo Trade Simulator

**Algo Trade Simulator** is a full-stack algorithmic trading platform that empowers users to backtest, simulate, and analyze quantitative trading strategies using live market data. It combines a powerful Python and FastAPI backend with a responsive React and TypeScript frontend to deliver a seamless, feature-rich experience.

At the heart of the simulator is a robust quantitative analysis engine that allows users to train and evaluate trading models based on historical data. The platform's initial machine learning capabilities are built around Simple Moving Average (SMA) crossover strategies—a cornerstone of technical analysis. Users can select a stock, define their short and long-term windows, and train the model on five years of historical data. The backend then calculates key performance metrics such as total return, annualized return, win rate, and max drawdown, providing a clear picture of the strategy's viability.

Once a strategy is trained, the application can generate real-time buy, sell, or hold signals based on the latest market data and momentum indicators. This predictive feature allows users to see how their trained models would perform in current market conditions, bridging the gap between historical backtesting and live trading.

Enhancing the user experience is an AI-powered chatbot, driven by Google Gemini, which acts as an intelligent trading assistant. The chatbot can provide market insights, answer questions about trading strategies, and even help users create new simulations, offering a conversational interface for complex financial analysis.

## Key Features:

*   **Quantitative Strategy Backtesting:** Train and evaluate SMA crossover strategies on 5-year historical market data.
*   **Performance Analytics:** Gain insights into your strategies with key metrics like total return, annualized return, win rate, Sharpe ratio, and max drawdown.
*   **Real-Time Signal Generation:** Get buy, sell, or hold recommendations based on your trained models and current market momentum.
*   **AI-Powered Chatbot:** Interact with a Google Gemini-powered assistant for market insights and educational guidance.
*   **Live Market Data:** Access real-time stock quotes, charts, and market data from Yahoo Finance.
*   **Interactive Dashboard:** Monitor your portfolio, track simulations, and view key analytics from a centralized home dashboard.
*   **Modern Tech Stack:** Built with React, TypeScript, Python, and FastAPI for a high-performance, scalable, and maintainable application.

Whether you're a seasoned quant or just beginning to explore the world of algorithmic trading, the Algo Trade Simulator provides a powerful and intuitive platform to test your ideas and hone your strategies.

## Getting started

### Prerequisites

- Node.js 18 or later
- Python 3.11 or later
- A running MongoDB instance (Atlas or local)
- A Google Gemini API key

### Setup

#### Backend

1.  Create a virtual environment and install dependencies:
    ```bash
    cd backend
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```
2.  Create a `.env` file in the `backend` directory. You can copy the provided `backend/.env.example` to get started.
    ```
    MONGO_URL="mongodb://localhost:27017"
    GOOGLE_API_KEY="your-google-api-key"
    ```
3.  Start the FastAPI server:
    ```bash
    uvicorn backend.main:app --reload --port 8000 --env-file backend/.env
    ```
    > To run without MongoDB during development, set `USE_IN_MEMORY_DB=true` in your `.env` file. All data is ephemeral and resets on restart.
4.  (Optional) Verify your setup:
    ```bash
    python backend/test.py
    ```

The API will be available at `http://localhost:8000` and includes automatically generated Swagger docs at `/docs`.

#### Frontend

1.  Install dependencies from the repository root:
    ```bash
    npm install
    ```
2.  (Optional) Create a `.env` file in the `client` directory to enable the login bypass for development:
    ```
    VITE_ENABLE_LOGIN_BYPASS=true
    ```
3.  Start the Vite dev server:
    ```bash
    npm run dev
    ```
4.  Open `http://localhost:5173` in your browser. The frontend proxies API requests to `http://localhost:8000` by default. You can override this by setting `VITE_API_BASE_URL` in `client/.env`.

## Configuration

### Backend environment variables

The backend recognises the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URL` | Preferred connection string for MongoDB | unset (falls back to defaults) |
| `MONGODB_URI` | Alternate connection string (used if `MONGO_URL` is unset) | `mongodb://localhost:27017` |
| `MONGO_URI` | Legacy connection string key (used if the others are unset) | unset |
| `MONGODB_DB` | Database name | `algo-trade-simulator` |
| `GOOGLE_API_KEY` | Google Gemini API key for the chatbot | unset |
| `FRONTEND_ORIGIN` | Allowed CORS origin for the web app | `http://localhost:5173` |
| `SESSION_DURATION_DAYS` | Optional override for session lifetime in days | `7` |
| `ENABLE_DEV_ENDPOINTS` | Enables development-only routes such as the login bypass helper | `false` |
| `USE_IN_MEMORY_DB` | Stores users, sessions, and simulations in memory for local testing (no MongoDB required) | `false` |
| `YAHOO_USER_AGENT` | Optional override for the header sent to Yahoo Finance endpoints | `Mozilla/5.0 (compatible; AlgoTradeSimulator/1.0; +https://example.com)` |
| `OLLAMA_URL` | Base URL for the local Ollama service | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model alias to use for chat completions | `mistral` |
| `OLLAMA_TIMEOUT_SECONDS` | Timeout (seconds) for Ollama responses | `30` |

> **Tip:** When deploying, supply a production MongoDB connection string and set `FRONTEND_ORIGIN` to your hosted frontend URL.

> If multiple Mongo variables are set, `MONGO_URL` wins, followed by `MONGODB_URI`, then the legacy `MONGO_URI`.

### Frontend environment variables

The Vite frontend honours the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Base URL for API requests | `http://localhost:8000` |
| `VITE_ENABLE_LOGIN_BYPASS` | Auto-sign in via the dev bypass endpoint when set to `true` | `false` |
| `VITE_LOGIN_BYPASS_EMAIL` | Optional email used when creating the bypass session | unset (backend default) |
| `VITE_LOGIN_BYPASS_NAME` | Optional name applied to the bypass session | unset (backend default) |

> To skip the login page during development, run the backend with `ENABLE_DEV_ENDPOINTS=true` and set `VITE_ENABLE_LOGIN_BYPASS=true` before starting the Vite dev server.
> You can also navigate to `http://localhost:5173/dev/auth/bypass` while the Vite dev server is running to trigger the bypass on demand.

## API overview

The FastAPI server exposes REST endpoints. Key routes include:

- `POST /auth/signup` – register a new user and return an access token
- `POST /auth/login` – authenticate an existing user
- `GET /auth/session` – validate a bearer token and retrieve the current user
- `GET /market/watchlist` – fetch live quotes for a comma separated list of symbols
- `GET /market/quote/{symbol}` – fetch a single market quote
- `GET /market/search` - look up tickers and exchanges that match a user query
- `GET /market/chart/{symbol}` - return candlestick-ready chart data with configurable range and interval
- `GET /simulations` – list saved simulations for the authenticated user
- `POST /simulations` – create a new simulation for the current user
- `PATCH /simulations/{id}` – update status or notes for a simulation
- `DELETE /simulations/{id}` – remove a simulation
- `GET /analytics/overview` - retrieve aggregated dashboard metrics for the signed-in user
- `GET /analytics/strategies` - list the built-in strategy catalogue shown on the info page
- `POST /analytics/train` - backtest/train the SMA crossover strategy on the last five years of data
- `POST /analytics/predict` - generate a live signal using the last trained strategy
- `GET /analytics/sparkline` - return sparkline-friendly price series for requested symbols
- `POST /chat` - query the hybrid chatbot (Ollama + DuckDuckGo fallback)

## Strategy lab & chatbot

1.  Ensure the backend can reach Yahoo Finance (no VPN/proxy required) and that your Google Gemini API key is configured.
2.  Train a strategy from the **Simulations** page or via `POST /analytics/train` with a symbol plus short/long SMA windows (default 20/60).
3.  Request a fresh prediction from the **Home** page or `POST /analytics/predict` to evaluate the current market regime.
4.  Use the **Chatbot** page to ask questions, run quick research, or say "create a simulation for AAPL with 25k" to auto-spin a test scenario.

Requests that require authentication expect an `Authorization: Bearer <token>` header. Tokens automatically expire after seven days.

## Development tips

- Run `npm run check` before committing frontend TypeScript changes to ensure type safety.
- Run `python -m compileall backend` to catch syntax errors in the FastAPI service.
- Update this README whenever you add or change developer-facing commands or environment variables.

## License

This project is released under the MIT License.
