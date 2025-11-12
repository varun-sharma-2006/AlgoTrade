# Algo Trade Simulator

A full-stack trading simulation platform with live market data, personalised strategy tracking, and persistent user accounts. The project now ships with a Python (FastAPI) backend and a React frontend so you can spin up the experience locally with minimal setup.

## Features

- **Live market data** sourced on demand from Yahoo Finance with resilient quote/chart fallbacks
- **Live market lab** page with ticker search, candlestick overlays, and intraday stats powered by Yahoo Finance
- **AI-assisted strategy lab** that trains and backtests SMA crossovers on 5-year history and generates live signals
- **Hybrid chatbot copilot** using local Ollama (Mistral) with DuckDuckGo fallback for research and automation, now including budget-aware, diversified allocation suggestions for short horizons
- **Simulation workspace** for creating, updating, and tracking algorithmic trading experiments
- **Home analytics dashboard** surfacing portfolio stats, trained strategies, and recent results
- **Session recovery** using browser storage so signed-in users can resume quickly

## Tech stack

| Area     | Technology |
|----------|------------|
| Frontend | React + Vite + TypeScript |
| Backend  | FastAPI, Motor (MongoDB), Passlib |
| Database | MongoDB |
| Market data | Yahoo Finance quote & chart APIs (via requests) |

## Getting started

### Prerequisites

- Node.js 18 or later
- Python 3.11 or later
- A running MongoDB instance (Atlas or local)

### Backend setup

1. Create a virtual environment and install dependencies:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   pip install -r requirements.txt
   ```
2. Configure environment variables (defaults are provided below). You can create a `.env` file in `backend/` if desired.
3. Start the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   > To run without MongoDB during development, export `USE_IN_MEMORY_DB=true`. All data is ephemeral and resets on restart.
4. (Optional) Verify MongoDB connectivity:
   ```bash
   python test.py
   ```

The API will be available at `http://localhost:8000` and includes automatically generated Swagger docs at `/docs`.

### Frontend setup

1. Install dependencies from the repository root:
   ```bash
   npm install
   ```
2. Start the Vite dev server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser. The frontend proxies API requests to `http://localhost:8000` by default. You can override this by setting `VITE_API_BASE_URL` in `client/.env`.

## Configuration

The backend recognises the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URL` | Preferred connection string for MongoDB | unset (falls back to defaults) |
| `MONGODB_URI` | Alternate connection string (used if `MONGO_URL` is unset) | `mongodb://localhost:27017` |
| `MONGO_URI` | Legacy connection string key (used if the others are unset) | unset |
| `MONGODB_DB` | Database name | `algo-trade-simulator` |
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

### Frontend environment

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

1. Ensure the backend can reach Yahoo Finance (no VPN/proxy required) and that Ollama is running locally with the Mistral model pulled (`ollama run mistral`).
2. Train a strategy from the **Simulations** page or via `POST /analytics/train` with a symbol plus short/long SMA windows (default 20/60).
3. Request a fresh prediction from the **Home** page or `POST /analytics/predict` to evaluate the current market regime.
4. Use the **Chatbot** page to ask questions, run quick research, or say "create a simulation for AAPL with 25k" to auto-spin a test scenario.


Requests that require authentication expect an `Authorization: Bearer <token>` header. Tokens automatically expire after seven days.

## Development tips

- Run `npm run check` before committing frontend TypeScript changes to ensure type safety.
- Run `python -m compileall backend` to catch syntax errors in the FastAPI service.
- Update this README whenever you add or change developer-facing commands or environment variables.

## License

This project is released under the MIT License.
