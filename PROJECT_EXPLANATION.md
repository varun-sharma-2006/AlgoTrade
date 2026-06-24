# Algo Trade Simulator - Project Explanation

## 1. High-Level Overview (The "Elevator Pitch")
**Algo Trade Simulator** is a full-stack algorithmic trading platform designed for quantitative analysis and strategy backtesting. It allows users to simulate trading scenarios, train momentum-based strategies on historical data, and receive real-time signals powered by AI. I built this to bridge the gap between complex quantitative tools and intuitive user experiences, providing a playground for both beginner and experienced traders to validate their ideas.

---

## 2. Product Walkthrough (Features)
The platform is organized into four main functional areas:

*   **Market Monitoring:** A real-time watchlist and interactive candlestick charts (powered by Yahoo Finance) that keep users updated on market trends.
*   **Strategy Lab:** The core "engine" where users can train Simple Moving Average (SMA) crossover strategies. By adjusting short and long-term momentum windows, users get instant feedback on historical performance metrics like Sharpe Ratio and Max Drawdown.
*   **Simulations Management:** A CRUD-based system where users can track hypothetical portfolios, assign specific strategies to tickers, and manage their starting capital and notes.
*   **AI Trading Assistant:** A Google Gemini-integrated chatbot that acts as a financial analyst. It can provide market insights, explain complex trading concepts, and even help automate simulation creation through natural language.

---

## 3. Technical Architecture

### **Frontend: React & TypeScript**
*   **Type Safety:** Used TypeScript throughout the project to ensure data consistency between the backend API and the UI components.
*   **Modular Components:** Built a library of reusable UI components (e.g., `StrategyTrainer`, `ChatbotPanel`, `SparklineChart`) to maintain a clean and scalable codebase.
*   **State Management:** Leveraged React Hooks (`useCallback`, `useMemo`, `useEffect`) for efficient re-renders and handling complex asynchronous data flows from multiple endpoints.

### **Backend: FastAPI & Python**
*   **Asynchronous Processing:** Built with FastAPI and `asyncio` to handle concurrent market data requests and LLM interactions without blocking.
*   **Quantitative Logic:** Implemented a quantitative engine that calculates SMA, annualized returns, and risk metrics (Sharpe, Drawdown) using historical price series.
*   **Data Integration:** Developed a robust abstraction layer for Yahoo Finance, ensuring the app remains functional even if external APIs face rate limits or latency issues.
*   **Security:** Implemented Bearer token authentication with bcrypt password hashing to secure user sessions and simulation data.

### Database & Persistence (The Role of MongoDB)
*   **Primary Persistence Layer:** MongoDB serves as the source of truth for all user-generated data, including authentication credentials, active sessions, and historical trading simulations.
*   **Asynchronous Integration:** Used the `motor` library (an async MongoDB driver) to ensure database operations are non-blocking. This is critical for maintaining high concurrency in the FastAPI backend, especially when multiple users are running backtests simultaneously.
*   **Document-Oriented Flexibility:**
    *   **Simulations:** Stored as flexible documents, allowing for different strategy parameters (like `shortWindow` vs `lookback`) without rigid schema migrations.
    *   **Trained Models:** Results of strategy training (metrics, sample price series) are stored as nested documents, making it efficient to retrieve a complete "snapshot" of a strategy's performance in a single query.
*   **Security & Isolation:** Implemented user-level data isolation by indexing collections on `userId` (stored as `ObjectId`). Every query is scoped to the authenticated user's ID to prevent cross-account data leaks.
*   **Dual-Store Architecture:** Designed a flexible persistence layer that supports both **MongoDB** and an **In-Memory Store**. This allowed for rapid development/testing without dependency on a live database.
*   **Thread-Safe In-Memory Store:** Implemented `asyncio.Lock` in the in-memory implementation to ensure data consistency and prevent race conditions during concurrent requests, mimicking the ACID properties of a real database.

---

## 4. Technical Challenges & Problem Solving

### **Challenge 1: Handling Data Inconsistency from External APIs**
*   **Problem:** The market data from Yahoo Finance can sometimes be incomplete or return inconsistent fields (e.g., missing `previousClose` or varying currency symbols).
*   **Solution:** I implemented a defensive data-fetching layer with multiple fallbacks. If the "fast info" from `yfinance` is missing, the system automatically falls back to historical daily bars to derive the necessary price data. I also built a robust "Offline Mode" with mock data to ensure a seamless developer experience.

### **Challenge 2: Managing Timezone-Naive vs. Timezone-Aware Datetimes**
*   **Problem:** Python’s `datetime` objects and MongoDB’s storage often lead to "naive vs aware" conflicts during comparisons, especially when dealing with global stock markets.
*   **Solution:** I standardized all datetime logic to **UTC**. I implemented middleware to ensure every timestamp retrieved from the database or external APIs is explicitly converted to UTC before processing, preventing logic errors in trade signal generation.

### **Challenge 3: Integrating LLMs for Financial Context**
*   **Problem:** General LLMs (like Gemini) can sometimes provide generic or irrelevant advice when asked about specific trading strategies.
*   **Solution:** I utilized specialized prompt engineering and a "hybrid" response system. For specific investment-related queries, the backend intercepts the message, fetches real-time market data, and injects that context into the LLM's prompt, ensuring the advice is grounded in current market reality.

---

## 5. Potential Future Improvements
*   **Advanced Strategies:** Moving beyond SMA to include RSI, MACD, and Mean Reversion models.
*   **Live Paper Trading:** Integrating with a broker API (like Alpaca) to allow users to execute trades in a sandbox environment.
*   **Real-time WebSockets:** Transitioning from polling to WebSockets for even faster market price updates.
