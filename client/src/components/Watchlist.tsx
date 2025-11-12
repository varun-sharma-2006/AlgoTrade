import type { MarketQuote } from "../types";

interface WatchlistProps {
  quotes: MarketQuote[];
  onRefresh: () => void;
  loading: boolean;
}

function formatCurrency(value: number | null | undefined, currency?: string | null) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  return value.toLocaleString(undefined, {
    style: "currency",
    currency: currency ?? "USD",
  });
}

export function Watchlist({ quotes, onRefresh, loading }: WatchlistProps) {
  return (
    <div className="card">
      <div className="header" style={{ marginBottom: "1rem" }}>
        <div>
          <h2>Live market watchlist</h2>
          <p style={{ color: "rgba(226,232,240,0.7)", margin: 0 }}>
            Quotes powered by Yahoo Finance via yfinance. Values refresh on demand.
          </p>
        </div>
        <button type="button" onClick={onRefresh} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {quotes.length === 0 ? (
        <div className="empty-state">Add tickers or refresh to load market data.</div>
      ) : (
        <table className="watchlist-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Price</th>
              <th>Change</th>
              <th>Prev. close</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {quotes.map((quote) => {
              const changeValue = typeof quote.change === "number" ? quote.change : null;
              const changePercent = typeof quote.changePercent === "number" ? quote.changePercent : null;
              const changeClass = changeValue === null ? "trend-neutral" : changeValue >= 0 ? "trend-positive" : "trend-negative";
              return (
                <tr key={quote.symbol}>
                  <td>
                    <span className="badge">{quote.symbol}</span>
                  </td>
                  <td>{formatCurrency(quote.price, quote.currency)}</td>
                  <td className={changeClass}>
                    {changeValue === null || changePercent === null ? (
                      "—"
                    ) : (
                      <>
                        {changeValue >= 0 ? "+" : ""}
                        {changeValue.toFixed(2)} ({changePercent.toFixed(2)}%)
                      </>
                    )}
                  </td>
                  <td>{formatCurrency(quote.previousClose ?? null, quote.currency)}</td>
                  <td>{new Date(quote.updated).toLocaleTimeString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
