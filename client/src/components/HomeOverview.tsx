import { useMemo } from "react";
import type { OverviewResponse, PredictionResult, SparklineSeries, User } from "../types";
import { SparklineChart } from "./SparklineChart";

interface HomeOverviewProps {
  user: User;
  overview: OverviewResponse | null;
  loading: boolean;
  onRefresh: () => void;
  onRequestPrediction: (symbol: string) => void;
  latestPrediction: PredictionResult | null;
  sparklines: SparklineSeries[];
}

function formatNumber(value: number, options: Intl.NumberFormatOptions = {}) {
  return new Intl.NumberFormat(undefined, options).format(value);
}

export function HomeOverview({
  user,
  overview,
  loading,
  onRefresh,
  onRequestPrediction,
  latestPrediction,
  sparklines,
}: HomeOverviewProps) {
  const totals = overview?.totals;
  const trainedSymbols = overview?.strategiesTrained ?? [];
  const sparklineMap = useMemo(() => {
    const map = new Map<string, SparklineSeries>();
    sparklines.forEach((series) => {
      map.set(series.symbol, series);
    });
    return map;
  }, [sparklines]);

  return (
    <section className="home-overview">
      <header className="header">
        <div>
          <h1>Welcome back, {user.name}</h1>
          <p style={{ color: "rgba(226,232,240,0.7)", marginTop: "0.35rem" }}>
            Your personalised snapshot across simulations, strategies, and live signals.
          </p>
        </div>
        <button type="button" onClick={onRefresh} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </header>

      <div className="stats-grid">
        <div className="stat-card">
          <span className="label">Total simulations</span>
          <strong className="value">{totals ? totals.totalSimulations : "–"}</strong>
        </div>
        <div className="stat-card">
          <span className="label">Active</span>
          <strong className="value">{totals ? totals.activeSimulations : "–"}</strong>
        </div>
        <div className="stat-card">
          <span className="label">Completed</span>
          <strong className="value">{totals ? totals.completedSimulations : "–"}</strong>
        </div>
        <div className="stat-card">
          <span className="label">Avg. starting capital</span>
          <strong className="value">
            {totals ? `$${formatNumber(totals.averageStartingCapital, { maximumFractionDigits: 0 })}` : "–"}
          </strong>
        </div>
        <div className="stat-card">
          <span className="label">Capital allocated</span>
          <strong className="value">
            {totals ? `$${formatNumber(totals.totalStartingCapital, { maximumFractionDigits: 0 })}` : "–"}
          </strong>
        </div>
        <div className="stat-card">
          <span className="label">Trained models</span>
          <strong className="value">{totals ? totals.trainedModels : "–"}</strong>
        </div>
      </div>

      <div className="panel-grid">
        <section className="panel">
          <header>
            <h2>Watchlist sparklines</h2>
            <span className="hint">Latest closes for your default symbols</span>
          </header>
          {overview?.watchlist?.length ? (
            <ul className="list">
              {overview.watchlist.map((symbol) => {
                const sparkline = sparklineMap.get(symbol);
                const latest = sparkline?.points?.at(-1)?.close;
                return (
                  <li key={symbol}>
                    <div>
                      <strong>{symbol}</strong>
                      <span className="subtle">{latest ? `$${latest.toFixed(2)}` : "–"}</span>
                    </div>
                    <SparklineChart points={sparkline?.points ?? []} />
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="empty">No watchlist data available.</p>
          )}
        </section>

        <section className="panel">
          <header>
            <h2>Recent simulations</h2>
            <span className="hint">Latest five experiments</span>
          </header>
          {overview?.recentSimulations?.length ? (
            <ul className="list">
              {overview.recentSimulations.map((item) => (
                <li key={item.id}>
                  <div>
                    <strong>{item.symbol}</strong>
                    <span className="subtle">{item.strategy}</span>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <span className="money">${formatNumber(item.startingCapital)}</span>
                    <span className="subtle">{new Date(item.createdAt).toLocaleDateString()}</span>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="empty">No simulations yet. Jump to the Simulations workspace to create one.</p>
          )}
        </section>

        <section className="panel">
          <header>
            <h2>Live signal</h2>
            <span className="hint">Run predictions on your trained symbols</span>
          </header>
          {trainedSymbols.length ? (
            <div className="predictor">
              <label>
                <span>Select a trained symbol</span>
                <select
                  onChange={(event) => {
                    const value = event.target.value;
                    if (value) {
                      onRequestPrediction(value.split(" ")[0]);
                      event.target.selectedIndex = 0;
                    }
                  }}
                  defaultValue=""
                >
                  <option value="" disabled>
                    Choose symbol
                  </option>
                  {trainedSymbols.map((entry) => (
                    <option key={entry} value={entry}>
                      {entry}
                    </option>
                  ))}
                </select>
              </label>
              {latestPrediction ? (
                <div className="prediction">
                  <strong>{latestPrediction.symbol}</strong>
                  <p>{latestPrediction.summary}</p>
                  <span className="subtle">
                    Signal: <em>{latestPrediction.signal.toUpperCase()}</em>, confidence {Math.round(latestPrediction.confidence * 100)}%
                  </span>
                </div>
              ) : (
                <p className="empty">Run a prediction to see the current momentum outlook.</p>
              )}
            </div>
          ) : (
            <p className="empty">Train a strategy first to unlock live predictions.</p>
          )}
        </section>
      </div>
    </section>
  );
}
