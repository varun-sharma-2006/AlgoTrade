import { useState, type FormEvent } from "react";
import type { PredictionResult, StrategyMetrics, TrainingPayload, TrainingResult } from "../types";

interface StrategyTrainerProps {
  onTrain: (payload: TrainingPayload) => Promise<void> | void;
  onPredict: (symbol: string) => Promise<void> | void;
  training: TrainingResult | null;
  prediction: PredictionResult | null;
  loading: boolean;
}

function MetricRow({ label, value, suffix = "" }: { label: string; value: number; suffix?: string }) {
  return (
    <li>
      <span>{label}</span>
      <strong>
        {Number.isFinite(value) ? value.toFixed(2) : "–"}
        {suffix}
      </strong>
    </li>
  );
}

function Metrics({ metrics }: { metrics: StrategyMetrics }) {
  return (
    <ul className="metrics">
      <MetricRow label="Total return" value={metrics.totalReturn * 100} suffix="%" />
      <MetricRow label="Annualised return" value={metrics.annualizedReturn * 100} suffix="%" />
      <MetricRow label="Win rate" value={metrics.winRate * 100} suffix="%" />
      <MetricRow label="Sharp ratio" value={metrics.sharpe} />
      <MetricRow label="Max drawdown" value={metrics.maxDrawdown * 100} suffix="%" />
      <li>
        <span>Trades</span>
        <strong>{metrics.trades}</strong>
      </li>
    </ul>
  );
}

export function StrategyTrainer({ onTrain, onPredict, training, prediction, loading }: StrategyTrainerProps) {
  const [symbol, setSymbol] = useState("AAPL");
  const [shortWindow, setShortWindow] = useState(20);
  const [longWindow, setLongWindow] = useState(60);

  const handleTrain = (event: FormEvent) => {
    event.preventDefault();
    onTrain({ symbol, shortWindow, longWindow, strategyId: "sma_cross" });
  };

  return (
    <section className="panel">
      <header>
        <h2>Strategy lab</h2>
        <span className="hint">Train SMA crossovers and inspect the results</span>
      </header>

      <form className="form-grid" onSubmit={handleTrain}>
        <label>
          <span>Symbol</span>
          <input value={symbol} onChange={(event) => setSymbol(event.target.value.toUpperCase())} maxLength={6} />
        </label>
        <label>
          <span>Short window</span>
          <input
            type="number"
            min={5}
            max={180}
            value={shortWindow}
            onChange={(event) => setShortWindow(Number(event.target.value))}
          />
        </label>
        <label>
          <span>Long window</span>
          <input
            type="number"
            min={20}
            max={365}
            value={longWindow}
            onChange={(event) => setLongWindow(Number(event.target.value))}
          />
        </label>
        <div className="actions">
          <button type="submit" disabled={loading}>
            {loading ? "Training..." : "Train strategy"}
          </button>
          <button
            type="button"
            onClick={() => onPredict(symbol)}
            disabled={loading || !training}
            style={{ marginLeft: "0.5rem" }}
          >
            {loading ? "Working..." : "Predict now"}
          </button>
        </div>
      </form>

      {training ? (
        <div className="training-result">
          <div className="summary">
            <strong>{training.symbol}</strong>
            <span className="subtle">
              {training.strategyId} · windows {training.shortWindow}/{training.longWindow}
            </span>
            <span className="subtle">Trained {new Date(training.trainedAt).toLocaleString()}</span>
          </div>
          <Metrics metrics={training.metrics} />
        </div>
      ) : (
        <p className="empty">Train a symbol to view performance metrics.</p>
      )}

      {prediction ? (
        <div className="prediction">
          <strong>Latest signal · {prediction.symbol}</strong>
          <p>{prediction.summary}</p>
          <span className="subtle">Signal: {prediction.signal.toUpperCase()} · Confidence {Math.round(prediction.confidence * 100)}%</span>
        </div>
      ) : null}
    </section>
  );
}
