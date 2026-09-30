import { useState, type FormEvent } from "react";
import type { PredictionResult, StrategyId, StrategyMetrics, TrainingPayload, TrainingResult } from "../types";
import { SparklineChart } from "./SparklineChart";

interface StrategyTrainerProps {
  onTrain: (payload: TrainingPayload) => Promise<void> | void;
  onPredict: (symbol: string) => Promise<void> | void;
  training: TrainingResult | null;
  prediction: PredictionResult | null;
  loading: boolean;
}

interface ParamField {
  key: keyof TrainingPayload;
  label: string;
  min: number;
  max: number;
  step?: number;
  initial: number;
}

export type BuiltInStrategyId = Exclude<StrategyId, "custom">;

export const STRATEGY_FORMS: Record<BuiltInStrategyId, { label: string; fields: ParamField[] }> = {
  "sma-crossover": {
    label: "SMA crossover",
    fields: [
      { key: "shortWindow", label: "Short window", min: 2, max: 180, initial: 20 },
      { key: "longWindow", label: "Long window", min: 5, max: 365, initial: 60 },
    ],
  },
  "mean-reversion": {
    label: "Mean reversion (Bollinger)",
    fields: [
      { key: "lookback", label: "Lookback", min: 5, max: 200, initial: 20 },
      { key: "deviation", label: "Band width (std devs)", min: 0.5, max: 5, step: 0.5, initial: 2 },
    ],
  },
  "trend-follow": {
    label: "Breakout (Donchian)",
    fields: [{ key: "channel", label: "Channel length", min: 5, max: 200, initial: 20 }],
  },
  "buy-hold": { label: "Buy & hold (baseline)", fields: [] },
};

export function strategyLabel(strategyId: string): string {
  if (strategyId === "custom") return "Custom rules";
  return STRATEGY_FORMS[strategyId as BuiltInStrategyId]?.label ?? strategyId;
}

const initialParams = Object.fromEntries(
  Object.values(STRATEGY_FORMS).flatMap((form) => form.fields.map((field) => [field.key, field.initial])),
) as Record<string, number>;

const pct = (value: number) => (Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "–");

function MetricRow({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <li title={hint}>
      <span>{label}</span>
      <strong>{value}</strong>
    </li>
  );
}

export function Metrics({ metrics }: { metrics: StrategyMetrics }) {
  const beat = metrics.excessReturn >= 0;
  return (
    <ul className="metrics">
      <MetricRow label="Strategy return" value={pct(metrics.totalReturn)} />
      <MetricRow label="Buy & hold return" value={pct(metrics.buyHoldReturn)} />
      <MetricRow
        label={beat ? "Beat buy & hold by" : "Lagged buy & hold by"}
        value={pct(Math.abs(metrics.excessReturn))}
      />
      <MetricRow label="Annualised return" value={pct(metrics.annualizedReturn)} />
      <MetricRow label="Sharpe ratio" value={metrics.sharpe.toFixed(2)} hint="Annualised, risk-free rate 0" />
      <MetricRow label="Max drawdown" value={pct(metrics.maxDrawdown)} />
      <MetricRow label="Win rate" value={pct(metrics.winRate)} hint="Share of closed trades that made money" />
      <MetricRow label="Trades" value={`${metrics.trades} (${metrics.closedTrades} closed)`} />
      <MetricRow label="Avg trade" value={pct(metrics.avgTradeReturn)} />
      <MetricRow label="Time in market" value={pct(metrics.exposure)} />
    </ul>
  );
}

/** Equity curve, metrics and recent trades for a backtest (used by the strategy lab and the builder). */
export function BacktestResults({ training, description }: { training: TrainingResult; description?: string }) {
  const equity = training.sample.map((point) => ({ timestamp: point.timestamp, close: point.equity }));
  const params = Object.entries(training.parameters ?? {})
    .map(([key, value]) => `${key} ${value}`)
    .join(", ");
  return (
    <div className="training-result">
      <div className="summary">
        <strong>{training.symbol}</strong>
        <span className="subtle">{description ?? [strategyLabel(training.strategyId), params].filter(Boolean).join(" · ")}</span>
        <span className="subtle">
          {new Date(training.period.start).toLocaleDateString()} – {new Date(training.period.end).toLocaleDateString()} ·{" "}
          {training.metrics.feeBps} bps fee per trade
        </span>
      </div>
      {equity.length ? (
        <div className="equity-curve">
          <span className="subtle">Equity curve (last {equity.length} days, starts at 1.0)</span>
          <SparklineChart points={equity} stretch />
        </div>
      ) : null}
      <Metrics metrics={training.metrics} />
      {training.trades.length || training.openTrade ? (
        <table className="trades-table">
          <thead>
            <tr>
              <th>Entry</th>
              <th>Exit</th>
              <th>Return</th>
            </tr>
          </thead>
          <tbody>
            {[...training.trades].reverse().slice(0, 5).map((trade) => (
              <tr key={trade.entryDate}>
                <td>
                  {new Date(trade.entryDate).toLocaleDateString()} @ {trade.entryPrice.toFixed(2)}
                </td>
                <td>
                  {new Date(trade.exitDate).toLocaleDateString()} @ {trade.exitPrice.toFixed(2)}
                </td>
                <td className={trade.return >= 0 ? "positive" : "negative"}>{pct(trade.return)}</td>
              </tr>
            ))}
            {training.openTrade ? (
              <tr>
                <td>
                  {new Date(training.openTrade.entryDate).toLocaleDateString()} @{" "}
                  {training.openTrade.entryPrice.toFixed(2)}
                </td>
                <td>open</td>
                <td className={training.openTrade.return >= 0 ? "positive" : "negative"}>
                  {pct(training.openTrade.return)}
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      ) : (
        <p className="empty">The strategy never entered a trade in this period.</p>
      )}
    </div>
  );
}

export function StrategyTrainer({ onTrain, onPredict, training, prediction, loading }: StrategyTrainerProps) {
  const [symbol, setSymbol] = useState("AAPL");
  const [strategyId, setStrategyId] = useState<BuiltInStrategyId>("sma-crossover");
  const [params, setParams] = useState<Record<string, number>>(initialParams);
  const form = STRATEGY_FORMS[strategyId];

  const handleTrain = (event: FormEvent) => {
    event.preventDefault();
    const values = Object.fromEntries(form.fields.map((field) => [field.key, params[field.key]]));
    onTrain({ symbol, strategyId, ...values });
  };

  return (
    <section className="panel strategy-lab">
      <header>
        <h2>Strategy lab</h2>
        <span className="hint">Backtest a strategy on 2 years of daily data, fees included</span>
      </header>

      <form className="form-grid" onSubmit={handleTrain}>
        <label>
          <span>Symbol</span>
          <input value={symbol} onChange={(event) => setSymbol(event.target.value.toUpperCase())} maxLength={12} />
        </label>
        <label>
          <span>Strategy</span>
          <select value={strategyId} onChange={(event) => setStrategyId(event.target.value as BuiltInStrategyId)}>
            {Object.entries(STRATEGY_FORMS).map(([id, option]) => (
              <option key={id} value={id}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        {form.fields.map((field) => (
          <label key={field.key}>
            <span>{field.label}</span>
            <input
              type="number"
              min={field.min}
              max={field.max}
              step={field.step ?? 1}
              value={params[field.key]}
              onChange={(event) => setParams((previous) => ({ ...previous, [field.key]: Number(event.target.value) }))}
            />
          </label>
        ))}
        <div className="actions">
          <button type="submit" disabled={loading}>
            {loading ? "Training..." : "Run backtest"}
          </button>
          <button
            type="button"
            onClick={() => onPredict(symbol)}
            disabled={loading || !training}
            className="button-ghost"
          >
            {loading ? "Working..." : "Today's signal"}
          </button>
        </div>
      </form>

      {training ? (
        <BacktestResults training={training} />
      ) : (
        <p className="empty">Run a backtest to see how the strategy would have traded.</p>
      )}

      {prediction ? (
        <div className="prediction">
          <strong>Today's signal · {prediction.symbol}</strong>
          <p>{prediction.summary}</p>
          <span className="subtle">
            Signal: {prediction.signal.toUpperCase()} · Strength {Math.round(prediction.confidence * 100)}%
          </span>
        </div>
      ) : null}
    </section>
  );
}
