import { useState, type FormEvent } from "react";
import { downloadFile } from "../api";
import type {
  BarInterval,
  ExecutionMode,
  OptimizeResult,
  ReviewResult,
  PredictionResult,
  RobustnessPayload,
  RobustnessResult,
  SizingMode,
  StrategyId,
  StrategyMetrics,
  TrainingPayload,
  TrainingResult,
  WalkForwardPayload,
  WalkForwardResult,
  WalkForwardStrategyId,
} from "../types";
import {
  LineChart,
  ModelReportCard,
  MonthlyHeatmap,
  RiskTable,
  RobustnessResults,
  WalkForwardResults,
} from "./Research";
import { OptimizePanel, ReviewPanel, ShareBox, TaxCard } from "./Insights";

interface StrategyTrainerProps {
  onTrain: (payload: TrainingPayload) => Promise<void> | void;
  onPredict: (symbol: string) => Promise<void> | void;
  onWalkForward?: (payload: WalkForwardPayload) => Promise<WalkForwardResult>;
  onRobustness?: (payload: RobustnessPayload) => Promise<RobustnessResult>;
  onOptimize?: (payload: TrainingPayload) => Promise<OptimizeResult>;
  onReview?: (payload: RobustnessPayload) => Promise<ReviewResult>;
  /** Creates a shareable report and returns its path (/r/<id>). */
  onShare?: (payload: TrainingPayload) => Promise<string>;
  onNotebook?: (payload: TrainingPayload) => Promise<Record<string, unknown>>;
  training: TrainingResult | null;
  prediction: PredictionResult | null;
  loading: boolean;
}

export const DEFAULT_SLIPPAGE_BPS = 5;
const WALK_FORWARD_IDS: WalkForwardStrategyId[] = [
  "sma-crossover",
  "mean-reversion",
  "trend-follow",
  "regime-switch",
  "ml-logistic",
];

interface ParamField {
  key: keyof TrainingPayload;
  label: string;
  min: number;
  max: number;
  step?: number;
  initial: number;
  /** Shown as a select instead of a number input. */
  options?: Array<{ value: number; label: string }>;
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
  "regime-switch": {
    label: "Regime switching (trend vs range)",
    fields: [
      { key: "erWindow", label: "Efficiency-ratio window", min: 5, max: 120, initial: 20 },
      { key: "erThreshold", label: "Trending above ER", min: 0.05, max: 0.95, step: 0.05, initial: 0.3 },
    ],
  },
  "ml-logistic": {
    label: "Machine learning",
    fields: [
      {
        key: "modelType",
        label: "Model",
        min: 0,
        max: 1,
        initial: 0,
        options: [
          { value: 0, label: "Logistic regression" },
          { value: 1, label: "Gradient-boosted trees" },
        ],
      },
      {
        key: "horizon",
        label: "Predict",
        min: 1,
        max: 20,
        initial: 1,
        options: [
          { value: 1, label: "Tomorrow's direction" },
          { value: 5, label: "1 week ahead" },
          { value: 10, label: "2 weeks ahead" },
          { value: 20, label: "1 month ahead" },
        ],
      },
      { key: "threshold", label: "Buy when P(up) ≥", min: 0.3, max: 0.8, step: 0.01, initial: 0.52 },
      { key: "trainWindow", label: "Training window (days)", min: 126, max: 1000, step: 21, initial: 504 },
    ],
  },
  "buy-hold": { label: "Buy & hold (baseline)", fields: [] },
};

export function strategyLabel(strategyId: string): string {
  if (strategyId === "custom") return "Custom rules";
  return STRATEGY_FORMS[strategyId as BuiltInStrategyId]?.label ?? strategyId;
}

export const initialParams = Object.fromEntries(
  Object.values(STRATEGY_FORMS).flatMap((form) => form.fields.map((field) => [field.key, field.initial])),
) as Record<string, number>;

const pct = (value: number) => (Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "–");
const growth = (value: number) => `$${value.toFixed(2)}`;
const drawdownPct = (value: number) => `${(value * 100).toFixed(1)}%`;

const EXECUTION_LABELS: Record<ExecutionMode, string> = {
  next_open: "next day's open",
  close: "same day's close",
};

function MetricRow({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <li title={hint}>
      <span>{label}</span>
      <strong>{value}</strong>
    </li>
  );
}

/** Headline results and trading statistics. With `detailed`, also the risk ratios (when there is no risk table). */
export function Metrics({ metrics, detailed = true }: { metrics: StrategyMetrics; detailed?: boolean }) {
  const beat = metrics.excessReturn >= 0;
  const sellFee = metrics.sellFeeBps ?? metrics.feeBps;
  const slippage = metrics.slippageBps ?? 0;
  const costs =
    sellFee === metrics.feeBps
      ? `${+(metrics.feeBps + slippage).toFixed(2)} bps`
      : `${+(metrics.feeBps + slippage).toFixed(2)} / ${+(sellFee + slippage).toFixed(2)} bps`;
  return (
    <ul className="metrics">
      <MetricRow label="Strategy return" value={pct(metrics.totalReturn)} />
      <MetricRow label="Buy & hold return" value={pct(metrics.buyHoldReturn)} />
      <MetricRow
        label={beat ? "Beat buy & hold by" : "Lagged buy & hold by"}
        value={pct(Math.abs(metrics.excessReturn))}
      />
      {detailed ? (
        <>
          <MetricRow label="Annualised return" value={pct(metrics.annualizedReturn)} />
          <MetricRow
            label="Sharpe ratio"
            value={metrics.sharpe.toFixed(2)}
            hint={`Annualised, net of a ${((metrics.riskFreeRate ?? 0) * 100).toFixed(1)}% risk-free rate`}
          />
          <MetricRow label="Max drawdown" value={pct(metrics.maxDrawdown)} />
        </>
      ) : null}
      {metrics.var95 !== undefined ? (
        <MetricRow
          label="1-day VaR / CVaR (95%)"
          value={`${pct(metrics.var95)} / ${pct(metrics.cvar95 ?? 0)}`}
          hint="Value at Risk: the loss exceeded on 1 day in 20. CVaR: the average loss on those days."
        />
      ) : null}
      <MetricRow label="Win rate" value={pct(metrics.winRate)} hint="Share of closed trades that made money" />
      {metrics.profitFactor !== undefined ? (
        <MetricRow
          label="Profit factor"
          value={metrics.profitFactor === null ? "no losing trades" : metrics.profitFactor.toFixed(2)}
          hint="Gains of winning trades divided by losses of losing trades (above 1 = profitable)"
        />
      ) : null}
      <MetricRow
        label="Trades"
        value={`${metrics.trades} (${metrics.closedTrades} closed${metrics.shortTrades ? `, ${metrics.shortTrades} short` : ""})`}
      />
      <MetricRow label="Avg trade" value={pct(metrics.avgTradeReturn)} />
      <MetricRow label="Time in market" value={pct(metrics.exposure)} />
      {metrics.avgGrossExposure !== undefined && metrics.sizing && metrics.sizing !== "full" ? (
        <MetricRow label="Average position size" value={pct(metrics.avgGrossExposure)} hint="Average share of equity invested" />
      ) : null}
      {metrics.turnover !== undefined ? (
        <MetricRow
          label="Turnover"
          value={`${metrics.turnover.toFixed(1)}× a year`}
          hint="Value traded per year as a multiple of the portfolio"
        />
      ) : null}
      <MetricRow
        label={sellFee === metrics.feeBps ? "Cost per trade" : "Cost per buy / sell"}
        value={costs}
        hint={`${metrics.feeBps} bps fee on buys, ${sellFee} bps on sells, plus ${slippage} bps slippage`}
      />
      {metrics.execution ? (
        <MetricRow label="Orders fill at" value={EXECUTION_LABELS[metrics.execution]} />
      ) : null}
      {typeof metrics.overnightReturn === "number" && typeof metrics.intradayReturn === "number" ? (
        <MetricRow
          label="Earned overnight / during the day"
          value={`${pct(metrics.overnightReturn)} / ${pct(metrics.intradayReturn)}`}
          hint="Price moves while invested: from one close to the next open (news, earnings) vs open to close"
        />
      ) : null}
      {typeof metrics.earningsReturn === "number" ? (
        <MetricRow
          label={metrics.avoidedEarnings ? "Earnings days (skipped)" : "Earned on earnings days"}
          value={`${pct(metrics.earningsReturn)} (stock ${pct(metrics.buyHoldEarningsReturn ?? 0)}, ${metrics.earningsDays} days)`}
          hint="Sum of daily returns on the day of and after each earnings report (SEC filing dates)"
        />
      ) : null}
      {typeof metrics.impactCost === "number" ? (
        <MetricRow
          label="Market impact paid"
          value={`${pct(metrics.impactCost)} (largest order ${pct(metrics.maxParticipation ?? 0)} of daily volume)`}
          hint="Extra cost of moving the price, growing with order size relative to the stock's traded value"
        />
      ) : null}
    </ul>
  );
}

const dateOf = (timestamp: string) => new Date(timestamp).toLocaleDateString();

/** Equity curve, metrics and recent trades for a backtest (used by the strategy lab and the builder). */
export function BacktestResults({ training, description }: { training: TrainingResult; description?: string }) {
  const dates = training.sample.map((point) => point.timestamp);
  const hasBuyHold = training.sample.some((point) => typeof point.buyHold === "number");
  const hasRolling = training.sample.some((point) => typeof point.rollingSharpe === "number");
  const hasBeta = training.sample.some((point) => typeof point.rollingBeta === "number");
  const params = Object.entries(training.parameters ?? {})
    .map(([key, value]) => `${key} ${value}`)
    .join(", ");
  const m = training.metrics;
  const trades = [...training.trades].reverse().slice(0, 8);
  return (
    <div className="training-result">
      <div className="summary">
        <strong>{training.symbol}</strong>
        <span className="subtle">{description ?? [strategyLabel(training.strategyId), params].filter(Boolean).join(" · ")}</span>
        <span className="subtle">
          {new Date(training.period.start).toLocaleDateString()} – {new Date(training.period.end).toLocaleDateString()} ·{" "}
          {training.costs
            ? `${training.costs.model}: ${training.costs.description}`
            : `${m.feeBps} bps fee per trade`}
          {m.slippageBps ? ` + ${m.slippageBps} bps slippage` : ""}
          {m.allowShort ? ` · shorting on (${m.borrowBps ?? 0} bps/yr borrow)` : ""}
          {m.sizing && m.sizing !== "full" ? ` · ${m.sizing === "fixed" ? "fixed-fraction" : "volatility-targeted"} sizing` : ""}
        </span>
      </div>
      <LineChart
        label="Growth of $1"
        dates={dates}
        series={[
          { label: "Strategy", color: "#e3c27f", values: training.sample.map((p) => p.equity) },
          ...(hasBuyHold
            ? [{ label: "Buy & hold", color: "#8fb7ff", values: training.sample.map((p) => p.buyHold), dashed: true }]
            : []),
        ]}
        format={growth}
      />
      {training.sample.some((p) => typeof p.drawdown === "number") ? (
        <LineChart
          label="Drawdown (fall from the previous peak)"
          dates={dates}
          height={120}
          series={[
            { label: "Strategy", color: "#f07a7a", values: training.sample.map((p) => p.drawdown), fill: true },
            { label: "Buy & hold", color: "#8fb7ff", values: training.sample.map((p) => p.buyHoldDrawdown), dashed: true },
          ]}
          format={drawdownPct}
        />
      ) : null}
      {hasRolling ? (
        <LineChart
          label="Rolling 6-month Sharpe ratio and beta (is the edge stable over time?)"
          dates={dates}
          height={130}
          series={[
            { label: "Sharpe", color: "#e3c27f", values: training.sample.map((p) => p.rollingSharpe) },
            ...(hasBeta
              ? [{ label: "Beta", color: "#8fb7ff", values: training.sample.map((p) => p.rollingBeta), dashed: true }]
              : []),
          ]}
          format={(v) => v.toFixed(2)}
        />
      ) : null}
      {training.buyHold ? (
        <RiskTable metrics={training.metrics} buyHold={training.buyHold} benchmark={training.benchmark} />
      ) : null}
      <Metrics metrics={training.metrics} detailed={!training.buyHold} />
      {training.tax ? <TaxCard tax={training.tax} /> : null}
      {training.news ? (
        <p className={`subtle ${training.news.used ? "" : "negative"}`}>
          {training.news.used
            ? `News tone feature on: GDELT articles matching ${training.news.detail}.`
            : `News tone feature off: ${training.news.detail}.`}
        </p>
      ) : null}
      {training.earnings?.reason ? <p className="subtle">Earnings: {training.earnings.reason}.</p> : null}
      {training.monthly?.length ? <MonthlyHeatmap monthly={training.monthly} /> : null}
      {training.model ? <ModelReportCard report={training.model} /> : null}
      {trades.length || training.openTrade ? (
        <table className="trades-table">
          <thead>
            <tr>
              <th>Side</th>
              <th>Entry</th>
              <th>Exit</th>
              <th>Return</th>
            </tr>
          </thead>
          <tbody>
            {training.openTrade ? (
              <tr>
                <td>{training.openTrade.side ?? "long"}</td>
                <td>
                  {dateOf(training.openTrade.entryDate)} @ {training.openTrade.entryPrice.toFixed(2)}
                </td>
                <td>open</td>
                <td className={training.openTrade.return >= 0 ? "positive" : "negative"}>
                  {pct(training.openTrade.return)}
                </td>
              </tr>
            ) : null}
            {trades.map((trade) => (
              <tr key={`${trade.entryDate}-${trade.side}`}>
                <td>{trade.side ?? "long"}</td>
                <td>
                  {dateOf(trade.entryDate)} @ {trade.entryPrice.toFixed(2)}
                </td>
                <td>
                  {dateOf(trade.exitDate)} @ {trade.exitPrice.toFixed(2)}
                </td>
                <td className={trade.return >= 0 ? "positive" : "negative"}>{pct(trade.return)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="empty">The strategy never entered a trade in this period.</p>
      )}
    </div>
  );
}

export interface TradingSettings {
  interval: BarInterval;
  capital: number | null;
  marketImpact: boolean;
  avoidEarnings: boolean;
  newsFeatures: boolean;
  execution: ExecutionMode;
  sizing: SizingMode;
  sizeFraction: number;
  targetVol: number;
  maxLeverage: number;
  allowShort: boolean;
  borrowBps: number;
  riskFreePct: number;
  slippageBps: number;
}

export const DEFAULT_TRADING: TradingSettings = {
  interval: "1d",
  capital: null,
  marketImpact: false,
  avoidEarnings: false,
  newsFeatures: false,
  execution: "next_open",
  sizing: "full",
  sizeFraction: 0.5,
  targetVol: 0.15,
  maxLeverage: 1,
  allowShort: false,
  borrowBps: 50,
  riskFreePct: 4,
  slippageBps: DEFAULT_SLIPPAGE_BPS,
};

export function tradingPayload(settings: TradingSettings) {
  return {
    interval: settings.interval,
    ...(settings.capital ? { capital: settings.capital } : {}),
    marketImpact: settings.marketImpact,
    ...(settings.avoidEarnings ? { avoidEarnings: true } : {}),
    ...(settings.newsFeatures ? { newsFeatures: true } : {}),
    execution: settings.execution,
    sizing: settings.sizing,
    sizeFraction: settings.sizeFraction,
    targetVol: settings.targetVol,
    maxLeverage: settings.maxLeverage,
    allowShort: settings.allowShort,
    borrowBps: settings.borrowBps,
    riskFreeRate: settings.riskFreePct / 100,
    slippageBps: settings.slippageBps,
  };
}

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

/** Execution, position sizing, shorting and cost settings shared by the lab's backtests. */
export function TradingSettingsFields({
  value,
  onChange,
  shortable = true,
  machineLearning = false,
}: {
  value: TradingSettings;
  onChange: (next: TradingSettings) => void;
  shortable?: boolean;
  machineLearning?: boolean;
}) {
  const set = <K extends keyof TradingSettings>(key: K, next: TradingSettings[K]) => onChange({ ...value, [key]: next });
  return (
    <fieldset className="trading-settings">
      <legend>Execution &amp; risk</legend>
      <label>
        <span>Bars</span>
        <select value={value.interval} onChange={(e) => set("interval", e.target.value as BarInterval)}>
          <option value="1d">Daily (5 years)</option>
          <option value="1h">Hourly (last 2 years)</option>
        </select>
      </label>
      <label>
        <span>Orders fill at</span>
        <select value={value.execution} onChange={(e) => set("execution", e.target.value as ExecutionMode)}>
          <option value="next_open">Next day's open (realistic)</option>
          <option value="close">Same day's close (optimistic)</option>
        </select>
      </label>
      <label>
        <span>Position size</span>
        <select value={value.sizing} onChange={(e) => set("sizing", e.target.value as SizingMode)}>
          <option value="full">100% of equity</option>
          <option value="fixed">Fixed fraction</option>
          <option value="vol-target">Volatility target</option>
        </select>
      </label>
      {value.sizing === "fixed" ? (
        <label>
          <span>Fraction of equity (%)</span>
          <input
            type="number"
            min={5}
            max={100}
            step={5}
            value={Math.round(value.sizeFraction * 100)}
            onChange={(e) => set("sizeFraction", clamp(Number(e.target.value), 5, 100) / 100)}
          />
        </label>
      ) : null}
      {value.sizing === "vol-target" ? (
        <>
          <label>
            <span>Target volatility (%/yr)</span>
            <input
              type="number"
              min={2}
              max={100}
              step={1}
              value={Math.round(value.targetVol * 100)}
              onChange={(e) => set("targetVol", clamp(Number(e.target.value), 2, 100) / 100)}
            />
          </label>
          <label>
            <span>Max leverage</span>
            <input
              type="number"
              min={0.25}
              max={3}
              step={0.25}
              value={value.maxLeverage}
              onChange={(e) => set("maxLeverage", clamp(Number(e.target.value), 0.25, 3))}
            />
          </label>
        </>
      ) : null}
      <label>
        <span>Slippage (bps per trade)</span>
        <input
          type="number"
          min={0}
          max={100}
          step={1}
          value={value.slippageBps}
          onChange={(e) => set("slippageBps", clamp(Number(e.target.value), 0, 100))}
        />
      </label>
      <label>
        <span>Risk-free rate (%/yr)</span>
        <input
          type="number"
          min={0}
          max={25}
          step={0.25}
          value={value.riskFreePct}
          onChange={(e) => set("riskFreePct", clamp(Number(e.target.value), 0, 25))}
        />
      </label>
      <label>
        <span>Capital (tax &amp; impact)</span>
        <input
          type="number"
          min={0}
          step="any"
          placeholder="auto"
          value={value.capital ?? ""}
          title="Starting capital in the stock's currency; defaults to 100,000 (₹10 lakh for Indian stocks)"
          onChange={(e) => set("capital", Number(e.target.value) > 0 ? Number(e.target.value) : null)}
        />
      </label>
      <label className="checkbox">
        <input type="checkbox" checked={value.marketImpact} onChange={(e) => set("marketImpact", e.target.checked)} />
        <span>Market impact</span>
      </label>
      <label className="checkbox" title="Stay out of the market the day before and the day of each earnings report (US stocks)">
        <input type="checkbox" checked={value.avoidEarnings} onChange={(e) => set("avoidEarnings", e.target.checked)} />
        <span>Skip earnings (US)</span>
      </label>
      {shortable ? (
        <label className="checkbox">
          <input type="checkbox" checked={value.allowShort} onChange={(e) => set("allowShort", e.target.checked)} />
          <span>Allow short selling</span>
        </label>
      ) : null}
      {machineLearning ? (
        <label className="checkbox" title="Adds the average tone of news about the company over the previous 7 days (GDELT) as a feature">
          <input type="checkbox" checked={value.newsFeatures} onChange={(e) => set("newsFeatures", e.target.checked)} />
          <span>News tone feature</span>
        </label>
      ) : null}
      {shortable && value.allowShort ? (
        <label>
          <span>Borrow fee (bps/yr)</span>
          <input
            type="number"
            min={0}
            max={5000}
            step={25}
            value={value.borrowBps}
            onChange={(e) => set("borrowBps", clamp(Number(e.target.value), 0, 5000))}
          />
        </label>
      ) : null}
    </fieldset>
  );
}

export function ParamInput({
  field,
  value,
  onChange,
}: {
  field: ParamField;
  value: number;
  onChange: (next: number) => void;
}) {
  return (
    <label>
      <span>{field.label}</span>
      {field.options ? (
        <select value={value} onChange={(event) => onChange(Number(event.target.value))}>
          {field.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          type="number"
          min={field.min}
          max={field.max}
          step={field.step ?? 1}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      )}
    </label>
  );
}

export function StrategyTrainer({
  onTrain,
  onPredict,
  onWalkForward,
  onRobustness,
  onOptimize,
  onReview,
  onShare,
  onNotebook,
  training,
  prediction,
  loading,
}: StrategyTrainerProps) {
  const [busy, setBusy] = useState<string | null>(null);
  const [toolError, setToolError] = useState<string | null>(null);
  const [optimized, setOptimized] = useState<OptimizeResult | null>(null);
  const [review, setReview] = useState<ReviewResult | null>(null);
  const [sharePath, setSharePath] = useState<string | null>(null);
  const [symbol, setSymbol] = useState("AAPL");
  const [strategyId, setStrategyId] = useState<BuiltInStrategyId>("sma-crossover");
  const [params, setParams] = useState<Record<string, number>>(initialParams);
  const [trading, setTrading] = useState<TradingSettings>(DEFAULT_TRADING);
  const [walkForward, setWalkForward] = useState<WalkForwardResult | null>(null);
  const [walkLoading, setWalkLoading] = useState(false);
  const [walkError, setWalkError] = useState<string | null>(null);
  const [robust, setRobust] = useState<RobustnessResult | null>(null);
  const [robustLoading, setRobustLoading] = useState(false);
  const [robustError, setRobustError] = useState<string | null>(null);
  const form = STRATEGY_FORMS[strategyId];
  const canWalkForward = WALK_FORWARD_IDS.includes(strategyId as WalkForwardStrategyId);

  const payload = (): TrainingPayload => {
    const values = Object.fromEntries(form.fields.map((field) => [field.key, params[field.key]]));
    return { symbol, strategyId, ...values, ...tradingPayload(trading) };
  };

  const handleTrain = (event: FormEvent) => {
    event.preventDefault();
    onTrain(payload());
  };

  const handleWalkForward = async () => {
    if (!onWalkForward || !canWalkForward) return;
    setWalkLoading(true);
    setWalkError(null);
    try {
      setWalkForward(
        await onWalkForward({
          symbol,
          strategyId: strategyId as WalkForwardStrategyId,
          slippageBps: trading.slippageBps,
          execution: trading.execution,
          allowShort: trading.allowShort,
        }),
      );
    } catch (error) {
      setWalkForward(null);
      setWalkError(error instanceof Error ? error.message : "The walk-forward test failed.");
    } finally {
      setWalkLoading(false);
    }
  };

  const runTool = async (name: string, action: () => Promise<void>) => {
    setBusy(name);
    setToolError(null);
    try {
      await action();
    } catch (error) {
      setToolError(error instanceof Error ? error.message : `The ${name} failed.`);
    } finally {
      setBusy(null);
    }
  };

  const exportTradesCsv = () => {
    if (!training) return;
    const rows = [
      ["side", "entry_date", "entry_price", "exit_date", "exit_price", "return"],
      ...[...training.trades, ...(training.openTrade ? [training.openTrade] : [])].map((t) => [
        t.side ?? "long",
        t.entryDate,
        t.entryPrice.toFixed(4),
        t === training.openTrade ? "open" : t.exitDate,
        t.exitPrice.toFixed(4),
        t.return.toFixed(6),
      ]),
    ];
    downloadFile(`${training.symbol}-${training.strategyId}-trades.csv`, rows.map((r) => r.join(",")).join("\n"), "text/csv");
  };

  const handleRobustness = async () => {
    if (!onRobustness) return;
    setRobustLoading(true);
    setRobustError(null);
    try {
      setRobust(await onRobustness(payload()));
    } catch (error) {
      setRobust(null);
      setRobustError(error instanceof Error ? error.message : "The robustness check failed.");
    } finally {
      setRobustLoading(false);
    }
  };

  return (
    <section className="panel strategy-lab">
      <header>
        <h2>Strategy lab</h2>
        <span className="hint">
          Backtest on 2 years of daily data with fees and slippage, against buy & hold and the market index
        </span>
      </header>

      <form className="form-grid" onSubmit={handleTrain}>
        <label>
          <span>Symbol</span>
          <input value={symbol} onChange={(event) => setSymbol(event.target.value.toUpperCase())} maxLength={20} />
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
          <ParamInput
            key={field.key}
            field={field}
            value={params[field.key]}
            onChange={(next) => setParams((previous) => ({ ...previous, [field.key]: next }))}
          />
        ))}
        <TradingSettingsFields
          value={trading}
          onChange={setTrading}
          shortable={strategyId !== "buy-hold"}
          machineLearning={strategyId === "ml-logistic"}
        />
        <div className="actions">
          <button type="submit" disabled={loading}>
            {loading ? "Training..." : "Run backtest"}
          </button>
          {onWalkForward ? (
            <button
              type="button"
              className="button-ghost"
              onClick={handleWalkForward}
              disabled={walkLoading || !canWalkForward}
              title={canWalkForward ? "Tune on each past year, test on the next quarter" : "Buy & hold has nothing to tune"}
            >
              {walkLoading ? "Testing..." : "Walk-forward test"}
            </button>
          ) : null}
          {onRobustness ? (
            <button
              type="button"
              className="button-ghost"
              onClick={handleRobustness}
              disabled={robustLoading}
              title="Monte Carlo, parameter sensitivity, deflated Sharpe and probability of overfitting"
            >
              {robustLoading ? "Checking..." : "Robustness check"}
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => onPredict(symbol)}
            disabled={loading || !training}
            className="button-ghost"
          >
            {loading ? "Working..." : "Today's signal"}
          </button>
        </div>
        <div className="actions tool-actions">
          {onOptimize ? (
            <button
              type="button"
              className="button-ghost"
              disabled={busy !== null || strategyId === "buy-hold"}
              title="Search nearby settings, then check the best one for overfitting"
              onClick={() => void runTool("optimiser", async () => setOptimized(await onOptimize(payload())))}
            >
              {busy === "optimiser" ? "Optimising..." : "Optimise"}
            </button>
          ) : null}
          {onReview ? (
            <button
              type="button"
              className="button-ghost"
              disabled={busy !== null}
              title="A sceptical review: costs, luck, overfitting, execution sensitivity"
              onClick={() => void runTool("review", async () => setReview(await onReview(payload())))}
            >
              {busy === "review" ? "Reviewing..." : "Review this backtest"}
            </button>
          ) : null}
          {onShare ? (
            <button
              type="button"
              className="button-ghost"
              disabled={busy !== null}
              title="Create a public link with the results and robustness checks"
              onClick={() => void runTool("share", async () => setSharePath(await onShare(payload())))}
            >
              {busy === "share" ? "Creating link..." : "Share"}
            </button>
          ) : null}
          {onNotebook ? (
            <button
              type="button"
              className="button-ghost"
              disabled={busy !== null || strategyId === "ml-logistic"}
              title={strategyId === "ml-logistic" ? "The ML strategy can't be exported" : "A Jupyter notebook that reproduces this backtest"}
              onClick={() =>
                void runTool("notebook export", async () => {
                  const nb = await onNotebook(payload());
                  downloadFile(`${symbol}-${strategyId}.ipynb`, JSON.stringify(nb, null, 1));
                })
              }
            >
              Notebook
            </button>
          ) : null}
          <button type="button" className="button-ghost" disabled={!training} onClick={exportTradesCsv}>
            Trades CSV
          </button>
          <button type="button" className="button-ghost" disabled={!training} onClick={() => window.print()}>
            Print / PDF
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

      {toolError ? <div className="error-banner walk-forward-slot">{toolError}</div> : null}
      {sharePath ? (
        <div className="walk-forward-slot">
          <ShareBox path={sharePath} onClose={() => setSharePath(null)} />
        </div>
      ) : null}
      {review ? (
        <div className="walk-forward-slot">
          <ReviewPanel review={review} />
        </div>
      ) : null}
      {optimized ? (
        <div className="walk-forward-slot">
          <OptimizePanel
            result={optimized}
            onUse={(best) => {
              setParams((previous) => ({ ...previous, ...best }));
              setOptimized(null);
            }}
          />
        </div>
      ) : null}

      {robustError ? <div className="error-banner walk-forward-slot">{robustError}</div> : null}
      {robust ? (
        <div className="walk-forward-slot">
          <RobustnessResults result={robust} />
        </div>
      ) : null}

      {walkError ? <div className="error-banner walk-forward-slot">{walkError}</div> : null}
      {walkForward ? (
        <div className="walk-forward-slot">
          <WalkForwardResults result={walkForward} />
        </div>
      ) : null}
    </section>
  );
}
