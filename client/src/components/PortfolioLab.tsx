import { useState, type FormEvent } from "react";
import type { BasketPayload, BasketResult, CustomStrategy, Rebalance, Weighting } from "../types";
import { LineChart, MonthlyHeatmap, pct, signedPct } from "./Research";
import { ParamInput, STRATEGY_FORMS, initialParams, strategyLabel, type BuiltInStrategyId } from "./StrategyTrainer";

interface PortfolioLabProps {
  onRun: (payload: BasketPayload) => Promise<BasketResult>;
  customStrategies?: CustomStrategy[];
}

const BASKETS: Array<{ name: string; symbols: string[] }> = [
  { name: "US mega-caps", symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"] },
  {
    name: "NIFTY leaders",
    symbols: ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS"],
  },
  { name: "US sectors (ETFs)", symbols: ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"] },
  { name: "Global assets", symbols: ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"] },
];

const WEIGHTINGS: Array<{ value: Weighting; label: string; hint: string }> = [
  { value: "equal", label: "Equal weight", hint: "The same slice for every stock" },
  { value: "inverse-vol", label: "Inverse volatility", hint: "Calmer stocks get more money" },
  { value: "risk-parity", label: "Risk parity", hint: "Each stock contributes the same risk, using correlations" },
];

const ratio = (value: number | undefined) => (value === undefined || !Number.isFinite(value) ? "–" : value.toFixed(2));
const tone = (value: number) => (value > 0 ? "positive" : value < 0 ? "negative" : "");

function parseSymbols(text: string) {
  return Array.from(new Set(text.split(/[\s,;]+/).map((s) => s.trim().toUpperCase()).filter(Boolean)));
}

export function PortfolioLab({ onRun, customStrategies = [] }: PortfolioLabProps) {
  const [symbolsText, setSymbolsText] = useState(BASKETS[0].symbols.join(", "));
  const [strategy, setStrategy] = useState<string>("sma-crossover");
  const [params, setParams] = useState<Record<string, number>>(initialParams);
  const [weighting, setWeighting] = useState<Weighting>("equal");
  const [rebalance, setRebalance] = useState<Rebalance>("monthly");
  const [topN, setTopN] = useState(0);
  const [allowShort, setAllowShort] = useState(false);
  const [result, setResult] = useState<BasketResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const custom = strategy.startsWith("custom:") ? customStrategies.find((s) => `custom:${s.id}` === strategy) : null;
  const builtIn = custom ? null : (strategy as BuiltInStrategyId);
  const fields = builtIn ? STRATEGY_FORMS[builtIn].fields : [];
  const symbols = parseSymbols(symbolsText);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(
        await onRun({
          symbols,
          strategyId: custom ? "custom" : (builtIn as string),
          parameters: Object.fromEntries(fields.map((field) => [field.key, params[field.key]])),
          rules: custom ? custom.rules : null,
          weighting,
          rebalance,
          topN,
          allowShort,
        }),
      );
    } catch (runError) {
      setResult(null);
      setError(runError instanceof Error ? runError.message : "The portfolio backtest failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="portfolio-lab">
      <header className="header">
        <div>
          <span className="eyebrow">Many stocks at once</span>
          <h1>Portfolio lab</h1>
          <p>
            Run one strategy on a whole basket with real portfolio weighting and rebalancing, and see whether its edge shows up
            on most stocks or just one lucky ticker.
          </p>
        </div>
      </header>

      <form className="panel" onSubmit={submit}>
        <header>
          <h2>Basket</h2>
          <span className="hint">2 to 30 tickers; only days every stock traded are used</span>
        </header>
        <div className="preset-row">
          <span className="hint">Start from:</span>
          {BASKETS.map((preset) => (
            <button
              key={preset.name}
              type="button"
              className="button-ghost chip"
              onClick={() => setSymbolsText(preset.symbols.join(", "))}
            >
              {preset.name}
            </button>
          ))}
        </div>
        <label>
          <span>Symbols ({symbols.length})</span>
          <textarea rows={2} value={symbolsText} onChange={(event) => setSymbolsText(event.target.value)} />
        </label>
        <div className="form-grid">
          <label>
            <span>Strategy</span>
            <select value={strategy} onChange={(event) => setStrategy(event.target.value)}>
              {Object.entries(STRATEGY_FORMS).map(([id, option]) => (
                <option key={id} value={id}>
                  {option.label}
                </option>
              ))}
              {customStrategies.map((s) => (
                <option key={s.id} value={`custom:${s.id}`}>
                  Custom: {s.name}
                </option>
              ))}
            </select>
          </label>
          {fields.map((field) => (
            <ParamInput
              key={field.key}
              field={field}
              value={params[field.key]}
              onChange={(next) => setParams((previous) => ({ ...previous, [field.key]: next }))}
            />
          ))}
          <label>
            <span>Weighting</span>
            <select value={weighting} onChange={(event) => setWeighting(event.target.value as Weighting)}>
              {WEIGHTINGS.map((w) => (
                <option key={w.value} value={w.value} title={w.hint}>
                  {w.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Rebalance</span>
            <select value={rebalance} onChange={(event) => setRebalance(event.target.value as Rebalance)}>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
              <option value="quarterly">Quarterly</option>
            </select>
          </label>
          <label>
            <span>Hold only the top N by momentum</span>
            <input
              type="number"
              min={0}
              max={30}
              value={topN}
              onChange={(event) => setTopN(Math.max(0, Math.min(30, Number(event.target.value))))}
              title="0 holds every stock; otherwise only the N with the best 6-month return at each rebalance"
            />
          </label>
          {builtIn !== "buy-hold" ? (
            <label className="checkbox">
              <input type="checkbox" checked={allowShort} onChange={(event) => setAllowShort(event.target.checked)} />
              <span>Allow short selling</span>
            </label>
          ) : null}
        </div>
        <p className="hint">{WEIGHTINGS.find((w) => w.value === weighting)?.hint}. Each stock's slice is invested only while its strategy signal is on; otherwise it stays in cash. Orders decided at a close fill at the next close.</p>
        {error ? <div className="error-banner">{error}</div> : null}
        <div className="actions">
          <button type="submit" disabled={busy || symbols.length < 2}>
            {busy ? "Running…" : "Run portfolio backtest"}
          </button>
        </div>
      </form>

      {result ? <BasketResults result={result} /> : null}
    </section>
  );
}

export function BasketResults({ result }: { result: BasketResult }) {
  const m = result.metrics;
  const summary = result.crossSection.summary;
  const dates = result.curve.map((p) => p.timestamp);
  return (
    <div className="basket-results">
      <div className="stats-grid">
        <div className="stat-card">
          <span className="label">Portfolio return</span>
          <strong className={`value plain ${tone(m.totalReturn)}`}>{signedPct(m.totalReturn)}</strong>
          <span className="subtle">
            {signedPct(m.annualizedReturn)} a year · equal-weight hold {signedPct(m.equalWeightReturn)}
          </span>
        </div>
        <div className="stat-card">
          <span className="label">Sharpe / max drawdown</span>
          <strong className="value plain">{ratio(m.sharpe)}</strong>
          <span className="subtle">
            drawdown {pct(m.maxDrawdown, 1)} · hold Sharpe {ratio(result.equalWeight.sharpe)}
          </span>
        </div>
        <div className="stat-card">
          <span className="label">Consistency across stocks</span>
          <strong className="value plain">{pct(summary.shareBeatBuyHold, 0)}</strong>
          <span className="subtle">
            beat buy &amp; hold on their own · median Sharpe {ratio(summary.medianSharpe)}
          </span>
        </div>
      </div>

      <div className="panel">
        <header>
          <h2>
            {strategyLabel(result.strategyId)} on {result.symbols.length} stocks
          </h2>
          <span className="hint">
            {result.weighting} weights, {result.rebalance} rebalancing{result.topN ? `, top ${result.topN} by momentum` : ""} ·{" "}
            {result.metrics.rebalances} rebalances · turnover {m.turnover.toFixed(1)}× a year · costs paid {pct(m.costPaid, 2)} of
            starting capital
            {result.missing.length ? ` · no data for ${result.missing.join(", ")}` : ""}
          </span>
        </header>
        <LineChart
          label="Growth of $1"
          dates={dates}
          series={[
            { label: "Strategy portfolio", color: "#e3c27f", values: result.curve.map((p) => p.equity) },
            { label: "Equal-weight hold", color: "#8fb7ff", values: result.curve.map((p) => p.equalWeight), dashed: true },
          ]}
          format={(v) => `$${v.toFixed(2)}`}
        />
        <LineChart
          label="Drawdown"
          dates={dates}
          height={110}
          series={[{ label: "Strategy portfolio", color: "#f07a7a", values: result.curve.map((p) => p.drawdown), fill: true }]}
          format={(v) => `${(v * 100).toFixed(1)}%`}
        />
        {result.benchmark ? (
          <p className="subtle risk-note">
            Against the {result.benchmark.name} ({signedPct(result.benchmark.totalReturn)}): beta {ratio(result.benchmark.beta)},
            correlation {ratio(result.benchmark.correlation)}, alpha {signedPct(result.benchmark.alpha)} a year. 1-day VaR (95%){" "}
            {pct(m.var95, 2)}.
          </p>
        ) : null}
        <MonthlyHeatmap monthly={result.monthly} />
      </div>

      <div className="portfolio-grid">
        <div className="panel">
          <header>
            <h2>Holdings</h2>
            <span className="hint">Contribution = each stock's share of the portfolio's return</span>
          </header>
          <div className="table-scroll">
            <table className="trades-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Avg weight</th>
                  <th>Now</th>
                  <th>Invested</th>
                  <th>Contribution</th>
                  <th>Stock</th>
                </tr>
              </thead>
              <tbody>
                {result.holdings.map((h) => (
                  <tr key={h.symbol}>
                    <td>
                      <span className="badge">{h.symbol}</span>
                    </td>
                    <td>{pct(h.avgWeight, 1)}</td>
                    <td>{pct(h.finalWeight, 1)}</td>
                    <td>{pct(h.timeInMarket, 0)}</td>
                    <td className={tone(h.contribution)}>{signedPct(h.contribution)}</td>
                    <td>{signedPct(h.buyHoldReturn)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="panel">
          <header>
            <h2>Each stock alone</h2>
            <span className="hint">
              Profitable on {pct(summary.shareProfitable, 0)}, better Sharpe than holding on{" "}
              {pct(summary.shareSharpeAboveBuyHold, 0)}; Sharpe middle half {ratio(summary.sharpeP25)} to{" "}
              {ratio(summary.sharpeP75)}
            </span>
          </header>
          <div className="table-scroll">
            <table className="trades-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Sharpe</th>
                  <th>Return</th>
                  <th>vs hold</th>
                  <th>Max DD</th>
                </tr>
              </thead>
              <tbody>
                {result.crossSection.rows.map((row) => (
                  <tr key={row.symbol}>
                    <td>
                      <span className="badge">{row.symbol}</span>
                    </td>
                    <td>{ratio(row.sharpe)}</td>
                    <td className={tone(row.totalReturn)}>{signedPct(row.totalReturn)}</td>
                    <td className={tone(row.excessReturn)}>{signedPct(row.excessReturn)}</td>
                    <td>{pct(row.maxDrawdown, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
