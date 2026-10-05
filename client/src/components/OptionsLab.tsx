import { useMemo, useState, type FormEvent } from "react";
import type { OptionsPayload, OptionsResult } from "../types";
import { LineChart, pct, signedPct } from "./Research";

interface OptionsLabProps {
  onRun: (payload: OptionsPayload) => Promise<OptionsResult>;
}

/** Standard normal CDF (Abramowitz-Stegun approximation, accurate to ~1e-7). */
function normCdf(x: number) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp((-x * x) / 2);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

export function blackScholes(kind: "call" | "put", spot: number, strike: number, years: number, rate: number, vol: number) {
  if (years <= 0 || vol <= 0) return kind === "call" ? Math.max(spot - strike, 0) : Math.max(strike - spot, 0);
  const d1 = (Math.log(spot / strike) + (rate + (vol * vol) / 2) * years) / (vol * Math.sqrt(years));
  const d2 = d1 - vol * Math.sqrt(years);
  return kind === "call"
    ? spot * normCdf(d1) - strike * Math.exp(-rate * years) * normCdf(d2)
    : strike * Math.exp(-rate * years) * normCdf(-d2) - spot * normCdf(-d1);
}

const STRATEGY_TEXT: Record<OptionsPayload["strategy"], string> = {
  "covered-call": "Own the stock and sell a call above the price: earn the premium, give up gains above the strike.",
  "cash-secured-put": "Hold cash and sell a put below the price: earn the premium, buy the dip at the strike if it falls.",
};

/** Profit at expiry per share for the strategy, at a range of stock prices. */
function PayoffCalculator() {
  const [strategy, setStrategy] = useState<OptionsPayload["strategy"]>("covered-call");
  const [spot, setSpot] = useState(100);
  const [otm, setOtm] = useState(5);
  const [days, setDays] = useState(30);
  const [vol, setVol] = useState(25);
  const strike = strategy === "covered-call" ? spot * (1 + otm / 100) : spot * (1 - otm / 100);
  const kind = strategy === "covered-call" ? "call" : "put";
  const premium = blackScholes(kind, spot, strike, days / 365, 0.04, vol / 100);
  const prices = useMemo(() => Array.from({ length: 41 }, (_, i) => spot * (0.7 + i * 0.015)), [spot]);
  const profit = prices.map((price) =>
    strategy === "covered-call"
      ? price - spot + premium - Math.max(price - strike, 0)
      : premium - Math.max(strike - price, 0),
  );
  const breakeven = strategy === "covered-call" ? spot - premium : strike - premium;
  return (
    <div className="panel">
      <header>
        <h2>Payoff at expiry</h2>
        <span className="hint">{STRATEGY_TEXT[strategy]}</span>
      </header>
      <div className="form-grid wide-grid">
        <label>
          <span>Strategy</span>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value as OptionsPayload["strategy"])}>
            <option value="covered-call">Covered call</option>
            <option value="cash-secured-put">Cash-secured put</option>
          </select>
        </label>
        <label>
          <span>Stock price</span>
          <input type="number" min={1} value={spot} onChange={(e) => setSpot(Math.max(1, Number(e.target.value)))} />
        </label>
        <label>
          <span>Strike distance (%)</span>
          <input type="number" min={0} max={30} value={otm} onChange={(e) => setOtm(Math.max(0, Math.min(30, Number(e.target.value))))} />
        </label>
        <label>
          <span>Days to expiry</span>
          <input type="number" min={1} max={365} value={days} onChange={(e) => setDays(Math.max(1, Math.min(365, Number(e.target.value))))} />
        </label>
        <label>
          <span>Implied volatility (%)</span>
          <input type="number" min={1} max={200} value={vol} onChange={(e) => setVol(Math.max(1, Math.min(200, Number(e.target.value))))} />
        </label>
      </div>
      <ul className="model-stats">
        <li>
          <span>Strike</span>
          <strong>{strike.toFixed(2)}</strong>
        </li>
        <li>
          <span>Premium (Black-Scholes)</span>
          <strong>{premium.toFixed(2)}</strong>
          <small>{pct(premium / spot, 2)} of the price</small>
        </li>
        <li>
          <span>Break-even</span>
          <strong>{breakeven.toFixed(2)}</strong>
        </li>
        <li>
          <span>Maximum profit</span>
          <strong>{(strategy === "covered-call" ? strike - spot + premium : premium).toFixed(2)}</strong>
          <small>per share</small>
        </li>
      </ul>
      <LineChart
        label="Profit per share at expiry, by the stock's price at expiry"
        dates={prices.map((price) => price.toFixed(2))}
        xFormat={(i) => `price ${prices[i].toFixed(2)}`}
        series={[{ label: "Profit", color: "#e3c27f", values: profit, fill: true }]}
        format={(v) => v.toFixed(2)}
        height={170}
      />
    </div>
  );
}

export function OptionsLab({ onRun }: OptionsLabProps) {
  const [payload, setPayload] = useState<OptionsPayload>({
    symbol: "SPY",
    strategy: "covered-call",
    otm: 0.05,
    days: 21,
    volPremium: 1.1,
    costBps: 5,
  });
  const [result, setResult] = useState<OptionsResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = <K extends keyof OptionsPayload>(key: K, value: OptionsPayload[K]) =>
    setPayload((previous) => ({ ...previous, [key]: value }));

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await onRun(payload));
    } catch (runError) {
      setResult(null);
      setError(runError instanceof Error ? runError.message : "The options backtest failed.");
    } finally {
      setBusy(false);
    }
  };

  const m = result?.metrics;
  return (
    <section className="options-lab">
      <header className="header">
        <div>
          <span className="eyebrow">Option income</span>
          <h1>Options lab</h1>
          <p>
            Explore covered calls and cash-secured puts: a payoff calculator, and a 2-year backtest that sells a new option every
            month. Historical option prices aren't freely available, so premiums are modelled with Black-Scholes from each day's
            recent volatility; treat results as estimates.
          </p>
        </div>
      </header>

      <PayoffCalculator />

      <form className="panel" onSubmit={submit}>
        <header>
          <h2>Backtest an income strategy</h2>
          <span className="hint">Sell a new option every N trading days, settle it at expiry, repeat</span>
        </header>
        <div className="form-grid wide-grid">
          <label>
            <span>Symbol</span>
            <input value={payload.symbol} maxLength={20} onChange={(e) => set("symbol", e.target.value.toUpperCase())} />
          </label>
          <label>
            <span>Strategy</span>
            <select value={payload.strategy} onChange={(e) => set("strategy", e.target.value as OptionsPayload["strategy"])}>
              <option value="covered-call">Covered call</option>
              <option value="cash-secured-put">Cash-secured put</option>
            </select>
          </label>
          <label>
            <span>Strike distance (%)</span>
            <input
              type="number"
              min={0}
              max={30}
              value={Math.round(payload.otm * 100)}
              onChange={(e) => set("otm", Math.max(0, Math.min(30, Number(e.target.value))) / 100)}
            />
          </label>
          <label>
            <span>Trading days per option</span>
            <input
              type="number"
              min={5}
              max={63}
              value={payload.days}
              onChange={(e) => set("days", Math.max(5, Math.min(63, Number(e.target.value))))}
            />
          </label>
          <label>
            <span>Implied / realised volatility</span>
            <input
              type="number"
              min={0.5}
              max={2}
              step={0.05}
              value={payload.volPremium}
              onChange={(e) => set("volPremium", Math.max(0.5, Math.min(2, Number(e.target.value))))}
            />
          </label>
          <label>
            <span>Cost per option (bps)</span>
            <input
              type="number"
              min={0}
              max={100}
              value={payload.costBps}
              onChange={(e) => set("costBps", Math.max(0, Math.min(100, Number(e.target.value))))}
            />
          </label>
        </div>
        {error ? <div className="error-banner">{error}</div> : null}
        <div className="actions">
          <button type="submit" disabled={busy}>
            {busy ? "Backtesting…" : "Run backtest"}
          </button>
        </div>
      </form>

      {result && m ? (
        <div className="panel">
          <header>
            <h2>
              {result.symbol} · {result.strategy === "covered-call" ? "covered call" : "cash-secured put"}
            </h2>
            <span className="hint">
              {result.metrics.rolls} options written, {result.metrics.assigned} finished in the money · modelled premiums
            </span>
          </header>
          <ul className="model-stats">
            <li>
              <span>Strategy return</span>
              <strong className={m.excessReturn >= 0 ? "positive" : "negative"}>{signedPct(m.totalReturn)}</strong>
              <small>vs buy &amp; hold {signedPct(m.buyHoldReturn)}</small>
            </li>
            <li>
              <span>Premium income</span>
              <strong>{pct(m.premiumYield, 1)}</strong>
              <small>a year, before payouts</small>
            </li>
            <li>
              <span>Volatility</span>
              <strong>{pct(m.volatility, 1)}</strong>
              <small>vs {pct(result.buyHold.volatility, 1)} for the stock</small>
            </li>
            <li>
              <span>Sharpe / max drawdown</span>
              <strong>
                {m.sharpe.toFixed(2)} / {pct(m.maxDrawdown, 1)}
              </strong>
              <small>
                stock: {result.buyHold.sharpe.toFixed(2)} / {pct(result.buyHold.maxDrawdown, 1)}
              </small>
            </li>
          </ul>
          <LineChart
            label="Growth of $1"
            dates={result.curve.map((p) => p.timestamp)}
            series={[
              { label: "Option strategy", color: "#e3c27f", values: result.curve.map((p) => p.equity) },
              { label: "Buy & hold", color: "#8fb7ff", values: result.curve.map((p) => p.buyHold), dashed: true },
            ]}
            format={(v) => `$${v.toFixed(2)}`}
          />
        </div>
      ) : null}
    </section>
  );
}
