import { useState, type FormEvent } from "react";
import type { CustomStrategy, SipPayload, SipResult } from "../types";
import { LineChart, signedPct } from "./Research";
import { STRATEGY_FORMS, type BuiltInStrategyId } from "./StrategyTrainer";

interface SipPlannerProps {
  onRun: (payload: SipPayload) => Promise<SipResult>;
  customStrategies?: CustomStrategy[];
}

const PRESETS = [
  { label: "NIFTY 50", symbol: "^NSEI" },
  { label: "NIFTY Bank", symbol: "^NSEBANK" },
  { label: "Reliance", symbol: "RELIANCE.NS" },
  { label: "S&P 500 ETF", symbol: "SPY" },
  { label: "Nasdaq 100 ETF", symbol: "QQQ" },
  { label: "Gold ETF (India)", symbol: "GOLDBEES.NS" },
];

const money = (value: number, currency: string) =>
  value.toLocaleString(undefined, { style: "currency", currency, maximumFractionDigits: 0 });

function Outcome({
  title,
  hint,
  outcome,
  invested,
  currency,
  rate,
  rateLabel,
}: {
  title: string;
  hint: string;
  outcome: SipResult["sip"];
  invested: number;
  currency: string;
  rate: number | null | undefined;
  rateLabel: string;
}) {
  return (
    <div className="stat-card">
      <span className="label">{title}</span>
      <strong className="value">{money(outcome.value, currency)}</strong>
      <span className={`subtle ${outcome.gain >= 0 ? "positive" : "negative"}`}>
        {outcome.gain >= 0 ? "+" : "−"}
        {money(Math.abs(outcome.gain), currency)} on {money(invested, currency)} · {rateLabel}{" "}
        {rate === null || rate === undefined ? "–" : signedPct(rate)}
      </span>
      <span className="subtle">
        {hint} Tax if redeemed today: {money(outcome.taxIfRedeemed, currency)}
        {outcome.cashWaiting ? ` · ${money(outcome.cashWaiting, currency)} still waiting in cash` : ""}
      </span>
    </div>
  );
}

export function SipPlanner({ onRun, customStrategies = [] }: SipPlannerProps) {
  const [symbol, setSymbol] = useState("^NSEI");
  const [monthly, setMonthly] = useState(10000);
  const [stepUp, setStepUp] = useState(10);
  const [years, setYears] = useState(3);
  const [timing, setTiming] = useState("sma-crossover");
  const [result, setResult] = useState<SipResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const custom = timing.startsWith("custom:") ? customStrategies.find((s) => `custom:${s.id}` === timing) : null;
    try {
      setResult(
        await onRun({
          symbol,
          monthly,
          stepUp: stepUp / 100,
          years,
          timingStrategy: timing === "none" ? null : custom ? "custom" : timing,
          timingRules: custom ? custom.rules : null,
        }),
      );
    } catch (runError) {
      setResult(null);
      setError(runError instanceof Error ? runError.message : "The SIP simulation failed.");
    } finally {
      setBusy(false);
    }
  };

  const currency = result?.currency ?? "INR";
  return (
    <section className="sip-planner">
      <header className="header">
        <div>
          <span className="eyebrow">Systematic investing</span>
          <h1>SIP planner</h1>
          <p>
            See what a monthly SIP would have grown to, compared with investing the same money at once, and with a SIP that waits
            in cash until a strategy says the trend is up. Returns are XIRR, and tax is estimated as if you redeemed everything
            today.
          </p>
        </div>
      </header>

      <form className="panel" onSubmit={submit}>
        <div className="preset-row">
          <span className="hint">Popular:</span>
          {PRESETS.map((preset) => (
            <button key={preset.symbol} type="button" className="button-ghost chip" onClick={() => setSymbol(preset.symbol)}>
              {preset.label}
            </button>
          ))}
        </div>
        <div className="form-grid wide-grid">
          <label>
            <span>Symbol or fund</span>
            <input value={symbol} maxLength={20} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
          </label>
          <label>
            <span>Monthly amount</span>
            <input type="number" min={1} step="any" value={monthly} onChange={(e) => setMonthly(Math.max(1, Number(e.target.value)))} />
          </label>
          <label>
            <span>Yearly step-up (%)</span>
            <input type="number" min={0} max={50} value={stepUp} onChange={(e) => setStepUp(Math.max(0, Math.min(50, Number(e.target.value))))} />
          </label>
          <label>
            <span>Years</span>
            <select value={years} onChange={(e) => setYears(Number(e.target.value))}>
              {[1, 2, 3, 4].map((y) => (
                <option key={y} value={y}>
                  {y} {y === 1 ? "year" : "years"}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Timed SIP: invest only while</span>
            <select value={timing} onChange={(e) => setTiming(e.target.value)}>
              <option value="none">(no timed SIP)</option>
              {(Object.keys(STRATEGY_FORMS) as BuiltInStrategyId[])
                .filter((id) => id !== "buy-hold")
                .map((id) => (
                  <option key={id} value={id}>
                    {STRATEGY_FORMS[id].label} says buy
                  </option>
                ))}
              {customStrategies.map((s) => (
                <option key={s.id} value={`custom:${s.id}`}>
                  Custom: {s.name}
                </option>
              ))}
            </select>
          </label>
        </div>
        {error ? <div className="error-banner">{error}</div> : null}
        <div className="actions">
          <button type="submit" disabled={busy}>
            {busy ? "Simulating…" : "Simulate SIP"}
          </button>
        </div>
      </form>

      {result ? (
        <>
          <div className="stats-grid">
            <Outcome
              title="Monthly SIP"
              hint={`${result.months} instalments, the last one ${money(result.lastInstalment, currency)}.`}
              outcome={result.sip}
              invested={result.invested}
              currency={currency}
              rate={result.sip.xirr}
              rateLabel="XIRR"
            />
            <Outcome
              title="Lump sum on day one"
              hint="The same total, invested at the start."
              outcome={result.lumpSum}
              invested={result.invested}
              currency={currency}
              rate={result.lumpSum.cagr}
              rateLabel="CAGR"
            />
            {result.timed ? (
              <Outcome
                title={`Timed SIP (${result.timingStrategy})`}
                hint="Each instalment waits in cash until the signal is on."
                outcome={result.timed}
                invested={result.invested}
                currency={currency}
                rate={result.timed.xirr}
                rateLabel="XIRR"
              />
            ) : null}
          </div>
          <div className="panel">
            <LineChart
              label={`${result.symbol}: value of each approach vs money put in`}
              dates={result.series.map((p) => p.date)}
              series={[
                { label: "SIP", color: "#e3c27f", values: result.series.map((p) => p.sip) },
                { label: "Lump sum", color: "#8fb7ff", values: result.series.map((p) => p.lumpSum), dashed: true },
                ...(result.timed ? [{ label: "Timed SIP", color: "#5fd49a", values: result.series.map((p) => p.timed) }] : []),
                { label: "Invested", color: "rgba(238,232,220,0.45)", values: result.series.map((p) => p.invested), dashed: true },
              ]}
              format={(v) => money(v, currency)}
              height={240}
            />
            <p className="subtle">
              Lump sum usually wins in rising markets because the money is invested longer; SIPs reduce the regret of investing
              everything right before a fall. Fees and slippage are included on every purchase.{" "}
              {result.region === "IN"
                ? "Tax uses Indian equity rules (12.5% LTCG above ₹1.25 lakh, 20% STCG, plus cess), lot by lot."
                : "Tax uses US short- and long-term rates, lot by lot."}
            </p>
          </div>
        </>
      ) : null}
    </section>
  );
}
