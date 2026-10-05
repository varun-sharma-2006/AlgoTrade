import { Fragment, useState, type FormEvent } from "react";
import type { CustomStrategy, LeaderboardResult, StrategyRules } from "../types";
import { pct, signedPct } from "./Research";

interface LeaderboardProps {
  onRun: (symbols: string[], custom: Array<{ name: string; rules: StrategyRules }>) => Promise<LeaderboardResult>;
  customStrategies?: CustomStrategy[];
}

const SETS: Array<{ name: string; symbols: string[] }> = [
  { name: "US leaders", symbols: ["AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ"] },
  { name: "India leaders", symbols: ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "LT.NS"] },
  { name: "Index ETFs", symbols: ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD"] },
  { name: "Crypto", symbols: ["BTC-USD", "ETH-USD", "SOL-USD"] },
];

export function Leaderboard({ onRun, customStrategies = [] }: LeaderboardProps) {
  const [symbolsText, setSymbolsText] = useState(SETS[0].symbols.join(", "));
  const [includeCustom, setIncludeCustom] = useState(true);
  const [result, setResult] = useState<LeaderboardResult | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const symbols = Array.from(new Set(symbolsText.split(/[\s,;]+/).map((s) => s.trim().toUpperCase()).filter(Boolean)));

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const custom = includeCustom ? customStrategies.slice(0, 10).map((s) => ({ name: s.name, rules: s.rules })) : [];
      setResult(await onRun(symbols.slice(0, 10), custom));
    } catch (runError) {
      setResult(null);
      setError(runError instanceof Error ? runError.message : "The leaderboard failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="leaderboard">
      <header className="header">
        <div>
          <span className="eyebrow">Which strategy is most robust?</span>
          <h1>Leaderboard</h1>
          <p>
            Every built-in strategy and your saved ones, backtested on the same stocks and period. Ranked by a robustness score:
            the typical (median) Sharpe ratio, weighted by how often the strategy beat simply holding. A strategy that wins on
            one stock and loses on the rest ranks low.
          </p>
        </div>
      </header>
      <form className="panel" onSubmit={submit}>
        <div className="preset-row">
          <span className="hint">Test on:</span>
          {SETS.map((set) => (
            <button key={set.name} type="button" className="button-ghost chip" onClick={() => setSymbolsText(set.symbols.join(", "))}>
              {set.name}
            </button>
          ))}
        </div>
        <label>
          <span>Symbols ({symbols.length}, up to 10)</span>
          <textarea rows={2} value={symbolsText} onChange={(event) => setSymbolsText(event.target.value)} />
        </label>
        {customStrategies.length ? (
          <label className="checkbox">
            <input type="checkbox" checked={includeCustom} onChange={(e) => setIncludeCustom(e.target.checked)} />
            <span>Include my {customStrategies.length} saved strategies</span>
          </label>
        ) : null}
        {error ? <div className="error-banner">{error}</div> : null}
        <div className="actions">
          <button type="submit" disabled={busy || !symbols.length}>
            {busy ? "Ranking…" : "Rank strategies"}
          </button>
        </div>
      </form>

      {result ? (
        <div className="panel">
          <header>
            <h2>Ranking on {result.symbols.join(", ")}</h2>
            <span className="hint">
              2 years, default settings, fees and next-day fills{result.missing.length ? ` · no data for ${result.missing.join(", ")}` : ""}
              . Click a row for each stock's result.
            </span>
          </header>
          <div className="table-scroll">
            <table className="simulation-table leaderboard-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Strategy</th>
                  <th title="Median Sharpe x (0.5 + 0.5 x share of stocks with a better Sharpe than holding)">Score</th>
                  <th>Median Sharpe</th>
                  <th>Worst Sharpe</th>
                  <th>Beat holding (Sharpe)</th>
                  <th>Median return</th>
                  <th>Median max DD</th>
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row, index) => {
                  const baseline = row.strategyId === "buy-hold";
                  return (
                    <Fragment key={`${row.strategyId}-${row.name}`}>
                      <tr className="clickable" onClick={() => setOpen(open === row.name ? null : row.name)}>
                        <td>{index + 1}</td>
                        <td>
                          <strong>{row.name}</strong>
                          {baseline ? <span className="subtle"> · baseline</span> : null}
                        </td>
                        <td>{row.score.toFixed(2)}</td>
                        <td>{row.medianSharpe.toFixed(2)}</td>
                        <td className={row.worstSharpe < 0 ? "negative" : ""}>{row.worstSharpe.toFixed(2)}</td>
                        <td>{baseline ? "–" : pct(row.shareBetterSharpe, 0)}</td>
                        <td>{signedPct(row.medianReturn)}</td>
                        <td>{pct(row.medianMaxDrawdown, 1)}</td>
                      </tr>
                      {open === row.name ? (
                        <tr className="ledger-row">
                          <td colSpan={8}>
                            <div className="chip-list">
                              {row.results.map((r) => (
                                <span key={r.symbol} className={`result-chip ${r.excessReturn >= 0 ? "positive" : "negative"}`}>
                                  {r.symbol}: Sharpe {r.sharpe.toFixed(2)}, {signedPct(r.totalReturn)} ({signedPct(r.excessReturn)} vs hold)
                                </span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      ) : null}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </section>
  );
}
