import { useState } from "react";
import type { FactorAttribution, OptimizeResult, ReviewFinding, ReviewResult, ReviewVerdict, TaxReport } from "../types";
import { pct, signedPct } from "./Research";

const amount = (value: number, currency: string) =>
  value.toLocaleString(undefined, { style: "currency", currency, maximumFractionDigits: 0 });
const ratio = (value: number | null | undefined) =>
  value === null || value === undefined || !Number.isFinite(value) ? "–" : value.toFixed(2);

/** Capital-gains tax on the backtest's closed trades, by tax year. */
export function TaxCard({ tax }: { tax: TaxReport }) {
  const paid = tax.totalTax > 0;
  return (
    <div className="insight-card">
      <header>
        <span className="eyebrow">After tax</span>
        <strong>
          {signedPct(tax.afterTaxReturn)} <small className="subtle">vs {signedPct(tax.preTaxReturn)} before tax</small>
        </strong>
        <span className="subtle">
          {tax.rules}. On {amount(tax.capital, tax.currency)} starting capital: {amount(tax.totalTax, tax.currency)} tax
          {tax.carryForwardLoss > 0 ? `, ${amount(tax.carryForwardLoss, tax.currency)} of losses carried forward` : ""}.
        </span>
      </header>
      {paid ? (
        <div className="risk-table-wrap">
          <table className="trades-table">
            <thead>
              <tr>
                <th>Tax year</th>
                <th>Short-term gain</th>
                <th>Long-term gain</th>
                <th>Tax</th>
              </tr>
            </thead>
            <tbody>
              {tax.years.map((row) => (
                <tr key={row.year}>
                  <td>{row.year}</td>
                  <td>{amount(row.shortTermGain, tax.currency)}</td>
                  <td>{amount(row.longTermGain, tax.currency)}</td>
                  <td>{amount(row.tax, tax.currency)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="subtle">No tax: no year ended with a taxable net gain.</p>
      )}
      <p className="subtle">Estimates for education; surcharges, state taxes and personal circumstances are not included.</p>
    </div>
  );
}

const LEVEL_ICON: Record<ReviewFinding["level"], string> = { good: "✓", warn: "!", bad: "✕" };

export function VerdictBadge({ verdict }: { verdict: ReviewVerdict }) {
  const tone = verdict.bad >= 2 ? "bad" : verdict.bad === 1 ? "warn" : verdict.label === "Promising" ? "good" : "warn";
  return (
    <div className={`verdict verdict-${tone}`}>
      <strong>{verdict.label}</strong>
      <span>{verdict.text}</span>
      <small className="subtle">
        {verdict.good} passed · {verdict.warnings} warnings · {verdict.bad} red flags
      </small>
    </div>
  );
}

export function FindingsList({ findings }: { findings: ReviewFinding[] }) {
  return (
    <ul className="findings">
      {findings.map((f) => (
        <li key={f.title} className={`finding-${f.level}`}>
          <i aria-hidden="true">{LEVEL_ICON[f.level]}</i>
          <div>
            <strong>{f.title}</strong>
            <span>{f.detail}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}

/** The sceptical-quant review of a backtest. */
export function ReviewPanel({ review }: { review: ReviewResult }) {
  return (
    <div className="walk-forward review-panel">
      <header>
        <span className="eyebrow">Backtest review</span>
        <strong>{review.symbol} · should you trust this result?</strong>
        <span className="subtle">
          Every check below is computed from the numbers; the summary {review.summary ? "was written by AI from them" : "is rule-based"}.
        </span>
      </header>
      <VerdictBadge verdict={review.verdict} />
      {review.summary ? <p className="model-verdict">{review.summary}</p> : null}
      <FindingsList findings={review.findings} />
    </div>
  );
}

const paramText = (params: Record<string, number>) =>
  Object.entries(params)
    .filter(([key]) => !["trainWindow", "horizon", "modelType"].includes(key))
    .map(([key, value]) => `${key} ${value}`)
    .join(", ");

/** The best setting found, with the guardrails that say whether to believe it. */
export function OptimizePanel({ result, onUse }: { result: OptimizeResult; onUse?: (params: Record<string, number>) => void }) {
  return (
    <div className="walk-forward optimize-panel">
      <header>
        <span className="eyebrow">Optimiser</span>
        <strong>
          Best of {result.trials} settings: {paramText(result.best.parameters)}
        </strong>
        <span className="subtle">
          Sharpe {ratio(result.best.sharpe)} and {signedPct(result.best.totalReturn)} return (yours:{" "}
          {paramText(result.current.parameters)}, Sharpe {ratio(result.current.sharpe)}).
        </span>
      </header>
      <ul className="model-stats">
        <li>
          <span>Deflated Sharpe</span>
          <strong className={(result.deflatedSharpe?.deflatedSharpe ?? 0) >= 0.95 ? "positive" : "negative"}>
            {pct(result.deflatedSharpe?.deflatedSharpe, 0)}
          </strong>
          <small>confidence it isn't luck (95% bar)</small>
        </li>
        <li>
          <span>Probability of overfitting</span>
          <strong className={(result.pbo?.pbo ?? 1) < 0.5 ? "positive" : "negative"}>{pct(result.pbo?.pbo, 0)}</strong>
          <small>winner falls to the bottom half out of sample</small>
        </li>
        {result.walkForward ? (
          <li>
            <span>Walk-forward (out of sample)</span>
            <strong>{signedPct(result.walkForward.outOfSampleAnnualized)}</strong>
            <small>a year, vs {signedPct(result.walkForward.inSampleAnnualized)} tuned</small>
          </li>
        ) : null}
      </ul>
      {result.warnings.length ? (
        <ul className="findings">
          {result.warnings.map((warning) => (
            <li key={warning} className="finding-bad">
              <i aria-hidden="true">✕</i>
              <div>
                <span>{warning}</span>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="model-verdict positive">The best setting passes the overfitting checks. Still paper-trade it first.</p>
      )}
      {onUse ? (
        <div className="actions">
          <button type="button" className="button-ghost" onClick={() => onUse(result.best.parameters)}>
            {result.trustworthy ? "Use these settings" : "Use them anyway (likely overfit)"}
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** Fama-French five factors plus momentum: what explains the strategy's returns. */
export function FactorTable({ factors }: { factors: FactorAttribution }) {
  return (
    <div className="factor-table">
      <span className="subtle">
        Factor attribution ({factors.days} days to {factors.end}, R² {ratio(factors.rSquared)}): alpha{" "}
        <strong className={factors.alphaSignificant && factors.alpha > 0 ? "positive" : ""}>{signedPct(factors.alpha)}</strong> a
        year, t = {factors.alphaT.toFixed(1)}
        {factors.alphaSignificant ? " (significant)" : " (not significant: |t| < 2)"}.
      </span>
      <div className="risk-table-wrap">
        <table className="trades-table">
          <thead>
            <tr>
              <th>Factor</th>
              <th>Loading</th>
              <th>t-stat</th>
            </tr>
          </thead>
          <tbody>
            {factors.loadings.map((row) => (
              <tr key={row.factor}>
                <td title={row.factor}>{row.label}</td>
                <td>{row.beta.toFixed(2)}</td>
                <td className={Math.abs(row.t) >= 2 ? "positive" : "subtle"}>{row.t.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** A shareable link to a report, with a copy button. */
export function ShareBox({ path, onClose }: { path: string; onClose: () => void }) {
  const url = `${window.location.origin}${path}`;
  const [copied, setCopied] = useState(false);
  return (
    <div className="share-box" role="status">
      <span className="eyebrow">Shareable report</span>
      <div className="share-row">
        <input readOnly value={url} aria-label="Report link" onFocus={(e) => e.currentTarget.select()} />
        <button
          type="button"
          className="button-ghost"
          onClick={() => {
            void navigator.clipboard
              ?.writeText(url)
              .then(() => setCopied(true))
              .catch(() => setCopied(false));
          }}
        >
          {copied ? "Copied" : "Copy"}
        </button>
        <a className="button-ghost link-button" href={url} target="_blank" rel="noreferrer">
          Open
        </a>
        <button type="button" className="button-ghost" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>
      <span className="subtle">
        Anyone with the link can view it. The server re-ran the backtest, so the numbers can't be edited.
      </span>
    </div>
  );
}
