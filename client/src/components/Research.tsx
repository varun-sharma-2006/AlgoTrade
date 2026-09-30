import { useId, useState } from "react";
import type { BenchmarkStats, ModelReport, RiskStats, StrategyMetrics, WalkForwardResult } from "../types";

export const pct = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || !Number.isFinite(value) ? "–" : `${(value * 100).toFixed(digits)}%`;
export const signedPct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "–"
    : `${value >= 0 ? "+" : "−"}${Math.abs(value * 100).toFixed(digits)}%`;
const ratio = (value: number | null | undefined) =>
  value === null || value === undefined || !Number.isFinite(value) ? "–" : value.toFixed(2);
const shortDate = (timestamp: string) =>
  new Date(timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "2-digit" });

export interface ChartSeries {
  label: string;
  color: string;
  values: Array<number | null | undefined>;
  dashed?: boolean;
  /** Shade the area between the line and zero (used for drawdowns). */
  fill?: boolean;
}

/** Line chart for one or more daily series, with a hover read-out and a legend. */
export function LineChart({
  dates,
  series,
  format,
  height = 200,
  label,
}: {
  dates: string[];
  series: ChartSeries[];
  format: (value: number) => string;
  height?: number;
  label: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const gradient = useId().replace(/:/g, "");
  const width = 800;
  const pad = { top: 12, right: 8, bottom: 8, left: 8 };
  const all = series.flatMap((s) => s.values.filter((v): v is number => typeof v === "number" && Number.isFinite(v)));
  if (dates.length < 2 || !all.length) {
    return null;
  }
  const hasFill = series.some((s) => s.fill);
  const min = Math.min(...all, hasFill ? 0 : Infinity);
  const max = Math.max(...all, hasFill ? 0 : -Infinity);
  const range = max - min || 1;
  const x = (i: number) => pad.left + (i / (dates.length - 1)) * (width - pad.left - pad.right);
  const y = (v: number) => pad.top + (1 - (v - min) / range) * (height - pad.top - pad.bottom);
  const path = (values: ChartSeries["values"]) => {
    let d = "";
    let pen = false;
    values.forEach((v, i) => {
      if (typeof v !== "number" || !Number.isFinite(v)) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  return (
    <figure className="line-chart">
      <figcaption>
        <span className="line-chart-title">{label}</span>
        <span className="line-chart-legend">
          {series.map((s) => (
            <span key={s.label}>
              <i style={{ background: s.color }} className={s.dashed ? "dashed" : undefined} />
              {s.label}
              {hover !== null && typeof s.values[hover] === "number" ? (
                <strong>{format(s.values[hover] as number)}</strong>
              ) : null}
            </span>
          ))}
          {hover !== null ? <span className="subtle">{shortDate(dates[hover])}</span> : null}
        </span>
      </figcaption>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        style={{ height }}
        role="img"
        aria-label={label}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          const position = (event.clientX - box.left) / box.width;
          setHover(Math.max(0, Math.min(dates.length - 1, Math.round(position * (dates.length - 1)))));
        }}
      >
        <defs>
          <linearGradient id={gradient} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(240,122,122,0.05)" />
            <stop offset="100%" stopColor="rgba(240,122,122,0.35)" />
          </linearGradient>
        </defs>
        {hasFill || (min < 0 && max > 0) ? (
          <line x1={pad.left} x2={width - pad.right} y1={y(0)} y2={y(0)} stroke="rgba(238,232,220,0.18)" vectorEffect="non-scaling-stroke" />
        ) : null}
        {series.map((s) =>
          s.fill ? (
            <path
              key={`${s.label}-fill`}
              d={`${path(s.values)} L${x(dates.length - 1)},${y(0)} L${x(0)},${y(0)} Z`}
              fill={`url(#${gradient})`}
            />
          ) : null,
        )}
        {series.map((s) => (
          <path
            key={s.label}
            d={path(s.values)}
            fill="none"
            stroke={s.color}
            strokeWidth={s.dashed ? 1.4 : 2}
            strokeDasharray={s.dashed ? "5 5" : undefined}
            vectorEffect="non-scaling-stroke"
          />
        ))}
        {hover !== null ? (
          <line
            x1={x(hover)}
            x2={x(hover)}
            y1={pad.top}
            y2={height - pad.bottom}
            stroke="rgba(243,220,166,0.5)"
            vectorEffect="non-scaling-stroke"
          />
        ) : null}
      </svg>
      <div className="value-chart-axis">
        <span>{shortDate(dates[0])}</span>
        <span>{shortDate(dates[Math.floor((dates.length - 1) / 2)])}</span>
        <span>{shortDate(dates[dates.length - 1])}</span>
      </div>
    </figure>
  );
}

const RISK_ROWS: Array<{ key: keyof RiskStats; label: string; hint: string; format: (v: number) => string; better: "high" | "low" }> = [
  { key: "totalReturn", label: "Total return", hint: "Over the whole backtest window", format: (v) => signedPct(v), better: "high" },
  { key: "annualizedReturn", label: "Annualised return", hint: "Compound yearly growth rate (CAGR)", format: (v) => signedPct(v), better: "high" },
  { key: "volatility", label: "Volatility", hint: "Annualised standard deviation of daily returns", format: (v) => pct(v, 1), better: "low" },
  { key: "sharpe", label: "Sharpe ratio", hint: "Return per unit of volatility, annualised", format: ratio, better: "high" },
  { key: "sortino", label: "Sortino ratio", hint: "Like Sharpe, but only losing days count as risk", format: ratio, better: "high" },
  { key: "maxDrawdown", label: "Max drawdown", hint: "Largest fall from a peak", format: (v) => pct(v, 1), better: "low" },
  { key: "calmar", label: "Calmar ratio", hint: "Annualised return divided by max drawdown", format: ratio, better: "high" },
];

/** Strategy vs buy & hold vs the S&P 500 on return and risk. */
export function RiskTable({
  metrics,
  buyHold,
  benchmark,
}: {
  metrics: StrategyMetrics;
  buyHold: RiskStats;
  benchmark?: BenchmarkStats | null;
}) {
  const strategy = metrics as unknown as RiskStats;
  return (
    <div className="risk-table-wrap">
      <table className="risk-table">
        <thead>
          <tr>
            <th scope="col">Risk & return</th>
            <th scope="col">Strategy</th>
            <th scope="col">Buy & hold</th>
            <th scope="col">{benchmark?.name ?? "S&P 500"}</th>
          </tr>
        </thead>
        <tbody>
          {RISK_ROWS.map((row) => {
            const mine = strategy[row.key];
            const theirs = buyHold[row.key];
            const wins =
              typeof mine === "number" && typeof theirs === "number" && mine !== theirs
                ? (row.better === "high") === mine > theirs
                : null;
            return (
              <tr key={row.key}>
                <th scope="row" title={row.hint}>
                  {row.label}
                </th>
                <td className={wins === null ? "" : wins ? "positive" : "negative"}>
                  {typeof mine === "number" ? row.format(mine) : "–"}
                </td>
                <td>{row.format(theirs)}</td>
                <td>{benchmark ? row.format(benchmark[row.key]) : "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {benchmark ? (
        <p className="subtle risk-note">
          Against the {benchmark.name}: beta {benchmark.beta.toFixed(2)}, correlation {benchmark.correlation.toFixed(2)}, alpha{" "}
          {signedPct(benchmark.alpha)} a year. Green or red marks where the strategy did better or worse than buy & hold.
        </p>
      ) : (
        <p className="subtle risk-note">Green or red marks where the strategy did better or worse than buy & hold.</p>
      )}
    </div>
  );
}

/** How well the machine-learning model predicted direction on days it had never seen. */
export function ModelReportCard({ report }: { report: ModelReport }) {
  const edge = report.accuracy - report.baselineAccuracy;
  const top = Math.max(...report.featureWeights.map((w) => Math.abs(w.weight)), 1e-9);
  return (
    <div className="model-report">
      <header>
        <span className="eyebrow">Machine-learning model</span>
        <strong>{report.model}</strong>
        <span className="subtle">
          Trained on the previous {report.trainWindow} trading days, refitted every {report.retrainEvery} days ({report.refits}{" "}
          fits). Each prediction only uses data up to that day.
        </span>
      </header>
      <ul className="model-stats">
        <li>
          <span>Direction accuracy</span>
          <strong>{pct(report.accuracy, 1)}</strong>
          <small>out of sample, {report.predictions} days</small>
        </li>
        <li>
          <span>Always-up baseline</span>
          <strong>{pct(report.baselineAccuracy, 1)}</strong>
          <small>guessing the majority direction</small>
        </li>
        <li>
          <span>ROC-AUC</span>
          <strong>{ratio(report.auc)}</strong>
          <small>0.5 = no skill</small>
        </li>
        <li>
          <span>Up days when long</span>
          <strong>{pct(report.precisionWhenLong, 1)}</strong>
          <small>{report.daysLong} days in the market</small>
        </li>
      </ul>
      <p className={`model-verdict ${edge > 0.01 ? "positive" : ""}`}>
        {edge > 0.01
          ? `The model beat the baseline by ${(edge * 100).toFixed(1)} points. Check the backtest too: a small accuracy edge can still lose to buy & hold after costs.`
          : "The model did not beat simply guessing the majority direction. Daily moves are close to a coin flip from price data alone, which is exactly what an honest test should reveal."}
        {report.latestProbability !== null ? ` Today's predicted chance of a rise: ${pct(report.latestProbability, 1)}.` : ""}
      </p>
      <div className="feature-weights">
        <span className="subtle">What the latest model weighs most (standardised coefficients; right pushes towards "up")</span>
        {report.featureWeights.map((w) => (
          <div key={w.feature} className="feature-row">
            <span>{w.feature}</span>
            <div className="feature-bar">
              <i
                className={w.weight >= 0 ? "up" : "down"}
                style={{
                  width: `${(Math.abs(w.weight) / top) * 50}%`,
                  left: w.weight >= 0 ? "50%" : `${50 - (Math.abs(w.weight) / top) * 50}%`,
                }}
              />
            </div>
            <small>{w.weight >= 0 ? "+" : "−"}{Math.abs(w.weight).toFixed(3)}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

const paramText = (params: Record<string, number>) =>
  Object.entries(params)
    .filter(([key]) => key !== "trainWindow")
    .map(([key, value]) => `${key} ${value}`)
    .join(", ");

/** Tune on each past year, trade the next quarter, roll forward. */
export function WalkForwardResults({ result }: { result: WalkForwardResult }) {
  const m = result.metrics;
  const decay = m.inSampleAnnualized - m.outOfSampleAnnualized;
  return (
    <div className="walk-forward">
      <header>
        <span className="eyebrow">Walk-forward test</span>
        <strong>
          {result.symbol} · {result.folds.length} out-of-sample quarters
        </strong>
        <span className="subtle">
          Each quarter, all {result.gridSize} parameter sets are backtested on the previous {result.trainDays} trading days; the one
          with the best Sharpe trades the next {result.testDays} days, which it has never seen. {result.feeBps + result.slippageBps}{" "}
          bps cost per trade.
        </span>
      </header>
      <ul className="model-stats">
        <li>
          <span>Tuned (in sample)</span>
          <strong>{signedPct(m.inSampleAnnualized)}</strong>
          <small>a year, on the data it was tuned on</small>
        </li>
        <li>
          <span>Out of sample</span>
          <strong className={m.outOfSampleAnnualized >= m.buyHoldAnnualized ? "positive" : "negative"}>
            {signedPct(m.outOfSampleAnnualized)}
          </strong>
          <small>a year, on unseen data</small>
        </li>
        <li>
          <span>Buy & hold</span>
          <strong>{signedPct(m.buyHoldAnnualized)}</strong>
          <small>a year, same period</small>
        </li>
        <li>
          <span>Sharpe (OOS vs B&H)</span>
          <strong>
            {m.outOfSampleSharpe.toFixed(2)} / {m.buyHoldSharpe.toFixed(2)}
          </strong>
          <small>beat buy & hold in {m.foldsBeatBuyHold} of {result.folds.length} quarters</small>
        </li>
      </ul>
      <p className="model-verdict">
        {decay > 0.02
          ? `Tuning flattered the strategy by ${(decay * 100).toFixed(1)} points a year: that gap is curve fitting, and the out-of-sample number is the realistic one.`
          : "Out-of-sample results held up close to the tuned ones, a sign the settings are not just fitted to the past."}{" "}
        The most common choice ({paramText(m.mostChosenParams) || "default"}) was picked in {m.mostChosenCount} of {result.folds.length}{" "}
        quarters.
      </p>
      <LineChart
        label="Out-of-sample growth of $1"
        dates={result.curve.map((p) => p.timestamp)}
        series={[
          { label: "Walk-forward", color: "#e3c27f", values: result.curve.map((p) => p.equity) },
          { label: "Buy & hold", color: "#8fb7ff", values: result.curve.map((p) => p.buyHold), dashed: true },
        ]}
        format={(v) => `$${v.toFixed(2)}`}
      />
      <div className="risk-table-wrap">
        <table className="trades-table folds-table">
          <thead>
            <tr>
              <th>Test quarter</th>
              <th>Chosen settings</th>
              <th>Tuned Sharpe</th>
              <th title="Green when it beat buy & hold that quarter">Return vs B&H</th>
              <th>Buy & hold</th>
            </tr>
          </thead>
          <tbody>
            {result.folds.map((fold) => (
              <tr key={fold.testStart}>
                <td>
                  {shortDate(fold.testStart)} – {shortDate(fold.testEnd)}
                </td>
                <td>{paramText(fold.params) || "default"}</td>
                <td>{fold.trainSharpe.toFixed(2)}</td>
                <td className={fold.testReturn >= fold.buyHoldReturn ? "positive" : "negative"}>{signedPct(fold.testReturn)}</td>
                <td>{signedPct(fold.buyHoldReturn)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
