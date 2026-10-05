import { useId, useState } from "react";
import type {
  BenchmarkStats,
  ModelReport,
  MonthlyReturn,
  RiskStats,
  RobustnessResult,
  StrategyMetrics,
  WalkForwardResult,
} from "../types";

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
  const importance = report.weightKind === "importance";
  const horizon = report.horizon ?? 1;
  return (
    <div className="model-report">
      <header>
        <span className="eyebrow">Machine-learning model</span>
        <strong>{report.model}</strong>
        <span className="subtle">
          Predicts whether the price is higher {horizon === 1 ? "tomorrow" : `in ${horizon} trading days`} from{" "}
          {report.features?.length ?? report.featureWeights.length} features. Trained on the previous {report.trainWindow} trading
          days, refitted every {report.retrainEvery} days ({report.refits} fits). Each prediction only uses data up to that day.
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
        <span className="subtle">
          {importance
            ? "What the latest model relies on most (share of the trees' total split gain)"
            : 'What the latest model weighs most (standardised coefficients; right pushes towards "up")'}
        </span>
        {report.featureWeights.map((w) => (
          <div key={w.feature} className="feature-row">
            <span>{w.feature}</span>
            <div className={`feature-bar${importance ? " importance" : ""}`}>
              <i
                className={w.weight >= 0 ? "up" : "down"}
                style={
                  importance
                    ? { width: `${(w.weight / top) * 100}%`, left: 0 }
                    : {
                        width: `${(Math.abs(w.weight) / top) * 50}%`,
                        left: w.weight >= 0 ? "50%" : `${50 - (Math.abs(w.weight) / top) * 50}%`,
                      }
                }
              />
            </div>
            <small>{importance ? pct(w.weight, 1) : `${w.weight >= 0 ? "+" : "−"}${Math.abs(w.weight).toFixed(3)}`}</small>
          </div>
        ))}
      </div>
      {report.calibration?.length ? <CalibrationTable rows={report.calibration} /> : null}
      {report.thresholdScan?.length ? (
        <ThresholdScan rows={report.thresholdScan} suggested={report.suggestedThreshold ?? null} current={report.threshold} />
      ) : null}
    </div>
  );
}

/** Predicted probability vs how often the price really rose: a well-calibrated model sits on the diagonal. */
function CalibrationTable({ rows }: { rows: NonNullable<ModelReport["calibration"]> }) {
  return (
    <div className="calibration">
      <span className="subtle">
        Calibration: when the model says 55%, does the price rise 55% of the time? Each bar is a tenth of the predictions.
      </span>
      <div className="calibration-bars" role="img" aria-label="Predicted versus actual frequency of rises">
        {rows.map((row, index) => (
          <div key={index} className="calibration-bar" title={`Predicted ${pct(row.predicted, 1)}, rose ${pct(row.actual, 1)} of ${row.count} days`}>
            <i className="actual" style={{ height: `${row.actual * 100}%` }} />
            <i className="predicted" style={{ bottom: `${row.predicted * 100}%` }} />
            <small>{(row.predicted * 100).toFixed(0)}</small>
          </div>
        ))}
      </div>
      <span className="subtle calibration-legend">
        <i className="swatch actual" /> share of days the price rose <i className="swatch predicted" /> average predicted chance
      </span>
    </div>
  );
}

/** Net-of-cost return of each buy threshold on the days before the backtest window. */
function ThresholdScan({
  rows,
  suggested,
  current,
}: {
  rows: NonNullable<ModelReport["thresholdScan"]>;
  suggested: number | null;
  current: number;
}) {
  const top = Math.max(...rows.map((r) => Math.abs(r.return)), 1e-9);
  return (
    <div className="threshold-scan">
      <span className="subtle">
        Choosing the threshold out of sample: each threshold's return after costs on the days <em>before</em> the backtest
        window. {suggested !== null ? `The best there was ${suggested.toFixed(2)}` : "Not enough earlier history to choose one"}
        {suggested !== null && Math.abs(suggested - current) > 1e-9 ? `; this backtest used ${current.toFixed(2)}.` : "."}
      </span>
      <div className="scan-bars">
        {rows.map((row) => (
          <div
            key={row.threshold}
            className={`scan-bar${row.threshold === suggested ? " best" : ""}${Math.abs(row.threshold - current) < 1e-9 ? " current" : ""}`}
            title={`${row.threshold.toFixed(2)}: ${signedPct(row.return)} with ${row.trades} trades`}
          >
            <i
              className={row.return >= 0 ? "up" : "down"}
              style={{ height: `${(Math.abs(row.return) / top) * 50}%`, [row.return >= 0 ? "bottom" : "top"]: "50%" }}
            />
          </div>
        ))}
      </div>
      <div className="value-chart-axis">
        <span>{rows[0].threshold.toFixed(2)}</span>
        <span>buy threshold</span>
        <span>{rows[rows.length - 1].threshold.toFixed(2)}</span>
      </div>
    </div>
  );
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Background for a return: green for gains, red for losses, stronger for bigger moves. */
export function heat(value: number | null | undefined, scale: number) {
  if (value === null || value === undefined || !Number.isFinite(value)) return undefined;
  const alpha = Math.min(Math.abs(value) / (scale || 1), 1) * 0.55 + 0.06;
  return value >= 0 ? `rgba(95, 212, 154, ${alpha})` : `rgba(240, 122, 122, ${alpha})`;
}

/** Calendar of monthly returns, one row per year, with the year's total. */
export function MonthlyHeatmap({ monthly }: { monthly: MonthlyReturn[] }) {
  if (!monthly.length) return null;
  const byYear = new Map<string, Map<number, number>>();
  for (const m of monthly) {
    const [year, month] = m.month.split("-");
    if (!byYear.has(year)) byYear.set(year, new Map());
    byYear.get(year)!.set(Number(month) - 1, m.return);
  }
  const scale = Math.max(...monthly.map((m) => Math.abs(m.return)), 0.01);
  return (
    <div className="risk-table-wrap">
      <table className="heatmap-table">
        <caption>Monthly returns (partial first and last months)</caption>
        <thead>
          <tr>
            <th scope="col">Year</th>
            {MONTHS.map((m) => (
              <th key={m} scope="col">
                {m}
              </th>
            ))}
            <th scope="col">Year</th>
          </tr>
        </thead>
        <tbody>
          {Array.from(byYear.entries()).map(([year, months]) => {
            const total = Array.from(months.values()).reduce((acc, r) => acc * (1 + r), 1) - 1;
            return (
              <tr key={year}>
                <th scope="row">{year}</th>
                {MONTHS.map((_, i) => {
                  const value = months.get(i);
                  return (
                    <td key={i} style={{ background: heat(value, scale) }}>
                      {value === undefined ? "" : (value * 100).toFixed(1)}
                    </td>
                  );
                })}
                <td className={total >= 0 ? "positive" : "negative"}>{signedPct(total)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Grid of Sharpe ratios over the strategy's settings; a broad bright region means a robust strategy. */
function SensitivityHeatmap({ grid }: { grid: NonNullable<RobustnessResult["sensitivity"]> }) {
  const twoD = Boolean(grid.yParam);
  const rows = twoD ? grid.yValues : [null];
  const cell = (x: number, y: number | null) => grid.cells.find((c) => c.x === x && c.y === y);
  const isCurrent = (x: number, y: number | null) =>
    grid.current[grid.xParam] === x && (!twoD || grid.current[grid.yParam as string] === y);
  const scale = Math.max(...grid.cells.filter((c) => c.valid).map((c) => Math.abs(c.sharpe ?? 0)), 0.25);
  return (
    <div className="risk-table-wrap">
      <table className="heatmap-table sensitivity">
        <caption>
          Sharpe ratio by setting ({grid.xParam}
          {twoD ? ` across, ${grid.yParam} down` : ""}); the outlined cell is the one you tested
        </caption>
        <thead>
          <tr>
            <th scope="col">{twoD ? `${grid.yParam} \\ ${grid.xParam}` : grid.xParam}</th>
            {grid.xValues.map((x) => (
              <th key={x} scope="col">
                {x}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((y) => (
            <tr key={String(y)}>
              <th scope="row">{y ?? "Sharpe"}</th>
              {grid.xValues.map((x) => {
                const c = cell(x, y);
                return (
                  <td
                    key={x}
                    className={isCurrent(x, y) ? "current" : undefined}
                    style={{ background: c?.valid ? heat(c.sharpe, scale) : undefined }}
                    title={
                      c?.valid
                        ? `Return ${signedPct(c.totalReturn)}, max drawdown ${pct(c.maxDrawdown, 1)}, ${c.trades} trades`
                        : "Not a valid setting"
                    }
                  >
                    {c?.valid ? ratio(c.sharpe) : "·"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const verdictClass = (good: boolean) => (good ? "positive" : "negative");

/** Monte Carlo bands, parameter sensitivity, deflated Sharpe and probability of overfitting. */
export function RobustnessResults({ result }: { result: RobustnessResult }) {
  const mc = result.monteCarlo;
  const dsr = result.deflatedSharpe;
  const pbo = result.pbo;
  const grid = result.sensitivity;
  return (
    <div className="walk-forward robustness">
      <header>
        <span className="eyebrow">Robustness check</span>
        <strong>
          {result.symbol} · luck, skill or overfitting?
        </strong>
        <span className="subtle">
          Four independent checks on the same backtest window ({shortDate(result.period.start)} – {shortDate(result.period.end)}).
        </span>
      </header>

      {mc ? (
        <>
          <ul className="model-stats">
            <li>
              <span>Median outcome</span>
              <strong>{signedPct(mc.finalReturn.p50)}</strong>
              <small>
                90% range {signedPct(mc.finalReturn.p5)} to {signedPct(mc.finalReturn.p95)}
              </small>
            </li>
            <li>
              <span>Chance of losing money</span>
              <strong className={verdictClass(mc.probLoss < 0.25)}>{pct(mc.probLoss, 0)}</strong>
              <small>over a window this long</small>
            </li>
            <li>
              <span>Chance of beating buy & hold</span>
              <strong className={verdictClass((mc.probBeatBuyHold ?? 0) >= 0.5)}>{pct(mc.probBeatBuyHold, 0)}</strong>
              <small>resampled on the same days</small>
            </li>
            <li>
              <span>Bad-case drawdown</span>
              <strong>{pct(mc.maxDrawdown.p95, 1)}</strong>
              <small>1 in 20 paths fell further (median {pct(mc.maxDrawdown.p50, 1)})</small>
            </li>
          </ul>
          <LineChart
            label={`Monte Carlo: ${mc.paths} resampled paths (${mc.block}-day blocks), growth of $1`}
            dates={mc.fan.map((p) => p.timestamp)}
            series={[
              { label: "95th pct", color: "rgba(95,212,154,0.7)", values: mc.fan.map((p) => p.p95), dashed: true },
              { label: "75th", color: "rgba(227,194,127,0.55)", values: mc.fan.map((p) => p.p75), dashed: true },
              { label: "Median", color: "#e3c27f", values: mc.fan.map((p) => p.p50) },
              { label: "25th", color: "rgba(227,194,127,0.55)", values: mc.fan.map((p) => p.p25), dashed: true },
              { label: "5th pct", color: "rgba(240,122,122,0.75)", values: mc.fan.map((p) => p.p5), dashed: true },
            ]}
            format={(v) => `$${v.toFixed(2)}`}
          />
        </>
      ) : (
        <p className="empty">Not enough trading days for a Monte Carlo test.</p>
      )}

      {dsr || pbo ? (
        <ul className="model-stats">
          {dsr ? (
            <>
              <li>
                <span>Deflated Sharpe ratio</span>
                <strong className={verdictClass(dsr.deflatedSharpe >= 0.95)}>{pct(dsr.deflatedSharpe, 0)}</strong>
                <small>
                  confidence the Sharpe ({ratio(dsr.sharpe)}) beats the best of {dsr.trials} random tries (
                  {ratio(dsr.expectedMaxSharpe)})
                </small>
              </li>
              <li>
                <span>Probabilistic Sharpe</span>
                <strong>{pct(dsr.probabilisticSharpe, 0)}</strong>
                <small>confidence the true Sharpe is above 0</small>
              </li>
            </>
          ) : null}
          {pbo ? (
            <li>
              <span>Probability of overfitting</span>
              <strong className={verdictClass(pbo.pbo < 0.5)}>{pct(pbo.pbo, 0)}</strong>
              <small>
                in-sample winner landed in the bottom half out of sample in {pct(pbo.pbo, 0)} of {pbo.combinations} splits
              </small>
            </li>
          ) : null}
        </ul>
      ) : null}

      {grid ? (
        <>
          <SensitivityHeatmap grid={grid} />
          <p className="model-verdict">
            {grid.positiveShare >= 0.7
              ? `${pct(grid.positiveShare, 0)} of the settings had a positive Sharpe ratio: the idea works across a broad range, not just at one lucky setting.`
              : grid.positiveShare >= 0.4
                ? `Only ${pct(grid.positiveShare, 0)} of the settings had a positive Sharpe ratio: results depend noticeably on the exact settings.`
                : `Just ${pct(grid.positiveShare, 0)} of the settings had a positive Sharpe ratio: a good result here is most likely luck or curve fitting.`}{" "}
            Median Sharpe across settings {ratio(grid.medianSharpe)}, best {ratio(grid.bestSharpe)}. The grid uses full-size
            positions and the same costs.
          </p>
        </>
      ) : null}
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
