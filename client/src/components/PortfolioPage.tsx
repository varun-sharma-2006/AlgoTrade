import { useCallback, useEffect, useMemo, useState } from "react";
import type { PortfolioPosition, PortfolioResponse } from "../types";
import { strategyLabel } from "./StrategyTrainer";

interface PortfolioPageProps {
  onLoad: () => Promise<PortfolioResponse>;
  onOpenSimulations: () => void;
}

const DONUT_COLORS = ["#e3c27f", "#8fb7ff", "#5fd49a", "#f0a37a", "#c9a4f5", "#7fd6d6", "#f07a9a", "#b8b8b8"];

export const money = (value: number, currency = "USD") =>
  value.toLocaleString(undefined, { style: "currency", currency, maximumFractionDigits: 2 });

const signedMoney = (value: number) => `${value >= 0 ? "+" : "−"}${money(Math.abs(value))}`;
const signedPct = (value: number) => `${value >= 0 ? "+" : "−"}${Math.abs(value * 100).toFixed(2)}%`;
const tone = (value: number) => (value > 0 ? "positive" : value < 0 ? "negative" : "");

/** Area chart of portfolio value with min/max labels and a hover read-out. */
export function ValueChart({ points }: { points: Array<{ date: string; value: number }> }) {
  const [hover, setHover] = useState<number | null>(null);
  const width = 800;
  const height = 240;
  const pad = { top: 16, right: 12, bottom: 26, left: 12 };
  if (points.length < 2) {
    return <p className="empty">The chart appears after the first full trading day.</p>;
  }
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const x = (i: number) => pad.left + (i / (points.length - 1)) * (width - pad.left - pad.right);
  const y = (v: number) => pad.top + (1 - (v - min) / range) * (height - pad.top - pad.bottom);
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
  const area = `${line} L${x(points.length - 1)},${height - pad.bottom} L${x(0)},${height - pad.bottom} Z`;
  const rising = values[values.length - 1] >= values[0];
  const stroke = rising ? "#e3c27f" : "#f07a7a";
  const active = hover !== null ? points[hover] : null;
  const ticks = [0, Math.floor((points.length - 1) / 2), points.length - 1];

  return (
    <div className="value-chart">
      <div className="value-chart-readout">
        {active ? (
          <>
            <strong>{money(active.value)}</strong>
            <span className="subtle">{new Date(active.date).toLocaleDateString(undefined, { dateStyle: "medium" })}</span>
          </>
        ) : (
          <span className="subtle">Hover the chart for daily values</span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        onMouseLeave={() => setHover(null)}
        onMouseMove={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - box.left) / box.width;
          setHover(Math.max(0, Math.min(points.length - 1, Math.round(ratio * (points.length - 1)))));
        }}
        role="img"
        aria-label={`Portfolio value from ${money(values[0])} to ${money(values[values.length - 1])}`}
      >
        <defs>
          <linearGradient id="portfolio-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={rising ? "rgba(212,175,106,0.32)" : "rgba(240,122,122,0.25)"} />
            <stop offset="100%" stopColor="rgba(212,175,106,0)" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map((f) => (
          <line
            key={f}
            x1={pad.left}
            x2={width - pad.right}
            y1={pad.top + f * (height - pad.top - pad.bottom)}
            y2={pad.top + f * (height - pad.top - pad.bottom)}
            stroke="rgba(238,232,220,0.06)"
            strokeDasharray="3 6"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        <path d={area} fill="url(#portfolio-fill)" />
        <path d={line} fill="none" stroke={stroke} strokeWidth={2} vectorEffect="non-scaling-stroke" />
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
        {ticks.map((i) => (
          <span key={i}>{new Date(points[i].date).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span>
        ))}
      </div>
      <div className="value-chart-range subtle">
        Low {money(min)} · High {money(max)}
      </div>
    </div>
  );
}

/** Donut of current value per position. */
export function AllocationDonut({ allocation }: { allocation: PortfolioResponse["allocation"] }) {
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;
  return (
    <div className="allocation">
      <svg viewBox="0 0 200 200" className="donut" role="img" aria-label="Allocation by position">
        <circle cx="100" cy="100" r={radius} fill="none" stroke="rgba(238,232,220,0.06)" strokeWidth="22" />
        {allocation.map((slice, index) => {
          const length = slice.weight * circumference;
          const dash = (
            <circle
              key={slice.id}
              cx="100"
              cy="100"
              r={radius}
              fill="none"
              stroke={DONUT_COLORS[index % DONUT_COLORS.length]}
              strokeWidth="22"
              strokeDasharray={`${Math.max(length - 2, 0)} ${circumference}`}
              strokeDashoffset={-offset}
              transform="rotate(-90 100 100)"
            />
          );
          offset += length;
          return dash;
        })}
        <text x="100" y="96" textAnchor="middle" className="donut-count">
          {allocation.length}
        </text>
        <text x="100" y="118" textAnchor="middle" className="donut-label">
          {allocation.length === 1 ? "position" : "positions"}
        </text>
      </svg>
      <ul className="allocation-legend">
        {allocation.map((slice, index) => (
          <li key={slice.id}>
            <span className="swatch" style={{ background: DONUT_COLORS[index % DONUT_COLORS.length] }} />
            <span className="legend-symbol">{slice.symbol}</span>
            <span className="subtle">{(slice.weight * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function PositionRow({ position }: { position: PortfolioPosition }) {
  const beatHold = position.buyHoldValue !== undefined ? position.value - position.buyHoldValue : null;
  return (
    <tr>
      <td>
        <span className="badge">{position.symbol}</span>
      </td>
      <td>
        <div>{position.strategy}</div>
        <span className="subtle">{strategyLabel(position.strategyId)}</span>
      </td>
      <td>
        {position.error ? (
          <span className="subtle">{position.error}</span>
        ) : (
          <span className={`status-pill ${position.inMarket ? "live" : "done"}`}>{position.inMarket ? "Invested" : "In cash"}</span>
        )}
      </td>
      <td>{position.startDate ? new Date(position.startDate).toLocaleDateString() : "–"}</td>
      <td>{money(position.startingCapital)}</td>
      <td>
        <strong>{money(position.value)}</strong>
      </td>
      <td className={tone(position.pnl)}>
        {signedMoney(position.pnl)}
        <span className="subtle">{signedPct(position.pnlPct)}</span>
      </td>
      <td className={beatHold === null || position.strategyId === "buy-hold" ? "" : tone(beatHold)}>
        {position.strategyId === "buy-hold" ? <span className="subtle">Baseline</span> : beatHold === null ? "–" : signedMoney(beatHold)}
      </td>
    </tr>
  );
}

export function PortfolioPage({ onLoad, onOpenSimulations }: PortfolioPageProps) {
  const [data, setData] = useState<PortfolioResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await onLoad());
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Unable to load your portfolio.");
    } finally {
      setLoading(false);
    }
  }, [onLoad]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const summary = data?.summary;
  const best = useMemo(
    () => (data?.positions.length ? [...data.positions].sort((a, b) => b.pnlPct - a.pnlPct)[0] : null),
    [data],
  );

  return (
    <section className="portfolio">
      <header className="header">
        <div>
          <span className="eyebrow">Paper portfolio</span>
          <h1>Portfolio</h1>
          <p>Every simulation, replayed on real prices with its strategy's buy and sell rules, fees included.</p>
        </div>
        <button type="button" className="button-ghost" onClick={() => void refresh()} disabled={loading}>
          {loading ? "Updating…" : "Refresh"}
        </button>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      {data && data.positions.length === 0 ? (
        <div className="panel portfolio-empty">
          <h2>Your portfolio is empty</h2>
          <p className="subtle">
            Launch a simulation (you can backdate it up to a year) and it will appear here with its live value.
          </p>
          <div>
            <button type="button" onClick={onOpenSimulations}>
              Create a simulation
            </button>
          </div>
        </div>
      ) : null}

      {summary && data && data.positions.length ? (
        <>
          <div className="stats-grid">
            <div className="stat-card">
              <span className="label">Portfolio value</span>
              <strong className="value">{money(summary.totalValue)}</strong>
              <span className="subtle">{money(summary.totalCapital)} invested</span>
            </div>
            <div className="stat-card">
              <span className="label">Total return</span>
              <strong className={`value plain ${tone(summary.pnl)}`}>{signedMoney(summary.pnl)}</strong>
              <span className={`subtle ${tone(summary.pnl)}`}>{signedPct(summary.pnlPct)}</span>
            </div>
            <div className="stat-card">
              <span className="label">Today</span>
              <strong className={`value plain ${tone(summary.dayChange)}`}>{signedMoney(summary.dayChange)}</strong>
              <span className={`subtle ${tone(summary.dayChange)}`}>{signedPct(summary.dayChangePct)}</span>
            </div>
          </div>

          <div className="portfolio-grid">
            <div className="panel">
              <header>
                <h2>Value over time</h2>
                <span className="hint">Simulations count at their starting capital until their start date</span>
              </header>
              <ValueChart points={data.history} />
            </div>
            <div className="panel">
              <header>
                <h2>Allocation</h2>
                <span className="hint">
                  {summary.inMarket} of {summary.positions} invested
                  {best ? ` · best: ${best.symbol} ${signedPct(best.pnlPct)}` : ""}
                </span>
              </header>
              <AllocationDonut allocation={data.allocation} />
            </div>
          </div>

          <div className="panel holdings">
            <header>
              <h2>Holdings</h2>
              <span className="hint">“vs buy &amp; hold” is how much the strategy made or lost compared with holding the stock</span>
            </header>
            <div className="table-scroll">
              <table className="simulation-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Strategy</th>
                    <th>Position</th>
                    <th>Started</th>
                    <th>Invested</th>
                    <th>Value</th>
                    <th>P&amp;L</th>
                    <th>vs buy &amp; hold</th>
                  </tr>
                </thead>
                <tbody>
                  {data.positions.map((position) => (
                    <PositionRow key={position.id} position={position} />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : null}

      {!data && loading ? <p className="empty">Valuing your simulations…</p> : null}
    </section>
  );
}
