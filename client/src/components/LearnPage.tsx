import { useState, type ReactNode } from "react";
import type { OptimizeResult, TrainingPayload, TrainingResult } from "../types";
import { OptimizePanel } from "./Insights";
import { pct, signedPct } from "./Research";

interface LearnPageProps {
  runBacktest: (payload: TrainingPayload) => Promise<TrainingResult>;
  runOptimize: (payload: TrainingPayload) => Promise<OptimizeResult>;
  onStartTour: () => void;
}

type Row = { label: string; result: TrainingResult };

function ResultsTable({ rows }: { rows: Row[] }) {
  return (
    <div className="risk-table-wrap">
      <table className="trades-table">
        <thead>
          <tr>
            <th>Setup</th>
            <th>Return</th>
            <th>Sharpe</th>
            <th>Max drawdown</th>
            <th>Trades</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ label, result }) => (
            <tr key={label}>
              <td>{label}</td>
              <td className={result.metrics.totalReturn >= 0 ? "positive" : "negative"}>{signedPct(result.metrics.totalReturn)}</td>
              <td>{result.metrics.sharpe.toFixed(2)}</td>
              <td>{pct(result.metrics.maxDrawdown, 1)}</td>
              <td>{result.metrics.trades}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Lesson({
  number,
  title,
  children,
  action,
  run,
}: {
  number: number;
  title: string;
  children: ReactNode;
  action: string;
  run: () => Promise<ReactNode>;
}) {
  const [output, setOutput] = useState<ReactNode>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <article className="panel lesson">
      <header>
        <span className="eyebrow">Lesson {number}</span>
        <h2>{title}</h2>
      </header>
      <div className="lesson-body">{children}</div>
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            setError(null);
            try {
              setOutput(await run());
            } catch (runError) {
              setError(runError instanceof Error ? runError.message : "The experiment failed.");
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Running on live data…" : action}
        </button>
      </div>
      {error ? <div className="error-banner">{error}</div> : null}
      {output}
    </article>
  );
}

export function LearnPage({ runBacktest, runOptimize, onStartTour }: LearnPageProps) {
  return (
    <section className="learn">
      <header className="header">
        <div>
          <span className="eyebrow">Learn by experiment</span>
          <h1>Three lessons every trader should learn</h1>
          <p>Each lesson runs a real experiment on live market data, so you see the effect yourself instead of taking it on faith.</p>
        </div>
        <button type="button" className="button-ghost" onClick={onStartTour}>
          Take the app tour
        </button>
      </header>

      <Lesson
        number={1}
        title="Buy & hold is hard to beat"
        action="Compare 4 strategies on the S&P 500"
        run={async () => {
          const base = { symbol: "SPY" } as const;
          const rows: Row[] = [];
          for (const [label, payload] of [
            ["Buy & hold", { ...base, strategyId: "buy-hold" }],
            ["SMA 20/60 crossover", { ...base, strategyId: "sma-crossover", shortWindow: 20, longWindow: 60 }],
            ["Bollinger mean reversion", { ...base, strategyId: "mean-reversion", lookback: 20, deviation: 2 }],
            ["20-day breakout", { ...base, strategyId: "trend-follow", channel: 20 }],
          ] as Array<[string, TrainingPayload]>) {
            rows.push({ label, result: await runBacktest(payload) });
          }
          const hold = rows[0].result.metrics.totalReturn;
          const beat = rows.slice(1).filter((r) => r.result.metrics.totalReturn > hold).length;
          return (
            <>
              <ResultsTable rows={rows} />
              <p className="model-verdict">
                {beat === 0
                  ? "None of the active strategies beat simply holding. In a rising market, every day out of the market is a day of missed gains, and every trade pays costs."
                  : `${beat} of 3 strategies beat holding here. Check their Sharpe ratio and drawdown too: beating buy & hold over one period is often luck.`}
              </p>
            </>
          );
        }}
      >
        <p>
          Most active strategies spend time in cash and pay fees on every trade. To be worth it, a strategy must earn more than
          the market, or the same with much less risk. Over most periods, buy &amp; hold on a broad index wins.
        </p>
      </Lesson>

      <Lesson
        number={2}
        title="Optimising settings fools you"
        action="Optimise an SMA crossover on NVDA"
        run={async () => {
          const result = await runOptimize({ symbol: "NVDA", strategyId: "sma-crossover", shortWindow: 20, longWindow: 60 });
          return <OptimizePanel result={result} />;
        }}
      >
        <p>
          Try enough settings and one will look great on past data by pure chance. The deflated Sharpe ratio asks whether the
          best result beats what the best of that many random tries would show, and the probability of overfitting checks
          whether the in-sample winner keeps winning on data it hasn't seen.
        </p>
      </Lesson>

      <Lesson
        number={3}
        title="Costs and execution decide the result"
        action="Run one strategy with optimistic and realistic assumptions"
        run={async () => {
          const base: TrainingPayload = { symbol: "AAPL", strategyId: "mean-reversion", lookback: 10, deviation: 1.5 };
          const rows: Row[] = [
            {
              label: "No slippage, fill at the signal's close",
              result: await runBacktest({ ...base, slippageBps: 0, execution: "close" }),
            },
            { label: "5 bps slippage, next-day open", result: await runBacktest({ ...base, slippageBps: 5, execution: "next_open" }) },
            {
              label: "20 bps slippage, next-day open, market impact",
              result: await runBacktest({ ...base, slippageBps: 20, execution: "next_open", marketImpact: true }),
            },
          ];
          const drop = rows[0].result.metrics.totalReturn - rows[2].result.metrics.totalReturn;
          return (
            <>
              <ResultsTable rows={rows} />
              <p className="model-verdict">
                The same rules lost {pct(drop, 1)} of return just by being honest about how trades are filled. Strategies that
                trade often are the most sensitive.
              </p>
            </>
          );
        }}
      >
        <p>
          A backtest that buys at the exact close that produced the signal, with no slippage, is impossible to trade. Real orders
          fill later, at worse prices, and big orders move the price.
        </p>
      </Lesson>
    </section>
  );
}

const TOUR_STEPS: Array<{ title: string; text: string; page?: string }> = [
  {
    title: "Welcome to Algo Trade Simulator",
    text: "Test trading strategies honestly on real prices: fees, slippage, next-day fills, taxes and overfitting checks included. Here's a 1-minute tour.",
  },
  {
    title: "Strategy lab",
    text: "Pick a stock and a strategy, run a backtest, then use Robustness check, Review and Optimise to see if the result is skill or luck. Share it with one click.",
    page: "simulations",
  },
  {
    title: "Strategy builder",
    text: "Describe a strategy in plain English or build rules by hand, backtest it, walk-forward test it, and export it to TradingView or a Jupyter notebook.",
    page: "builder",
  },
  {
    title: "Portfolio and alerts",
    text: "Invest paper money with any strategy. The Portfolio page shows every fill, today's signals, and sends email or Telegram alerts when a strategy wants to trade.",
    page: "portfolio",
  },
  {
    title: "More tools",
    text: "Portfolio lab and Leaderboard test strategies across many stocks; SIP planner and Options lab cover long-term investing and option income. Start with Learn if you're new.",
    page: "learn",
  },
];

export const TOUR_KEY = "algo-trade-tour-done";

export function OnboardingTour({ onNavigate, onClose }: { onNavigate: (page: string) => void; onClose: () => void }) {
  const [step, setStep] = useState(0);
  const current = TOUR_STEPS[step];
  const finish = () => {
    try {
      window.localStorage.setItem(TOUR_KEY, "1");
    } catch {
      /* storage blocked: the tour simply shows again next time */
    }
    onClose();
  };
  return (
    <div className="tour-backdrop" role="dialog" aria-modal="true" aria-labelledby="tour-title">
      <div className="tour-card">
        <span className="eyebrow">
          Step {step + 1} of {TOUR_STEPS.length}
        </span>
        <h2 id="tour-title">{current.title}</h2>
        <p>{current.text}</p>
        <div className="actions">
          {step < TOUR_STEPS.length - 1 ? (
            <button
              type="button"
              onClick={() => {
                const next = TOUR_STEPS[step + 1];
                if (next.page) onNavigate(next.page);
                setStep(step + 1);
              }}
            >
              Next
            </button>
          ) : (
            <button type="button" onClick={finish}>
              Start exploring
            </button>
          )}
          <button type="button" className="button-ghost" onClick={finish}>
            Skip tour
          </button>
        </div>
      </div>
    </div>
  );
}
