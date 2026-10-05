import { useEffect, useState } from "react";
import type { SharedReport, StrategyRules } from "../types";
import { LogoMark } from "./Icons";
import { FindingsList, VerdictBadge } from "./Insights";
import { RobustnessResults } from "./Research";
import { BacktestResults } from "./StrategyTrainer";
import { describeRules } from "./StrategyBuilder";

interface ReportPageProps {
  reportId: string;
  onLoad: (id: string) => Promise<SharedReport>;
  /** Saves a shared custom strategy to the signed-in viewer's account (absent when signed out). */
  onSaveRules?: (name: string, rules: StrategyRules) => Promise<void>;
}

/** A shared backtest report, readable by anyone with the link. */
export function ReportPage({ reportId, onLoad, onSaveRules }: ReportPageProps) {
  const [report, setReport] = useState<SharedReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);

  useEffect(() => {
    onLoad(reportId)
      .then((data) => {
        setReport(data);
        document.title = `${data.title} · Algo Trade Simulator`;
      })
      .catch((loadError) => setError(loadError instanceof Error ? loadError.message : "This report couldn't be loaded."));
  }, [reportId, onLoad]);

  const rules = report?.content.backtest.rules ?? null;
  return (
    <div className="report-page">
      <header className="report-header">
        <a href="/" className="brand-row">
          <LogoMark size={36} />
          <strong>Algo Trade Simulator</strong>
        </a>
        <a className="button-ghost link-button" href="/">
          Try it yourself
        </a>
      </header>
      {error ? <div className="error-banner">{error}</div> : null}
      {!report && !error ? <p className="empty">Loading the report…</p> : null}
      {report ? (
        <main className="report-body">
          <div className="header">
            <div>
              <span className="eyebrow">Shared backtest</span>
              <h1>{report.title}</h1>
              <p className="subtle">
                By {report.author} · {new Date(report.createdAt).toLocaleDateString(undefined, { dateStyle: "medium" })} · results
                computed by the server on real prices, with fees, slippage and next-day fills
              </p>
              {rules ? <p>{describeRules(rules)}</p> : null}
            </div>
          </div>
          <div className="panel">
            <VerdictBadge verdict={report.content.review.verdict} />
            <FindingsList findings={report.content.review.findings} />
          </div>
          <div className="panel">
            <BacktestResults training={report.content.backtest} />
          </div>
          {report.content.robustness ? (
            <div className="panel">
              <RobustnessResults result={report.content.robustness} />
            </div>
          ) : null}
          <div className="panel report-cta">
            <h2>Test your own idea</h2>
            <p className="subtle">
              Backtest any stock with 6 strategies or your own rules, check it for overfitting, and paper-trade it with alerts.
              For education only; not financial advice.
            </p>
            <div className="actions">
              <a className="link-button" href="/">
                Open Algo Trade Simulator
              </a>
              {rules && onSaveRules ? (
                <button
                  type="button"
                  className="button-ghost"
                  disabled={saved !== null}
                  onClick={() =>
                    void onSaveRules(report.title.slice(0, 60), rules)
                      .then(() => setSaved("Saved to your strategies."))
                      .catch((saveError) => setSaved(saveError instanceof Error ? saveError.message : "Couldn't save."))
                  }
                >
                  Save these rules to my account
                </button>
              ) : null}
              {saved ? <span className="subtle">{saved}</span> : null}
            </div>
          </div>
        </main>
      ) : null}
    </div>
  );
}
