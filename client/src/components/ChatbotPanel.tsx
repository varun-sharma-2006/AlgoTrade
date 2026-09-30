import { FormEvent, useEffect, useRef, useState } from "react";
import type { ChatAction, ChatMessage } from "../types";
import { FormattedText } from "./FormattedText";
import { signedPct } from "./Research";

interface ChatbotPanelProps {
  messages: ChatMessage[];
  loading: boolean;
  onSend: (message: string) => Promise<void> | void;
}

type Row = Record<string, unknown>;
const num = (row: Row, key: string) => (typeof row[key] === "number" ? (row[key] as number) : null);
const fixed = (value: number | null) => (value === null ? "–" : value.toFixed(2));

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <li>
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </li>
  );
}

/** The result of a backend action the copilot ran (backtest, comparison, walk-forward, simulation). */
export function ChatActionCard({ action }: { action: ChatAction }) {
  const data = action.data as Row;
  if (action.type === "backtest") {
    const beat = (num(data, "strategyReturn") ?? 0) >= (num(data, "buyHoldReturn") ?? 0);
    return (
      <div className="action-card">
        <strong>{action.label}</strong>
        <ul>
          <Stat label="Strategy" value={signedPct(num(data, "strategyReturn"))} tone={beat ? "positive" : "negative"} />
          <Stat label="Buy & hold" value={signedPct(num(data, "buyHoldReturn"))} />
          <Stat label="S&P 500" value={signedPct(num(data, "sp500Return"))} />
          <Stat label="Sharpe" value={fixed(num(data, "sharpe"))} />
          <Stat label="Max drawdown" value={signedPct(-(num(data, "maxDrawdown") ?? 0))} />
          <Stat label="Trades" value={String(num(data, "trades") ?? "–")} />
        </ul>
      </div>
    );
  }
  if (action.type === "comparison") {
    const rows = (data.rows as Row[]) ?? [];
    return (
      <div className="action-card">
        <strong>{action.label}</strong>
        <table className="trades-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Return</th>
              <th>Sharpe</th>
              <th>Max DD</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={String(row.strategyId)}>
                <td>{String(row.strategy)}</td>
                <td>{signedPct(num(row, "strategyReturn"))}</td>
                <td>{fixed(num(row, "sharpe"))}</td>
                <td>{signedPct(-(num(row, "maxDrawdown") ?? 0))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  if (action.type === "walkforward") {
    return (
      <div className="action-card">
        <strong>{action.label}</strong>
        <ul>
          <Stat label="Tuned, a year" value={signedPct(num(data, "tunedInSampleAnnualized"))} />
          <Stat label="Out of sample, a year" value={signedPct(num(data, "outOfSampleAnnualized"))} />
          <Stat label="Buy & hold (total)" value={signedPct(num(data, "buyHoldReturn"))} />
          <Stat label="Out of sample (total)" value={signedPct(num(data, "outOfSampleReturn"))} />
          <Stat label="Quarters beating B&H" value={`${num(data, "foldsBeatBuyHold") ?? 0} / ${num(data, "folds") ?? 0}`} />
        </ul>
      </div>
    );
  }
  return (
    <div className="action-card">
      <strong>{action.label}</strong>
    </div>
  );
}

export function ChatbotPanel({ messages, loading, onSend }: ChatbotPanelProps) {
  const [input, setInput] = useState("");
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!input.trim()) {
      return;
    }
    await onSend(input.trim());
    setInput("");
  };

  return (
    <section className="chatbot">
      <header>
        <span className="eyebrow">AI copilot</span>
        <h2>Trading copilot</h2>
        <span className="hint">Ask about any stock or strategy. It can run real backtests for you, on live market data.</span>
      </header>

      <div className="chat-window" ref={containerRef}>
        {messages.length === 0 ? (
          <p className="empty">
            Try “Backtest AAPL with SMA 20/60”, “Compare strategies on NVDA”, “Walk-forward test MSFT with Bollinger”, “Which stock
            should I buy and why?” or “What is RSI?”.
          </p>
        ) : (
          messages.map((message, index) => (
            <article key={`${message.timestamp}-${index}`} className={`bubble ${message.role}`}>
              <div className="content">
                <FormattedText text={message.content} />
              </div>
              {message.citations?.length ? (
                <ul className="citations">
                  {message.citations.map((citation) => (
                    <li key={citation}>
                      <a href={citation} target="_blank" rel="noreferrer">{citation}</a>
                    </li>
                  ))}
                </ul>
              ) : null}
              {message.actions?.length ? (
                <div className="action-cards">
                  {message.actions.map((action, actionIndex) => (
                    <ChatActionCard key={`${action.type}-${actionIndex}`} action={action} />
                  ))}
                </div>
              ) : null}
            </article>
          ))
        )}
      </div>

      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={loading ? "Thinking…" : "Ask about any stock, strategy or market move…"}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
    </section>
  );
}
