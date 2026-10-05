import { useCallback, useEffect, useState, type FormEvent } from "react";
import type { AlertSettings, AlertSettingsResponse, DailySignal } from "../types";

interface AlertsPanelProps {
  onLoadSignals: () => Promise<DailySignal[]>;
  onLoadSettings: () => Promise<AlertSettingsResponse>;
  onSaveSettings: (settings: AlertSettings) => Promise<AlertSettingsResponse>;
  onTest: () => Promise<unknown>;
}

const SIGNAL_TONE: Record<string, string> = { buy: "positive", cover: "positive", sell: "negative", short: "negative" };

/** What each active simulation's strategy says at the latest close, and where to send trade alerts. */
export function AlertsPanel({ onLoadSignals, onLoadSettings, onSaveSettings, onTest }: AlertsPanelProps) {
  const [signals, setSignals] = useState<DailySignal[] | null>(null);
  const [config, setConfig] = useState<AlertSettingsResponse | null>(null);
  const [email, setEmail] = useState(false);
  const [chatId, setChatId] = useState("");
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const [loadedSignals, loadedConfig] = await Promise.all([onLoadSignals(), onLoadSettings()]);
      setSignals(loadedSignals);
      setConfig(loadedConfig);
      setEmail(loadedConfig.settings.email);
      setChatId(loadedConfig.settings.telegramChatId ?? "");
    } catch (error) {
      setMessage({ kind: "error", text: error instanceof Error ? error.message : "Couldn't load alerts." });
    }
  }, [onLoadSignals, onLoadSettings]);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      setConfig(await onSaveSettings({ email, telegramChatId: chatId.trim() || null }));
      setMessage({ kind: "ok", text: "Alert settings saved." });
    } catch (error) {
      setMessage({ kind: "error", text: error instanceof Error ? error.message : "Couldn't save alert settings." });
    } finally {
      setBusy(false);
    }
  };

  const test = async () => {
    setBusy(true);
    setMessage(null);
    try {
      await onTest();
      setMessage({ kind: "ok", text: "Test alert sent." });
    } catch (error) {
      setMessage({ kind: "error", text: error instanceof Error ? error.message : "The test alert failed." });
    } finally {
      setBusy(false);
    }
  };

  const actionable = signals?.filter((s) => s.actionable) ?? [];
  return (
    <div className="portfolio-grid alerts-grid">
      <div className="panel">
        <header>
          <h2>Today's signals</h2>
          <span className="hint">
            {signals === null
              ? "Checking your strategies…"
              : actionable.length
                ? `${actionable.length} of ${signals.length} simulations want to trade at the next open`
                : "No simulation wants to trade at the next open"}
          </span>
        </header>
        {signals?.length ? (
          <ul className="signal-list">
            {signals.map((s) => (
              <li key={s.simulationId} className={s.actionable ? "actionable" : undefined}>
                <span className="badge">{s.symbol}</span>
                <strong className={SIGNAL_TONE[s.signal] ?? ""}>{s.signal.toUpperCase()}</strong>
                <span className="subtle">
                  {s.strategy} · close {s.price.toFixed(2)} {s.currency} on {s.date}
                </span>
                <p>{s.summary}</p>
              </li>
            ))}
          </ul>
        ) : signals ? (
          <p className="empty">No active simulations.</p>
        ) : null}
      </div>
      <form className="panel" onSubmit={save}>
        <header>
          <h2>Trade alerts</h2>
          <span className="hint">Sent once per signal after each market close</span>
        </header>
        <label className="checkbox">
          <input type="checkbox" checked={email} onChange={(event) => setEmail(event.target.checked)} />
          <span>Email me when a strategy signals a trade</span>
        </label>
        <label>
          <span>Telegram chat id (optional)</span>
          <input
            value={chatId}
            inputMode="numeric"
            placeholder="e.g. 123456789"
            maxLength={40}
            onChange={(event) => setChatId(event.target.value.replace(/[^\d-]/g, ""))}
          />
        </label>
        <p className="hint">
          For Telegram, message the app's bot once, then get your chat id from @userinfobot.
          {config && !config.available.email && !config.available.telegram
            ? " This server has no email or Telegram sender configured yet (SMTP_* or TELEGRAM_BOT_TOKEN), so alerts are saved but not sent."
            : config && !config.available.email
              ? " Email isn't configured on this server; Telegram is."
              : config && !config.available.telegram
                ? " Telegram isn't configured on this server; email is."
                : ""}
        </p>
        {message ? <div className={message.kind === "error" ? "error-banner" : "success-banner"}>{message.text}</div> : null}
        <div className="actions">
          <button type="submit" disabled={busy}>
            Save
          </button>
          <button type="button" className="button-ghost" disabled={busy} onClick={() => void test()}>
            Send a test alert
          </button>
        </div>
      </form>
    </div>
  );
}
