import { useCallback, useEffect, useState, type FormEvent } from "react";
import type { AlertSettings, AlertSettingsResponse, Condition, ConditionOp, DailySignal, WatchAlert } from "../types";
import { OPS, OperandPicker } from "./StrategyBuilder";

interface AlertsPanelProps {
  onLoadSignals: () => Promise<DailySignal[]>;
  onLoadSettings: () => Promise<AlertSettingsResponse>;
  onSaveSettings: (settings: AlertSettings) => Promise<AlertSettingsResponse>;
  onTest: () => Promise<unknown>;
  watch?: {
    load: () => Promise<WatchAlert[]>;
    add: (payload: { symbol: string; condition: Condition; note?: string }) => Promise<unknown>;
    remove: (id: string) => Promise<unknown>;
  };
  /** Admins with Alpaca configured: mirror a simulation's trades as paper orders. */
  onToggleMirror?: (simulationId: string, enabled: boolean) => Promise<unknown>;
}

const fmt = (value: number | null | undefined) =>
  value === null || value === undefined ? "–" : Math.abs(value) >= 1000 ? value.toFixed(0) : value.toFixed(2);

/** "Tell me when NVDA's RSI(14) is below 30": rules checked after every close, alerting once per day. */
function WatchAlerts({ watch }: { watch: NonNullable<AlertsPanelProps["watch"]> }) {
  const [alerts, setAlerts] = useState<WatchAlert[] | null>(null);
  const [symbol, setSymbol] = useState("NVDA");
  const [condition, setCondition] = useState<Condition>({
    left: { kind: "rsi", period: 14 },
    op: "<",
    right: { kind: "value", value: 30 },
  });
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setAlerts(await watch.load());
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Couldn't load alerts.");
    }
  }, [watch]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const add = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      await watch.add({ symbol: symbol.trim().toUpperCase(), condition, note: note.trim() || undefined });
      setNote("");
      await refresh();
    } catch (addError) {
      setError(addError instanceof Error ? addError.message : "Couldn't add the alert.");
    }
  };

  return (
    <div className="panel watch-panel">
      <header>
        <h2>Price &amp; indicator alerts</h2>
        <span className="hint">Checked after every close; you're notified the first day a condition is true</span>
      </header>
      <form className="watch-form" onSubmit={add}>
        <input aria-label="Alert symbol" value={symbol} maxLength={20} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
        <OperandPicker label="Alert left" value={condition.left} onChange={(left) => setCondition({ ...condition, left })} />
        <select
          aria-label="Alert comparison"
          value={condition.op}
          onChange={(e) => setCondition({ ...condition, op: e.target.value as ConditionOp })}
        >
          {OPS.map((option) => (
            <option key={option.op} value={option.op}>
              {option.label}
            </option>
          ))}
        </select>
        <OperandPicker label="Alert right" value={condition.right} onChange={(right) => setCondition({ ...condition, right })} />
        <input aria-label="Alert note" placeholder="note (optional)" value={note} maxLength={120} onChange={(e) => setNote(e.target.value)} />
        <button type="submit">Add alert</button>
      </form>
      {error ? <div className="error-banner">{error}</div> : null}
      {alerts?.length ? (
        <ul className="signal-list">
          {alerts.map((alert) => (
            <li key={alert.id} className={alert.status?.triggered ? "actionable" : undefined}>
              <span className="badge">{alert.symbol}</span>
              <strong className={alert.status?.triggered ? "positive" : ""}>
                {alert.status ? (alert.status.triggered ? "TRUE NOW" : "not yet") : "no data"}
              </strong>
              <span className="subtle">
                {alert.status?.description ?? ""}
                {alert.status ? ` · now ${fmt(alert.status.left)} vs ${fmt(alert.status.right)}` : ""}
                {alert.note ? ` · ${alert.note}` : ""}
              </span>
              <p>
                <button type="button" className="button-danger chip" onClick={() => void watch.remove(alert.id).then(refresh)}>
                  Delete
                </button>
              </p>
            </li>
          ))}
        </ul>
      ) : alerts ? (
        <p className="empty">No alerts yet.</p>
      ) : null}
    </div>
  );
}

const SIGNAL_TONE: Record<string, string> = { buy: "positive", cover: "positive", sell: "negative", short: "negative" };

/** What each active simulation's strategy says at the latest close, and where to send trade alerts. */
export function AlertsPanel({
  onLoadSignals,
  onLoadSettings,
  onSaveSettings,
  onTest,
  watch,
  onToggleMirror,
}: AlertsPanelProps) {
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
  const canMirror = Boolean(config?.available.broker && onToggleMirror);
  return (
    <>
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
                {canMirror ? (
                  <label className="checkbox mirror-toggle">
                    <input
                      type="checkbox"
                      checked={Boolean(s.brokerMirror)}
                      onChange={(event) => {
                        const enabled = event.target.checked;
                        void onToggleMirror?.(s.simulationId, enabled).then(() =>
                          setSignals((list) =>
                            (list ?? []).map((item) =>
                              item.simulationId === s.simulationId ? { ...item, brokerMirror: enabled } : item,
                            ),
                          ),
                        );
                      }}
                    />
                    <span>Mirror trades to Alpaca paper</span>
                  </label>
                ) : null}
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
    {watch ? <WatchAlerts watch={watch} /> : null}
    </>
  );
}
