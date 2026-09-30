import { useState, type FormEvent } from "react";
import type { Condition, ConditionOp, CustomStrategy, Operand, OperandKind, StrategyRules, TrainingResult } from "../types";
import { BacktestResults } from "./StrategyTrainer";

interface StrategyBuilderProps {
  saved: CustomStrategy[];
  onBacktest: (symbol: string, rules: StrategyRules) => Promise<TrainingResult>;
  onSave: (name: string, description: string, rules: StrategyRules) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

const MAX_CONDITIONS = 5;

const OPERAND_KINDS: Array<{ kind: OperandKind; label: string }> = [
  { kind: "price", label: "Price" },
  { kind: "sma", label: "SMA" },
  { kind: "ema", label: "EMA" },
  { kind: "rsi", label: "RSI" },
  { kind: "value", label: "Number" },
];

const OPS: Array<{ op: ConditionOp; label: string }> = [
  { op: ">", label: "is above" },
  { op: "<", label: "is below" },
  { op: "crosses_above", label: "crosses above" },
  { op: "crosses_below", label: "crosses below" },
];

const price = (): Operand => ({ kind: "price" });
const sma = (period: number): Operand => ({ kind: "sma", period });
const ema = (period: number): Operand => ({ kind: "ema", period });
const rsi = (period = 14): Operand => ({ kind: "rsi", period });
const num = (value: number): Operand => ({ kind: "value", value });

export const PRESETS: Array<{ name: string; description: string; rules: StrategyRules }> = [
  {
    name: "RSI dip buyer",
    description: "Buy oversold dips, sell into strength, with a stop-loss.",
    rules: { entry: [{ left: rsi(), op: "<", right: num(30) }], exit: [{ left: rsi(), op: ">", right: num(60) }], stopLoss: 0.08 },
  },
  {
    name: "Golden cross",
    description: "Classic 50/200-day moving-average crossover.",
    rules: {
      entry: [{ left: sma(50), op: "crosses_above", right: sma(200) }],
      exit: [{ left: sma(50), op: "crosses_below", right: sma(200) }],
    },
  },
  {
    name: "Trend + momentum",
    description: "Only buy uptrends that aren't overbought.",
    rules: {
      entry: [
        { left: price(), op: ">", right: sma(50) },
        { left: ema(12), op: "crosses_above", right: ema(26) },
        { left: rsi(), op: "<", right: num(70) },
      ],
      exit: [{ left: price(), op: "<", right: sma(50) }],
      stopLoss: 0.07,
      takeProfit: 0.25,
    },
  },
  {
    name: "Pullback in uptrend",
    description: "Buy when price dips under the 20-day average while above the 200-day.",
    rules: {
      entry: [
        { left: price(), op: ">", right: sma(200) },
        { left: price(), op: "crosses_below", right: sma(20) },
      ],
      exit: [{ left: price(), op: "crosses_above", right: sma(20) }],
      stopLoss: 0.06,
    },
  },
];

function operandLabel(operand: Operand): string {
  if (operand.kind === "price") return "Price";
  if (operand.kind === "value") return String(operand.value ?? 0);
  return `${operand.kind.toUpperCase()}(${operand.period ?? "?"})`;
}

const OP_WORDS: Record<ConditionOp, string> = {
  ">": "is above",
  "<": "is below",
  crosses_above: "crosses above",
  crosses_below: "crosses below",
};

/** Plain-English summary; mirrors backend/rules.py `describe`. */
export function describeRules(rules: StrategyRules): string {
  const cond = (c: Condition) => `${operandLabel(c.left)} ${OP_WORDS[c.op]} ${operandLabel(c.right)}`;
  const exits = rules.exit.map(cond);
  if (rules.stopLoss) exits.push(`price falls ${+(rules.stopLoss * 100).toFixed(2)}% below entry`);
  if (rules.takeProfit) exits.push(`price rises ${+(rules.takeProfit * 100).toFixed(2)}% above entry`);
  const buy = `Buy when ${rules.entry.map(cond).join(" and ")}`;
  return `${buy}; ${exits.length ? `sell when ${exits.join(" or ")}` : "hold once bought"}.`;
}

function OperandPicker({ value, onChange, label }: { value: Operand; onChange: (next: Operand) => void; label: string }) {
  return (
    <div className="operand">
      <select
        aria-label={`${label} type`}
        value={value.kind}
        onChange={(event) => {
          const kind = event.target.value as OperandKind;
          if (kind === "price") onChange({ kind });
          else if (kind === "value") onChange({ kind, value: value.value ?? (kind === "value" ? 50 : 0) });
          else onChange({ kind, period: value.period ?? (kind === "rsi" ? 14 : 20) });
        }}
      >
        {OPERAND_KINDS.map((option) => (
          <option key={option.kind} value={option.kind}>
            {option.label}
          </option>
        ))}
      </select>
      {value.kind === "sma" || value.kind === "ema" || value.kind === "rsi" ? (
        <input
          aria-label={`${label} period`}
          type="number"
          min={2}
          max={250}
          value={value.period ?? 14}
          onChange={(event) => onChange({ ...value, period: Number(event.target.value) })}
          title="Period in days"
        />
      ) : null}
      {value.kind === "value" ? (
        <input
          aria-label={`${label} value`}
          type="number"
          step="any"
          value={value.value ?? 0}
          onChange={(event) => onChange({ ...value, value: Number(event.target.value) })}
        />
      ) : null}
    </div>
  );
}

function ConditionList({
  title,
  hint,
  conditions,
  onChange,
  required,
}: {
  title: string;
  hint: string;
  conditions: Condition[];
  onChange: (next: Condition[]) => void;
  required?: boolean;
}) {
  const update = (index: number, next: Condition) => onChange(conditions.map((c, i) => (i === index ? next : c)));
  return (
    <div className="rule-group">
      <div className="rule-group-head">
        <strong>{title}</strong>
        <span className="hint">{hint}</span>
      </div>
      {conditions.map((condition, index) => (
        <div className="rule-row" key={index}>
          <span className="rule-joiner">{index === 0 ? "If" : title.startsWith("Buy") ? "and" : "or"}</span>
          <OperandPicker label={`${title} ${index + 1} left`} value={condition.left} onChange={(left) => update(index, { ...condition, left })} />
          <select
            aria-label={`${title} ${index + 1} comparison`}
            value={condition.op}
            onChange={(event) => update(index, { ...condition, op: event.target.value as ConditionOp })}
          >
            {OPS.map((option) => (
              <option key={option.op} value={option.op}>
                {option.label}
              </option>
            ))}
          </select>
          <OperandPicker label={`${title} ${index + 1} right`} value={condition.right} onChange={(right) => update(index, { ...condition, right })} />
          <button
            type="button"
            className="button-danger icon-button"
            aria-label={`Remove ${title.toLowerCase()} rule ${index + 1}`}
            disabled={required && conditions.length === 1}
            onClick={() => onChange(conditions.filter((_, i) => i !== index))}
          >
            ×
          </button>
        </div>
      ))}
      <button
        type="button"
        className="button-ghost add-rule"
        disabled={conditions.length >= MAX_CONDITIONS}
        onClick={() => onChange([...conditions, { left: price(), op: ">", right: sma(50) }])}
      >
        + Add rule
      </button>
    </div>
  );
}

export function StrategyBuilder({ saved, onBacktest, onSave, onDelete }: StrategyBuilderProps) {
  const [name, setName] = useState(PRESETS[0].name);
  const [description, setDescription] = useState(PRESETS[0].description);
  const [rules, setRules] = useState<StrategyRules>(PRESETS[0].rules);
  const [symbol, setSymbol] = useState("AAPL");
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [busy, setBusy] = useState<"test" | "save" | null>(null);
  const [message, setMessage] = useState<{ kind: "error" | "ok"; text: string } | null>(null);

  const load = (preset: { name: string; description?: string | null; rules: StrategyRules }) => {
    setName(preset.name);
    setDescription(preset.description ?? "");
    setRules({ ...preset.rules, exit: preset.rules.exit ?? [] });
    setResult(null);
    setMessage(null);
  };

  const clean = (): StrategyRules => ({
    entry: rules.entry,
    exit: rules.exit,
    stopLoss: rules.stopLoss || null,
    takeProfit: rules.takeProfit || null,
  });

  const runBacktest = async (event: FormEvent) => {
    event.preventDefault();
    setBusy("test");
    setMessage(null);
    try {
      setResult(await onBacktest(symbol, clean()));
    } catch (error) {
      setMessage({ kind: "error", text: error instanceof Error ? error.message : "Backtest failed." });
    } finally {
      setBusy(null);
    }
  };

  const save = async () => {
    if (!name.trim()) {
      setMessage({ kind: "error", text: "Give your strategy a name first." });
      return;
    }
    setBusy("save");
    setMessage(null);
    try {
      await onSave(name.trim(), description.trim(), clean());
      setMessage({ kind: "ok", text: `Saved “${name.trim()}”. You can now use it when creating a simulation.` });
    } catch (error) {
      setMessage({ kind: "error", text: error instanceof Error ? error.message : "Couldn't save the strategy." });
    } finally {
      setBusy(null);
    }
  };

  const percentInput = (key: "stopLoss" | "takeProfit", label: string, max: number) => (
    <label>
      <span>{label}</span>
      <div className="suffix-input">
        <input
          type="number"
          min={0}
          max={max}
          step={0.5}
          placeholder="off"
          value={rules[key] ? +(rules[key]! * 100).toFixed(2) : ""}
          onChange={(event) => {
            const raw = event.target.value;
            setRules((previous) => ({ ...previous, [key]: raw === "" || Number(raw) <= 0 ? null : Number(raw) / 100 }));
          }}
        />
        <em>%</em>
      </div>
    </label>
  );

  return (
    <section className="builder">
      <header className="header">
        <div>
          <span className="eyebrow">Design your own</span>
          <h1>Strategy builder</h1>
          <p>Combine price, moving averages and RSI into buy and sell rules, backtest them, and save the ones that work.</p>
        </div>
      </header>

      <div className="preset-row">
        <span className="hint">Start from a preset:</span>
        {PRESETS.map((preset) => (
          <button key={preset.name} type="button" className="button-ghost chip" onClick={() => load(preset)}>
            {preset.name}
          </button>
        ))}
      </div>

      <div className="builder-grid">
        <form className="panel" onSubmit={runBacktest}>
          <header>
            <h2>Rules</h2>
            <span className="hint">Decisions use each day's closing data and take effect the next day</span>
          </header>
          <div className="flex-row">
            <label style={{ flex: "2 1 220px" }}>
              <span>Name</span>
              <input value={name} maxLength={60} onChange={(event) => setName(event.target.value)} />
            </label>
            <label style={{ flex: "1 1 120px" }}>
              <span>Test on symbol</span>
              <input value={symbol} maxLength={12} onChange={(event) => setSymbol(event.target.value.toUpperCase())} />
            </label>
          </div>
          <label>
            <span>Description (optional)</span>
            <input value={description} maxLength={300} onChange={(event) => setDescription(event.target.value)} />
          </label>

          <ConditionList
            title="Buy rules"
            hint="All must be true"
            required
            conditions={rules.entry}
            onChange={(entry) => setRules((previous) => ({ ...previous, entry }))}
          />
          <ConditionList
            title="Sell rules"
            hint="Any one is enough"
            conditions={rules.exit}
            onChange={(exit) => setRules((previous) => ({ ...previous, exit }))}
          />

          <div className="flex-row">
            {percentInput("stopLoss", "Stop-loss", 99)}
            {percentInput("takeProfit", "Take-profit", 1000)}
          </div>

          <div className="rule-summary">
            <span className="eyebrow">In plain English</span>
            <p>{describeRules(rules)}</p>
          </div>

          {message ? <div className={message.kind === "error" ? "error-banner" : "success-banner"}>{message.text}</div> : null}

          <div className="actions">
            <button type="submit" disabled={busy !== null}>
              {busy === "test" ? "Backtesting…" : "Run backtest"}
            </button>
            <button type="button" className="button-ghost" disabled={busy !== null} onClick={() => void save()}>
              {busy === "save" ? "Saving…" : "Save strategy"}
            </button>
          </div>
        </form>

        <div className="builder-side">
          <div className="panel">
            <header>
              <h2>Backtest</h2>
              <span className="hint">2 years of daily data, fees and slippage included</span>
            </header>
            {result ? (
              <BacktestResults training={result} description={describeRules(result.rules ?? rules)} />
            ) : (
              <p className="empty">Run a backtest to see how your rules would have traded.</p>
            )}
          </div>

          <div className="panel">
            <header>
              <h2>Your strategies</h2>
              <span className="hint">Saved strategies appear in the simulation form</span>
            </header>
            {saved.length ? (
              <ul className="saved-strategies">
                {saved.map((strategy) => (
                  <li key={strategy.id}>
                    <div>
                      <strong>{strategy.name}</strong>
                      <span className="subtle">{describeRules(strategy.rules)}</span>
                    </div>
                    <div className="actions">
                      <button type="button" className="button-ghost" onClick={() => load(strategy)}>
                        Edit
                      </button>
                      <button type="button" className="button-danger" onClick={() => void onDelete(strategy.id)}>
                        Delete
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty">No saved strategies yet.</p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
