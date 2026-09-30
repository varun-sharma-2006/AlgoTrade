import { FormEvent, useState } from "react";
import type { CustomStrategy, SimulationInput } from "../types";
import { STRATEGY_FORMS, type BuiltInStrategyId } from "./StrategyTrainer";

interface SimulationFormProps {
  onSubmit: (payload: SimulationInput) => Promise<void> | void;
  loading: boolean;
  customStrategies?: CustomStrategy[];
}

const localDate = (date: Date) =>
  `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;

export function SimulationForm({ onSubmit, loading, customStrategies = [] }: SimulationFormProps) {
  const today = localDate(new Date());
  const yearAgo = localDate(new Date(Date.now() - 364 * 24 * 60 * 60 * 1000));
  const [symbol, setSymbol] = useState("AAPL");
  const [choice, setChoice] = useState<string>("buy-hold"); // a built-in id, or "custom:<saved id>"
  const [startingCapital, setStartingCapital] = useState(10000);
  const [startDate, setStartDate] = useState(localDate(new Date(Date.now() - 90 * 24 * 60 * 60 * 1000)));
  const [notes, setNotes] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setValidationError(null);
    if (!symbol.trim()) {
      setValidationError("Symbol is required.");
      return;
    }
    if (!Number.isFinite(startingCapital) || startingCapital <= 0) {
      setValidationError("Starting capital must be a positive number.");
      return;
    }
    if (startDate && (startDate > today || startDate < yearAgo)) {
      setValidationError("Pick a start date within the last year.");
      return;
    }

    let strategy: Pick<SimulationInput, "strategy" | "strategyId" | "parameters" | "rules">;
    if (choice.startsWith("custom:")) {
      const saved = customStrategies.find((item) => `custom:${item.id}` === choice);
      if (!saved) {
        setValidationError("That saved strategy no longer exists.");
        return;
      }
      strategy = { strategy: saved.name, strategyId: "custom", parameters: {}, rules: saved.rules };
    } else {
      const form = STRATEGY_FORMS[choice as BuiltInStrategyId];
      const parameters = Object.fromEntries(form.fields.map((field) => [field.key, field.initial]));
      strategy = { strategy: form.label.replace(" (baseline)", ""), strategyId: choice, parameters };
    }

    await onSubmit({
      symbol: symbol.trim().toUpperCase(),
      startingCapital,
      notes: notes.trim() || undefined,
      startDate: startDate || undefined,
      ...strategy,
    });
    setNotes("");
  };

  return (
    <div className="card">
      <h2>New simulation</h2>
      <p className="hint">Invest paper money in a stock with a strategy. Backdate it to see how it would have done.</p>

      {validationError && <div className="error-banner">{validationError}</div>}

      <form onSubmit={handleSubmit} className="form-grid">
        <div className="flex-row">
          <label style={{ flex: "1 1 120px" }}>
            <span>Symbol</span>
            <input value={symbol} onChange={(event) => setSymbol(event.target.value)} placeholder="AAPL" maxLength={12} />
          </label>
          <label style={{ flex: "1 1 160px" }}>
            <span>Starting capital (USD)</span>
            <input
              type="number"
              min={100}
              step={100}
              value={startingCapital}
              onChange={(event) => setStartingCapital(Number(event.target.value))}
            />
          </label>
        </div>

        <div className="flex-row">
          <label style={{ flex: "2 1 200px" }}>
            <span>Strategy</span>
            <select value={choice} onChange={(event) => setChoice(event.target.value)}>
              <optgroup label="Built-in">
                {Object.entries(STRATEGY_FORMS).map(([id, form]) => (
                  <option key={id} value={id}>
                    {form.label}
                  </option>
                ))}
              </optgroup>
              {customStrategies.length ? (
                <optgroup label="Your strategies">
                  {customStrategies.map((item) => (
                    <option key={item.id} value={`custom:${item.id}`}>
                      {item.name}
                    </option>
                  ))}
                </optgroup>
              ) : null}
            </select>
          </label>
          <label style={{ flex: "1 1 150px" }}>
            <span>Start date</span>
            <input type="date" value={startDate} min={yearAgo} max={today} onChange={(event) => setStartDate(event.target.value)} />
          </label>
        </div>

        <label>
          <span>Notes (optional)</span>
          <textarea
            rows={2}
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Why this trade? What would make you exit?"
          />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Creating simulation..." : "Launch simulation"}
        </button>
      </form>
    </div>
  );
}
