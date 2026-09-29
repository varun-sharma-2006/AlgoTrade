import { FormEvent, useState } from "react";
import type { SimulationInput } from "../types";

interface SimulationFormProps {
  onSubmit: (payload: SimulationInput) => Promise<void> | void;
  loading: boolean;
}

const DEFAULT_FORM: SimulationInput = {
  symbol: "AAPL",
  strategy: "Momentum",
  startingCapital: 10000,
  notes: "",
};

export function SimulationForm({ onSubmit, loading }: SimulationFormProps) {
  const [form, setForm] = useState<SimulationInput>(DEFAULT_FORM);
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setValidationError(null);

    if (!form.symbol.trim()) {
      setValidationError("Symbol is required.");
      return;
    }

    if (!form.strategy.trim()) {
      setValidationError("Strategy is required.");
      return;
    }

    if (!Number.isFinite(form.startingCapital) || form.startingCapital <= 0) {
      setValidationError("Starting capital must be a positive number.");
      return;
    }

    await onSubmit({
      ...form,
      symbol: form.symbol.trim().toUpperCase(),
      strategy: form.strategy.trim(),
      notes: form.notes?.trim() || undefined,
    });

    setForm((prev) => ({ ...prev, notes: "" }));
  };

  return (
    <div className="card">
      <h2>New simulation</h2>
      <p className="hint">Pick a ticker, a strategy and your starting capital.</p>

      {validationError && <div className="error-banner">{validationError}</div>}

      <form onSubmit={handleSubmit} className="form-grid">
        <div className="flex-row">
          <label style={{ flex: "1 1 140px" }}>
            <span>Symbol</span>
            <input
              value={form.symbol}
              onChange={(event) => setForm((prev) => ({ ...prev, symbol: event.target.value }))}
              placeholder="AAPL"
              maxLength={6}
            />
          </label>

          <label style={{ flex: "1 1 200px" }}>
            <span>Strategy</span>
            <input
              value={form.strategy}
              onChange={(event) => setForm((prev) => ({ ...prev, strategy: event.target.value }))}
              placeholder="Momentum"
            />
          </label>

          <label style={{ flex: "1 1 200px" }}>
            <span>Starting capital (USD)</span>
            <input
              type="number"
              min={100}
              step={100}
              value={form.startingCapital}
              onChange={(event) =>
                setForm((prev) => ({ ...prev, startingCapital: Number(event.target.value) }))
              }
            />
          </label>
        </div>

        <label>
          <span>Notes (optional)</span>
          <textarea
            rows={3}
            value={form.notes ?? ""}
            onChange={(event) => setForm((prev) => ({ ...prev, notes: event.target.value }))}
            placeholder="Describe entry rules, time horizon, or goals."
          />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Creating simulation..." : "Launch simulation"}
        </button>
      </form>
    </div>
  );
}
