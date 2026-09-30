import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PortfolioPage } from "../components/PortfolioPage";
import { PRESETS, StrategyBuilder, describeRules } from "../components/StrategyBuilder";
import { SimulationForm } from "../components/SimulationForm";
import type { PortfolioResponse, TrainingResult } from "../types";

afterEach(cleanup);

describe("describeRules", () => {
  it("reads like a sentence", () => {
    expect(describeRules(PRESETS[0].rules)).toBe(
      "Buy when RSI(14) is below 30; sell when RSI(14) is above 60 or price falls 8% below entry.",
    );
    expect(describeRules({ entry: PRESETS[0].rules.entry, exit: [] })).toBe("Buy when RSI(14) is below 30; hold once bought.");
  });
});

describe("StrategyBuilder", () => {
  const result: TrainingResult = {
    symbol: "AAPL",
    strategyId: "custom",
    parameters: {},
    rules: PRESETS[1].rules,
    metrics: {
      totalReturn: 0.1, annualizedReturn: 0.05, buyHoldReturn: 0.2, excessReturn: -0.1, winRate: 0.5, trades: 2,
      closedTrades: 1, avgTradeReturn: 0.03, sharpe: 0.7, maxDrawdown: 0.1, exposure: 0.5, feeBps: 10,
    },
    trades: [],
    openTrade: null,
    sample: [],
    period: { start: "2024-01-01", end: "2026-01-01", days: 500 },
    trainedAt: "2026-01-01",
  };

  it("builds rules, backtests and saves", async () => {
    const onBacktest = vi.fn().mockResolvedValue(result);
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<StrategyBuilder saved={[]} onBacktest={onBacktest} onSave={onSave} onDelete={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Golden cross" }));
    expect(screen.getByText(/Buy when SMA\(50\) crosses above SMA\(200\)/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/Stop-loss/), { target: { value: "5" } });
    fireEvent.click(screen.getAllByRole("button", { name: "+ Add rule" })[0]); // add a second buy rule

    fireEvent.click(screen.getByRole("button", { name: "Run backtest" }));
    await waitFor(() => expect(onBacktest).toHaveBeenCalled());
    const [symbol, rules] = onBacktest.mock.calls[0];
    expect(symbol).toBe("AAPL");
    expect(rules.entry).toHaveLength(2);
    expect(rules.stopLoss).toBeCloseTo(0.05);
    expect(await screen.findByText("Lagged buy & hold by")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Save strategy" }));
    await waitFor(() => expect(onSave).toHaveBeenCalledWith("Golden cross", expect.any(String), expect.objectContaining({ stopLoss: 0.05 })));
    expect(await screen.findByText(/Saved “Golden cross”/)).toBeInTheDocument();
  });
});

describe("SimulationForm", () => {
  it("sends the chosen saved strategy with its rules and a start date", async () => {
    const onSubmit = vi.fn();
    const saved = [{ id: "s1", name: "My RSI", rules: PRESETS[0].rules, createdAt: "2026-01-01" }];
    render(<SimulationForm onSubmit={onSubmit} loading={false} customStrategies={saved} />);
    fireEvent.change(screen.getByLabelText("Strategy"), { target: { value: "custom:s1" } });
    fireEvent.click(screen.getByRole("button", { name: "Launch simulation" }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    const payload = onSubmit.mock.calls[0][0];
    expect(payload).toMatchObject({ symbol: "AAPL", strategy: "My RSI", strategyId: "custom", rules: PRESETS[0].rules });
    expect(payload.startDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});

describe("PortfolioPage", () => {
  const data: PortfolioResponse = {
    summary: { totalValue: 11250, totalCapital: 10000, pnl: 1250, pnlPct: 0.125, dayChange: -50, dayChangePct: -0.0044, positions: 2, inMarket: 1 },
    history: [
      { date: "2026-01-01", value: 10000 },
      { date: "2026-01-02", value: 11250 },
    ],
    allocation: [
      { id: "a", symbol: "MSFT", value: 7000, weight: 0.622 },
      { id: "b", symbol: "NVDA", value: 4250, weight: 0.378 },
    ],
    positions: [
      { id: "a", symbol: "MSFT", strategy: "Buy & hold", strategyId: "buy-hold", status: "active", startingCapital: 6000, currency: "USD", value: 7000, pnl: 1000, pnlPct: 0.1667, inMarket: true, startDate: "2026-01-01", buyHoldValue: 7000 },
      { id: "b", symbol: "NVDA", strategy: "RSI dip", strategyId: "custom", status: "active", startingCapital: 4000, currency: "USD", value: 4250, pnl: 250, pnlPct: 0.0625, inMarket: false, startDate: "2026-01-01", buyHoldValue: 4500 },
    ],
  };

  it("shows totals, allocation and holdings", async () => {
    render(<PortfolioPage onLoad={() => Promise.resolve(data)} onOpenSimulations={vi.fn()} />);
    expect(await screen.findByText("Holdings")).toBeInTheDocument();
    expect(screen.getByText("Portfolio value").nextSibling?.textContent).toMatch(/11,250/);
    const pills = Array.from(document.querySelectorAll(".status-pill"), (el) => el.textContent);
    expect(pills).toEqual(["Invested", "In cash"]);
    expect(screen.getByText("62.2%")).toBeInTheDocument();
    expect(screen.getByText(/−\$250\.00/)).toBeInTheDocument(); // RSI dip lagged buy & hold by $250
  });

  it("guides new users to create a simulation", async () => {
    const onOpen = vi.fn();
    const empty = { ...data, positions: [], allocation: [], history: [], summary: { ...data.summary, positions: 0 } };
    render(<PortfolioPage onLoad={() => Promise.resolve(empty)} onOpenSimulations={onOpen} />);
    fireEvent.click(await screen.findByRole("button", { name: "Create a simulation" }));
    expect(onOpen).toHaveBeenCalled();
  });
});
