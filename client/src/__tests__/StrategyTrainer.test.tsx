import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Metrics, StrategyTrainer } from "../components/StrategyTrainer";
import type { StrategyMetrics, TrainingResult } from "../types";

afterEach(cleanup);

const metrics: StrategyMetrics = {
  totalReturn: 0.1,
  annualizedReturn: 0.05,
  buyHoldReturn: 0.3,
  excessReturn: -0.2,
  winRate: 0.5,
  trades: 3,
  closedTrades: 2,
  avgTradeReturn: 0.04,
  sharpe: 0.8,
  maxDrawdown: 0.12,
  exposure: 0.4,
  feeBps: 10,
};

describe("StrategyTrainer", () => {
  it("sends only the parameters of the selected strategy", () => {
    const onTrain = vi.fn();
    render(<StrategyTrainer onTrain={onTrain} onPredict={vi.fn()} training={null} prediction={null} loading={false} />);

    fireEvent.change(screen.getByLabelText("Strategy"), { target: { value: "trend-follow" } });
    expect(screen.queryByLabelText("Short window")).toBeNull();
    fireEvent.change(screen.getByLabelText("Channel length"), { target: { value: "55" } });
    fireEvent.click(screen.getByRole("button", { name: "Run backtest" }));

    expect(onTrain).toHaveBeenCalledWith({
      symbol: "AAPL",
      strategyId: "trend-follow",
      channel: 55,
      interval: "1d",
      marketImpact: false,
      execution: "next_open",
      sizing: "full",
      sizeFraction: 0.5,
      targetVol: 0.15,
      maxLeverage: 1,
      allowShort: false,
      borrowBps: 50,
      riskFreeRate: 0.04,
      slippageBps: 5,
    });
  });

  it("disables the signal button until a strategy is trained", () => {
    render(<StrategyTrainer onTrain={vi.fn()} onPredict={vi.fn()} training={null} prediction={null} loading={false} />);
    expect(screen.getByRole("button", { name: "Today's signal" })).toBeDisabled();
  });

  it("renders backtest results with trades", () => {
    const training: TrainingResult = {
      symbol: "MSFT",
      strategyId: "sma-crossover",
      parameters: { shortWindow: 20, longWindow: 60 },
      metrics,
      trades: [{ entryDate: "2026-01-02", entryPrice: 100, exitDate: "2026-02-02", exitPrice: 110, return: 0.098 }],
      openTrade: null,
      sample: [
        { timestamp: "2026-01-02", close: 100, equity: 1, position: 0 },
        { timestamp: "2026-01-03", close: 101, equity: 1.01, position: 1 },
      ],
      period: { start: "2024-01-01", end: "2026-01-01", days: 500 },
      trainedAt: "2026-01-01T00:00:00Z",
    };
    render(<StrategyTrainer onTrain={vi.fn()} onPredict={vi.fn()} training={training} prediction={null} loading={false} />);
    expect(screen.getByText("MSFT")).toBeInTheDocument();
    expect(screen.getByText("9.80%")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Today's signal" })).toBeEnabled();
  });
});

describe("Metrics", () => {
  it("says when the strategy lagged buy and hold", () => {
    render(<Metrics metrics={metrics} />);
    expect(screen.getByText("Lagged buy & hold by")).toBeInTheDocument();
    expect(screen.getByText("20.00%")).toBeInTheDocument();
  });
});
