import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatActionCard } from "../components/ChatbotPanel";
import { ModelReportCard, RiskTable, WalkForwardResults } from "../components/Research";
import { BacktestResults, StrategyTrainer } from "../components/StrategyTrainer";
import type { ModelReport, RiskStats, StrategyMetrics, TrainingResult, WalkForwardResult } from "../types";

afterEach(cleanup);

const risk = (overrides: Partial<RiskStats> = {}): RiskStats => ({
  totalReturn: 0.3,
  annualizedReturn: 0.14,
  volatility: 0.25,
  sharpe: 0.7,
  sortino: 1.0,
  maxDrawdown: 0.3,
  calmar: 0.47,
  ...overrides,
});

const metrics: StrategyMetrics = {
  ...risk({ totalReturn: 0.1, annualizedReturn: 0.05, volatility: 0.12, sharpe: 0.9, maxDrawdown: 0.1 }),
  buyHoldReturn: 0.3,
  excessReturn: -0.2,
  winRate: 0.5,
  trades: 3,
  closedTrades: 2,
  avgTradeReturn: 0.04,
  exposure: 0.4,
  feeBps: 10,
  slippageBps: 5,
};

const model: ModelReport = {
  model: "Logistic regression",
  predictions: 500,
  accuracy: 0.51,
  baselineAccuracy: 0.53,
  upDays: 0.53,
  auc: 0.5,
  precisionWhenLong: 0.52,
  daysLong: 300,
  threshold: 0.52,
  trainWindow: 504,
  retrainEvery: 21,
  refits: 46,
  latestProbability: 0.55,
  featureWeights: [
    { feature: "1-day return", weight: -0.2 },
    { feature: "RSI (14)", weight: 0.1 },
  ],
};

const walkForward: WalkForwardResult = {
  symbol: "AAPL",
  strategyId: "sma-crossover",
  trainDays: 252,
  testDays: 63,
  gridSize: 13,
  folds: [
    {
      trainStart: "2023-01-03",
      testStart: "2024-01-03",
      testEnd: "2024-04-03",
      params: { shortWindow: 20, longWindow: 50 },
      trainSharpe: 1.4,
      trainReturn: 0.3,
      testReturn: 0.02,
      buyHoldReturn: 0.05,
      trades: 2,
    },
  ],
  curve: [
    { timestamp: "2024-01-03", equity: 1, buyHold: 1, drawdown: 0 },
    { timestamp: "2024-04-03", equity: 1.02, buyHold: 1.05, drawdown: 0 },
  ],
  metrics: {
    outOfSampleReturn: 0.02,
    outOfSampleAnnualized: 0.08,
    outOfSampleSharpe: 0.5,
    outOfSampleMaxDrawdown: 0.1,
    inSampleAnnualized: 0.3,
    buyHoldReturn: 0.05,
    buyHoldAnnualized: 0.2,
    buyHoldSharpe: 0.9,
    foldsBeatBuyHold: 0,
    mostChosenParams: { shortWindow: 20, longWindow: 50 },
    mostChosenCount: 1,
    costBps: 15,
  },
  period: { start: "2024-01-03", end: "2024-04-03", days: 63 },
  feeBps: 10,
  slippageBps: 5,
};

describe("Risk and research panels", () => {
  it("compares the strategy with buy & hold and the S&P 500", () => {
    render(
      <RiskTable
        metrics={metrics}
        buyHold={risk()}
        benchmark={{ ...risk({ sharpe: 1.1 }), symbol: "^GSPC", name: "S&P 500", beta: 0.8, alpha: 0.01, correlation: 0.6 }}
      />,
    );
    expect(screen.getByText("Sortino ratio")).toBeInTheDocument();
    expect(screen.getByText("S&P 500")).toBeInTheDocument();
    expect(screen.getByText(/beta 0\.80/)).toBeInTheDocument();
    // Lower volatility than buy & hold is marked as better.
    expect(screen.getByText("12.0%").className).toBe("positive");
  });

  it("says plainly when the model did not beat the baseline", () => {
    render(<ModelReportCard report={model} />);
    expect(screen.getByText("51.0%")).toBeInTheDocument();
    expect(screen.getByText(/did not beat simply guessing/)).toBeInTheDocument();
    expect(screen.getByText("1-day return")).toBeInTheDocument();
  });

  it("shows how much tuning flattered a strategy in a walk-forward test", () => {
    render(<WalkForwardResults result={walkForward} />);
    expect(screen.getByText(/Tuning flattered the strategy by 22\.0 points/)).toBeInTheDocument();
    expect(screen.getByText("shortWindow 20, longWindow 50")).toBeInTheDocument();
  });

  it("renders charts, the risk table and the model card for an ML backtest", () => {
    const training: TrainingResult = {
      symbol: "NVDA",
      strategyId: "ml-logistic",
      parameters: { threshold: 0.52, trainWindow: 504 },
      metrics,
      buyHold: risk(),
      benchmark: null,
      model,
      trades: [],
      openTrade: null,
      sample: [
        { timestamp: "2026-01-02", close: 100, equity: 1, buyHold: 1, drawdown: 0, buyHoldDrawdown: 0, position: 0 },
        { timestamp: "2026-01-05", close: 90, equity: 0.95, buyHold: 0.9, drawdown: -0.05, buyHoldDrawdown: -0.1, position: 1 },
      ],
      period: { start: "2024-01-01", end: "2026-01-05", days: 500 },
      trainedAt: "2026-01-05T00:00:00Z",
    };
    render(<BacktestResults training={training} />);
    expect(screen.getByRole("img", { name: "Growth of $1" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Drawdown/ })).toBeInTheDocument();
    expect(screen.getByText("Risk & return")).toBeInTheDocument();
    expect(screen.getByText("15 bps")).toBeInTheDocument();
    expect(screen.getByText("Machine-learning model")).toBeInTheDocument();
  });
});

describe("Strategy lab extras", () => {
  it("sends the ML settings and runs a walk-forward test", async () => {
    const onTrain = vi.fn();
    const onWalkForward = vi.fn().mockResolvedValue(walkForward);
    render(
      <StrategyTrainer
        onTrain={onTrain}
        onPredict={vi.fn()}
        onWalkForward={onWalkForward}
        training={null}
        prediction={null}
        loading={false}
      />,
    );
    fireEvent.change(screen.getByLabelText("Strategy"), { target: { value: "ml-logistic" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText("Predict"), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText("Buy when P(up) ≥"), { target: { value: "0.55" } });
    fireEvent.change(screen.getByLabelText("Slippage (bps per trade)"), { target: { value: "8" } });
    fireEvent.click(screen.getByLabelText("Allow short selling"));
    fireEvent.click(screen.getByRole("button", { name: "Run backtest" }));
    expect(onTrain).toHaveBeenCalledWith({
      symbol: "AAPL",
      strategyId: "ml-logistic",
      modelType: 1,
      horizon: 5,
      threshold: 0.55,
      trainWindow: 504,
      execution: "next_open",
      sizing: "full",
      sizeFraction: 0.5,
      targetVol: 0.15,
      maxLeverage: 1,
      allowShort: true,
      borrowBps: 50,
      riskFreeRate: 0.04,
      slippageBps: 8,
    });

    fireEvent.click(screen.getByRole("button", { name: "Walk-forward test" }));
    expect(onWalkForward).toHaveBeenCalledWith({
      symbol: "AAPL",
      strategyId: "ml-logistic",
      slippageBps: 8,
      execution: "next_open",
      allowShort: true,
    });
    await waitFor(() => expect(screen.getByText("AAPL · 1 out-of-sample quarters")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Strategy"), { target: { value: "buy-hold" } });
    expect(screen.getByRole("button", { name: "Walk-forward test" })).toBeDisabled();
  });
});

describe("Copilot action cards", () => {
  it("shows backtest numbers the copilot ran", () => {
    render(
      <ChatActionCard
        action={{
          type: "backtest",
          label: "Backtest · AAPL · SMA 20/60",
          data: { strategyReturn: 0.42, buyHoldReturn: 0.43, sp500Return: 0.33, sharpe: 1.02, maxDrawdown: 0.145, trades: 6 },
        }}
      />,
    );
    expect(screen.getByText("Backtest · AAPL · SMA 20/60")).toBeInTheDocument();
    expect(screen.getByText("+42.0%").className).toBe("negative");
    expect(screen.getByText("1.02")).toBeInTheDocument();
  });
});
