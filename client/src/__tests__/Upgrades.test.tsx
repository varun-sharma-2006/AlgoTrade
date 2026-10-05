import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AlertsPanel } from "../components/AlertsPanel";
import { PortfolioLab } from "../components/PortfolioLab";
import { PortfolioPage } from "../components/PortfolioPage";
import { ModelReportCard, MonthlyHeatmap, RobustnessResults } from "../components/Research";
import { StrategyBuilder, describeRules } from "../components/StrategyBuilder";
import type { BasketResult, ModelReport, PortfolioResponse, RobustnessResult } from "../types";

afterEach(cleanup);

const band = { p5: -0.1, p25: 0.02, p50: 0.08, p75: 0.15, p95: 0.3 };

const robustness: RobustnessResult = {
  symbol: "AAPL",
  strategyId: "sma-crossover",
  parameters: { shortWindow: 20, longWindow: 60 },
  metrics: { totalReturn: 0.1, sharpe: 0.7, maxDrawdown: 0.12, buyHoldReturn: 0.2, trades: 6 },
  monteCarlo: {
    paths: 500,
    block: 20,
    finalReturn: band,
    maxDrawdown: { p5: 0.05, p25: 0.08, p50: 0.11, p75: 0.15, p95: 0.24 },
    sharpe: band,
    probLoss: 0.18,
    probBeatBuyHold: 0.35,
    fan: [
      { step: 0, timestamp: "2025-01-02", p5: 1, p25: 1, p50: 1, p75: 1, p95: 1 },
      { step: 10, timestamp: "2025-06-02", p5: 0.9, p25: 1.0, p50: 1.05, p75: 1.1, p95: 1.2 },
    ],
  },
  sensitivity: {
    xParam: "shortWindow",
    xValues: [10, 20],
    yParam: "longWindow",
    yValues: [60, 100],
    cells: [
      { x: 10, y: 60, valid: true, sharpe: 0.4, totalReturn: 0.05, maxDrawdown: 0.1, trades: 8 },
      { x: 20, y: 60, valid: true, sharpe: 0.7, totalReturn: 0.1, maxDrawdown: 0.12, trades: 6 },
      { x: 10, y: 100, valid: true, sharpe: -0.2, totalReturn: -0.03, maxDrawdown: 0.2, trades: 4 },
      { x: 20, y: 100, valid: false },
    ],
    current: { shortWindow: 20, longWindow: 60 },
    positiveShare: 0.67,
    medianSharpe: 0.4,
    bestSharpe: 0.7,
  },
  deflatedSharpe: {
    trials: 16,
    sharpe: 0.7,
    expectedMaxSharpe: 1.1,
    deflatedSharpe: 0.21,
    probabilisticSharpe: 0.84,
    skew: -0.3,
    kurtosis: 5,
  },
  pbo: { pbo: 0.43, combinations: 70, trials: 16, slices: 8, medianLogit: 0.2, winnerOutOfSampleSharpe: 0.3 },
  period: { start: "2025-01-02", end: "2026-01-02", days: 252 },
};

describe("Research upgrades", () => {
  it("shows Monte Carlo bands, deflated Sharpe, PBO and the sensitivity grid", () => {
    render(<RobustnessResults result={robustness} />);
    expect(screen.getByText("18%")).toBeInTheDocument(); // chance of losing money
    expect(screen.getByText("Deflated Sharpe ratio")).toBeInTheDocument();
    expect(screen.getByText("21%").className).toBe("negative");
    expect(screen.getByText("43%").className).toBe("positive");
    expect(screen.getByRole("img", { name: /Monte Carlo/ })).toBeInTheDocument();
    const current = document.querySelector(".heatmap-table td.current");
    expect(current?.textContent).toBe("0.70");
    expect(screen.getByText(/results depend noticeably|broad range/)).toBeInTheDocument();
  });

  it("draws a monthly heatmap with yearly totals", () => {
    render(
      <MonthlyHeatmap
        monthly={[
          { month: "2025-11", year: 2025, return: 0.1 },
          { month: "2025-12", year: 2025, return: -0.05 },
          { month: "2026-01", year: 2026, return: 0.02 },
        ]}
      />,
    );
    expect(screen.getByText("10.0")).toBeInTheDocument();
    expect(screen.getByText("+4.5%")).toBeInTheDocument(); // 1.1 * 0.95 - 1
  });

  it("shows boosted-tree importances, calibration and the out-of-sample threshold", () => {
    const report: ModelReport = {
      model: "Gradient-boosted trees",
      modelType: 1,
      weightKind: "importance",
      horizon: 5,
      features: ["RSI (14)", "1-day return"],
      predictions: 400,
      accuracy: 0.53,
      baselineAccuracy: 0.52,
      upDays: 0.52,
      auc: 0.54,
      precisionWhenLong: 0.55,
      daysLong: 200,
      threshold: 0.52,
      trainWindow: 504,
      retrainEvery: 63,
      refits: 8,
      latestProbability: 0.6,
      featureWeights: [
        { feature: "RSI (14)", weight: 0.7 },
        { feature: "1-day return", weight: 0.3 },
      ],
      calibration: [
        { predicted: 0.45, actual: 0.47, count: 200 },
        { predicted: 0.58, actual: 0.56, count: 200 },
      ],
      thresholdScan: [
        { threshold: 0.5, return: 0.04, trades: 20 },
        { threshold: 0.55, return: 0.09, trades: 10 },
      ],
      suggestedThreshold: 0.55,
    };
    render(<ModelReportCard report={report} />);
    expect(screen.getByText(/in 5 trading days/)).toBeInTheDocument();
    expect(screen.getByText("70.0%")).toBeInTheDocument();
    expect(screen.getByText(/The best there was 0.55; this backtest used 0.52/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Predicted versus actual/ })).toBeInTheDocument();
  });
});

describe("PortfolioLab", () => {
  it("sends the basket, weighting and strategy settings", async () => {
    const result: BasketResult = {
      symbols: ["AAPL", "MSFT"],
      strategyId: "trend-follow",
      parameters: { channel: 20 },
      weighting: "risk-parity",
      rebalance: "quarterly",
      topN: 0,
      metrics: {
        totalReturn: 0.2,
        annualizedReturn: 0.1,
        volatility: 0.15,
        sharpe: 0.66,
        sortino: 0.9,
        maxDrawdown: 0.1,
        calmar: 1,
        var95: 0.015,
        cvar95: 0.02,
        equalWeightReturn: 0.25,
        excessReturn: -0.05,
        turnover: 3,
        costPaid: 0.004,
        rebalances: 8,
      },
      equalWeight: { totalReturn: 0.25, annualizedReturn: 0.12, volatility: 0.2, sharpe: 0.6, sortino: 0.8, maxDrawdown: 0.15, calmar: 0.8 },
      benchmark: null,
      holdings: [
        { symbol: "AAPL", avgWeight: 0.4, finalWeight: 0.5, contribution: 0.12, timeInMarket: 0.7, buyHoldReturn: 0.3 },
        { symbol: "MSFT", avgWeight: 0.35, finalWeight: 0, contribution: 0.08, timeInMarket: 0.6, buyHoldReturn: 0.2 },
      ],
      curve: [
        { timestamp: "2025-01-02", equity: 1, equalWeight: 1, drawdown: 0 },
        { timestamp: "2026-01-02", equity: 1.2, equalWeight: 1.25, drawdown: -0.02 },
      ],
      monthly: [],
      period: { start: "2025-01-02", end: "2026-01-02", days: 252 },
      missing: [],
      crossSection: {
        rows: [
          { symbol: "AAPL", sharpe: 0.8, totalReturn: 0.2, buyHoldReturn: 0.3, excessReturn: -0.1, buyHoldSharpe: 0.9, maxDrawdown: 0.1, trades: 5 },
        ],
        summary: { count: 1, medianSharpe: 0.8, shareBeatBuyHold: 0, shareProfitable: 1, shareSharpeAboveBuyHold: 0 },
      },
      riskFreeRate: 0.04,
    };
    const onRun = vi.fn().mockResolvedValue(result);
    render(<PortfolioLab onRun={onRun} />);
    fireEvent.change(screen.getByLabelText(/Symbols/), { target: { value: "aapl, msft msft" } });
    fireEvent.change(screen.getByLabelText("Strategy"), { target: { value: "trend-follow" } });
    fireEvent.change(screen.getByLabelText("Weighting"), { target: { value: "risk-parity" } });
    fireEvent.change(screen.getByLabelText("Rebalance"), { target: { value: "quarterly" } });
    fireEvent.click(screen.getByRole("button", { name: "Run portfolio backtest" }));
    expect(onRun).toHaveBeenCalledWith({
      symbols: ["AAPL", "MSFT"],
      strategyId: "trend-follow",
      parameters: { channel: 20 },
      rules: null,
      weighting: "risk-parity",
      rebalance: "quarterly",
      topN: 0,
      allowShort: false,
    });
    expect(await screen.findByText("Each stock alone")).toBeInTheDocument();
    expect(screen.getByText("+12.0%")).toBeInTheDocument();
  });
});

describe("StrategyBuilder upgrades", () => {
  it("describes short rules with trailing and time exits, and exports Pine Script", async () => {
    expect(
      describeRules({
        entry: [{ left: { kind: "macd_hist" }, op: "crosses_below", right: { kind: "value", value: 0 } }],
        exit: [],
        side: "short",
        trailingStop: 0.1,
        maxHoldDays: 20,
      }),
    ).toBe(
      "Sell short when MACD histogram crosses below 0; cover when price rises 10% from its lowest close since entry or 20 trading days have passed.",
    );

    const onExportPine = vi.fn().mockResolvedValue("//@version=5\nstrategy(\"x\")");
    render(
      <StrategyBuilder
        saved={[]}
        onBacktest={vi.fn()}
        onSave={vi.fn()}
        onDelete={vi.fn()}
        onExportPine={onExportPine}
      />,
    );
    fireEvent.change(screen.getByLabelText("Entry needs"), { target: { value: "any" } });
    fireEvent.change(screen.getByLabelText(/Trailing stop/), { target: { value: "8" } });
    fireEvent.click(screen.getByRole("button", { name: "Export to TradingView" }));
    await waitFor(() => expect(screen.getByLabelText("Pine Script").textContent).toContain("//@version=5"));
    expect(onExportPine).toHaveBeenCalledWith(
      "RSI dip buyer",
      expect.objectContaining({ entryMode: "any", trailingStop: 0.08, side: "long" }),
    );
  });
});

describe("Portfolio ledger and alerts", () => {
  it("expands a position's fills and shows today's signals", async () => {
    const data: PortfolioResponse = {
      summary: {
        totalValue: 1100,
        totalCapital: 1000,
        pnl: 100,
        pnlPct: 0.1,
        dayChange: 5,
        dayChangePct: 0.005,
        positions: 1,
        inMarket: 1,
        baseCurrency: "USD",
        execution: "next_open",
      },
      history: [
        { date: "2026-01-01", value: 1000 },
        { date: "2026-01-02", value: 1100 },
      ],
      allocation: [{ id: "a", symbol: "TCS.NS", value: 1100, weight: 1 }],
      positions: [
        {
          id: "a",
          symbol: "TCS.NS",
          strategy: "Buy & hold",
          strategyId: "buy-hold",
          status: "active",
          startingCapital: 1000,
          currency: "INR",
          baseCurrency: "USD",
          value: 1100,
          pnl: 100,
          pnlPct: 0.1,
          inMarket: true,
          fxConverted: true,
          fxReturn: -0.02,
          pendingOrder: "sell",
          ledger: [
            { date: "2026-01-02", side: "buy", price: 4000, shares: 20.8, notional: 83200, fee: 98, slippage: 41.6, stop: false },
          ],
        },
      ],
    };
    const signals = [
      {
        simulationId: "a",
        symbol: "TCS.NS",
        strategy: "Buy & hold",
        signal: "sell",
        summary: "Exit rule triggered.",
        date: "2026-01-02",
        price: 4000,
        currency: "INR",
        actionable: true,
      },
    ];
    const alerts = (
      <AlertsPanel
        onLoadSignals={() => Promise.resolve(signals)}
        onLoadSettings={() =>
          Promise.resolve({ settings: { email: true, telegramChatId: null }, available: { email: false, telegram: false } })
        }
        onSaveSettings={vi.fn()}
        onTest={vi.fn()}
      />
    );
    render(<PortfolioPage onLoad={() => Promise.resolve(data)} onOpenSimulations={vi.fn()} alerts={alerts} />);
    fireEvent.click(await screen.findByRole("button", { name: "Show TCS.NS trades" }));
    expect(screen.getByText(/INR moved −2.00% against USD/)).toBeInTheDocument();
    expect(screen.getByText(/sell at open/)).toBeInTheDocument();
    expect(await screen.findByText("1 of 1 simulations want to trade at the next open")).toBeInTheDocument();
    expect(screen.getByText(/no email or Telegram sender configured/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Email me/)).toBeChecked();
  });
});
