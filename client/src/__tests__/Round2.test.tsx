import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AlertsPanel } from "../components/AlertsPanel";
import { OptimizePanel, ReviewPanel, TaxCard } from "../components/Insights";
import { Leaderboard } from "../components/Leaderboard";
import { LearnPage, OnboardingTour } from "../components/LearnPage";
import { OptionsLab, blackScholes } from "../components/OptionsLab";
import { ReportPage } from "../components/ReportPage";
import { SipPlanner } from "../components/SipPlanner";
import { StrategyBuilder } from "../components/StrategyBuilder";
import { StrategyTrainer } from "../components/StrategyTrainer";
import type { OptimizeResult, SharedReport, SipResult, TrainingResult } from "../types";

afterEach(cleanup);

const training: TrainingResult = {
  symbol: "AAPL",
  strategyId: "sma-crossover",
  parameters: { shortWindow: 20, longWindow: 60 },
  metrics: {
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
  },
  trades: [{ entryDate: "2026-01-02", entryPrice: 100, exitDate: "2026-02-02", exitPrice: 110, return: 0.098 }],
  openTrade: null,
  sample: [
    { timestamp: "2026-01-02", close: 100, equity: 1, position: 0 },
    { timestamp: "2026-01-03", close: 101, equity: 1.01, position: 1 },
  ],
  period: { start: "2024-01-01", end: "2026-01-01", days: 500 },
  trainedAt: "2026-01-01T00:00:00Z",
};

const optimized: OptimizeResult = {
  symbol: "NVDA",
  strategyId: "sma-crossover",
  current: { parameters: { shortWindow: 20, longWindow: 60 }, sharpe: 0.5, totalReturn: 0.2 },
  best: { parameters: { shortWindow: 40, longWindow: 60 }, sharpe: 0.9, totalReturn: 0.5, maxDrawdown: 0.2, trades: 8 },
  trials: 38,
  deflatedSharpe: { trials: 38, sharpe: 0.9, expectedMaxSharpe: 1.2, deflatedSharpe: 0.3, probabilisticSharpe: 0.9, skew: 0, kurtosis: 3 },
  pbo: { pbo: 0.6, combinations: 70, trials: 38, slices: 8, medianLogit: -0.1, winnerOutOfSampleSharpe: 0.1 },
  walkForward: null,
  warnings: ["After trying 38 settings, there is only 30% confidence the best one's Sharpe ratio isn't luck."],
  trustworthy: false,
};

describe("insight cards", () => {
  it("shows taxes, review findings and optimiser guardrails", () => {
    const onUse = vi.fn();
    render(
      <>
        <TaxCard
          tax={{
            region: "IN",
            crypto: false,
            currency: "INR",
            capital: 1_000_000,
            rules: "India: STCG 20%",
            totalTax: 20800,
            preTaxReturn: 0.2,
            afterTaxReturn: 0.1792,
            carryForwardLoss: 0,
            years: [{ year: "FY2025-26", shortTermGain: 100000, longTermGain: 0, taxableLongTerm: 0, tax: 20800 }],
          }}
        />
        <ReviewPanel
          review={{
            symbol: "AAPL",
            strategyId: "trend-follow",
            verdict: { label: "Not trustworthy", text: "Several red flags", good: 0, bad: 2, warnings: 0 },
            findings: [
              { level: "bad", title: "Lost to buy & hold", detail: "-20%" },
              { level: "bad", title: "Too few trades to judge", detail: "3 trades" },
            ],
            summary: null,
          }}
        />
        <OptimizePanel result={optimized} onUse={onUse} />
      </>,
    );
    expect(screen.getByText("FY2025-26")).toBeInTheDocument();
    expect(screen.getByText(/\+17\.9%/)).toBeInTheDocument();
    expect(screen.getByText("Not trustworthy")).toBeInTheDocument();
    expect(screen.getByText(/only 30% confidence/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use them anyway (likely overfit)" }));
    expect(onUse).toHaveBeenCalledWith({ shortWindow: 40, longWindow: 60 });
  });
});

describe("Strategy lab tools", () => {
  it("runs the optimiser, review and share actions with the form's settings", async () => {
    const onOptimize = vi.fn().mockResolvedValue(optimized);
    const onShare = vi.fn().mockResolvedValue("/r/abc123");
    render(
      <StrategyTrainer
        onTrain={vi.fn()}
        onPredict={vi.fn()}
        onOptimize={onOptimize}
        onShare={onShare}
        training={training}
        prediction={null}
        loading={false}
      />,
    );
    fireEvent.change(screen.getByLabelText("Bars"), { target: { value: "1h" } });
    fireEvent.click(screen.getByLabelText("Market impact"));
    fireEvent.click(screen.getByRole("button", { name: "Optimise" }));
    await screen.findByText(/Best of 38 settings/);
    expect(onOptimize).toHaveBeenCalledWith(expect.objectContaining({ interval: "1h", marketImpact: true, strategyId: "sma-crossover" }));
    fireEvent.click(screen.getByRole("button", { name: "Use them anyway (likely overfit)" }));
    expect(screen.getByLabelText("Short window")).toHaveValue(40);

    fireEvent.click(screen.getByRole("button", { name: "Share" }));
    expect(await screen.findByLabelText("Report link")).toHaveValue(`${window.location.origin}/r/abc123`);
  });
});

describe("Strategy builder from text", () => {
  it("fills the rules from a plain-English description", async () => {
    const onFromText = vi.fn().mockResolvedValue({
      source: "parser",
      notes: "",
      rules: {
        entry: [{ left: { kind: "rsi", period: 2 }, op: "<", right: { kind: "value", value: 10 } }],
        exit: [],
        stopLoss: 0.05,
      },
    });
    render(<StrategyBuilder saved={[]} onBacktest={vi.fn()} onSave={vi.fn()} onDelete={vi.fn()} onFromText={onFromText} />);
    fireEvent.change(screen.getByLabelText("Describe a strategy in plain English"), {
      target: { value: "buy when rsi(2) is below 10, 5% stop" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Build rules" }));
    expect(await screen.findByText(/Built by the offline parser/)).toBeInTheDocument();
    expect(screen.getByText(/Buy when RSI\(2\) is below 10; sell when price falls 5% below entry\./)).toBeInTheDocument();
  });
});

describe("SIP, options and leaderboard pages", () => {
  it("sends SIP settings and shows XIRR", async () => {
    const result: SipResult = {
      symbol: "^NSEI",
      currency: "INR",
      region: "IN",
      timingStrategy: "Simple moving average crossover",
      months: 36,
      years: 3,
      invested: 400000,
      lastInstalment: 12100,
      sip: { value: 500000, gain: 100000, xirr: 0.14, taxIfRedeemed: 3000 },
      lumpSum: { value: 520000, gain: 120000, cagr: 0.09, taxIfRedeemed: 0 },
      timed: { value: 480000, gain: 80000, xirr: 0.12, taxIfRedeemed: 1000, cashWaiting: 0 },
      series: [
        { date: "2023-01-02", invested: 10000, sip: 10000, lumpSum: 400000, timed: 10000 },
        { date: "2026-01-02", invested: 400000, sip: 500000, lumpSum: 520000, timed: 480000 },
      ],
      period: { start: "2023-01-02", end: "2026-01-02" },
    };
    const onRun = vi.fn().mockResolvedValue(result);
    render(<SipPlanner onRun={onRun} />);
    fireEvent.change(screen.getByLabelText("Monthly amount"), { target: { value: "5000" } });
    fireEvent.click(screen.getByRole("button", { name: "Simulate SIP" }));
    expect(onRun).toHaveBeenCalledWith({
      symbol: "^NSEI",
      monthly: 5000,
      stepUp: 0.1,
      years: 3,
      timingStrategy: "sma-crossover",
      timingRules: null,
    });
    expect(await screen.findByText(/XIRR \+14\.0%/)).toBeInTheDocument();
  });

  it("prices options with Black-Scholes and shows the payoff", () => {
    expect(blackScholes("call", 100, 100, 1, 0.05, 0.2)).toBeCloseTo(10.4506, 3);
    expect(blackScholes("put", 100, 100, 1, 0.05, 0.2)).toBeCloseTo(5.5735, 3);
    render(<OptionsLab onRun={vi.fn()} />);
    expect(screen.getByText("Premium (Black-Scholes)")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Profit per share/ })).toBeInTheDocument();
  });

  it("ranks strategies and expands a row", async () => {
    const onRun = vi.fn().mockResolvedValue({
      symbols: ["AAPL"],
      missing: [],
      rows: [
        {
          strategyId: "trend-follow",
          name: "Breakout",
          symbols: 1,
          medianSharpe: 0.8,
          worstSharpe: 0.8,
          medianReturn: 0.2,
          medianMaxDrawdown: 0.1,
          shareBeatBuyHold: 1,
          shareBetterSharpe: 1,
          score: 0.8,
          results: [{ symbol: "AAPL", sharpe: 0.8, totalReturn: 0.2, excessReturn: 0.05, maxDrawdown: 0.1 }],
        },
      ],
    });
    render(<Leaderboard onRun={onRun} />);
    fireEvent.click(screen.getByRole("button", { name: "Rank strategies" }));
    fireEvent.click(await screen.findByText("Breakout"));
    expect(screen.getByText(/AAPL: Sharpe 0\.80/)).toBeInTheDocument();
    expect(onRun.mock.calls[0][0]).toEqual(["AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ"]);
  });
});

describe("learning, tour and shared reports", () => {
  it("runs a lesson and walks through the tour", async () => {
    const runBacktest = vi.fn().mockResolvedValue(training);
    render(<LearnPage runBacktest={runBacktest} runOptimize={vi.fn()} onStartTour={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Compare 4 strategies on the S&P 500" }));
    expect(await screen.findByText(/None of the active strategies beat simply holding/)).toBeInTheDocument();
    expect(runBacktest).toHaveBeenCalledTimes(4);

    const onNavigate = vi.fn();
    const onClose = vi.fn();
    render(<OnboardingTour onNavigate={onNavigate} onClose={onClose} />);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(onNavigate).toHaveBeenCalledWith("simulations");
    fireEvent.click(screen.getByRole("button", { name: "Skip tour" }));
    expect(onClose).toHaveBeenCalled();
  });

  it("renders a shared report and saves its rules", async () => {
    const rules = { entry: [{ left: { kind: "price" as const }, op: ">" as const, right: { kind: "sma" as const, period: 50 } }], exit: [] };
    const report: SharedReport = {
      id: "abc123",
      title: "Trend rules on AAPL",
      author: "Ravi",
      createdAt: "2026-10-01T00:00:00Z",
      content: {
        backtest: { ...training, strategyId: "custom", rules },
        robustness: null,
        review: { verdict: { label: "Inconclusive", text: "Not enough evidence", good: 1, bad: 0, warnings: 1 }, findings: [] },
        settings: { symbol: "AAPL", strategyId: "custom" },
      },
    };
    const onSaveRules = vi.fn().mockResolvedValue(undefined);
    render(<ReportPage reportId="abc123" onLoad={() => Promise.resolve(report)} onSaveRules={onSaveRules} />);
    expect(await screen.findByText("Trend rules on AAPL")).toBeInTheDocument();
    expect(screen.getByText("Inconclusive")).toBeInTheDocument();
    expect(screen.getByText("Buy when Price is above SMA(50); hold once bought.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Save these rules to my account" }));
    await waitFor(() => expect(onSaveRules).toHaveBeenCalledWith("Trend rules on AAPL", rules));
  });
});

describe("watch alerts", () => {
  it("adds an indicator alert and lists its status", async () => {
    const add = vi.fn().mockResolvedValue({});
    const load = vi
      .fn()
      .mockResolvedValueOnce([])
      .mockResolvedValue([
        {
          id: "w1",
          symbol: "NVDA",
          condition: { left: { kind: "rsi", period: 14 }, op: "<", right: { kind: "value", value: 30 } },
          note: null,
          status: { triggered: true, left: 25.3, right: 30, date: "2026-10-05", price: 100, currency: "USD", description: "RSI(14) is below 30" },
        },
      ]);
    render(
      <AlertsPanel
        onLoadSignals={() => Promise.resolve([])}
        onLoadSettings={() => Promise.resolve({ settings: { email: false, telegramChatId: null }, available: { email: true, telegram: true } })}
        onSaveSettings={vi.fn()}
        onTest={vi.fn()}
        watch={{ load, add, remove: vi.fn() }}
      />,
    );
    expect(await screen.findByText("No alerts yet.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Add alert" }));
    await waitFor(() =>
      expect(add).toHaveBeenCalledWith({
        symbol: "NVDA",
        condition: { left: { kind: "rsi", period: 14 }, op: "<", right: { kind: "value", value: 30 } },
        note: undefined,
      }),
    );
    expect(await screen.findByText("TRUE NOW")).toBeInTheDocument();
    expect(screen.getByText(/now 25\.30 vs 30\.00/)).toBeInTheDocument();
  });
});
