import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PortfolioLab } from "../components/PortfolioLab";
import { PushToggle, keyBytes } from "../components/PushToggle";
import { StrategyTrainer } from "../components/StrategyTrainer";

afterEach(cleanup);

describe("push notifications", () => {
  it("decodes the server key and explains when the browser can't do push", async () => {
    expect(Array.from(keyBytes("AQID_w"))).toEqual([1, 2, 3, 255]);
    render(<PushToggle api={{ getKey: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn() }} />);
    expect(await screen.findByText(/doesn't support push notifications/)).toBeInTheDocument();
  });
});

describe("survivorship tools", () => {
  it("draws a point-in-time sample and flags stocks that joined the index later", async () => {
    const onSample = vi.fn().mockResolvedValue({ asOf: "2024-10-06", symbols: ["AAA", "OLD"], leftSince: ["OLD"], membersThen: 504 });
    const onSurvivorship = vi.fn().mockResolvedValue({
      asOf: "2024-10-06",
      membersThen: 504,
      inIndexThen: ["AAA"],
      joinedLater: [{ symbol: "NEW", date: "2025-03-03" }],
      notInIndex: [],
      removedSince: [{ symbol: "OLD", date: "2025-03-03" }],
      survivorShare: 0.93,
    });
    render(<PortfolioLab onRun={vi.fn()} onSample={onSample} onSurvivorship={onSurvivorship} />);
    fireEvent.click(screen.getByRole("button", { name: "Point-in-time S&P 500 sample" }));
    expect(await screen.findByText(/OLD left the index since/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Symbols/)).toHaveValue("AAA, OLD");
    fireEvent.click(screen.getByRole("button", { name: "Check survivorship bias" }));
    expect(await screen.findByText("1 of your stocks weren't in the index yet")).toBeInTheDocument();
    expect(onSurvivorship).toHaveBeenCalledWith(["AAA", "OLD"]);
  });
});

describe("earnings and news options", () => {
  it("sends skip-earnings and, for the ML strategy, the news feature", () => {
    const onTrain = vi.fn();
    render(<StrategyTrainer onTrain={onTrain} onPredict={vi.fn()} training={null} prediction={null} loading={false} />);
    expect(screen.queryByLabelText("News tone feature")).toBeNull();
    fireEvent.click(screen.getByLabelText("Skip earnings (US)"));
    fireEvent.change(screen.getByLabelText("Strategy"), { target: { value: "ml-logistic" } });
    fireEvent.click(screen.getByLabelText("News tone feature"));
    fireEvent.click(screen.getByRole("button", { name: "Run backtest" }));
    expect(onTrain).toHaveBeenCalledWith(
      expect.objectContaining({ strategyId: "ml-logistic", avoidEarnings: true, newsFeatures: true }),
    );
  });
});
