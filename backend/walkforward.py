"""Walk-forward testing: tune a strategy's parameters on one period, then trade them on the next.

For each fold, every parameter set in the strategy's grid is backtested on the training window (the
previous `train_days` bars) and the one with the best Sharpe ratio is traded on the following `test_days`
bars, which it has never seen. Folds roll forward until the data runs out, and the test windows are
stitched into a single out-of-sample equity curve. The position carries over between folds, as it would
for a trader who re-tunes every quarter.

Comparing the tuned (in-sample) returns with the out-of-sample returns shows how much of a backtest's
edge was curve fitting.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from backend import risk, strategies

GRIDS: dict[str, list[dict[str, float]]] = {
    "sma-crossover": [
        {"shortWindow": s, "longWindow": long_} for s in (10, 20, 30, 50) for long_ in (50, 100, 150, 200) if s < long_
    ],
    "mean-reversion": [{"lookback": lb, "deviation": d} for lb in (10, 20, 30) for d in (1.5, 2.0, 2.5)],
    "trend-follow": [{"channel": c} for c in (10, 20, 40, 55)],
    # The model's probabilities don't depend on the threshold, so this grid costs one model run.
    "ml-logistic": [{"threshold": t, "trainWindow": 504} for t in (0.5, 0.52, 0.55, 0.58)],
}
TRAIN_DAYS = 252
TEST_DAYS = 63


def run(
    closes: list[float],
    timestamps: list[str],
    strategy_id: str,
    cost_bps: float,
    *,
    train_days: int = TRAIN_DAYS,
    test_days: int = TEST_DAYS,
) -> dict[str, Any]:
    grid = GRIDS[strategy_id]
    targets = [strategies.signals(strategy_id, closes, params)[0] for params in grid]
    warm = max(strategies.warmup(strategy_id, params) for params in grid)
    last = len(closes) - 1
    first_test = warm + train_days
    if first_test + 10 > last:
        raise ValueError("Not enough price history for a walk-forward test")

    folds = []
    curve: list[dict[str, Any]] = []
    value, held = 1.0, 0
    test_start = first_test
    while test_start + 10 <= last:
        test_end = min(test_start + test_days, last)
        train_start = test_start - train_days
        scored = []
        for k, target in enumerate(targets):
            trained = strategies.simulate(closes, target, train_start, test_start, cost_bps)
            stats = risk.summary(trained["dailyReturns"], [1.0, *trained["equity"]])
            scored.append((stats["sharpe"], -k, k, stats))  # ties go to the first (simplest) grid entry
        _, _, best, train_stats = max(scored)
        tested = strategies.simulate(closes, targets[best], test_start, test_end, cost_bps, held=held, value=value)
        start_value = value
        value, held = tested["equity"][-1], tested["held"]
        folds.append(
            {
                "trainStart": timestamps[train_start],
                "testStart": timestamps[test_start],
                "testEnd": timestamps[test_end],
                "params": grid[best],
                "trainSharpe": train_stats["sharpe"],
                "trainReturn": train_stats["totalReturn"],
                "testReturn": value / start_value - 1,
                "buyHoldReturn": closes[test_end] / closes[test_start] - 1,
                "trades": len(tested["trades"]),
            }
        )
        # Each fold's first bar is the previous fold's last one; keep it once.
        skip = 1 if curve else 0
        for i, equity in zip(range(test_start + skip, test_end + 1), tested["equity"][skip:], strict=True):
            curve.append({"timestamp": timestamps[i], "equity": equity, "buyHold": closes[i] / closes[first_test]})
        test_start = test_end

    equity = [point["equity"] for point in curve]
    daily = [equity[k] / equity[k - 1] - 1 for k in range(1, len(equity))]
    oos = risk.summary(daily, [1.0, *equity])
    buy_hold_equity = [point["buyHold"] for point in curve]
    buy_hold = risk.summary(
        [buy_hold_equity[k] / buy_hold_equity[k - 1] - 1 for k in range(1, len(buy_hold_equity))], buy_hold_equity
    )
    chosen = Counter(tuple(sorted(fold["params"].items())) for fold in folds)
    in_sample = risk.mean([risk.annualise(fold["trainReturn"], train_days) for fold in folds])
    for point, dd in zip(curve, risk.drawdowns(equity), strict=True):
        point["drawdown"] = dd
    return {
        "strategyId": strategy_id,
        "trainDays": train_days,
        "testDays": test_days,
        "gridSize": len(grid),
        "folds": folds,
        "curve": curve,
        "metrics": {
            "outOfSampleReturn": oos["totalReturn"],
            "outOfSampleAnnualized": oos["annualizedReturn"],
            "outOfSampleSharpe": oos["sharpe"],
            "outOfSampleMaxDrawdown": oos["maxDrawdown"],
            "inSampleAnnualized": in_sample,
            "buyHoldReturn": buy_hold["totalReturn"],
            "buyHoldAnnualized": buy_hold["annualizedReturn"],
            "buyHoldSharpe": buy_hold["sharpe"],
            "foldsBeatBuyHold": sum(1 for fold in folds if fold["testReturn"] > fold["buyHoldReturn"]),
            "mostChosenParams": dict(chosen.most_common(1)[0][0]),
            "mostChosenCount": chosen.most_common(1)[0][1],
            "costBps": cost_bps,
        },
        "period": {"start": timestamps[first_test], "end": timestamps[last], "days": last - first_test + 1},
    }
