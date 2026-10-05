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
from backend import rules as rule_engine

GRIDS: dict[str, list[dict[str, float]]] = {
    "sma-crossover": [
        {"shortWindow": s, "longWindow": long_} for s in (10, 20, 30, 50) for long_ in (50, 100, 150, 200) if s < long_
    ],
    "mean-reversion": [{"lookback": lb, "deviation": d} for lb in (10, 20, 30) for d in (1.5, 2.0, 2.5)],
    "trend-follow": [{"channel": c} for c in (10, 20, 40, 55)],
    "regime-switch": [{"erWindow": w, "erThreshold": t} for w in (10, 20, 40) for t in (0.2, 0.3, 0.45)],
    # The model's probabilities don't depend on the threshold, so this grid costs one model run.
    "ml-logistic": [{"threshold": t, "trainWindow": 504} for t in (0.5, 0.52, 0.55, 0.58)],
}
TRAIN_DAYS = 252
TEST_DAYS = 63
# Custom rules: every indicator period and every stop / target scaled by these factors.
CUSTOM_SCALES = [{"periodScale": p, "stopScale": s} for p in (0.75, 1.0, 1.25) for s in (0.75, 1.0, 1.25)]


def scale_rules(rules: dict[str, Any], periodScale: float = 1.0, stopScale: float = 1.0) -> dict[str, Any]:  # noqa: N803
    """A copy of Builder rules with indicator periods and stop / target distances scaled."""

    def operand(spec: dict[str, Any]) -> dict[str, Any]:
        if spec.get("period"):
            return spec | {"period": max(2, min(250, round(spec["period"] * periodScale)))}
        return dict(spec)

    def conditions(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        return [c | {"left": operand(c["left"]), "right": operand(c["right"])} for c in items or []]

    scaled = rules | {"entry": conditions(rules["entry"]), "exit": conditions(rules.get("exit"))}
    for key in ("stopLoss", "trailingStop"):
        if rules.get(key):
            scaled[key] = min(rules[key] * stopScale, 0.95)
    if rules.get("takeProfit"):
        scaled["takeProfit"] = min(rules["takeProfit"] * stopScale, 10)
    if rules.get("maxHoldDays"):
        scaled["maxHoldDays"] = max(1, round(rules["maxHoldDays"] * periodScale))
    return scaled


def run(
    closes: list[float],
    timestamps: list[str],
    strategy_id: str,
    cost_bps: float,
    *,
    train_days: int = TRAIN_DAYS,
    test_days: int = TEST_DAYS,
    opens: list[float] | None = None,
    execution: str = "close",
    sell_cost_bps: float | None = None,
    allow_short: bool = False,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trading = {"opens": opens, "execution": execution, "sell_cost_bps": sell_cost_bps}
    if strategy_id == "custom":
        # Builder rules have no named parameters, so tune scaled variants of them instead.
        if not rules:
            raise ValueError("A custom walk-forward test needs rules")
        variants = [(label, scale_rules(rules, **label)) for label in CUSTOM_SCALES]
        grid = [label for label, _ in variants]
        targets = [rule_engine.plan(closes, variant)[0] for _, variant in variants]
        warm = max(rule_engine.warmup(variant) for _, variant in variants)
    else:
        grid = GRIDS[strategy_id]
        targets = [strategies.signals(strategy_id, closes, params, allow_short=allow_short)[0] for params in grid]
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
            trained = strategies.simulate(closes, target, train_start, test_start, cost_bps, **trading)
            stats = risk.summary(trained["dailyReturns"], [1.0, *trained["equity"]])
            scored.append((stats["sharpe"], -k, k, stats))  # ties go to the first (simplest) grid entry
        _, _, best, train_stats = max(scored)
        tested = strategies.simulate(
            closes, targets[best], test_start, test_end, cost_bps, held=held, value=value, **trading
        )
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
            "execution": execution,
        },
        "period": {"start": timestamps[first_test], "end": timestamps[last], "days": last - first_test + 1},
    }
