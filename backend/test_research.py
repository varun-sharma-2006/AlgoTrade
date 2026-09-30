"""Risk metrics, slippage, the evaluation window, the ML strategy and walk-forward testing."""

import math
import random
from datetime import date, timedelta

import pytest

from backend import ml, risk, strategies, walkforward


def dated(n, start=date(2020, 1, 1)):
    return [f"{start + timedelta(days=i)}T14:30:00+00:00" for i in range(n)]


def random_walk(n, seed=7, drift=0.0004, vol=0.015):
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift + rng.gauss(0, vol)))
    return closes


def test_risk_summary_and_sortino_only_counts_losing_days():
    returns = [0.01, -0.01, 0.02, -0.005, 0.0]
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    stats = risk.summary(returns, equity)
    assert stats["totalReturn"] == pytest.approx(equity[-1] - 1)
    assert stats["volatility"] == pytest.approx(risk.stdev(returns) * math.sqrt(252))
    downside = math.sqrt((0.01**2 + 0.005**2) / 5)
    assert stats["sortino"] == pytest.approx(risk.mean(returns) / downside * math.sqrt(252))
    assert stats["maxDrawdown"] == pytest.approx(0.01)
    assert risk.drawdowns([1, 2, 1, 3]) == [0, 0, -0.5, 0]


def test_benchmark_beta_of_a_doubled_index_is_two():
    index = random_walk(120, seed=1)
    ts = dated(120)
    equity = [1.0]
    for i in range(1, 120):
        equity.append(equity[-1] * (1 + 2 * (index[i] / index[i - 1] - 1)))
    stats = risk.benchmark(ts, equity, ts, index)
    assert stats["beta"] == pytest.approx(2.0)
    assert stats["correlation"] == pytest.approx(1.0)
    assert stats["totalReturn"] == pytest.approx(index[-1] / index[0] - 1)
    assert risk.benchmark(ts[:10], equity[:10], ts, index) is None  # too few shared days


def test_slippage_is_charged_on_both_sides_of_every_trade():
    closes = [100.0, 100.0, 110.0, 121.0]
    no_cost = strategies.backtest(closes, dated(4), "buy-hold", {}, fee_bps=0)
    costly = strategies.backtest(closes, dated(4), "buy-hold", {}, fee_bps=0, slippage_bps=50)
    assert no_cost["metrics"]["totalReturn"] == pytest.approx(0.21)
    # One entry and no exit: 0.5% slippage once.
    assert costly["metrics"]["totalReturn"] == pytest.approx(1.21 * 0.995 - 1)
    assert costly["metrics"]["slippageBps"] == 50


def test_backtest_window_starts_in_cash_after_the_warm_up():
    closes = [100 + i for i in range(300)]
    ts = dated(300)
    start = strategies.evaluation_start(ts, 100)
    assert ts[start][:10] >= str(date(2020, 1, 1) + timedelta(days=199))
    report = strategies.backtest(closes, ts, "sma-crossover", {"shortWindow": 5, "longWindow": 20}, 0, start=start)
    # The SMAs are already crossed at the window start, so it buys on day one of the window.
    assert report["sample"][0]["timestamp"] == ts[start]
    assert report["metrics"]["totalReturn"] == pytest.approx(closes[-1] / closes[start] - 1)
    assert report["metrics"]["buyHoldReturn"] == pytest.approx(closes[-1] / closes[start] - 1)
    assert report["period"]["days"] == 300 - start
    assert [p["drawdown"] for p in report["sample"]] == [0] * len(report["sample"])
    assert strategies.evaluation_start(["2026-0001", "2026-0002"], 100) == 0  # undated test data
    assert strategies.period_days("2y") == 730 and strategies.period_days("6mo") == 180


def test_ml_never_uses_the_future():
    closes = random_walk(700)
    base = ml.probabilities(closes, 504)["probs"]
    cut = 500
    changed = closes[: cut + 1] + [c * 1.5 for c in closes[cut + 1 :]]
    after = ml.probabilities(changed, 504)["probs"]
    assert base[: cut + 1] == after[: cut + 1]
    assert base[cut + 1 :] != after[cut + 1 :]
    first = next(i for i, p in enumerate(base) if p is not None)
    assert first >= ml.MIN_TRAIN  # needs a year of labelled history first


def test_ml_learns_a_real_pattern_and_reports_honestly():
    # Tomorrow's move reverses today's: a pattern the 1-day return feature can learn.
    rng = random.Random(3)
    closes, r = [100.0], 0.01
    for _ in range(800):
        r = -0.9 * r + rng.gauss(0, 0.004)
        closes.append(closes[-1] * (1 + r))
    report = ml.report(closes, {"threshold": 0.5, "trainWindow": 504}, start=400)
    assert report["accuracy"] > 0.8 > report["baselineAccuracy"]
    assert report["auc"] > 0.85
    assert report["featureWeights"][0]["feature"] in {
        "1-day return",
        "Bollinger z-score (20)",
        "Price vs 10-day average",
    }
    target, lines = strategies.signals("ml-logistic", closes, {"threshold": 0.5, "trainWindow": 504})
    assert set(target) == {0, 1} and len(lines["probUp"]) == len(closes)
    assert strategies.validate("ml-logistic", {"threshold": 0.9, "trainWindow": 504})
    assert strategies.validate("ml-logistic", {"threshold": 0.55, "trainWindow": 504}) is None


def test_ml_solver_matches_a_known_system():
    assert ml._solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0]) == pytest.approx([0.8, 1.4])


def test_walk_forward_trains_only_on_the_past():
    closes = random_walk(900, seed=11)
    ts = dated(900)
    result = walkforward.run(closes, ts, "trend-follow", 15)
    folds = result["folds"]
    assert len(folds) >= 5
    for previous, fold in zip(folds, folds[1:], strict=False):
        assert fold["testStart"] == previous["testEnd"]
    assert all(f["trainStart"] < f["testStart"] < f["testEnd"] for f in folds)
    assert result["curve"][0]["timestamp"] == folds[0]["testStart"]
    assert len({p["timestamp"] for p in result["curve"]}) == len(result["curve"])

    # Rewriting the last bars must not change which parameters were picked for earlier folds.
    changed = closes[:-60] + [c * 0.5 for c in closes[-60:]]
    again = walkforward.run(changed, ts, "trend-follow", 15)
    assert [f["params"] for f in again["folds"][:-2]] == [f["params"] for f in folds[:-2]]
    metrics = result["metrics"]
    assert metrics["mostChosenCount"] <= len(folds) and 0 <= metrics["foldsBeatBuyHold"] <= len(folds)

    with pytest.raises(ValueError):
        walkforward.run(closes[:200], ts[:200], "trend-follow", 15)
