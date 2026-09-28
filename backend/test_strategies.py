import math

import pytest

from backend import strategies


def dates(n):
    return [f"2026-01-{i:03d}" for i in range(n)]


def test_moving_average_and_drawdown():
    assert strategies.moving_average([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]
    assert strategies.compute_drawdown([100, 120, 90, 130]) == pytest.approx(0.25)


def test_flat_strategy_on_rising_prices_has_no_trades():
    closes = [100 + i for i in range(80)]
    report = strategies.backtest(closes, dates(80), "mean-reversion", {"lookback": 20, "deviation": 2}, fee_bps=0)
    assert report["metrics"]["trades"] == 0
    assert report["metrics"]["totalReturn"] == 0
    assert report["metrics"]["buyHoldReturn"] == pytest.approx(79 / 100)


def test_sma_crossover_trade_returns_include_fees():
    # Down, then a sustained rally: one entry on the crossover, held to the end.
    closes = [100 - i for i in range(40)] + [61 + 2 * i for i in range(60)]
    report = strategies.backtest(closes, dates(100), "sma-crossover", {"shortWindow": 5, "longWindow": 20}, fee_bps=10)
    metrics = report["metrics"]
    assert metrics["trades"] >= 1
    open_trade = report["openTrade"]
    assert open_trade is not None
    gross = open_trade["exitPrice"] / open_trade["entryPrice"] - 1
    assert open_trade["return"] == pytest.approx((1 + gross) * (1 - 0.001) - 1, rel=1e-9)
    assert 0 < metrics["exposure"] < 1
    assert metrics["totalReturn"] > 0


def test_breakout_enters_and_exits():
    closes = [100.0] * 30 + [110 + i for i in range(10)] + [100 - i for i in range(20)]
    report = strategies.backtest(closes, dates(60), "trend-follow", {"channel": 20}, fee_bps=0)
    assert report["metrics"]["closedTrades"] == 1
    assert report["openTrade"] is None
    assert len(report["sample"]) == 60


def test_validation_and_signal():
    assert strategies.validate("sma-crossover", {"shortWindow": 50, "longWindow": 20})
    assert strategies.validate("nope", {})
    assert strategies.validate("trend-follow", {"channel": 20}) is None
    falling = [100 - i * 0.2 + math.sin(i) * 0.05 for i in range(100)]
    assert strategies.current_signal("trend-follow", falling, {"channel": 20})["signal"] == "wait"
    flat_then_breakout = [100.0] * 100 + [120.0]
    outlook = strategies.current_signal("trend-follow", flat_then_breakout, {"channel": 20})
    assert outlook["signal"] == "buy"
    assert 0 <= outlook["confidence"] <= 1
