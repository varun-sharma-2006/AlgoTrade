"""Execution realism, sizing, shorting, costs, robustness checks, baskets, FX, alerts and Pine export."""

import math
import random
from datetime import date, timedelta

import pytest

from backend import basket, costs, ml, portfolio, risk, robustness, rules, strategies
from backend.config import settings


def dated(n, start=date(2021, 1, 4)):
    return [f"{start + timedelta(days=i)}T14:30:00+00:00" for i in range(n)]


def random_walk(n, seed=7, drift=0.0004, vol=0.015):
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift + rng.gauss(0, vol)))
    return closes


ALWAYS = {"left": {"kind": "price"}, "op": ">", "right": {"kind": "value", "value": 0}}


# ---------- Properties of the simulator ----------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_always_long_without_costs_equals_buy_and_hold(seed):
    closes = random_walk(300, seed=seed)
    run = strategies.simulate(closes, [1] * 300, 0, 299, 0.0)
    # Buys at the first close, then holds: exactly the stock's return.
    assert run["equity"][-1] == pytest.approx(closes[-1] / closes[0], rel=1e-12)
    flat = strategies.simulate(closes, [0] * 300, 0, 299, 25.0)
    assert flat["equity"][-1] == 1.0 and flat["trades"] == [] and flat["turnover"] == 0


def test_next_open_execution_pays_the_overnight_gap():
    closes = [100.0, 100.0, 110.0, 110.0]
    opens = [100.0, 105.0, 108.0, 112.0]
    target = [1, 1, 1, 1]
    at_close = strategies.simulate(closes, target, 0, 3, 0.0)
    at_open = strategies.simulate(closes, target, 0, 3, 0.0, opens=opens, execution="next_open")
    assert at_close["equity"][-1] == pytest.approx(1.1)
    # The day-0 decision fills at day 1's open (105), so the 100 -> 105 gap is missed.
    assert at_open["equity"][-1] == pytest.approx(110 / 105)
    assert at_open["fills"][0]["index"] == 1 and at_open["fills"][0]["price"] == 105.0
    # Without opens it falls back to trading at the close.
    assert strategies.simulate(closes, target, 0, 3, 0.0, execution="next_open")["equity"][-1] == pytest.approx(1.1)


def test_buy_and_sell_costs_can_differ():
    closes = [100.0] * 4
    run = strategies.simulate(closes, [0, 1, 0, 0], 0, 3, 10.0, sell_cost_bps=30.0)
    assert run["equity"][-1] == pytest.approx((1 - 0.001) * (1 - 0.003))
    assert run["trades"][0]["exitValue"] == pytest.approx(run["equity"][-1])


def test_fixed_and_vol_target_sizing():
    closes = [100.0, 110.0, 99.0, 108.9]
    full = strategies.simulate(closes, strategies.size_positions([1] * 4, closes, "full"), 0, 3, 0.0)
    half = strategies.simulate(closes, strategies.size_positions([1] * 4, closes, "fixed", fraction=0.5), 0, 3, 0.0)
    assert full["equity"][-1] == pytest.approx(1.089)
    assert half["equity"][-1] == pytest.approx(1.05 * 0.95 * 1.05)

    noisy = random_walk(200, vol=0.04)
    sized = strategies.size_positions([1] * 200, noisy, "vol-target", target_vol=0.10, max_leverage=1.0)
    assert all(0 < w <= 1.0 for w in sized)
    assert max(sized[30:]) < 0.5  # ~64% annual volatility needs far less than a full position for a 10% target
    capped = strategies.size_positions([1] * 200, [100 + 0.001 * i for i in range(200)], "vol-target", max_leverage=2.0)
    assert max(capped) == 2.0


def test_shorting_profits_from_a_decline_and_pays_borrow():
    closes = [100 + i for i in range(30)] + [129 - 2 * i for i in range(40)]
    ts = dated(70)
    long_only = strategies.backtest(closes, ts, "sma-crossover", {"shortWindow": 3, "longWindow": 10}, 0)
    both = strategies.backtest(closes, ts, "sma-crossover", {"shortWindow": 3, "longWindow": 10}, 0, allow_short=True)
    assert both["metrics"]["shortTrades"] >= 0 and both["openTrade"]["side"] == "short"
    assert both["metrics"]["totalReturn"] > long_only["metrics"]["totalReturn"]
    costly = strategies.backtest(
        closes, ts, "sma-crossover", {"shortWindow": 3, "longWindow": 10}, 0, allow_short=True, borrow_bps=2000
    )
    assert costly["metrics"]["totalReturn"] < both["metrics"]["totalReturn"]
    assert strategies.current_signal("sma-crossover", closes, {"shortWindow": 3, "longWindow": 10}, allow_short=True)[
        "signal"
    ] in {"short", "hold"}


# ---------- Rule engine ----------


def test_intraday_stop_fills_at_the_stop_or_the_gapped_open():
    closes = [100.0, 100.0, 99.0, 98.0, 98.0]
    bars = {
        "open": [100.0, 100.0, 99.5, 85.0, 98.0],
        "high": [100.0, 101.0, 100.0, 99.0, 99.0],
        "low": [100.0, 99.0, 88.0, 84.0, 97.0],
    }
    spec = {"entry": [ALWAYS], "exit": [], "stopLoss": 0.1}
    target, _, fills = rules.plan(closes, spec, bars=bars)
    assert fills == {2: 90.0}  # the low of 88 crossed the 90 stop during day 2
    assert target[2] == 0 and target[3] == 1  # flat after the stop, back in on the next day's close
    run = strategies.simulate(closes, target, 0, 4, 0.0, fills=fills)
    assert run["trades"][0]["exitPrice"] == 90.0 and run["trades"][0]["exitValue"] == pytest.approx(0.9)

    gap = {"open": [100.0, 100.0, 80.0], "high": [100.0, 100.0, 81.0], "low": [100.0, 100.0, 79.0]}
    _, _, gap_fills = rules.plan([100.0, 100.0, 80.0], spec, bars=gap)
    assert gap_fills == {2: 80.0}  # gapped through the stop: filled at the open


def test_trailing_stop_time_exit_any_mode_and_short_side():
    up_then_down = [100.0, 110.0, 120.0, 130.0, 120.0, 115.0]
    trailed = rules.signals(up_then_down, {"entry": [ALWAYS], "trailingStop": 0.05})[0]
    assert trailed[:4] == [1, 1, 1, 1] and trailed[4] == 0  # 120 is more than 5% below the 130 peak

    timed = rules.signals([100.0] * 8, {"entry": [ALWAYS], "maxHoldDays": 3})[0]
    assert timed[:4] == [1, 1, 1, 0]

    never = {"left": {"kind": "price"}, "op": "<", "right": {"kind": "value", "value": 0}}
    assert set(rules.signals([100.0] * 5, {"entry": [never, ALWAYS]})[0]) == {0}
    assert set(rules.signals([100.0] * 5, {"entry": [never, ALWAYS], "entryMode": "any"})[0]) == {1}

    short = rules.signals([100.0, 100.0, 95.0, 111.0], {"entry": [ALWAYS], "side": "short", "stopLoss": 0.1})[0]
    assert short == [-1, -1, -1, 0]  # 111 is more than 10% above the 100 short entry
    assert "Sell short when" in rules.describe({"entry": [ALWAYS], "side": "short", "trailingStop": 0.1})


def test_new_indicators():
    closes = [100.0 + i for i in range(60)]
    line, signal_line, hist = rules.macd(closes)
    assert line[24] is None and line[25] is not None and signal_line[33] is not None
    assert hist[-1] == pytest.approx(line[-1] - signal_line[-1])
    assert rules.atr(closes, 14)[-1] == pytest.approx(1.0)  # close-to-close moves of 1
    cache = {}
    highest = rules.operand_series(closes, {"kind": "highest", "period": 5}, cache)
    assert highest[10] == closes[9]  # previous 5 days only
    roc = rules.operand_series(closes, {"kind": "roc", "period": 10}, cache)
    assert roc[20] == pytest.approx((120 / 110 - 1) * 100)
    assert rules.operand_series(closes, {"kind": "volume"}, cache, bars=None)[-1] is None
    assert (
        rules.warmup({"entry": [{"left": {"kind": "macd"}, "op": ">", "right": {"kind": "value", "value": 0}}]}) == 36
    )


def test_pine_export():
    spec = {
        "entry": [
            {"left": {"kind": "sma", "period": 50}, "op": "crosses_above", "right": {"kind": "sma", "period": 200}},
            {"left": {"kind": "macd_hist"}, "op": ">", "right": {"kind": "value", "value": 0}},
        ],
        "exit": [{"left": {"kind": "rsi", "period": 14}, "op": ">", "right": {"kind": "value", "value": 70}}],
        "stopLoss": 0.08,
        "takeProfit": 0.2,
        "maxHoldDays": 30,
    }
    script = rules.to_pine(spec, 'Golden "cross"')
    assert script.startswith("//@version=5")
    assert "ta.crossover(ta.sma(close, 50), ta.sma(close, 200))" in script
    assert "[macdLine, macdSignal, macdHist] = ta.macd(close, 12, 26, 9)" in script
    assert "stop=strategy.position_avg_price * 0.92, limit=strategy.position_avg_price * 1.2" in script
    assert "bar_index - entryBar >= 30" in script and "\"Golden 'cross'\"" in script


# ---------- Costs and risk ----------


def test_nse_costs_and_benchmarks():
    india = costs.cost_model("RELIANCE.NS", 10, 5)
    assert india["model"] == "NSE delivery"
    assert india["buyFeeBps"] == pytest.approx(10 + 0.297 + 0.01 + 1.5 + 0.18 * 0.307)
    assert india["sellBps"] == pytest.approx(india["buyBps"] - 1.5)
    us = costs.cost_model("AAPL", 10, 5)
    assert us["buyBps"] == us["sellBps"] == 15
    assert costs.benchmark_for("TCS.BO") == ("^NSEI", "NIFTY 50") and costs.benchmark_for("MSFT")[0] == "^GSPC"


def test_tail_risk_profit_factor_monthly_and_risk_free():
    returns = [-0.05] + [0.01] * 19
    tail = risk.tail_risk(returns)
    assert tail["var95"] == pytest.approx(0.05) and tail["cvar95"] == pytest.approx(0.05)
    assert risk.profit_factor([0.1, 0.2, -0.1]) == pytest.approx(3.0)
    assert risk.profit_factor([0.1]) is None
    months = risk.monthly_returns(["2026-01-30", "2026-01-31", "2026-02-02"], [1.0, 1.1, 1.21], 1.0)
    assert [m["month"] for m in months] == ["2026-01", "2026-02"]
    assert months[0]["return"] == pytest.approx(0.1) and months[1]["return"] == pytest.approx(0.1)
    rising = [0.001 + 0.0001 * math.sin(i) for i in range(100)]
    eq = [1.0]
    for r in rising:
        eq.append(eq[-1] * (1 + r))
    assert risk.summary(rising, eq, 0.05)["sharpe"] < risk.summary(rising, eq)["sharpe"]
    assert risk.rolling_sharpe(rising, 20)[19] is None and risk.rolling_sharpe(rising, 20)[20] is not None


def test_backtest_reports_new_metrics_and_series():
    closes = random_walk(400)
    ts = dated(400)
    bars = {"open": [c * 0.999 for c in closes], "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes]}
    report = strategies.backtest(
        closes,
        ts,
        "mean-reversion",
        {"lookback": 20, "deviation": 1.5},
        10,
        start=100,
        bars=bars,
        execution="next_open",
        sizing={"mode": "vol-target", "targetVol": 0.1},
        allow_short=True,
        risk_free=0.03,
    )
    m = report["metrics"]
    assert m["execution"] == "next_open" and m["sizing"] == "vol-target" and m["allowShort"]
    for key in ("var95", "cvar95", "turnover", "avgGrossExposure", "longTrades", "shortTrades"):
        assert key in m
    assert 0 < m["avgGrossExposure"] <= 1
    assert report["monthly"] and report["sample"][-1]["rollingSharpe"] is not None
    assert all(t["side"] in {"long", "short"} for t in report["trades"])


def test_golden_backtest_snapshot():
    """Pins the simulator's output on a fixed series, so an unintended change in the engine shows up here."""
    closes = random_walk(500, seed=42)
    ts = dated(500)
    report = strategies.backtest(closes, ts, "sma-crossover", {"shortWindow": 10, "longWindow": 30}, 10, start=100)
    m = report["metrics"]
    assert m["closedTrades"] == GOLDEN["closedTrades"]
    assert m["totalReturn"] == pytest.approx(GOLDEN["totalReturn"], rel=1e-9)
    assert m["sharpe"] == pytest.approx(GOLDEN["sharpe"], rel=1e-9)
    assert m["maxDrawdown"] == pytest.approx(GOLDEN["maxDrawdown"], rel=1e-9)


# Close-mode results, identical to the engine before next-open execution, sizing and shorting were added.
GOLDEN = {
    "closedTrades": 7,
    "totalReturn": 0.15815320811449207,
    "sharpe": 0.5431849946743128,
    "maxDrawdown": 0.19422383550201394,
}


# ---------- Strategies ----------


def test_regime_switch_and_efficiency_ratio():
    straight = [100.0 + i for i in range(40)]
    assert strategies.efficiency_ratio(straight, 10)[-1] == pytest.approx(1.0)
    zigzag = [100.0 + (i % 2) for i in range(40)]
    assert strategies.efficiency_ratio(zigzag, 10)[-1] < 0.2
    closes = random_walk(300, seed=5)
    target, lines = strategies.signals("regime-switch", closes, {"erWindow": 20, "erThreshold": 0.3})
    assert set(target) <= {0, 1} and "middleBand" in lines
    assert strategies.validate("regime-switch", {"erWindow": 2, "erThreshold": 0.3})
    outlook = strategies.current_signal("regime-switch", closes, {"erWindow": 20, "erThreshold": 0.3})
    assert "Efficiency ratio" in outlook["summary"]


def test_ml_boosted_model_and_multi_day_horizon_have_no_look_ahead():
    rng = random.Random(3)
    closes, r = [100.0], 0.01
    for _ in range(800):
        r = -0.9 * r + rng.gauss(0, 0.004)
        closes.append(closes[-1] * (1 + r))
    boosted = ml.report(closes, {"threshold": 0.5, "trainWindow": 504, "modelType": 1}, start=400)
    assert boosted["model"] == "Gradient-boosted trees" and boosted["auc"] > 0.75
    assert abs(sum(w["weight"] for w in boosted["featureWeights"]) - 1) < 1e-9
    assert sum(b["count"] for b in boosted["calibration"]) == boosted["predictions"]

    walk = random_walk(700)
    cut = 500
    for horizon in (1, 5):
        base = ml.probabilities(walk, 504, horizon=horizon)["probs"]
        changed = walk[: cut + 1] + [c * 1.5 for c in walk[cut + 1 :]]
        after = ml.probabilities(changed, 504, horizon=horizon)["probs"]
        assert base[: cut + 1] == after[: cut + 1]


def test_ml_extra_features_and_out_of_sample_threshold():
    closes = random_walk(800, seed=9)
    bars = {
        "volume": [1000 + (i * 37) % 500 for i in range(800)],
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
    }
    market = random_walk(800, seed=10)
    rows, names = ml.feature_table(closes, bars, market)
    assert len(names) == 12 and len(next(r for r in rows if r)) == 12
    report = ml.report(
        closes, {"threshold": 0.52, "trainWindow": 504}, start=700, cost_bps=15, bars=bars, market=market
    )
    assert len(report["features"]) == 12 and report["thresholdScan"]
    # The suggested threshold only depends on days before the evaluation window.
    changed = closes[:700] + [c * 2 for c in closes[700:]]
    again = ml.report(changed, {"threshold": 0.52, "trainWindow": 504}, start=700, cost_bps=15)
    plain = ml.report(closes, {"threshold": 0.52, "trainWindow": 504}, start=700, cost_bps=15)
    assert again["suggestedThreshold"] == plain["suggestedThreshold"]


# ---------- Robustness ----------


def test_monte_carlo_bands_are_ordered():
    returns = [0.001 + 0.01 * math.sin(i * 1.7) for i in range(300)]
    mc = robustness.monte_carlo(returns, [0.0] * 300, paths=200)
    for key in ("finalReturn", "maxDrawdown", "sharpe"):
        band = mc[key]
        assert band["p5"] <= band["p25"] <= band["p50"] <= band["p75"] <= band["p95"]
    assert mc["fan"][0]["p50"] == 1.0 and mc["fan"][-1]["step"] == 300
    assert 0 <= mc["probLoss"] <= 1 and mc["probBeatBuyHold"] is not None
    assert robustness.monte_carlo([0.01] * 10) is None


def test_deflated_sharpe_penalises_many_trials():
    rng = random.Random(1)
    returns = [0.0008 + rng.gauss(0, 0.01) for _ in range(500)]
    one = robustness.deflated_sharpe(returns, [robustness.daily_sharpe(returns)])
    many = robustness.deflated_sharpe(returns, [rng.gauss(0, 0.03) for _ in range(50)])
    assert many["expectedMaxSharpe"] > one["expectedMaxSharpe"] == 0
    assert many["deflatedSharpe"] < one["deflatedSharpe"]
    assert one["probabilisticSharpe"] == pytest.approx(one["deflatedSharpe"])


def test_pbo_separates_noise_from_a_real_edge():
    rng = random.Random(4)
    noise = [[rng.gauss(0, 0.01) for _ in range(400)] for _ in range(10)]
    edge = [row[:] for row in noise]
    edge[0] = [0.004 + rng.gauss(0, 0.002) for _ in range(400)]  # consistently the best everywhere
    assert robustness.pbo(edge)["pbo"] == 0.0
    assert 0.2 < robustness.pbo(noise)["pbo"] < 0.8
    assert robustness.pbo(noise[:2]) is None


def test_sensitivity_grid_marks_invalid_cells():
    closes = random_walk(500)
    grid, trials = robustness.sensitivity(closes, 250, "sma-crossover", {"shortWindow": 20, "longWindow": 60}, 15, 15)
    cells = grid["cells"]
    assert len(cells) == 7 * 6
    assert all(not c["valid"] for c in cells if c["x"] >= c["y"])
    assert len(trials) == sum(1 for c in cells if c["valid"])
    assert 0 <= grid["positiveShare"] <= 1


# ---------- Baskets ----------


def test_basket_alignment_and_weightings():
    a = random_walk(300, seed=1, vol=0.01)
    b = random_walk(300, seed=2, vol=0.03)
    ts = dated(300)
    data = {"A": (a, ts), "B": (b[:-5], ts[:-5])}
    stamps, closes = basket.align(data)
    assert len(stamps) == 295 and len(closes["A"]) == len(closes["B"]) == 295

    weights = basket.target_weights({"A": a, "B": b}, 299, "inverse-vol", 0)
    assert weights["A"] > weights["B"] and sum(weights.values()) == pytest.approx(1)
    assert basket.risk_parity([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx([0.5, 0.5])
    parity = basket.risk_parity([[4.0, 0.0], [0.0, 1.0]])
    assert parity[1] == pytest.approx(2 * parity[0])  # half the volatility, twice the weight

    held = basket.run({"A": (a, ts), "B": (b, ts)}, "buy-hold", {}, costs={"A": (0, 0), "B": (0, 0)}, eval_days=150)
    assert held["metrics"]["rebalances"] >= 4 and len(held["holdings"]) == 2
    assert held["curve"][0]["equity"] == 1.0
    momentum = basket.target_weights({"A": a, "B": b, "C": [100.0] * 300}, 299, "equal", 1)
    assert sorted(momentum.values()) == [0.0, 0.0, 1.0]


# ---------- Portfolio valuation ----------


def test_fx_conversion_and_ledger():
    closes = [100.0, 100.0, 100.0, 100.0]
    days = [f"2026-01-0{i + 1}T14:30:00+00:00" for i in range(4)]
    fx = portfolio.align_fx(
        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], {"2026-01-01": 0.01, "2026-01-03": 0.02}
    )
    assert fx == [0.01, 0.01, 0.02, 0.02]
    valued = portfolio.value_simulation(
        closes,
        days,
        strategy_id="buy-hold",
        params={},
        rules=None,
        start_date="2026-01-01",
        capital=1000,
        fee_bps=10,
        sell_fee_bps=20,
        slippage_bps=5,
        fx=fx,
    )
    # The stock didn't move, but the currency doubled against the base currency.
    assert valued["value"] == pytest.approx(1000 * 0.9985 * 2) and valued["fxReturn"] == pytest.approx(1.0)
    entry = valued["ledger"][0]
    assert entry["side"] == "buy" and entry["date"] == "2026-01-02"
    assert entry["notional"] == pytest.approx(100_000) and entry["fee"] == pytest.approx(100)
    assert entry["slippage"] == pytest.approx(50) and entry["shares"] == pytest.approx(1000)
    assert portfolio.align_fx(["2026-01-01"], {"*": 0.01}) == [0.01]


# ---------- API ----------


def auth(client, email="quant@example.com"):
    token = client.post("/auth/signup", json={"email": email, "password": "secret123", "name": "Quant"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_train_with_execution_sizing_and_shorting(client):
    headers = auth(client)
    body = {
        "symbol": "AAPL",
        "strategyId": "sma-crossover",
        "shortWindow": 5,
        "longWindow": 20,
        "execution": "next_open",
        "sizing": "vol-target",
        "targetVol": 0.2,
        "allowShort": True,
        "riskFreeRate": 0.02,
    }
    result = client.post("/analytics/train", headers=headers, json=body)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["metrics"]["execution"] == "next_open" and data["metrics"]["riskFreeRate"] == 0.02
    assert data["costs"]["model"] == "Flat fee" and "dailyReturns" not in data and data["monthly"]


def test_robustness_basket_and_pine_endpoints(client):
    headers = auth(client)
    robust = client.post(
        "/analytics/robustness",
        headers=headers,
        json={"symbol": "AAPL", "strategyId": "trend-follow", "channel": 20, "paths": 100},
    )
    assert robust.status_code == 200, robust.text
    data = robust.json()
    assert data["monteCarlo"]["paths"] == 100 and data["sensitivity"]["xParam"] == "channel"
    assert "timestamp" in data["monteCarlo"]["fan"][0] and data["deflatedSharpe"]["trials"] >= 1

    result = client.post(
        "/analytics/basket",
        headers=headers,
        json={"symbols": ["aapl", "MSFT", "msft"], "strategyId": "buy-hold", "weighting": "inverse-vol"},
    )
    assert result.status_code == 200, result.text
    basket_data = result.json()
    assert basket_data["symbols"] == ["AAPL", "MSFT"] and basket_data["crossSection"]["summary"]["count"] == 2
    assert client.post("/analytics/basket", headers=headers, json={"symbols": ["AAPL"]}).status_code == 422

    pine = client.post("/strategies/pine", headers=headers, json={"name": "RSI", "rules": {"entry": [ALWAYS]}})
    assert pine.status_code == 200 and "strategy.entry" in pine.json()["script"]


def test_alert_settings_signals_and_cron(client, monkeypatch):
    from backend.conftest import fake_chart
    from backend.routes import portfolio as portfolio_routes

    monkeypatch.setattr(portfolio_routes, "fetch_chart", fake_chart)
    headers = auth(client)
    assert client.get("/alerts/settings", headers=headers).json()["settings"] == {
        "email": False,
        "telegramChatId": None,
    }
    saved = client.put("/alerts/settings", headers=headers, json={"email": True, "telegramChatId": "12345"})
    assert saved.status_code == 200 and saved.json()["settings"]["telegramChatId"] == "12345"
    assert client.put("/alerts/settings", headers=headers, json={"telegramChatId": "abc"}).status_code == 422

    sim = {"symbol": "AAPL", "strategy": "Breakout", "strategyId": "trend-follow", "startingCapital": 100}
    assert client.post("/simulations", headers=headers, json=sim).status_code == 200
    signals = client.get("/alerts/signals", headers=headers).json()
    assert signals and signals[0]["symbol"] == "AAPL" and "signal" in signals[0]

    assert client.get("/cron/daily").status_code == 503  # no secret configured
    monkeypatch.setattr(settings, "cron_secret", "s3cret")
    assert client.get("/cron/daily", headers={"Authorization": "Bearer nope"}).status_code == 401
    run = client.get("/cron/daily", headers={"Authorization": "Bearer s3cret"})
    assert run.status_code == 200 and run.json()["simulations"] == 1
