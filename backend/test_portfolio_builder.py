"""Strategy Builder rules, portfolio valuation, and their API routes (the API tests run against both stores)."""

from datetime import date, timedelta

import pytest

from backend import portfolio, rules
from backend.routes import portfolio as portfolio_routes
from backend.schemas import StrategyRules
from backend.stores import now


def closes_up_down():
    return [100.0 - i for i in range(30)] + [71.0 + 2 * i for i in range(40)]


def test_indicators():
    assert rules.ema([1, 2, 3, 4], 2)[1:] == [1.5, pytest.approx(2.5), pytest.approx(3.5)]
    assert rules.rsi([1.0 + i for i in range(20)], 14)[-1] == 100.0  # only gains
    falling = rules.rsi([20.0 - i for i in range(20)], 14)
    assert falling[13] is None and falling[-1] == 0.0


def test_crossover_rules_and_description():
    spec = {
        "entry": [
            {"left": {"kind": "sma", "period": 5}, "op": "crosses_above", "right": {"kind": "sma", "period": 20}}
        ],
        "exit": [{"left": {"kind": "price"}, "op": "<", "right": {"kind": "value", "value": 50}}],
        "stopLoss": None,
        "takeProfit": None,
    }
    target, indicators = rules.signals(closes_up_down(), spec)
    first_buy = target.index(1)
    assert 30 < first_buy < 50 and all(target[first_buy:])  # bought during the rally, never exited
    assert set(indicators) == {"SMA5", "SMA20"}
    assert rules.warmup(spec) == 21
    assert rules.describe(spec) == "Buy when SMA(5) crosses above SMA(20); sell when Price is below 50."


def test_stop_loss_and_take_profit():
    prices = [100.0, 100.0, 95.0, 89.0, 100.0, 125.0]
    always = {"left": {"kind": "price"}, "op": ">", "right": {"kind": "value", "value": 0}}
    stop = rules.signals(prices, {"entry": [always], "exit": [], "stopLoss": 0.1, "takeProfit": None})[0]
    assert stop[:4] == [1, 1, 1, 0] and stop[4] == 1  # stopped out at 89 (-11%), re-entered next day
    take = rules.signals(prices, {"entry": [always], "exit": [], "stopLoss": None, "takeProfit": 0.2})[0]
    assert take[-1] == 0  # 125 is +25% over the 100 entry


def test_rules_schema_validation():
    with pytest.raises(ValueError):
        StrategyRules.model_validate({"entry": []})
    with pytest.raises(ValueError):
        StrategyRules.model_validate({"entry": [{"left": {"kind": "sma"}, "op": ">", "right": {"kind": "price"}}]})


def test_value_simulation_buy_hold_and_cash():
    closes = [100.0, 110.0, 121.0, 110.0]
    days = [f"2026-01-0{i + 1}T14:30:00+00:00" for i in range(4)]
    held = portfolio.value_simulation(
        closes, days, strategy_id="buy-hold", params={}, rules=None, start_date="2026-01-02", capital=1000, fee_bps=0
    )
    assert held["startDate"] == "2026-01-02" and held["startPrice"] == 110.0
    assert held["value"] == pytest.approx(1000.0) and held["inMarket"]
    assert held["dayChange"] == pytest.approx(1000 - 1100)
    assert [p["value"] for p in held["history"]] == pytest.approx([1000, 1100, 1000])

    never = {"entry": [{"left": {"kind": "price"}, "op": ">", "right": {"kind": "value", "value": 1e6}}]}
    cash = portfolio.value_simulation(
        closes, days, strategy_id="custom", params={}, rules=never, start_date="2026-01-01", capital=500, fee_bps=10
    )
    assert cash["value"] == 500 and not cash["inMarket"] and cash["trades"] == 0


def test_combine_counts_capital_before_start():
    a = {
        "id": "a",
        "symbol": "A",
        "startingCapital": 100,
        "value": 120,
        "dayChange": 5,
        "inMarket": True,
        "history": [{"date": "2026-01-01", "value": 100}, {"date": "2026-01-02", "value": 120}],
    }
    b = {
        "id": "b",
        "symbol": "B",
        "startingCapital": 50,
        "value": 40,
        "dayChange": -10,
        "inMarket": False,
        "history": [{"date": "2026-01-02", "value": 40}],
    }
    result = portfolio.combine([a, b])
    assert [p["value"] for p in result["history"]] == [150, 160]
    assert result["summary"]["totalValue"] == 160 and result["summary"]["pnl"] == 10
    assert result["summary"]["inMarket"] == 1
    assert [x["symbol"] for x in result["allocation"]] == ["A", "B"]


RSI_DIP = {
    "name": "RSI dip",
    "rules": {
        "entry": [{"left": {"kind": "rsi", "period": 14}, "op": "<", "right": {"kind": "value", "value": 60}}],
        "exit": [{"left": {"kind": "rsi", "period": 14}, "op": ">", "right": {"kind": "value", "value": 70}}],
        "stopLoss": 0.08,
    },
}


def auth(client, email="pat@example.com"):
    token = client.post("/auth/signup", json={"email": email, "password": "secret123", "name": "Pat"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_custom_strategy_crud_and_backtest(client):
    headers = auth(client)
    saved = client.post("/strategies/custom", headers=headers, json=RSI_DIP)
    assert saved.status_code == 200, saved.text
    assert client.get("/strategies/custom", headers=headers).json()[0]["name"] == "RSI dip"
    assert client.get("/strategies/custom", headers=auth(client, "other@example.com")).json() == []

    body = {"symbol": "AAPL", "strategyId": "custom", "rules": RSI_DIP["rules"]}
    trained = client.post("/analytics/train", headers=headers, json=body)
    assert trained.status_code == 200, trained.text
    assert trained.json()["rules"]["stopLoss"] == 0.08
    signal = client.post("/analytics/predict", headers=headers, json={**body, "parameters": {}}).json()
    assert signal["signal"] in {"buy", "hold", "sell", "wait"} and "Rules: Buy when RSI(14)" in signal["summary"]
    assert (
        client.post("/analytics/train", headers=headers, json={"symbol": "AAPL", "strategyId": "custom"}).status_code
        == 422
    )

    assert client.delete(f"/strategies/custom/{saved.json()['id']}", headers=headers).status_code == 204
    assert client.get("/strategies/custom", headers=headers).json() == []


def test_portfolio_values_backdated_simulations(client, monkeypatch):
    from backend.conftest import fake_chart

    def dated_chart(symbol, range_value="1mo", interval="1d"):
        chart = fake_chart(symbol)
        start = date.today() - timedelta(days=len(chart["points"]) - 1)
        for i, point in enumerate(chart["points"]):
            point["timestamp"] = f"{start + timedelta(days=i)}T14:30:00+00:00"
        return chart

    monkeypatch.setattr(portfolio_routes, "fetch_chart", dated_chart)
    headers = auth(client)
    week_ago = (date.today() - timedelta(days=7)).isoformat()
    for body in (
        {"symbol": "AAPL", "strategy": "Buy & hold", "startingCapital": 1000, "startDate": week_ago},
        {"symbol": "MSFT", "strategy": "Breakout", "strategyId": "trend-follow", "startingCapital": 500},
        {
            "symbol": "NVDA",
            "strategy": "RSI dip",
            "strategyId": "custom",
            "rules": RSI_DIP["rules"],
            "startingCapital": 250,
            "startDate": week_ago,
        },
    ):
        response = client.post("/simulations", headers=headers, json=body)
        assert response.status_code == 200, response.text

    data = client.get("/portfolio", headers=headers).json()
    summary = data["summary"]
    assert summary["positions"] == 3 and summary["totalCapital"] == 1750
    aapl = next(p for p in data["positions"] if p["symbol"] == "AAPL")
    # The fake chart rises 1/day, so a week of buy & hold made money (minus the entry fee).
    assert aapl["pnl"] > 0 and aapl["inMarket"] and aapl["startDate"] == week_ago
    assert summary["totalValue"] == pytest.approx(sum(p["value"] for p in data["positions"]))
    assert data["history"][-1]["value"] == pytest.approx(summary["totalValue"])
    assert abs(sum(a["weight"] for a in data["allocation"]) - 1) < 1e-9


def test_simulation_validation(client):
    headers = auth(client)
    base = {"symbol": "AAPL", "strategy": "X", "startingCapital": 100}
    utc_today = now().date()  # the server's clock
    future = (utc_today + timedelta(days=3)).isoformat()
    too_old = (utc_today - timedelta(days=400)).isoformat()
    assert client.post("/simulations", headers=headers, json=base | {"startDate": future}).status_code == 422
    # A visitor ahead of UTC may already be on tomorrow's date; that still counts as today.
    ahead = client.post("/simulations", headers=headers, json=base | {"startDate": str(utc_today + timedelta(days=1))})
    assert ahead.status_code == 200
    assert client.post("/simulations", headers=headers, json=base | {"startDate": too_old}).status_code == 422
    assert client.post("/simulations", headers=headers, json=base | {"strategyId": "astrology"}).status_code == 422
    assert client.post("/simulations", headers=headers, json=base | {"strategyId": "custom"}).status_code == 422
    ok = client.post("/simulations", headers=headers, json=base | {"strategyId": "sma-crossover"}).json()
    assert ok["parameters"] == {"shortWindow": 20, "longWindow": 60} and ok["startDate"] == utc_today.isoformat()
