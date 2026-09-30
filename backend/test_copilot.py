"""Copilot actions: the tools, the offline request parser, Gemini function calling and the chat endpoint."""

from datetime import date, timedelta

import pytest

from backend import copilot, gemini
from backend.config import settings
from backend.routes import analytics, chat
from backend.schemas import ChatRequest


def long_chart(symbol, range_value="1mo", interval="1d"):
    """Four years of gently trending, wavy prices ending today."""
    n = 1000
    start = date.today() - timedelta(days=n - 1)
    points = []
    for i in range(n):
        close = 100 * (1.0004**i) * (1 + 0.05 * ((i % 40) - 20) / 20)
        points.append({"timestamp": f"{start + timedelta(days=i)}T14:30:00+00:00", "close": close})
    return {"symbol": symbol.upper(), "points": points}


@pytest.fixture
def market(monkeypatch):
    monkeypatch.setattr(analytics, "fetch_chart", long_chart)
    analytics._benchmark_cache.clear()
    yield
    analytics._benchmark_cache.clear()


def test_run_backtest_tool_returns_a_summary_and_an_action(market):
    box = copilot.Toolbox()
    result = box.call("run_backtest", {"symbol": "aapl", "strategy": "sma-crossover", "shortWindow": 10})
    assert result["symbol"] == "AAPL" and result["strategy"] == "SMA 10/60"
    assert result["sp500Return"] is not None and result["costPerTradeBps"] == 15
    assert box.actions[0]["type"] == "backtest"
    assert box.call("run_backtest", {"symbol": "aapl", "strategy": "sma-crossover", "shortWindow": 10}) == result
    assert len(box.actions) == 1  # a repeated call is answered from memory
    assert "error" in box.call("run_backtest", {"symbol": "AAPL", "strategy": "astrology"})
    assert "error" in box.call("run_backtest", {"strategy": "buy-hold"})


def test_compare_and_walk_forward_tools(market):
    box = copilot.Toolbox()
    ranked = box.call("compare_strategies", {"symbol": "MSFT"})["rankedBySharpe"]
    assert {row["strategyId"] for row in ranked} == {
        "sma-crossover",
        "mean-reversion",
        "trend-follow",
        "ml-logistic",
        "buy-hold",
    }
    assert [r["sharpe"] for r in ranked] == sorted((r["sharpe"] for r in ranked), reverse=True)
    wf = box.call("walk_forward_test", {"symbol": "MSFT", "strategy": "trend-follow"})
    assert wf["folds"] >= 3 and "outOfSampleAnnualized" in wf
    assert [a["type"] for a in box.actions] == ["comparison", "walkforward"]


def test_create_simulation_tool_validates_and_runs_once(market):
    created = []

    def create(payload):
        created.append(payload)
        return {"id": "sim1", "symbol": payload.symbol, "startDate": str(payload.startDate)}

    box = copilot.Toolbox(create)
    month_ago = (date.today() - timedelta(days=30)).isoformat()
    args = {"symbol": "NVDA", "strategy": "ml-logistic", "startingCapital": 5000, "startDate": month_ago}
    assert box.call("create_simulation", args)["created"] is True
    box.call("create_simulation", args)
    assert len(created) == 1 and created[0].parameters == {"threshold": 0.52, "trainWindow": 504}
    too_old = args | {"startDate": "2000-01-01"}
    assert "at most one year" in box.call("create_simulation", too_old)["error"]
    assert "error" in copilot.Toolbox().call("create_simulation", args)  # no store available


def test_local_action_understands_common_requests(market):
    box = copilot.Toolbox()
    reply = copilot.local_action("Backtest AAPL with SMA 10/30", box)
    assert reply.startswith("Backtest: AAPL · SMA 10/30")
    assert "buy & hold" in reply and "S&P 500" in reply
    assert copilot.local_action("compare strategies for tesla", box).startswith("All built-in strategies on TSLA")
    assert "Walk-forward test: MSFT" in copilot.local_action("walk-forward test MSFT with bollinger", box)
    assert "Model:" in copilot.local_action("backtest NVDA with machine learning", box)
    assert copilot.local_action("what is RSI?", box) is None
    assert copilot.local_action("backtest a strategy", box) is None  # no ticker


class FakeResponse:
    def __init__(self, content, status=200):
        self.status_code = status
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"candidates": [{"content": self._content}]}


def test_gemini_calls_tools_and_sends_back_the_results(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_models", ["model-a"])
    monkeypatch.setattr(gemini, "_cooldown_until", {})
    bodies = []
    replies = iter(
        [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "run_backtest", "args": {"symbol": "AAPL", "strategy": "buy-hold"}}},
                    {"thoughtSignature": "sig"},
                ],
            },
            {"role": "model", "parts": [{"text": "Buy & hold returned 12%."}]},
        ]
    )

    def fake_post(url, json, headers, timeout):
        bodies.append(json)
        return FakeResponse(next(replies))

    monkeypatch.setattr(gemini.requests, "post", fake_post)
    calls = []

    def run_tool(name, args):
        calls.append((name, args))
        return {"strategyReturn": 0.12}

    reply = gemini.ask_gemini(ChatRequest(message="backtest AAPL"), None, copilot.TOOL_DECLARATIONS, run_tool)
    assert reply == "Buy & hold returned 12%."
    assert calls == [("run_backtest", {"symbol": "AAPL", "strategy": "buy-hold"})]
    assert bodies[0]["tools"][0]["functionDeclarations"][0]["name"] == "run_backtest"
    followup = bodies[1]["contents"]
    assert followup[-2]["parts"][1] == {"thoughtSignature": "sig"}  # the model's turn is sent back unchanged
    assert followup[-1]["parts"][0]["functionResponse"] == {
        "name": "run_backtest",
        "response": {"strategyReturn": 0.12},
    }


def test_chat_runs_a_backtest_without_gemini(client, monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    monkeypatch.setattr(chat, "load_closes", lambda symbols: {})
    token = client.post("/dev/auth/bypass").json()["token"]
    response = client.post(
        "/chat", headers={"Authorization": f"Bearer {token}"}, json={"message": "Backtest AAPL with SMA 20/60"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"].startswith("Backtest: AAPL · SMA 20/60")
    assert body["actions"][0]["type"] == "backtest"
