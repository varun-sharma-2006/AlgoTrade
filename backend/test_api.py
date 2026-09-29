"""End-to-end API tests; the `client` fixture (backend/conftest.py) runs each against both stores."""

import pytest


def auth_headers(client, email="jane@example.com"):
    response = client.post("/auth/signup", json={"email": email, "password": "secret123", "name": "Jane"})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_signup_login_and_duplicate(client):
    auth_headers(client)
    assert (
        client.post(
            "/auth/signup", json={"email": "jane@example.com", "password": "secret123", "name": "J"}
        ).status_code
        == 409
    )
    assert client.post("/auth/login", json={"email": "jane@example.com", "password": "wrong"}).status_code == 401
    login = client.post("/auth/login", json={"email": "JANE@example.com", "password": "secret123"})
    assert login.status_code == 200 and login.json()["user"]["name"] == "Jane"


def test_auth_required(client):
    assert client.get("/simulations").status_code == 401
    assert client.get("/simulations", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_dev_bypass_is_idempotent(client):
    first = client.post("/dev/auth/bypass").json()
    second = client.post("/dev/auth/bypass").json()
    assert first["user"]["id"] == second["user"]["id"]


def test_simulation_crud(client):
    headers = auth_headers(client)
    created = client.post(
        "/simulations", headers=headers, json={"symbol": "msft", "strategy": "Momentum", "startingCapital": 5000}
    ).json()
    assert created["symbol"] == "MSFT" and created["status"] == "active"
    sim_id = created["id"]
    assert len(client.get("/simulations", headers=headers).json()) == 1
    updated = client.patch(f"/simulations/{sim_id}", headers=headers, json={"status": "completed"})
    assert updated.json()["status"] == "completed"
    overview = client.get("/analytics/overview", headers=headers).json()
    assert overview["totals"]["completedSimulations"] == 1
    # Another user can't see or delete it.
    other = auth_headers(client, "other@example.com")
    assert client.get("/simulations", headers=other).json() == []
    assert client.delete(f"/simulations/{sim_id}", headers=other).status_code == 404
    assert client.delete(f"/simulations/{sim_id}", headers=headers).status_code == 204
    assert client.delete("/simulations/not-an-id", headers=headers).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"symbol": "AAPL", "strategyId": "sma-crossover", "shortWindow": 10, "longWindow": 30},
        {"symbol": "AAPL", "strategyId": "mean-reversion", "lookback": 20, "deviation": 2},
        {"symbol": "AAPL", "strategyId": "trend-follow", "channel": 20},
    ],
)
def test_train_and_predict_each_strategy(client, body):
    headers = auth_headers(client)
    assert client.post("/analytics/predict", headers=headers, json={"symbol": "AAPL"}).status_code == 404
    trained = client.post("/analytics/train", headers=headers, json=body)
    assert trained.status_code == 200, trained.text
    result = trained.json()
    assert result["strategyId"] == body["strategyId"]
    assert {"totalReturn", "buyHoldReturn", "winRate", "sharpe", "maxDrawdown", "exposure"} <= result["metrics"].keys()
    prediction = client.post("/analytics/predict", headers=headers, json={"symbol": "aapl"}).json()
    assert prediction["signal"] in {"buy", "hold", "sell", "wait"}
    assert prediction["strategyId"] == body["strategyId"]
    assert client.get("/analytics/overview", headers=headers).json()["totals"]["trainedModels"] == 1


def test_train_rejects_bad_parameters(client):
    headers = auth_headers(client)
    bad = {"symbol": "AAPL", "strategyId": "sma-crossover", "shortWindow": 50, "longWindow": 20}
    assert client.post("/analytics/train", headers=headers, json=bad).status_code == 422
    unknown = {"symbol": "AAPL", "strategyId": "astrology"}
    assert client.post("/analytics/train", headers=headers, json=unknown).status_code == 422


def test_health_and_strategies(client):
    assert client.get("/health").json()["status"] == "ok"
    ids = {s["id"] for s in client.get("/analytics/strategies").json()}
    assert ids == {"sma-crossover", "mean-reversion", "trend-follow"}
