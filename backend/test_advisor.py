from backend import advisor


def rising(n=130, start=100.0, step=0.004):
    # Steady uptrend with a small wobble so RSI is not pinned at 100.
    return [start * (1 + step) ** i * (1 + (0.01 if i % 3 else -0.012)) for i in range(n)]


def falling(n=130, start=100.0):
    return list(reversed(rising(n, start)))


def fake_loader(data):
    return lambda symbols: {s: data[s] for s in symbols if s in data}


def test_snapshot_detects_trends():
    up = advisor.build_snapshot("UP", rising())
    down = advisor.build_snapshot("DOWN", falling())
    assert up.uptrend and up.return_3m > 0
    assert down.downtrend and down.signal == "AVOID for now"
    assert advisor.build_snapshot("SHORT", [1.0] * 10) is None


def test_extract_symbols_handles_names_and_tickers():
    assert advisor.extract_symbols("should I buy apple or TSLA?") == ["AAPL", "TSLA"]
    assert advisor.extract_symbols("what is RSI and SMA") == []
    assert "RELIANCE.NS" in advisor.extract_symbols("analyse reliance")


def test_parse_budget_and_horizon():
    assert advisor.parse_budget("I have $5,000 for 30 days") == 5000
    assert advisor.parse_budget("invest 2 lakh rupees") == 200000
    assert advisor.parse_budget("what about 5 days") is None
    assert advisor.parse_horizon_days("for 2 weeks") == 14


def test_recommend_ranks_uptrend_first_and_gives_plan():
    data = {sym: falling() for sym in advisor.US_UNIVERSE}
    data["MSFT"] = rising()
    data["SPY"] = rising()
    result = advisor.local_answer("which stock should i buy and why with 10000", fake_loader(data))
    reply = result["reply"]
    assert "Top pick: MSFT" in reply
    assert "Stop-loss" in reply and "Sizing for your 10,000" in reply


def test_recommend_without_setups_says_wait():
    data = {sym: falling() for sym in advisor.US_UNIVERSE + ["SPY"]}
    reply = advisor.local_answer("give me a stock to buy", fake_loader(data))["reply"]
    assert "No stock passes the buy rules" in reply
    assert "Caution: the broad market" in reply


def test_concepts_and_fallbacks():
    loader = fake_loader({})
    assert "Relative Strength Index" in advisor.local_answer("what is rsi?", loader)["reply"]
    assert "golden cross" in advisor.local_answer("explain a golden cross", loader)["reply"]
    assert "couldn't find price history" in advisor.local_answer("analyse ZZZZ", loader)["reply"]
    assert "overloaded" in advisor.local_answer("capital of france?", loader)["reply"]
