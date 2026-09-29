import pytest

from backend import market


def chart_payload(closes, adj=None, meta=None):
    n = len(closes)
    indicators = {"quote": [{"open": closes, "high": closes, "low": closes, "close": closes, "volume": [100] * n}]}
    if adj is not None:
        indicators["adjclose"] = [{"adjclose": adj}]
    return {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "USD", "exchangeTimezoneName": "America/New_York", **(meta or {})},
                    "timestamp": [1_700_000_000 + i * 86_400 for i in range(n)],
                    "indicators": indicators,
                }
            ],
            "error": None,
        }
    }


def test_parse_chart_skips_incomplete_bars_and_adjusts():
    payload = chart_payload([10.0, None, 20.0], adj=[5.0, None, 20.0])
    chart = market.parse_chart(payload, "aapl", "1mo", "1d")
    assert chart["symbol"] == "AAPL" and chart["currency"] == "USD"
    assert [p["close"] for p in chart["points"]] == [5.0, 20.0]
    assert chart["points"][0]["open"] == 5.0  # OHLC scaled by the same dividend/split factor
    assert chart["points"][0]["timestamp"].endswith("+00:00")


def test_parse_chart_errors():
    with pytest.raises(market.MarketDataError):
        market.parse_chart({"chart": {"result": None, "error": {"code": "Not Found"}}}, "ZZZ", "1mo", "1d")
    with pytest.raises(market.MarketDataError):
        market.parse_chart(chart_payload([None, None]), "X", "1mo", "1d")


def test_quote_compares_with_previous_session():
    chart = market.parse_chart(chart_payload([100.0, 110.0], meta={"regularMarketPrice": 110.0}), "X", "5d", "1d")
    quote = market._quote_from_chart(chart)
    assert quote["previousClose"] == 100.0
    assert quote["changePercent"] == pytest.approx(10.0)


def test_fetch_quotes_skips_unknown_symbols(monkeypatch):
    def fake_chart(symbol, range_value, interval):
        if symbol == "NOPE":
            raise market.MarketDataError("symbol not found")
        return market.parse_chart(chart_payload([1.0, 2.0]), symbol, range_value, interval)

    monkeypatch.setattr(market, "yahoo_chart", fake_chart)
    monkeypatch.setattr(market.settings, "use_in_memory_db", False)
    assert [q["symbol"] for q in market.fetch_quotes(["msft", "NOPE"])] == ["MSFT"]
