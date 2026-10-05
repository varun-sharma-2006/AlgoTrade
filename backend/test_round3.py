"""Earnings dates and blackouts, survivorship (point-in-time S&P 500), news tone and Web Push."""

import base64
import random
from datetime import date, timedelta

import pytest

from backend import earnings, ml, news, push, strategies, survivorship
from backend.config import settings
from backend.scripts import vapid_keys


def dated(n, start=date(2025, 1, 6)):
    return [f"{start + timedelta(days=i)}T14:30:00+00:00" for i in range(n)]


# ---------- Earnings ----------


def test_earnings_dates_from_sec_filings(monkeypatch):
    monkeypatch.setattr(settings, "sec_contact_email", "")
    assert earnings.earnings_dates("AAPL") is None  # no contact email: feature off, no request made

    monkeypatch.setattr(settings, "sec_contact_email", "someone@example.com")
    earnings._cache.clear()
    responses = {
        earnings.TICKERS_URL: {
            "0": {"cik_str": 320193, "ticker": "AAPL"},
            "1": {"cik_str": 1067983, "ticker": "BRK.B"},
        },
        earnings.SUBMISSIONS_URL.format(cik=320193): {
            "filings": {
                "recent": {
                    "form": ["8-K", "10-Q", "8-K", "8-K/A", "8-K"],
                    "filingDate": ["2026-07-30", "2026-08-01", "2026-05-01", "2026-04-30", "2026-03-10"],
                    "items": ["2.02,9.01", "", "5.07", "2.02", "8.01"],
                }
            }
        },
    }
    monkeypatch.setattr(earnings, "_get", lambda url: responses[url])
    assert earnings.earnings_dates("AAPL") == ["2026-04-30", "2026-07-30"]
    assert earnings.cik_for("BRK-B") == 1067983 and earnings.earnings_dates("ZZZZ") is None


def test_earnings_windows_and_blackout():
    stamps = dated(12)  # 2025-01-06 .. 2025-01-17
    blackout, reaction = earnings.windows(stamps, ["2025-01-09", "2025-01-30"])
    assert reaction == {3, 4} and blackout == {2, 3}
    closes = [100.0, 101, 102, 103, 110, 99, 100, 101, 102, 103, 104, 105]
    always = strategies.backtest(closes, stamps, "buy-hold", {}, 0, reaction_bars=reaction)
    skip = strategies.backtest(closes, stamps, "buy-hold", {}, 0, blackout=blackout, reaction_bars=reaction)
    assert always["metrics"]["earningsReturn"] == pytest.approx((103 / 102 - 1) + (110 / 103 - 1))
    assert skip["metrics"]["earningsReturn"] == 0 and skip["metrics"]["avoidedEarnings"]
    assert [p["position"] for p in skip["sample"]][2:4] == [0, 0]

    # With next-open fills the blackout starts a day earlier, and no earnings-day return is earned at all.
    opens = [c * 1.01 for c in closes]
    early, reaction = earnings.windows(stamps, ["2025-01-09"], next_open=True)
    assert early == {1, 2, 3}
    gapped = strategies.backtest(
        closes,
        stamps,
        "buy-hold",
        {},
        0,
        bars={"open": opens},
        execution="next_open",
        blackout=early,
        reaction_bars=reaction,
    )
    assert gapped["metrics"]["earningsReturn"] == pytest.approx(0)


# ---------- Survivorship ----------


CONSTITUENTS = """<table id="constituents"><tr><th>Symbol</th><th>Security</th></tr>
<tr><td><a>AAA</a></td><td>A</td></tr><tr><td>BRK.B</td><td>B</td></tr><tr><td>NEW</td><td>N</td></tr></table>"""
CHANGES = """<table id="changes"><tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td rowspan="2">March 3, 2025</td><td>NEW</td><td>New Co</td><td>OLD</td><td>Old Co</td><td>Reason</td></tr>
<tr><td>XYZ</td><td>Xyz</td><td>GONE</td><td>Gone Co</td><td>Reason</td></tr>
<tr><td>January 2, 2020</td><td>AAA</td><td>A</td><td></td><td></td><td>Reason</td></tr></table>"""


def test_point_in_time_membership():
    members = survivorship.parse_constituents(CONSTITUENTS)
    changes = survivorship.parse_changes(CHANGES)
    assert members == ["AAA", "BRK-B", "NEW"]
    assert changes == [
        {"date": "2025-03-03", "added": "NEW", "removed": "OLD"},
        {"date": "2025-03-03", "added": "XYZ", "removed": "GONE"},
        {"date": "2020-01-02", "added": "AAA", "removed": None},
    ]
    assert survivorship.members_on("2024-12-31", members, changes) == {"AAA", "BRK-B", "OLD", "GONE"}
    assert survivorship.members_on("2019-06-01", members, changes) == {"BRK-B", "OLD", "GONE"}
    result = survivorship.check(["NEW", "AAA", "TSLA"], "2024-12-31", (members, changes))
    assert result["inIndexThen"] == ["AAA"] and result["joinedLater"] == [{"symbol": "NEW", "date": "2025-03-03"}]
    assert result["notInIndex"] == ["TSLA"] and {r["symbol"] for r in result["removedSince"]} == {"OLD", "GONE"}
    assert result["survivorShare"] == pytest.approx(0.5)


def test_survivorship_routes(client, monkeypatch):
    members, changes = survivorship.parse_constituents(CONSTITUENTS), survivorship.parse_changes(CHANGES)
    monkeypatch.setattr(survivorship, "load", lambda: (members, changes))
    token = client.post("/auth/signup", json={"email": "s@example.com", "password": "secret123", "name": "S"}).json()[
        "token"
    ]
    headers = {"Authorization": f"Bearer {token}"}
    check = client.post("/analytics/survivorship", headers=headers, json={"symbols": ["NEW"], "years": 3}).json()
    assert check["joinedLater"][0]["symbol"] == "NEW"
    sample = client.get("/analytics/sp500-sample?size=3&years=3&seed=1", headers=headers).json()
    assert set(sample["symbols"]) <= {"AAA", "BRK-B", "OLD", "GONE"} and len(sample["symbols"]) == 3
    monkeypatch.setattr(survivorship, "load", lambda: None)
    assert client.get("/analytics/sp500-sample", headers=headers).status_code == 502


# ---------- News tone ----------


def test_news_query_alignment_and_ml_feature(monkeypatch):
    assert news.company_query("Apple Inc.", "AAPL") == '"apple"'
    assert news.company_query("Reliance Industries Limited", "RELIANCE.NS") == '"reliance industries"'
    assert news.company_query(None, "BRK-B") == '"BRK"'
    stamps = dated(10)
    tone = {"2025-01-06": 5.0, "2025-01-07": -1.0, "2025-01-10": 3.0}
    series = news.aligned(stamps, tone, window=2)
    # Day t only sees the two days before it, never its own day.
    assert series[:6] == [None, 5.0, 2.0, -1.0, None, 3.0]

    closes = [100 * (1 + 0.01 * random.Random(i).gauss(0, 1)) for i in range(400)]
    with_news = ml.feature_table(closes, {"news": [0.5] * 400})
    without = ml.feature_table(closes, {})
    assert with_news[1][-1] == ml.NEWS_FEATURE and len(with_news[1]) == len(without[1]) + 1
    sparse = ml.feature_table(closes, {"news": [None] * 399 + [1.0]})
    assert ml.NEWS_FEATURE not in sparse[1]  # too little coverage: feature skipped

    news._cache.clear()

    class Busy:
        status_code = 429
        text = "Please limit requests"

    monkeypatch.setattr(news.requests, "get", lambda *a, **k: Busy())
    assert news.daily_tone('"nobody"') is None


# ---------- Web Push ----------


def test_vapid_keys_and_unconfigured_push(monkeypatch):
    public, private = vapid_keys.generate()
    assert len(base64.urlsafe_b64decode(public + "==")) == 65 and len(base64.urlsafe_b64decode(private + "=")) == 32
    monkeypatch.setattr(settings, "vapid_public_key", "")
    assert not push.configured() and push.send({"endpoint": "https://x"}, "t", "b") == "failed"


def test_push_subscription_routes(client, monkeypatch):
    token = client.post("/auth/signup", json={"email": "p@example.com", "password": "secret123", "name": "P"}).json()[
        "token"
    ]
    headers = {"Authorization": f"Bearer {token}"}
    sub = {"endpoint": "https://push.example.com/abc", "keys": {"p256dh": "k", "auth": "a"}}
    monkeypatch.setattr(settings, "vapid_public_key", "")
    assert client.get("/push/key").json() == {"publicKey": None, "available": False}
    assert client.post("/push/subscribe", headers=headers, json=sub).status_code == 503

    public, private = vapid_keys.generate()
    monkeypatch.setattr(settings, "vapid_public_key", public)
    monkeypatch.setattr(settings, "vapid_private_key", private)
    sent = []
    monkeypatch.setattr(push, "send", lambda s, title, body, url="/": sent.append((s["endpoint"], title)) or "gone")
    assert client.post("/push/subscribe", headers=headers, json=sub).json()["devices"] == 1
    assert client.get("/push/key").json()["publicKey"] == public
    test = client.post("/alerts/test", headers=headers)
    # The (fake) push service says the subscription is gone, so nothing was delivered and it was removed.
    assert sent == [("https://push.example.com/abc", "BUY TEST")] and test.status_code == 503
    assert client.post("/alerts/test", headers=headers).status_code == 422  # no channels left
    client.post("/push/subscribe", headers=headers, json=sub)
    assert client.post("/push/unsubscribe", headers=headers, json=sub).json() == {"subscribed": False}
