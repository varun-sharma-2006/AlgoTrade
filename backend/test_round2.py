"""Intraday/crypto annualisation, market impact, taxes, SIPs, factors, options, reviews, reports and alerts."""

import io
import json
import math
import random
import zipfile
from datetime import date, timedelta

import pytest

from backend import (
    factors,
    nlrules,
    notebook,
    options,
    pricecache,
    review,
    risk,
    robustness,
    sip,
    strategies,
    tax,
    walkforward,
)
from backend.config import settings


def dated(n, start=date(2021, 1, 4)):
    return [f"{start + timedelta(days=i)}T14:30:00+00:00" for i in range(n)]


def random_walk(n, seed=7, drift=0.0004, vol=0.015):
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift + rng.gauss(0, vol)))
    return closes


# ---------- Annualisation for crypto and hourly bars ----------


def test_bars_per_year_and_annualisation():
    daily = dated(300)
    hourly = [f"2026-03-{d:02d}T{h:02d}:30:00+00:00" for d in range(1, 29) for h in range(14, 21)]
    assert strategies.bars_per_year(daily) == 252
    assert strategies.bars_per_year(daily, "BTC-USD") == 365
    assert strategies.bars_per_year(hourly) == 252 * 7
    assert strategies.is_crypto("eth-inr") and not strategies.is_crypto("BRK-B")
    assert risk.annualise(0.21, 504) == pytest.approx(0.1)
    assert risk.annualise(0.21, 730, 365) == pytest.approx(0.1)
    returns = [0.01, -0.01] * 50
    assert risk.summary(returns, [1.0] * 101, 0, 365)["volatility"] == pytest.approx(
        risk.stdev(returns) * math.sqrt(365)
    )


def test_intraday_backtest_compares_daily_closes_with_the_index():
    stamps = [f"{date(2025, 1, 1) + timedelta(days=d)}T{h:02d}:00:00+00:00" for d in range(120) for h in range(14, 21)]
    closes = random_walk(len(stamps), vol=0.004)
    index = random_walk(120, seed=3)
    bench = ([f"{date(2025, 1, 1) + timedelta(days=d)}T21:00:00+00:00" for d in range(120)], index)
    report = strategies.backtest(
        closes, stamps, "buy-hold", {}, 0, start=0, benchmark=bench, periods=strategies.bars_per_year(stamps)
    )
    assert report["metrics"]["periodsPerYear"] == 252 * 7
    assert report["benchmark"] is not None and abs(report["benchmark"]["beta"]) < 1  # daily, not hourly, pairs


# ---------- Market impact and overnight/intraday split ----------


def test_market_impact_grows_with_order_size():
    closes = random_walk(200, vol=0.02)
    bars = {"volume": [10_000] * 200, "open": closes[:1] + closes[:-1]}
    ts = dated(200)
    args = dict(fee_bps=0, start=50, bars=bars)
    params = {"shortWindow": 5, "longWindow": 20}
    small = strategies.backtest(closes, ts, "sma-crossover", params, impact={"capital": 1_000}, **args)
    large = strategies.backtest(closes, ts, "sma-crossover", params, impact={"capital": 10_000_000}, **args)
    none = strategies.backtest(closes, ts, "sma-crossover", params, **args)
    assert none["metrics"]["impactCost"] is None
    assert 0 < small["metrics"]["impactCost"] < large["metrics"]["impactCost"]
    assert large["metrics"]["maxParticipation"] > 1  # bigger than a whole day's trading
    assert large["metrics"]["totalReturn"] < small["metrics"]["totalReturn"] <= none["metrics"]["totalReturn"]


def test_overnight_and_intraday_split_adds_up():
    closes = [100.0, 102.0, 101.0, 105.0]
    opens = [100.0, 101.0, 103.0, 104.0]
    run = strategies.simulate(closes, [1, 1, 1, 1], 0, 3, 0.0, opens=opens)
    assert run["overnight"] == pytest.approx((101 / 100 - 1) + (103 / 102 - 1) + (104 / 101 - 1))
    assert run["intraday"] == pytest.approx((102 / 101 - 1) + (101 / 103 - 1) + (105 / 104 - 1))


# ---------- Taxes ----------


def trade(entry, exit_, gain, side="long"):
    return {"entryDate": entry, "exitDate": exit_, "gain": gain, "side": side}


def test_india_tax_rates_exemption_and_set_off():
    capital = 1_000_000
    report = tax.tax_report(
        [
            trade("2024-05-01", "2025-06-01", 0.3),  # long-term, FY2025-26: ₹3,00,000
            trade("2025-05-01", "2025-07-01", -0.05),  # short-term loss: ₹-50,000, offsets the long-term gain
            trade("2025-08-01", "2025-09-01", 0.1),  # short-term: ₹1,00,000 but the loss already went to LTCG first?
        ],
        capital,
        1.35,
        region="IN",
        currency="INR",
    )
    year = report["years"][0]
    assert year["year"] == "FY2025-26"
    # Net short-term +50,000 (100k - 50k); long-term 300k minus the 1.25 lakh exemption.
    expected = 50_000 * 0.20 * 1.04 + (300_000 - 125_000) * 0.125 * 1.04
    assert report["totalTax"] == pytest.approx(expected)
    assert report["afterTaxReturn"] == pytest.approx(0.35 - expected / capital)

    losses_carried = tax.tax_report(
        [trade("2025-01-01", "2025-02-01", -0.2), trade("2025-06-01", "2025-07-01", 0.1)],
        capital,
        0.9,
        region="IN",
    )
    assert [y["year"] for y in losses_carried["years"]] == ["FY2024-25", "FY2025-26"]
    assert losses_carried["totalTax"] == 0 and losses_carried["carryForwardLoss"] == pytest.approx(100_000)


def test_india_crypto_has_no_loss_set_off_and_us_rules():
    crypto = tax.tax_report(
        [trade("2025-05-01", "2025-06-01", 0.1), trade("2025-06-02", "2025-07-01", -0.1)],
        100_000,
        1.0,
        region="IN",
        crypto=True,
    )
    assert crypto["totalTax"] == pytest.approx(10_000 * 0.30 * 1.04)
    us = tax.tax_report(
        [trade("2023-01-02", "2024-03-01", 0.2), trade("2024-04-01", "2024-05-01", 0.1, side="short")],
        10_000,
        1.3,
        region="US",
        us_short_rate=0.3,
        us_long_rate=0.1,
    )
    assert us["totalTax"] == pytest.approx(2_000 * 0.1 + 1_000 * 0.3)


# ---------- SIP ----------


def monthly_prices(months, growth):
    stamps, closes = [], []
    price = 100.0
    for m in range(months):
        for d in (1, 15):
            stamps.append(f"{2022 + m // 12}-{m % 12 + 1:02d}-{d:02d}T14:30:00+00:00")
            closes.append(price)
        price *= 1 + growth
    return stamps, closes


def test_xirr_and_sip_against_lump_sum():
    assert sip.xirr([(date(2025, 1, 1), -100), (date(2026, 1, 1), 110)]) == pytest.approx(0.1, abs=1e-6)
    stamps, closes = monthly_prices(24, 0.01)
    result = sip.run(closes, stamps, monthly=1000, start=0, step_up=0.1)
    assert result["months"] == 24 and result["invested"] == pytest.approx(12 * 1000 + 12 * 1100)
    # Prices rise 1% at each month start; the series ends two weeks after the last instalment, with no rise.
    assert 0.10 < result["sip"]["xirr"] < 1.01**12 - 1
    assert result["lumpSum"]["value"] > result["sip"]["value"]  # rising prices favour investing early
    waiting = sip.run(closes, stamps, monthly=1000, start=0, timing=[0] * len(closes))
    assert waiting["timed"]["cashWaiting"] == pytest.approx(waiting["invested"])
    always = sip.run(closes, stamps, monthly=1000, start=0, timing=[1] * len(closes))
    assert always["timed"]["value"] == pytest.approx(always["sip"]["value"])
    assert result["sip"]["taxIfRedeemed"] > 0


# ---------- Factors ----------


def test_factor_regression_and_csv_parser():
    rng = random.Random(5)
    market = [rng.gauss(0, 0.01) for _ in range(300)]
    size = [rng.gauss(0, 0.005) for _ in range(300)]
    y = [0.0004 + 1.5 * m - 0.5 * s + rng.gauss(0, 0.0005) for m, s in zip(market, size, strict=True)]
    beta, errors, r2 = factors.regress(y, [[m, s] for m, s in zip(market, size, strict=True)])
    assert beta == pytest.approx([0.0004, 1.5, -0.5], abs=0.01) and r2 > 0.98

    text = "Some description\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20260105,1.00,0.50,-0.20,0.10,0.00,0.02\n20260106,-0.50,0.10,0.10,0.00,0.10,0.02\n\nAnnual,x\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", text)
    rows = factors._parse(buffer.getvalue())
    assert rows["2026-01-05"] == pytest.approx(
        {"Mkt-RF": 0.01, "SMB": 0.005, "HML": -0.002, "RMW": 0.001, "CMA": 0.0, "RF": 0.0002}
    )

    stamps = dated(101)
    table = {
        ts[:10]: {"Mkt-RF": m, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0, "Mom": 0.0, "RF": 0.0}
        for ts, m in zip(stamps[1:], market[:100], strict=True)
    }
    for i, row in enumerate(table.values()):  # avoid perfectly collinear zero columns
        for k, name in enumerate(("SMB", "HML", "RMW", "CMA", "Mom")):
            row[name] = math.sin(i * (k + 1.3)) * 0.001
    strat = [2 * m for m in market[:100]]
    attribution = factors.attribution(stamps, strat, table)
    market_row = next(row for row in attribution["loadings"] if row["factor"] == "Mkt-RF")
    assert market_row["beta"] == pytest.approx(2.0, abs=1e-6) and attribution["rSquared"] == pytest.approx(1.0)


# ---------- Robustness additions ----------


def test_random_entries_and_sixty_forty():
    closes = [100 * 1.001**i for i in range(400)]
    lucky = robustness.random_entries(closes, 0, [20] * 5, 10.0, 0)
    assert lucky["percentile"] == 1.0
    unlucky = robustness.random_entries(closes, 0, [20] * 5, -1.0, 0)
    assert unlucky["percentile"] == 0.0
    assert robustness.random_entries(closes, 0, [500], 0.1, 0) is None
    stamps = dated(200)
    mix = robustness.sixty_forty((stamps, [100 * 1.01**i for i in range(200)]), (stamps, [100.0] * 200), stamps[0])
    assert 0 < mix["totalReturn"] < 1.01**199 - 1


def test_walk_forward_custom_rules_scales_variants():
    spec = {
        "entry": [
            {"left": {"kind": "sma", "period": 20}, "op": "crosses_above", "right": {"kind": "sma", "period": 60}}
        ],
        "exit": [
            {"left": {"kind": "sma", "period": 20}, "op": "crosses_below", "right": {"kind": "sma", "period": 60}}
        ],
        "stopLoss": 0.1,
    }
    scaled = walkforward.scale_rules(spec, periodScale=0.75, stopScale=1.25)
    assert scaled["entry"][0]["left"]["period"] == 15 and scaled["stopLoss"] == pytest.approx(0.125)
    result = walkforward.run(random_walk(900, seed=4), dated(900), "custom", 10, rules=spec)
    assert result["gridSize"] == 9 and set(result["folds"][0]["params"]) == {"periodScale", "stopScale"}


# ---------- Plain English, review, options, notebook ----------


def test_plain_english_parser():
    parsed = nlrules.parse(
        "Buy when RSI(2) is below 10 and price is above the 200-day SMA; sell when RSI(2) goes above 70, 5% stop loss"
    )
    rules = parsed["rules"]
    assert rules["entry"][0] == {
        "left": {"kind": "rsi", "period": 2},
        "op": "<",
        "right": {"kind": "value", "value": 10.0},
    }
    assert rules["entry"][1]["right"] == {"kind": "sma", "period": 200}
    assert rules["exit"][0]["op"] == ">" and rules["stopLoss"] == 0.05
    short = nlrules.parse("go short on a new 20-day low, cover on a 10-day high, trailing stop of 8%")["rules"]
    assert short["side"] == "short" and short["trailingStop"] == 0.08
    assert short["entry"][0]["right"] == {"kind": "lowest", "period": 20}
    assert nlrules.parse("the weather is nice")["rules"] is None


def test_review_findings_flag_weak_backtests():
    report = {
        "metrics": {
            "excessReturn": -0.2,
            "sharpe": 0.3,
            "closedTrades": 3,
            "turnover": 30,
            "feeBps": 10,
            "sellFeeBps": 10,
            "slippageBps": 5,
            "maxDrawdown": 0.2,
            "overnightReturn": 0.3,
            "intradayReturn": 0.02,
            "maxParticipation": None,
        },
        "buyHold": {"sharpe": 0.9, "maxDrawdown": 0.3},
        "_trades": [{"gain": 0.3}, {"gain": 0.01}, {"gain": -0.02}],
    }
    items = review.findings(
        report, robustness={"pbo": {"pbo": 0.8}, "deflatedSharpe": {"deflatedSharpe": 0.2, "trials": 30}}
    )
    titles = {f["title"]: f["level"] for f in items}
    assert titles["Lost to buy & hold"] == "bad" and titles["Too few trades to judge"] == "bad"
    assert titles["One trade did most of the work"] == "warn" and titles["Costs eat a lot"] == "warn"
    assert titles["Profits come from overnight gaps"] == "warn" and titles["Probability of overfitting"] == "bad"
    assert review.verdict(items)["label"] == "Not trustworthy"
    mixed = review.findings(report | {"metrics": report["metrics"] | {"excessReturn": 0.2}})
    assert mixed[0]["title"] == "More return, but rougher" and mixed[0]["level"] == "warn"


def test_black_scholes_parity_and_option_income():
    call = options.black_scholes("call", 100, 105, 0.5, 0.03, 0.25)
    put = options.black_scholes("put", 100, 105, 0.5, 0.03, 0.25)
    assert call - put == pytest.approx(100 - 105 * math.exp(-0.03 * 0.5))
    flat = [100.0 + 0.5 * math.sin(i) for i in range(300)]
    covered = options.backtest(flat, dated(300), 0, strategy="covered-call", risk_free=0.0, cost_bps=0)
    assert covered["metrics"]["totalReturn"] > 0 and covered["metrics"]["rolls"] >= 10
    crash = [100.0] * 50 + [60.0] * 250
    put_seller = options.backtest(crash, dated(300), 0, strategy="cash-secured-put", risk_free=0.0)
    assert put_seller["metrics"]["assigned"] >= 1 and put_seller["metrics"]["totalReturn"] < 0
    with pytest.raises(ValueError):
        options.backtest(flat, dated(300), 0, strategy="iron-condor")


def test_notebook_reproduces_the_close_mode_backtest():
    pd = pytest.importorskip("pandas")
    closes = random_walk(600, seed=8)
    stamps = [date(2021, 1, 4) + timedelta(days=i) for i in range(600)]
    params = {"shortWindow": 10, "longWindow": 30}
    nb = notebook.build(
        "TEST",
        "sma-crossover",
        params,
        None,
        title="t",
        execution="close",
        buy_cost_bps=10,
        sell_cost_bps=10,
        risk_free=0.0,
        allow_short=False,
        backtest_days=300,
        summary="s",
    )
    json.dumps(nb)
    code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code" and not c["source"].startswith("%pip")]
    namespace = {
        "df": pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1}, index=pd.to_datetime(stamps)
        )
    }
    exec("import numpy as np\nimport pandas as pd", namespace)
    exec(code[1], namespace)  # settings
    exec(code[3], namespace)  # signal
    exec(code[4], namespace)  # backtest
    ts = [f"{d}T00:00:00+00:00" for d in stamps]
    start = strategies.evaluation_start(ts, 300)
    app = strategies.backtest(closes, ts, "sma-crossover", params, 10, start=start)
    assert namespace["equity"].iloc[-1] - 1 == pytest.approx(app["metrics"]["totalReturn"], abs=1e-9)

    custom = notebook.build(
        "TEST", "custom", {}, {"entry": [{"left": {"kind": "rsi", "period": 14}, "op": "<", "right": {"kind": "value", "value": 30}}]},
        title="t", execution="next_open", buy_cost_bps=10, sell_cost_bps=10, risk_free=0.0, allow_short=False,
        backtest_days=300, summary="s",
    )  # fmt: skip
    for cell in custom["cells"]:
        if cell["cell_type"] == "code" and not cell["source"].startswith("%pip"):
            compile(cell["source"], "cell", "exec")
    with pytest.raises(ValueError):
        notebook.build("X", "ml-logistic", {}, None, title="", execution="close", buy_cost_bps=0, sell_cost_bps=0,
                       risk_free=0, allow_short=False, backtest_days=1, summary="")  # fmt: skip


def test_price_cache_merge_keeps_the_window():
    old = {"points": [{"timestamp": f"2026-01-0{i}T00:00:00", "close": i} for i in range(1, 8)], "range": "5y"}
    recent = {"points": [{"timestamp": f"2026-01-0{i}T00:00:00", "close": i * 10} for i in range(6, 10)]}
    merged = pricecache.merge(old, recent)
    assert [p["close"] for p in merged["points"]] == [3, 4, 5, 60, 70, 80, 90] and merged["range"] == "5y"


# ---------- API ----------


def auth(client, email="round2@example.com"):
    token = client.post("/auth/signup", json={"email": email, "password": "secret123", "name": "Ravi Kumar"}).json()[
        "token"
    ]
    return {"Authorization": f"Bearer {token}"}


def dated_chart(symbol, range_value="1mo", interval="1d"):
    from backend.conftest import fake_chart

    chart = fake_chart(symbol)
    chart["currency"] = "INR" if symbol.upper().endswith(".NS") else "USD"
    closes = random_walk(1200, seed=len(symbol))
    start = date.today() - timedelta(days=1199)
    chart["points"] = [
        {"timestamp": f"{start + timedelta(days=i)}T14:30:00+00:00", "open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 5000}
        for i, c in enumerate(closes)
    ]  # fmt: skip
    return chart


@pytest.fixture
def dated_market(monkeypatch):
    from backend.routes import analytics
    from backend.routes import portfolio as portfolio_routes

    monkeypatch.setattr(analytics, "fetch_chart", dated_chart)
    monkeypatch.setattr(portfolio_routes, "fetch_chart", dated_chart)
    monkeypatch.setattr(factors, "load_factors", lambda: None)
    monkeypatch.setattr(settings, "google_api_key", None)


def test_train_reports_tax_impact_and_interval(client, dated_market):
    headers = auth(client)
    body = {
        "symbol": "RELIANCE.NS",
        "strategyId": "sma-crossover",
        "shortWindow": 5,
        "longWindow": 20,
        "marketImpact": True,
    }
    data = client.post("/analytics/train", headers=headers, json=body).json()
    assert data["tax"]["region"] == "IN" and data["tax"]["capital"] == 1_000_000
    assert data["metrics"]["impactCost"] is not None and "_trades" not in data
    hourly = client.post("/analytics/train", headers=headers, json=body | {"interval": "1h", "symbol": "AAPL"}).json()
    assert hourly["interval"] == "1h" and hourly["tax"]["region"] == "US"


def test_research_endpoints(client, dated_market):
    headers = auth(client)
    base = {"symbol": "AAPL", "strategyId": "sma-crossover", "shortWindow": 10, "longWindow": 40}
    opt = client.post("/analytics/optimize", headers=headers, json=base)
    assert opt.status_code == 200, opt.text
    assert opt.json()["trials"] > 10 and "shortWindow" in opt.json()["best"]["parameters"]
    rev = client.post("/analytics/review", headers=headers, json=base | {"paths": 100}).json()
    assert rev["verdict"]["label"] and rev["findings"] and rev["summary"] is None  # no Gemini key
    board = client.post("/analytics/leaderboard", headers=headers, json={"symbols": ["AAPL", "MSFT"]}).json()
    assert board["rows"] and board["rows"][0]["score"] >= board["rows"][-1]["score"]
    plan = client.post(
        "/analytics/sip", headers=headers, json={"symbol": "TCS.NS", "monthly": 5000, "timingStrategy": "sma-crossover"}
    )
    assert plan.status_code == 200, plan.text
    assert plan.json()["region"] == "IN" and plan.json()["timed"]["value"] > 0
    opt_lab = client.post("/analytics/options", headers=headers, json={"symbol": "SPY", "strategy": "cash-secured-put"})
    assert opt_lab.status_code == 200 and opt_lab.json()["metrics"]["rolls"] > 5
    rules = client.post(
        "/strategies/from-text", headers=headers, json={"text": "buy when rsi is below 30, sell when rsi is above 60"}
    )
    assert rules.json()["source"] == "parser" and rules.json()["rules"]["entry"][0]["left"]["kind"] == "rsi"
    assert client.post("/strategies/from-text", headers=headers, json={"text": "hello there"}).status_code == 422
    nb = client.post("/strategies/notebook", headers=headers, json=base)
    assert nb.status_code == 200 and nb.json()["nbformat"] == 4
    wf = client.post(
        "/analytics/walk-forward",
        headers=headers,
        json={"symbol": "AAPL", "strategyId": "custom", "rules": {"entry": [{"left": {"kind": "price"}, "op": ">", "right": {"kind": "sma", "period": 50}}]}},
    )  # fmt: skip
    assert wf.status_code == 200, wf.text


def test_reports_are_public_and_owned(client, dated_market):
    headers = auth(client)
    created = client.post(
        "/reports", headers=headers, json={"symbol": "AAPL", "strategyId": "trend-follow", "paths": 100}
    )
    assert created.status_code == 200, created.text
    report_id = created.json()["id"]
    assert created.json()["path"] == f"/r/{report_id}" and created.json()["author"] == "Ravi"
    public = client.get(f"/reports/{report_id}").json()  # no login needed
    assert public["content"]["backtest"]["symbol"] == "AAPL" and "userId" not in public
    assert public["content"]["review"]["verdict"]["label"]
    assert client.get("/reports", headers=headers).json()[0]["id"] == report_id
    page = client.get(f"/share/{report_id}")
    assert page.status_code == 200 and "refresh" in page.text  # no built frontend in tests: redirect to the app
    assert client.delete(f"/reports/{report_id}", headers=auth(client, "other@example.com")).status_code == 404
    assert client.delete(f"/reports/{report_id}", headers=headers).status_code == 204
    assert client.get(f"/reports/{report_id}").status_code == 404


def test_watch_alerts_rate_limit_and_broker_guard(client, dated_market, monkeypatch):
    headers = auth(client)
    alert = {
        "symbol": "nvda",
        "condition": {"left": {"kind": "price"}, "op": ">", "right": {"kind": "value", "value": 1}},
        "note": "always",
    }
    created = client.post("/alerts/watch", headers=headers, json=alert)
    assert created.status_code == 200, created.text
    listed = client.get("/alerts/watch", headers=headers).json()
    assert listed[0]["symbol"] == "NVDA" and listed[0]["status"]["triggered"] is True
    monkeypatch.setattr(settings, "cron_secret", "s")
    run = client.get("/cron/daily", headers={"Authorization": "Bearer s"}).json()
    assert run["watchAlerts"] == 1 and run["signals"] == 1
    again = client.get("/cron/daily", headers={"Authorization": "Bearer s"}).json()
    assert again["signals"] == 0  # already alerted today
    assert client.delete(f"/alerts/watch/{created.json()['id']}", headers=headers).status_code == 204

    sim = client.post(
        "/simulations", headers=headers, json={"symbol": "AAPL", "strategy": "x", "startingCapital": 100}
    ).json()
    assert client.patch(f"/simulations/{sim['id']}", headers=headers, json={"brokerMirror": True}).status_code == 403

    monkeypatch.setattr(settings, "heavy_requests_per_minute", 1)
    body = {"symbol": "AAPL", "strategyId": "trend-follow", "paths": 100}
    assert client.post("/analytics/robustness", headers=headers, json=body).status_code == 200
    assert client.post("/analytics/robustness", headers=headers, json=body).status_code == 429
