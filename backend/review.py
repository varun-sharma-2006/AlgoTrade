"""An honest review of a backtest: the checks a sceptical quant would run before trusting the result.

Every finding is computed from the numbers (so it is reproducible and testable); an LLM, when available, only
turns the findings into a short summary and is told not to add numbers of its own.
"""

from __future__ import annotations

from typing import Any

GOOD, WARN, BAD = "good", "warn", "bad"


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def findings(
    report: dict[str, Any],
    *,
    close_report: dict[str, Any] | None = None,
    robustness: dict[str, Any] | None = None,
    walk_forward: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Ordered list of {level, title, detail} for a backtest (plus optional extra tests)."""
    m = report["metrics"]
    bh = report["buyHold"]
    out: list[dict[str, str]] = []

    def add(level: str, title: str, detail: str) -> None:
        out.append({"level": level, "title": title, "detail": detail})

    excess = m["excessReturn"]
    if excess >= 0 and m["sharpe"] >= bh["sharpe"]:
        add(
            GOOD,
            "Beat buy & hold",
            f"{_pct(excess)} more return with a higher Sharpe ratio ({m['sharpe']:.2f} vs {bh['sharpe']:.2f}).",
        )
    elif excess >= 0:
        add(
            WARN,
            "More return, but rougher",
            f"{_pct(excess)} vs buy & hold, but a lower Sharpe ratio ({m['sharpe']:.2f} vs {bh['sharpe']:.2f}): "
            "the extra return came with more risk.",
        )
    elif m["sharpe"] >= bh["sharpe"]:
        add(
            WARN,
            "Less return, but smoother",
            f"It made {_pct(excess)} less than buy & hold, but its Sharpe ratio was higher ({m['sharpe']:.2f} vs "
            f"{bh['sharpe']:.2f}), with a {m['maxDrawdown'] * 100:.1f}% worst fall vs {bh['maxDrawdown'] * 100:.1f}%.",
        )
    else:
        add(
            BAD,
            "Lost to buy & hold",
            f"{_pct(excess)} vs simply holding, and a lower Sharpe ratio ({m['sharpe']:.2f} vs {bh['sharpe']:.2f}).",
        )

    closed = m["closedTrades"]
    if closed < 10:
        add(
            BAD if closed < 5 else WARN,
            "Too few trades to judge",
            f"Only {closed} closed trades. With so few, a couple of lucky trades can explain the whole result.",
        )

    gains = sorted((t["gain"] for t in report.get("_trades") or [] if t["gain"] > 0), reverse=True)
    total_gain = sum(gains)
    if gains and total_gain > 0 and closed >= 3:
        share = gains[0] / total_gain
        if share > 0.5:
            add(
                WARN,
                "One trade did most of the work",
                f"The best trade made {share * 100:.0f}% of all the winning trades' profit.",
            )

    turnover = m.get("turnover") or 0.0
    cost_drag = turnover * (m["feeBps"] + m.get("sellFeeBps", m["feeBps"]) + 2 * m.get("slippageBps", 0)) / 2 / 10_000
    if cost_drag > 0.03:
        add(
            WARN,
            "Costs eat a lot",
            f"Trading {turnover:.1f}× the portfolio a year costs about {cost_drag * 100:.1f}% a year in fees and slippage.",
        )

    if close_report is not None:
        diff = close_report["metrics"]["totalReturn"] - m["totalReturn"]
        if abs(diff) > 0.05:
            add(
                WARN,
                "Sensitive to execution",
                f"Filling at the same day's close instead of the next open changes the return by {_pct(diff)}. "
                "Results that depend on the exact fill price are fragile.",
            )

    overnight, intraday = m.get("overnightReturn"), m.get("intradayReturn")
    if overnight is not None and intraday is not None and abs(overnight) + abs(intraday) > 0.05:
        share = overnight / (abs(overnight) + abs(intraday))
        if share > 0.6:
            add(
                WARN,
                "Profits come from overnight gaps",
                f"Most of the price gain while invested ({share * 100:.0f}%) happened between the close and the next "
                "open, which includes earnings and news gaps you can't react to.",
            )

    earned = m.get("earningsReturn")
    total = m.get("totalReturn")
    if earned is not None and total is not None and total > 0.02 and earned > 0.5 * total:
        add(
            WARN,
            "Profits depend on earnings days",
            f"About {_pct(earned)} of the return came on the {m.get('earningsDays') or 'few'} days around earnings "
            "announcements. Try the 'skip earnings' option to see the strategy without them.",
        )

    participation = m.get("maxParticipation")
    if participation is not None and participation > 0.01:
        add(
            WARN,
            "Large orders for this stock",
            f"The biggest order was {participation * 100:.1f}% of a normal day's traded value; real fills would move the price.",
        )

    if robustness:
        dsr = robustness.get("deflatedSharpe")
        if dsr:
            p = dsr["deflatedSharpe"]
            add(
                GOOD if p >= 0.95 else BAD if p < 0.5 else WARN,
                "Deflated Sharpe ratio",
                f"{p * 100:.0f}% confidence the Sharpe ratio beats what the best of {dsr['trials']} random settings "
                "would show by luck (95% is the usual bar).",
            )
        pbo = robustness.get("pbo")
        if pbo:
            add(
                BAD if pbo["pbo"] > 0.5 else GOOD if pbo["pbo"] < 0.25 else WARN,
                "Probability of overfitting",
                f"The best in-sample setting fell into the bottom half out of sample in {pbo['pbo'] * 100:.0f}% of splits.",
            )
        grid = robustness.get("sensitivity")
        if grid:
            share = grid["positiveShare"]
            add(
                GOOD if share >= 0.7 else BAD if share < 0.4 else WARN,
                "Nearby settings",
                f"{share * 100:.0f}% of nearby settings had a positive Sharpe ratio.",
            )
        mc = robustness.get("monteCarlo")
        if mc:
            add(
                GOOD if mc["probLoss"] < 0.2 else WARN,
                "Range of outcomes",
                f"Resampling the returns gave a {mc['probLoss'] * 100:.0f}% chance of losing money; the 1-in-20 bad "
                f"case lost {mc['maxDrawdown']['p95'] * 100:.0f}% from a peak.",
            )
        rand = robustness.get("randomEntries")
        if rand:
            pct = rand["percentile"]
            add(
                GOOD if pct >= 0.9 else BAD if pct < 0.6 else WARN,
                "Better than random timing?",
                f"It beat {pct * 100:.0f}% of strategies with the same trades placed at random times.",
            )
        factor = robustness.get("factors")
        if factor:
            add(
                GOOD if factor["alphaSignificant"] and factor["alpha"] > 0 else WARN,
                "Factor alpha",
                f"After market, size, value, profitability, investment and momentum, the unexplained return was "
                f"{_pct(factor['alpha'])} a year (t = {factor['alphaT']:.1f}; |t| ≥ 2 counts as significant).",
            )

    if walk_forward:
        wm = walk_forward["metrics"]
        decay = wm["inSampleAnnualized"] - wm["outOfSampleAnnualized"]
        add(
            BAD if decay > 0.1 else WARN if decay > 0.03 else GOOD,
            "Walk-forward",
            f"Tuned settings made {_pct(wm['inSampleAnnualized'])} a year in sample but {_pct(wm['outOfSampleAnnualized'])} "
            "on data they hadn't seen.",
        )
    return out


def verdict(items: list[dict[str, str]]) -> dict[str, Any]:
    """A one-line overall rating from the findings."""
    bad = sum(1 for f in items if f["level"] == BAD)
    good = sum(1 for f in items if f["level"] == GOOD)
    if bad >= 2:
        label, text = (
            "Not trustworthy",
            "Several red flags: treat this backtest as luck or overfitting until proven otherwise.",
        )
    elif bad == 1:
        label, text = "Doubtful", "There is a serious weakness to fix before relying on this strategy."
    elif good >= max(3, len(items) // 2):
        label, text = "Promising", "It passes most checks. Paper-trade it before risking money."
    else:
        label, text = "Inconclusive", "No fatal flaw, but not enough evidence of a real edge yet."
    return {"label": label, "text": text, "good": good, "bad": bad, "warnings": len(items) - good - bad}


SUMMARY_PROMPT = (
    "You review trading backtests for a retail investor. Using ONLY the findings provided, write 3 or 4 short "
    "sentences in plain English: what the result says, the biggest risk, and what to test next. Do not introduce "
    "any number that is not in the findings. No markdown, no headings, no advice to buy or sell."
)


def summarise(symbol: str, strategy: str, items: list[dict[str, str]], overall: dict[str, Any]) -> str | None:
    from backend.gemini import ask_text

    lines = "\n".join(f"- [{f['level']}] {f['title']}: {f['detail']}" for f in items)
    return ask_text(SUMMARY_PROMPT, f"{strategy} on {symbol}. Overall: {overall['label']}.\n{lines}")
