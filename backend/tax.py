"""After-tax returns: capital-gains tax on a backtest's closed trades.

India (listed shares and equity funds, from 23 July 2024): gains on positions held more than 12 months are
long-term and taxed at 12.5% above a ₹1.25 lakh exemption per financial year (April to March); shorter
holdings are short-term and taxed at 20%. Both get the 4% health and education cess. Short-term losses can
be set off against short- or long-term gains, long-term losses only against long-term gains, and unused
losses carry forward. Short selling counts as short-term. Crypto ("virtual digital assets") pays a flat 30%
plus cess on each profitable trade, and losses can't be set off at all.

United States: positions held more than a year are long-term (15% by default), everything else is short-term
at your ordinary income rate (24% by default). Net losses on one side offset gains on the other, and unused
losses carry forward (the $3,000 a year deduction against other income is ignored).

Taxes are worked out per tax year and subtracted from the final value; in reality they would be paid each
year, which would slightly reduce compounding. Surcharges, state taxes and the US net investment income tax
are not included.
"""

from __future__ import annotations

from datetime import date
from typing import Any

INDIA_SHORT_RATE = 0.20
INDIA_LONG_RATE = 0.125
INDIA_LTCG_EXEMPTION = 125_000.0
INDIA_CRYPTO_RATE = 0.30
INDIA_CESS = 0.04
LONG_TERM_DAYS = 365


def _date(timestamp: str) -> date | None:
    try:
        return date.fromisoformat(timestamp[:10])
    except ValueError:
        return None


def _tax_year(day: date | None, region: str) -> str:
    if day is None:
        return "unknown"
    if region == "IN":
        start = day.year if day.month >= 4 else day.year - 1
        return f"FY{start}-{str(start + 1)[2:]}"
    return str(day.year)


def tax_report(
    trades: list[dict[str, Any]],
    capital: float,
    final_equity: float,
    *,
    region: str,
    crypto: bool = False,
    currency: str = "USD",
    us_short_rate: float = 0.24,
    us_long_rate: float = 0.15,
) -> dict[str, Any]:
    """Tax on closed trades. Each trade has entryDate, exitDate, side and gain (in units of starting capital)."""
    years: dict[str, dict[str, float]] = {}
    for trade in trades:
        entry, exit_ = _date(trade["entryDate"]), _date(trade["exitDate"])
        held = (exit_ - entry).days if entry and exit_ else 0
        long_term = trade["side"] == "long" and held > LONG_TERM_DAYS
        gain = trade["gain"] * capital
        year = years.setdefault(_tax_year(exit_, region), {"short": 0.0, "long": 0.0, "flat": 0.0})
        if region == "IN" and crypto:
            year["flat"] += max(gain, 0.0)  # every profitable trade is taxed; losses are simply lost
        elif long_term:
            year["long"] += gain
        else:
            year["short"] += gain

    india = region == "IN"
    short_rate = INDIA_SHORT_RATE * (1 + INDIA_CESS) if india else us_short_rate
    long_rate = INDIA_LONG_RATE * (1 + INDIA_CESS) if india else us_long_rate
    carry_short = carry_long = 0.0
    rows = []
    total = 0.0
    for name in sorted(years):
        st, lt, flat = years[name]["short"], years[name]["long"], years[name]["flat"]
        if india and crypto:
            tax = flat * INDIA_CRYPTO_RATE * (1 + INDIA_CESS)
            rows.append({"year": name, "shortTermGain": flat, "longTermGain": 0.0, "taxableLongTerm": 0.0, "tax": tax})
            total += tax
            continue
        # Losses carried in from earlier years.
        use = min(carry_short, max(st, 0.0))
        st, carry_short = st - use, carry_short - use
        use = min(carry_short, max(lt, 0.0))
        lt, carry_short = lt - use, carry_short - use
        use = min(carry_long, max(lt, 0.0))
        lt, carry_long = lt - use, carry_long - use
        if not india:  # in the US, long-term losses can also offset short-term gains
            use = min(carry_long, max(st, 0.0))
            st, carry_long = st - use, carry_long - use
        # This year's losses on one side against gains on the other.
        if st < 0:
            use = min(-st, max(lt, 0.0))
            lt, st = lt - use, st + use
            carry_short += -st
            st = 0.0
        if lt < 0:
            if not india:
                use = min(-lt, max(st, 0.0))
                st, lt = st - use, lt + use
            carry_long += -lt
            lt = 0.0
        taxable_long = max(lt - INDIA_LTCG_EXEMPTION, 0.0) if india else lt
        tax = st * short_rate + taxable_long * long_rate
        total += tax
        rows.append(
            {
                "year": name,
                "shortTermGain": years[name]["short"],
                "longTermGain": years[name]["long"],
                "taxableLongTerm": taxable_long,
                "tax": tax,
            }
        )

    pre_tax = final_equity - 1
    if india and crypto:
        rules = "India, crypto: flat 30% + 4% cess on each profitable trade; losses can't be set off"
    elif india:
        rules = "India: STCG 20%, LTCG 12.5% above ₹1.25 lakh a year (both + 4% cess); April-March years"
    else:
        rules = f"Short-term {us_short_rate:.0%}, long-term (over a year) {us_long_rate:.0%}; calendar years"
    return {
        "region": region,
        "crypto": crypto,
        "currency": currency,
        "capital": capital,
        "rules": rules,
        "totalTax": total,
        "preTaxReturn": pre_tax,
        "afterTaxReturn": pre_tax - total / capital if capital else pre_tax,
        "carryForwardLoss": carry_short + carry_long,
        "years": rows,
    }
