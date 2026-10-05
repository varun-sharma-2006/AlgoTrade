"""Trading costs and benchmarks per market.

US stocks pay a flat fee (TRADING_FEE_BPS) on every buy and sell. Indian stocks on the NSE/BSE pay the
real delivery charges instead, which differ between buying and selling:

    STT 0.1% on both sides, exchange transaction charge ~0.00297%, SEBI fee 0.0001%,
    stamp duty 0.015% on buys only, brokerage (0 for most discount brokers on delivery),
    and 18% GST on brokerage + exchange + SEBI charges.

Slippage is added to both sides in every market.
"""

from __future__ import annotations

from typing import Any

# NSE delivery charges, in basis points of traded value.
NSE_STT_BPS = 10.0
NSE_EXCHANGE_BPS = 0.297
NSE_SEBI_BPS = 0.01
NSE_STAMP_BUY_BPS = 1.5
GST_RATE = 0.18

BENCHMARKS = {
    "US": ("^GSPC", "S&P 500"),
    "IN": ("^NSEI", "NIFTY 50"),
    "CRYPTO": ("BTC-USD", "Bitcoin"),
}


def market(symbol: str) -> str:
    """Market of a symbol: CRYPTO for crypto pairs, IN for NSE/BSE listings and Indian indices, US otherwise."""
    from backend.strategies import is_crypto

    upper = symbol.upper()
    if is_crypto(upper):
        return "CRYPTO"
    if upper.endswith((".NS", ".BO")) or upper in {"^NSEI", "^BSESN", "^NSEBANK"}:
        return "IN"
    return "US"


def tax_region(symbol: str, base_currency: str = "USD") -> tuple[str, bool]:
    """Which capital-gains rules apply: Indian rules for NSE/BSE stocks and for crypto held by an INR-based
    investor, US rules otherwise; and whether the asset is crypto."""
    kind = market(symbol)
    if kind == "CRYPTO":
        return ("IN" if base_currency.upper() == "INR" or symbol.upper().endswith("-INR") else "US"), True
    return ("IN" if kind == "IN" else "US"), False


def benchmark_for(symbol: str) -> tuple[str, str]:
    """The index a symbol is compared with: NIFTY 50 for Indian stocks, Bitcoin for crypto, the S&P 500 otherwise."""
    return BENCHMARKS[market(symbol)]


def cost_model(symbol: str, fee_bps: float, slippage_bps: float, india_brokerage_bps: float = 0.0) -> dict[str, Any]:
    """Cost of a buy and of a sell, in basis points of traded value, with a breakdown for display."""
    if market(symbol) == "IN":
        taxed = india_brokerage_bps + NSE_EXCHANGE_BPS + NSE_SEBI_BPS
        common = india_brokerage_bps + NSE_STT_BPS + NSE_EXCHANGE_BPS + NSE_SEBI_BPS + GST_RATE * taxed
        buy, sell = common + NSE_STAMP_BUY_BPS, common
        return {
            "model": "NSE delivery",
            "description": "STT 0.1% each side, stamp duty 0.015% on buys, exchange and SEBI charges, 18% GST, "
            f"brokerage {india_brokerage_bps:g} bps",
            "buyFeeBps": buy,
            "sellFeeBps": sell,
            "slippageBps": slippage_bps,
            "buyBps": buy + slippage_bps,
            "sellBps": sell + slippage_bps,
        }
    return {
        "model": "Flat fee",
        "description": f"{fee_bps:g} bps commission on every buy and sell",
        "buyFeeBps": fee_bps,
        "sellFeeBps": fee_bps,
        "slippageBps": slippage_bps,
        "buyBps": fee_bps + slippage_bps,
        "sellBps": fee_bps + slippage_bps,
    }
