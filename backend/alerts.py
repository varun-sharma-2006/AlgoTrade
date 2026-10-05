"""Daily trade alerts for paper-trading simulations.

After the market closes, a scheduler (Vercel Cron or GitHub Actions) calls /cron/daily. For every active
simulation the job re-runs its strategy on the latest prices; when the strategy says to trade (buy, sell,
short or cover), the owner gets one message per day per simulation, by email and/or Telegram, depending on
their alert settings. Each alert is recorded on the simulation, so a repeated run never sends it twice.
"""

from __future__ import annotations

import asyncio
import smtplib
from collections import defaultdict
from collections.abc import Callable
from email.message import EmailMessage
from typing import Any

import requests

from backend import strategies
from backend.config import logger, settings

ACTIONABLE = {"buy", "sell", "short", "cover"}
ChartLoader = Callable[[str], dict[str, Any] | None]


def simulation_signal(sim: dict[str, Any], chart: dict[str, Any] | None) -> dict[str, Any] | None:
    """Today's action for one simulation's strategy, or None without price data."""
    if not chart or len(chart.get("points") or []) < 3:
        return None
    points = chart["points"]
    closes = [p["close"] for p in points]
    bars = {key: [p.get(key) for p in points] for key in ("open", "high", "low", "volume")}
    strategy_id = sim.get("strategyId") or "buy-hold"
    params = strategies.DEFAULT_PARAMS.get(strategy_id, {}) | (sim.get("parameters") or {})
    rules = sim.get("rules")
    if (
        strategies.validate(strategy_id, params, rules)
        or len(closes) < strategies.warmup(strategy_id, params, rules) + 2
    ):
        return None
    outlook = strategies.current_signal(strategy_id, closes, params, rules, bars=bars)
    return {
        "simulationId": sim["id"],
        "symbol": sim["symbol"],
        "strategy": sim.get("strategy") or strategy_id,
        "signal": outlook["signal"],
        "summary": outlook["summary"],
        "date": points[-1]["timestamp"][:10],
        "price": closes[-1],
        "currency": chart.get("currency") or "USD",
        "actionable": outlook["signal"] in ACTIONABLE,
    }


def compose(name: str, items: list[dict[str, Any]]) -> tuple[str, str]:
    """Subject and plain-text body for one user's alerts."""
    subject = (
        f"Algo Trade: {items[0]['signal'].upper()} {items[0]['symbol']}"
        if len(items) == 1
        else f"Algo Trade: {len(items)} trade signals"
    )
    lines = [f"Hi {name.split(' ')[0] if name else 'there'},", "", "Your paper-trading strategies signalled today:", ""]
    for item in items:
        lines.append(
            f"- {item['signal'].upper()} {item['symbol']} ({item['strategy']}) at {item['price']:.2f} {item['currency']}"
            f" on {item['date']}"
        )
        lines.append(f"  {item['summary']}")
    lines += [
        "",
        "Orders decided at the close are filled at the next open in the simulator.",
        "This is a paper-trading simulation, not investment advice.",
    ]
    return subject, "\n".join(lines)


def email_configured() -> bool:
    return bool(settings.smtp_host and (settings.smtp_from or settings.smtp_user))


def telegram_configured() -> bool:
    return bool(settings.telegram_bot_token)


def send_email(to: str, subject: str, body: str) -> bool:
    if not email_configured():
        return False
    message = EmailMessage()
    message["From"] = settings.smtp_from or settings.smtp_user
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
            server.starttls()
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)
        return True
    except (OSError, smtplib.SMTPException) as exc:
        logger.warning("Alert email to %s failed: %s", to, exc)
        return False


def send_telegram(chat_id: str, text: str) -> bool:
    if not telegram_configured() or not chat_id:
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=10,
        )
        return response.ok
    except requests.RequestException as exc:
        logger.warning("Telegram alert failed: %s", exc)
        return False


def deliver(user: dict[str, Any], prefs: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, bool]:
    subject, body = compose(user.get("name", ""), items)
    sent = {"email": False, "telegram": False}
    if prefs.get("email") and user.get("email"):
        sent["email"] = send_email(user["email"], subject, body)
    if prefs.get("telegramChatId"):
        sent["telegram"] = send_telegram(prefs["telegramChatId"], f"{subject}\n\n{body}")
    return sent


async def run_daily(store: Any, load_chart: ChartLoader) -> dict[str, Any]:
    """Check every active simulation and send each owner their new trade signals."""
    sims = await store.list_active_simulations()
    symbols = sorted({sim["symbol"] for sim in sims})
    charts = dict(zip(symbols, await asyncio.gather(*(asyncio.to_thread(load_chart, s) for s in symbols)), strict=True))
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checked = 0
    for sim in sims:
        signal = simulation_signal(sim, charts.get(sim["symbol"]))
        if not signal:
            continue
        checked += 1
        key = f"{signal['date']}:{signal['signal']}"
        if signal["actionable"] and sim.get("lastAlert") != key:
            by_user[str(sim["userId"])].append(signal | {"key": key})
    delivered = 0
    for user_id, items in by_user.items():
        user = await store.get_user(user_id)
        prefs = await store.get_alert_settings(user_id)
        if user and (prefs.get("email") or prefs.get("telegramChatId")):
            sent = await asyncio.to_thread(deliver, user, prefs, items)
            delivered += int(any(sent.values()))
        # Marked even without a channel, so turning alerts on later doesn't replay old signals.
        for item in items:
            await store.mark_alerted(item["simulationId"], item["key"])
    return {
        "simulations": len(sims),
        "checked": checked,
        "signals": sum(len(items) for items in by_user.values()),
        "usersNotified": delivered,
    }
