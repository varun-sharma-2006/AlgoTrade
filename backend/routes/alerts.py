from __future__ import annotations

import asyncio
import hmac
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from backend import alerts, broker, push
from backend.config import settings
from backend.deps import get_current_user, get_db, is_admin
from backend.routes.portfolio import _chart
from backend.schemas import AlertSettings, PushSubscription, WatchAlertInput
from backend.stores import Store

router = APIRouter(tags=["alerts"])


def _channels(user: dict[str, Any]) -> dict[str, bool]:
    return {
        "email": alerts.email_configured(),
        "telegram": alerts.telegram_configured(),
        # Admins can mirror simulations as Alpaca paper orders when the server has Alpaca keys.
        "broker": broker.configured() and is_admin(user),
        "push": push.configured(),
    }


@router.get("/alerts/settings")
async def get_alert_settings(
    user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    return {"settings": await store.get_alert_settings(user["id"]), "available": _channels(user)}


@router.put("/alerts/settings")
async def put_alert_settings(
    payload: AlertSettings, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    saved = await store.set_alert_settings(user["id"], payload.model_dump())
    return {"settings": saved, "available": _channels(user)}


@router.get("/alerts/watch")
async def list_watch_alerts(
    user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> list[dict[str, Any]]:
    """The user's price / indicator alerts, each with whether it holds at the latest close."""
    watches = await store.list_watch_alerts(user["id"])
    symbols = sorted({w["symbol"] for w in watches})
    if not symbols:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
        charts = dict(zip(symbols, pool.map(_chart, symbols), strict=True))
    out = []
    for watch in watches:
        status_ = alerts.watch_status(watch, charts.get(watch["symbol"]))
        out.append(
            {"id": watch["id"], "symbol": watch["symbol"], "condition": watch["condition"], "note": watch.get("note")}
            | ({"status": status_} if status_ else {"status": None})
        )
    return out


@router.post("/alerts/watch")
async def add_watch_alert(
    payload: WatchAlertInput, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    try:
        return await store.add_watch_alert(user["id"], payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/alerts/watch/{alert_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_watch_alert(
    alert_id: str, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> Response:
    try:
        await store.delete_watch_alert(user["id"], alert_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Alert not found") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/alerts/test")
async def test_alert(
    user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    prefs = await store.get_alert_settings(user["id"])
    subscriptions = await store.list_push_subscriptions(user["id"])
    if not (prefs.get("email") or prefs.get("telegramChatId") or subscriptions):
        raise HTTPException(
            status_code=422, detail="Turn on email or push notifications, or add a Telegram chat id first"
        )
    sample = {
        "simulationId": "test",
        "symbol": "TEST",
        "strategy": "Test alert",
        "signal": "buy",
        "summary": "This is a test message from Algo Trade. Real alerts look like this.",
        "date": "today",
        "price": 100.0,
        "currency": "USD",
    }
    sent = await asyncio.to_thread(alerts.deliver, user, prefs, [sample], subscriptions)
    for endpoint in sent.pop("gone"):
        await store.remove_push_subscription(endpoint)
    if not any(sent.values()):
        raise HTTPException(status_code=503, detail="No alert channel is configured on the server, or sending failed")
    return {"sent": sent}


@router.get("/push/key")
async def push_key() -> dict[str, Any]:
    """The server's public VAPID key, which browsers need to subscribe to push notifications."""
    return {"publicKey": settings.vapid_public_key or None, "available": push.configured()}


@router.post("/push/subscribe")
async def push_subscribe(
    payload: PushSubscription, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    if not push.configured():
        raise HTTPException(status_code=503, detail="Push notifications aren't configured on this server")
    await store.add_push_subscription(user["id"], payload.model_dump(exclude_none=True))
    return {"subscribed": True, "devices": len(await store.list_push_subscriptions(user["id"]))}


@router.post("/push/unsubscribe")
async def push_unsubscribe(
    payload: PushSubscription, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    await store.remove_push_subscription(payload.endpoint, user["id"])
    return {"subscribed": False}


@router.get("/alerts/signals")
async def todays_signals(
    user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> list[dict[str, Any]]:
    """What each of the user's active simulations would do at the latest close (no messages are sent)."""
    sims = [s for s in await store.list_simulations(user["id"]) if str(s.get("status", "")).lower() == "active"]
    symbols = sorted({s["symbol"] for s in sims})
    if not symbols:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
        charts = dict(zip(symbols, pool.map(_chart, symbols), strict=True))
    out = [alerts.simulation_signal(sim, charts.get(sim["symbol"])) for sim in sims]
    return sorted((s for s in out if s), key=lambda s: (not s["actionable"], s["symbol"]))


async def _run_cron(authorization: str, store: Store) -> dict[str, Any]:
    if not settings.cron_secret:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="CRON_SECRET is not set")
    if not hmac.compare_digest(authorization, f"Bearer {settings.cron_secret}"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad cron secret")
    return await alerts.run_daily(store, _chart)


# Vercel Cron sends GET with "Authorization: Bearer $CRON_SECRET"; other schedulers can POST.
@router.get("/cron/daily", include_in_schema=False)
async def cron_daily_get(authorization: str = Header(""), store: Store = Depends(get_db)) -> dict[str, Any]:
    return await _run_cron(authorization, store)


@router.post("/cron/daily", include_in_schema=False)
async def cron_daily_post(authorization: str = Header(""), store: Store = Depends(get_db)) -> dict[str, Any]:
    return await _run_cron(authorization, store)
