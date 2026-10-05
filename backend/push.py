"""Web Push notifications (free, standard browser push with VAPID keys; no third-party service).

Each signed-in device that turns notifications on stores a push subscription; the daily alert job and the
test button send to every subscription the user has. Subscriptions the browser has revoked (HTTP 404 / 410)
are removed.
"""

from __future__ import annotations

import json
from typing import Any

from backend.config import logger, settings


def configured() -> bool:
    return bool(settings.vapid_public_key and settings.vapid_private_key)


def send(subscription: dict[str, Any], title: str, body: str, url: str = "/") -> str:
    """ "sent", "gone" (the subscription is dead and should be deleted) or "failed"."""
    if not configured():
        return "failed"
    from pywebpush import WebPushException, webpush

    try:
        webpush(
            subscription_info=subscription,
            data=json.dumps({"title": title, "body": body[:400], "url": url}),
            vapid_private_key=settings.vapid_private_key,
            vapid_claims={"sub": settings.vapid_subject},
            ttl=12 * 3600,
        )
        return "sent"
    except WebPushException as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (404, 410):
            return "gone"
        logger.warning("Web push failed (%s): %s", status, str(exc)[:200])
        return "failed"
    except Exception as exc:  # malformed subscription keys and the like
        logger.warning("Web push failed: %s", str(exc)[:200])
        return "failed"
