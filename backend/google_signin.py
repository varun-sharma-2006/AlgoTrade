"""Verifies "Sign in with Google" ID tokens from Google Identity Services."""

from __future__ import annotations

from typing import Any

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from backend.config import settings

_transport = google_requests.Request()


def verify_credential(credential: str) -> dict[str, Any]:
    """Claims of a genuine Google ID token issued to this app for a verified email; raises ValueError otherwise.

    verify_oauth2_token checks Google's signature, the expiry, the issuer (accounts.google.com) and that the
    token was issued for our client ID, so a token minted for another site can't be replayed here.
    """
    claims = id_token.verify_oauth2_token(credential, _transport, settings.google_client_id, clock_skew_in_seconds=10)
    if not claims.get("email") or not claims.get("email_verified"):
        raise ValueError("This Google account's email address is not verified")
    return claims
