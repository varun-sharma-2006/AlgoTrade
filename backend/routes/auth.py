from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from google.auth import exceptions as google_exceptions

from backend import google_signin
from backend.config import logger, settings
from backend.deps import get_current_user, get_db, is_admin
from backend.schemas import DevAuthBypassRequest, GoogleLoginRequest, LoginRequest, SignupRequest
from backend.stores import Store

router = APIRouter(tags=["auth"])


def _with_role(user: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in user.items() if key != "token"} | {"isAdmin": is_admin(user)}


async def _signed_in(
    store: Store, user: dict[str, Any], request: Request, provider: str, picture: str | None = None
) -> dict[str, Any]:
    await store.record_login(user, picture, provider, request.headers.get("user-agent"))
    session = await store.create_session(user["id"])
    return {"token": session["token"], "user": _with_role(user | ({"picture": picture} if picture else {}))}


def _password_auth_allowed() -> None:
    if settings.google_client_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This app uses Sign in with Google")


@router.get("/auth/config")
async def auth_config() -> dict[str, Any]:
    """Tells the login page which sign-in methods are available."""
    google = bool(settings.google_client_id)
    return {
        "googleClientId": settings.google_client_id or None,
        "passwordLogin": not google,
        "devBypass": settings.enable_dev_endpoints and not google,
    }


@router.post("/auth/google")
async def google_login(payload: GoogleLoginRequest, request: Request, store: Store = Depends(get_db)) -> dict[str, Any]:
    if not settings.google_client_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Google sign-in is not configured")
    try:
        claims = await asyncio.to_thread(google_signin.verify_credential, payload.credential)
    except google_exceptions.TransportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Couldn't reach Google, try again"
        ) from exc
    except (ValueError, google_exceptions.GoogleAuthError) as exc:
        logger.warning("Rejected Google sign-in: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Google sign-in could not be verified"
        ) from exc
    email = claims["email"].lower()
    name = claims.get("name") or email.split("@")[0]
    user = await store.ensure_user(email, name) | {"name": name}
    return await _signed_in(store, user, request, "google", claims.get("picture"))


@router.post("/auth/signup")
async def signup(payload: SignupRequest, request: Request, store: Store = Depends(get_db)) -> dict[str, Any]:
    _password_auth_allowed()
    try:
        user = await store.create_user(payload.email, payload.name, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return await _signed_in(store, user, request, "password")


@router.post("/auth/login")
async def login(payload: LoginRequest, request: Request, store: Store = Depends(get_db)) -> dict[str, Any]:
    _password_auth_allowed()
    user = await store.get_user_by_credentials(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return await _signed_in(store, user, request, "password")


@router.post("/dev/auth/bypass")
async def dev_auth_bypass(
    request: Request, payload: DevAuthBypassRequest | None = None, store: Store = Depends(get_db)
) -> dict[str, Any]:
    if not settings.enable_dev_endpoints or settings.google_client_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dev endpoints are disabled")
    email = (payload.email if payload and payload.email else "user@example.com").lower()
    name = payload.name if payload and payload.name else "User"
    return await _signed_in(store, await store.ensure_user(email, name), request, "dev")


@router.get("/auth/me")
async def me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return _with_role(user)
