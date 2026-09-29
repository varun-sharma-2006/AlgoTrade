"""Sign in with Google, sign-in tracking and the admin Visitors page (runs against both stores)."""

import pytest
from google.auth import exceptions as google_exceptions

from backend import google_signin
from backend.config import settings

ADMIN = "owner@gmail.com"
TOKENS = {
    "good-token-alice-000000": {
        "email": "Alice@gmail.com",
        "email_verified": True,
        "name": "Alice",
        "picture": "https://x/a.png",
    },
    "good-token-owner-000000": {"email": ADMIN, "email_verified": True, "name": "Owner"},
}


def fake_verify(credential):
    if credential == "network-down-000000000":
        raise google_exceptions.TransportError("unreachable")
    if credential not in TOKENS:
        raise ValueError("Token used too late / wrong audience")
    return TOKENS[credential]


@pytest.fixture
def google(monkeypatch):
    monkeypatch.setattr(settings, "google_client_id", "test-client.apps.googleusercontent.com")
    monkeypatch.setattr(settings, "admin_emails", [ADMIN])
    monkeypatch.setattr(google_signin, "verify_credential", fake_verify)


def sign_in(client, credential, agent="Mozilla/5.0 (iPhone) Safari"):
    return client.post("/auth/google", json={"credential": credential}, headers={"User-Agent": agent})


def bearer(response):
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_config_reports_google_and_disables_other_logins(client, google):
    config = client.get("/auth/config").json()
    assert config == {
        "googleClientId": "test-client.apps.googleusercontent.com",
        "passwordLogin": False,
        "devBypass": False,
    }
    body = {"email": "x@example.com", "password": "secret123", "name": "X"}
    assert client.post("/auth/signup", json=body).status_code == 403
    assert client.post("/auth/login", json={"email": "x@example.com", "password": "secret123"}).status_code == 403
    assert client.post("/dev/auth/bypass").status_code == 403


def test_google_sign_in_creates_account_and_session(client, google):
    response = sign_in(client, "good-token-alice-000000")
    assert response.status_code == 200
    user = response.json()["user"]
    assert user["email"] == "alice@gmail.com" and user["name"] == "Alice" and user["isAdmin"] is False
    me = client.get("/auth/me", headers=bearer(response)).json()
    assert me["email"] == "alice@gmail.com" and me["picture"] == "https://x/a.png"
    # Signing in again reuses the same account.
    assert sign_in(client, "good-token-alice-000000").json()["user"]["id"] == user["id"]


def test_invalid_and_unreachable_google(client, google):
    assert sign_in(client, "forged-token-00000000000").status_code == 401
    assert sign_in(client, "network-down-000000000").status_code == 503


def test_google_not_configured(client):
    assert client.post("/auth/google", json={"credential": "good-token-alice-000000"}).status_code == 404


def test_admin_sees_visitors_and_others_cannot(client, google):
    alice = sign_in(client, "good-token-alice-000000")
    sign_in(client, "good-token-alice-000000", agent="Mozilla/5.0 (Windows NT 10.0) Chrome/140")
    owner = sign_in(client, "good-token-owner-000000")
    assert owner.json()["user"]["isAdmin"] is True

    assert client.get("/admin/visitors", headers=bearer(alice)).status_code == 403
    assert client.get("/admin/visitors").status_code == 401

    data = client.get("/admin/visitors", headers=bearer(owner)).json()
    assert data["totalUsers"] == 2
    by_email = {u["email"]: u for u in data["users"]}
    assert by_email["alice@gmail.com"]["loginCount"] == 2 and by_email["alice@gmail.com"]["lastLoginAt"]
    assert "password_hash" not in by_email["alice@gmail.com"]
    assert [entry["email"] for entry in data["logins"]] == [ADMIN, "alice@gmail.com", "alice@gmail.com"]
    assert data["logins"][1]["userAgent"].endswith("Chrome/140")
    assert all(entry["provider"] == "google" for entry in data["logins"])
