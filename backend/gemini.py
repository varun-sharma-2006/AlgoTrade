"""Gemini chat client with multi-model fallback and retries."""

from __future__ import annotations

import time

import requests

from backend.config import logger, settings
from backend.schemas import ChatRequest

SYSTEM_PROMPT = (
    "You are the trading copilot inside an algorithmic trading simulator. Answer every question directly and "
    "specifically, including which stocks look attractive and why: name tickers, cite the live numbers you are "
    "given (trend vs SMAs, momentum, RSI, volatility), and give a concrete plan with entry, stop-loss, target "
    "and position sizing when relevant. Use short paragraphs or bullet lists in plain text (no markdown tables). "
    "For questions outside trading, still answer helpfully. End with a one-line reminder that this is educational, "
    "not financial advice."
)
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_cooldown_until: dict[str, float] = {}


def ask_gemini(payload: ChatRequest, context: str | None = None) -> str | None:
    """Try the configured models for a few rounds; return None if none of them answer in time."""
    if not settings.google_api_key:
        return None
    contents = [
        {"role": "model" if item.role in {"assistant", "model"} else "user", "parts": [{"text": item.content}]}
        for item in payload.history[-10:]
        if item.content.strip()
    ]
    while contents and contents[0]["role"] == "model":  # Gemini expects the conversation to start with the user
        contents.pop(0)
    question = payload.message if not context else f"{payload.message}\n\n[{context}]"
    contents.append({"role": "user", "parts": [{"text": question}]})
    body = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents}

    deadline = time.monotonic() + settings.gemini_budget_seconds
    attempt = 0
    while time.monotonic() < deadline:
        available = [m for m in settings.gemini_models if _cooldown_until.get(m, 0) <= time.time()]
        if not available:
            return None
        for model in available:
            remaining = deadline - time.monotonic()
            if remaining < 3:
                return None
            try:
                response = requests.post(
                    API_URL.format(model=model),
                    json=body,
                    headers={"x-goog-api-key": settings.google_api_key},
                    timeout=min(20, remaining),
                )
                if response.status_code in (404, 429):
                    # Retired model or exhausted quota: skip it for a while instead of retrying every message.
                    _cooldown_until[model] = time.time() + (3600 if response.status_code == 404 else 120)
                response.raise_for_status()
                parts = response.json()["candidates"][0]["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts).strip()
                if text:
                    return text
            except Exception as exc:  # overload, network, or malformed response
                logger.warning("Gemini model %s failed: %s", model, str(exc)[:160])
        attempt += 1
        time.sleep(min(1.5 * attempt, max(deadline - time.monotonic(), 0)))
    return None
