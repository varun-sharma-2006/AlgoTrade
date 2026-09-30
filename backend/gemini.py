"""Gemini chat client with multi-model fallback, retries and function calling."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

import requests

from backend.config import logger, settings
from backend.schemas import ChatRequest

SYSTEM_PROMPT = (
    "You are the trading copilot inside an algorithmic trading simulator. Answer every question directly and "
    "specifically, including which stocks look attractive and why: name tickers, cite the live numbers you are "
    "given (trend vs SMAs, momentum, RSI, volatility), and give a concrete plan with entry, stop-loss, target "
    "and position sizing when relevant. Use short paragraphs or bullet lists in plain text (no markdown tables). "
    "For questions outside trading, still answer helpfully. "
    "You have tools that run the app's real backtester: when the user asks how a strategy would have done, to "
    "backtest, compare strategies, run a walk-forward test or create a simulation, call the tool instead of "
    "guessing, then explain the result with its exact numbers (return vs buy & hold and the S&P 500, Sharpe, "
    "drawdown, trades) and say plainly when the strategy lost to buy & hold. Only create a simulation when the "
    "user explicitly asks for one. End with a one-line reminder that this is educational, not financial advice."
)
MAX_TOOL_ROUNDS = 3
TOOL_TIME_BONUS = 15.0  # seconds added to the budget after a tool runs, so Gemini can explain its result
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_cooldown_until: dict[str, float] = {}


def ask_gemini(
    payload: ChatRequest,
    context: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    run_tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> str | None:
    """Try the configured models for a few rounds; return None if none of them answer in time.

    With `tools` (Gemini function declarations) and `run_tool(name, args)`, the model may call backend
    functions; their results are sent back until it answers in text.
    """
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
    body: dict[str, Any] = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}}
    if tools and run_tool:
        body["tools"] = [{"functionDeclarations": tools}]

    deadline = time.monotonic() + settings.gemini_budget_seconds
    attempt = 0
    while time.monotonic() < deadline:
        available = [m for m in settings.gemini_models if _cooldown_until.get(m, 0) <= time.time()]
        if not available:
            return None
        for model in available:
            if deadline - time.monotonic() < 3:
                return None
            # Each model starts from the original conversation; the caller memoises tool results.
            conversation = copy.deepcopy(contents)
            try:
                for _ in range(MAX_TOOL_ROUNDS + 1):
                    content = _generate(model, body | {"contents": conversation}, deadline)
                    parts = content.get("parts") or []
                    calls = [part["functionCall"] for part in parts if "functionCall" in part]
                    if not calls or not run_tool:
                        text = "".join(part.get("text", "") for part in parts).strip()
                        if text:
                            return text
                        break
                    # Send the model's turn back unchanged (it carries thought signatures), then the results.
                    conversation.append(content)
                    results = [
                        {"functionResponse": {"name": c["name"], "response": run_tool(c["name"], c.get("args") or {})}}
                        for c in calls
                    ]
                    conversation.append({"role": "user", "parts": results})
                    deadline = max(deadline, time.monotonic() + TOOL_TIME_BONUS)
            except Exception as exc:  # overload, network, or malformed response
                logger.warning("Gemini model %s failed: %s", model, str(exc)[:160])
        attempt += 1
        time.sleep(min(1.5 * attempt, max(deadline - time.monotonic(), 0)))
    return None


def _generate(model: str, body: dict[str, Any], deadline: float) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining < 2:
        raise TimeoutError("out of time")
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
    content = response.json()["candidates"][0]["content"]
    content.setdefault("role", "model")
    return content
