from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends

from backend import advisor
from backend.config import logger
from backend.deps import get_current_user
from backend.gemini import ask_gemini
from backend.market import load_closes
from backend.schemas import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat", dependencies=[Depends(get_current_user)])
async def chat(payload: ChatRequest) -> dict[str, Any]:
    try:
        context = await asyncio.to_thread(advisor.market_context, payload.message, load_closes)
    except Exception as exc:
        logger.warning("Market context failed: %s", exc)
        context = None

    reply = await asyncio.to_thread(ask_gemini, payload, context)
    if reply is not None:
        return {"reply": reply, "citations": [], "actions": []}

    # Gemini is rate-limited or overloaded: answer from live market data instead.
    local = await asyncio.to_thread(advisor.local_answer, payload.message, load_closes)
    return {
        "reply": f"{local['reply']}\n\nEducational insight, not financial advice.",
        "citations": local["citations"],
        "actions": [],
    }
