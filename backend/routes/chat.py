from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends

from backend import advisor, copilot
from backend.config import logger
from backend.deps import get_current_user, get_db
from backend.gemini import ask_gemini
from backend.market import load_closes
from backend.schemas import ChatRequest, SimulationInput
from backend.stores import Store

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(
    payload: ChatRequest, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()

    def create_simulation(simulation: SimulationInput) -> dict[str, Any]:
        # Tools run in a worker thread, while the store is async and lives on the request's event loop.
        return asyncio.run_coroutine_threadsafe(store.add_simulation(user["id"], simulation), loop).result(timeout=15)

    toolbox = copilot.Toolbox(create_simulation)
    try:
        context = await asyncio.to_thread(advisor.market_context, payload.message, load_closes)
    except Exception as exc:
        logger.warning("Market context failed: %s", exc)
        context = None

    reply = await asyncio.to_thread(ask_gemini, payload, context, copilot.TOOL_DECLARATIONS, toolbox.call)
    if reply is not None:
        return {"reply": reply, "citations": [], "actions": toolbox.actions}

    # Gemini is rate-limited or overloaded: run the request ourselves, or answer from live market data.
    note = "\n\nEducational insight, not financial advice."
    if toolbox.actions:  # a tool ran before Gemini gave up, so report its result
        return {"reply": describe_actions(toolbox.actions) + note, "citations": [], "actions": toolbox.actions}
    action_reply = await asyncio.to_thread(copilot.local_action, payload.message, toolbox)
    if action_reply:
        return {"reply": action_reply + note, "citations": [], "actions": toolbox.actions}
    local = await asyncio.to_thread(advisor.local_answer, payload.message, load_closes)
    return {"reply": f"{local['reply']}{note}", "citations": local["citations"], "actions": []}


def describe_actions(actions: list[dict[str, Any]]) -> str:
    texts = []
    for action in actions:
        data = action["data"]
        if action["type"] == "backtest":
            texts.append(copilot.describe_backtest(data))
        elif action["type"] == "comparison":
            texts.append(copilot.describe_comparison(data["rows"][0]["symbol"], data["rows"]))
        elif action["type"] == "walkforward":
            texts.append(copilot.describe_walk_forward(data))
        else:
            texts.append(action["label"] + ".")
    return "\n\n".join(texts)
