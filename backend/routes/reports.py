"""Shareable backtest reports.

A report is created from a strategy's settings, not from numbers sent by the browser: the server re-runs the
backtest (and the robustness checks), so a shared link always shows a result the app really produced. Anyone
with the link can view it at /r/<id>; /share/<id> serves the app's page with link-preview tags (title,
description, image) so the link unfurls nicely on LinkedIn, X or WhatsApp.
"""

from __future__ import annotations

import asyncio
import html
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse

from backend import review
from backend.config import logger, settings
from backend.deps import get_current_user, get_db, heavy_limit
from backend.routes.analytics import _public, run_backtest, run_robustness, strategy_params, trading_options
from backend.routes.research import STRATEGY_LABELS
from backend.schemas import ReportPayload
from backend.stores import Store

router = APIRouter(tags=["reports"])

MAX_REPORT_POINTS = 400
_index_cache: dict[str, tuple[float, str]] = {}
TRUSTED_SUFFIXES = (".vercel.app", ".hf.space", ".onrender.com")


def _thin(points: list[dict[str, Any]], limit: int = MAX_REPORT_POINTS) -> list[dict[str, Any]]:
    if len(points) <= limit:
        return points
    step = -(-len(points) // limit)
    return points[::step] + ([points[-1]] if (len(points) - 1) % step else [])


def build_content(payload: ReportPayload) -> dict[str, Any]:
    params = strategy_params(payload)
    rules = payload.rules.model_dump() if payload.rules else None
    opts = trading_options(payload)
    report = run_backtest(payload.symbol, payload.strategyId, params, rules, None, opts)
    robust = run_robustness(payload, report) if payload.includeRobustness else None
    items = review.findings(report, robustness=robust)
    backtest = _public(report)
    backtest["sample"] = _thin(backtest["sample"])
    backtest["trades"] = backtest["trades"][-20:]
    if robust and robust.get("monteCarlo"):
        robust["monteCarlo"]["fan"] = _thin(robust["monteCarlo"]["fan"], 80)
    return {
        "backtest": backtest,
        "robustness": robust,
        "review": {"findings": items, "verdict": review.verdict(items)},
        "settings": payload.model_dump(exclude={"title", "includeRobustness"}),
    }


@router.post("/reports", dependencies=[Depends(heavy_limit)])
async def create_report(
    payload: ReportPayload, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> dict[str, Any]:
    content = await asyncio.to_thread(build_content, payload)
    label = STRATEGY_LABELS.get(payload.strategyId, payload.strategyId)
    title = payload.title.strip() or f"{label} on {payload.symbol.upper()}"
    created = await store.create_report(user, title, content)
    return created | {"path": f"/r/{created['id']}"}


@router.get("/reports")
async def my_reports(user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)) -> list[dict]:
    return await store.list_reports(user["id"])


@router.get("/reports/{report_id}")
async def get_report(report_id: str, store: Store = Depends(get_db)) -> dict[str, Any]:
    """Public: anyone with the link can read a report (the owner's email is never included)."""
    record = await store.get_report(report_id)
    if not record:
        raise HTTPException(status_code=404, detail="Report not found")
    return {key: record[key] for key in ("id", "title", "author", "createdAt", "content")}


@router.delete("/reports/{report_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_report(
    report_id: str, user: dict[str, Any] = Depends(get_current_user), store: Store = Depends(get_db)
) -> Response:
    try:
        await store.delete_report(user["id"], report_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Report not found") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _origin(request: Request) -> str | None:
    """The site's public origin: PUBLIC_URL, or the request's host when it is a known hosting domain."""
    if settings.public_url:
        return settings.public_url.rstrip("/")
    host = urlparse(str(request.url)).hostname or ""
    if host in {"localhost", "127.0.0.1"} or host.endswith(TRUSTED_SUFFIXES):
        scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
        port = f":{request.url.port}" if request.url.port and host in {"localhost", "127.0.0.1"} else ""
        return f"{scheme}://{host}{port}"
    return None


def _index_html(origin: str | None) -> str | None:
    if settings.static_dir and Path(settings.static_dir, "index.html").is_file():
        return Path(settings.static_dir, "index.html").read_text(encoding="utf-8")
    if not origin:
        return None
    cached = _index_cache.get(origin)
    if cached and time.time() - cached[0] < 600:
        return cached[1]
    try:
        response = requests.get(f"{origin}/index.html", timeout=8)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Couldn't load index.html for a share page: %s", exc)
        return None
    _index_cache[origin] = (time.time(), response.text)
    return response.text


def describe(record: dict[str, Any]) -> str:
    backtest = record["content"]["backtest"]
    m = backtest["metrics"]
    text = (
        f"{backtest['symbol']}: {m['totalReturn'] * 100:+.1f}% vs buy & hold {m['buyHoldReturn'] * 100:+.1f}%, "
        f"Sharpe {m['sharpe']:.2f}, max drawdown {m['maxDrawdown'] * 100:.1f}%"
    )
    verdict = (record["content"].get("review") or {}).get("verdict")
    if verdict:
        text += f". Robustness verdict: {verdict['label']}"
    return text + ". Backtested with fees, slippage and next-day fills on Algo Trade Simulator."


@router.get("/share/{report_id}", include_in_schema=False)
async def share_page(report_id: str, request: Request, store: Store = Depends(get_db)) -> HTMLResponse:
    """The app's page for /r/<id>, with Open Graph tags so the shared link shows a preview."""
    record = await store.get_report(report_id)
    origin = _origin(request)
    page = await asyncio.to_thread(_index_html, origin)
    if page is None:
        target = f"{origin or ''}/r/{report_id}"
        return HTMLResponse(f'<meta http-equiv="refresh" content="0;url={html.escape(target)}">', status_code=200)
    if record:
        title = html.escape(f"{record['title']} · Algo Trade Simulator")
        description = html.escape(describe(record))
        url = html.escape(f"{origin or ''}/r/{report_id}")
        image = html.escape(f"{origin or ''}/og.png")
        tags = (
            f'<meta property="og:type" content="article"><meta property="og:title" content="{title}">'
            f'<meta property="og:description" content="{description}"><meta property="og:url" content="{url}">'
            f'<meta property="og:image" content="{image}"><meta name="twitter:card" content="summary_large_image">'
            f'<meta name="description" content="{description}">'
        )
        page = page.replace("<head>", f"<head>{tags}", 1)
        page = re.sub(r"<title>.*?</title>", lambda _: f"<title>{title}</title>", page, count=1, flags=re.S)
    return HTMLResponse(page)
