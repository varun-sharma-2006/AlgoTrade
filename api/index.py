"""Vercel serverless entry point: the FastAPI backend, served under /api.

vercel.json builds the React app as static files and rewrites /api/* to this function, so the frontend
and API share one origin. Settings default to the demo configuration; override them in the Vercel
project's Environment Variables (e.g. MONGO_URL for persistent data, GOOGLE_API_KEY for Gemini).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("ENABLE_DEV_ENDPOINTS", "true")
os.environ.setdefault("USE_IN_MEMORY_DB", "false" if os.getenv("MONGO_URL") else "true")

from fastapi import FastAPI  # noqa: E402

from backend.main import app as backend_app  # noqa: E402
from backend.routes.reports import share_page  # noqa: E402

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
# vercel.json rewrites /r/<id> here: the app's page with link-preview tags for a shared report.
app.add_api_route("/r/{report_id}", share_page, methods=["GET"], include_in_schema=False)
app.mount("/api", backend_app)
