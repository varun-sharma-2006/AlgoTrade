# Repository guidelines

- Keep the README in sync with any new scripts or environment variables, and add new variables to the `.env.example` files.
- Backend: run `ruff check backend`, `ruff format backend` and `pytest backend` after editing (dev tools are in `backend/requirements-dev.txt`).
- Frontend: run `npm run check` and `npm test` after editing TypeScript sources.
- Routes use the shared store interface in `backend/stores.py`; add new persistence methods to both `MongoStore` and `InMemoryStore`, and cover them in `backend/test_api.py` (which runs against both).
- When you point the backend at a new MongoDB instance, run `python -m backend.scripts.check_setup` to confirm connectivity.
- The chatbot uses Gemini (`GOOGLE_API_KEY`, `GEMINI_MODELS`) and falls back to `backend/advisor.py` when no model responds.
