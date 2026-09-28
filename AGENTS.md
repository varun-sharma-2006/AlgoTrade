# Repository guidelines

- Keep the README in sync with any new scripts or environment variables.
- Run `npm run check` after editing TypeScript sources.
- Run `python -m compileall backend` after editing the FastAPI backend.
- When you point the backend at a new MongoDB instance, optionally run `python test.py` to confirm connectivity.

- The chatbot uses Gemini (`GOOGLE_API_KEY`, `GEMINI_MODELS`) and falls back to a local reply when no model responds.
