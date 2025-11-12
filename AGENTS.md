# Repository guidelines

- Keep the README in sync with any new scripts or environment variables.
- Run `npm run check` after editing TypeScript sources.
- Run `python -m compileall backend` after editing the FastAPI backend.
- When you point the backend at a new MongoDB instance, optionally run `python test.py` to confirm connectivity.

- Confirm the local Ollama service is running (or mock responses) before testing chatbot interactions.
