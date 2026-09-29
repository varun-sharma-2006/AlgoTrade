# Single-container build: the API also serves the built frontend, so the whole app runs at one URL.
# Used by the Hugging Face Space (free, no card). For separate services use docker-compose.yml.
FROM node:20-alpine AS web
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json vite.config.ts ./
COPY client client
# Empty base URL = same origin as the API.
ENV VITE_API_BASE_URL="" VITE_ENABLE_LOGIN_BYPASS=true
RUN npm run build

FROM python:3.12-slim
# Hugging Face runs containers as uid 1000; give it a writable home for yfinance's cache.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY --from=web /app/dist dist
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 PORT=7860 STATIC_DIR=/app/dist \
    USE_IN_MEMORY_DB=true ENABLE_DEV_ENDPOINTS=true
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
