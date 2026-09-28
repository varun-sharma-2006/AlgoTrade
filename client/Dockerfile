# Build from the repository root: docker build -f client/Dockerfile .
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json vite.config.ts ./
COPY client client
# Vite inlines VITE_* variables at build time.
ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_ENABLE_LOGIN_BYPASS=false
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL VITE_ENABLE_LOGIN_BYPASS=$VITE_ENABLE_LOGIN_BYPASS
RUN npm run build

FROM nginx:1.27-alpine
COPY client/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
