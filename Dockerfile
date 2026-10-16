FROM node:22-alpine AS frontend-build

WORKDIR /build/pactio-frontend
COPY pactio-frontend/pactio-frontend/package.json ./
RUN npm install
COPY pactio-frontend/pactio-frontend/ ./
RUN npm run build

FROM python:3.11-slim AS application

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONTRACT_LENS_ARTIFACTS=/app/artifacts

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
COPY api.py ./
COPY artifacts/bilstm_attention.pt ./artifacts/bilstm_attention.pt
COPY artifacts/tokenizer.json ./artifacts/tokenizer.json
RUN python -m pip install --no-cache-dir .

COPY --from=frontend-build /build/pactio-frontend/dist ./pactio-frontend/pactio-frontend/dist

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}"]
