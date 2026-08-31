FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=prod

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv + locked deps (the lockfile is the single source of truth)
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/
RUN uv sync --locked --no-dev --no-install-project

# Copy source
COPY src ./src
COPY README.md ./README.md

ENV PATH="/app/.venv/bin:${PATH}"
ENV PORT=8080
CMD uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT}