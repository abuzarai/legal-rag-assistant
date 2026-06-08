FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on

WORKDIR /app

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv + deps using lockfile
RUN pip install --upgrade pip uv \
    && uv pip install --system .

# Copy source
COPY src ./src
COPY README.md ./README.md

ENV PORT=8080
CMD uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT}
