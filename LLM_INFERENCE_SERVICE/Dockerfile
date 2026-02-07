# syntax=docker/dockerfile:1.7

# ---------- Builder ----------
FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /install

# System deps required for torch/tokenizers builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.cargo/bin/uv /usr/local/bin/uv

COPY requirements.txt .

# Cache wheels between builds (THIS is where BuildKit matters)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# ---------- Final Image ----------
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy only installed packages, not compilers
COPY --from=builder /usr/local /usr/local

# Copy application
COPY app ./app

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app.flask_app:app"]
