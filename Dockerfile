FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-default-groups

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups

FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

ENV PATH="/app/.venv/bin:$PATH"
# Settings.HOST defaults to 127.0.0.1 (correct for local `uv run` dev -- binding 0.0.0.0
# there would expose the dev API to the whole LAN by default). Inside a container,
# 127.0.0.1 is only reachable from within the container's own network namespace --
# `docker run -p` port-forwarding can never reach it. Override for the container only,
# not the code-level default.
ENV HOST=0.0.0.0

WORKDIR /app

CMD ["python", "-m", "src"]
