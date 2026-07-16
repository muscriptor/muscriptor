# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS web-builder
WORKDIR /web

# pnpm is the project's package manager (pinned via package.json#packageManager).
RUN corepack enable

COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY web/ ./
# The GA4 measurement ID is inlined into the JS bundle by Vite; left unset
# (the default), the frontend is built with analytics disabled entirely.
ARG VITE_GA_MEASUREMENT_ID=
ENV VITE_GA_MEASUREMENT_ID=$VITE_GA_MEASUREMENT_ID
# vite outDir is ../muscriptor/web_dist, i.e. /muscriptor/web_dist in this stage.
RUN pnpm run build


FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS runtime


ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1

WORKDIR /app

# fluidsynth is required at runtime for MIDI auralization (the /auralize endpoint).
RUN apt-get update \
    && apt-get install -y --no-install-recommends fluidsynth \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --no-install-project

# Pre-fetch the soundfonts (253 MB: .sf2 for /auralize, .sf3 for the web UI)
# into the HF hub cache so a fresh container doesn't download them on first
# use. Runs off just the two files it needs, before the full source COPY,
# so day-to-day code changes don't re-trigger the download.
COPY muscriptor/soundfonts.py muscriptor/utils/download.py /tmp/prewarm/
RUN /app/.venv/bin/python -c "\
import sys; sys.path.insert(0, '/tmp/prewarm'); \
import soundfonts, download; \
download.download_if_necessary(soundfonts.SF2_URL); \
download.download_if_necessary(soundfonts.SF3_URL)" \
    && rm -rf /tmp/prewarm

COPY pyproject.toml uv.lock README.md ./
COPY muscriptor/ ./muscriptor/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

COPY --from=web-builder /muscriptor/web_dist ./muscriptor/web_dist


EXPOSE 8000
ENTRYPOINT ["uv", "run", "muscriptor", "serve", "--host", "0.0.0.0"]
