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
# ffmpeg is the fallback decoder for audio containers libsndfile can't read
# (notably AAC/M4A uploads) — see muscriptor/utils/audio.py:_read_via_ffmpeg.
# The rest are what the MuseScore AppImage below needs from the system: its own
# AppRun substitutes bundled fallbacks for OpenGL/jack/nss/pipewire, but these
# five it expects to find installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fluidsynth \
        ffmpeg \
        curl \
        libasound2 \
        libgl1 \
        libegl1 \
        libfontconfig1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


ARG MUSESCORE_URL=https://github.com/musescore/MuseScore/releases/download/v4.7.4/MuseScore-Studio-4.7.4.260706075-x86_64.AppImage
RUN curl -fsSL -o /tmp/musescore.AppImage "$MUSESCORE_URL" \
    && chmod +x /tmp/musescore.AppImage \
    && cd /opt && /tmp/musescore.AppImage --appimage-extract > /dev/null \
    && mv squashfs-root musescore \
    && rm /tmp/musescore.AppImage \
    && QT_QPA_PLATFORM=offscreen MU_QT_QPA_PLATFORM=offscreen \
       /opt/musescore/AppRun --version

ENV MUSCRIPTOR_MUSESCORE=/opt/musescore/AppRun

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
