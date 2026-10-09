# The docs assets are plain JS/CSS whatever the target, so npm runs natively: under QEMU (arm64) node dies of SIGILL.
FROM --platform=$BUILDPLATFORM node:22-alpine AS docs-assets
ENV SCARF_ANALYTICS=false
RUN npm install --omit=dev --prefix /assets swagger-ui-dist@5.18.2 redoc@2.4.0 \
    && mkdir -p /static \
    && cp /assets/node_modules/swagger-ui-dist/swagger-ui-bundle.js \
          /assets/node_modules/swagger-ui-dist/swagger-ui.css \
          /assets/node_modules/swagger-ui-dist/favicon-32x32.png /static/ \
    && cp /assets/node_modules/redoc/bundles/redoc.standalone.js /static/

FROM python:3.13-slim AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-default-groups --group service --no-install-project \
    && /opt/venv/bin/python -c 'import importlib.util; assert all(importlib.util.find_spec(n) is None for n in ("PyQt6", "qtpy", "pyqtgraph", "qtawesome"))'

FROM python:3.13-slim
ARG PUID=1000
ARG PGID=1000
ARG VERSION=0.0.0-alpha.1
ARG REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/3ll3d00d/beqdesigner" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git openssh-client libsndfile1 \
    && ffmpeg -demuxers 2>/dev/null | grep -q dvdvideo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g "${PGID}" beq && useradd -m -u "${PUID}" -g beq beq \
    && mkdir -p /work /queue /config /app /opt/venv \
    && chown -R beq:beq /work /queue /app
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=docs-assets /static /app/static
COPY src/main/python /app/src/main/python
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src/main/python \
    BEQ_SERVICE_STATIC=/app/static PYTHONUNBUFFERED=1
WORKDIR /app
USER beq
EXPOSE 8080
VOLUME ["/work", "/queue"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).close()" || exit 1
ENTRYPOINT ["python", "-m", "pipeline.service", "--profile", "/config/profile.yaml", "--service-config", "/config/service.yaml"]
