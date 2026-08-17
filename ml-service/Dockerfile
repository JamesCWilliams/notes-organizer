# Build context is the REPO ROOT, not ml-service/, because the service shares
# stroke preprocessing with the training project (see embedding/strokes.py).
# Build via docker compose, or: docker build -f ml-service/Dockerfile .
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

# Runtime deps only (--no-dev skips pytest/eval tooling). The lockfile pins
# CPU-only torch from the pytorch index, so no CUDA libs. The cache mount
# persists uv's download cache across builds without baking it into the image
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=ml-service/uv.lock,target=uv.lock \
    --mount=type=bind,source=ml-service/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev

COPY ml-service/app.py .
COPY ml-service/inference/ inference/
COPY ml-service/utils/ utils/
COPY ml-service/embedding/ embedding/

# Shared preprocessing, at the path embedding/strokes.py expects (/training).
# Copied rather than vendored so there is still exactly one prep.py.
COPY training/stroke_encoder/__init__.py training/stroke_encoder/prep.py /training/stroke_encoder/

EXPOSE 5000

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "app.py"]
