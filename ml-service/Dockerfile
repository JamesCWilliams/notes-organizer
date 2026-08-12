FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

# Runtime deps only (--no-dev skips pytest/eval tooling). The lockfile pins
# CPU-only torch from the pytorch index, so no CUDA libs. The cache mount
# persists uv's download cache across builds without baking it into the image
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev

COPY app.py .
COPY inference/ inference/
COPY utils/ utils/

EXPOSE 5000

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "app.py"]
