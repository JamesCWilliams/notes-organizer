FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first: the default PyPI wheel drags in ~3GB of CUDA libs the
# service never uses (inference is pinned to CPU). The cache mount persists
# pip's download cache across builds without baking it into the image layer.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY app.py .
COPY inference/ inference/
COPY utils/ utils/

EXPOSE 5000

CMD ["python", "app.py"]
