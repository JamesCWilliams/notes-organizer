# ml-service

This runs a small Flask server that takes a handwriting image plus the raw pen strokes and spits back transcribed text using a pretrained model from Microsoft (TrOCR). The strokes are used to split the canvas into rows of writing (`utils/rows.py`), then each row is cropped out of the image and transcribed separately with beam search, best candidate wins.

It also produces embeddings for note similarity, three of them per note:

- **text**: the transcription, embedded with MiniLM (`embedding/text.py`).
- **image**: the canvas, embedded with SigLIP (`embedding/image.py`). SigLIP puts images and text in one shared space, so typed queries can later be matched against image vectors directly.
- **strokes**: the pen trajectory itself, embedded with our own encoder (`embedding/strokes.py`), trained in `../training`. This is the one that sees *how* the note was drawn: stroke order, shape, and dynamics that neither the transcription nor the flattened picture retains.

All vectors are unit length, cosine similarity between two of them is just a dot product, and vectors are only comparable to others from the same model, which is why each one is stored with the model name that produced it.

The stroke encoder is optional: if nothing has been exported to `models/stroke-encoder-*/` yet, that embedding comes back `null` and everything else still works.

Everything runs on CPU: torch is deliberately pinned to the CPU-only wheels (the CUDA ones drag in ~3GB of libraries this service never uses).

## getting started

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`, no requirements.txt anymore). From the `ml-service` folder:

```bash
uv run python app.py
```

That creates/updates `.venv` from the lockfile automatically on first run and starts the server on port 5000. To add or upgrade dependencies use `uv add <package>` / `uv lock --upgrade` so the lockfile stays in sync.

Alternatively, `bash dev.sh` from the repo root starts this service and the Tauri frontend together.

## the endpoints

Both take the same JSON body:

```json
{
  "image": "data:image/png;base64,...",
  "strokes": [[[x, y, pressure], ...], ...]
}
```

`image` is the rendered canvas as a base64 PNG data URL, `strokes` is one array per pen stroke of `[x, y, pressure]` points (bare `[x, y]` also accepted).

`POST /transcribe` returns `{ "text": "..." }` with rows joined by newlines. This is what the Analyze button uses.

`POST /analyze` runs transcription plus both embeddings — it's what enriches `.canvasnote` files on save:

```json
{
  "text": "...",
  "embeddings": {
    "text":    { "model": "all-MiniLM-L6-v2", "dim": 384, "vector": [...] },
    "image":   { "model": "google/siglip-base-patch16-224", "dim": 768, "vector": [...] },
    "strokes": { "model": "stroke-encoder-v2", "dim": 192, "vector": [...] }
  }
}
```

`embeddings.text` is `null` when no text was recognized (a pure doodle);
`embeddings.strokes` is `null` when no stroke encoder has been exported.

## running with docker

Download the model weights first (one time, ~2.2GB across TrOCR and the embedding models):

```bash
cd ml-service
uv run python download_models.py
```

This saves the weights under `ml-service/models/`. That folder is mounted into the container as a volume so they don't get re-downloaded on every build. There's some warning about pooler weights being missing, don't worry about it, it is spam.

Then from the repo root:

```bash
bash dev.sh --docker    # or: docker compose up --build
```

The image installs from the lockfile (`uv sync --frozen --no-dev`), so test/eval dependencies stay out of it. No GPU needed, see the CPU note above.

## debug logging

Set `ML_DEBUG=true` (dev.sh does this) and every prediction dumps its inputs and outputs under `debug/transcription/<timestamp>/`:

```
input_image.png     the canvas as received
strokes.json        the raw strokes as received
row_0.png, ...      the per-row crops fed to TrOCR
predictions.json    per-row top-5 candidates with confidences, plus timings
```

Handy for figuring out whether a bad transcription came from row segmentation or from the model. When running Dockerized the folder is volume-mounted, so dumps land on the host either way.

## testing

Unit tests (no model needed, runs fast):

```bash
uv run pytest tests/test_image.py
```

Full endpoint test (downloads the model on first run if `models/` is empty, ~1.3GB):

```bash
uv run pytest tests/test_endpoint.py
```

## checking if the model actually works

Grab some handwriting samples from the IAM dataset:

```bash
uv run python evaluation/download_iam_samples.py --count 20
```

Then run the evaluation to see Word Error Rate:

```bash
uv run python evaluation/evaluate.py
```

WER is basically "what percentage of words did it get wrong." Somewhere around 3-8% is what you'd expect on lots of clean handwriting. I'm getting like 20% from a small sample (20) of a large research dataset.

## layout

```
app.py               the flask server, just routes
pyproject.toml       dependencies (uv manages these; uv.lock pins them)
download_models.py   fetches all model weights into models/, safe to rerun
inference/
  transcribe.py      loads TrOCR, runs beam-search transcription per row
embedding/
  text.py            MiniLM text embeddings of transcriptions
  image.py           SigLIP image embeddings (+ text tower for cross-modal queries)
  strokes.py         our own stroke-trajectory encoder, trained in ../training
utils/
  rows.py            stroke-based row segmentation and cropping
  image.py           decodes the base64 image from the request
evaluation/
  evaluate.py        computes WER over sample images
  download_iam_samples.py   grabs samples from huggingface so you don't have to find them yourself
  samples/           put .png + .txt pairs here (matched by filename)
tests/
  test_image.py      tests for the image decoding util
  test_endpoint.py   tests for the flask endpoint (runs real inference)
models/              model weights live here (download_models.py fills it)
debug/               per-prediction dumps when ML_DEBUG=true
```
