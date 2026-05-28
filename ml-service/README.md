# ml-service

This runs a small Flask server that takes a handwriting image and spits back transcribed text using a pretrained model from Microsoft (TrOCR).

## getting started

Make sure you're in the `ml-service` folder, then set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the server:

```bash
python app.py
```

It'll be listening on port 5000. The one endpoint is `POST /transcribe`, send it a JSON body with an `image` key containing a base64-encoded PNG and it'll return `{ "text": "..." }`.

If you have a GPU it'll use it automatically, otherwise it falls back to CPU.

## running with docker

Download the model weights first (one time, ~1.3GB):

```bash
cd ml-service
python download_model.py
```

This saves the weights to `ml-service/models/trocr-base-handwritten/`. That folder is mounted into the container as a volume so they don't get re-downloaded on every build. There's some warning about pooler weights being missing, don't worry about it, it is spam.

Then from the repo root:

```bash
docker compose up --build
```

That builds the image and starts the server on port 5000. To stop it: `ctrl+c`, then `docker compose down`.

**GPU requirement:** the container expects an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to be installed on the host. If you don't have that set up, the container will fail to start. In that case just run it locally with `python app.py` instead — it'll fall back to CPU.

## testing

Unit tests (no model needed, runs fast):

```bash
pytest tests/test_image.py
```

Full endpoint test (downloads the model on first run, ~1.3GB):

```bash
pytest tests/test_endpoint.py
```

## checking if the model actually works

Grab some handwriting samples from the IAM dataset:

```bash
python evaluation/download_iam_samples.py --count 20
```

Then run the evaluation to see Word Error Rate:

```bash
python evaluation/evaluate.py
```

WER is basically "what percentage of words did it get wrong." Somewhere around 3-8% is what you'd expect on lots of clean handwriting. I'm getting like 20% from a small sample (20) of a large research dataset.

## layout

```
app.py               the flask server, just routes
inference/
  transcribe.py      loads the model and does the actual inference
utils/
  image.py           decodes the base64 image from the request
evaluation/
  evaluate.py        computes WER over sample images
  download_iam_samples.py   grabs samples from huggingface so you don't have to find them yourself
  samples/           put .png + .txt pairs here (matched by filename)
tests/
  test_image.py      tests for the image decoding util
  test_endpoint.py   tests for the flask endpoint
models/              model weights live here if you download them manually
```
