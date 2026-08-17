# notes-organizer

Draw handwritten notes on a canvas, get them transcribed by a local ML model, and (eventually) have the app organize them for you.

There are three separate pieces:

- **the app**: a [Tauri](https://tauri.app/) desktop app (React + Vite) with a drawing canvas. Strokes are captured with pressure via [perfect-freehand](https://github.com/steveruizok/perfect-freehand). Notes save as `.canvasnote` files: JSON holding the raw strokes, and ML analysis: the transcription and embeddings.
- **the ML service**: a local Flask server that takes the canvas image and strokes, segments the writing into rows based on the strokes, and transcribes it with Microsoft's TrOCR model. Also computes embeddings using three separate ML models for each modality (canvas, text, and strokes). Runs entirely on your machine, CPU only, nothing leaves your computer. Details in [ml-service/README.md](ml-service/README.md).
- **model training**: a separate python environment for training new ML models. This is mainly for training stroke encoders, as there aren't really any open-source pretrained models for the use-case. This could also be used for fine-tuning models later. Details in [training/README.md](training/README.md).

Hitting **Analyze** shows the transcription of what's on the canvas. Hitting **Save** writes the strokes to a file immediately, then asks the ML service for a transcription and tucks that into the file too (if the service is down the file still saves fine, the transcription is just left out).

The longer-term idea is to use those stored transcriptions, as well as the final canvas and the stroke data, for similarity: finding related notes, grouping by topic, searching your own handwriting, etc. hence the name.

"Similarity" is the cosine similarity of the embedding vectors. The embeddings should be similar for related content and dissimilar for unrelated content. Eventually all three embeddings can be used in combination for retrieval tasks.

## running it

You'll need [Node](https://nodejs.org/), [Rust](https://www.rust-lang.org/tools/install) (Tauri needs it), and [uv](https://docs.astral.sh/uv/) for the Python side.

```bash
npm install        # first time only
bash dev.sh
```

`dev.sh` starts the ML service (first run downloads the pretrained model weights, a couple GB), waits for it to come up, then launches the app. Add `--docker` if you'd rather run the ML service in a container.

## layout

```
src/               the React frontend (canvas, save/load, talking to the ML service)
src-tauri/         Tauri shell (Rust)
ml-service/        Flask server which handles transcriptions and embedding, has its own README
saves/             .canvasnote files end up wherever you save them; samples live here
dev.sh             starts everything for development
training/          Training environment for new ML models, has its own README
```
