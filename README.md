# notes-organizer

Draw handwritten notes on a canvas, get them transcribed by a local ML model, and (eventually) have the app organize them for you.

There are two separate pieces:

- **the app**: a [Tauri](https://tauri.app/) desktop app (React + Vite) with a drawing canvas. Strokes are captured with pressure via [perfect-freehand](https://github.com/steveruizok/perfect-freehand). Notes save as `.canvasnote` files: JSON holding the raw strokes plus, when available, the transcription.
- **the ML service**: a local Flask server that takes the canvas image and strokes, segments the writing into rows, and transcribes it with Microsoft's TrOCR model. Runs entirely on your machine, CPU only, nothing leaves your computer. Details in [ml-service/README.md](ml-service/README.md).

Hitting **Analyze** shows the transcription of what's on the canvas. Hitting **Save** writes the strokes to a file immediately, then asks the ML service for a transcription and tucks that into the file too (if the service is down the file still saves fine, the transcription is just left out).

The longer-term idea is to use those stored transcriptions for similarity: finding related notes, grouping by topic, searching your own handwriting, etc. hence the name.

## running it

You'll need [Node](https://nodejs.org/), [Rust](https://www.rust-lang.org/tools/install) (Tauri needs it), and [uv](https://docs.astral.sh/uv/) for the Python side.

```bash
npm install        # first time only
bash dev.sh
```

`dev.sh` starts the ML service (first run downloads the ~1.3GB TrOCR weights), waits for it to come up, then launches the app. Add `--docker` if you'd rather run the ML service in a container.

## layout

```
src/               the React frontend (canvas, save/load, talking to the ML service)
src-tauri/         Tauri shell (Rust)
ml-service/        Flask + TrOCR transcription service, has its own README
saves/             .canvasnote files end up wherever you save them; samples live here
dev.sh             starts everything for development
```
