# training

Trains the stroke encoder: a small transformer that embeds raw pen strokes, so
notes can be compared by *how they were drawn* and not just by how they look or
what they say. This is the one modality where off-the-shelf models don't exist,
because nobody else stores the pen trajectory.

This is a **separate uv project** from `ml-service` on purpose. Training wants
CUDA torch; the service deliberately pins the CPU-only wheels. Two projects,
two lockfiles, neither constrains the other. The service never imports this
package, trained models cross the boundary as TorchScript artifacts.

## setup

```bash
cd training
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

## data

QuickDraw downloads itself (doodles, no labels needed since the objective is
self-supervised):

```bash
# all 345 categories, 1000 drawings each
uv run python -c "
from stroke_encoder.data import fetch_quickdraw, all_categories
fetch_quickdraw(all_categories(), per_category=1000)"
```

It streams and stops early rather than downloading whole files, and skips
categories already on disk, so it's safe to interrupt and rerun. Categories
are only a diversity knob, nothing ever trains on the labels, and more
variety is better for a general-purpose encoder, so take all 345 rather than
the `DEFAULT_CATEGORIES` starter list.

**IAM-OnDB** (handwriting) can't be downloaded automatically: it needs a free
registration at
<https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database>. Get
`lineStrokes-all.tar.gz` (pen trajectories; `lineImages` is the offline
pixel version and is not what we want) and unpack it anywhere under
`data/iam_ondb/`, the loader recurses. `ascii-all.tar.gz` is worth grabbing
at the same time: it isn't used yet, but it pairs each line with its text,
which is the supervision an eventual stroke-to-text alignment head would need.
I never received the validation email when using a normal email address,
but combining a VPN with a temporary email address (<https://temp-mail.org/en/>) was suggested
on Reddit and worked for me. I'm not sure which part is important.

Its coordinates are already screen-oriented (y down), the same as the app and
QuickDraw, so no flip is applied, verified by rendering lines and reading
them back against the ascii transcriptions.

Training runs on QuickDraw alone if IAM-OnDB is missing, but mixing in real
handwriting is what makes the encoder work on actual notes.

### sizing and the mixing ratio

Drawings are packed into flat numpy arrays (`StrokeStore`), which matters more
than it sounds: as Python lists the same data costs ~10x more (measured 3 KB
for an 18-point doodle, 95 KB for a 628-point IAM line), and every DataLoader
worker that forks and touches those objects multiplies it again. Packed, a
full corpus is a few hundred MB and forks cleanly. IAM-OnDB alone is 95 MB
packed versus 1.1 GB as lists.

The thing actually worth tuning is the **mixing ratio**, not the category
list. All of QuickDraw is ~345k–1.7M doodles against IAM-OnDB's 12k
handwriting lines, which would leave handwriting at well under 1% of what the
encoder ever sees, for an app that is mostly handwriting. `--handwriting-share`
(default 0.35) oversamples IAM to fix that; repeats are free, since only
indices are duplicated and each repeat is augmented differently.

## training

```bash
uv run python -m stroke_encoder.train_contrastive --epochs 20
```

Two augmented views of the same drawing are pulled together, every other
drawing in the batch is pushed away (NT-Xent). The encoder treats handwriting and 
diagrams as the same kind of input, which is the whole point.

### batching, and why there is no --batch-size

Batches are formed by a **token budget**, not a sample count. Drawing lengths
are strongly bimodal, QuickDraw doodles run ~40 points, IAM handwriting lines
hit the 256 cap, and a batch pads to its longest member, so a random mix
makes every doodle pay for a handwriting line. Sorting by length first cut
padding from 42% to 7%, and lets short batches hold ~6000 drawings while long
ones shrink to ~200, keeping VRAM roughly flat either way.

The trade-off is that the number of in-batch negatives now varies with
sequence length. Batches also become length-homogeneous, which removes
sequence length as a trivial cue the model could otherwise lean on.

**Tune `--token-budget` to your card, and measure rather than guess.** On a
16 GB card:

| budget | peak VRAM | throughput |
|--------|-----------|------------|
| 260k (default) | 9.8 GB | 11,900 drawings/s |
| 400k | 16.6 GB | 7,200/s |
| 600k | 27 GB (spilled) | 1,400/s |

Note that overshooting does **not** raise a clean out-of-memory error. On
Windows/WSL the driver silently spills into shared system memory over PCIe and
everything just gets several times slower, the 600k row is not an error, it is
a "working" run at 8x the cost.

The augmentations in `augment.py` *define* what "the same drawing" means, so
they're where the real design work is. Two choices already made there: rotation
is small jitter rather than an invariance (a rotated page is a different note),
and stroke order is never shuffled (draw order is signal a pixel model can't
see).

Best-val checkpoints land in `checkpoints/`.

## exporting to the service

```bash
uv run python -m stroke_encoder.export --name stroke-encoder-v2
```

Writes TorchScript + `meta.json` into `ml-service/models/stroke-encoder-v2/`,
which `ml-service/embedding/strokes.py` loads with `torch.jit.load`, no model
class needed on the serving side. Unlike the downloaded HuggingFace weights,
these are committed to git: they're small, and nobody else can re-download
them.

Every retrain that changes the architecture or the features gets a **new name**
(`-v3`, ...). The name is written into each save file's embedding record, so
old notes keep their old vectors and similarity search only ever compares
vectors from the same model.

## the preprocessing contract

`prep.py` is imported by *both* this project and `ml-service`, and that is
deliberate: a note embedded at serve time must be preprocessed exactly the way
training data was. If the two ever drift apart, embeddings degrade silently,
with no error anywhere. Any change to prep means a new `PREP_VERSION` and a
retrained encoder, `export.py` refuses to export a checkpoint whose prep
version doesn't match the tree.

## layout

```
stroke_encoder/
  prep.py               strokes -> (dx, dy, pressure, pen_up) tensor  [SHARED WITH ml-service]
  augment.py            stroke-space augmentations; defines "same drawing"
  model.py              transformer encoder + projection head + NT-Xent loss
  data.py               QuickDraw download, IAM-OnDB parser, packed store, dataset
  train_contrastive.py  the training loop
  export.py             checkpoint -> TorchScript into ml-service/models/
data/                   downloaded datasets (gitignored)
checkpoints/            training output (gitignored)
```
