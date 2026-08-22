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

Packing, and the pass that measures batch lengths, are both cached under
`data/cache/`, keyed on the name/size/mtime of every source file plus the
contents of `prep.py` and `augment.py`. Downloading more QuickDraw categories,
unpacking more IAM, or editing either module invalidates the cache on its own,
so there is nothing to remember to clear. A warm start memory-maps the points
array instead of reparsing ~500 MB of ndjson and XML, so workers share one copy
through the page cache rather than one per forked process.

The thing actually worth tuning is the **mixing ratio**, not the category
list. All of QuickDraw is ~345k–1.7M doodles against IAM-OnDB's 12k
handwriting lines, which would leave handwriting at well under 1% of what the
encoder ever sees, for an app that is mostly handwriting. `--handwriting-share`
(default 0.35) oversamples IAM to fix that; repeats are free, since only
indices are duplicated and each repeat is augmented differently.

## training

```bash
uv run python -m stroke_encoder.train_contrastive --epochs 100
```

Two augmented views of the same drawing are pulled together, every other
drawing in the batch is pushed away (NT-Xent). The encoder treats handwriting and 
diagrams as the same kind of input, which is the whole point.

Everything that buys batch size is on by default, because batch size is what
NT-Xent turns into negatives. Training is bf16 autocast with no `GradScaler`:
loss scaling exists to stop fp16's narrow exponent range flushing small
gradients to zero, and bfloat16 has the same exponent range as fp32.

### batching, and why there is no --batch-size

Batches are formed by a **token budget**, not a sample count. Drawing lengths
vary a lot, and a batch pads to its longest member, so a random mix makes every
short drawing pay for the longest one in the batch. Sorting by length first cut
padding from 42% to 7%, and lets short batches hold thousands of drawings while
long ones shrink to a few hundred.

The trade-off is that the number of in-batch negatives now varies with
sequence length. Batches also become length-homogeneous, which removes
sequence length as a trivial cue the model could otherwise lean on.

Bucketing must be fed `prep.prepared_length`, **not** the stored point count.
Since prep v2 resamples to a constant spacing it can *add* points, so a sparse
41-point QuickDraw polyline becomes ~139 and the raw count underestimates by up
to 11x. Sizing batches off the raw count silently builds batches 3-4x over
budget, which is enough to exhaust host RAM.

It must also be fed the length of the drawing **as augmented**, since that is
what actually reaches the model, so `data.cached_lengths` measures a few
augmented draws per drawing and keeps the worst. That is an estimate over
samples, never a bound: a batch pads to the longest of the thousands of
drawings in it, and one of them will always beat its estimate. So
`make_collate` enforces the budget where the true length is finally known,
trimming an overshooting batch by dropping whole drawings (both views) at
random. Random rather than longest-first, which would systematically thin out
long handwriting lines and quietly reshape what the encoder trains on.

Between the two, batches run ~88% full and never exceed the budget.

#### --max-batch, and why the budget alone is not enough

The token budget bounds `drawings x length`. It does **not** bound the drawing
count, which grows as lengths shrink, and NT-Xent builds a `2N x 2N` similarity
matrix that grows with the *square* of it.

That is not a theoretical concern. Four drawings out of 342,600 in QuickDraw
reduce to 2 points, and a bucket whose longest member is 2 points takes
`budget/2` drawings, which puts a 5 GB similarity matrix on the card before the
backward pass. Those four drawings were setting peak VRAM for the entire run,
forcing the budget down to 100k, where every *other* batch then ran at a quarter
of the size it could have. Past about 400k there is a hard failure too: the
efficient-attention kernel refuses batches over 65,535 rows.

`--max-batch` (default 8192 drawings) caps the count so the two are decoupled.
Measured cost of the loss alone, which is what this bounds:

| `--max-batch` | NT-Xent peak |
|---------------|--------------|
| 4096 | 0.84 GB |
| **8192 (default)** | **3.29 GB** |
| 16384 | 12.98 GB — the loss, not the encoder, is now the ceiling |

#### gradient checkpointing

On by default (`--no-grad-checkpointing` to disable). Encoder layers are
recomputed during the backward pass instead of keeping their activations, which
is what makes the current token budget affordable:

| | peak reserved at 400k tokens | ms/step |
|---|------------------------------|---------|
| without | 11.73 GB | 308 |
| **with** | **4.62 GB** | 403 |

31% slower per step, for 2.5x the batch. That is a good trade *here* and not in
general: every drawing in the batch is a negative, so batch size is the main
thing determining how hard the task is, and this is the cheapest place to buy
it. It lives in `train_contrastive.encode`, not in `StrokeEncoder`, because
`torch.utils.checkpoint` is not TorchScript-able and `export.py` scripts the
model — so the forward the service runs stays exactly the forward that was
trained. `tests/test_checkpointing.py` pins the two together.

#### tuning --token-budget

**Measure, don't guess, and re-measure after any change to `prep.py`.** With
checkpointing on, peak scales close to linearly at roughly 1.1 GB per 100k
tokens, but that mapping depends on the whole length distribution, and changing
preprocessing moves it. Figures measured under prep v1 stopped holding the
moment v2 changed how long a drawing is.

Measured on a 16 GB card (RTX 4070 Ti SUPER) at prep v2, checkpointing on,
`--max-batch 8192`, worst-case batch per budget:

| budget | peak reserved | |
|--------|---------------|---|
| 400k | 4.62 GB | wastes most of the card |
| 800k | 8.96 GB | |
| **1M (default)** | **11.2 GB** | ~12.4 GB in a real run, once fragmentation is counted |
| 1.2M | 13.45 GB | no room left for the desktop |

Note the desktop's own ~2.8 GB comes out of the same 16 GB, so the usable
ceiling is well below what `nvidia-smi` advertises.

Each epoch prints its own peak, so tune against the real thing:

```
epoch   1  train 7.8895  val 5.7154  (14s)  peak 11.2 GB  1,410 drawings/s
```

Bigger is better for contrastive learning, since every other drawing in the
batch is a negative, so the few minutes of tuning are worth it.

Overshooting used **not** to raise a clean out-of-memory error. On Windows/WSL
the driver silently spills into shared system memory over PCIe and everything
just gets several times slower — a peak pinned near capacity with throughput
well down, rather than a crash. `--vram-fraction` (default 0.85) caps the
allocator so that becomes a real `torch.OutOfMemoryError` instead, which is why
the budget is now tunable in minutes rather than by watching for a slowdown.
The epoch line still reports both numbers.

### augmentations

`augment.py` *defines* what "the same drawing" means, so it's where the real
design work is. Three choices made there:

* rotation is small jitter, not an invariance, since a rotated page is a
  different note;
* stroke order is never shuffled, because draw order is signal a pixel model
  can't see;
* positional noise is capped against the drawing's own point spacing, not just
  scaled to its size.

That last one matters more than it looks. Noise applied per point stretches each
segment by about `sqrt(1 + 4(sigma/step)^2)`, which is nothing for QuickDraw's
sparse polylines but severe for IAM-OnDB's ~614-point lines, where consecutive
points sit closer together than sigma itself. Uncapped it inflated IAM's
prepared length **1.81x** while leaving QuickDraw at 1.00x, so the augmentation
was ~10x stronger for handwriting than for doodles: the exact sample-rate
dependence `prep.py` exists to remove, reintroduced a stage earlier. Capped,
both sources sit at 1.01-1.02x.

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
tests/                  pins the checkpointed forward to the exported one
data/                   downloaded datasets, and the packing cache (gitignored)
checkpoints/            training output (gitignored)
```
