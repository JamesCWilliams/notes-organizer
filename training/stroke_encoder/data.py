"""
Training data: doodles from QuickDraw and handwriting from IAM-OnDB, both
converted into the app's own stroke format and then mixed into one unlabeled
pile. The contrastive objective needs no labels, so the two sources need no
alignment beyond sharing a representation.

QuickDraw downloads itself (see fetch_quickdraw). IAM-OnDB requires a free
registration, so it has to be downloaded by hand, see the README.

Drawings are packed into a few large numpy arrays (StrokeStore) rather than
kept as Python lists. Lists of floats cost ~10x their data size in object
overhead, measured at 3 KB for an 18-point QuickDraw doodle and 95 KB for a
628-point IAM line, which puts a full corpus at ~6 GB, multiplied again by
every DataLoader worker that forks and touches it. Packed, the same corpus is
about 500 MB and forks cleanly.

Packing and the length measurement it feeds are both deterministic functions of
the files on disk, and both are slow, so both are cached under data/cache.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from .augment import augment
from .prep import prepare, prepared_length

DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'
QUICKDRAW_DIR = DATA_ROOT / 'quickdraw'
IAM_DIR = DATA_ROOT / 'iam_ondb'
CACHE_DIR = DATA_ROOT / 'cache'

_QUICKDRAW_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/{form}/{category}.ndjson'
_CATEGORY_LIST_URL = (
    'https://raw.githubusercontent.com/googlecreativelab/'
    'quickdraw-dataset/master/categories.txt'
)

# Pressure for sources that don't record it (QuickDraw, IAM-OnDB). Matches
# prep's default so packed and unpacked paths agree.
_DEFAULT_PRESSURE = 0.5

# Augmented draws measured per drawing when estimating batch lengths. Wants to
# be more than 2: each drawing is realized as two views, so the length a batch
# pads to is already a max over 2 draws, and estimating over the same number of
# draws only matches it half the time. Measured mean fill of the token budget,
# and the fraction of drawings surviving make_collate's trim: 0.79 and 72% at
# 2 draws, 0.88 and 81% at 4. Past that it is diminishing, since a batch's
# length is a max over thousands of drawings and some one of them will always
# beat its estimate. Each draw costs a prep pass over the corpus, but is cached.
_LENGTH_DRAWS = 4


def all_categories() -> list[str]:
    """Fetches the full list of 345 QuickDraw category names."""
    response = requests.get(_CATEGORY_LIST_URL, timeout=30)
    response.raise_for_status()
    return [line.strip() for line in response.text.splitlines() if line.strip()]


def fetch_quickdraw(categories=None, per_category: int = 5000, raw: bool = False) -> None:
    """Downloads QuickDraw categories into data/quickdraw/, one ndjson each.

    Streams and stops after per_category drawings rather than pulling whole
    files (some are hundreds of MB). Already-downloaded categories are skipped.

    raw=True keeps Google's original recordings, which include per-point
    timestamps; the default 'simplified' form is smaller and faster but has
    been resampled, which destroys timing. We do not currently use timing
    (see prep.FEATURE_DIM), so simplified is the sensible default.
    """
    categories = categories or all_categories()
    QUICKDRAW_DIR.mkdir(parents=True, exist_ok=True)
    form = 'raw' if raw else 'simplified'

    for category in tqdm(categories, desc=f'quickdraw ({form})'):
        target = QUICKDRAW_DIR / f'{category.replace(" ", "_")}.ndjson'
        if target.exists():
            continue

        url = _QUICKDRAW_URL.format(form=form, category=quote(category))
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 404:
            # Category names must match Google's list exactly ('sailboat', not
            # 'boat'); see all_categories().
            raise SystemExit(f'"{category}" is not a QuickDraw category (404 at {url})')
        response.raise_for_status()

        lines = []
        for line in response.iter_lines():
            if line:
                lines.append(line.decode('utf-8'))
            if len(lines) >= per_category:
                break
        response.close()

        # Write only after a complete read, so an interrupted download does
        # not leave a half-file that the skip-if-exists check would honor.
        target.write_text('\n'.join(lines))


def quickdraw_to_strokes(drawing) -> list:
    """QuickDraw's [[xs], [ys]] (or [[xs], [ys], [ts]]) into our stroke format."""
    return [
        [[float(x), float(y)] for x, y in zip(stroke[0], stroke[1])]
        for stroke in drawing
        if len(stroke) >= 2 and len(stroke[0])
    ]


def iter_quickdraw(limit_per_category: int | None = None) -> Iterator[list]:
    """Yields drawings from every downloaded category, one at a time."""
    for path in sorted(QUICKDRAW_DIR.glob('*.ndjson')):
        with path.open() as handle:
            for i, line in enumerate(handle):
                if limit_per_category is not None and i >= limit_per_category:
                    break
                if not line.strip():
                    continue
                strokes = quickdraw_to_strokes(json.loads(line)['drawing'])
                if strokes:
                    yield strokes


def iter_iam_ondb() -> Iterator[list]:
    """Yields IAM-OnDB handwriting lines, if the dataset has been downloaded.

    Expects the lineStrokes XML tree unpacked anywhere under data/iam_ondb/.
    Yields nothing when it is absent, so training still runs on QuickDraw
    alone.
    """
    if not IAM_DIR.exists():
        return

    for path in sorted(IAM_DIR.rglob('*.xml')):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue

        strokes = []
        for stroke in root.iter('Stroke'):
            points = []
            for point in stroke.iter('Point'):
                x, y = point.get('x'), point.get('y')
                if x is not None and y is not None:
                    points.append([float(x), float(y)])
            if points:
                strokes.append(points)
        if strokes:
            yield strokes


class StrokeStore:
    """Many drawings packed into three flat arrays.

    points is (P, 3) float32 of every point in the corpus; stroke_ends and
    drawing_ends are exclusive end indices that carve it back up. Indexing
    returns numpy views, so nothing is copied until an augmentation does it.
    """

    def __init__(self, points: np.ndarray, stroke_ends: np.ndarray, drawing_ends: np.ndarray):
        self.points = points
        self.stroke_ends = stroke_ends
        self.drawing_ends = drawing_ends

    @classmethod
    def build(cls, sources: dict[str, Iterable[list]]) -> tuple['StrokeStore', dict[str, range]]:
        """Packs named sources into one store.

        Returns the store and, per source, the range of drawing indices it
        occupies — which is what lets training rebalance an over-represented
        source without duplicating any point data.
        """
        chunks: list[np.ndarray] = []
        stroke_ends: list[int] = []
        drawing_ends: list[int] = []
        spans: dict[str, range] = {}
        point_count = 0

        for name, drawings in sources.items():
            start = len(drawing_ends)
            for drawing in tqdm(drawings, desc=f'packing {name}'):
                added = False
                for stroke in drawing:
                    array = np.asarray(stroke, dtype=np.float32)
                    if array.ndim != 2 or array.shape[1] < 2 or not len(array):
                        continue
                    if array.shape[1] == 2:
                        pressure = np.full((len(array), 1), _DEFAULT_PRESSURE, dtype=np.float32)
                        array = np.hstack([array, pressure])
                    chunks.append(array[:, :3])
                    point_count += len(array)
                    stroke_ends.append(point_count)
                    added = True
                if added:
                    drawing_ends.append(len(stroke_ends))
            spans[name] = range(start, len(drawing_ends))

        if not chunks:
            raise ValueError('no drawings to pack')

        return cls(
            np.concatenate(chunks, axis=0),
            np.asarray(stroke_ends, dtype=np.int64),
            np.asarray(drawing_ends, dtype=np.int64),
        ), spans

    def save(self, directory: Path) -> None:
        """Writes the three arrays out so the next run can skip packing."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / 'points.npy', self.points)
        np.save(directory / 'stroke_ends.npy', self.stroke_ends)
        np.save(directory / 'drawing_ends.npy', self.drawing_ends)

    @classmethod
    def load(cls, directory: Path) -> 'StrokeStore':
        """Reads a saved store back, memory-mapping the bulk of it.

        points is the big array and is only ever read, so mapping it keeps one
        copy in the page cache for every DataLoader worker instead of one per
        forked process. The two offset arrays are small and get read on every
        __getitem__, so those are loaded properly.
        """
        return cls(
            np.load(directory / 'points.npy', mmap_mode='r'),
            np.load(directory / 'stroke_ends.npy'),
            np.load(directory / 'drawing_ends.npy'),
        )

    def __len__(self) -> int:
        return len(self.drawing_ends)

    def __getitem__(self, index: int) -> list[np.ndarray]:
        first_stroke = 0 if index == 0 else int(self.drawing_ends[index - 1])
        last_stroke = int(self.drawing_ends[index])
        strokes = []
        for s in range(first_stroke, last_stroke):
            start = 0 if s == 0 else int(self.stroke_ends[s - 1])
            strokes.append(self.points[start:int(self.stroke_ends[s])])
        return strokes

    def nbytes(self) -> int:
        return self.points.nbytes + self.stroke_ends.nbytes + self.drawing_ends.nbytes


def _hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
    return digest.hexdigest()[:16]


def _corpus_key(limit_per_category: int | None) -> str:
    """Identifies the corpus by its source files, without reading them.

    Name, size and mtime of every input file. Cheap enough to run on every
    start, and it means downloading more QuickDraw categories or unpacking more
    IAM invalidates the cache on its own rather than relying on anyone
    remembering to clear it.
    """
    files = sorted(QUICKDRAW_DIR.glob('*.ndjson')) + sorted(IAM_DIR.rglob('*.xml'))
    return _hash(repr(limit_per_category), *(
        f'{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}' for path in files
    ))


def load_or_build_store(limit_per_category: int | None = None):
    """The packed corpus, from cache when the source files have not changed.

    Packing parses ~500 MB of ndjson and XML for a result identical run to run.
    Caching turns that into a memory-mapped read.
    """
    directory = CACHE_DIR / f'store-{_corpus_key(limit_per_category)}'
    spans_path = directory / 'spans.json'
    if spans_path.exists():
        spans = {name: range(*bounds) for name, bounds in json.loads(spans_path.read_text()).items()}
        print(f'loaded packed corpus from {directory}')
        return StrokeStore.load(directory), spans

    store, spans = StrokeStore.build({
        'quickdraw': iter_quickdraw(limit_per_category),
        'iam': iter_iam_ondb(),
    })
    store.save(directory)
    spans_path.write_text(json.dumps({name: [s.start, s.stop] for name, s in spans.items()}))
    return store, spans


def cached_lengths(store: StrokeStore, limit_per_category: int | None = None) -> np.ndarray:
    """Longest sequence each drawing is likely to produce once augmented.

    Bucketing has to predict the length of the tensor that reaches the model,
    and that is prepare(augment(drawing)), never the drawing as stored. The two
    differ because the augmentations can lengthen the pen path, and prep
    resamples at a constant spacing, so a longer path becomes more points.

    Taking the worst of a few augmented draws beats predicting the inflation
    analytically, which would need rederiving every time augment.py changed.
    It is an estimate over samples rather than a bound, so make_collate still
    enforces the budget once the true length is known.

    Cached because this is the slowest thing between launching a run and the
    first epoch. The key covers the corpus, prep and augment, so editing either
    module recomputes rather than silently reusing stale lengths.
    """
    here = Path(__file__).parent
    recipe = _hash(
        str(_LENGTH_DRAWS),
        # Hashed rather than keyed on PREP_VERSION, which is a human promise to
        # bump a string. Stale lengths silently build over-budget batches, so
        # this one should not depend on anybody remembering.
        (here / 'prep.py').read_text(),
        (here / 'augment.py').read_text(),
    )
    path = CACHE_DIR / f'store-{_corpus_key(limit_per_category)}' / f'lengths-{recipe}.npy'
    if path.exists():
        counts = np.load(path)
        if len(counts) == len(store):
            print(f'loaded prepared lengths from {path.name}')
            return counts

    rng = np.random.default_rng(0)
    counts = np.empty(len(store), dtype=np.int64)
    for i in tqdm(range(len(store)), desc=f'measuring lengths ({_LENGTH_DRAWS} draws)'):
        counts[i] = max(prepared_length(augment(store[i], rng)) for _ in range(_LENGTH_DRAWS))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, counts)
    return counts


class LengthBucketSampler(Sampler):
    """Yields batches of similar-length drawings, sized by a token budget.

    Padding is charged per batch at the longest member, so mixing a 39-point
    doodle with a 256-point handwriting line makes everyone pay 256. Sorting by
    length first cut padding from 42% to 7%, and lets short batches hold
    thousands of drawings while only the genuinely long ones shrink.

    The trade-off for contrastive training is that the number of in-batch
    negatives now varies with sequence length. Batches also become
    length-homogeneous, which removes sequence length as a trivial cue the
    model could otherwise use to tell samples apart.

    max_drawings is not optional in practice: the budget bounds drawings x
    length, leaving the count unbounded as lengths shrink, while NT-Xent builds
    a (2N, 2N) matrix that grows with its square. Four drawings out of 342,600
    reduce to 2 points, and a bucket of 2-point drawings takes budget/2 of
    them, which put a 5 GB similarity matrix on the card before the backward
    pass and set peak VRAM for the whole run.
    """

    def __init__(
        self,
        lengths,
        token_budget: int,
        max_drawings: int = 8192,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths)
        self.token_budget = token_budget
        self.max_drawings = max_drawings
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._cache: tuple[int, list[np.ndarray]] | None = None

    def _batches(self) -> list[np.ndarray]:
        # DataLoader asks for len() as well as iterating, and bucketing sorts
        # the whole corpus, so memoize until the epoch actually advances.
        if self._cache is not None and self._cache[0] == self.epoch:
            return self._cache[1]

        rng = np.random.default_rng((self.seed, self.epoch))
        # Jitter the sort key so identical-length items don't clump into the
        # exact same batch every epoch.
        keys = self.lengths + (rng.random(len(self.lengths)) * 8 if self.shuffle else 0)
        order = np.argsort(keys, kind='stable')

        batches, start, longest = [], 0, 0
        for position, index in enumerate(order):
            longest = max(longest, int(self.lengths[index]))
            count = position - start + 1
            if (longest * count > self.token_budget or count > self.max_drawings) \
                    and position > start:
                batches.append(order[start:position])
                start, longest = position, int(self.lengths[index])
        if start < len(order):
            batches.append(order[start:])

        if self.shuffle:
            rng.shuffle(batches)
        self._cache = (self.epoch, batches)
        return batches

    def __iter__(self):
        for batch in self._batches():
            yield batch.tolist()
        self.epoch += 1

    def __len__(self) -> int:
        return len(self._batches())


class ContrastiveStrokes(Dataset):
    """Yields two independently augmented views of the same drawing.

    indices may repeat entries, which is how an under-represented source gets
    oversampled, the augmentations make each repeat a different view.
    """

    def __init__(self, store: StrokeStore, indices=None, seed: int = 0):
        self.store = store
        self.indices = np.arange(len(store)) if indices is None else np.asarray(indices)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        drawing = self.store[int(self.indices[index])]
        # Seeded per call so DataLoader workers don't share an augmentation
        # stream, and so repeated indices get different views.
        rng = np.random.default_rng((self.seed, index, np.random.randint(1 << 30)))
        return prepare(augment(drawing, rng)), prepare(augment(drawing, rng))
