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
"""

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
from .prep import prepare

DATA_ROOT = Path(__file__).resolve().parents[1] / 'data'
QUICKDRAW_DIR = DATA_ROOT / 'quickdraw'
IAM_DIR = DATA_ROOT / 'iam_ondb'

_QUICKDRAW_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/{form}/{category}.ndjson'
_CATEGORY_LIST_URL = (
    'https://raw.githubusercontent.com/googlecreativelab/'
    'quickdraw-dataset/master/categories.txt'
)

# Pressure for sources that don't record it (QuickDraw, IAM-OnDB). Matches
# prep's default so packed and unpacked paths agree.
_DEFAULT_PRESSURE = 0.5


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

    def point_counts(self) -> np.ndarray:
        """Points per drawing, straight from the offsets, no parsing.

        Augmentation only ever removes points, so this is an upper bound on
        the sequence length a drawing will produce, which is what bucketing
        needs.
        """
        stroke_starts = np.concatenate([[0], self.stroke_ends[:-1]])
        per_stroke = self.stroke_ends - stroke_starts
        ends = self.drawing_ends
        cumulative = np.concatenate([[0], np.cumsum(per_stroke)])
        return (cumulative[ends] - cumulative[np.concatenate([[0], ends[:-1]])]).astype(np.int64)


class LengthBucketSampler(Sampler):
    """Yields batches of similar-length drawings, sized by a token budget.

    Padding is charged per batch at the longest member, so mixing a 39-point
    doodle with a 256-point handwriting line makes everyone pay 256. Sorting
    by length first means short batches can be enormous and only the genuinely
    long batches are small, the GPU sees a roughly constant number of real
    tokens either way.

    The trade-off for contrastive training is that batch size, and so the
    number of in-batch negatives, now varies. Batches also become
    length-homogeneous, which removes sequence length as a trivial cue the
    model could otherwise use to tell samples apart.
    """

    def __init__(self, lengths, token_budget: int, shuffle: bool = True, seed: int = 0):
        self.lengths = np.asarray(lengths)
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def _batches(self) -> list[np.ndarray]:
        rng = np.random.default_rng((self.seed, self.epoch))
        # Jitter the sort key so identical-length items don't clump into the
        # exact same batch every epoch.
        keys = self.lengths + (rng.random(len(self.lengths)) * 8 if self.shuffle else 0)
        order = np.argsort(keys, kind='stable')

        batches, start, longest = [], 0, 0
        for position, index in enumerate(order):
            longest = max(longest, int(self.lengths[index]))
            if longest * (position - start + 1) > self.token_budget and position > start:
                batches.append(order[start:position])
                start, longest = position, int(self.lengths[index])
        if start < len(order):
            batches.append(order[start:])

        if self.shuffle:
            rng.shuffle(batches)
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
