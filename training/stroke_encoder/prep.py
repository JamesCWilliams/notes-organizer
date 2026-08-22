"""
Turns raw canvas strokes into the tensor the stroke encoder consumes.

THIS IS SHARED BETWEEN TRAINING AND SERVING. ml-service imports it so that a
note embedded at serve time is preprocessed exactly the way training data was.
If the two ever drift, embeddings silently degrade with no error anywhere,
so any change here means a new PREP_VERSION and a retrained encoder.

Input is the app's own stroke format: a list of strokes, each a list of
[x, y, pressure] points ([x, y] is accepted, pressure defaults to 0.5).

Output is a (T, 4) float32 array of (dx, dy, pressure, pen_up):
  dx, dy   offset from the previous point, in normalized units
  pressure raw pen pressure, 0..1
  pen_up   1.0 on the last point of a stroke, else 0.0

Normalization translates and scales the drawing into a unit box but does NOT
rotate it: a page of horizontal lines and a page of vertical lines should not
embed identically.

Points are then resampled to a constant spacing along the pen path. Every
source records at a different rate: QuickDraw is decimated to sparse polylines,
IAM-OnDB is raw ~100 Hz tablet capture, the app gets browser pointer events.
Measured mean step length differed 10x between them (0.16 vs 0.015 normalized
units) for reasons having nothing to do with what was drawn.
Point count and step length alone classified drawing-vs-handwriting at 99.7%,
so a model trained on raw offsets could score well while learning only which
device recorded the strokes. Resampling collapses all sources onto one spacing,
leaving path shape rather than pen sample rate. Point counts still differ
afterwards, but now because a turbulent doodle genuinely needs more points than
a straight line.
"""

import numpy as np

PREP_VERSION = 'v2'

# Points per drawing fed to the model. Longer inputs are decimated, shorter
# ones are padded by the caller (see pad_batch).
MAX_POINTS = 256

# Distance between resampled points, in normalized units (the drawing's longer
# side is 1.0). Chosen from measured path lengths: the median normalized path
# runs 2.3 units for IAM-OnDB handwriting, 3.6 for the app's own notes and 4.0
# for QuickDraw, so 0.03 yields a median of 77-135 points across all three and
# leaves headroom under MAX_POINTS. Only ~5% of QuickDraw drawings exceed the
# cap, and _spacing widens the step for those rather than letting them clip.
TARGET_SPACING = 0.03

# (dx, dy, pressure, pen_up)
FEATURE_DIM = 4

_DEFAULT_PRESSURE = 0.5


def _clean(strokes) -> list[np.ndarray]:
    """Drops empty strokes and normalizes each to an (N, 3) float32 array."""
    cleaned = []
    for raw_stroke in strokes:
        if len(raw_stroke) == 0:
            continue
        points = np.asarray(raw_stroke, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            continue
        if points.shape[1] == 2:  # bare [x, y]
            pressure = np.full((len(points), 1), _DEFAULT_PRESSURE, dtype=np.float32)
            points = np.hstack([points, pressure])
        cleaned.append(points[:, :3])
    return cleaned


def _decimate(strokes: list[np.ndarray], budget: int) -> list[np.ndarray]:
    """Thins strokes to fit the point budget, keeping every stroke's endpoints.

    Each stroke keeps a share of the budget proportional to its length, so a
    long stroke is not thinned down to the same handful of points as a dot.
    """
    total = sum(len(s) for s in strokes)
    if total <= budget:
        return strokes

    thinned = []
    for stroke in strokes:
        if len(stroke) <= 2:
            thinned.append(stroke)
            continue
        keep = max(2, round(budget * len(stroke) / total))
        # Evenly spaced sample that always includes the first and last point.
        idx = np.unique(np.linspace(0, len(stroke) - 1, keep).round().astype(int))
        thinned.append(stroke[idx])
    return thinned


def _normalize(strokes: list[np.ndarray]) -> list[np.ndarray]:
    """Translates and scales a drawing into a unit box, preserving aspect ratio."""
    xy = np.vstack([s[:, :2] for s in strokes])
    lo = xy.min(axis=0)
    span = float(max(xy.max(axis=0) - lo))
    # A single point, or a perfectly straight horizontal/vertical line, has
    # zero span on at least one axis; scaling by the larger span keeps aspect
    # ratio intact and avoids dividing by zero.
    scale = span if span > 1e-6 else 1.0

    out = []
    for stroke in strokes:
        moved = stroke.copy()
        moved[:, :2] = (stroke[:, :2] - lo) / scale
        out.append(moved)
    return out


def _arc_lengths(stroke: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative path length per point, with coincident points removed.

    Repeated points contribute zero-length segments, which would make the
    interpolation below ambiguous, so they are dropped here rather than
    special-cased later.
    """
    steps = np.linalg.norm(np.diff(stroke[:, :2], axis=0), axis=1)
    keep = np.concatenate([[True], steps > 1e-12])
    stroke = stroke[keep]
    if len(stroke) < 2:
        return stroke, np.zeros(len(stroke), dtype=np.float64)
    steps = np.linalg.norm(np.diff(stroke[:, :2], axis=0), axis=1)
    return stroke, np.concatenate([[0.0], np.cumsum(steps)])


def _spacing(strokes: list[np.ndarray], max_points: int) -> float:
    """Resampling step: TARGET_SPACING, widened if the drawing would overflow.

    Each stroke costs one point beyond its length's share, since both endpoints
    are always kept, so the budget accounts for the stroke count. A drawing long
    enough to need a wider step keeps a uniform spacing internally; it just
    carries less detail, which is a better failure than clipping the tail of the
    drawing off entirely.
    """
    total = 0.0
    for stroke in strokes:
        _, arc = _arc_lengths(stroke)
        total += float(arc[-1]) if len(arc) else 0.0
    if total < 1e-9:
        return TARGET_SPACING
    budget = max(max_points - len(strokes), 1)
    return max(TARGET_SPACING, total / budget)


def _resample(stroke: np.ndarray, spacing: float) -> np.ndarray:
    """Resamples one stroke to evenly spaced points along its own path.

    Pressure is interpolated alongside position. Endpoints are always kept, so
    stroke shape survives, and a dot (no path length) stays a single point.
    """
    stroke, arc = _arc_lengths(stroke)
    if len(stroke) < 2:
        return stroke
    total = float(arc[-1])
    if total < 1e-9:
        return stroke[:1]

    count = max(2, round(total / spacing) + 1)
    targets = np.linspace(0.0, total, count)
    out = np.empty((count, stroke.shape[1]), dtype=np.float32)
    for column in range(stroke.shape[1]):
        out[:, column] = np.interp(targets, arc, stroke[:, column])
    return out


def prepare(strokes, max_points: int = MAX_POINTS) -> np.ndarray:
    """Preprocesses one drawing. Returns a (T, 4) float32 array, T <= max_points.

    Returns an empty (0, 4) array for input with no usable points; callers
    should skip embedding those rather than feeding them to the model.
    """
    cleaned = _clean(strokes)
    if not cleaned:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    # Normalize before resampling so the spacing is in the drawing's own units
    # and means the same thing whatever the input device's coordinate scale was.
    cleaned = _normalize(cleaned)
    spacing = _spacing(cleaned, max_points)
    cleaned = [_resample(stroke, spacing) for stroke in cleaned]
    # _spacing sizes the drawing to the budget, but rounding per stroke can
    # still overshoot, so keep the hard cap as a backstop.
    cleaned = _decimate(cleaned, max_points)

    total_points = sum(len(s) for s in cleaned)
    features = np.zeros((total_points, FEATURE_DIM), dtype=np.float32)
    offset = 0
    previous = None
    for stroke in cleaned:
        points = stroke[:, :2]
        for i, point in enumerate(points):
            features[offset + i, 0:2] = point if previous is None else point - previous
            features[offset + i, 2] = stroke[i, 2]
            previous = point
        features[offset + len(points) - 1, 3] = 1.0  # pen lifts after this point
        offset += len(points)

    return features[:max_points]


def prepared_length(strokes, max_points: int = MAX_POINTS) -> int:
    """How many points prepare() will return for this drawing.

    Batching needs this, and the stored point count will not do: resampling can
    *add* points, turning a sparse 41-point QuickDraw polyline into ~139, so
    bucketing on the raw count silently overshoots the token budget.

    This used to recompute the count from the arc lengths to avoid building the
    array. It measured 11% faster and had to mirror _resample's rounding exactly
    or batches drifted out of budget with nothing to signal it, which is a poor
    trade for a pass whose result is cached anyway.
    """
    return len(prepare(strokes, max_points))

def pad_batch(sequences: list[np.ndarray], max_points: int = MAX_POINTS):
    """Pads variable-length sequences into (B, T, 4) values and a (B, T) mask.

    The mask is True where a position is padding, matching the convention
    torch's src_key_padding_mask expects.
    """
    length = min(max_points, max(1, max(len(s) for s in sequences)))
    values = np.zeros((len(sequences), length, FEATURE_DIM), dtype=np.float32)
    mask = np.ones((len(sequences), length), dtype=bool)
    for i, seq in enumerate(sequences):
        n = min(len(seq), length)
        if n:
            values[i, :n] = seq[:n]
            mask[i, :n] = False
    return values, mask
