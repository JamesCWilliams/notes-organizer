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
"""

import numpy as np

PREP_VERSION = 'v1'

# Points per drawing fed to the model. Longer inputs are decimated, shorter
# ones are padded by the caller (see pad_batch).
MAX_POINTS = 256

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


def prepare(strokes, max_points: int = MAX_POINTS) -> np.ndarray:
    """Preprocesses one drawing. Returns a (T, 4) float32 array, T <= max_points.

    Returns an empty (0, 4) array for input with no usable points; callers
    should skip embedding those rather than feeding them to the model.
    """
    cleaned = _clean(strokes)
    if not cleaned:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    cleaned = _decimate(cleaned, max_points)

    xy = np.vstack([s[:, :2] for s in cleaned])
    lo = xy.min(axis=0)
    span = float(max(xy.max(axis=0) - lo))
    # A single point, or a perfectly straight horizontal/vertical line, has
    # zero span on at least one axis; scaling by the larger span keeps aspect
    # ratio intact and avoids dividing by zero.
    scale = span if span > 1e-6 else 1.0

    features = np.zeros((len(xy), FEATURE_DIM), dtype=np.float32)
    offset = 0
    previous = None
    for stroke in cleaned:
        points = (stroke[:, :2] - lo) / scale
        for i, point in enumerate(points):
            features[offset + i, 0:2] = point if previous is None else point - previous
            features[offset + i, 2] = stroke[i, 2]
            previous = point
        features[offset + len(points) - 1, 3] = 1.0  # pen lifts after this point
        offset += len(points)

    return features[:max_points]


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
