"""
Stroke-space augmentations, applied to the app's raw stroke format before
prep.prepare() runs.

These define what the encoder is asked to treat as "the same drawing", so they
are a design decision, not a detail. Two deliberate choices:

  * rotation is kept small, a rotated page is a different note so this is
    jitter, not an invariance;
  * stroke order is never shuffled, the sequence in which strokes were drawn
    is signal a pixel-based model cannot see, and throwing it away would give
    up the main advantage of embedding strokes at all.
"""

import numpy as np


def augment(strokes, rng: np.random.Generator) -> list[np.ndarray]:
    """Returns a randomly perturbed copy of one drawing as (N, 3) arrays.

    Accepts either the app's nested-list stroke format or the numpy views a
    StrokeStore hands out; never mutates its input, so store views are safe
    to pass in directly.
    """
    arrays = [np.asarray(s, dtype=np.float32) for s in strokes if len(s)]
    if not arrays:
        return arrays

    arrays = _drop_strokes(arrays, rng)
    arrays = _affine(arrays, rng)
    arrays = _resample(arrays, rng)
    return _jitter_points(arrays, rng)


def _drop_strokes(strokes: list[np.ndarray], rng, p: float = 0.1):
    """Randomly removes strokes so a partial drawing stays near the whole."""
    if len(strokes) < 3:
        return strokes
    kept = [s for s in strokes if rng.random() > p]
    return kept if kept else strokes


def _affine(strokes: list[np.ndarray], rng, max_rotation_deg: float = 8.0):
    """Small rotation, anisotropic scale, and shear about the drawing's center."""
    xy = np.vstack([s[:, :2] for s in strokes])
    center = (xy.min(axis=0) + xy.max(axis=0)) / 2

    angle = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    shear = np.array([[1.0, rng.uniform(-0.1, 0.1)], [0.0, 1.0]], dtype=np.float32)
    scale = rng.uniform(0.85, 1.15, size=2).astype(np.float32)
    transform = (rotation @ shear) * scale

    out = []
    for stroke in strokes:
        moved = stroke.copy()
        moved[:, :2] = (stroke[:, :2] - center) @ transform.T + center
        out.append(moved)
    return out


def _resample(strokes: list[np.ndarray], rng, p: float = 0.5):
    """Randomly thins points within strokes, mimicking a different sample rate.

    Endpoints are always kept so stroke shape survives.
    """
    if rng.random() > p:
        return strokes

    keep_frac = rng.uniform(0.6, 1.0)
    out = []
    for stroke in strokes:
        if len(stroke) <= 3:
            out.append(stroke)
            continue
        keep = max(3, round(len(stroke) * keep_frac))
        idx = np.unique(np.linspace(0, len(stroke) - 1, keep).round().astype(int))
        out.append(stroke[idx])
    return out


def _jitter_points(
    strokes: list[np.ndarray], rng, sigma_frac: float = 0.004, step_frac: float = 0.1
):
    """Adds positional noise scaled to the drawing's size, capped by its density.

    The cap is what keeps this from inflating path length. Noise applied
    independently per point stretches each segment by about
    sqrt(1 + 4 (sigma/step)^2), which is nothing for QuickDraw's sparse
    polylines, where points sit ~40x sigma apart, but compounds badly for
    IAM-OnDB's ~614-point lines, where they sit closer together than sigma
    itself.

    Uncapped that inflated IAM's prepared length 1.81x, since prep resamples at
    a constant spacing and duly turned the extra wobble into 1.81x the points.
    Two things wrong with that: it made batch lengths unpredictable, and it made
    the augmentation ~10x stronger for handwriting than for doodles, which is
    the sample-rate dependence prep.py exists to remove, reintroduced one stage
    earlier.

    Capping at a fraction of the mean step holds the stretch under ~2% for any
    recording density, and leaves the sparse case exactly as it was.
    """
    xy = np.vstack([s[:, :2] for s in strokes])
    span = float(max(xy.max(axis=0) - xy.min(axis=0)))

    steps = [np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1) for s in strokes if len(s) > 1]
    sigma = max(span, 1e-6) * sigma_frac
    if steps:
        mean_step = float(np.concatenate(steps).mean())
        sigma = min(sigma, mean_step * step_frac)

    out = []
    for stroke in strokes:
        noisy = stroke.copy()
        noisy[:, :2] += rng.normal(0.0, sigma, size=stroke[:, :2].shape).astype(np.float32)
        out.append(noisy)
    return out
