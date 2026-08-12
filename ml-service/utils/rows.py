import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Pressure assigned to points that arrive as bare [x, y]
_DEFAULT_PRESSURE = 0.5

# Strokes shorter than this (px) are ignored when estimating the handwriting
# scale, i-dots and periods would drag the median down.
_MIN_SCALE_HEIGHT = 2.0

# The thresholds below are multiples of the handwriting scale (median stroke
# height), so they hold regardless of canvas resolution or writing size.

# A carriage return is a leftward jump of at least this
_LEFT_JUMP_FACTOR = 2.5
# combined with a downward shift of at least this.
_DOWN_SHIFT_FACTOR = 0.5
# A downward shift this large splits a row on its own, so a new line that
# starts further right than a short previous line ended still triggers
_DOWN_ONLY_FACTOR = 1.5

# Pixels of padding added on all sides of each row crop. Also covers the
# rendered stroke width, which extends beyond the raw point coordinates
# (perfect-freehand draws ~8px around each point at the frontend's size 16)
_CROP_PADDING = 12


class StrokeDataError(ValueError):
    """Raised when the strokes payload is malformed."""


@dataclass
class Stroke:
    """One pen stroke: an (N, 3) float64 array of x, y, pressure per point."""
    points: np.ndarray

    @property
    def xy(self) -> np.ndarray:
        return self.points[:, :2]

    @property
    def pressure(self) -> np.ndarray:
        return self.points[:, 2]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) in canvas pixels."""
        x_min, y_min = self.xy.min(axis=0)
        x_max, y_max = self.xy.max(axis=0)
        return float(x_min), float(y_min), float(x_max), float(y_max)

    @property
    def height(self) -> float:
        _, y_min, _, y_max = self.bbox
        return y_max - y_min

    @property
    def center_y(self) -> float:
        _, y_min, _, y_max = self.bbox
        return (y_min + y_max) / 2


def load_strokes(raw: list[list[list[float]]]) -> list[Stroke]:
    """
    Converts the raw strokes payload into Stroke objects.

    Points may be [x, y, pressure] or bare [x, y] (pressure defaults).
    Empty strokes are dropped; single-point strokes (dots) are kept.
    Raises StrokeDataError on anything malformed or non-finite.
    """
    if not isinstance(raw, list):
        raise StrokeDataError('strokes must be a list of strokes')

    strokes: list[Stroke] = []
    for i, raw_stroke in enumerate(raw):
        if not isinstance(raw_stroke, list):
            raise StrokeDataError(f'stroke {i} must be a list of points')
        if not raw_stroke:
            continue

        points = np.empty((len(raw_stroke), 3), dtype=np.float64)
        for j, point in enumerate(raw_stroke):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise StrokeDataError(
                    f'stroke {i} point {j} must be [x, y] or [x, y, pressure]')
            try:
                points[j, 0] = point[0]
                points[j, 1] = point[1]
                points[j, 2] = point[2] if len(point) > 2 else _DEFAULT_PRESSURE
            except (TypeError, ValueError):
                raise StrokeDataError(
                    f'stroke {i} point {j} has non-numeric values') from None

        if not np.isfinite(points).all():
            raise StrokeDataError(f'stroke {i} contains non-finite values')
        strokes.append(Stroke(points))

    return strokes


def measure_handwriting_scale(strokes: list[Stroke]) -> float:
    """Median stroke height, ignoring near-zero-height strokes like dots."""
    heights = [s.height for s in strokes if s.height >= _MIN_SCALE_HEIGHT]
    if not heights:
        return 1.0
    return float(np.median(heights))


def overlaps_row_band(stroke: Stroke, row: list[Stroke]) -> bool:
    """True if the stroke vertically overlaps the row's band by at least half
    the stroke's own height (dots count as overlapping on any contact)."""
    band_top = min(s.bbox[1] for s in row)
    band_bottom = max(s.bbox[3] for s in row)
    _, top, _, bottom = stroke.bbox
    overlap = min(bottom, band_bottom) - max(top, band_top)
    return overlap > 0.5 * max(bottom - top, 1.0)


def split_into_rows(strokes: list[Stroke], scale: float) -> list[list[Stroke]]:
    """Splits temporally ordered strokes into rows at "carriage returns":
    a big right-to-left jump between consecutive strokes plus a downward
    shift means the pen moved to a new line."""
    rows = [[strokes[0]]]
    for stroke in strokes[1:]:
        prev = rows[-1][-1]
        dx = stroke.bbox[0] - prev.bbox[2] # negative = jumped left
        dy = stroke.center_y - prev.center_y # positive = moved down

        carriage_return = (
            (dx < -_LEFT_JUMP_FACTOR * scale and dy > _DOWN_SHIFT_FACTOR * scale) # x goes right-to-left and y goes down
            or dy > _DOWN_ONLY_FACTOR * scale # only y goes down
        )
        # Delayed i-dots / t-crosses / fix-ups land back inside the current
        # row's band, keep them there regardless of the jump
        if carriage_return and not overlaps_row_band(stroke, rows[-1]):
            rows.append([stroke])
        else:
            rows[-1].append(stroke)

    # Temporal order is usually top-to-bottom already, but don't trust it
    rows.sort(key=lambda row: min(s.bbox[1] for s in row))
    return rows


def segment_rows(image: Image.Image, strokes: list[list[list[float]]]) -> list[Image.Image]:
    """
    Segments a handwritten image into one crop per row of text using the
    stroke data. TrOCR was trained on line-level images, so whole rows
    match its training distribution better than word crops.

    Returns PIL Image crops ordered top-to-bottom.
    Falls back to [image] if there are no usable strokes.
    """
    loaded = load_strokes(strokes)
    if not loaded:
        return [image]

    scale = measure_handwriting_scale(loaded)
    img_w, img_h = image.size
    result: list[Image.Image] = []
    for row in split_into_rows(loaded, scale):
        x_min = min(s.bbox[0] for s in row)
        y_min = min(s.bbox[1] for s in row)
        x_max = max(s.bbox[2] for s in row)
        y_max = max(s.bbox[3] for s in row)

        left = max(0, int(x_min) - _CROP_PADDING)
        top = max(0, int(y_min) - _CROP_PADDING)
        right = min(img_w, int(x_max) + _CROP_PADDING)
        bottom = min(img_h, int(y_max) + _CROP_PADDING)
        
        result.append(image.crop((left, top, right, bottom)))

    return result
