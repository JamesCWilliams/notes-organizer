"""
This file takes a whole screen grab and tries to split it into "lines" of text.
Super buggy and doesn't work great, but anything with the same signature will
work downstream.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
#WORK TO BE DONE HERE.
# Set ML_DEBUG_CROPS=1 to save line crops to /tmp/ml_crops/ for inspection.
_DEBUG = os.environ.get('ML_DEBUG_CROPS') == '1'

# Fraction of the row width that must be ink for a row to count as "inked".
# e.g. 0.003 means at least 0.3% of pixels in the row have ink.
_INK_THRESHOLD_FRAC = 0.003

# A gap between lines must be at least this many consecutive blank rows.
# Prevents a single noisy row inside a gap from merging two lines.
_MIN_GAP_ROWS = 4

# Pixels of padding added on all sides of each detected line crop.
_CROP_PADDING = 8


def split_into_lines(image: Image.Image) -> list[Image.Image]:
    """
    Splits a multi-line handwritten image into single-line crops.

    Uses a horizontal projection profile to find blank gaps between lines,
    then crops each line to its tight vertical and horizontal ink bounds.
    Falls back to [image] if no line boundaries are found.
    """
    gray = np.array(image.convert('L'))     # (H, W), 0=black 255=white
    h, w = gray.shape
    ink = 255 - gray                        # ink is bright

    # Normalize row sums by width so threshold is canvas-size independent.
    ink_per_row = ink.sum(axis=1) / w       # values in [0, 255]
    blank = ink_per_row < (_INK_THRESHOLD_FRAC * 255)

    # Find contiguous runs of non-blank rows (text lines), requiring that
    # gaps between them are at least _MIN_GAP_ROWS wide.
    line_bands: list[tuple[int, int]] = []
    in_line = False
    start = 0
    gap_count = 0

    for y, is_blank in enumerate(blank):
        if not is_blank:
            if not in_line:
                in_line = True
                start = y
            gap_count = 0
        else:
            if in_line:
                gap_count += 1
                if gap_count >= _MIN_GAP_ROWS:
                    in_line = False
                    line_bands.append((start, y - gap_count))
    if in_line:
        line_bands.append((start, h))

    if not line_bands:
        return [image]

    crops = []
    for top, bottom in line_bands:
        # Crop vertically with padding.
        crop_top = max(0, top - _CROP_PADDING)
        crop_bottom = min(h, bottom + _CROP_PADDING)

        # Find horizontal ink bounds within this line band and crop those too.
        band_ink = ink[crop_top:crop_bottom, :]
        ink_per_col = band_ink.sum(axis=0)
        inked_cols = np.where(ink_per_col > 0)[0]
        if len(inked_cols):
            crop_left = max(0, int(inked_cols[0]) - _CROP_PADDING)
            crop_right = min(w, int(inked_cols[-1]) + _CROP_PADDING)
        else:
            crop_left, crop_right = 0, w

        crops.append(image.crop((crop_left, crop_top, crop_right, crop_bottom)))

    if _DEBUG:
        debug_dir = Path('debug/ml_crops')
        debug_dir.mkdir(exist_ok=True, parents=True)
        for i, crop in enumerate(crops):
            crop.save(debug_dir / f'line_{i}.png')
        print(f'[debug] {len(crops)} crop(s) saved to {debug_dir}')

    return crops
