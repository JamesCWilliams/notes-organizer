import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Set ML_DEBUG_CROPS=1 to save word crops to debug/ml_crops/ for inspection.
_DEBUG = os.environ.get('ML_DEBUG_CROPS') == '1'

# Pixels darker than this (out of 255) are treated as ink.
_BINARY_THRESHOLD = 200

# Dilation kernel dimensions as multiples of the estimated character height.
# Wider kernel = merges more horizontally (risk: merges adjacent words).
# Taller kernel = merges more vertically (risk: merges adjacent lines).
_KERNEL_W_FACTOR = 1.7
_KERNEL_H_FACTOR = 0.4

# Contours smaller than this area (px²) are treated as noise.
_MIN_BLOB_AREA = 50

# Two word-blobs are on the same row if their vertical overlap exceeds this
# fraction of the shorter blob's height.
_ROW_OVERLAP_FRAC = 0.4

# Pixels of padding added on all sides of each word crop.
_CROP_PADDING = 10


def segment_words(image: Image.Image) -> list[list[Image.Image]]:
    """
    Segments a handwritten image into word crops, grouped into rows.

    Returns a list of rows, where each row is a list of PIL Image crops
    ordered left-to-right. Rows are ordered top-to-bottom.
    Falls back to [[image]] if no blobs are found.
    """
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray.shape

    # Threshold: ink 255, background 0.
    _, binary = cv2.threshold(gray, _BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Get raw ink blobs to estimate character height before dilation.
    raw_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = [cv2.boundingRect(c) for c in raw_contours
                 if cv2.contourArea(c) > _MIN_BLOB_AREA]

    if not raw_boxes:
        return [[image]]

    median_h = float(np.median([h for (_, _, _, h) in raw_boxes]))

    # Dilate to merge letter strokes into word blobs.
    kw = max(1, int(median_h * _KERNEL_W_FACTOR))
    kh = max(1, int(median_h * _KERNEL_H_FACTOR))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(binary, kernel)

    # Find word-level blobs on the dilated image.
    word_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_boxes = [cv2.boundingRect(c) for c in word_contours
                  if cv2.contourArea(c) > _MIN_BLOB_AREA]

    if not word_boxes:
        return [[image]]

    # Group boxes into rows by vertical overlap, then sort rows top-to-bottom
    # and words within each row left-to-right.
    word_boxes.sort(key=lambda b: b[1])  # pre-sort by top edge
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in word_boxes:
        x, y, w, h = box
        placed = False
        for row in rows:
            # Compare against the full vertical span of the row so far.
            row_top = min(b[1] for b in row)
            row_bottom = max(b[1] + b[3] for b in row)
            overlap = min(y + h, row_bottom) - max(y, row_top)
            if overlap > _ROW_OVERLAP_FRAC * min(h, row_bottom - row_top):
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    rows.sort(key=lambda row: min(b[1] for b in row))
    for row in rows:
        row.sort(key=lambda b: b[0])

    # Crop each word from the original (undilated) image.
    result: list[list[Image.Image]] = []
    for row in rows:
        row_crops = []
        for (x, y, w, h) in row:
            left = max(0, x - _CROP_PADDING)
            top = max(0, y - _CROP_PADDING)
            right = min(img_w, x + w + _CROP_PADDING)
            bottom = min(img_h, y + h + _CROP_PADDING)
            row_crops.append(image.crop((left, top, right, bottom)))
        result.append(row_crops)

    if _DEBUG:
        debug_dir = Path('debug/ml_crops')
        debug_dir.mkdir(parents=True, exist_ok=True)
        for i, row_crops in enumerate(result):
            for j, crop in enumerate(row_crops):
                crop.save(debug_dir / f'row{i}_word{j}.png')
        total = sum(len(r) for r in result)
        print(f'[debug] {total} word crop(s) across {len(result)} row(s) saved to {debug_dir}')

    return result
