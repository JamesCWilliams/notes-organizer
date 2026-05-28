"""
Computes Word Error Rate (WER) over a set of sample images.

Sample format — evaluation/samples/
  some_sample.png        the handwriting image
  some_sample.txt        the ground truth transcription (same stem)

Usage:
  python evaluation/evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jiwer import wer
from PIL import Image

from inference.transcribe import transcribe

SAMPLES_DIR = Path(__file__).parent / 'samples'


def load_samples() -> list[tuple[Image.Image, str]]:
    samples = []
    for img_path in sorted(SAMPLES_DIR.glob('*.png')):
        txt_path = img_path.with_suffix('.txt')
        if not txt_path.exists():
            continue
        image = Image.open(img_path).convert('RGB')
        ground_truth = txt_path.read_text().strip()
        samples.append((image, ground_truth))
    return samples


def main():
    samples = load_samples()
    if not samples:
        print('No samples found in evaluation/samples/. Add .png + .txt pairs.')
        return

    predictions = [transcribe(img) for img, _ in samples]
    ground_truths = [gt for _, gt in samples]

    score = wer(ground_truths, predictions)
    print(f'Samples evaluated : {len(samples)}')
    print(f'Word Error Rate   : {score:.2%}')

    for (_, gt), pred in zip(samples, predictions):
        print(f'\n  expected : {gt}')
        print(f'  got      : {pred}')


if __name__ == '__main__':
    main()
