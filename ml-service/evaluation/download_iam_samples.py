"""
Downloads a small number of IAM handwriting line samples from HuggingFace
and saves them as .png + .txt pairs in evaluation/samples/.

Usage:
  python evaluation/download_iam_samples.py --count 20
"""

import argparse
from pathlib import Path

from datasets import load_dataset

SAMPLES_DIR = Path(__file__).parent / 'samples'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=20, help='Number of samples to download')
    args = parser.parse_args()

    SAMPLES_DIR.mkdir(exist_ok=True)

    print(f'Downloading {args.count} IAM samples...')
    dataset = load_dataset('Teklia/IAM-line', split=f'test[:{args.count}]')

    for i, sample in enumerate(dataset):

        image = sample['image']
        text = sample['text'].strip()
        stem = f'iam_{i:04d}'

        image.save(SAMPLES_DIR / f'{stem}.png')
        (SAMPLES_DIR / f'{stem}.txt').write_text(text)

        print(f'  [{i + 1}/{args.count}] {stem}: "{text}"')

    print(f'\nDone. Samples saved to {SAMPLES_DIR}')


if __name__ == '__main__':
    main()
