"""
Downloads TrOCR weights into ml-service/models/ so they don't get
re-fetched from HuggingFace every time the container starts.

Run this once before building the docker image:
  python download_model.py
"""

from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = 'microsoft/trocr-base-handwritten'
MODEL_DIR = Path(__file__).parent / 'models' / 'trocr-base-handwritten'

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f'Downloading {MODEL_NAME} into {MODEL_DIR}...')
TrOCRProcessor.from_pretrained(MODEL_NAME).save_pretrained(MODEL_DIR)
VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).save_pretrained(MODEL_DIR)
print('Done.')
