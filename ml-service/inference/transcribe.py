from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from utils.rows import segment_rows

MODEL_NAME = 'microsoft/trocr-base-handwritten'
LOCAL_MODEL_DIR = Path(__file__).parent.parent / 'models' / 'trocr-base-handwritten'

model_source = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else MODEL_NAME

device = torch.device('cpu')

processor = TrOCRProcessor.from_pretrained(model_source)
model = VisionEncoderDecoderModel.from_pretrained(model_source).to(device)


def _run_trocr(crop: Image.Image) -> str:
    pixel_values = processor(images=crop, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def transcribe(image: Image.Image, strokes: list[list[list[float]]]) -> str:
    return '\n'.join(_run_trocr(row) for row in segment_rows(image, strokes))
