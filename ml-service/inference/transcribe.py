from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from utils.segments import segment_words

MODEL_NAME = 'microsoft/trocr-base-handwritten'
LOCAL_MODEL_DIR = Path(__file__).parent.parent / 'models' / 'trocr-base-handwritten'

model_source = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else MODEL_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = TrOCRProcessor.from_pretrained(model_source)
model = VisionEncoderDecoderModel.from_pretrained(model_source).to(device)


# TrOCR was trained on IAM lines ~128px tall. Padding short crops to this
# height before the 384x384 resize keeps the stretch ratio consistent.
_TARGET_LINE_HEIGHT = 128


def _pad_to_height(image: Image.Image, target_h: int) -> Image.Image:
    w, h = image.size
    if h >= target_h:
        return image
    padded = Image.new('RGB', (w, target_h), (255, 255, 255))
    padded.paste(image, (0, (target_h - h) // 2))
    return padded


def _run_trocr(crop: Image.Image) -> str:
    padded = _pad_to_height(crop, _TARGET_LINE_HEIGHT)
    pixel_values = processor(images=padded, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=32)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def transcribe(image: Image.Image) -> str:
    rows = segment_words(image)
    return '\n'.join(
        ' '.join(_run_trocr(word) for word in row)
        for row in rows
    )
