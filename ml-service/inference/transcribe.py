from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_NAME = 'microsoft/trocr-base-handwritten'
LOCAL_MODEL_DIR = Path(__file__).parent.parent / 'models' / 'trocr-base-handwritten'

model_source = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else MODEL_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = TrOCRProcessor.from_pretrained(model_source)
model = VisionEncoderDecoderModel.from_pretrained(model_source).to(device)


def transcribe(image: Image.Image) -> str:
    pixel_values = processor(images=image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
