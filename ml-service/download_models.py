"""
Downloads model weights into ml-service/models/ so they don't get
re-fetched from HuggingFace every time the container starts.

Run this once before building the docker image:
  uv run python download_models.py

Already-downloaded models are skipped, so it's safe to rerun after adding a
new model. (If a download died halfway, delete that model's folder first.)
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor, TrOCRProcessor, VisionEncoderDecoderModel

MODEL_DIR = Path(__file__).parent / 'models'


def _download_trocr(target: Path) -> None:
    name = 'microsoft/trocr-base-handwritten'
    TrOCRProcessor.from_pretrained(name).save_pretrained(target)
    VisionEncoderDecoderModel.from_pretrained(name).save_pretrained(target)


def _download_minilm(target: Path) -> None:
    SentenceTransformer('all-MiniLM-L6-v2', device='cpu').save(str(target))


def _download_siglip(target: Path) -> None:
    name = 'google/siglip-base-patch16-224'
    AutoProcessor.from_pretrained(name).save_pretrained(target)
    AutoModel.from_pretrained(name).save_pretrained(target)


# Folder names must match what the loaders expect (inference/transcribe.py
# and embedding/*.py check models/<name> before falling back to HuggingFace).
DOWNLOADERS = {
    'trocr-base-handwritten': _download_trocr,  # OCR (~1.3GB)
    'minilm': _download_minilm,                 # text embeddings (~90MB)
    'siglip': _download_siglip,                 # text+image embeddings (~800MB)
}

for name, download in DOWNLOADERS.items():
    target = MODEL_DIR / name
    if target.exists() and any(target.iterdir()):
        print(f'{name}: already downloaded, skipping')
        continue
    target.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {name} into {target}...')
    download(target)

print('Done.')
