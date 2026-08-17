"""
Image embeddings of the rendered canvas, via SigLIP.

SigLIP embeds images and text into one shared space, so a typed search query
(embed_query) can be compared directly against stored image vectors. That is
the whole reason for using it over a pure-vision model: cross-modal search
comes for free.

Vectors are unit length (cosine similarity = dot product) and only comparable
to other vectors from this same model, including across the two towers:
embed_query vectors match embed_image vectors, not text.py's.
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

_MODEL_NAME = 'google/siglip-base-patch16-224'
_LOCAL_MODEL_DIR = Path(__file__).parent.parent / 'models' / 'siglip'

# Loaded once at import from models/ if download_models.py has run, otherwise
# from HuggingFace. Same pattern as inference/transcribe.py.
_model_source = str(_LOCAL_MODEL_DIR) if _LOCAL_MODEL_DIR.exists() else _MODEL_NAME
# The bos/eos_token_id warnings printed during this load are harmless: the
# SigLIP config on the hub carries CLIP's token ids, and we never generate
# text with it, we only embed.
print(f'Loading SigLIP from {_model_source}...', flush=True)
_processor = AutoProcessor.from_pretrained(_model_source)
_model = AutoModel.from_pretrained(_model_source)
_model.eval()


def _package(out) -> dict:
    # transformers 5.x returns an output object from get_image_features /
    # get_text_features (older versions returned the pooled tensor directly);
    # the pooled embedding lives in .pooler_output.
    feats = out.pooler_output if hasattr(out, 'pooler_output') else out
    vec = feats[0]
    vec = vec / vec.norm()
    return {'model': _MODEL_NAME, 'dim': int(vec.shape[0]), 'vector': vec.tolist()}


def embed_image(image: Image.Image) -> dict:
    """Embed a canvas image as a unit-length vector."""
    inputs = _processor(images=image.convert('RGB'), return_tensors='pt')
    with torch.no_grad():
        feats = _model.get_image_features(**inputs)
    return _package(feats)


def embed_query(text: str) -> dict:
    """Embed a search query into the same space as embed_image.

    SigLIP's text tower was trained with fixed-length padding, so
    padding='max_length' is required, default padding quietly degrades
    the embeddings.
    """
    inputs = _processor(
        text=text, return_tensors='pt', padding='max_length', truncation=True,
    )
    with torch.no_grad():
        feats = _model.get_text_features(**inputs)
    return _package(feats)
