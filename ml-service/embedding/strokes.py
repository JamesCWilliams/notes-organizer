"""
Stroke embeddings: what the note looked like to draw, rather than what it says
(text.py) or how the page looks (image.py).

The model is a TorchScript artifact exported from the training project, loaded
with torch.jit.load, no model class is needed here, which is what lets the
service stay on CPU-only torch while training uses CUDA.

Preprocessing is imported from the training project rather than reimplemented.
That import path is load-bearing: a note embedded here must be preprocessed
exactly the way training data was, and a divergence would degrade embeddings
silently. See training/README.md, "the preprocessing contract".

Unlike the other two encoders, this one is optional: if no model has been
exported yet, embed_strokes returns None and the service runs without it.
"""

import json
import sys
from pathlib import Path

import torch

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_ROOT = _SERVICE_ROOT.parent / 'training'

# Newest export wins, so retraining to stroke-encoder-v3 is picked up by
# dropping the folder in, no code change.
_CANDIDATES = sorted((_SERVICE_ROOT / 'models').glob('stroke-encoder-*'), reverse=True)
_MODEL_DIR = _CANDIDATES[0] if _CANDIDATES else None

_model = None
_meta: dict = {}
prepare = None
pad_batch = None

if _MODEL_DIR is None:
    print('No stroke encoder exported; stroke embeddings disabled.', flush=True)
elif not _TRAINING_ROOT.exists():
    print(f'{_TRAINING_ROOT} missing; stroke embeddings disabled.', flush=True)
else:
    if str(_TRAINING_ROOT) not in sys.path:
        sys.path.insert(0, str(_TRAINING_ROOT))
    from stroke_encoder.prep import PREP_VERSION, pad_batch, prepare

    _meta = json.loads((_MODEL_DIR / 'meta.json').read_text())
    if _meta.get('prep_version') != PREP_VERSION:
        raise RuntimeError(
            f'{_MODEL_DIR.name} was trained with prep {_meta.get("prep_version")}, '
            f'but training/ is now {PREP_VERSION}. Re-export or retrain, '
            f'embedding with mismatched preprocessing fails silently.'
        )

    print(f'Loading stroke encoder from {_MODEL_DIR}...', flush=True)
    _model = torch.jit.load(str(_MODEL_DIR / 'model.pt'), map_location='cpu')
    _model.eval()


def embed_strokes(strokes) -> dict | None:
    """Embed raw strokes as a unit-length vector.

    Returns None when no encoder is available, or when the strokes contain no
    usable points.
    """
    if _model is None or prepare is None or pad_batch is None:
        return None

    features = prepare(strokes)
    if not len(features):
        return None

    values, mask = pad_batch([features], _meta.get('max_points', 256))
    with torch.no_grad():
        vector = _model(torch.from_numpy(values), torch.from_numpy(mask))[0]

    return {
        'model': _meta['model'],
        'dim': int(vector.shape[0]),
        'vector': vector.tolist(),
    }
