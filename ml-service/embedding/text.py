"""
Text embeddings of note transcriptions, for note-to-note and query-to-note
semantic similarity.

Vectors are unit length, so cosine similarity between two of them is a plain
dot product. Vectors are only comparable to others from the same model; the
'model' field stored alongside each vector is the guard rail.
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

_MODEL_NAME = 'all-MiniLM-L6-v2'
_LOCAL_MODEL_DIR = Path(__file__).parent.parent / 'models' / 'minilm'

# Loaded once at import from models/ if download_models.py has run, otherwise
# from HuggingFace. Same pattern as inference/transcribe.py.
_model_source = str(_LOCAL_MODEL_DIR) if _LOCAL_MODEL_DIR.exists() else _MODEL_NAME
print(f'Loading MiniLM from {_model_source}...', flush=True)
_model = SentenceTransformer(_model_source, device='cpu')


def embed_text(text: str) -> dict:
    """Embed a transcription (or a search query) as a unit-length vector.

    Callers should skip empty/whitespace-only text; embedding it produces a
    meaningless but valid-looking vector.
    """
    vec = _model.encode(text, normalize_embeddings=True)
    return {'model': _MODEL_NAME, 'dim': len(vec), 'vector': vec.tolist()}
