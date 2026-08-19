"""
Decides whether a segmented row is handwriting or a drawing, so TrOCR is never
asked to read a doodle.

TrOCR always returns *something*, and it is confident about it: a drawn
triangle came back as "2 1" at 0.883, higher than most genuine handwriting in
the same debug set. Its own confidence is useless as a filter, measured at AUC
0.492 over 46 rows, which is chance. Two other approaches were tried and
rejected before this one:

  * comparing log P(text | row) against log P(text | blank image), to ask
    whether the image explained the text at all. Principled and needs no extra
    model, but only reached AUC 0.859.
  * rescoring candidates with a small language model's prior. This scored 0.379,
    i.e. worse than chance, and consistently so across prompts and length
    normalizations. Language models measure how *predictable* a string is, and
    "6 0" is highly predictable precisely because it carries no information,
    while a real word is not. Surprisal is not implausibility.

SigLIP works because it is the only model here that was trained to compare
images against text descriptions, so it can be asked the question directly
rather than through a proxy. It is also already loaded for embedding, so this
costs one extra image embedding per row and no new dependency.

The prompt pair and threshold below come from 46 debug rows (6 drawings, 40
handwriting). All eight prompt pairs tried scored AUC 0.94-1.00, so the result
does not hinge on the exact wording; this pair was picked for the widest margin
between the two classes.
"""

from PIL import Image

from embedding.image import embed_image, embed_query

# Ranking a row's image between these two descriptions separates drawings from
# handwriting; the score is the difference of the two cosine similarities.
_TEXT_PROMPT = 'handwritten words'
_DRAWING_PROMPT = 'a simple drawn shape'

# Measured on the debug rows, drawings topped out at -0.0762 and handwriting
# bottomed out at -0.0181, so this sits at the midpoint of a 0.058-wide gap.
# Raise it to reject more aggressively, lower it to keep more transcriptions.
_THRESHOLD = -0.047

# Prompt vectors never change, so embed them once. Both towers of SigLIP share
# one space, which is what makes this comparison meaningful at all.
_text_vector = embed_query(_TEXT_PROMPT)['vector']
_drawing_vector = embed_query(_DRAWING_PROMPT)['vector']


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def text_score(row: Image.Image) -> float:
    """How much more like handwriting than like a drawing this row looks.

    Positive is not the decision boundary; see _THRESHOLD. Both vectors are unit
    length, so each similarity is a cosine and the difference is bounded by 2.
    """
    vector = embed_image(row)['vector']
    return _dot(vector, _text_vector) - _dot(vector, _drawing_vector)


def looks_like_text(row: Image.Image) -> tuple[bool, float]:
    """Returns (keep, score) for one row crop.

    Rows that fail are dropped before transcription rather than after, which
    also saves the ~700 ms TrOCR would spend inventing words for a drawing.
    """
    score = text_score(row)
    return score > _THRESHOLD, score
