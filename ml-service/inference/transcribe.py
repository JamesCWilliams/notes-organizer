from datetime import datetime
from pathlib import Path
import json
from time import perf_counter_ns

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput

from utils.rows import segment_rows

MODEL_NAME = 'microsoft/trocr-base-handwritten'
LOCAL_MODEL_DIR = Path(__file__).parent.parent / 'models' / 'trocr-base-handwritten'
DEBUG_ROOT = Path(__file__).parent.parent / 'debug' / 'transcription'

model_source = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else MODEL_NAME

device = torch.device('cpu')

processor = TrOCRProcessor.from_pretrained(model_source)
model = VisionEncoderDecoderModel.from_pretrained(model_source)

# Beams searched (and kept) per row
_NUM_BEAMS = 5


def _run_trocr(crop: Image.Image) -> list[dict]:
    """Beam-search decode one row crop.

    Returns one candidate per beam, best first: {'text', 'confidence'}.
    Confidence is exp of the length-normalized sequence log-prob, i.e. the
    geometric mean token probability, in (0, 1].
    """
    inputs = processor(images=crop, return_tensors='pt')  # type: ignore
    pixel_values = inputs['pixel_values'].to(device)
    out = model.generate(  # type: ignore
        pixel_values,
        max_new_tokens=128,
        num_beams=_NUM_BEAMS,
        num_return_sequences=_NUM_BEAMS,
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert isinstance(out, GenerateBeamEncoderDecoderOutput)
    assert out.sequences_scores is not None

    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    confidences = out.sequences_scores.exp().tolist()
    candidates = [
        {'text': text, 'confidence': round(confidence, 4)}
        for text, confidence in zip(texts, confidences)
    ]

    candidates.sort(key=lambda c: c['confidence'], reverse=True)
    return candidates


def _save_debug_run(
    image: Image.Image,
    strokes: list[list[list[float]]],
    rows: list[Image.Image],
    row_candidates: list[list[dict]],
    text: str,
    seg_time: float,
    row_times: list[float]
) -> None:
    """Saves one request's inputs and outputs to a timestamped run directory,
    pairing row_<i>.png with predictions.json['per row'][i]."""
    run_dir = DEBUG_ROOT / datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    run_dir.mkdir(parents=True, exist_ok=True)

    image.save(run_dir / 'input_image.png')
    with open(run_dir / 'strokes.json', 'w') as file:
        json.dump(strokes, file, indent=4)

    for i, row in enumerate(rows):
        row.save(run_dir / f'row_{i}.png')

    summary = {
        'per row': [
            {
                'prediction': candidates[0]['text'],
                'confidence': candidates[0]['confidence'],
                'transcription time': f'{row_times[i]:.6f} ms',
                'candidates': candidates
            }
            for (i, candidates) in enumerate(row_candidates)
        ],
        'joined': text,
        'segmentation time': f'{seg_time:.6f} ms',
        'total elapsed time': f'{seg_time + sum(row_times):.6f} ms'
    }
    with open(run_dir / 'predictions.json', 'w') as file:
        json.dump(summary, file, indent=4)


def transcribe(image: Image.Image, strokes: list[list[list[float]]], debug: bool = False) -> str:
    segment_rows_t0 = perf_counter_ns()
    rows = segment_rows(image, strokes)
    segment_rows_t1 = perf_counter_ns()
    seg_time = (segment_rows_t1 - segment_rows_t0) / 1e6

    row_times = []
    row_candidates = []
    for row in rows:
        this_row_t0 = perf_counter_ns()
        this_row_candidates = _run_trocr(row)
        this_row_t1 = perf_counter_ns()
        this_row_time = (this_row_t1 - this_row_t0) / 1e6
        row_times.append(this_row_time)
        row_candidates.append(this_row_candidates)

    text = '\n'.join(candidates[0]['text'] for candidates in row_candidates)

    if debug:
        _save_debug_run(image, strokes, rows, row_candidates, text, seg_time, row_times)

    return text
