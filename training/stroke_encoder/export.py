"""
Exports a trained checkpoint into ml-service as a TorchScript artifact.

  uv run python -m stroke_encoder.export --name stroke-encoder-v2

TorchScript is what keeps the two projects decoupled: ml-service loads the
result with torch.jit.load and never imports this package's model code, so
training can use CUDA torch while the service stays on the CPU wheels.

The projection head is deliberately not exported, the contrastive loss was
computed on it, but the encoder output is the representation we serve.
"""

import argparse
import json
from pathlib import Path

import torch

from .model import StrokeEncoder
from .prep import FEATURE_DIM, MAX_POINTS, PREP_VERSION

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / 'checkpoints'
SERVING_DIR = Path(__file__).resolve().parents[2] / 'ml-service' / 'models'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='stroke-encoder-v2')
    args = parser.parse_args()

    checkpoint = torch.load(CHECKPOINT_DIR / f'{args.name}.pt', map_location='cpu')
    if checkpoint['prep_version'] != PREP_VERSION:
        raise SystemExit(
            f'checkpoint was trained with prep {checkpoint["prep_version"]}, '
            f'this tree is {PREP_VERSION}, retrain before exporting.'
        )

    model = StrokeEncoder(**checkpoint['config'])
    model.load_state_dict(checkpoint['encoder'])
    model.eval()

    target = SERVING_DIR / args.name
    target.mkdir(parents=True, exist_ok=True)

    scripted = torch.jit.script(model)
    scripted.save(str(target / 'model.pt'))

    # Sanity check the artifact in isolation, the way the service will load it.
    reloaded = torch.jit.load(str(target / 'model.pt'), map_location='cpu')
    values = torch.zeros(1, 8, FEATURE_DIM)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    with torch.no_grad():
        out = reloaded(values, mask)

    (target / 'meta.json').write_text(json.dumps({
        'model': args.name,
        'dim': int(out.shape[-1]),
        'prep_version': PREP_VERSION,
        'max_points': MAX_POINTS,
        'trained_epoch': checkpoint.get('epoch'),
        'val_loss': checkpoint.get('val_loss'),
    }, indent=2) + '\n')

    print(f'exported {args.name} (dim {out.shape[-1]}) to {target}')


if __name__ == '__main__':
    main()
