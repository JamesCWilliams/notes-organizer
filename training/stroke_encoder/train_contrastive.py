"""
Self-supervised contrastive pretraining for the stroke encoder.

  uv run python -m stroke_encoder.train_contrastive --epochs 20 --batch-size 512

Batch size is the knob that matters most: every other drawing in the batch is
a negative, so bigger batches make the task harder and the embeddings better.
Gradient accumulation does NOT substitute for it here.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    ContrastiveStrokes,
    LengthBucketSampler,
    StrokeStore,
    iter_iam_ondb,
    iter_quickdraw,
)
from .model import ProjectionHead, StrokeEncoder, nt_xent
from .prep import MAX_POINTS, PREP_VERSION, pad_batch

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / 'checkpoints'


def collate(batch):
    """Pads both views into (2B, T, 4) values and mask, views stacked in order."""
    views = [view for pair in batch for view in pair]
    values, mask = pad_batch(views, MAX_POINTS)
    return torch.from_numpy(values), torch.from_numpy(mask)


def balance(spans, handwriting_share: float) -> np.ndarray:
    """Builds a sample index that oversamples handwriting to a target share.

    IAM-OnDB is ~12k lines against QuickDraw's hundreds of thousands, so left
    alone the encoder would see almost nothing but doodles, while the app it
    serves is mostly handwriting. Repeating indices costs nothing: the stored
    points are untouched and each repeat is augmented differently.
    """
    doodles, handwriting = spans['quickdraw'], spans['iam']
    if not len(handwriting) or handwriting_share <= 0:
        return np.asarray(doodles, dtype=np.int64)
    if not len(doodles):
        return np.asarray(handwriting, dtype=np.int64)

    # Solve repeats * len(hw) / (repeats * len(hw) + len(qd)) = share
    wanted = handwriting_share * len(doodles) / (1.0 - handwriting_share)
    repeats = max(1, round(wanted / len(handwriting)))
    index = np.concatenate([
        np.asarray(doodles, dtype=np.int64),
        np.tile(np.asarray(handwriting, dtype=np.int64), repeats),
    ])
    actual = repeats * len(handwriting) / len(index)
    print(f'handwriting oversampled {repeats}x -> {actual:.0%} of samples')
    return index


def run_epoch(model, head, loader, optimizer, scaler, device, temperature, train=True):
    model.train(train)
    head.train(train)
    total, batches = 0.0, 0

    for values, mask in tqdm(loader, leave=False, desc='train' if train else 'val'):
        values, mask = values.to(device, non_blocking=True), mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(train), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'
        ):
            z = head(model(values, mask))
            # Views were interleaved by collate: [a0, b0, a1, b1, ...]
            loss = nt_xent(z[0::2], z[1::2], temperature)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()

        total += loss.item()
        batches += 1

    return total / max(batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--token-budget', type=int, default=260_000,
                        help='padded points per batch across both views. Batch '
                             'size floats with sequence length to keep VRAM '
                             'flat; 260k peaks near 10 GB. Raising it past what '
                             'the card holds does not fail loudly, it spills to '
                             'shared memory and gets several times SLOWER')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--d-model', type=int, default=192)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit-per-category', type=int, default=None,
                        help='cap drawings loaded per QuickDraw category (for quick runs)')
    parser.add_argument('--handwriting-share', type=float, default=0.35,
                        help='target fraction of samples drawn from IAM-OnDB; '
                             'it is far smaller than QuickDraw, so it gets '
                             'oversampled to keep handwriting from being drowned out')
    parser.add_argument('--name', default='stroke-encoder-v2',
                        help='checkpoint name; also becomes the served model name')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if device.type == 'cpu':
        print('WARNING: contrastive training on CPU will be slow; large batches are the point.')

    store, spans = StrokeStore.build({
        'quickdraw': iter_quickdraw(args.limit_per_category),
        'iam': iter_iam_ondb(),
    })
    if not len(store):
        raise SystemExit(
            'No training data found. Run:\n'
            '  uv run python -c "from stroke_encoder.data import fetch_quickdraw; fetch_quickdraw()"'
        )
    print(
        f'{len(store)} drawings packed into {store.nbytes() / 1e6:.0f} MB '
        f'({len(spans["quickdraw"])} quickdraw, {len(spans["iam"])} iam)'
    )

    sample_index = balance(spans, args.handwriting_share)
    counts = np.minimum(store.point_counts(), MAX_POINTS)

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(sample_index))
    val_size = max(1, int(0.02 * len(sample_index)))
    splits = {'val': shuffled[:val_size], 'train': shuffled[val_size:]}

    loaders = {}
    for split, positions in splits.items():
        indices = sample_index[positions]
        # Each batch carries two views, so the token budget is split in half.
        sampler = LengthBucketSampler(
            counts[indices], args.token_budget // 2, shuffle=split == 'train'
        )
        loaders[split] = DataLoader(
            ContrastiveStrokes(store, indices),
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=device.type == 'cuda',
            persistent_workers=args.workers > 0,
        )
    train_loader, val_loader = loaders['train'], loaders['val']
    print(f'{len(train_loader)} train batches/epoch at {args.token_budget:,} tokens')

    model = StrokeEncoder(d_model=args.d_model, num_layers=args.layers).to(device)
    head = ProjectionHead(args.d_model).to(device)
    print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M encoder params')

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=0.05
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best = float('inf')

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = run_epoch(model, head, train_loader, optimizer, scaler, device, args.temperature, True)
        val_loss = run_epoch(model, head, val_loader, optimizer, scaler, device, args.temperature, False)
        schedule.step()
        print(f'epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  ({time.time() - started:.0f}s)')

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    'name': args.name,
                    'prep_version': PREP_VERSION,
                    'config': {'d_model': args.d_model, 'num_layers': args.layers},
                    'encoder': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                },
                CHECKPOINT_DIR / f'{args.name}.pt',
            )
            print(f'  saved {CHECKPOINT_DIR / f"{args.name}.pt"}')


if __name__ == '__main__':
    main()
