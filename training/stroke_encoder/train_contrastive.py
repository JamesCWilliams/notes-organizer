"""
Self-supervised contrastive pretraining for the stroke encoder.

  uv run python -m stroke_encoder.train_contrastive --epochs 100

Batch size is the knob that matters most: every other drawing in the batch is
a negative, so bigger batches make the task harder and the embeddings better.
Naive gradient accumulation does NOT substitute for it here, since the
negatives have to be inside one loss to push against each other. There is no
--batch-size, batches are sized by --token-budget instead; see the README.

Everything that buys batch size is therefore on by default: gradient
checkpointing trades recomputation for the encoder activations, and the token
budget is set from measured peak VRAM rather than guessed.
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    ContrastiveStrokes,
    LengthBucketSampler,
    cached_lengths,
    load_or_build_store,
)
from .model import ProjectionHead, StrokeEncoder, nt_xent
from .prep import MAX_POINTS, PREP_VERSION, pad_batch

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / 'checkpoints'


def encode(model, values, mask, grad_checkpointing: bool):
    """model(values, mask), optionally recomputing layers in the backward pass.

    Checkpointing lives here rather than in StrokeEncoder because export.py
    scripts the model and torch.utils.checkpoint is not scriptable. Keeping it
    on the training side means the forward the service runs is exactly the one
    that was trained, the same contract prep.py is held to.

    The layer loop mirrors nn.TransformerEncoder, and is equivalent only
    because this encoder has norm=None and enable_nested_tensor=False, so there
    is no final norm and no fast path to skip. tests/ pins the two together.
    """
    if not grad_checkpointing:
        return model(values, mask)

    h = model.embed_points(values)
    attention_mask = model.attention_mask(mask)
    for layer in model.encoder.layers:
        # positional args: src, src_mask, src_key_padding_mask
        h = checkpoint(layer, h, None, attention_mask, use_reentrant=False)
    return model.pool(h, mask)


def make_collate(token_budget: int):
    """Builds the collate_fn, closing over the budget it has to keep.

    LengthBucketSampler can only predict a batch's padded length, from lengths
    measured on sampled augmentations, and a batch's length is the maximum over
    every drawing in it, drawn fresh each epoch. Here the true length is known,
    so here the budget becomes a guarantee rather than an intention.

    Overshooting batches are trimmed by dropping whole drawings, both views
    together, uniformly at random. Random rather than longest-first, which
    would systematically thin out long handwriting lines and quietly reshape
    what the encoder trains on. NT-Xent minds only that pairs stay intact, not
    that N varies between batches.
    """
    def collate(batch):
        """Pads both views into (2B, T, 4) values and mask, views stacked in order."""
        longest = min(MAX_POINTS, max(1, max(len(v) for pair in batch for v in pair)))
        affordable = max(1, token_budget // (2 * longest))
        if len(batch) > affordable:
            batch = [batch[i] for i in np.random.choice(len(batch), affordable, replace=False)]

        views = [view for pair in batch for view in pair]
        values, mask = pad_batch(views, MAX_POINTS)
        return torch.from_numpy(values), torch.from_numpy(mask)

    return collate


def seed_everything(seed: int) -> None:
    """Fixes weight init and every random stream the main process draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Gives each DataLoader worker its own reproducible numpy stream.

    torch reseeds workers itself but does not cover numpy's global RNG, which
    augment() reaches through ContrastiveStrokes. Left alone, forked workers
    share one stream and the augmentations stop being reproducible however
    carefully the main process was seeded. torch.initial_seed() already differs
    per worker, so deriving from it keeps them independent and tied to --seed.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


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


def run_epoch(model, head, loader, optimizer, device, temperature,
              train=True, grad_checkpointing=False):
    model.train(train)
    head.train(train)
    total, batches = 0.0, 0

    for values, mask in tqdm(loader, leave=False, desc='train' if train else 'val'):
        values, mask = values.to(device, non_blocking=True), mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(train), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'
        ):
            # Checkpointing only pays off where there is a backward pass to
            # recompute for; under no_grad it would be pure overhead.
            z = head(encode(model, values, mask, grad_checkpointing and train))
            # Views were interleaved by collate: [a0, b0, a1, b1, ...]
            loss = nt_xent(z[0::2], z[1::2], temperature)

        if train:
            optimizer.zero_grad(set_to_none=True)
            # No GradScaler: loss scaling exists to keep fp16's narrow exponent
            # range from flushing small gradients to zero, and bfloat16 has the
            # same exponent range as fp32, so there is nothing to rescue.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0
            )
            optimizer.step()

        total += loss.item()
        batches += 1

    return total / max(batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--token-budget', type=int, default=1_000_000,
                        help='padded points per batch across both views; ~1.1 GB '
                             'of VRAM per 100k. Re-measure after changing prep.py. '
                             'See the README on tuning it')
    parser.add_argument('--max-batch', type=int, default=8192,
                        help='hard cap on drawings per batch. The budget bounds '
                             'drawings x length, not the count, and NT-Xent grows '
                             'with the square of the count')
    parser.add_argument('--grad-checkpointing', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='recompute encoder layers in the backward pass rather '
                             'than storing activations: 2.5x the batch for 31%% '
                             'more time per step')
    parser.add_argument('--vram-fraction', type=float, default=0.85,
                        help='cap the allocator at this fraction of the card, so '
                             'an over-budget batch raises instead of spilling to '
                             'shared memory and running slowly. 0 disables')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--d-model', type=int, default=192)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=10,
                        help='seeds weight init, shuffling and augmentation, so '
                             'rerunning the same config yields the same encoder')
    parser.add_argument('--limit-per-category', type=int, default=None,
                        help='cap drawings loaded per QuickDraw category (for quick runs)')
    parser.add_argument('--handwriting-share', type=float, default=0.35,
                        help='target fraction of samples drawn from IAM-OnDB, which '
                             'is far smaller than QuickDraw and gets oversampled to '
                             'keep handwriting from being drowned out')
    parser.add_argument('--name', default='stroke-encoder-v3',
                        help='checkpoint name; also becomes the served model name')
    args = parser.parse_args()

    # Batch shapes vary enormously here, which is the pattern that fragments
    # CUDA's caching allocator: measured 1.33x reserved-over-allocated without
    # this, 1.11x with it, worth ~0.8 GB. Set before the first allocation, and
    # only if the caller has not asked for something else.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}  seed: {args.seed}')
    if device.type == 'cpu':
        print('WARNING: contrastive training on CPU will be slow; large batches are the point.')
    elif args.vram_fraction:
        # The desktop holds a couple of GB of the same card, so the usable
        # ceiling is well under what nvidia-smi advertises. Capping the
        # allocator makes an over-budget batch raise here rather than spilling
        # to shared memory and running at a fraction of the speed.
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'allocator capped at {args.vram_fraction:.0%} of {total:.1f} GB '
              f'= {args.vram_fraction * total:.1f} GB')

    store, spans = load_or_build_store(args.limit_per_category)
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
    counts = cached_lengths(store, args.limit_per_category)

    # Seeded so the same run always holds out the same drawings for validation;
    # otherwise val loss would not be comparable between runs either.
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(sample_index))
    val_size = max(1, int(0.02 * len(sample_index)))
    splits = {'val': shuffled[:val_size], 'train': shuffled[val_size:]}

    loaders = {}
    for split, positions in splits.items():
        indices = sample_index[positions]
        # Each batch carries two views, so the token budget is split in half.
        sampler = LengthBucketSampler(
            counts[indices], args.token_budget // 2,
            max_drawings=args.max_batch,
            shuffle=split == 'train', seed=args.seed,
        )
        loaders[split] = DataLoader(
            ContrastiveStrokes(store, indices, seed=args.seed),
            batch_sampler=sampler,
            collate_fn=make_collate(args.token_budget),
            num_workers=args.workers,
            worker_init_fn=seed_worker,
            pin_memory=device.type == 'cuda',
            persistent_workers=args.workers > 0,
        )
    train_loader, val_loader = loaders['train'], loaders['val']
    print(f'{len(train_loader)} train batches/epoch at {args.token_budget:,} tokens'
          f'{", grad checkpointing" if args.grad_checkpointing else ""}')

    model = StrokeEncoder(d_model=args.d_model, num_layers=args.layers).to(device)
    head = ProjectionHead(args.d_model).to(device)
    print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M encoder params')

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=0.05
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best = float('inf')

    drawings_per_epoch = sum(len(positions) for positions in splits.values())

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        train_loss = run_epoch(model, head, train_loader, optimizer, device,
                               args.temperature, True, args.grad_checkpointing)
        val_loss = run_epoch(model, head, val_loader, optimizer, device,
                             args.temperature, False, args.grad_checkpointing)
        schedule.step()

        elapsed = time.time() - started
        # Peak and throughput together are what --token-budget is tuned against.
        usage = ''
        if device.type == 'cuda':
            usage = (f'  peak {torch.cuda.max_memory_reserved() / 1e9:.1f} GB'
                     f'  {drawings_per_epoch / max(elapsed, 1e-9):,.0f} drawings/s')
        print(
            f'epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  '
            f'({elapsed:.0f}s){usage}'
        )

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    'name': args.name,
                    'prep_version': PREP_VERSION,
                    'seed': args.seed,
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
