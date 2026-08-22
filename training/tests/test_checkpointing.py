"""Pins the checkpointed training forward to the plain one export.py ships.

train_contrastive.encode reimplements StrokeEncoder.forward's layer loop so it
can wrap each layer in torch.utils.checkpoint, which is not TorchScript-able
and so cannot live in the model. That duplication is the risk: if the two ever
disagree, training optimizes something the service does not run, and nothing
anywhere raises. These tests are what make the duplication safe to keep.
"""

import torch

from stroke_encoder.model import StrokeEncoder
from stroke_encoder.prep import FEATURE_DIM
from stroke_encoder.train_contrastive import encode


def _model_and_batch(seed: int = 0):
    torch.manual_seed(seed)
    model = StrokeEncoder(d_model=32, nhead=4, num_layers=3, dim_feedforward=64)
    # eval() so dropout is off; with it on the two paths draw from the RNG in
    # a different order and would differ for reasons that are not drift.
    model.eval()

    values = torch.randn(6, 20, FEATURE_DIM)
    mask = torch.zeros(6, 20, dtype=torch.bool)
    mask[0, 12:] = True   # a padded row
    mask[1, 1:] = True    # a row down to a single real point
    return model, values, mask


def test_checkpointed_forward_matches_plain_forward():
    model, values, mask = _model_and_batch()
    with torch.no_grad():
        plain = encode(model, values, mask, grad_checkpointing=False)
        checkpointed = encode(model, values, mask, grad_checkpointing=True)
    torch.testing.assert_close(plain, checkpointed)


def test_checkpointed_forward_matches_module_call():
    """encode(..., False) must be the model's own forward, not a copy of it."""
    model, values, mask = _model_and_batch()
    with torch.no_grad():
        torch.testing.assert_close(model(values, mask), encode(model, values, mask, True))


def test_checkpointed_backward_matches_plain_gradients():
    """Recomputation has to reproduce the gradients, not just the outputs."""
    grads = {}
    for checkpointing in (False, True):
        model, values, mask = _model_and_batch()
        encode(model, values, mask, checkpointing).sum().backward()
        grads[checkpointing] = {n: p.grad.clone() for n, p in model.named_parameters()} # type: ignore

    assert grads[False].keys() == grads[True].keys()
    for name in grads[False]:
        torch.testing.assert_close(grads[False][name], grads[True][name], msg=name)


def test_all_padding_row_embeds_as_zeros():
    """An empty drawing must not poison the batch.

    Attention over a fully masked row is a softmax of all -inf; without
    attention_mask() opening the row back up that is NaN, and NaN * 0 keeps it
    NaN through the pooling mask.
    """
    model, values, mask = _model_and_batch()
    mask[2, :] = True
    with torch.no_grad():
        out = encode(model, values, mask, grad_checkpointing=True)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[2], torch.zeros_like(out[2]))


def test_all_padding_row_does_not_poison_gradients():
    """The real damage: NaN in one row reaches every shared parameter."""
    model, values, mask = _model_and_batch()
    mask[2, :] = True
    encode(model, values, mask, grad_checkpointing=True).sum().backward()

    bad = [name for name, p in model.named_parameters() if not torch.isfinite(p.grad).all()] # type: ignore
    assert not bad, f'non-finite gradients in {bad}'


def test_all_padding_row_does_not_change_its_neighbours():
    """Opening the row up must not perturb the drawings sharing its batch."""
    model, values, mask = _model_and_batch()
    with torch.no_grad():
        before = encode(model, values, mask, grad_checkpointing=True)
    mask[2, :] = True
    with torch.no_grad():
        after = encode(model, values, mask, grad_checkpointing=True)

    intact = [0, 1, 3, 4, 5]
    torch.testing.assert_close(before[intact], after[intact])
