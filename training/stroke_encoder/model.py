"""
The stroke encoder: a small transformer over (dx, dy, pressure, pen_up) points.

Trained with a contrastive objective, so there is no task head and no label
vocabulary anywhere in here, the encoder is domain-agnostic by construction
and sees handwriting and drawings as the same kind of input.

forward() returns a unit-length embedding, which is exactly what serving
needs, so the exported artifact runs the same code path as training. The
projection head used by the contrastive loss is a separate module and is
thrown away at export (standard SimCLR practice: the representation before
the projection transfers better than the one the loss is computed on).
"""

import torch
import torch.nn.functional as F
from torch import nn

from .prep import FEATURE_DIM, MAX_POINTS


class StrokeEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        max_points: int = MAX_POINTS,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(FEATURE_DIM, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_points, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is a no-op with norm_first=True and only emits
        # a warning, so turn it off explicitly.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 4) features. mask: (B, T) bool, True where padding.

        Returns (B, d_model) unit-length embeddings.
        """
        h = self.input_proj(x) + self.pos_emb[:, : x.size(1)]
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.norm(h)

        # Masked mean pool; clamp guards an all-padding row from dividing by 0.
        keep = (~mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, dim=-1)


class ProjectionHead(nn.Module):
    """Training-only MLP that the contrastive loss is computed on."""

    def __init__(self, d_model: int, hidden: int = 384, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE / NT-Xent over two augmented views of the same batch.

    z1[i] and z2[i] are views of the same drawing (the positive pair); every
    other drawing in the batch is a negative, which is why batch size matters
    so much for this loss.
    """
    batch = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.t()) / temperature
    # A sample must not match itself.
    sim.fill_diagonal_(float('-inf'))

    targets = torch.cat([
        torch.arange(batch, 2 * batch, device=z.device),
        torch.arange(0, batch, device=z.device),
    ])
    return F.cross_entropy(sim, targets)
