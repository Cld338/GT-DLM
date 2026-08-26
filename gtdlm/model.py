"""Small bidirectional Transformer models for the mechanism experiment."""

import math
from typing import Optional

import torch
from torch import nn


def immediate_gap_boundaries(
    tokens: torch.Tensor, gap_id: int, pad_id: int = 0
):
    """Return immediate left/right token ids, populated only at gap positions."""
    left = torch.full_like(tokens, pad_id)
    right = torch.full_like(tokens, pad_id)
    left[:, 1:] = tokens[:, :-1]
    right[:, :-1] = tokens[:, 1:]
    gaps = tokens == gap_id
    pad = torch.full_like(tokens, pad_id)
    return torch.where(gaps, left, pad), torch.where(gaps, right, pad), gaps


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        layers: int,
        max_positions: int,
        max_steps: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_positions, d_model)
        self.step_embedding = nn.Embedding(max_steps, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def augment_embeddings(
        self, tokens: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        return hidden

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, width = tokens.shape
        positions = torch.arange(width, device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) * self.scale
        hidden = hidden + self.position_embedding(positions)
        if steps is not None:
            hidden = hidden + self.step_embedding(steps).unsqueeze(1)
        hidden = self.augment_embeddings(tokens, hidden)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        return self.norm(hidden)


class BoundaryEncoder(Encoder):
    """Encoder that injects immediate interval anchors into gap embeddings."""

    def __init__(self, *args, gap_id: int, pad_id: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gap_id = gap_id
        self.pad_id = pad_id
        d_model = self.token_embedding.embedding_dim
        # Zero initialization makes this a residual intervention: at step zero,
        # the shared backbone is identical to the non-boundary model.
        self.left_boundary_scale = nn.Parameter(torch.zeros(d_model))
        self.right_boundary_scale = nn.Parameter(torch.zeros(d_model))

    def augment_embeddings(
        self, tokens: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        left, right, gaps = immediate_gap_boundaries(
            tokens, self.gap_id, self.pad_id
        )
        boundary = (
            self.token_embedding(left) * self.left_boundary_scale
            + self.token_embedding(right) * self.right_boundary_scale
        )
        return hidden + boundary * gaps.unsqueeze(-1)


class GapTreeModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        action_size: int,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        max_positions: int = 96,
        max_steps: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size, d_model, nhead, layers, max_positions, max_steps, dropout
        )
        self.action_head = nn.Linear(d_model, action_size)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.action_head(self.encoder(tokens, padding_mask, steps))


class GapTreeChildModel(nn.Module):
    """Gap model that predicts whether an emitted token has child gaps."""

    def __init__(
        self,
        vocab_size: int,
        action_size: int,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        max_positions: int = 96,
        max_steps: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size, d_model, nhead, layers, max_positions, max_steps, dropout
        )
        self.action_head = nn.Linear(d_model, action_size)
        self.child_head = nn.Linear(d_model, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ):
        hidden = self.encoder(tokens, padding_mask, steps)
        return self.action_head(hidden), self.child_head(hidden)


class GapTreeBoundaryModel(nn.Module):
    """Direct-child model with explicit left/right boundary features."""

    def __init__(
        self,
        vocab_size: int,
        action_size: int,
        gap_id: int,
        pad_id: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        max_positions: int = 96,
        max_steps: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BoundaryEncoder(
            vocab_size,
            d_model,
            nhead,
            layers,
            max_positions,
            max_steps,
            dropout,
            gap_id=gap_id,
            pad_id=pad_id,
        )
        self.action_head = nn.Linear(d_model, action_size)
        self.child_head = nn.Linear(d_model, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ):
        hidden = self.encoder(tokens, padding_mask, steps)
        return self.action_head(hidden), self.child_head(hidden)


class GapTreeConditionalBoundaryModel(nn.Module):
    """Boundary-aware model with child decisions conditioned on the pivot token."""

    def __init__(
        self,
        vocab_size: int,
        action_size: int,
        gap_id: int,
        pad_id: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        max_positions: int = 96,
        max_steps: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BoundaryEncoder(
            vocab_size,
            d_model,
            nhead,
            layers,
            max_positions,
            max_steps,
            dropout,
            gap_id=gap_id,
            pad_id=pad_id,
        )
        self.action_head = nn.Linear(d_model, action_size)
        self.child_head = nn.Linear(2 * d_model, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ):
        hidden = self.encoder(tokens, padding_mask, steps)
        return self.action_head(hidden), hidden

    def predict_children(
        self, hidden: torch.Tensor, chosen_tokens: torch.Tensor
    ) -> torch.Tensor:
        token_features = self.encoder.token_embedding(chosen_tokens)
        return self.child_head(torch.cat((hidden, token_features), dim=-1))


class GapTreeFactorizedBoundaryModel(nn.Module):
    """Separate STOP hazard from token identity for multimodal text gaps."""

    def __init__(
        self,
        vocab_size: int,
        gap_id: int,
        pad_id: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        max_positions: int = 96,
        max_steps: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BoundaryEncoder(
            vocab_size,
            d_model,
            nhead,
            layers,
            max_positions,
            max_steps,
            dropout,
            gap_id=gap_id,
            pad_id=pad_id,
        )
        self.token_head = nn.Linear(d_model, vocab_size)
        self.stop_head = nn.Linear(d_model, 1)
        self.child_head = nn.Linear(2 * d_model, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ):
        hidden = self.encoder(tokens, padding_mask, steps)
        return self.token_head(hidden), self.stop_head(hidden).squeeze(-1), hidden

    def predict_children(
        self, hidden: torch.Tensor, chosen_tokens: torch.Tensor
    ) -> torch.Tensor:
        token_features = self.encoder.token_embedding(chosen_tokens)
        return self.child_head(torch.cat((hidden, token_features), dim=-1))


class GapTreeJointTopologyBoundaryModel(nn.Module):
    """Factorized STOP/token heads with one joint four-way child topology."""

    def __init__(
        self,
        vocab_size: int,
        gap_id: int,
        pad_id: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        max_positions: int = 96,
        max_steps: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BoundaryEncoder(
            vocab_size,
            d_model,
            nhead,
            layers,
            max_positions,
            max_steps,
            dropout,
            gap_id=gap_id,
            pad_id=pad_id,
        )
        self.token_head = nn.Linear(d_model, vocab_size)
        self.stop_head = nn.Linear(d_model, 1)
        self.topology_head = nn.Linear(2 * d_model, 4)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ):
        hidden = self.encoder(tokens, padding_mask, steps)
        return self.token_head(hidden), self.stop_head(hidden).squeeze(-1), hidden

    def predict_topology(
        self, hidden: torch.Tensor, chosen_tokens: torch.Tensor
    ) -> torch.Tensor:
        token_features = self.encoder.token_embedding(chosen_tokens)
        return self.topology_head(torch.cat((hidden, token_features), dim=-1))


class GapTreeCoupledFrontierBoundaryModel(GapTreeJointTopologyBoundaryModel):
    """Joint per-node topology plus a non-scalable 16-way depth-1 pair head."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = self.encoder.token_embedding.embedding_dim
        self.topology_pair_head = nn.Linear(4 * d_model, 16)

    def predict_topology_pair(
        self,
        hidden_pairs: torch.Tensor,
        chosen_token_pairs: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_pairs.size(-2) != 2 or chosen_token_pairs.size(-1) != 2:
            raise ValueError("topology pair head requires exactly two gaps")
        token_features = self.encoder.token_embedding(chosen_token_pairs)
        features = torch.cat((hidden_pairs, token_features), dim=-1)
        return self.topology_pair_head(features.flatten(start_dim=-2))


class GapTreeRefinedTopologyBoundaryModel(GapTreeJointTopologyBoundaryModel):
    """One scalable topology-denoising pass over provisional frontier choices."""

    def __init__(self, *args, refinement_dim: int = 128, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = self.encoder.token_embedding.embedding_dim
        self.refinement_input = nn.Linear(2 * d_model, refinement_dim)
        self.provisional_topology_embedding = nn.Embedding(5, refinement_dim)
        self.refinement_layer = nn.TransformerEncoderLayer(
            d_model=refinement_dim,
            nhead=4,
            dim_feedforward=4 * refinement_dim,
            dropout=kwargs.get("dropout", 0.1),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refinement_norm = nn.LayerNorm(refinement_dim)
        self.refined_topology_head = nn.Linear(refinement_dim, 4)

    def refine_topology(
        self,
        hidden: torch.Tensor,
        chosen_tokens: torch.Tensor,
        provisional_topology: torch.Tensor,
        gap_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_features = self.encoder.token_embedding(chosen_tokens)
        token_features = token_features * gap_mask.unsqueeze(-1)
        features = self.refinement_input(torch.cat((hidden, token_features), dim=-1))
        provisional_features = self.provisional_topology_embedding(
            provisional_topology
        )
        features = features + provisional_features * gap_mask.unsqueeze(-1)
        features = self.refinement_layer(features, src_key_padding_mask=padding_mask)
        return self.refined_topology_head(self.refinement_norm(features))


class GapTreeBlockConditionalTopologyBoundaryModel(
    GapTreeRefinedTopologyBoundaryModel
):
    """Two-block frontier factorization with explicit conditional sampling.

    Alternating gap positions are sampled by the marginal topology head. The
    remaining positions are predicted after observing those samples. This gives
    cross-gap dependence a direct conditional-likelihood training signal.
    """

    conditional_block_topology = True
    topology_stages = 2


class GapTreeSymmetricBlockConditionalTopologyBoundaryModel(
    GapTreeBlockConditionalTopologyBoundaryModel
):
    """Block-conditional topology with both alternating orders in training."""

    symmetric_block_topology = True


class GapTreeThreeStageTopologyBoundaryModel(
    GapTreeBlockConditionalTopologyBoundaryModel
):
    """Three-stage chain-rule factorization over round-robin frontier blocks."""

    topology_stages = 3


class IntervalInsideBoundaryModel(nn.Module):
    """Tree-local action model whose latent pivots admit an inside objective."""

    def __init__(
        self,
        vocab_size: int,
        gap_id: int,
        pad_id: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        max_positions: int = 256,
        max_steps: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BoundaryEncoder(
            vocab_size,
            d_model,
            nhead,
            layers,
            max_positions,
            max_steps,
            dropout,
            gap_id=gap_id,
            pad_id=pad_id,
        )
        self.interval_projection = nn.Linear(3 * d_model, d_model)
        self.interval_norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, vocab_size)
        self.stop_head = nn.Linear(d_model, 1)
        self.topology_head = nn.Linear(2 * d_model, 4)

    def encode(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(tokens, padding_mask)

    def interval_hidden(
        self,
        context_hidden: torch.Tensor,
        left_boundary: torch.Tensor,
        right_boundary: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if depths is not None:
            context_hidden = context_hidden + self.encoder.step_embedding(depths)
        left = self.encoder.token_embedding(left_boundary)
        right = self.encoder.token_embedding(right_boundary)
        hidden = self.interval_projection(
            torch.cat((context_hidden, left, right), dim=-1)
        )
        return self.interval_norm(torch.nn.functional.gelu(hidden))

    def interval_logits(
        self,
        context_hidden: torch.Tensor,
        left_boundary: torch.Tensor,
        right_boundary: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
    ):
        hidden = self.interval_hidden(
            context_hidden, left_boundary, right_boundary, depths
        )
        return self.token_head(hidden), self.stop_head(hidden).squeeze(-1), hidden

    def topology_logits(
        self, interval_hidden: torch.Tensor, chosen_tokens: torch.Tensor
    ) -> torch.Tensor:
        token = self.encoder.token_embedding(chosen_tokens)
        return self.topology_head(torch.cat((interval_hidden, token), dim=-1))


class GapTreeSharedRegimeBoundaryModel(nn.Module):
    """Joint topology conditioned on one root-sampled shared branching regime."""

    def __init__(
        self,
        vocab_size: int,
        gap_id: int,
        pad_id: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        max_positions: int = 96,
        max_steps: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BoundaryEncoder(
            vocab_size,
            d_model,
            nhead,
            layers,
            max_positions,
            max_steps,
            dropout,
            gap_id=gap_id,
            pad_id=pad_id,
        )
        self.token_head = nn.Linear(d_model, vocab_size)
        self.stop_head = nn.Linear(d_model, 1)
        self.regime_embedding = nn.Embedding(3, d_model)
        self.topology_head = nn.Linear(3 * d_model, 4)
        # Conditional on a non-empty gap, lengths 1--2, 3--5, and 6--8 have
        # masses 2/8, 3/8, and 3/8 under the experiment's corruption prior.
        self.register_buffer(
            "regime_prior", torch.tensor([0.25, 0.375, 0.375])
        )

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ):
        hidden = self.encoder(tokens, padding_mask, steps)
        return self.token_head(hidden), self.stop_head(hidden).squeeze(-1), hidden

    def predict_topology(
        self,
        hidden: torch.Tensor,
        chosen_tokens: torch.Tensor,
        regimes: torch.Tensor,
    ) -> torch.Tensor:
        token_features = self.encoder.token_embedding(chosen_tokens)
        regime_features = self.regime_embedding(regimes)
        if regime_features.dim() == 2:
            regime_features = regime_features.unsqueeze(1).expand_as(hidden)
        return self.topology_head(
            torch.cat((hidden, token_features, regime_features), dim=-1)
        )


class LengthMaskedModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_span: int,
        d_model: int = 128,
        nhead: int = 4,
        layers: int = 3,
        max_positions: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size, d_model, nhead, layers, max_positions, 1, dropout
        )
        self.length_head = nn.Linear(d_model, max_span + 1)
        self.token_head = nn.Linear(d_model, vocab_size)

    def predict_length(self, tokens: torch.Tensor, gap_index: int = 2) -> torch.Tensor:
        hidden = self.encoder(tokens)
        return self.length_head(hidden[:, gap_index])

    def predict_tokens(
        self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.token_head(self.encoder(tokens, padding_mask))
