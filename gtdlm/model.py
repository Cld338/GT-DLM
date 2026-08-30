"""Small bidirectional Transformer models for the mechanism experiment."""

import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


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


class PretrainedIntervalEncoder(nn.Module):
    """Masked-language prompt encoder with interval boundary states.

    The pretrained backbone runs once per observed prompt. Only its mask-token
    state is needed by the interval chart; all latent tree states then reuse
    that context and differ through boundary and depth embeddings. The native
    vocabulary path feeds pretrained token ids directly, avoiding the lossy
    decode/re-tokenize bridge used by the historical custom-BPE experiments.
    """

    def __init__(
        self,
        vocab_size: int,
        gap_id: int,
        pad_id: int,
        source_tokenizer,
        model_name: str,
        cache_dir: str,
        max_steps: int = 32,
        max_length: int = 256,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
        random_init_backbone: bool = False,
        backbone=None,
        pretrained_tokenizer=None,
        initialize_custom_embeddings: bool = True,
        native_vocabulary: bool = False,
        fixed_mask_count: int = 0,
    ) -> None:
        super().__init__()
        if backbone is None or pretrained_tokenizer is None:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            pretrained_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_fast=True,
                local_files_only=local_files_only,
            )
            if random_init_backbone:
                config = AutoConfig.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                backbone = AutoModel.from_config(config)
            else:
                backbone = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
        if pretrained_tokenizer.mask_token_id is None:
            raise ValueError("pretrained tokenizer must define a mask token")
        self.backbone = backbone
        self.pretrained_tokenizer = pretrained_tokenizer
        self.source_tokenizer = source_tokenizer
        self.gap_id = gap_id
        self.pad_id = pad_id
        self.max_length = max_length
        self.native_vocabulary = native_vocabulary
        self.fixed_mask_count = int(fixed_mask_count)
        if self.fixed_mask_count < 0:
            raise ValueError("fixed_mask_count must be nonnegative")
        if self.fixed_mask_count and not native_vocabulary:
            raise ValueError("fixed mask banks require native vocabulary")
        if native_vocabulary and int(pretrained_tokenizer.mask_token_id) != gap_id:
            raise ValueError("native vocabulary requires GAP to be the mask token")
        d_model = int(backbone.config.hidden_size)
        if native_vocabulary:
            self.token_embedding = backbone.get_input_embeddings()
            if self.token_embedding.num_embeddings != vocab_size:
                raise ValueError(
                    "native vocabulary size does not match pretrained embeddings"
                )
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.step_embedding = nn.Embedding(max_steps, d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self._keep_prompt_states = False
        self.prompt_states = None
        self.prompt_mask = None
        self.mask_bank_states = None
        if initialize_custom_embeddings and not native_vocabulary:
            self.initialize_custom_token_embeddings()
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.config.hidden_size)

    def initialize_custom_token_embeddings(self) -> None:
        """Map every custom-BPE token into the pretrained embedding space."""
        pretrained_embeddings = self.backbone.get_input_embeddings().weight.detach()
        strings = [
            self.source_tokenizer.decode([index], skip_special_tokens=False)
            for index in range(self.token_embedding.num_embeddings)
        ]
        initialized = self.token_embedding.weight.detach().clone()
        for start in range(0, len(strings), 256):
            encoded = self.pretrained_tokenizer(
                strings[start : start + 256],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            ids = encoded["input_ids"].to(pretrained_embeddings.device)
            mask = encoded["attention_mask"].to(
                device=pretrained_embeddings.device,
                dtype=pretrained_embeddings.dtype,
            )
            counts = mask.sum(dim=1, keepdim=True)
            means = (
                pretrained_embeddings[ids] * mask.unsqueeze(-1)
            ).sum(dim=1) / counts.clamp_min(1.0)
            usable = counts.squeeze(1).gt(0).cpu()
            initialized[start : start + len(means)][usable] = means[usable].cpu()
        with torch.no_grad():
            self.token_embedding.weight.copy_(initialized)

    def render_prompts(
        self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ):
        rows = tokens.detach().cpu().tolist()
        padding_rows = (
            padding_mask.detach().cpu().tolist()
            if padding_mask is not None
            else [[False] * len(row) for row in rows]
        )
        texts = []
        custom_gap_positions = []
        for row, row_padding in zip(rows, padding_rows):
            valid = [token for token, padded in zip(row, row_padding) if not padded]
            gaps = [index for index, token in enumerate(valid) if token == self.gap_id]
            if len(gaps) != 1:
                raise ValueError("pretrained exact-inside encoder requires one gap")
            gap = gaps[0]
            if gap == 0 or gap == len(valid) - 1:
                raise ValueError("gap must have structural boundary tokens")
            left = self.source_tokenizer.decode(
                valid[1:gap], skip_special_tokens=False
            )
            right = self.source_tokenizer.decode(
                valid[gap + 1 : -1], skip_special_tokens=False
            )
            texts.append(left + self.pretrained_tokenizer.mask_token + right)
            custom_gap_positions.append(gap)
        return texts, custom_gap_positions

    def native_model_inputs(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        mask_counts: Optional[Sequence[int]] = None,
    ):
        """Build native-token inputs, optionally expanding the single gap."""
        if not self.native_vocabulary:
            raise ValueError("native_model_inputs requires native vocabulary")
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_id)
        if mask_counts is None:
            return {
                "input_ids": tokens.masked_fill(padding_mask, self.pad_id),
                "attention_mask": (~padding_mask).to(torch.long),
            }
        rows = []
        for row, padded, count in zip(tokens, padding_mask, mask_counts):
            valid = row[~padded].tolist()
            gaps = [index for index, token in enumerate(valid) if token == self.gap_id]
            if len(gaps) != 1:
                raise ValueError("native pretrained encoder requires one gap")
            gap = gaps[0]
            expanded = (
                valid[:gap]
                + [self.gap_id] * max(1, int(count))
                + valid[gap + 1 :]
            )[: self.max_length]
            rows.append(expanded)
        width = max(len(row) for row in rows)
        input_ids = tokens.new_full((len(rows), width), self.pad_id)
        attention_mask = tokens.new_zeros((len(rows), width))
        for index, row in enumerate(rows):
            input_ids[index, :len(row)] = torch.tensor(row, device=tokens.device)
            attention_mask[index, :len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if steps is not None:
            raise ValueError("prompt encoder does not accept generation steps")
        if self.native_vocabulary:
            counts = (
                [self.fixed_mask_count] * len(tokens)
                if self.fixed_mask_count
                else None
            )
            model_inputs = self.native_model_inputs(tokens, padding_mask, counts)
        else:
            texts, _ = self.render_prompts(tokens, padding_mask)
            encoded = self.pretrained_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(tokens.device) for key, value in encoded.items()
            }
        mask_matches = model_inputs["input_ids"].eq(
            int(self.pretrained_tokenizer.mask_token_id)
        )
        counts = mask_matches.sum(dim=1)
        expected_masks = self.fixed_mask_count or 1
        if not bool(counts.eq(expected_masks).all()):
            raise ValueError(
                "encoded prompts must retain exactly {} mask token(s)".format(
                    expected_masks
                )
            )
        mask_positions = mask_matches.to(torch.int64).argmax(dim=1)
        hidden = self.backbone(**model_inputs).last_hidden_state
        rows = torch.arange(len(tokens), device=tokens.device)
        if self.fixed_mask_count:
            banks = hidden.new_zeros(
                (len(tokens), self.fixed_mask_count, hidden.size(-1))
            )
            for row in range(len(tokens)):
                positions = mask_matches[row].nonzero().flatten()
                banks[row] = hidden[row, positions]
            # Keep raw backbone mask states for the pretrained MLM head. The
            # pooled context is used only by STOP/topology queries.
            self.mask_bank_states = banks
            context = self.context_norm(banks.mean(dim=1))
        else:
            self.mask_bank_states = None
            context = self.context_norm(hidden[rows, mask_positions])
        custom_gaps = tokens.eq(self.gap_id).unsqueeze(-1).to(context.dtype)
        if self._keep_prompt_states:
            # Stash the full sequence for interval heads that attend over it.
            # Pooling to the mask state alone discards most of what the
            # backbone computed; see research/LIKELIHOOD_DECOMPOSITION.md.
            self.prompt_states = hidden
            self.prompt_mask = model_inputs["attention_mask"].bool()
        return context.unsqueeze(1) * custom_gaps

    def keep_prompt_states(self, enabled: bool = True) -> None:
        """Retain the backbone's full sequence output from the next encode."""
        self._keep_prompt_states = enabled
        if not enabled:
            self.prompt_states = None
            self.prompt_mask = None


class PretrainedIntervalInsideModel(nn.Module):
    """Exact interval model conditioned by a pretrained masked encoder."""

    def __init__(
        self,
        vocab_size: int,
        gap_id: int,
        pad_id: int,
        source_tokenizer,
        model_name: str = "distilroberta-base",
        cache_dir: str = ".hf_cache/hub",
        max_steps: int = 32,
        max_length: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
        random_init_backbone: bool = False,
        backbone=None,
        pretrained_tokenizer=None,
        initialize_custom_embeddings: bool = True,
        tie_token_embeddings: bool = True,
        prompt_attention: bool = False,
        native_vocabulary: bool = False,
        pretrained_lm_head=None,
        fixed_mask_count: int = 0,
    ) -> None:
        super().__init__()
        if prompt_attention and fixed_mask_count:
            raise ValueError("prompt attention and fixed mask bank are exclusive")
        if native_vocabulary:
            if backbone is None and pretrained_lm_head is None:
                from transformers import (
                    AutoConfig,
                    AutoModelForMaskedLM,
                    AutoTokenizer,
                )

                if pretrained_tokenizer is None:
                    pretrained_tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        use_fast=True,
                        local_files_only=local_files_only,
                    )
                if random_init_backbone:
                    config = AutoConfig.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                    )
                    masked_lm = AutoModelForMaskedLM.from_config(config)
                else:
                    masked_lm = AutoModelForMaskedLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                    )
                backbone = masked_lm.base_model
                pretrained_lm_head = getattr(masked_lm, "lm_head", None)
            if backbone is None or pretrained_lm_head is None:
                raise ValueError(
                    "native vocabulary needs both a backbone and pretrained MLM head"
                )
        self.encoder = PretrainedIntervalEncoder(
            vocab_size,
            gap_id,
            pad_id,
            source_tokenizer,
            model_name,
            cache_dir,
            max_steps=max_steps,
            max_length=max_length,
            freeze_backbone=freeze_backbone,
            gradient_checkpointing=gradient_checkpointing,
            local_files_only=local_files_only,
            random_init_backbone=random_init_backbone,
            backbone=backbone,
            pretrained_tokenizer=pretrained_tokenizer,
            initialize_custom_embeddings=initialize_custom_embeddings,
            native_vocabulary=native_vocabulary,
            fixed_mask_count=fixed_mask_count,
        )
        d_model = self.encoder.hidden_size
        # With prompt_attention each interval builds a query from its own
        # boundary tokens and depth and attends over the backbone's sequence
        # output, instead of every node in the chart sharing one pooled vector.
        # Cost is O(D n^2 L): the token head runs per interval record, not per
        # chart cell, and all records of an example share the same keys.
        self.prompt_attention = prompt_attention
        self.fixed_mask_count = int(fixed_mask_count)
        self.requires_record_owners = bool(prompt_attention or fixed_mask_count)
        self.interval_projection = (
            None if fixed_mask_count else nn.Linear(
                (4 if prompt_attention else 3) * d_model, d_model
            )
        )
        if prompt_attention:
            self.prompt_query = nn.Linear(3 * d_model, d_model)
            self.prompt_key = nn.Linear(d_model, d_model)
            self.prompt_value = nn.Linear(d_model, d_model)
            self.prompt_norm = nn.LayerNorm(d_model)
        if fixed_mask_count:
            # Queries depend only on the observed prompt summary, generated
            # boundaries and depth. The bank width is constant, so no target
            # length is available to this selection.
            self.mask_bank_query = nn.Linear(3 * d_model, d_model)
            self.mask_bank_residual = nn.Linear(3 * d_model, d_model)
            self.mask_bank_residual_scale = nn.Parameter(torch.zeros(()))
        self.interval_norm = nn.LayerNorm(d_model)
        self.interval_dropout = nn.Dropout(dropout)
        if native_vocabulary:
            # Keep RoBERTa's dense/GELU/layer-norm/decoder stack intact rather
            # than relearning a custom-vocabulary projection from averaged
            # input embeddings.
            self.token_head = pretrained_lm_head
        else:
            self.token_head = nn.Linear(d_model, vocab_size)
            if tie_token_embeddings:
                self.token_head.weight = self.encoder.token_embedding.weight
        self.stop_head = nn.Linear(d_model, 1)
        self.topology_head = nn.Linear(2 * d_model, 4)

    @property
    def d_model(self) -> int:
        return self.encoder.hidden_size

    def encode(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(tokens, padding_mask)

    def _attend_prompt(self, features: torch.Tensor, owners: torch.Tensor):
        """Let each interval record read the backbone's own sequence output.

        ``features`` is the per-record ``[context, left, right]`` concatenation
        and ``owners`` maps each record to its example. Records of one example
        share keys, so the attention is padded per example and batched, which
        keeps it at a few tens of thousands of scores rather than materializing
        one key matrix per record.
        """
        states = self.encoder.prompt_states
        mask = self.encoder.prompt_mask
        if states is None:
            raise ValueError(
                "prompt attention needs encoder.keep_prompt_states() before encode"
            )
        if int(owners.max()) >= states.size(0):
            # Callers that encode in chunks must accumulate the states too, or
            # the stashed batch silently belongs to a different set of prompts.
            raise ValueError(
                "owner index {} exceeds the {} stashed prompts; accumulate "
                "prompt states across chunks before attending".format(
                    int(owners.max()), states.size(0)
                )
            )
        queries = self.prompt_query(features)
        keys = self.prompt_key(states)
        values = self.prompt_value(states)

        batch = states.size(0)
        counts = torch.bincount(owners, minlength=batch)
        width = int(counts.max()) if counts.numel() else 0
        if width == 0:
            return torch.zeros_like(queries)
        # Slot each record within its example so queries can be padded.
        order = torch.argsort(owners, stable=True)
        ranks = torch.empty_like(order)
        starts = torch.cumsum(counts, 0) - counts
        ranks[order] = (
            torch.arange(owners.numel(), device=owners.device) - starts[owners[order]]
        )
        padded = queries.new_zeros((batch, width, queries.size(-1)))
        padded[owners, ranks] = queries

        scale = queries.size(-1) ** 0.5
        scores = torch.bmm(padded, keys.transpose(1, 2)) / scale
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attended = torch.bmm(scores.softmax(dim=-1), values)
        return self.prompt_norm(attended[owners, ranks])

    def _attend_mask_bank(self, features: torch.Tensor, owners: torch.Tensor):
        """Select one MLM-compatible state from a fixed, length-blind bank."""
        states = self.encoder.mask_bank_states
        if states is None:
            raise ValueError("fixed mask bank needs encoder states before scoring")
        if int(owners.max()) >= states.size(0):
            raise ValueError("fixed mask bank owner exceeds encoded batch")
        queries = self.mask_bank_query(features)
        owned_states = states[owners]
        scores = torch.bmm(
            queries.unsqueeze(1), owned_states.transpose(1, 2)
        ).squeeze(1) / (queries.size(-1) ** 0.5)
        attended = torch.bmm(
            scores.softmax(dim=-1).unsqueeze(1), owned_states
        ).squeeze(1)
        # Start exactly in the pretrained mask-state space; let training add a
        # bounded node-specific correction only if validation supports it.
        correction = torch.tanh(self.mask_bank_residual(features))
        return attended + self.mask_bank_residual_scale.tanh() * correction

    def interval_hidden(
        self,
        context_hidden: torch.Tensor,
        left_boundary: torch.Tensor,
        right_boundary: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
        owners: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if depths is not None:
            context_hidden = context_hidden + self.encoder.step_embedding(depths)
        left = self.encoder.token_embedding(left_boundary)
        right = self.encoder.token_embedding(right_boundary)
        features = torch.cat((context_hidden, left, right), dim=-1)
        if self.fixed_mask_count:
            if owners is None:
                raise ValueError("fixed mask bank needs per-record owner indices")
            hidden = self._attend_mask_bank(features, owners)
            return self.interval_dropout(hidden)
        if self.prompt_attention:
            if owners is None:
                raise ValueError("prompt attention needs per-record owner indices")
            features = torch.cat(
                (features, self._attend_prompt(features, owners)), dim=-1
            )
        hidden = self.interval_projection(features)
        return self.interval_dropout(
            self.interval_norm(torch.nn.functional.gelu(hidden))
        )

    def interval_logits(
        self,
        context_hidden: torch.Tensor,
        left_boundary: torch.Tensor,
        right_boundary: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
        owners: Optional[torch.Tensor] = None,
    ):
        hidden = self.interval_hidden(
            context_hidden, left_boundary, right_boundary, depths, owners
        )
        return self.token_head(hidden), self.stop_head(hidden).squeeze(-1), hidden

    def topology_logits(
        self, interval_hidden: torch.Tensor, chosen_tokens: torch.Tensor
    ) -> torch.Tensor:
        token = self.encoder.token_embedding(chosen_tokens)
        return self.topology_head(torch.cat((interval_hidden, token), dim=-1))


class PretrainedGapFrontierModel(nn.Module):
    """Re-encode the current partial canvas and score every open gap in parallel.

    Unlike :class:`PretrainedIntervalInsideModel`, this model has no persistent
    mask bank and no target-length-indexed chart. A gap is represented by the
    native masked-LM state at its current position in the partial sequence.
    After a frontier is expanded, generated tokens and newly-created gaps are
    fed through the backbone again on the next round.

    Lexical and structural learning are intentionally separated. The native MLM
    head and backbone receive token-loss gradients, while root/degree/direction
    heads read a detached backbone state through their own adapter.
    """

    def __init__(
        self,
        vocab_size: int,
        gap_id: int,
        pad_id: int,
        model_name: str = "distilroberta-base",
        cache_dir: str = ".hf_cache/hub",
        max_steps: int = 32,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
        random_init_backbone: bool = False,
        backbone=None,
        pretrained_tokenizer=None,
        pretrained_lm_head=None,
        detach_structure_encoder: bool = True,
        token_conditioned_topology: bool = False,
        marginal_preserving_joint: bool = False,
        direct_joint_actions: bool = False,
        joint_rank: int = 32,
        joint_sinkhorn_iterations: int = 12,
        zero_joint_interaction: bool = False,
    ) -> None:
        super().__init__()
        joint_modes = sum((
            bool(token_conditioned_topology),
            bool(marginal_preserving_joint),
            bool(direct_joint_actions),
        ))
        if joint_modes > 1:
            raise ValueError(
                "token-conditioned, marginal-preserving, and direct joint "
                "actions are mutually exclusive"
            )
        if joint_rank < 1 or joint_sinkhorn_iterations < 1:
            raise ValueError("joint rank and Sinkhorn iterations must be positive")
        if backbone is None or pretrained_lm_head is None:
            from transformers import (
                AutoConfig,
                AutoModelForMaskedLM,
                AutoTokenizer,
            )

            if pretrained_tokenizer is None:
                pretrained_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    use_fast=True,
                    local_files_only=local_files_only,
                )
            if random_init_backbone:
                config = AutoConfig.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                masked_lm = AutoModelForMaskedLM.from_config(config)
            else:
                masked_lm = AutoModelForMaskedLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            backbone = masked_lm.base_model
            pretrained_lm_head = getattr(masked_lm, "lm_head", None)
        if pretrained_tokenizer is None or pretrained_lm_head is None:
            raise ValueError(
                "pretrained frontier model needs a tokenizer and native MLM head"
            )
        if pretrained_tokenizer.mask_token_id is None:
            raise ValueError("pretrained tokenizer must define a mask token")
        if int(pretrained_tokenizer.mask_token_id) != int(gap_id):
            raise ValueError("GAP must be the pretrained tokenizer's mask token")
        if int(backbone.get_input_embeddings().num_embeddings) != int(vocab_size):
            raise ValueError("native vocabulary size does not match the backbone")

        self.backbone = backbone
        self.pretrained_tokenizer = pretrained_tokenizer
        self.token_embedding = backbone.get_input_embeddings()
        self.token_head = pretrained_lm_head
        self.gap_id = int(gap_id)
        self.pad_id = int(pad_id)
        self.detach_structure_encoder = bool(detach_structure_encoder)
        d_model = int(backbone.config.hidden_size)
        self.step_embedding = nn.Embedding(max_steps, d_model)
        self.gap_type_embedding = nn.Embedding(2, d_model)
        self.structure_adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        # Termination, branching degree, and unary direction are separate so
        # chain collapse can be measured and calibrated directly.
        self.root_stop_head = nn.Linear(d_model, 1)
        self.degree_head = nn.Linear(d_model, 3)
        self.direction_head = nn.Linear(d_model, 2)
        # Growing the tree and emitting a token become one decision when the
        # branching heads read the token the node just emitted.  The projection
        # is zero-initialized, so the model starts as the token-independent
        # frontier policy and any difference is attributable to the coupling.
        # Held-out length calibration lives outside the policy, as it does for
        # the scaffold model.  These are non-persistent so every checkpoint
        # written before calibration existed still loads strictly.
        self.register_buffer(
            "calibration_root_bias", torch.zeros(()), persistent=False
        )
        self.register_buffer(
            "calibration_degree_bias",
            torch.zeros(max_steps, 3),
            persistent=False,
        )
        self.token_conditioned_topology = bool(token_conditioned_topology)
        self.token_condition = (
            nn.Linear(d_model, d_model, bias=False)
            if self.token_conditioned_topology
            else None
        )
        if self.token_condition is not None:
            nn.init.zeros_(self.token_condition.weight)
        self.marginal_preserving_joint = bool(marginal_preserving_joint)
        self.direct_joint_actions = bool(direct_joint_actions)
        self.joint_rank = int(joint_rank)
        self.joint_sinkhorn_iterations = int(joint_sinkhorn_iterations)
        joint_actions = self.marginal_preserving_joint or self.direct_joint_actions
        self.joint_token_projection = (
            nn.Linear(d_model, self.joint_rank, bias=False)
            if joint_actions
            else None
        )
        self.joint_marker_projection = (
            nn.Linear(d_model, 4 * self.joint_rank)
            if joint_actions
            else None
        )
        if self.joint_marker_projection is not None:
            # Zero interaction gives the independent product of the native MLM
            # marginal and the existing marker marginal exactly at init.
            nn.init.zeros_(self.joint_marker_projection.weight)
            nn.init.zeros_(self.joint_marker_projection.bias)
        self.zero_joint_interaction = bool(zero_joint_interaction)
        if self.zero_joint_interaction:
            if not joint_actions:
                raise ValueError(
                    "zeroing the interaction requires a joint action mode"
                )
            # The ablation `research/SEMANTIC_BRANCHING.md` prescribes: hold the
            # token/marker interaction at its zero init so the joint table stays
            # exactly the independent product in training *and* rollout, while
            # every other part of the model trains as usual.  Freezing rather
            # than re-zeroing keeps these out of the optimizer entirely, so
            # weight decay cannot drift them off zero.
            for parameter in self.joint_token_projection.parameters():
                parameter.requires_grad_(False)
            for parameter in self.joint_marker_projection.parameters():
                parameter.requires_grad_(False)

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @property
    def d_model(self) -> int:
        return int(self.backbone.config.hidden_size)

    def structure_logits_from_hidden(
        self,
        hidden: torch.Tensor,
        steps: torch.Tensor,
        structure_token_ids: Optional[torch.Tensor] = None,
    ):
        """Branching logits, optionally conditioned on each node's own token.

        `structure_token_ids` carries the emitted token at every open gap and a
        negative value elsewhere.  During training that is the gold pivot; at
        inference it is the token the node just sampled, so one backbone pass
        still serves both decisions.
        """
        steps = steps.clamp(0, self.step_embedding.num_embeddings - 1)
        structure_input = hidden.detach() if self.detach_structure_encoder else hidden
        root_types = steps.eq(0).to(torch.long)
        structure_input = (
            structure_input
            + self.step_embedding(steps).unsqueeze(1)
            + self.gap_type_embedding(root_types).unsqueeze(1)
        )
        structure = self.structure_adapter(structure_input)
        # The root decides emptiness *before* any token exists, so it must not
        # read one.  Conditioning it would also leak the label during training,
        # where an empty span is exactly the case with no gold pivot to feed.
        root_stop = (
            self.root_stop_head(structure).squeeze(-1)
            + self.calibration_root_bias
        )
        if self.token_condition is not None and structure_token_ids is not None:
            valid = structure_token_ids.ge(0)
            vectors = self.token_embedding(
                structure_token_ids.clamp_min(0)
            ).detach()
            structure = self.structure_adapter(
                structure_input
                + self.token_condition(vectors)
                * valid.unsqueeze(-1).to(structure_input.dtype)
            )
        return (
            root_stop,
            self.degree_head(structure)
            + self.calibration_degree_bias[steps].unsqueeze(1),
            self.direction_head(structure),
        )

    @staticmethod
    def marker_log_probs(
        degree_logits: torch.Tensor,
        direction_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Convert degree/direction logits to leaf/left/right/both log masses."""
        degree = degree_logits.log_softmax(dim=-1)
        direction = direction_logits.log_softmax(dim=-1)
        return torch.stack((
            degree[..., 0],
            degree[..., 1] + direction[..., 0],
            degree[..., 1] + direction[..., 1],
            degree[..., 2],
        ), dim=-1)

    def joint_action_log_probs(
        self,
        token_logits: torch.Tensor,
        degree_logits: torch.Tensor,
        direction_logits: torch.Tensor,
        hidden: torch.Tensor,
        steps: torch.Tensor,
        generated_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the configured joint distribution over token/marker actions.

        Inputs contain only active non-empty nodes, so the returned tensor is
        ``[nodes, generated vocabulary, 4]``. The direct mode globally
        normalizes a learned low-rank joint energy, so token identity and the
        branch marker are genuinely one decision during both training and
        rollout. The marginal-preserving mode instead applies iterative
        proportional fitting to retain the supplied lexical and topology
        marginals. Root emptiness remains a preceding independent event.
        """
        if not (self.marginal_preserving_joint or self.direct_joint_actions):
            raise ValueError("joint actions are disabled")
        if token_logits.dim() != 2 or hidden.dim() != 2:
            raise ValueError("joint action inputs must be flattened node tensors")
        generated_token_ids = generated_token_ids.to(token_logits.device)
        token_logp = token_logits.index_select(
            -1, generated_token_ids
        ).log_softmax(dim=-1)
        marker_logp = self.marker_log_probs(
            degree_logits, direction_logits
        )
        if self.zero_joint_interaction:
            # Structural guarantee for the ablation: the interaction is not
            # computed at all, so no loaded checkpoint can reintroduce it and
            # the vocabulary-wide einsum is skipped.
            log_joint = token_logp.unsqueeze(-1) + marker_logp.unsqueeze(-2)
        else:
            clipped_steps = steps.clamp(
                0, self.step_embedding.num_embeddings - 1
            )
            root_types = clipped_steps.eq(0).to(torch.long)
            context = (
                hidden.detach()
                + self.step_embedding(clipped_steps)
                + self.gap_type_embedding(root_types)
            )
            marker_features = self.joint_marker_projection(context).view(
                context.size(0), 4, self.joint_rank
            )
            token_vectors = self.token_embedding(generated_token_ids).detach()
            token_features = self.joint_token_projection(token_vectors)
            interaction = torch.einsum(
                "vr,nmr->nvm", token_features, marker_features
            ) / math.sqrt(self.joint_rank)
            log_joint = (
                token_logp.unsqueeze(-1)
                + marker_logp.unsqueeze(-2)
                + interaction
            )
        if self.direct_joint_actions:
            # Training and rollout consume this same joint table. At zero
            # interaction it nests the independent token/marker product; a
            # learned interaction may move either marginal when supported by
            # joint likelihood.
            return log_joint - torch.logsumexp(
                log_joint.flatten(start_dim=-2), dim=-1
            ).view(-1, 1, 1)
        # Alternating log-domain row/column scaling is stable in mixed
        # precision and differentiable through the interaction and marker head.
        for _ in range(self.joint_sinkhorn_iterations):
            log_joint = log_joint + (
                token_logp - torch.logsumexp(log_joint, dim=-1)
            ).unsqueeze(-1)
            log_joint = log_joint + (
                marker_logp - torch.logsumexp(log_joint, dim=-2)
            ).unsqueeze(-2)
        # Finish on the token marginal, which is the lexical invariant. With
        # four marker columns the preceding iterations make column error tiny.
        log_joint = log_joint + (
            token_logp - torch.logsumexp(log_joint, dim=-1)
        ).unsqueeze(-1)
        return log_joint

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
        structure_token_ids: Optional[torch.Tensor] = None,
    ):
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_id)
        if steps is None:
            steps = torch.zeros(
                tokens.size(0), dtype=torch.long, device=tokens.device
            )
        hidden = self.backbone(
            input_ids=tokens.masked_fill(padding_mask, self.pad_id),
            attention_mask=(~padding_mask).to(torch.long),
        ).last_hidden_state
        token_logits = self.token_head(hidden)
        root_stop, degree, direction = self.structure_logits_from_hidden(
            hidden, steps, structure_token_ids
        )
        return token_logits, root_stop, degree, direction, hidden


class PretrainedScaffoldTopologyModel(nn.Module):
    """Shape-only frontier policy with a shared latent regime per round.

    The frozen pretrained encoder supplies optional prompt evidence.  Shape is
    governed primarily by explicit depth-indexed priors.  A small, zero-gated
    residual can use context, while a round-level categorical regime couples
    sibling decisions without sharing any lexical latent or mask bank.
    """

    def __init__(
        self,
        vocab_size: int,
        gap_id: int,
        pad_id: int,
        model_name: str = "distilroberta-base",
        cache_dir: str = ".hf_cache/hub",
        max_steps: int = 16,
        regimes: int = 4,
        residual_dim: int = 128,
        state_feedback: bool = False,
        prompt_conditioned: bool = False,
        state_bins: int = 17,
        semantic_codes: int = 0,
        continuous_semantic: bool = False,
        semantic_seed: int = 29,
        semantic_injection_scale: float = 0.25,
        dropout: float = 0.1,
        local_files_only: bool = False,
        backbone=None,
        pretrained_tokenizer=None,
    ) -> None:
        super().__init__()
        if regimes < 1:
            raise ValueError("regimes must be positive")
        if state_bins < 2:
            raise ValueError("state_bins must be at least two")
        if semantic_codes < 0:
            raise ValueError("semantic_codes cannot be negative")
        if semantic_codes and continuous_semantic:
            raise ValueError(
                "discrete and continuous semantic states are exclusive"
            )
        if backbone is None:
            from transformers import AutoModel, AutoTokenizer

            if pretrained_tokenizer is None:
                pretrained_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    use_fast=True,
                    local_files_only=local_files_only,
                )
            backbone = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        if pretrained_tokenizer is None:
            raise ValueError("a pretrained tokenizer is required")
        if int(pretrained_tokenizer.mask_token_id) != int(gap_id):
            raise ValueError("GAP must be the pretrained mask token")
        if int(backbone.get_input_embeddings().num_embeddings) != int(vocab_size):
            raise ValueError("native vocabulary size does not match the backbone")

        self.backbone = backbone
        self.gap_id = int(gap_id)
        self.pad_id = int(pad_id)
        self.regimes = int(regimes)
        self.max_steps = int(max_steps)
        self.state_feedback = bool(state_feedback)
        # A prompt-conditioned policy reads one context vector fixed at round
        # zero instead of the evolving scaffold, which keeps the branching
        # process context-free *given the prompt* and therefore keeps the
        # total-progeny chart exact per prompt.
        self.prompt_conditioned = bool(prompt_conditioned)
        self.state_bins = int(state_bins)
        self.semantic_codes = int(semantic_codes)
        self.continuous_semantic = bool(continuous_semantic)
        self.semantic_injection_scale = float(semantic_injection_scale)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        d_model = int(backbone.config.hidden_size)
        if self.continuous_semantic:
            with torch.no_grad():
                token_vectors = (
                    backbone.get_input_embeddings().weight.detach().float()
                )
                semantic_mean = token_vectors.mean(dim=0)
                centered = token_vectors - semantic_mean
                semantic_scale = centered.norm(dim=-1).mean()
            self.register_buffer(
                "semantic_embedding_mean",
                semantic_mean.to(backbone.get_input_embeddings().weight.dtype),
                persistent=False,
            )
            self.register_buffer(
                "semantic_embedding_scale",
                semantic_scale.to(backbone.get_input_embeddings().weight.dtype),
                persistent=False,
            )
        if self.semantic_codes:
            with torch.no_grad():
                token_vectors = (
                    backbone.get_input_embeddings().weight.detach().float()
                )
                generator = torch.Generator(device=token_vectors.device)
                generator.manual_seed(int(semantic_seed))
                projection = torch.randn(
                    d_model,
                    self.semantic_codes,
                    generator=generator,
                    device=token_vectors.device,
                )
                normalized_tokens = F.normalize(token_vectors, dim=-1)
                normalized_projection = F.normalize(projection, dim=0)
                token_codes = (
                    normalized_tokens @ normalized_projection
                ).argmax(dim=-1)
                code_vectors = token_vectors.new_zeros(
                    self.semantic_codes, d_model
                )
                code_vectors.index_add_(0, token_codes, token_vectors)
                code_counts = torch.bincount(
                    token_codes, minlength=self.semantic_codes
                ).clamp_min(1).unsqueeze(-1)
                code_vectors = code_vectors / code_counts
                code_vectors = code_vectors - token_vectors.mean(
                    dim=0, keepdim=True
                )
                code_vectors = F.normalize(code_vectors, dim=-1)
                code_vectors = code_vectors * token_vectors.norm(
                    dim=-1
                ).mean()
            self.register_buffer(
                "semantic_token_codes", token_codes, persistent=False
            )
            self.register_buffer(
                "semantic_code_vectors",
                code_vectors.to(backbone.get_input_embeddings().weight.dtype),
                persistent=False,
            )
        self.residual_dim = residual_dim
        self.local_adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, residual_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(residual_dim),
        )
        self.global_adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, residual_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(residual_dim),
        )

        self.root_stop_prior = nn.Parameter(torch.zeros(()))
        self.regime_prior = nn.Embedding(max_steps, regimes)
        self.regime_transition = nn.Parameter(
            torch.zeros(max_steps, regimes, regimes)
        )
        self.degree_prior = nn.Parameter(torch.zeros(max_steps, regimes, 3))
        self.direction_prior = nn.Parameter(torch.zeros(max_steps, regimes, 2))
        nn.init.normal_(self.degree_prior, std=0.01)
        nn.init.normal_(self.direction_prior, std=0.01)
        if self.state_feedback:
            self.open_regime_prior = nn.Embedding(state_bins, regimes)
            self.completed_regime_prior = nn.Embedding(state_bins, regimes)
            self.open_degree_prior = nn.Embedding(
                state_bins, regimes * 3
            )
            self.completed_degree_prior = nn.Embedding(
                state_bins, regimes * 3
            )
            nn.init.zeros_(self.open_regime_prior.weight)
            nn.init.zeros_(self.completed_regime_prior.weight)
            nn.init.zeros_(self.open_degree_prior.weight)
            nn.init.zeros_(self.completed_degree_prior.weight)
        self.root_residual = nn.Linear(residual_dim, 1)
        self.regime_residual = nn.Linear(residual_dim, regimes)
        self.degree_residual = nn.Linear(residual_dim, regimes * 3)
        self.direction_residual = nn.Linear(residual_dim, regimes * 2)
        self.semantic_head = (
            nn.Linear(residual_dim, self.semantic_codes)
            if self.semantic_codes
            else None
        )
        self.semantic_vector_head = (
            nn.Linear(residual_dim, d_model)
            if self.continuous_semantic
            else None
        )

        # Zero gates make the initial policy an explicit global shape prior.
        # Context can enter only when held-out likelihood supports it.
        self.root_gate = nn.Parameter(torch.zeros(()))
        self.regime_gate = nn.Parameter(torch.zeros(max_steps))
        self.degree_gate = nn.Parameter(torch.zeros(max_steps))
        self.direction_gate = nn.Parameter(torch.zeros(max_steps))
        # Held-out calibration is disabled during ordinary training.  Keeping
        # these explicit makes post-hoc shape calibration reproducible without
        # touching the frozen encoder or learned residual policy.
        self.calibration_root_bias = nn.Parameter(
            torch.zeros(()), requires_grad=False
        )
        self.calibration_regime_bias = nn.Parameter(
            torch.zeros(max_steps, regimes), requires_grad=False
        )
        self.calibration_degree_bias = nn.Parameter(
            torch.zeros(max_steps, 3), requires_grad=False
        )
        self.calibration_direction_bias = nn.Parameter(
            torch.zeros(max_steps, 2), requires_grad=False
        )

    @property
    def d_model(self) -> int:
        return int(self.backbone.config.hidden_size)

    def topology_state_dict(self):
        """Return the small trainable state without duplicating the backbone."""
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("backbone.")
        }

    def token_semantic_states(self, token_ids: torch.Tensor) -> torch.Tensor:
        if not self.continuous_semantic:
            raise ValueError("continuous semantic states are disabled")
        embeddings = self.backbone.get_input_embeddings().weight[token_ids]
        centered = embeddings - self.semantic_embedding_mean
        return F.normalize(centered, dim=-1) * self.semantic_embedding_scale

    def load_topology_state_dict(self, state_dict) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        non_backbone_missing = [
            name
            for name in missing
            if not name.startswith("backbone.")
            and not name.startswith("calibration_")
            and name != "regime_transition"
        ]
        if non_backbone_missing or unexpected:
            raise RuntimeError(
                "invalid topology state: missing={} unexpected={}".format(
                    non_backbone_missing, unexpected
                )
            )

    def prompt_shape_context(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        source: str = "pooled",
    ) -> torch.Tensor:
        """Encode the round-zero canvas once into a fixed per-prompt context.

        Every later round reuses this vector, so the shape policy depends on
        the prompt but not on the partially grown scaffold.  That is what makes
        `conditional_scaffold_length_distribution` exact rather than an
        approximation of an evolving process. source="gap" retains the
        backbone state at the native mask position instead of diluting it with
        a sequence-wide mean. It still does not reveal or predict a length.
        """
        if source not in ("pooled", "gap"):
            raise ValueError("prompt shape context source must be pooled or gap")
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_id)
        input_ids = tokens.masked_fill(padding_mask, self.pad_id)
        with torch.no_grad():
            hidden = self.backbone(
                input_ids=input_ids,
                attention_mask=(~padding_mask).to(torch.long),
            ).last_hidden_state
        if source == "gap":
            gaps = tokens.eq(self.gap_id) & ~padding_mask
            if not bool(gaps.any(dim=1).all()):
                raise ValueError("every prompt must contain a gap")
            positions = gaps.to(torch.long).argmax(dim=1)
            rows = torch.arange(tokens.size(0), device=tokens.device)
            prompt_hidden = hidden[rows, positions]
        else:
            observed = (~padding_mask).to(hidden.dtype).unsqueeze(-1)
            prompt_hidden = (
                (hidden * observed).sum(1) / observed.sum(1).clamp_min(1.0)
            )
        return self.global_adapter(prompt_hidden)

    def conditional_shape_logits(
        self,
        context: torch.Tensor,
        step: int,
        open_count: int,
        completed_count: int,
    ):
        """Shape logits for one chart state, from the prompt context alone.

        `structure_logits` reads the current canvas, so its degree logits
        differ per node and per round.  Here every open node in a round shares
        one distribution determined by the prompt, the round, and the realized
        counts, which is exactly the exchangeability the chart assumes.
        """
        step = min(max(int(step), 0), self.max_steps - 1)
        batch = context.size(0)
        root = (
            self.root_stop_prior
            + self.calibration_root_bias
            + torch.tanh(self.root_gate) * self.root_residual(context).squeeze(-1)
        )
        regime = (
            self.regime_prior.weight[step]
            + self.calibration_regime_bias[step]
            + torch.tanh(self.regime_gate[step]) * self.regime_residual(context)
        )
        degree = (
            self.degree_prior[step]
            + self.calibration_degree_bias[step].unsqueeze(0)
            + torch.tanh(self.degree_gate[step])
            * self.degree_residual(context).view(batch, self.regimes, 3)
        )
        direction = (
            self.direction_prior[step]
            + self.calibration_direction_bias[step].unsqueeze(0)
            + torch.tanh(self.direction_gate[step])
            * self.direction_residual(context).view(batch, self.regimes, 2)
        )
        if self.state_feedback:
            open_index = min(max(int(open_count), 0), self.state_bins - 1)
            completed_index = min(
                max(int(completed_count), 0), self.state_bins - 1
            )
            regime = (
                regime
                + self.open_regime_prior.weight[open_index]
                + self.completed_regime_prior.weight[completed_index]
            )
            degree = degree + (
                self.open_degree_prior.weight[open_index]
                + self.completed_degree_prior.weight[completed_index]
            ).view(self.regimes, 3).unsqueeze(0)
        return root, regime, degree, direction

    def structure_logits(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
        open_mask: Optional[torch.Tensor] = None,
        slot_codes: Optional[torch.Tensor] = None,
        slot_semantics: Optional[torch.Tensor] = None,
        semantic_scale: Optional[torch.Tensor] = None,
        semantic_requires_grad: bool = False,
        return_semantic: bool = False,
        return_continuous_semantic: bool = False,
    ):
        if padding_mask is None:
            padding_mask = tokens.eq(self.pad_id)
        if steps is None:
            steps = torch.zeros(
                tokens.size(0), dtype=torch.long, device=tokens.device
            )
        steps = steps.clamp(0, self.max_steps - 1)
        if open_mask is None:
            open_mask = tokens.eq(self.gap_id) & ~padding_mask
        input_ids = tokens.masked_fill(padding_mask, self.pad_id)
        def encode_scaffold():
            if self.continuous_semantic and slot_semantics is not None:
                valid_semantics = slot_semantics.abs().sum(dim=-1).gt(0)
                inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
                scale = (
                    self.semantic_injection_scale
                    if semantic_scale is None
                    else semantic_scale
                )
                if torch.is_tensor(scale):
                    while scale.dim() < inputs_embeds.dim():
                        scale = scale.unsqueeze(-1)
                inputs_embeds = inputs_embeds + (
                    scale
                    * slot_semantics.to(inputs_embeds.dtype)
                    * valid_semantics.unsqueeze(-1).to(inputs_embeds.dtype)
                )
                return self.backbone(
                    inputs_embeds=inputs_embeds,
                    attention_mask=(~padding_mask).to(torch.long),
                ).last_hidden_state
            elif self.semantic_codes and slot_codes is not None:
                valid_codes = slot_codes.ge(0) & ~padding_mask
                inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
                code_indices = slot_codes.clamp(0, self.semantic_codes - 1)
                code_residual = self.semantic_code_vectors[code_indices]
                inputs_embeds = inputs_embeds + (
                    self.semantic_injection_scale
                    * code_residual
                    * valid_codes.unsqueeze(-1).to(code_residual.dtype)
                )
                return self.backbone(
                    inputs_embeds=inputs_embeds,
                    attention_mask=(~padding_mask).to(torch.long),
                ).last_hidden_state
            return self.backbone(
                input_ids=input_ids,
                attention_mask=(~padding_mask).to(torch.long),
            ).last_hidden_state

        if semantic_requires_grad:
            hidden = encode_scaffold()
        else:
            with torch.no_grad():
                hidden = encode_scaffold()

        shape_hidden = hidden if semantic_requires_grad else hidden.detach()
        local = self.local_adapter(shape_hidden)
        observed = (~padding_mask).to(hidden.dtype).unsqueeze(-1)
        pooled = (shape_hidden * observed).sum(1) / observed.sum(1).clamp_min(1.0)
        global_hidden = self.global_adapter(pooled)
        batch, width = tokens.shape

        root = self.root_stop_prior + self.calibration_root_bias + torch.tanh(self.root_gate) * (
            self.root_residual(local).squeeze(-1)
        )
        regime = self.regime_prior(steps) + torch.tanh(
            self.regime_gate[steps]
        ).unsqueeze(-1) * self.regime_residual(global_hidden) + self.calibration_regime_bias[steps]
        degree = self.degree_prior[steps].unsqueeze(1).expand(
            batch, width, self.regimes, 3
        ) + torch.tanh(self.degree_gate[steps]).view(batch, 1, 1, 1) * (
            self.degree_residual(local).view(batch, width, self.regimes, 3)
        ) + self.calibration_degree_bias[steps].view(batch, 1, 1, 3)
        direction = self.direction_prior[steps].unsqueeze(1).expand(
            batch, width, self.regimes, 2
        ) + torch.tanh(self.direction_gate[steps]).view(batch, 1, 1, 1) * (
            self.direction_residual(local).view(batch, width, self.regimes, 2)
        ) + self.calibration_direction_bias[steps].view(batch, 1, 1, 2)
        if self.state_feedback:
            open_count = open_mask.sum(dim=1).clamp(
                0, self.state_bins - 1
            )
            mask_count = (
                tokens.eq(self.gap_id) & ~padding_mask
            ).sum(dim=1)
            completed_count = (mask_count - open_count).clamp(
                0, self.state_bins - 1
            )
            regime = (
                regime
                + self.open_regime_prior(open_count)
                + self.completed_regime_prior(completed_count)
            )
            state_degree = (
                self.open_degree_prior(open_count)
                + self.completed_degree_prior(completed_count)
            ).view(batch, self.regimes, 3)
            degree = degree + state_degree.unsqueeze(1)
        if return_semantic:
            if self.semantic_head is None:
                raise ValueError("semantic codes are disabled")
            return (
                root,
                regime,
                degree,
                direction,
                hidden,
                self.semantic_head(local),
            )
        if return_continuous_semantic:
            if self.semantic_vector_head is None:
                raise ValueError("continuous semantic states are disabled")
            semantic_state = F.normalize(
                self.semantic_vector_head(local), dim=-1
            ) * self.semantic_embedding_scale
            return (
                root,
                regime,
                degree,
                direction,
                hidden,
                semantic_state,
            )
        return root, regime, degree, direction, hidden

    @torch.inference_mode()
    def sample_regime_transition(
        self,
        previous_regimes: torch.Tensor,
        steps: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        steps = steps.clamp(0, self.max_steps - 1)
        logits = self.regime_transition[steps, previous_regimes]
        return torch.multinomial(
            logits.softmax(dim=-1), 1, generator=generator
        ).squeeze(-1)

    @torch.inference_mode()
    def sample_structure(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        steps: torch.Tensor,
        open_mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        forced_regimes: Optional[torch.Tensor] = None,
        slot_codes: Optional[torch.Tensor] = None,
        slot_semantics: Optional[torch.Tensor] = None,
        return_semantic_codes: bool = False,
        return_continuous_semantic: bool = False,
    ):
        if return_semantic_codes and return_continuous_semantic:
            raise ValueError("only one semantic representation can be returned")
        outputs = self.structure_logits(
            tokens,
            padding_mask,
            steps,
            open_mask,
            slot_codes=slot_codes,
            slot_semantics=slot_semantics,
            return_semantic=return_semantic_codes,
            return_continuous_semantic=return_continuous_semantic,
        )
        root, regime_logits, degree_logits, direction_logits = outputs[:4]
        stops = torch.rand(
            root.shape, device=root.device, generator=generator
        ) < root.sigmoid()
        regime_probabilities = regime_logits.softmax(dim=-1)
        sampled_regimes = torch.multinomial(
            regime_probabilities, 1, generator=generator
        ).squeeze(-1)
        if forced_regimes is None:
            regimes = sampled_regimes
        else:
            forced_regimes = forced_regimes.to(
                device=tokens.device, dtype=torch.long
            )
            regimes = torch.where(
                forced_regimes.ge(0), forced_regimes, sampled_regimes
            )
        rows = torch.arange(tokens.size(0), device=tokens.device)
        selected_degree = degree_logits[rows, :, regimes, :]
        selected_direction = direction_logits[rows, :, regimes, :]
        degrees = torch.multinomial(
            selected_degree.softmax(dim=-1).reshape(-1, 3),
            1,
            generator=generator,
        ).reshape(tokens.shape)
        directions = torch.multinomial(
            selected_direction.softmax(dim=-1).reshape(-1, 2),
            1,
            generator=generator,
        ).reshape(tokens.shape)
        if return_semantic_codes:
            semantic_logits = outputs[5]
            semantic = torch.multinomial(
                semantic_logits.softmax(dim=-1).reshape(
                    -1, self.semantic_codes
                ),
                1,
                generator=generator,
            ).reshape(tokens.shape)
            return stops, degrees, directions, regimes, semantic
        if return_continuous_semantic:
            return stops, degrees, directions, regimes, outputs[5]
        return stops, degrees, directions, regimes


class PretrainedUnifiedScaffoldModel(PretrainedScaffoldTopologyModel):
    """One masked LM jointly grows a scaffold and maintains token beliefs.

    Every round uses one shared backbone pass.  Its native MLM head supplies a
    token posterior for each open node while topology heads decide how that
    node branches.  The posterior embedding is stored on the completed node
    and is re-encoded on later rounds.  A zero-initialized gate makes the token
    path exactly the supplied masked-LM baseline before joint training.
    """

    def __init__(
        self,
        vocab_size: int,
        gap_id: int,
        pad_id: int,
        pretrained_lm_head,
        generated_token_ids: Sequence[int],
        posterior_topk: int = 32,
        freeze_lexical_model: bool = True,
        **kwargs,
    ) -> None:
        if pretrained_lm_head is None:
            raise ValueError("a pretrained MLM head is required")
        if not generated_token_ids:
            raise ValueError("generated token ids cannot be empty")
        kwargs.pop("continuous_semantic", None)
        kwargs.pop("semantic_codes", None)
        kwargs.pop("semantic_injection_scale", None)
        super().__init__(
            vocab_size,
            gap_id,
            pad_id,
            continuous_semantic=True,
            semantic_codes=0,
            semantic_injection_scale=1.0,
            **kwargs,
        )
        # Unified states come from the native token posterior, so the former
        # auxiliary vector-prediction head would be redundant and is removed.
        self.semantic_vector_head = None
        self.lm_head = pretrained_lm_head
        self.posterior_topk = max(1, int(posterior_topk))
        self.register_buffer(
            "generated_token_ids",
            torch.tensor(list(generated_token_ids), dtype=torch.long),
            persistent=True,
        )
        # Node-local token beliefs affect topology through their own attention
        # path.  They never perturb the MLM hidden states or token head, so
        # lexical baseline equivalence is exact even after this gate is learned.
        residual_dim = self.residual_dim
        self.posterior_query = nn.Linear(self.d_model, residual_dim, bias=False)
        self.posterior_key = nn.Linear(self.d_model, residual_dim, bias=False)
        self.posterior_value = nn.Linear(self.d_model, residual_dim, bias=False)
        self.posterior_norm = nn.LayerNorm(residual_dim)
        self.posterior_regime_head = nn.Linear(residual_dim, self.regimes)
        self.posterior_degree_head = nn.Linear(
            residual_dim, self.regimes * 3
        )
        self.posterior_direction_head = nn.Linear(
            residual_dim, self.regimes * 2
        )
        self.posterior_gate = nn.Parameter(torch.zeros(self.max_steps))
        self.freeze_lexical_model = bool(freeze_lexical_model)
        if freeze_lexical_model:
            for parameter in self.lm_head.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen pretrained components must not acquire stochastic dropout
        # drift while the small joint controller is trained.
        self.backbone.eval()
        if self.freeze_lexical_model:
            self.lm_head.eval()
        return self

    def topology_state_dict(self):
        """Save joint control parameters without duplicating the lexical LM."""
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("backbone.")
            and not name.startswith("lm_head.")
        }

    def load_topology_state_dict(self, state_dict) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = ("backbone.", "lm_head.", "calibration_")
        non_lexical_missing = [
            name
            for name in missing
            if not name.startswith(allowed_missing)
            and name != "regime_transition"
            and name != "generated_token_ids"
            and not name.startswith("posterior_")
        ]
        if non_lexical_missing or unexpected:
            raise RuntimeError(
                "invalid unified topology state: missing={} unexpected={}".format(
                    non_lexical_missing, unexpected
                )
            )

    def posterior_states(self, token_logits: torch.Tensor) -> torch.Tensor:
        """Convert native MLM probabilities to node-local soft embeddings."""
        generated = self.generated_token_ids.to(token_logits.device)
        allowed = token_logits.index_select(-1, generated)
        topk = min(self.posterior_topk, allowed.size(-1))
        values, indices = allowed.topk(topk, dim=-1)
        probabilities = values.softmax(dim=-1)
        token_ids = generated[indices]
        states = self.token_semantic_states(token_ids)
        return (probabilities.unsqueeze(-1) * states).sum(dim=-2)

    def unified_logits(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
        open_mask: Optional[torch.Tensor] = None,
        slot_semantics: Optional[torch.Tensor] = None,
        semantic_requires_grad: bool = False,
    ):
        if steps is None:
            steps = torch.zeros(
                tokens.size(0), dtype=torch.long, device=tokens.device
            )
        clipped_steps = steps.clamp(0, self.max_steps - 1)
        root, regime, degree, direction, hidden = self.structure_logits(
            tokens,
            padding_mask,
            clipped_steps,
            open_mask,
        )
        if slot_semantics is not None:
            if padding_mask is None:
                padding_mask = tokens.eq(self.pad_id)
            valid = slot_semantics.abs().sum(dim=-1).gt(0) & ~padding_mask
            query = self.posterior_query(hidden.detach())
            key = self.posterior_key(slot_semantics.to(query.dtype))
            value = self.posterior_value(slot_semantics.to(query.dtype))
            scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(
                query.size(-1)
            )
            scores = scores.masked_fill(~valid.unsqueeze(1), -1e4)
            attention = scores.softmax(dim=-1)
            attention = attention * valid.unsqueeze(1).to(attention.dtype)
            attention = attention / attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            local_posterior = self.posterior_norm(
                torch.matmul(attention, value)
            )
            valid_float = valid.to(value.dtype).unsqueeze(-1)
            global_posterior = self.posterior_norm(
                (value * valid_float).sum(dim=1)
                / valid_float.sum(dim=1).clamp_min(1.0)
            )
            has_posterior = valid.any(dim=1).to(value.dtype)
            local_posterior = (
                local_posterior * has_posterior.view(-1, 1, 1)
            )
            global_posterior = (
                global_posterior * has_posterior.unsqueeze(-1)
            )
            gate = torch.tanh(self.posterior_gate[clipped_steps])
            regime = regime + gate.unsqueeze(-1) * (
                self.posterior_regime_head(global_posterior)
            )
            degree = degree + gate.view(-1, 1, 1, 1) * (
                self.posterior_degree_head(local_posterior).view(
                    tokens.size(0), tokens.size(1), self.regimes, 3
                )
            )
            direction = direction + gate.view(-1, 1, 1, 1) * (
                self.posterior_direction_head(local_posterior).view(
                    tokens.size(0), tokens.size(1), self.regimes, 2
                )
            )
        return self.lm_head(hidden), root, regime, degree, direction, hidden

    @torch.inference_mode()
    def sample_unified_structure(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        steps: torch.Tensor,
        open_mask: torch.Tensor,
        slot_semantics: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        forced_regimes: Optional[torch.Tensor] = None,
    ):
        outputs = self.unified_logits(
            tokens,
            padding_mask,
            steps,
            open_mask,
            slot_semantics=slot_semantics,
        )
        token_logits, root, regime_logits, degree_logits, direction_logits = (
            outputs[:5]
        )
        stops = torch.rand(
            root.shape, device=root.device, generator=generator
        ) < root.sigmoid()
        sampled_regimes = torch.multinomial(
            regime_logits.softmax(dim=-1), 1, generator=generator
        ).squeeze(-1)
        if forced_regimes is None:
            regimes = sampled_regimes
        else:
            forced_regimes = forced_regimes.to(
                device=tokens.device, dtype=torch.long
            )
            regimes = torch.where(
                forced_regimes.ge(0), forced_regimes, sampled_regimes
            )
        rows = torch.arange(tokens.size(0), device=tokens.device)
        selected_degree = degree_logits[rows, :, regimes, :]
        selected_direction = direction_logits[rows, :, regimes, :]
        degrees = torch.multinomial(
            selected_degree.softmax(dim=-1).reshape(-1, 3),
            1,
            generator=generator,
        ).reshape(tokens.shape)
        directions = torch.multinomial(
            selected_direction.softmax(dim=-1).reshape(-1, 2),
            1,
            generator=generator,
        ).reshape(tokens.shape)
        return (
            stops,
            degrees,
            directions,
            regimes,
            self.posterior_states(token_logits),
            token_logits,
        )


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


class PretrainedLengthMaskedModel(nn.Module):
    """Learned length plus mask filling on the same pretrained backbone.

    The control `research/LIKELIHOOD_DECOMPOSITION.md` identifies as missing.
    Every published comparison of the exact tree objective against learned
    lengths plus masks has pitted a pretrained tree model against a 10M
    from-scratch baseline, so it differs in pretraining and capacity as well as
    objective. This class removes the first two differences by giving the
    baseline exactly the backbone the tree model gets.

    Both passes reuse `PretrainedIntervalEncoder.render_prompts`, so the two
    models see identical prompt text. Length prediction reads the single
    mask-token state, exactly as the tree model's root gap context does. Token
    prediction re-renders the prompt with the span's own number of mask tokens
    and reads each of their states, which is the masked-language-model task the
    backbone was pretrained on.
    """

    def __init__(
        self,
        vocab_size: int,
        max_span: int,
        gap_id: int,
        pad_id: int,
        source_tokenizer,
        model_name: str = "distilroberta-base",
        cache_dir: str = ".hf_cache/hub",
        max_length: int = 256,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
        random_init_backbone: bool = False,
        backbone=None,
        pretrained_tokenizer=None,
        initialize_custom_embeddings: bool = True,
        tie_token_embeddings: bool = True,
        bottleneck_context: bool = False,
        native_vocabulary: bool = False,
        pretrained_lm_head=None,
    ) -> None:
        super().__init__()
        if native_vocabulary:
            if backbone is None and pretrained_lm_head is None:
                from transformers import (
                    AutoConfig,
                    AutoModelForMaskedLM,
                    AutoTokenizer,
                )

                if pretrained_tokenizer is None:
                    pretrained_tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        use_fast=True,
                        local_files_only=local_files_only,
                    )
                if random_init_backbone:
                    config = AutoConfig.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                    )
                    masked_lm = AutoModelForMaskedLM.from_config(config)
                else:
                    masked_lm = AutoModelForMaskedLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                    )
                backbone = masked_lm.base_model
                pretrained_lm_head = getattr(masked_lm, "lm_head", None)
            if backbone is None or pretrained_lm_head is None:
                raise ValueError(
                    "native vocabulary needs both a backbone and pretrained MLM head"
                )
        self.encoder = PretrainedIntervalEncoder(
            vocab_size,
            gap_id,
            pad_id,
            source_tokenizer,
            model_name,
            cache_dir,
            max_length=max_length,
            freeze_backbone=freeze_backbone,
            gradient_checkpointing=gradient_checkpointing,
            local_files_only=local_files_only,
            random_init_backbone=random_init_backbone,
            backbone=backbone,
            pretrained_tokenizer=pretrained_tokenizer,
            initialize_custom_embeddings=initialize_custom_embeddings,
            native_vocabulary=native_vocabulary,
        )
        d_model = self.encoder.hidden_size
        self.max_span = max_span
        self.length_head = nn.Linear(d_model, max_span + 1)
        if native_vocabulary:
            self.token_head = pretrained_lm_head
        else:
            self.token_head = nn.Linear(d_model, vocab_size)
            if tie_token_embeddings:
                self.token_head.weight = self.encoder.token_embedding.weight
        # Diagnostic arm. With bottleneck_context the token pass reads the same
        # single mask-token summary vector the interval chart is restricted to,
        # plus a within-span position embedding, instead of one contextualized
        # state per masked position. The objective is unchanged, so comparing
        # the two isolates how much of the tree model's generation deficit is
        # its encoder integration rather than its objective.
        self.bottleneck_context = bottleneck_context
        self.span_position = (
            nn.Embedding(max_span, d_model) if bottleneck_context else None
        )

    @property
    def d_model(self) -> int:
        return self.encoder.hidden_size

    def _encode_with_masks(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        mask_counts: Optional[Sequence[int]],
        mask_residuals: Optional[Sequence[torch.Tensor]] = None,
        residual_scale: float = 0.0,
    ):
        """Encode each prompt with ``mask_counts[i]`` mask tokens in its gap.

        Returns the backbone states at those mask positions, left-padded into a
        ``[batch, max_count, hidden]`` tensor, with a boolean validity mask.
        """
        encoder = self.encoder
        if encoder.native_vocabulary:
            model_inputs = encoder.native_model_inputs(
                tokens, padding_mask, mask_counts
            )
        else:
            texts, _ = encoder.render_prompts(tokens, padding_mask)
            mask_token = encoder.pretrained_tokenizer.mask_token
            if mask_counts is not None:
                rendered = []
                for text, count in zip(texts, mask_counts):
                    # One mask is already present from render_prompts; a zero-length
                    # span still needs a position to read, so keep at least one.
                    rendered.append(text.replace(
                        mask_token, mask_token * max(1, int(count))
                    ))
                texts = rendered
            encoded = encoder.pretrained_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=encoder.max_length,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(tokens.device) for key, value in encoded.items()
            }
        input_ids = model_inputs["input_ids"]
        matches = input_ids.eq(
            int(encoder.pretrained_tokenizer.mask_token_id)
        )
        if residual_scale and mask_residuals is not None:
            inputs_embeds = encoder.backbone.get_input_embeddings()(input_ids)
            inputs_embeds = inputs_embeds.clone()
            for row, residuals in enumerate(mask_residuals):
                positions = matches[row].nonzero().flatten()
                count = min(int(positions.numel()), int(residuals.size(0)))
                if count:
                    inputs_embeds[row, positions[:count]] = (
                        inputs_embeds[row, positions[:count]]
                        + float(residual_scale)
                        * residuals[:count].to(
                            device=inputs_embeds.device,
                            dtype=inputs_embeds.dtype,
                        )
                    )
            backbone_inputs = {
                key: value
                for key, value in model_inputs.items()
                if key != "input_ids"
            }
            hidden = encoder.backbone(
                inputs_embeds=inputs_embeds, **backbone_inputs
            ).last_hidden_state
        else:
            hidden = encoder.backbone(**model_inputs).last_hidden_state
        width = int(matches.sum(dim=1).max().clamp_min(1))
        states = hidden.new_zeros((hidden.size(0), width, hidden.size(-1)))
        valid = torch.zeros(
            (hidden.size(0), width), dtype=torch.bool, device=hidden.device
        )
        for row in range(hidden.size(0)):
            positions = matches[row].nonzero().flatten()
            if not positions.numel():
                # Truncation removed the gap; leave the row invalid.
                continue
            positions = positions[:width]
            states[row, : positions.numel()] = hidden[row, positions]
            valid[row, : positions.numel()] = True
        return encoder.context_norm(states), valid

    def predict_length(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        states, _ = self._encode_with_masks(tokens, padding_mask, None)
        return self.length_head(states[:, 0])

    def predict_tokens(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask_counts: Optional[Sequence[int]] = None,
        mask_residuals: Optional[Sequence[torch.Tensor]] = None,
        residual_scale: float = 0.0,
    ):
        if self.bottleneck_context:
            # One mask, one summary vector, every span token predicted from it.
            summary, _ = self._encode_with_masks(tokens, padding_mask, None)
            width = max(1, max(int(c) for c in mask_counts)) if mask_counts else 1
            width = min(width, int(self.span_position.num_embeddings))
            positions = torch.arange(width, device=summary.device)
            states = summary[:, :1] + self.span_position(positions).unsqueeze(0)
            valid = torch.zeros(
                (states.size(0), width), dtype=torch.bool, device=states.device
            )
            for row, count in enumerate(mask_counts or [width] * states.size(0)):
                valid[row, :min(int(count), width)] = True
            return self.token_head(states), valid
        states, valid = self._encode_with_masks(
            tokens,
            padding_mask,
            mask_counts,
            mask_residuals=mask_residuals,
            residual_scale=residual_scale,
        )
        return self.token_head(states), valid


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
