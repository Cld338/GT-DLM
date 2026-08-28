"""Small bidirectional Transformer models for the mechanism experiment."""

import math
from typing import Optional, Sequence

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
        hidden = encoder.backbone(**model_inputs).last_hidden_state
        matches = model_inputs["input_ids"].eq(
            int(encoder.pretrained_tokenizer.mask_token_id)
        )
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
        states, valid = self._encode_with_masks(tokens, padding_mask, mask_counts)
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
