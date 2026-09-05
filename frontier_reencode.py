"""Training states, losses, and rollout for the re-encoded gap frontier model."""

import math
import random
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gtdlm.data import collate_compact_frontiers
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextGapProposalDataset,
    TextInfillingExample,
    TextVocabulary,
)
from gtdlm.tree import build_pivot_tree, make_tree_frontier, pivot_tree_depth


def topology_targets(
    left_targets: torch.Tensor, right_targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert child bits into offspring count and unary direction targets."""
    if left_targets.shape != right_targets.shape:
        raise ValueError("left and right child targets must align")
    valid = left_targets.ge(0) & right_targets.ge(0)
    degree = torch.full_like(left_targets, -100)
    direction = torch.full_like(left_targets, -100)
    degree[valid] = left_targets[valid] + right_targets[valid]
    unary = valid & degree.eq(1)
    # left-only -> 0; right-only -> 1.
    direction[unary] = right_targets[unary]
    return degree, direction


def truncated_length_convolution(
    left: torch.Tensor,
    right: torch.Tensor,
    cap: int,
) -> torch.Tensor:
    """Convolve length laws with a final overflow category.

    Inputs and output use categories ``0..cap, >cap``. Keeping overflow mass
    explicit makes the projected rollout objective penalize runaway branching
    without allocating a canvas for every possible descendant.
    """
    if cap < 1:
        raise ValueError("length cap must be positive")
    if left.shape != (cap + 2,) or right.shape != (cap + 2,):
        raise ValueError("length distributions must have cap + 2 entries")
    finite_size = cap + 1
    outer = left[:finite_size].unsqueeze(1) * right[:finite_size].unsqueeze(0)
    indices = (
        torch.arange(finite_size, device=left.device).unsqueeze(1)
        + torch.arange(finite_size, device=left.device).unsqueeze(0)
    ).flatten()
    full = left.new_zeros(2 * cap + 1).scatter_add(
        0, indices, outer.flatten()
    )
    finite_tensor = full[:finite_size]
    total_mass = left.sum() * right.sum()
    overflow = (total_mass - finite_tensor.sum()).clamp_min(0.0)
    return torch.cat((finite_tensor, overflow.unsqueeze(0)))


def truncated_length_convolution_batch(
    distributions: torch.Tensor,
    cap: int,
) -> torch.Tensor:
    """Self-convolve a batch of capped length distributions."""
    if distributions.dim() != 2 or distributions.size(1) != cap + 2:
        raise ValueError("batched length laws must have shape [batch, cap + 2]")
    finite_size = cap + 1
    finite = distributions[:, :finite_size]
    outer = finite.unsqueeze(2) * finite.unsqueeze(1)
    base = torch.arange(finite_size, device=distributions.device)
    indices = (base.unsqueeze(1) + base.unsqueeze(0)).flatten()
    full = distributions.new_zeros(
        distributions.size(0), 2 * cap + 1
    ).scatter_add(
        1,
        indices.unsqueeze(0).expand(distributions.size(0), -1),
        outer.flatten(start_dim=1),
    )
    finite_result = full[:, :finite_size]
    total_mass = distributions.sum(dim=1).square()
    overflow = (total_mass - finite_result.sum(dim=1)).clamp_min(0.0)
    return torch.cat((finite_result, overflow.unsqueeze(1)), dim=1)


def shift_length_distribution(
    distribution: torch.Tensor,
    amount: int,
    cap: int,
) -> torch.Tensor:
    """Add a fixed number of emitted tokens to a capped length law."""
    if amount < 0:
        raise ValueError("length shift cannot be negative")
    if distribution.shape != (cap + 2,):
        raise ValueError("length distribution must have cap + 2 entries")
    if amount > cap:
        return torch.cat((
            distribution.new_zeros(cap + 1), distribution.sum().unsqueeze(0)
        ))
    retained = distribution[: cap + 1 - amount]
    finite = torch.cat((distribution.new_zeros(amount), retained))
    overflow = (
        distribution[-1] + distribution[cap + 1 - amount : cap + 1].sum()
    )
    return torch.cat((finite, overflow.unsqueeze(0)))


def projected_total_progeny_distribution(
    degree_probabilities: torch.Tensor,
    cap: int,
    horizon: int,
    completed: int = 0,
    root_stop_probabilities: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project local degree laws to a differentiable terminal length law.

    Each active frontier node recursively reuses its current ``0/1/2`` child
    law for descendants up to ``horizon`` levels. This homogeneous projection
    is deliberately a cheap training surrogate, not a claim that future neural
    states are context-free. It supplies a proper terminal-length NLL while the
    ordinary joint action loss still trains every observed frontier state.
    """
    if degree_probabilities.dim() != 2 or degree_probabilities.size(-1) != 3:
        raise ValueError("degree probabilities must have shape [nodes, 3]")
    if horizon < 1:
        raise ValueError("rollout horizon must be positive")
    if root_stop_probabilities is not None and root_stop_probabilities.shape != (
        degree_probabilities.size(0),
    ):
        raise ValueError("root stop probabilities must align with active nodes")
    delta_zero = degree_probabilities.new_zeros(cap + 2)
    delta_zero[0] = 1.0
    node_count = degree_probabilities.size(0)
    total = delta_zero
    node = shift_length_distribution(delta_zero, 1, cap).unsqueeze(0).expand(
        node_count, -1
    )
    for _ in range(1, horizon):
        two_children = truncated_length_convolution_batch(node, cap)
        children = (
            degree_probabilities[:, 0:1] * delta_zero.unsqueeze(0)
            + degree_probabilities[:, 1:2] * node
            + degree_probabilities[:, 2:3] * two_children
        )
        node = torch.cat((
            children.new_zeros(node_count, 1),
            children[:, :cap],
            (children[:, -1] + children[:, cap]).unsqueeze(1),
        ), dim=1)
    if root_stop_probabilities is not None:
        stop = root_stop_probabilities.unsqueeze(1)
        node = stop * delta_zero.unsqueeze(0) + (1.0 - stop) * node
    for node_distribution in node:
        total = truncated_length_convolution(total, node_distribution, cap)
    return shift_length_distribution(total, completed, cap)


def projected_rollout_length_loss(
    model,
    batch: Dict[str, torch.Tensor],
    vocab: TextVocabulary,
    outputs: Tuple[torch.Tensor, ...],
    cap: int,
    horizon: int,
    detach_backbone: bool = False,
    root_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Proper length NLL from a projected frontier branching process."""
    token_logits, root_stop, degree_logits, _, hidden = outputs
    targets = batch.get("target_lengths")
    if targets is None:
        return token_logits.sum() * 0.0, torch.zeros(
            (), dtype=torch.long, device=token_logits.device
        )
    tokens = batch["tokens"].to(token_logits.device)
    padding = batch["padding"].to(token_logits.device)
    steps = batch["steps"].to(token_logits.device)
    node_ids = batch["node_ids"].to(token_logits.device)
    targets = targets.to(token_logits.device)
    valid = targets.ge(0)
    if root_only:
        valid = valid & steps.eq(0)
    if not bool(valid.any()):
        return token_logits.sum() * 0.0, valid.sum()
    if detach_backbone:
        root_stop, degree_logits, _ = model.structure_logits_from_hidden(
            hidden.detach(), steps
        )
    sample_weights = batch.get("sample_weights")
    if sample_weights is None:
        sample_weights = torch.ones(tokens.size(0), device=token_logits.device)
    else:
        sample_weights = sample_weights.to(token_logits.device)
    losses = []
    weights = []
    for row in valid.nonzero(as_tuple=False).flatten().tolist():
        gaps = tokens[row].eq(vocab.GAP) & ~padding[row]
        probabilities = degree_logits[row, gaps].float().softmax(dim=-1)
        completed = int((node_ids[row].ge(0) & ~tokens[row].eq(vocab.GAP)).sum())
        remaining_horizon = max(1, horizon - int(steps[row]))
        root_probabilities = (
            root_stop[row, gaps].float().sigmoid()
            if int(steps[row]) == 0
            else None
        )
        distribution = projected_total_progeny_distribution(
            probabilities,
            cap=cap,
            horizon=remaining_horizon,
            completed=completed,
            root_stop_probabilities=root_probabilities,
        )
        target = int(targets[row])
        category = target if target <= cap else cap + 1
        losses.append(-distribution[category].clamp_min(1e-8).log())
        weights.append(sample_weights[row])
    stacked_weights = torch.stack(weights)
    loss = (torch.stack(losses) * stacked_weights).sum() / stacked_weights.sum()
    return loss, valid.sum()


def apply_frontier_calibration_biases(model, values: Sequence[float]) -> None:
    """Apply root/base/slope biases used by Monte Carlo length calibration."""
    if len(values) != 7:
        raise ValueError("frontier calibration requires seven bias values")
    root, base, slope = values[0], values[1:4], values[4:7]
    steps = torch.arange(
        model.calibration_degree_bias.size(0),
        device=model.calibration_degree_bias.device,
    ).unsqueeze(-1).to(model.calibration_degree_bias.dtype)
    base_tensor = torch.tensor(
        base, device=model.calibration_degree_bias.device
    ).to(model.calibration_degree_bias.dtype)
    slope_tensor = torch.tensor(
        slope, device=model.calibration_degree_bias.device
    ).to(model.calibration_degree_bias.dtype)
    with torch.no_grad():
        model.calibration_root_bias.fill_(float(root))
        model.calibration_degree_bias.copy_(
            base_tensor.unsqueeze(0) + steps * slope_tensor.unsqueeze(0)
        )


class RandomFrontierDataset(Dataset):
    """Draw one balanced or near-balanced frontier per document and epoch."""

    def __init__(
        self,
        source: DynamicTextExampleDataset,
        vocab: TextVocabulary,
        strategy: str = "mixed",
        midpoint_probability: float = 0.7,
        seed_offset: int = 83,
    ) -> None:
        self.source = source
        self.vocab = vocab
        self.strategy = strategy
        self.midpoint_probability = midpoint_probability
        self.seed_offset = seed_offset

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.source[index]
        state_seed = self.source.example_seed(index) + self.seed_offset
        states = TextGapProposalDataset(
            [example],
            self.vocab,
            strategy=self.strategy,
            seed=state_seed,
            midpoint_probability=self.midpoint_probability,
        )
        choice = random.Random(state_seed + 17).randrange(len(states))
        state = dict(states[choice])
        # The input contains only the observed context, already emitted tokens,
        # and current gaps. Hidden span length is never included.
        state["sample_weight"] = float(len(states))
        # Prefix states let training reconstruct the exact gold-topology
        # derivation while replacing lexical ancestors with model samples.
        state["history_states"] = [
            dict(states[level]) for level in range(choice)
        ]
        state["source_example"] = example
        return state


def collate_frontiers_with_history(
    examples: Sequence[Dict[str, object]], pad_id: int
) -> Dict[str, object]:
    """Collate current frontiers while retaining raw derivation prefixes."""
    batch: Dict[str, object] = dict(collate_compact_frontiers(examples, pad_id))
    batch["history_states"] = [
        example.get("history_states", []) for example in examples
    ]
    batch["source_examples"] = [
        example.get("source_example") for example in examples
    ]
    return batch


@torch.no_grad()
def replace_with_generated_history(
    model,
    batch: Dict[str, object],
    vocab: TextVocabulary,
    device: torch.device,
    probability: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, int]:
    """Roll out lexical ancestors under gold topology and insert them.

    The topology and current targets remain teacher-forced. Only tokens emitted
    in earlier parallel rounds are sampled from the model, so the current
    frontier is trained on histories it can actually produce at inference.
    Returns the number of selected examples and replaced ancestor tokens.
    """
    if probability <= 0.0:
        return 0, 0
    histories = batch.get("history_states")
    if not isinstance(histories, list):
        raise ValueError("generated-history batches must retain history_states")
    selected = torch.rand(
        len(histories), device=device, generator=generator
    ).lt(probability).tolist()
    generated = [dict() for _ in histories]
    maximum_depth = max((len(states) for states in histories), default=0)
    generated_ids = torch.tensor(
        vocab.generated_token_ids, dtype=torch.long, device=device
    )
    was_training = model.training
    model.eval()
    try:
        for depth in range(maximum_depth):
            owners = [
                row for row, states in enumerate(histories)
                if selected[row] and depth < len(states)
            ]
            if not owners:
                continue
            states = []
            for owner in owners:
                state = dict(histories[owner][depth])
                tokens = list(state["tokens"])
                targets = list(state["targets"])
                node_ids = list(state.get("node_ids", []))
                for position, node_id in enumerate(node_ids):
                    if targets[position] < 0 and node_id in generated[owner]:
                        tokens[position] = generated[owner][node_id]
                state["tokens"] = tokens
                states.append(state)
            prefix = collate_compact_frontiers(states, vocab.PAD)
            tokens = prefix["tokens"].to(device)
            padding = prefix["padding"].to(device)
            steps = prefix["steps"].to(device)
            targets = prefix["targets"].to(device)
            node_ids = prefix["node_ids"].to(device)
            token_logits, _, degree_logits, direction_logits, hidden = model(
                tokens, padding, steps
            )
            active = targets.ge(0) & targets.lt(vocab.vocab_size)
            if not bool(active.any()):
                continue
            if getattr(model, "direct_joint_actions", False):
                active_steps = steps.unsqueeze(1).expand_as(tokens)[active]
                joint_logp = model.joint_action_log_probs(
                    token_logits[active],
                    degree_logits[active],
                    direction_logits[active],
                    hidden[active],
                    active_steps,
                    generated_ids,
                )
                token_probabilities = torch.logsumexp(
                    joint_logp, dim=-1
                ).exp()
            else:
                token_probabilities = token_logits[active].index_select(
                    -1, generated_ids
                ).softmax(dim=-1)
            sampled = generated_ids[torch.multinomial(
                token_probabilities, 1, generator=generator
            ).squeeze(-1)]
            active_locations = active.nonzero(as_tuple=False)
            for index, (prefix_row, position) in enumerate(active_locations.tolist()):
                owner = owners[prefix_row]
                node_id = int(node_ids[prefix_row, position])
                if node_id < 0:
                    raise ValueError("active frontier node is missing its node id")
                generated[owner][node_id] = int(sampled[index])
    finally:
        model.train(was_training)

    tokens = batch["tokens"]
    targets = batch["targets"]
    node_ids = batch["node_ids"]
    if not all(isinstance(value, torch.Tensor) for value in (tokens, targets, node_ids)):
        raise TypeError("collated frontier tensors are required")
    replacements = 0
    for row, use_generated in enumerate(selected):
        if not use_generated:
            continue
        for position, node_id in enumerate(node_ids[row].tolist()):
            if targets[row, position] < 0 and node_id in generated[row]:
                tokens[row, position] = generated[row][node_id]
                replacements += 1
    return sum(selected), replacements


class ScaffoldProposalDataset(Dataset):
    """Topology frontiers whose already-emitted nodes remain lexical slots."""

    def __init__(
        self,
        examples: Sequence[TextInfillingExample],
        vocab: TextVocabulary,
        strategy: str,
        seed: int,
        midpoint_probability: float = 0.7,
    ) -> None:
        self.examples: List[Dict[str, object]] = []
        for example_index, example in enumerate(examples):
            trees = []
            for gap_index, span in enumerate(example.spans):
                tree_seed = (
                    seed * 1_000_003
                    + example_index * 9_176
                    + gap_index * 7_919
                )
                trees.append(build_pivot_tree(
                    0,
                    len(span),
                    strategy=strategy,
                    rng=random.Random(tree_seed),
                    midpoint_probability=midpoint_probability,
                ))
            depth = max(1, *(pivot_tree_depth(tree) for tree in trees))
            for level in range(depth):
                tokens = [vocab.LEFT]
                targets = [-100]
                left_targets = [-100]
                right_targets = [-100]
                semantic_tokens = [-100]
                for gap_index, (span, tree) in enumerate(
                    zip(example.spans, trees)
                ):
                    context = list(example.segments[gap_index])
                    tokens.extend(context)
                    targets.extend([-100] * len(context))
                    left_targets.extend([-100] * len(context))
                    right_targets.extend([-100] * len(context))
                    semantic_tokens.extend([-100] * len(context))
                    frontier = make_tree_frontier(
                        span,
                        tree,
                        level,
                        vocab.GAP,
                        vocab.stop_action,
                    )
                    # Both completed nodes and current open gaps are rendered
                    # as native mask tokens. Targets identify the open subset.
                    tokens.extend([vocab.GAP] * len(frontier[0]))
                    targets.extend(frontier[1])
                    left_targets.extend(frontier[2])
                    right_targets.extend(frontier[3])
                    semantic_tokens.extend([
                        int(token) if int(target) == -100 else -100
                        for token, target in zip(frontier[0], frontier[1])
                    ])
                context = list(example.segments[-1])
                tokens.extend(context)
                targets.extend([-100] * len(context))
                left_targets.extend([-100] * len(context))
                right_targets.extend([-100] * len(context))
                semantic_tokens.extend([-100] * len(context))
                tokens.append(vocab.RIGHT)
                targets.append(-100)
                left_targets.append(-100)
                right_targets.append(-100)
                semantic_tokens.append(-100)
                self.examples.append({
                    "tokens": tokens,
                    "targets": targets,
                    "left_targets": left_targets,
                    "right_targets": right_targets,
                    "semantic_tokens": semantic_tokens,
                    "step": level,
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.examples[index]


class RandomScaffoldFrontierDataset(Dataset):
    """Draw one unbiased mask-scaffold frontier per document and epoch."""

    def __init__(
        self,
        source: DynamicTextExampleDataset,
        vocab: TextVocabulary,
        strategy: str = "mixed",
        midpoint_probability: float = 0.7,
        seed_offset: int = 131,
    ) -> None:
        self.source = source
        self.vocab = vocab
        self.strategy = strategy
        self.midpoint_probability = midpoint_probability
        self.seed_offset = seed_offset

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Dict[str, object]:
        state_seed = self.source.example_seed(index) + self.seed_offset
        states = ScaffoldProposalDataset(
            [self.source[index]],
            self.vocab,
            strategy=self.strategy,
            seed=state_seed,
            midpoint_probability=self.midpoint_probability,
        )
        choice = random.Random(state_seed + 17).randrange(len(states))
        state = dict(states[choice])
        state["sample_weight"] = float(len(states))
        return state


class DynamicScaffoldDerivationDataset(Dataset):
    """Return every scaffold frontier from one dynamic document corruption."""

    def __init__(
        self,
        source: DynamicTextExampleDataset,
        vocab: TextVocabulary,
        strategy: str = "mixed",
        midpoint_probability: float = 0.7,
        seed_offset: int = 173,
    ) -> None:
        self.source = source
        self.vocab = vocab
        self.strategy = strategy
        self.midpoint_probability = midpoint_probability
        self.seed_offset = seed_offset

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> List[Dict[str, object]]:
        seed = self.source.example_seed(index) + self.seed_offset
        states = ScaffoldProposalDataset(
            [self.source[index]],
            self.vocab,
            strategy=self.strategy,
            seed=seed,
            midpoint_probability=self.midpoint_probability,
        )
        return [dict(states[state]) for state in range(len(states))]


class FixedScaffoldDerivationDataset(Dataset):
    """Deterministic complete derivations for held-out evaluation."""

    def __init__(
        self,
        examples: Sequence[TextInfillingExample],
        vocab: TextVocabulary,
        strategy: str,
        seed: int,
        midpoint_probability: float = 0.7,
    ) -> None:
        self.derivations: List[List[Dict[str, object]]] = []
        for index, example in enumerate(examples):
            states = ScaffoldProposalDataset(
                [example],
                vocab,
                strategy=strategy,
                seed=seed + index * 9_176,
                midpoint_probability=midpoint_probability,
            )
            self.derivations.append([
                dict(states[state]) for state in range(len(states))
            ])

    def __len__(self) -> int:
        return len(self.derivations)

    def __getitem__(self, index: int) -> List[Dict[str, object]]:
        return self.derivations[index]


def scaffold_topology_losses(
    model,
    batch: Dict[str, torch.Tensor],
    vocab: TextVocabulary,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Marginalize one shared shape regime across every gap in a frontier."""
    tokens = batch["tokens"].to(device)
    padding = batch["padding"].to(device)
    steps = batch["steps"].to(device)
    targets = batch["targets"].to(device)
    left = batch["left_targets"].to(device)
    right = batch["right_targets"].to(device)
    weights = batch.get("sample_weights")
    weights = (
        torch.ones(tokens.size(0), device=device)
        if weights is None
        else weights.to(device)
    )
    degree_targets, direction_targets = topology_targets(left, right)
    open_mask = targets.ne(-100)
    continuous_semantic = bool(
        getattr(model, "continuous_semantic", False)
    )
    semantic_enabled = bool(getattr(model, "semantic_codes", 0))
    if continuous_semantic:
        semantic_tokens = batch["semantic_tokens"].to(device)
        slot_semantics = torch.zeros(
            *semantic_tokens.shape,
            model.d_model,
            device=device,
            dtype=model.semantic_embedding_mean.dtype,
        )
        completed = semantic_tokens.ge(0)
        slot_semantics[completed] = model.token_semantic_states(
            semantic_tokens[completed]
        )
        outputs = model.structure_logits(
            tokens,
            padding,
            steps,
            open_mask,
            slot_semantics=slot_semantics,
            return_continuous_semantic=True,
        )
        root, regime_logits, degree_logits, direction_logits, _, predicted_state = (
            outputs
        )
        lexical = targets.ge(0) & targets.lt(vocab.vocab_size)
        target_state = model.token_semantic_states(targets[lexical])
        semantic_terms = 1.0 - F.cosine_similarity(
            predicted_state[lexical].float(),
            target_state.float(),
            dim=-1,
        )
        semantic_weights = (
            weights.unsqueeze(1).expand_as(targets)[lexical]
        )
        semantic_loss = (
            semantic_terms * semantic_weights
        ).sum() / semantic_weights.sum().clamp_min(1.0)
        semantic_count = lexical.sum()
    elif semantic_enabled:
        semantic_tokens = batch["semantic_tokens"].to(device)
        slot_codes = torch.full_like(semantic_tokens, -1)
        completed = semantic_tokens.ge(0)
        slot_codes[completed] = model.semantic_token_codes[
            semantic_tokens[completed]
        ]
        outputs = model.structure_logits(
            tokens,
            padding,
            steps,
            open_mask,
            slot_codes=slot_codes,
            return_semantic=True,
        )
        root, regime_logits, degree_logits, direction_logits, _, semantic_logits = (
            outputs
        )
        semantic_targets = torch.full_like(targets, -100)
        lexical = targets.ge(0) & targets.lt(vocab.vocab_size)
        semantic_targets[lexical] = model.semantic_token_codes[targets[lexical]]
        semantic_terms = F.cross_entropy(
            semantic_logits.transpose(1, 2),
            semantic_targets,
            reduction="none",
            ignore_index=-100,
        )
        semantic_weights = weights.unsqueeze(1) * lexical.to(weights.dtype)
        semantic_loss = (
            semantic_terms * semantic_weights
        ).sum() / semantic_weights.sum().clamp_min(1.0)
        semantic_count = lexical.sum()
    else:
        root, regime_logits, degree_logits, direction_logits, _ = (
            model.structure_logits(tokens, padding, steps, open_mask)
        )
        semantic_loss = regime_logits.sum() * 0.0
        semantic_count = torch.zeros((), dtype=torch.long, device=device)

    root_rows = steps.eq(0) & open_mask.any(dim=1)
    root_positions = open_mask.to(torch.long).argmax(dim=1)
    root_values = root[torch.arange(tokens.size(0), device=device), root_positions]
    root_targets = targets.eq(vocab.stop_action).any(dim=1).to(root.dtype)
    if bool(root_rows.any()):
        root_terms = F.binary_cross_entropy_with_logits(
            root_values[root_rows], root_targets[root_rows], reduction="none"
        )
        root_loss = (
            root_terms * weights[root_rows]
        ).sum() / weights[root_rows].sum().clamp_min(1.0)
    else:
        root_loss = root.sum() * 0.0

    topology_terms = []
    topology_weights = []
    topology_nodes = 0
    direction_nodes = 0
    for row in range(tokens.size(0)):
        valid = degree_targets[row].ge(0)
        if not bool(valid.any()):
            continue
        regime_logp = regime_logits[row].log_softmax(dim=-1)
        degree_logp = degree_logits[row, valid].log_softmax(dim=-1)
        degree_target = degree_targets[row, valid]
        gathered_degree = degree_logp.gather(
            -1,
            degree_target.view(-1, 1, 1).expand(-1, model.regimes, 1),
        ).squeeze(-1).sum(0)
        joint = regime_logp + gathered_degree
        unary = direction_targets[row, valid].ge(0)
        if bool(unary.any()):
            direction_logp = direction_logits[row, valid][unary].log_softmax(
                dim=-1
            )
            direction_target = direction_targets[row, valid][unary]
            joint = joint + direction_logp.gather(
                -1,
                direction_target.view(-1, 1, 1).expand(
                    -1, model.regimes, 1
                ),
            ).squeeze(-1).sum(0)
            direction_nodes += int(unary.sum())
        topology_terms.append(-torch.logsumexp(joint, dim=0))
        topology_weights.append(weights[row])
        topology_nodes += int(valid.sum())
    if topology_terms:
        stacked_terms = torch.stack(topology_terms)
        stacked_weights = torch.stack(topology_weights)
        topology_loss = (
            stacked_terms * stacked_weights
        ).sum() / stacked_weights.sum().clamp_min(1.0)
    else:
        topology_loss = regime_logits.sum() * 0.0
    return {
        "root": root_loss,
        "topology": topology_loss,
        "semantic": semantic_loss,
        "root_count": root_rows.sum(),
        "frontier_count": torch.tensor(
            len(topology_terms), device=device, dtype=torch.long
        ),
        "degree_count": torch.tensor(
            topology_nodes, device=device, dtype=torch.long
        ),
        "direction_count": torch.tensor(
            direction_nodes, device=device, dtype=torch.long
        ),
        "semantic_count": semantic_count,
    }


def unified_scaffold_losses(
    model,
    batch: Dict[str, torch.Tensor],
    vocab: TextVocabulary,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Joint lexical/topology objective for the unified grow-and-denoise LM.

    A baseline pass forms soft token posteriors for completed nodes.  The joint
    pass re-encodes those node-local beliefs and predicts both the next tree
    expansion and all known lexical targets.  The KL term anchors the joint
    token distribution to the supplied masked-LM baseline.
    """
    tokens = batch["tokens"].to(device)
    padding = batch["padding"].to(device)
    steps = batch["steps"].to(device)
    targets = batch["targets"].to(device)
    semantic_tokens = batch["semantic_tokens"].to(device)
    left = batch["left_targets"].to(device)
    right = batch["right_targets"].to(device)
    weights = batch.get("sample_weights")
    weights = (
        torch.ones(tokens.size(0), device=device)
        if weights is None
        else weights.to(device)
    )
    open_mask = targets.ne(-100)
    degree_targets, direction_targets = topology_targets(left, right)

    # This pass is also the exact zero-gate lexical reference.  Posterior
    # states are predictions rather than gold token embeddings, preventing a
    # train/inference information leak into the shape controller.
    with torch.no_grad():
        reference_logits = model.unified_logits(
            tokens, padding, steps, open_mask
        )[0]
        predicted_states = model.posterior_states(reference_logits)
    slot_semantics = torch.zeros_like(predicted_states)
    completed = semantic_tokens.ge(0)
    slot_semantics[completed] = predicted_states[completed]
    outputs = model.unified_logits(
        tokens,
        padding,
        steps,
        open_mask,
        slot_semantics=slot_semantics,
        semantic_requires_grad=True,
    )
    token_logits, root, regime_logits, degree_logits, direction_logits = (
        outputs[:5]
    )

    lexical_targets = torch.where(targets.ge(0), targets, semantic_tokens)
    lexical = lexical_targets.ge(0) & lexical_targets.lt(vocab.vocab_size)
    lexical_ce_targets = lexical_targets.masked_fill(~lexical, -100)
    lexical_terms = F.cross_entropy(
        token_logits.transpose(1, 2),
        lexical_ce_targets,
        reduction="none",
        ignore_index=-100,
    )
    lexical_weights = weights.unsqueeze(1) * lexical.to(weights.dtype)
    lexical_loss = (
        lexical_terms * lexical_weights
    ).sum() / lexical_weights.sum().clamp_min(1.0)

    generated = model.generated_token_ids.to(device)
    joint_allowed = token_logits.index_select(-1, generated)
    reference_allowed = reference_logits.index_select(-1, generated)
    anchor_terms = F.kl_div(
        joint_allowed.log_softmax(dim=-1),
        reference_allowed.softmax(dim=-1),
        reduction="none",
    ).sum(dim=-1)
    anchor_loss = (
        anchor_terms * lexical_weights
    ).sum() / lexical_weights.sum().clamp_min(1.0)

    root_rows = steps.eq(0) & open_mask.any(dim=1)
    root_positions = open_mask.to(torch.long).argmax(dim=1)
    root_values = root[
        torch.arange(tokens.size(0), device=device), root_positions
    ]
    root_targets = targets.eq(vocab.stop_action).any(dim=1).to(root.dtype)
    if bool(root_rows.any()):
        root_terms = F.binary_cross_entropy_with_logits(
            root_values[root_rows], root_targets[root_rows], reduction="none"
        )
        root_loss = (
            root_terms * weights[root_rows]
        ).sum() / weights[root_rows].sum().clamp_min(1.0)
    else:
        root_loss = root.sum() * 0.0

    topology_terms = []
    topology_weights = []
    topology_nodes = 0
    direction_nodes = 0
    for row in range(tokens.size(0)):
        valid = degree_targets[row].ge(0)
        if not bool(valid.any()):
            continue
        regime_logp = regime_logits[row].log_softmax(dim=-1)
        degree_logp = degree_logits[row, valid].log_softmax(dim=-1)
        degree_target = degree_targets[row, valid]
        gathered_degree = degree_logp.gather(
            -1,
            degree_target.view(-1, 1, 1).expand(-1, model.regimes, 1),
        ).squeeze(-1).sum(0)
        joint = regime_logp + gathered_degree
        unary = direction_targets[row, valid].ge(0)
        if bool(unary.any()):
            direction_logp = direction_logits[row, valid][unary].log_softmax(
                dim=-1
            )
            direction_target = direction_targets[row, valid][unary]
            joint = joint + direction_logp.gather(
                -1,
                direction_target.view(-1, 1, 1).expand(
                    -1, model.regimes, 1
                ),
            ).squeeze(-1).sum(0)
            direction_nodes += int(unary.sum())
        topology_terms.append(-torch.logsumexp(joint, dim=0))
        topology_weights.append(weights[row])
        topology_nodes += int(valid.sum())
    if topology_terms:
        stacked_terms = torch.stack(topology_terms)
        stacked_weights = torch.stack(topology_weights)
        topology_loss = (
            stacked_terms * stacked_weights
        ).sum() / stacked_weights.sum().clamp_min(1.0)
    else:
        topology_loss = regime_logits.sum() * 0.0
    return {
        "root": root_loss,
        "topology": topology_loss,
        "lexical": lexical_loss,
        "anchor": anchor_loss,
        "root_count": root_rows.sum(),
        "frontier_count": torch.tensor(
            len(topology_terms), device=device, dtype=torch.long
        ),
        "degree_count": torch.tensor(
            topology_nodes, device=device, dtype=torch.long
        ),
        "direction_count": torch.tensor(
            direction_nodes, device=device, dtype=torch.long
        ),
        "lexical_count": lexical.sum(),
    }


def persistent_scaffold_losses(
    model,
    derivations: Sequence[Sequence[Dict[str, object]]],
    vocab: TextVocabulary,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Marginalize one shape regime shared by an entire tree derivation."""
    flat_states = [state for rows in derivations for state in rows]
    if not flat_states:
        raise ValueError("a derivation batch cannot be empty")
    owners = torch.tensor(
        [owner for owner, rows in enumerate(derivations) for _ in rows],
        dtype=torch.long,
        device=device,
    )
    batch = collate_compact_frontiers(flat_states, vocab.PAD)
    tokens = batch["tokens"].to(device)
    padding = batch["padding"].to(device)
    steps = batch["steps"].to(device)
    targets = batch["targets"].to(device)
    left = batch["left_targets"].to(device)
    right = batch["right_targets"].to(device)
    degree_targets, direction_targets = topology_targets(left, right)
    open_mask = targets.ne(-100)
    root, regime_logits, degree_logits, direction_logits, _ = (
        model.structure_logits(tokens, padding, steps, open_mask)
    )

    derivation_count = len(derivations)
    root_rows = torch.full(
        (derivation_count,), -1, dtype=torch.long, device=device
    )
    for row, owner in enumerate(owners.tolist()):
        if int(steps[row]) == 0:
            root_rows[owner] = row
    if bool(root_rows.lt(0).any()):
        raise ValueError("every derivation must contain a root frontier")
    root_positions = open_mask[root_rows].to(torch.long).argmax(dim=1)
    root_values = root[
        root_rows,
        root_positions,
    ]
    root_targets = targets[root_rows].eq(vocab.stop_action).any(dim=1).to(
        root.dtype
    )
    root_loss = F.binary_cross_entropy_with_logits(root_values, root_targets)

    regime_logp = regime_logits[root_rows].log_softmax(dim=-1)
    action_logp = torch.zeros_like(regime_logp)
    degree_nodes = 0
    direction_nodes = 0
    for row, owner in enumerate(owners.tolist()):
        valid = degree_targets[row].ge(0)
        if not bool(valid.any()):
            continue
        degree_logp = degree_logits[row, valid].log_softmax(dim=-1)
        degree_target = degree_targets[row, valid]
        action_logp[owner] = action_logp[owner] + degree_logp.gather(
            -1,
            degree_target.view(-1, 1, 1).expand(-1, model.regimes, 1),
        ).squeeze(-1).sum(0)
        degree_nodes += int(valid.sum())
        unary = direction_targets[row, valid].ge(0)
        if bool(unary.any()):
            direction_logp = direction_logits[row, valid][unary].log_softmax(
                dim=-1
            )
            direction_target = direction_targets[row, valid][unary]
            action_logp[owner] = action_logp[owner] + direction_logp.gather(
                -1,
                direction_target.view(-1, 1, 1).expand(
                    -1, model.regimes, 1
                ),
            ).squeeze(-1).sum(0)
            direction_nodes += int(unary.sum())
    topology_loss = -torch.logsumexp(regime_logp + action_logp, dim=-1).mean()
    return {
        "root": root_loss,
        "topology": topology_loss,
        "derivation_count": torch.tensor(
            derivation_count, device=device, dtype=torch.long
        ),
        "degree_count": torch.tensor(
            degree_nodes, device=device, dtype=torch.long
        ),
        "direction_count": torch.tensor(
            direction_nodes, device=device, dtype=torch.long
        ),
    }


def markov_scaffold_losses(
    model,
    derivations: Sequence[Sequence[Dict[str, object]]],
    vocab: TextVocabulary,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Marginalize a depth-wise Markov shape regime over each derivation.

    All sibling actions in one frontier share z_t. Adjacent frontier regimes
    are linked by p(z_t | z_(t-1)) and the complete latent chain is summed
    exactly with the HMM forward algorithm.
    """
    flat_states = [state for rows in derivations for state in rows]
    if not flat_states:
        raise ValueError("a derivation batch cannot be empty")
    owners = [owner for owner, rows in enumerate(derivations) for _ in rows]
    batch = collate_compact_frontiers(flat_states, vocab.PAD)
    tokens = batch["tokens"].to(device)
    padding = batch["padding"].to(device)
    steps = batch["steps"].to(device)
    targets = batch["targets"].to(device)
    left = batch["left_targets"].to(device)
    right = batch["right_targets"].to(device)
    degree_targets, direction_targets = topology_targets(left, right)
    open_mask = targets.ne(-100)
    root, regime_logits, degree_logits, direction_logits, _ = (
        model.structure_logits(tokens, padding, steps, open_mask)
    )

    derivation_count = len(derivations)
    rows_by_owner: List[List[int]] = [[] for _ in derivations]
    for row, owner in enumerate(owners):
        rows_by_owner[owner].append(row)
    for rows in rows_by_owner:
        rows.sort(key=lambda row: int(steps[row]))
        if not rows or int(steps[rows[0]]) != 0:
            raise ValueError("every derivation must contain a root frontier")

    root_rows = torch.tensor(
        [rows[0] for rows in rows_by_owner], dtype=torch.long, device=device
    )
    root_positions = open_mask[root_rows].to(torch.long).argmax(dim=1)
    root_values = root[root_rows, root_positions]
    root_targets = targets[root_rows].eq(vocab.stop_action).any(dim=1).to(
        root.dtype
    )
    root_loss = F.binary_cross_entropy_with_logits(root_values, root_targets)

    emissions: List[torch.Tensor] = []
    degree_nodes = 0
    direction_nodes = 0
    for row in range(tokens.size(0)):
        valid = degree_targets[row].ge(0)
        emission = regime_logits[row] * 0.0
        if bool(valid.any()):
            degree_logp = degree_logits[row, valid].log_softmax(dim=-1)
            degree_target = degree_targets[row, valid]
            emission = emission + degree_logp.gather(
                -1,
                degree_target.view(-1, 1, 1).expand(
                    -1, model.regimes, 1
                ),
            ).squeeze(-1).sum(0)
            degree_nodes += int(valid.sum())
            unary = direction_targets[row, valid].ge(0)
            if bool(unary.any()):
                direction_logp = direction_logits[row, valid][
                    unary
                ].log_softmax(dim=-1)
                direction_target = direction_targets[row, valid][unary]
                emission = emission + direction_logp.gather(
                    -1,
                    direction_target.view(-1, 1, 1).expand(
                        -1, model.regimes, 1
                    ),
                ).squeeze(-1).sum(0)
                direction_nodes += int(unary.sum())
        emissions.append(emission)

    derivation_logp = []
    for rows in rows_by_owner:
        first = rows[0]
        alpha = regime_logits[first].log_softmax(dim=-1) + emissions[first]
        for row in rows[1:]:
            step = steps[row].clamp(0, model.max_steps - 1)
            transition_logp = model.regime_transition[step].log_softmax(
                dim=-1
            )
            alpha = torch.logsumexp(
                alpha.unsqueeze(1) + transition_logp, dim=0
            ) + emissions[row]
        derivation_logp.append(torch.logsumexp(alpha, dim=-1))
    topology_loss = -torch.stack(derivation_logp).mean()
    return {
        "root": root_loss,
        "topology": topology_loss,
        "derivation_count": torch.tensor(
            derivation_count, device=device, dtype=torch.long
        ),
        "degree_count": torch.tensor(
            degree_nodes, device=device, dtype=torch.long
        ),
        "direction_count": torch.tensor(
            direction_nodes, device=device, dtype=torch.long
        ),
    }


def frontier_losses(
    model,
    batch: Dict[str, torch.Tensor],
    vocab: TextVocabulary,
    device: torch.device,
    rollout_length_cap: int = 0,
    rollout_length_horizon: int = 8,
    rollout_length_detach_backbone: bool = False,
    rollout_length_root_only: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute separated lexical, root-stop, degree, and direction losses."""
    tokens = batch["tokens"].to(device)
    padding = batch["padding"].to(device)
    steps = batch["steps"].to(device)
    targets = batch["targets"].to(device)
    left = batch["left_targets"].to(device)
    right = batch["right_targets"].to(device)
    sample_weights = batch.get("sample_weights")
    if sample_weights is None:
        sample_weights = torch.ones(tokens.size(0), device=device)
    else:
        sample_weights = sample_weights.to(device)
    position_weights = sample_weights.unsqueeze(1).expand_as(tokens)
    degree_targets, direction_targets = topology_targets(left, right)

    token_valid = targets.ge(0) & targets.lt(vocab.vocab_size)
    # Teacher-force the gold pivot into the branching heads when the model
    # treats emitting a token and growing the tree as one decision.
    structure_token_ids = (
        torch.where(token_valid, targets, torch.full_like(targets, -1))
        if getattr(model, "token_conditioned_topology", False)
        else None
    )
    node_features = {}
    if getattr(model, "per_node_frontier_features", False):
        node_features = {
            "node_depths": batch["node_depths"].to(device),
            "node_ages": batch["node_ages"].to(device),
        }
    if node_features:
        outputs = model(
            tokens,
            padding,
            steps,
            structure_token_ids=structure_token_ids,
            **node_features,
        )
    elif structure_token_ids is not None:
        outputs = model(tokens, padding, steps, structure_token_ids)
    else:
        outputs = model(tokens, padding, steps)
    token_logits, root_stop, degree_logits, direction_logits, _ = outputs
    root_valid = tokens.eq(vocab.GAP) & steps.unsqueeze(1).eq(0)
    root_targets = targets.eq(vocab.stop_action).to(root_stop.dtype)
    degree_valid = degree_targets.ge(0) & token_valid
    direction_valid = direction_targets.ge(0) & token_valid

    zero = token_logits.sum() * 0.0
    token_loss = (
        (
            F.cross_entropy(
                token_logits[token_valid],
                targets[token_valid],
                reduction="none",
            )
            * position_weights[token_valid]
        ).sum()
        / position_weights[token_valid].sum().clamp_min(1.0)
        if bool(token_valid.any())
        else zero
    )
    root_loss = (
        (
            F.binary_cross_entropy_with_logits(
                root_stop[root_valid],
                root_targets[root_valid],
                reduction="none",
            )
            * position_weights[root_valid]
        ).sum()
        / position_weights[root_valid].sum().clamp_min(1.0)
        if bool(root_valid.any())
        else zero
    )
    degree_loss = (
        (
            F.cross_entropy(
                degree_logits[degree_valid],
                degree_targets[degree_valid],
                reduction="none",
            )
            * position_weights[degree_valid]
        ).sum()
        / position_weights[degree_valid].sum().clamp_min(1.0)
        if bool(degree_valid.any())
        else zero
    )
    direction_loss = (
        (
            F.cross_entropy(
                direction_logits[direction_valid],
                direction_targets[direction_valid],
                reduction="none",
            )
            * position_weights[direction_valid]
        ).sum()
        / position_weights[direction_valid].sum().clamp_min(1.0)
        if bool(direction_valid.any())
        else zero
    )
    if bool(degree_valid.any()):
        node_degree = degree_targets[degree_valid]
        node_direction = direction_targets[degree_valid]
        marker_targets = torch.where(
            node_degree.eq(0),
            torch.zeros_like(node_degree),
            torch.where(
                node_degree.eq(2),
                torch.full_like(node_degree, 3),
                1 + node_direction,
            ),
        )
        node_degree_logp = degree_logits[degree_valid].log_softmax(dim=-1)
        node_direction_logp = direction_logits[degree_valid].log_softmax(dim=-1)
        independent_marker_logp = torch.stack((
            node_degree_logp[..., 0],
            node_degree_logp[..., 1] + node_direction_logp[..., 0],
            node_degree_logp[..., 1] + node_direction_logp[..., 1],
            node_degree_logp[..., 2],
        ), dim=-1)
        marker_terms = F.nll_loss(
            independent_marker_logp,
            marker_targets,
            reduction="none",
        )
        marker_weights = position_weights[degree_valid]
        marker_loss = (
            marker_terms * marker_weights
        ).sum() / marker_weights.sum().clamp_min(1.0)
        marker_count = degree_valid.sum()
    else:
        marker_loss = zero
        marker_count = torch.zeros((), dtype=torch.long, device=device)
    marginal_joint = bool(
        getattr(model, "marginal_preserving_joint", False)
    )
    direct_joint = bool(getattr(model, "direct_joint_actions", False))
    if (marginal_joint or direct_joint) and bool(degree_valid.any()):
        generated_ids = torch.tensor(
            vocab.generated_token_ids, dtype=torch.long, device=device
        )
        token_to_generated = torch.full(
            (vocab.vocab_size,), -1, dtype=torch.long, device=device
        )
        token_to_generated[generated_ids] = torch.arange(
            generated_ids.numel(), device=device
        )
        node_steps = steps.unsqueeze(1).expand_as(tokens)[degree_valid]
        joint_token_logits = token_logits[degree_valid]
        if marginal_joint:
            # In the copula experiment, the separate token loss is the only
            # lexical gradient path. Direct semantic branching instead learns
            # token and marker from the full joint likelihood.
            joint_token_logits = joint_token_logits.detach()
        joint_logp = model.joint_action_log_probs(
            joint_token_logits,
            degree_logits[degree_valid],
            direction_logits[degree_valid],
            outputs[-1][degree_valid],
            node_steps,
            generated_ids,
        )
        token_indices = token_to_generated[targets[degree_valid]]
        if bool(token_indices.lt(0).any()):
            raise ValueError("frontier target is outside generated vocabulary")
        rows = torch.arange(token_indices.numel(), device=device)
        gold_joint = joint_logp[
            rows, token_indices, marker_targets
        ]
        allowed_logp = token_logits[degree_valid].detach().index_select(
            -1, generated_ids
        ).log_softmax(dim=-1)
        gold_token = allowed_logp[rows, token_indices]
        # Marginal-preserving coupling trains only p(marker | token), leaving
        # the separate token loss as the lexical term. Direct semantic
        # branching trains the full p(token, marker) table because rollout
        # samples that same joint action without teacher forcing.
        joint_terms = (
            -(gold_joint - gold_token) if marginal_joint else -gold_joint
        )
        node_batch_rows = torch.arange(
            tokens.size(0), device=device
        ).unsqueeze(1).expand_as(tokens)[degree_valid]
        marginalized_nodes = torch.zeros(
            token_indices.numel(), dtype=torch.bool, device=device
        )
        compatible_node_tokens = batch.get("compatible_action_tokens")
        compatible_node_markers = batch.get("compatible_action_markers")
        if (
            direct_joint
            and compatible_node_tokens is not None
            and compatible_node_markers is not None
            and compatible_node_tokens.size(2) > 0
        ):
            compatible_node_tokens = compatible_node_tokens.to(device)[
                degree_valid
            ]
            compatible_node_markers = compatible_node_markers.to(device)[
                degree_valid
            ]
            for node in range(token_indices.numel()):
                valid_actions = (
                    compatible_node_tokens[node].ge(0)
                    & compatible_node_markers[node].ge(0)
                )
                if not bool(valid_actions.any()):
                    continue
                action_tokens = compatible_node_tokens[node, valid_actions]
                action_markers = compatible_node_markers[node, valid_actions]
                action_token_indices = token_to_generated[action_tokens]
                if bool(action_token_indices.lt(0).any()):
                    raise ValueError(
                        "compatible node action is outside generated vocabulary"
                    )
                joint_terms[node] = -torch.logsumexp(
                    joint_logp[
                        node, action_token_indices, action_markers
                    ],
                    dim=0,
                )
                marginalized_nodes[node] = True

        # Preserve the SSB-1 root-only metadata for every historical dataset.
        compatible_tokens = batch.get("compatible_root_tokens")
        compatible_markers = batch.get("compatible_root_markers")
        if (
            direct_joint
            and compatible_tokens is not None
            and compatible_markers is not None
            and compatible_tokens.size(1) > 0
        ):
            compatible_tokens = compatible_tokens.to(device)
            compatible_markers = compatible_markers.to(device)
            root_joint_nodes = (
                steps.index_select(0, node_batch_rows).eq(0)
                & ~marginalized_nodes
            )
            for node in root_joint_nodes.nonzero().flatten().tolist():
                batch_row = int(node_batch_rows[node])
                valid_actions = (
                    compatible_tokens[batch_row].ge(0)
                    & compatible_markers[batch_row].ge(0)
                )
                if not bool(valid_actions.any()):
                    continue
                action_tokens = compatible_tokens[batch_row, valid_actions]
                action_markers = compatible_markers[batch_row, valid_actions]
                action_token_indices = token_to_generated[action_tokens]
                if bool(action_token_indices.lt(0).any()):
                    raise ValueError(
                        "compatible root action is outside generated vocabulary"
                    )
                joint_terms[node] = -torch.logsumexp(
                    joint_logp[
                        node, action_token_indices, action_markers
                    ],
                    dim=0,
                )
        joint_weights = position_weights[degree_valid]
        joint_loss = (
            joint_terms * joint_weights
        ).sum() / joint_weights.sum().clamp_min(1.0)
        joint_count = degree_valid.sum()
    else:
        joint_loss = zero
        joint_count = torch.zeros((), dtype=torch.long, device=device)
    if rollout_length_cap > 0:
        rollout_length_loss, rollout_length_count = projected_rollout_length_loss(
            model,
            batch,
            vocab,
            outputs,
            cap=rollout_length_cap,
            horizon=rollout_length_horizon,
            detach_backbone=rollout_length_detach_backbone,
            root_only=rollout_length_root_only,
        )
    else:
        rollout_length_loss = zero
        rollout_length_count = torch.zeros((), dtype=torch.long, device=device)
    return {
        "token": token_loss,
        "root": root_loss,
        "degree": degree_loss,
        "direction": direction_loss,
        "marker": marker_loss,
        "joint": joint_loss,
        "rollout_length": rollout_length_loss,
        "token_count": token_valid.sum(),
        "root_count": root_valid.sum(),
        "degree_count": degree_valid.sum(),
        "direction_count": direction_valid.sum(),
        "marker_count": marker_count,
        "joint_count": joint_count,
        "rollout_length_count": rollout_length_count,
    }


def initial_region_canvas(
    example: TextInfillingExample, vocab: TextVocabulary
) -> List[Tuple[int, int]]:
    """Tag dynamic-canvas tokens by the source gap they belong to."""
    canvas: List[Tuple[int, int]] = [(vocab.LEFT, -1)]
    for gap_index in range(len(example.spans)):
        canvas.extend((token, -1) for token in example.segments[gap_index])
        canvas.append((vocab.GAP, gap_index))
    canvas.extend((token, -1) for token in example.segments[-1])
    canvas.append((vocab.RIGHT, -1))
    return canvas


def trajectory_energy_coefficients(
    generated_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score-function coefficients for the sample energy distance.

    The energy distance is ``2 E|X-Y| - E|X-X'| - E|Y-Y'|``. Its derivative
    with respect to the generated law assigns each sampled ``X`` the detached
    coefficient returned here. Unlike per-example squared length error, this
    distributional objective does not ask an unidentifiable prompt to predict
    one randomly corrupted length.
    """
    if generated_lengths.dim() != 1 or target_lengths.dim() != 1:
        raise ValueError("trajectory lengths must be one-dimensional")
    if generated_lengths.numel() < 2 or target_lengths.numel() < 1:
        raise ValueError("energy distance needs two generated and one target sample")
    generated = generated_lengths.float()
    targets = target_lengths.float()
    attraction = 2.0 * (generated[:, None] - targets[None, :]).abs().mean(dim=1)
    pairwise_generated = (generated[:, None] - generated[None, :]).abs()
    repulsion = 2.0 * pairwise_generated.sum(dim=1) / (
        generated.numel() - 1
    )
    coefficients = (attraction - repulsion) / max(float(scale), 1.0)
    generated_term = pairwise_generated.sum() / (
        generated.numel() * (generated.numel() - 1)
    )
    target_pairwise = (targets[:, None] - targets[None, :]).abs()
    target_term = target_pairwise.mean()
    energy = (
        2.0 * (generated[:, None] - targets[None, :]).abs().mean()
        - generated_term
        - target_term
    ) / max(float(scale), 1.0)
    return coefficients, energy


def sampled_trajectory_length_policy_loss(
    model,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    samples_per_prompt: int = 2,
    max_rounds: int = 8,
    max_decode_span: int = 16,
    seed: int = 1901,
    target_length_bank: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample re-encoded trajectories and return a structural policy loss.

    Sampling uses the model's exact current direct-joint action table. For the
    auxiliary score-function gradient, the sampled joint marker marginal is
    treated as a frozen offset around the live base marker logits. Consequently
    the length signal updates the structure adapter and root/degree/direction
    heads, but not the native token head, backbone, or learned token-marker
    interaction. Supervised joint NLL continues to train the full model.
    """
    if samples_per_prompt < 1:
        raise ValueError("trajectory samples per prompt must be positive")
    if len(examples) * samples_per_prompt < 2:
        raise ValueError("trajectory policy loss needs at least two rollouts")
    if any(example is None for example in examples):
        raise ValueError("trajectory batches must retain source examples")
    replicas = [
        (owner, example)
        for owner, example in enumerate(examples)
        for _ in range(samples_per_prompt)
    ]
    canvases = [initial_region_canvas(example, vocab) for _, example in replicas]
    rounds = [0] * len(replicas)
    unfinished = [False] * len(replicas)
    logp_rounds: List[torch.Tensor] = []
    generated_ids = torch.tensor(
        vocab.generated_token_ids, dtype=torch.long, device=device
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_rounds):
            active = [
                index for index, canvas in enumerate(canvases)
                if any(token == vocab.GAP for token, _ in canvas)
                and not unfinished[index]
            ]
            if not active:
                break
            width = max(len(canvases[index]) for index in active)
            tokens = torch.full(
                (len(active), width), vocab.PAD, dtype=torch.long, device=device
            )
            padding = torch.ones_like(tokens, dtype=torch.bool)
            steps = torch.tensor(
                [rounds[index] for index in active], dtype=torch.long, device=device
            )
            for row, index in enumerate(active):
                values = [token for token, _ in canvases[index]]
                tokens[row, : len(values)] = torch.tensor(values, device=device)
                padding[row, : len(values)] = False
            with torch.no_grad():
                hidden = model.backbone(
                    input_ids=tokens.masked_fill(padding, vocab.PAD),
                    attention_mask=(~padding).to(torch.long),
                ).last_hidden_state
                token_logits = model.token_head(hidden)
            root_stop, degree_logits, direction_logits = (
                model.structure_logits_from_hidden(hidden.detach(), steps)
            )
            gaps = tokens.eq(vocab.GAP) & ~padding
            gap_steps = steps.unsqueeze(1).expand_as(tokens)[gaps]
            with torch.no_grad():
                frozen_joint = model.joint_action_log_probs(
                    token_logits[gaps].detach(),
                    degree_logits[gaps].detach(),
                    direction_logits[gaps].detach(),
                    hidden[gaps].detach(),
                    gap_steps,
                    generated_ids,
                )
                sampled = torch.multinomial(
                    frozen_joint.exp().reshape(int(gaps.sum()), -1),
                    1,
                    generator=generator,
                ).squeeze(-1)
                token_indices = torch.div(sampled, 4, rounding_mode="floor")
                marker_values = sampled.remainder(4)
                stop_values = (
                    torch.rand(
                        root_stop.shape, device=device, generator=generator
                    ) < root_stop.detach().sigmoid()
                )
            base_marker = model.marker_log_probs(
                degree_logits[gaps], direction_logits[gaps]
            )
            frozen_marker = torch.logsumexp(frozen_joint, dim=-2)
            auxiliary_marker = (
                frozen_marker
                + base_marker
                - base_marker.detach()
            ).log_softmax(dim=-1)
            selected_marker_logp = auxiliary_marker.gather(
                1, marker_values.unsqueeze(1)
            ).squeeze(1)
            gap_rows = gaps.nonzero(as_tuple=False)[:, 0]
            row_contribution = root_stop.new_zeros(len(active)).scatter_add(
                0, gap_rows, selected_marker_logp
            )
            initial_rows = steps.eq(0)
            if bool(initial_rows.any()):
                root_locations = gaps & initial_rows.unsqueeze(1)
                root_logp = -F.binary_cross_entropy_with_logits(
                    root_stop[root_locations],
                    stop_values[root_locations].to(root_stop.dtype),
                    reduction="none",
                )
                root_rows = root_locations.nonzero(as_tuple=False)[:, 0]
                row_contribution = row_contribution.scatter_add(
                    0, root_rows, root_logp
                )
                # A stopped root emits no joint action.
                stopped_gap = stop_values[gaps] & initial_rows[gap_rows]
                row_contribution = row_contribution.scatter_add(
                    0, gap_rows[stopped_gap],
                    -selected_marker_logp[stopped_gap],
                )
            replica_contribution = root_stop.new_zeros(len(replicas))
            active_tensor = torch.tensor(active, device=device, dtype=torch.long)
            replica_contribution = replica_contribution.scatter_add(
                0, active_tensor, row_contribution
            )
            logp_rounds.append(replica_contribution)

            chosen_tokens = generated_ids[token_indices].cpu()
            chosen_markers = marker_values.cpu()
            stop_cpu = stop_values.cpu()
            gap_cursor = 0
            for row, replica_index in enumerate(active):
                expanded: List[Tuple[int, int]] = []
                initial = rounds[replica_index] == 0
                for position, (token, region) in enumerate(canvases[replica_index]):
                    if token != vocab.GAP:
                        expanded.append((token, region))
                        continue
                    stopped = initial and bool(stop_cpu[row, position])
                    pivot = int(chosen_tokens[gap_cursor])
                    marker = int(chosen_markers[gap_cursor])
                    gap_cursor += 1
                    if stopped:
                        continue
                    if marker in (1, 3):
                        expanded.append((vocab.GAP, region))
                    expanded.append((pivot, region))
                    if marker in (2, 3):
                        expanded.append((vocab.GAP, region))
                rounds[replica_index] += 1
                generated = sum(
                    token != vocab.GAP and region >= 0 for token, region in expanded
                )
                if generated > max_decode_span:
                    unfinished[replica_index] = True
                    expanded = [item for item in expanded if item[0] != vocab.GAP]
                canvases[replica_index] = expanded
    finally:
        model.train(was_training)
    if logp_rounds:
        trajectory_logp = torch.stack(logp_rounds).sum(dim=0)
    else:
        trajectory_logp = next(model.parameters()).sum() * 0.0 + torch.zeros(
            len(replicas), device=device
        )
    lengths = []
    for index, canvas in enumerate(canvases):
        failed = unfinished[index] or any(token == vocab.GAP for token, _ in canvas)
        lengths.append(
            max_decode_span + 1 if failed else sum(
                token != vocab.GAP and region >= 0 for token, region in canvas
            )
        )
    generated_lengths = torch.tensor(lengths, device=device)
    target_lengths = torch.tensor(
        list(target_length_bank)
        if target_length_bank is not None
        else [sum(len(span) for span in example.spans) for example in examples],
        device=device,
    )
    if target_lengths.numel() < 1:
        raise ValueError("trajectory target length bank cannot be empty")
    coefficients, energy = trajectory_energy_coefficients(
        generated_lengths, target_lengths, max_decode_span + 1
    )
    scale = coefficients.detach().abs().mean().clamp_min(1.0)
    policy_loss = (coefficients.detach() / scale * trajectory_logp).mean()
    return policy_loss, energy.detach(), generated_lengths.detach()


@torch.inference_mode()
def decode_frontier_model(
    model,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    max_rounds: int = 16,
    max_decode_span: int = 16,
    stochastic: bool = False,
    generator: Optional[torch.Generator] = None,
    sample_tokens: Optional[bool] = None,
    chunk_size: Optional[int] = None,
    selective_gap_fraction: float = 1.0,
    selective_gap_min: int = 1,
    root_lookahead_ranker: Optional[Dict[str, torch.Tensor]] = None,
    root_lookahead_token_k: int = 4,
    root_lookahead_candidate_batch_size: int = 4,
    root_lookahead_temperature: float = 1.0,
    root_lookahead_cache: Optional[Dict[object, object]] = None,
    defer_lookahead: bool = False,
    defer_lookahead_candidate_batch_size: int = 4,
    defer_lookahead_weight: float = 1.0,
    selection_policy: str = "confidence",
    selection_threshold: float = 0.0,
    return_action_logp: bool = False,
) -> Tuple[List[List[List[int]]], List[int], List[bool]]:
    """Score every open gap, then expand a configurable subset per round.

    Greedy decoding is useful for inspecting the conditional mode.  Stochastic
    decoding samples the actual unknown-length generative process; this matters
    when corruption length is independent of the visible prompt and therefore
    cannot be recovered as a deterministic per-prompt label.

    ``selective_gap_fraction < 1`` defers low-confidence descendant gaps.  Root
    gaps are always resolved together in the first round so their independent
    empty-span decisions retain the original semantics.  Confidence is the
    maximum probability of the joint token/marker action when available, and
    the maximum lexical probability otherwise.

    ``selection_policy`` chooses how many gaps that budget covers and which
    ones.  ``confidence`` keeps the fixed-share top-k rule.  ``threshold``
    replaces the fixed share with ``selection_threshold``, a probability every
    committed action must reach, so an easy frontier commits in one round and a
    doubtful one commits a single gap.  ``random`` keeps the fixed share but
    picks uninformed gaps, which prices the confidence ranking at equal NFE.
    """
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("decode chunk size must be positive")
    if not 0.0 < selective_gap_fraction <= 1.0:
        raise ValueError("selective_gap_fraction must be in (0,1]")
    if selective_gap_min < 1:
        raise ValueError("selective_gap_min must be positive")
    if root_lookahead_token_k < 1 or root_lookahead_candidate_batch_size < 1:
        raise ValueError("root lookahead sizes must be positive")
    if root_lookahead_temperature <= 0.0:
        raise ValueError("root lookahead temperature must be positive")
    if defer_lookahead_candidate_batch_size < 1:
        raise ValueError("DEFER lookahead candidate batch size must be positive")
    if defer_lookahead_weight < 0.0:
        raise ValueError("DEFER lookahead weight must be non-negative")
    if selection_policy not in {"confidence", "threshold", "random"}:
        raise ValueError("unknown selection policy: {}".format(selection_policy))
    if selection_policy == "threshold" and not 0.0 < selection_threshold <= 1.0:
        raise ValueError("threshold selection needs a probability in (0,1]")
    if selection_policy == "random" and generator is None:
        raise ValueError("random selection requires a generator")
    if return_action_logp and not (
        getattr(model, "marginal_preserving_joint", False)
        or getattr(model, "direct_joint_actions", False)
    ):
        raise ValueError(
            "the derivation log-probability needs the joint token/marker head"
        )
    selection_threshold_logp = (
        math.log(selection_threshold)
        if selection_policy == "threshold" else -math.inf
    )
    # The derivation log-probability sums every committed action and the root
    # empty decision, so candidates of different lengths stay comparable
    # without an invented normalizer.
    action_logp = [0.0] * len(examples)
    if chunk_size is not None and len(examples) > chunk_size:
        predictions: List[List[List[int]]] = []
        rounds: List[int] = []
        unfinished: List[bool] = []
        scores: List[float] = []
        for start in range(0, len(examples), chunk_size):
            chunk = decode_frontier_model(
                model,
                examples[start : start + chunk_size],
                vocab,
                device,
                max_rounds=max_rounds,
                max_decode_span=max_decode_span,
                stochastic=stochastic,
                generator=generator,
                sample_tokens=sample_tokens,
                chunk_size=None,
                selective_gap_fraction=selective_gap_fraction,
                selective_gap_min=selective_gap_min,
                root_lookahead_ranker=root_lookahead_ranker,
                root_lookahead_token_k=root_lookahead_token_k,
                root_lookahead_candidate_batch_size=(
                    root_lookahead_candidate_batch_size
                ),
                root_lookahead_temperature=root_lookahead_temperature,
                root_lookahead_cache=root_lookahead_cache,
                defer_lookahead=defer_lookahead,
                defer_lookahead_candidate_batch_size=(
                    defer_lookahead_candidate_batch_size
                ),
                defer_lookahead_weight=defer_lookahead_weight,
                selection_policy=selection_policy,
                selection_threshold=selection_threshold,
                return_action_logp=return_action_logp,
            )
            predictions.extend(chunk[0])
            rounds.extend(chunk[1])
            unfinished.extend(chunk[2])
            if return_action_logp:
                scores.extend(chunk[3])
        if return_action_logp:
            return predictions, rounds, unfinished, scores
        return predictions, rounds, unfinished
    model.eval()
    if sample_tokens is None:
        sample_tokens = stochastic
    canvases = [initial_region_canvas(example, vocab) for example in examples]
    canvas_depths = [
        [0 if token == vocab.GAP else -1 for token, _ in canvas]
        for canvas in canvases
    ]
    canvas_opened = [
        [0 if token == vocab.GAP else -1 for token, _ in canvas]
        for canvas in canvases
    ]
    rounds = [0] * len(examples)
    unfinished = [False] * len(examples)
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)

    for _ in range(max_rounds):
        active = [
            index
            for index, canvas in enumerate(canvases)
            if any(token == vocab.GAP for token, _ in canvas)
        ]
        if not active:
            break
        width = max(len(canvases[index]) for index in active)
        tokens = torch.full(
            (len(active), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        node_depths = torch.zeros_like(tokens)
        node_ages = torch.zeros_like(tokens)
        steps = torch.tensor(
            [rounds[index] for index in active],
            dtype=torch.long,
            device=device,
        )
        for row, index in enumerate(active):
            raw = [token for token, _ in canvases[index]]
            tokens[row, : len(raw)] = torch.tensor(raw, device=device)
            padding[row, : len(raw)] = False
            if getattr(model, "per_node_frontier_features", False):
                node_depths[row, : len(raw)] = torch.tensor(
                    canvas_depths[index], device=device
                ).clamp_min(0)
                opened = torch.tensor(canvas_opened[index], device=device)
                node_ages[row, : len(raw)] = torch.where(
                    opened.ge(0), steps[row] - opened, torch.zeros_like(opened)
                ).clamp_min(0)

        node_features = (
            {"node_depths": node_depths, "node_ages": node_ages}
            if getattr(model, "per_node_frontier_features", False) else {}
        )
        token_logits, root_stop, degree_logits, direction_logits, hidden = model(
            tokens, padding, steps, **node_features
        )
        allowed = token_logits.index_select(-1, generated_ids)
        gap_mask = tokens.eq(vocab.GAP) & ~padding
        gap_confidence = torch.full_like(tokens, -torch.inf, dtype=token_logits.dtype)
        joint_markers = None
        if (
            getattr(model, "marginal_preserving_joint", False)
            or getattr(model, "direct_joint_actions", False)
        ):
            gap_steps = steps.unsqueeze(1).expand_as(tokens)[gap_mask]
            joint_logp = model.joint_action_log_probs(
                token_logits[gap_mask],
                degree_logits[gap_mask],
                direction_logits[gap_mask],
                hidden[gap_mask],
                gap_steps,
                generated_ids,
            )
            gap_confidence[gap_mask] = joint_logp.amax(dim=(1, 2))
            chosen_device = generated_ids[allowed.argmax(dim=-1)]
            joint_markers_device = torch.zeros_like(tokens)
            nodes = int(gap_mask.sum())
            if stochastic and sample_tokens:
                samples = torch.multinomial(
                    joint_logp.exp().reshape(nodes, -1),
                    1,
                    generator=generator,
                ).squeeze(-1)
                token_indices = torch.div(samples, 4, rounding_mode="floor")
                marker_values = samples.remainder(4)
            elif stochastic:
                token_indices = allowed[gap_mask].argmax(dim=-1)
                rows = torch.arange(nodes, device=device)
                marker_values = torch.multinomial(
                    joint_logp[rows, token_indices].softmax(dim=-1),
                    1,
                    generator=generator,
                ).squeeze(-1)
            else:
                samples = joint_logp.reshape(nodes, -1).argmax(dim=-1)
                token_indices = torch.div(samples, 4, rounding_mode="floor")
                marker_values = samples.remainder(4)
            chosen_device[gap_mask] = generated_ids[token_indices]
            joint_markers_device[gap_mask] = marker_values
            if return_action_logp:
                node_rows = torch.arange(nodes, device=device)
                chosen_logp_device = torch.zeros_like(
                    tokens, dtype=joint_logp.dtype
                )
                chosen_logp_device[gap_mask] = joint_logp[
                    node_rows, token_indices, marker_values
                ]
            if root_lookahead_ranker is not None and bool(steps.eq(0).all()):
                from selective_semantic_branching.root_lookahead import (
                    rerank_root_actions,
                )

                root_tokens, root_markers, _ = rerank_root_actions(
                    model,
                    [canvases[index] for index in active],
                    token_logits,
                    degree_logits,
                    direction_logits,
                    tokens,
                    padding,
                    generated_ids,
                    vocab.GAP,
                    vocab.PAD,
                    root_lookahead_ranker,
                    token_k=root_lookahead_token_k,
                    candidate_batch_size=root_lookahead_candidate_batch_size,
                    stochastic=stochastic and bool(sample_tokens),
                    generator=generator,
                    temperature=root_lookahead_temperature,
                    cache=root_lookahead_cache,
                )
                root_positions = gap_mask.to(torch.long).argmax(dim=-1)
                root_rows = torch.arange(len(active), device=device)
                chosen_device[root_rows, root_positions] = root_tokens
                joint_markers_device[root_rows, root_positions] = root_markers
            chosen = chosen_device.cpu()
            joint_markers = joint_markers_device.cpu()
            if return_action_logp:
                chosen_logp = chosen_logp_device.cpu()
                stop_logp = torch.nn.functional.logsigmoid(root_stop).cpu()
                keep_logp = torch.nn.functional.logsigmoid(-root_stop).cpu()
        elif stochastic and sample_tokens:
            gap_confidence[gap_mask] = allowed[gap_mask].log_softmax(dim=-1).amax(
                dim=-1
            )
            token_probabilities = allowed.softmax(dim=-1)
            token_samples = torch.multinomial(
                token_probabilities.reshape(-1, token_probabilities.size(-1)),
                1,
                generator=generator,
            ).reshape(token_probabilities.shape[:-1])
            chosen = generated_ids[token_samples].cpu()
        else:
            gap_confidence[gap_mask] = allowed[gap_mask].log_softmax(dim=-1).amax(
                dim=-1
            )
            chosen = generated_ids[allowed.argmax(dim=-1)].cpu()
        if getattr(model, "token_conditioned_topology", False):
            # The node decides how to branch after seeing what it just emitted,
            # which needs no second backbone pass: only the small structure
            # adapter is recomputed.
            structure_token_ids = torch.full_like(tokens, -1)
            emitted = tokens.eq(vocab.GAP)
            structure_token_ids[emitted] = chosen.to(device)[emitted]
            root_stop, degree_logits, direction_logits = (
                model.structure_logits_from_hidden(
                    hidden, steps, structure_token_ids
                )
            )
        if stochastic:
            stop = (
                torch.rand(
                    root_stop.shape,
                    device=device,
                    generator=generator,
                )
                < root_stop.sigmoid()
            ).cpu()
            if joint_markers is None:
                degree_probabilities = degree_logits.softmax(dim=-1)
                degree = torch.multinomial(
                    degree_probabilities.reshape(-1, 3),
                    1,
                    generator=generator,
                ).reshape(degree_probabilities.shape[:-1]).cpu()
                direction_probabilities = direction_logits.softmax(dim=-1)
                direction = torch.multinomial(
                    direction_probabilities.reshape(-1, 2),
                    1,
                    generator=generator,
                ).reshape(direction_probabilities.shape[:-1]).cpu()
        else:
            stop = root_stop.gt(0).cpu()
            if joint_markers is None:
                degree = degree_logits.argmax(dim=-1).cpu()
                direction = direction_logits.argmax(dim=-1).cpu()
        if joint_markers is not None:
            degree = torch.where(
                joint_markers.eq(0),
                torch.zeros_like(joint_markers),
                torch.where(
                    joint_markers.eq(3),
                    torch.full_like(joint_markers, 2),
                    torch.ones_like(joint_markers),
                ),
            )
            direction = joint_markers.eq(2).to(torch.long)

        selection_scores = gap_confidence
        if defer_lookahead and selective_gap_fraction < 1.0:
            if joint_markers is None:
                raise ValueError("DEFER lookahead requires joint token/marker actions")
            from selective_semantic_branching.defer_lookahead import (
                predicted_defer_expand_scores,
            )

            defer_scores = predicted_defer_expand_scores(
                model,
                [canvases[index] for index in active],
                gap_mask,
                steps,
                chosen,
                joint_markers,
                gap_confidence,
                generated_ids,
                vocab.GAP,
                vocab.PAD,
                candidate_batch_size=defer_lookahead_candidate_batch_size,
            )
            selection_scores = (
                gap_confidence + defer_lookahead_weight * defer_scores
            )

        selected_gaps = torch.zeros_like(gap_mask)
        for row in range(len(active)):
            positions = gap_mask[row].nonzero().flatten()
            everything = (
                int(steps[row]) == 0
                or (
                    selection_policy != "threshold"
                    and selective_gap_fraction == 1.0
                )
            )
            if everything:
                selected_gaps[row, positions] = True
                continue
            confidence = selection_scores[row].index_select(0, positions)
            if selection_policy == "threshold":
                # An adaptive budget: commit every GAP the model is already
                # confident about, rather than a fixed share of the frontier.
                above = confidence.ge(selection_threshold_logp)
                count = max(selective_gap_min, int(above.sum()))
                count = min(len(positions), count)
            else:
                count = min(
                    len(positions),
                    max(
                        selective_gap_min,
                        int(math.ceil(
                            len(positions) * selective_gap_fraction
                        )),
                    ),
                )
            if selection_policy == "random":
                # Same budget, uninformed order. This prices the confidence
                # ranking itself at equal NFE.
                order = torch.randperm(
                    len(positions), device=positions.device, generator=generator
                )
                chosen_positions = positions.index_select(0, order[:count])
            else:
                chosen_positions = positions.index_select(
                    0, confidence.topk(count).indices
                )
            selected_gaps[row, chosen_positions] = True
        selected_gaps = selected_gaps.cpu()

        for row, index in enumerate(active):
            expanded: List[Tuple[int, int]] = []
            expanded_depths: List[int] = []
            expanded_opened: List[int] = []
            initial = rounds[index] == 0
            for position, (token, region) in enumerate(canvases[index]):
                if token != vocab.GAP:
                    expanded.append((token, region))
                    expanded_depths.append(-1)
                    expanded_opened.append(-1)
                    continue
                if not bool(selected_gaps[row, position]):
                    expanded.append((token, region))
                    expanded_depths.append(canvas_depths[index][position])
                    expanded_opened.append(canvas_opened[index][position])
                    continue
                if initial and bool(stop[row, position]):
                    if return_action_logp:
                        action_logp[index] += float(stop_logp[row, position])
                    continue
                if return_action_logp:
                    if initial:
                        action_logp[index] += float(keep_logp[row, position])
                    action_logp[index] += float(chosen_logp[row, position])
                pivot = int(chosen[row, position])
                child_count = int(degree[row, position])
                unary_direction = int(direction[row, position])
                left_child = child_count == 2 or (
                    child_count == 1 and unary_direction == 0
                )
                right_child = child_count == 2 or (
                    child_count == 1 and unary_direction == 1
                )
                if left_child:
                    expanded.append((vocab.GAP, region))
                    expanded_depths.append(canvas_depths[index][position] + 1)
                    expanded_opened.append(rounds[index] + 1)
                expanded.append((pivot, region))
                expanded_depths.append(-1)
                expanded_opened.append(-1)
                if right_child:
                    expanded.append((vocab.GAP, region))
                    expanded_depths.append(canvas_depths[index][position] + 1)
                    expanded_opened.append(rounds[index] + 1)
            rounds[index] += 1
            for gap_index in range(len(examples[index].spans)):
                generated = sum(
                    token != vocab.GAP and region == gap_index
                    for token, region in expanded
                )
                if generated > max_decode_span:
                    unfinished[index] = True
            if unfinished[index]:
                keep = [item[0] != vocab.GAP for item in expanded]
                expanded = [
                    item for item, retain in zip(expanded, keep) if retain
                ]
                expanded_depths = [
                    value for value, retain in zip(expanded_depths, keep)
                    if retain
                ]
                expanded_opened = [
                    value for value, retain in zip(expanded_opened, keep)
                    if retain
                ]
            canvases[index] = expanded
            canvas_depths[index] = expanded_depths
            canvas_opened[index] = expanded_opened

    predictions: List[List[List[int]]] = []
    for index, (example, canvas) in enumerate(zip(examples, canvases)):
        if any(token == vocab.GAP for token, _ in canvas):
            unfinished[index] = True
        predictions.append(
            [
                [
                    token
                    for token, region in canvas
                    if region == gap_index and token != vocab.GAP
                ]
                for gap_index in range(len(example.spans))
            ]
        )
    if return_action_logp:
        return predictions, rounds, unfinished, action_logp
    return predictions, rounds, unfinished


@torch.inference_mode()
def sample_frontier_rollouts(
    model,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    samples_per_prompt: int = 32,
    chunk_size: int = 64,
    max_rounds: int = 16,
    max_decode_span: int = 16,
    seed: int = 1901,
    sample_tokens: bool = True,
    selective_gap_fraction: float = 1.0,
    selective_gap_min: int = 1,
    root_lookahead_ranker: Optional[Dict[str, torch.Tensor]] = None,
    root_lookahead_token_k: int = 4,
    root_lookahead_candidate_batch_size: int = 4,
    root_lookahead_temperature: float = 1.0,
    defer_lookahead: bool = False,
    defer_lookahead_candidate_batch_size: int = 4,
    defer_lookahead_weight: float = 1.0,
    selection_policy: str = "confidence",
    selection_threshold: float = 0.0,
    return_scores: bool = False,
) -> Tuple[List[List[List[int]]], List[List[int]], List[List[bool]]]:
    """Draw ancestral samples without supplying a target length or canvas.

    With ``return_scores`` each sample also carries the log-probability of the
    derivation that produced it, including the root empty decision, which is the
    only score that compares candidates of different lengths without an invented
    normalizer.
    """
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if any(len(example.spans) != 1 for example in examples):
        raise ValueError("sampling metrics currently require one gap per prompt")

    samples: List[List[List[int]]] = [[] for _ in examples]
    rounds: List[List[int]] = [[] for _ in examples]
    unfinished: List[List[bool]] = [[] for _ in examples]
    scores: List[List[float]] = [[] for _ in examples]
    replicas = [
        (index, example)
        for index, example in enumerate(examples)
        for _ in range(samples_per_prompt)
    ]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    root_lookahead_cache = {} if root_lookahead_ranker is not None else None
    for start in range(0, len(replicas), chunk_size):
        batch = replicas[start : start + chunk_size]
        decoded = decode_frontier_model(
            model,
            [example for _, example in batch],
            vocab,
            device,
            max_rounds=max_rounds,
            max_decode_span=max_decode_span,
            stochastic=True,
            generator=generator,
            sample_tokens=sample_tokens,
            selective_gap_fraction=selective_gap_fraction,
            selective_gap_min=selective_gap_min,
            root_lookahead_ranker=root_lookahead_ranker,
            root_lookahead_token_k=root_lookahead_token_k,
            root_lookahead_candidate_batch_size=(
                root_lookahead_candidate_batch_size
            ),
            root_lookahead_temperature=root_lookahead_temperature,
            root_lookahead_cache=root_lookahead_cache,
            defer_lookahead=defer_lookahead,
            defer_lookahead_candidate_batch_size=(
                defer_lookahead_candidate_batch_size
            ),
            defer_lookahead_weight=defer_lookahead_weight,
            selection_policy=selection_policy,
            selection_threshold=selection_threshold,
            return_action_logp=return_scores,
        )
        predictions, batch_rounds, batch_unfinished = decoded[:3]
        batch_scores = decoded[3] if return_scores else [0.0] * len(batch)
        for (owner, _), prediction, steps, failed, score in zip(
            batch, predictions, batch_rounds, batch_unfinished, batch_scores
        ):
            samples[owner].append(prediction[0])
            rounds[owner].append(steps)
            unfinished[owner].append(failed)
            scores[owner].append(score)
        if device.type == "cuda":
            # Dynamic canvases have different widths across prompts. Under
            # Windows WDDM, retaining all allocator slabs caused reserved
            # memory to grow beyond 8 GiB even without root lookahead, despite
            # live tensors staying below 3 GiB. Return inactive slabs after
            # every replica chunk for every frontier rollout mode.
            torch.cuda.empty_cache()
    if return_scores:
        return samples, rounds, unfinished, scores
    return samples, rounds, unfinished


@torch.inference_mode()
def sample_frontier_scaffolds(
    model,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    samples_per_prompt: int = 32,
    chunk_size: int = 64,
    max_rounds: int = 16,
    max_decode_span: int = 16,
    seed: int = 1901,
    persistent_regime: bool = False,
    markov_regime: bool = False,
    return_codes: bool = False,
    return_states: bool = False,
    conditional_context_source: Optional[str] = None,
):
    """Sample topology while keeping every emitted token as a native mask slot.

    Length is the total progeny of the branching process.  It is never passed
    into the topology model and no fixed canvas is allocated.  The completed
    scaffold can subsequently be filled by a standard parallel masked LM.
    """
    if samples_per_prompt < 1 or chunk_size < 1:
        raise ValueError("sampling and chunk sizes must be positive")
    if persistent_regime and markov_regime:
        raise ValueError("persistent and Markov regimes are mutually exclusive")
    if return_codes and return_states:
        raise ValueError("return only one semantic representation")
    if conditional_context_source is not None and (
        persistent_regime or markov_regime
    ):
        raise ValueError("conditional context uses an independent regime per round")
    if any(len(example.spans) != 1 for example in examples):
        raise ValueError("scaffold evaluation currently requires one gap")
    lengths: List[List[int]] = [[] for _ in examples]
    all_rounds: List[List[int]] = [[] for _ in examples]
    all_unfinished: List[List[bool]] = [[] for _ in examples]
    all_slot_codes: List[List[List[int]]] = [[] for _ in examples]
    all_slot_states: List[List[torch.Tensor]] = [[] for _ in examples]
    replicas = [
        (owner, example)
        for owner, example in enumerate(examples)
        for _ in range(samples_per_prompt)
    ]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.eval()

    for start in range(0, len(replicas), chunk_size):
        batch = replicas[start : start + chunk_size]
        # item = (token, region, open, discrete code, continuous state).
        # Semantic values are node-local and only completed slots carry one.
        canvases = []
        for _, example in batch:
            canvas = [
                (vocab.LEFT, -1, False, -1, None)
            ]
            for region in range(len(example.spans)):
                canvas.extend(
                    (token, -1, False, -1, None)
                    for token in example.segments[region]
                )
                canvas.append((vocab.GAP, region, True, -1, None))
            canvas.extend(
                (token, -1, False, -1, None)
                for token in example.segments[-1]
            )
            canvas.append((vocab.RIGHT, -1, False, -1, None))
            canvases.append(canvas)
        fixed_context = None
        if conditional_context_source is not None:
            if bool(getattr(model, "semantic_codes", 0)) or bool(
                getattr(model, "continuous_semantic", False)
            ):
                raise ValueError("conditional scaffold does not use semantic slots")
            width = max(len(canvas) for canvas in canvases)
            prompt_tokens = torch.full(
                (len(canvases), width),
                vocab.PAD,
                dtype=torch.long,
                device=device,
            )
            prompt_padding = torch.ones_like(prompt_tokens, dtype=torch.bool)
            for row, canvas in enumerate(canvases):
                values = [token for token, _, _, _, _ in canvas]
                prompt_tokens[row, : len(values)] = torch.tensor(
                    values, device=device
                )
                prompt_padding[row, : len(values)] = False
            fixed_context = model.prompt_shape_context(
                prompt_tokens,
                prompt_padding,
                source=conditional_context_source,
            )
        rounds = [0] * len(batch)
        unfinished = [False] * len(batch)
        persistent_regimes = [-1] * len(batch)

        for _ in range(max_rounds):
            active = [
                index
                for index, canvas in enumerate(canvases)
                if any(is_gap for _, _, is_gap, _, _ in canvas)
            ]
            if not active:
                break
            width = max(len(canvases[index]) for index in active)
            tokens = torch.full(
                (len(active), width), vocab.PAD, dtype=torch.long, device=device
            )
            padding = torch.ones_like(tokens, dtype=torch.bool)
            open_mask = torch.zeros_like(tokens, dtype=torch.bool)
            slot_codes = torch.full_like(tokens, -1)
            continuous_enabled = bool(
                getattr(model, "continuous_semantic", False)
            )
            slot_semantics = (
                torch.zeros(
                    len(active),
                    width,
                    model.d_model,
                    device=device,
                    dtype=model.semantic_embedding_mean.dtype,
                )
                if continuous_enabled
                else None
            )
            steps = torch.tensor(
                [rounds[index] for index in active],
                dtype=torch.long,
                device=device,
            )
            for row, index in enumerate(active):
                raw = [token for token, _, _, _, _ in canvases[index]]
                tokens[row, : len(raw)] = torch.tensor(raw, device=device)
                padding[row, : len(raw)] = False
                open_values = [
                    is_gap for _, _, is_gap, _, _ in canvases[index]
                ]
                open_mask[row, : len(raw)] = torch.tensor(
                    open_values, device=device
                )
                code_values = [
                    code for _, _, _, code, _ in canvases[index]
                ]
                slot_codes[row, : len(raw)] = torch.tensor(
                    code_values, device=device
                )
                if slot_semantics is not None:
                    for position, item in enumerate(canvases[index]):
                        state = item[4]
                        if state is not None:
                            slot_semantics[row, position] = state.to(device)
            semantic_codes = None
            semantic_states = None
            if conditional_context_source is not None:
                stops = torch.zeros_like(tokens, dtype=torch.bool)
                degrees = torch.zeros_like(tokens)
                directions = torch.zeros_like(tokens)
                sampled_regimes = torch.zeros(
                    len(active), dtype=torch.long, device=device
                )
                for row, index in enumerate(active):
                    open_count = int(open_mask[row].sum())
                    completed_count = sum(
                        region >= 0 and not is_gap
                        for _, region, is_gap, _, _ in canvases[index]
                    )
                    root, regime, degree, direction = (
                        model.conditional_shape_logits(
                            fixed_context[index : index + 1],
                            rounds[index],
                            open_count,
                            completed_count,
                        )
                    )
                    regime_index = int(torch.multinomial(
                        regime[0].softmax(dim=-1),
                        1,
                        generator=generator,
                    ))
                    sampled_regimes[row] = regime_index
                    positions = open_mask[row].nonzero().flatten()
                    if rounds[index] == 0:
                        stop = torch.rand((), device=device, generator=generator)
                        stops[row, positions] = stop < root[0].sigmoid()
                    degree_probability = degree[
                        0, regime_index
                    ].softmax(dim=-1)
                    direction_probability = direction[
                        0, regime_index
                    ].softmax(dim=-1)
                    degrees[row, positions] = torch.multinomial(
                        degree_probability,
                        positions.numel(),
                        replacement=True,
                        generator=generator,
                    )
                    directions[row, positions] = torch.multinomial(
                        direction_probability,
                        positions.numel(),
                        replacement=True,
                        generator=generator,
                    )
                stops = stops.cpu()
                degrees = degrees.cpu()
                directions = directions.cpu()
            elif hasattr(model, "sample_structure"):
                previous = torch.tensor(
                    [persistent_regimes[index] for index in active],
                    dtype=torch.long,
                    device=device,
                )
                if persistent_regime:
                    forced_regimes = previous
                elif markov_regime:
                    forced_regimes = previous.clone()
                    transitioned = previous.ge(0)
                    if bool(transitioned.any()):
                        forced_regimes[transitioned] = (
                            model.sample_regime_transition(
                                previous[transitioned],
                                steps[transitioned],
                                generator=generator,
                            )
                        )
                else:
                    forced_regimes = None
                semantic_enabled = bool(getattr(model, "semantic_codes", 0))
                sampled = model.sample_structure(
                    tokens,
                    padding,
                    steps,
                    open_mask,
                    generator=generator,
                    forced_regimes=forced_regimes,
                    slot_codes=slot_codes if semantic_enabled else None,
                    slot_semantics=slot_semantics,
                    return_semantic_codes=semantic_enabled,
                    return_continuous_semantic=continuous_enabled,
                )
                stops, degrees, directions, sampled_regimes = sampled[:4]
                semantic_codes = (
                    sampled[4].cpu() if semantic_enabled else None
                )
                semantic_states = (
                    sampled[4].float().cpu()
                    if continuous_enabled
                    else None
                )
                if persistent_regime or markov_regime:
                    for row, index in enumerate(active):
                        if markov_regime or persistent_regimes[index] < 0:
                            persistent_regimes[index] = int(
                                sampled_regimes[row]
                            )
                stops = stops.cpu()
                degrees = degrees.cpu()
                directions = directions.cpu()
            else:
                _, root_logits, degree_logits, direction_logits, _ = model(
                    tokens, padding, steps
                )
                stops = (
                    torch.rand(
                        root_logits.shape, device=device, generator=generator
                    )
                    < root_logits.sigmoid()
                ).cpu()
                degree_probabilities = degree_logits.softmax(dim=-1)
                degrees = torch.multinomial(
                    degree_probabilities.reshape(-1, 3),
                    1,
                    generator=generator,
                ).reshape(degree_probabilities.shape[:-1]).cpu()
                direction_probabilities = direction_logits.softmax(dim=-1)
                directions = torch.multinomial(
                    direction_probabilities.reshape(-1, 2),
                    1,
                    generator=generator,
                ).reshape(direction_probabilities.shape[:-1]).cpu()

            for row, index in enumerate(active):
                expanded = []
                initial = rounds[index] == 0
                for position, (token, region, is_gap, code, state) in enumerate(
                    canvases[index]
                ):
                    if not is_gap:
                        expanded.append(
                            (token, region, False, code, state)
                        )
                        continue
                    if initial and bool(stops[row, position]):
                        continue
                    degree = int(degrees[row, position])
                    direction = int(directions[row, position])
                    left_child = degree == 2 or (degree == 1 and direction == 0)
                    right_child = degree == 2 or (degree == 1 and direction == 1)
                    if left_child:
                        expanded.append(
                            (vocab.GAP, region, True, -1, None)
                        )
                    node_code = (
                        int(semantic_codes[row, position])
                        if semantic_codes is not None
                        else -1
                    )
                    node_state = (
                        semantic_states[row, position].clone()
                        if semantic_states is not None
                        else None
                    )
                    expanded.append(
                        (
                            vocab.GAP,
                            region,
                            False,
                            node_code,
                            node_state,
                        )
                    )
                    if right_child:
                        expanded.append(
                            (vocab.GAP, region, True, -1, None)
                        )
                rounds[index] += 1
                slots = sum(
                    region >= 0 and not is_gap
                    for _, region, is_gap, _, _ in expanded
                )
                if slots > max_decode_span:
                    unfinished[index] = True
                    expanded = [
                        (token, region, False, code, state)
                        for token, region, is_gap, code, state in expanded
                        if not is_gap
                    ]
                canvases[index] = expanded

        for local, ((owner, _), canvas) in enumerate(zip(batch, canvases)):
            failed = unfinished[local] or any(
                is_gap for _, _, is_gap, _, _ in canvas
            )
            lengths[owner].append(
                sum(
                    region == 0 and not is_gap
                    for _, region, is_gap, _, _ in canvas
                )
            )
            all_rounds[owner].append(rounds[local])
            all_unfinished[owner].append(failed)
            all_slot_codes[owner].append([
                code
                for _, region, is_gap, code, _ in canvas
                if region == 0 and not is_gap
            ])
            states = [
                state
                for _, region, is_gap, _, state in canvas
                if region == 0 and not is_gap and state is not None
            ]
            all_slot_states[owner].append(
                torch.stack(states)
                if states
                else torch.empty(0, getattr(model, "d_model", 0))
            )
    if return_codes:
        return lengths, all_rounds, all_unfinished, all_slot_codes
    if return_states:
        return lengths, all_rounds, all_unfinished, all_slot_states
    return lengths, all_rounds, all_unfinished


@torch.inference_mode()
def sample_unified_scaffolds(
    model,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    samples_per_prompt: int = 32,
    chunk_size: int = 64,
    max_rounds: int = 16,
    max_decode_span: int = 16,
    seed: int = 1901,
    conditional_context_source: Optional[str] = None,
    skip_round_encoding: bool = False,
) -> Tuple[List[List[List[int]]], List[List[int]], List[List[bool]]]:
    """Grow shape and token beliefs together, then decode in parallel.

    No length or fixed output canvas is supplied.  Each expansion pass emits a
    topology decision and a soft lexical state for the same node.  Once the
    branching process terminates, the same MLM head fills every completed slot
    simultaneously.

    `skip_round_encoding` drops the per-round backbone pass.  It is available
    only in the conditional mode, where the shape policy reads a context fixed
    at round zero and therefore never consults the evolving canvas: the only
    thing that pass produces is the node-local token posterior carried into the
    final fill.  With it off, growth costs no backbone passes at all and a
    complete generation is two passes -- one for the round-zero context and one
    for the fill -- whatever length it produces.
    """
    if skip_round_encoding and conditional_context_source is None:
        raise ValueError(
            "skipping the round encoding requires a conditional context"
        )
    if samples_per_prompt < 1 or chunk_size < 1:
        raise ValueError("sampling and chunk sizes must be positive")
    if any(len(example.spans) != 1 for example in examples):
        raise ValueError("unified scaffold evaluation currently requires one gap")
    samples: List[List[List[int]]] = [[] for _ in examples]
    all_rounds: List[List[int]] = [[] for _ in examples]
    all_unfinished: List[List[bool]] = [[] for _ in examples]
    replicas = [
        (owner, example)
        for owner, example in enumerate(examples)
        for _ in range(samples_per_prompt)
    ]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.eval()

    for start in range(0, len(replicas), chunk_size):
        batch = replicas[start : start + chunk_size]
        # item = (token, region, is_open_gap, posterior_state).
        canvases = []
        for _, example in batch:
            canvas = [(vocab.LEFT, -1, False, None)]
            for region in range(len(example.spans)):
                canvas.extend(
                    (token, -1, False, None)
                    for token in example.segments[region]
                )
                canvas.append((vocab.GAP, region, True, None))
            canvas.extend(
                (token, -1, False, None)
                for token in example.segments[-1]
            )
            canvas.append((vocab.RIGHT, -1, False, None))
            canvases.append(canvas)
        fixed_context = None
        if conditional_context_source is not None:
            width = max(len(canvas) for canvas in canvases)
            prompt_tokens = torch.full(
                (len(canvases), width),
                vocab.PAD,
                dtype=torch.long,
                device=device,
            )
            prompt_padding = torch.ones_like(prompt_tokens, dtype=torch.bool)
            for row, canvas in enumerate(canvases):
                values = [token for token, _, _, _ in canvas]
                prompt_tokens[row, : len(values)] = torch.tensor(
                    values, device=device
                )
                prompt_padding[row, : len(values)] = False
            fixed_context = model.prompt_shape_context(
                prompt_tokens,
                prompt_padding,
                source=conditional_context_source,
            )
        rounds = [0] * len(batch)
        unfinished = [False] * len(batch)

        for _ in range(max_rounds):
            active = [
                index
                for index, canvas in enumerate(canvases)
                if any(is_open for _, _, is_open, _ in canvas)
            ]
            if not active:
                break
            width = max(len(canvases[index]) for index in active)
            tokens = torch.full(
                (len(active), width), vocab.PAD, dtype=torch.long, device=device
            )
            padding = torch.ones_like(tokens, dtype=torch.bool)
            open_mask = torch.zeros_like(tokens, dtype=torch.bool)
            slot_semantics = torch.zeros(
                len(active),
                width,
                model.d_model,
                device=device,
                dtype=model.semantic_embedding_mean.dtype,
            )
            steps = torch.tensor(
                [rounds[index] for index in active],
                dtype=torch.long,
                device=device,
            )
            for row, index in enumerate(active):
                canvas = canvases[index]
                raw = [token for token, _, _, _ in canvas]
                tokens[row, : len(raw)] = torch.tensor(raw, device=device)
                padding[row, : len(raw)] = False
                open_mask[row, : len(raw)] = torch.tensor(
                    [is_open for _, _, is_open, _ in canvas], device=device
                )
                for position, (_, _, _, state) in enumerate(canvas):
                    if state is not None:
                        slot_semantics[row, position] = state.to(device)

            if conditional_context_source is None:
                sampled = model.sample_unified_structure(
                    tokens,
                    padding,
                    steps,
                    open_mask,
                    slot_semantics=slot_semantics,
                    generator=generator,
                )
                stops, degrees, directions, _, posterior_states = sampled[:5]
            else:
                if skip_round_encoding:
                    posterior_states = None
                else:
                    token_logits = model.unified_logits(
                        tokens,
                        padding,
                        steps,
                        open_mask,
                        slot_semantics=slot_semantics,
                    )[0]
                    posterior_states = model.posterior_states(token_logits)
                stops = torch.zeros_like(tokens, dtype=torch.bool)
                degrees = torch.zeros_like(tokens)
                directions = torch.zeros_like(tokens)
                for row, index in enumerate(active):
                    open_count = int(open_mask[row].sum())
                    completed_count = sum(
                        region >= 0 and not is_open
                        for _, region, is_open, _ in canvases[index]
                    )
                    root, regime, degree, direction = (
                        model.conditional_shape_logits(
                            fixed_context[index : index + 1],
                            rounds[index],
                            open_count,
                            completed_count,
                        )
                    )
                    regime_index = int(torch.multinomial(
                        regime[0].softmax(dim=-1),
                        1,
                        generator=generator,
                    ))
                    positions = open_mask[row].nonzero().flatten()
                    if rounds[index] == 0:
                        stop = torch.rand((), device=device, generator=generator)
                        stops[row, positions] = stop < root[0].sigmoid()
                    degrees[row, positions] = torch.multinomial(
                        degree[0, regime_index].softmax(dim=-1),
                        positions.numel(),
                        replacement=True,
                        generator=generator,
                    )
                    directions[row, positions] = torch.multinomial(
                        direction[0, regime_index].softmax(dim=-1),
                        positions.numel(),
                        replacement=True,
                        generator=generator,
                    )
            stops = stops.cpu()
            degrees = degrees.cpu()
            directions = directions.cpu()
            if posterior_states is not None:
                posterior_states = posterior_states.float().cpu()

            for row, index in enumerate(active):
                expanded = []
                initial = rounds[index] == 0
                for position, (token, region, is_open, state) in enumerate(
                    canvases[index]
                ):
                    if not is_open:
                        expanded.append((token, region, False, state))
                        continue
                    if initial and bool(stops[row, position]):
                        continue
                    degree = int(degrees[row, position])
                    direction = int(directions[row, position])
                    left_child = degree == 2 or (
                        degree == 1 and direction == 0
                    )
                    right_child = degree == 2 or (
                        degree == 1 and direction == 1
                    )
                    if left_child:
                        expanded.append((vocab.GAP, region, True, None))
                    expanded.append((
                        vocab.GAP,
                        region,
                        False,
                        None
                        if posterior_states is None
                        else posterior_states[row, position].clone(),
                    ))
                    if right_child:
                        expanded.append((vocab.GAP, region, True, None))
                rounds[index] += 1
                slots = sum(
                    region >= 0 and not is_open
                    for _, region, is_open, _ in expanded
                )
                if slots > max_decode_span:
                    unfinished[index] = True
                    expanded = [
                        (token, region, False, state)
                        for token, region, is_open, state in expanded
                        if not is_open
                    ]
                canvases[index] = expanded

        # The same backbone/head now resolves all dynamically created slots in
        # one parallel pass; this is not a separate lexical model.
        width = max(len(canvas) for canvas in canvases)
        final_tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        final_padding = torch.ones_like(final_tokens, dtype=torch.bool)
        final_states = torch.zeros(
            len(batch),
            width,
            model.d_model,
            device=device,
            dtype=model.semantic_embedding_mean.dtype,
        )
        final_steps = torch.tensor(rounds, dtype=torch.long, device=device)
        for row, canvas in enumerate(canvases):
            raw = [token for token, _, _, _ in canvas]
            final_tokens[row, : len(raw)] = torch.tensor(raw, device=device)
            final_padding[row, : len(raw)] = False
            for position, (_, _, _, state) in enumerate(canvas):
                if state is not None:
                    final_states[row, position] = state.to(device)
        final_logits = model.unified_logits(
            final_tokens,
            final_padding,
            final_steps,
            torch.zeros_like(final_tokens, dtype=torch.bool),
            slot_semantics=final_states,
        )[0]
        generated = model.generated_token_ids.to(device)
        chosen = generated[
            final_logits.index_select(-1, generated).argmax(dim=-1)
        ].cpu()

        for local, ((owner, _), canvas) in enumerate(zip(batch, canvases)):
            failed = unfinished[local] or any(
                is_open for _, _, is_open, _ in canvas
            )
            positions = [
                position
                for position, (_, region, is_open, _) in enumerate(canvas)
                if region == 0 and not is_open
            ]
            prediction = (
                []
                if failed
                else [int(chosen[local, position]) for position in positions]
            )
            samples[owner].append(prediction)
            all_rounds[owner].append(rounds[local])
            all_unfinished[owner].append(failed)
    return samples, all_rounds, all_unfinished


@torch.inference_mode()
def fill_sampled_scaffolds(
    masked_model,
    examples: Sequence[TextInfillingExample],
    sampled_lengths: Sequence[Sequence[int]],
    unfinished: Sequence[Sequence[bool]],
    vocab: TextVocabulary,
    device: torch.device,
    batch_size: int = 64,
    sampled_codes: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    token_codes: Optional[torch.Tensor] = None,
    semantic_logit_bias: float = 0.0,
    sampled_states: Optional[Sequence[Sequence[torch.Tensor]]] = None,
    semantic_residual_scale: float = 0.0,
) -> List[List[List[int]]]:
    """Fill dynamically generated mask canvases with one parallel MLM pass."""
    if not (len(examples) == len(sampled_lengths) == len(unfinished)):
        raise ValueError("examples, lengths, and unfinished flags must align")
    if sampled_codes is not None and len(sampled_codes) != len(examples):
        raise ValueError("sampled semantic codes must align with examples")
    if sampled_states is not None and len(sampled_states) != len(examples):
        raise ValueError("sampled semantic states must align with examples")
    if semantic_logit_bias and (
        sampled_codes is None or token_codes is None
    ):
        raise ValueError("semantic bias needs sampled and vocabulary codes")
    if semantic_residual_scale and sampled_states is None:
        raise ValueError("semantic residual scale needs sampled states")
    samples: List[List[Optional[List[int]]]] = [
        [None] * len(rows) for rows in sampled_lengths
    ]
    records = [
        (
            owner,
            sample,
            example,
            int(length),
            bool(failed),
            (
                list(sampled_codes[owner][sample])
                if sampled_codes is not None
                else []
            ),
            (
                sampled_states[owner][sample]
                if sampled_states is not None
                else torch.empty(0)
            ),
        )
        for owner, (example, lengths, failures) in enumerate(
            zip(examples, sampled_lengths, unfinished)
        )
        for sample, (length, failed) in enumerate(zip(lengths, failures))
    ]
    masked_model.eval()
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        rows = [
            example.prompt(vocab)
            for _, _, example, _, _, _, _ in batch
        ]
        width = max(len(row) for row in rows)
        tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        for row, values in enumerate(rows):
            tokens[row, : len(values)] = torch.tensor(values, device=device)
            padding[row, : len(values)] = False
        counts = [length for _, _, _, length, _, _, _ in batch]
        residuals = [record[6] for record in batch]
        logits, _ = masked_model.predict_tokens(
            tokens,
            padding,
            counts,
            mask_residuals=(
                residuals if semantic_residual_scale else None
            ),
            residual_scale=semantic_residual_scale,
        )
        allowed = logits.index_select(-1, generated_ids)
        if semantic_logit_bias:
            generated_codes = token_codes.to(device).index_select(
                0, generated_ids
            )
            for row, record in enumerate(batch):
                codes = record[5]
                for position, code in enumerate(codes[: allowed.size(1)]):
                    if int(code) >= 0:
                        allowed[row, position] = allowed[row, position] + (
                            float(semantic_logit_bias)
                            * generated_codes.eq(int(code)).to(allowed.dtype)
                        )
        chosen = generated_ids[allowed.argmax(dim=-1)].cpu().tolist()
        for record, values in zip(batch, chosen):
            owner, sample, _, length, failed, _, _ = record
            samples[owner][sample] = [] if failed else values[:length]
    return [[row or [] for row in rows] for rows in samples]


def scaffold_length_distribution(
    model,
    max_length: int,
    max_rounds: Optional[int] = None,
) -> torch.Tensor:
    """Exact total-progeny distribution of the context-free shape policy.

    The final element is overflow/nontermination mass. A regime is shared by
    all open nodes in a round and marginalized before advancing the dynamic
    program. No target length or preallocated canvas is used by this process.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if max_rounds is None:
        max_rounds = model.max_steps
    max_rounds = min(int(max_rounds), int(model.max_steps))
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    root_logit = model.root_stop_prior + model.calibration_root_bias
    empty_probability = root_logit.sigmoid()
    zero = root_logit * 0.0
    terminal = [zero for _ in range(max_length + 1)]
    terminal[0] = empty_probability
    overflow = zero
    active = {(0, 1): 1.0 - empty_probability}

    for step in range(max_rounds):
        base_regime_logits = (
            model.regime_prior.weight[step]
            + model.calibration_regime_bias[step]
        )
        base_degree_logits = (
            model.degree_prior[step]
            + model.calibration_degree_bias[step].unsqueeze(0)
        )
        next_active: Dict[Tuple[int, int], torch.Tensor] = {}
        for (emitted, frontier_size), state_probability in active.items():
            new_emitted = emitted + frontier_size
            if new_emitted > max_length:
                overflow = overflow + state_probability
                continue
            regime_logits = base_regime_logits
            degree_logits = base_degree_logits
            if bool(getattr(model, "state_feedback", False)):
                open_index = min(frontier_size, model.state_bins - 1)
                completed_index = min(emitted, model.state_bins - 1)
                regime_logits = (
                    regime_logits
                    + model.open_regime_prior.weight[open_index]
                    + model.completed_regime_prior.weight[completed_index]
                )
                degree_logits = degree_logits + (
                    model.open_degree_prior.weight[open_index]
                    + model.completed_degree_prior.weight[completed_index]
                ).view(model.regimes, 3)
            regime_probability = regime_logits.softmax(dim=-1)
            degree_probability = degree_logits.softmax(dim=-1)
            offspring_by_regime: List[List[torch.Tensor]] = []
            for regime in range(model.regimes):
                polynomial = [degree_probability[regime, 0] * 0.0 + 1.0]
                for _ in range(frontier_size):
                    expanded = [
                        degree_probability[regime, 0] * 0.0
                        for _ in range(len(polynomial) + 2)
                    ]
                    for count, probability in enumerate(polynomial):
                        for degree in range(3):
                            expanded[count + degree] = (
                                expanded[count + degree]
                                + probability
                                * degree_probability[regime, degree]
                            )
                    polynomial = expanded
                offspring_by_regime.append(polynomial)
            for child_count in range(2 * frontier_size + 1):
                child_probability = sum(
                    regime_probability[regime]
                    * offspring_by_regime[regime][child_count]
                    for regime in range(model.regimes)
                )
                mass = state_probability * child_probability
                if child_count == 0:
                    terminal[new_emitted] = terminal[new_emitted] + mass
                else:
                    key = (new_emitted, child_count)
                    next_active[key] = next_active.get(key, zero) + mass
        active = next_active
    for state_probability in active.values():
        overflow = overflow + state_probability
    return torch.stack([*terminal, overflow])


def conditional_scaffold_length_distribution(
    model,
    context: torch.Tensor,
    max_length: int,
    max_rounds: Optional[int] = None,
) -> torch.Tensor:
    """Exact per-prompt total-progeny distribution, batched over prompts.

    `scaffold_length_distribution` marginalizes one process shared by every
    prompt.  This runs the same recursion with prompt-conditioned logits, which
    stays exact because `conditional_shape_logits` reads a context fixed at
    round zero: the process is context-free *given the prompt*.  The final
    column is overflow/nontermination mass.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if max_rounds is None:
        max_rounds = model.max_steps
    max_rounds = min(int(max_rounds), int(model.max_steps))
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    batch = context.size(0)
    root_logits, _, _, _ = model.conditional_shape_logits(context, 0, 1, 0)
    empty_probability = root_logits.sigmoid()
    zero = torch.zeros_like(empty_probability)
    terminal = [zero for _ in range(max_length + 1)]
    terminal[0] = empty_probability
    overflow = zero
    active = {(0, 1): 1.0 - empty_probability}

    for step in range(max_rounds):
        next_active: Dict[Tuple[int, int], torch.Tensor] = {}
        for (emitted, frontier_size), state_probability in active.items():
            new_emitted = emitted + frontier_size
            if new_emitted > max_length:
                overflow = overflow + state_probability
                continue
            _, regime_logits, degree_logits, _ = model.conditional_shape_logits(
                context, step, frontier_size, emitted
            )
            regime_probability = regime_logits.softmax(dim=-1)
            degree_probability = degree_logits.softmax(dim=-1)
            offspring_by_regime: List[List[torch.Tensor]] = []
            for regime in range(model.regimes):
                polynomial = [torch.ones_like(zero)]
                for _ in range(frontier_size):
                    expanded = [
                        zero for _ in range(len(polynomial) + 2)
                    ]
                    for count, probability in enumerate(polynomial):
                        for degree in range(3):
                            expanded[count + degree] = (
                                expanded[count + degree]
                                + probability
                                * degree_probability[:, regime, degree]
                            )
                    polynomial = expanded
                offspring_by_regime.append(polynomial)
            for child_count in range(2 * frontier_size + 1):
                child_probability = sum(
                    regime_probability[:, regime]
                    * offspring_by_regime[regime][child_count]
                    for regime in range(model.regimes)
                )
                mass = state_probability * child_probability
                if child_count == 0:
                    terminal[new_emitted] = terminal[new_emitted] + mass
                else:
                    key = (new_emitted, child_count)
                    next_active[key] = next_active.get(key, zero) + mass
        active = next_active
    for state_probability in active.values():
        overflow = overflow + state_probability
    return torch.stack([*terminal, overflow], dim=-1)


def sampled_length_probabilities(
    samples: Sequence[Sequence[Sequence[int]]],
    unfinished: Sequence[Sequence[bool]],
    support_max: int = 8,
) -> List[List[float]]:
    """Convert Monte Carlo rollouts to per-prompt length distributions."""
    probabilities: List[List[float]] = []
    for prompt_samples, prompt_unfinished in zip(samples, unfinished):
        if len(prompt_samples) != len(prompt_unfinished):
            raise ValueError("samples and unfinished flags must align")
        if not prompt_samples:
            raise ValueError("each prompt needs at least one sample")
        counts = [0] * (support_max + 2)
        for sequence, failed in zip(prompt_samples, prompt_unfinished):
            category = (
                support_max + 1
                if failed or len(sequence) > support_max
                else len(sequence)
            )
            counts[category] += 1
        probabilities.append(
            [count / len(prompt_samples) for count in counts]
        )
    return probabilities


@torch.inference_mode()
def frontier_structure_diagnostics(
    model,
    dataset: Dataset,
    vocab: TextVocabulary,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, object]:
    """Measure whether topology probabilities vary beyond a depth-only prior."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    root_rows: List[Tuple[int, float]] = []
    degree_rows: Dict[int, List[Tuple[int, List[float]]]] = {}
    model.eval()
    for batch in loader:
        tokens = batch["tokens"].to(device)
        padding = batch["padding"].to(device)
        steps = batch["steps"].to(device)
        targets = batch["targets"].to(device)
        left = batch["left_targets"].to(device)
        right = batch["right_targets"].to(device)
        degree_targets, _ = topology_targets(left, right)
        _, root_logits, degree_logits, _, _ = model(tokens, padding, steps)

        root_valid = tokens.eq(vocab.GAP) & steps.unsqueeze(1).eq(0)
        root_targets = targets.eq(vocab.stop_action)
        for target, probability in zip(
            root_targets[root_valid].cpu().tolist(),
            root_logits[root_valid].sigmoid().cpu().tolist(),
        ):
            root_rows.append((int(target), float(probability)))

        degree_valid = degree_targets.ge(0) & targets.ge(0) & targets.lt(
            vocab.vocab_size
        )
        degree_probabilities = degree_logits.softmax(dim=-1)
        for row in range(tokens.size(0)):
            valid = degree_valid[row]
            if not bool(valid.any()):
                continue
            step = int(steps[row])
            bucket = degree_rows.setdefault(step, [])
            for target, probabilities in zip(
                degree_targets[row, valid].cpu().tolist(),
                degree_probabilities[row, valid].cpu().tolist(),
            ):
                bucket.append((int(target), [float(x) for x in probabilities]))

    def summarize_degree(
        rows: Sequence[Tuple[int, List[float]]]
    ) -> Dict[str, object]:
        count = len(rows)
        target_counts = [sum(target == index for target, _ in rows) for index in range(3)]
        argmaxes = [max(range(3), key=lambda index: row[1][index]) for row in rows]
        mean_probabilities = [
            sum(probabilities[index] for _, probabilities in rows) / max(1, count)
            for index in range(3)
        ]
        probability_std = [
            math.sqrt(
                sum(
                    (probabilities[index] - mean_probabilities[index]) ** 2
                    for _, probabilities in rows
                )
                / max(1, count)
            )
            for index in range(3)
        ]
        return {
            "count": count,
            "target_distribution": [value / max(1, count) for value in target_counts],
            "predicted_mean_probabilities": mean_probabilities,
            "predicted_probability_std": probability_std,
            "argmax_distribution": [argmaxes.count(index) / max(1, count) for index in range(3)],
            "accuracy": sum(predicted == target for predicted, (target, _) in zip(argmaxes, rows)) / max(1, count),
            "mean_entropy_nats": -sum(
                sum(value * math.log(max(value, 1e-12)) for value in probabilities)
                for _, probabilities in rows
            ) / max(1, count),
        }

    root_probabilities = [probability for _, probability in root_rows]
    root_mean = sum(root_probabilities) / max(1, len(root_probabilities))
    all_degree_rows = [row for rows in degree_rows.values() for row in rows]
    return {
        "root": {
            "count": len(root_rows),
            "target_stop_rate": sum(target for target, _ in root_rows) / max(1, len(root_rows)),
            "predicted_stop_mean": root_mean,
            "predicted_stop_std": math.sqrt(
                sum((value - root_mean) ** 2 for value in root_probabilities)
                / max(1, len(root_probabilities))
            ),
            "argmax_stop_rate": sum(value >= 0.5 for value in root_probabilities) / max(1, len(root_probabilities)),
            "brier": sum((probability - target) ** 2 for target, probability in root_rows) / max(1, len(root_rows)),
        },
        "degree_all": summarize_degree(all_degree_rows),
        "degree_by_step": {
            str(step): summarize_degree(rows)
            for step, rows in sorted(degree_rows.items())
        },
    }


def greedy_length_probabilities(
    predictions: Sequence[Sequence[Sequence[int]]],
    unfinished: Sequence[bool],
    support_max: int = 8,
) -> List[List[float]]:
    """Represent a greedy rollout as one-hot probabilities for length metrics."""
    result: List[List[float]] = []
    for rows, failed in zip(predictions, unfinished):
        if len(rows) != 1:
            raise ValueError("single-gap length metrics need one prediction per row")
        category = support_max + 1 if failed or len(rows[0]) > support_max else len(rows[0])
        probabilities = [0.0] * (support_max + 2)
        probabilities[category] = 1.0
        result.append(probabilities)
    return result
