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
        return state


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

    token_logits, root_stop, degree_logits, direction_logits, _ = model(
        tokens, padding, steps
    )
    token_valid = targets.ge(0) & targets.lt(vocab.vocab_size)
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
    return {
        "token": token_loss,
        "root": root_loss,
        "degree": degree_loss,
        "direction": direction_loss,
        "token_count": token_valid.sum(),
        "root_count": root_valid.sum(),
        "degree_count": degree_valid.sum(),
        "direction_count": direction_valid.sum(),
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
) -> Tuple[List[List[List[int]]], List[int], List[bool]]:
    """Expand every open gap in one backbone pass per round.

    Greedy decoding is useful for inspecting the conditional mode.  Stochastic
    decoding samples the actual unknown-length generative process; this matters
    when corruption length is independent of the visible prompt and therefore
    cannot be recovered as a deterministic per-prompt label.
    """
    model.eval()
    if sample_tokens is None:
        sample_tokens = stochastic
    canvases = [initial_region_canvas(example, vocab) for example in examples]
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
        steps = torch.tensor(
            [rounds[index] for index in active],
            dtype=torch.long,
            device=device,
        )
        for row, index in enumerate(active):
            raw = [token for token, _ in canvases[index]]
            tokens[row, : len(raw)] = torch.tensor(raw, device=device)
            padding[row, : len(raw)] = False

        token_logits, root_stop, degree_logits, direction_logits, _ = model(
            tokens, padding, steps
        )
        allowed = token_logits.index_select(-1, generated_ids)
        if stochastic and sample_tokens:
            token_probabilities = allowed.softmax(dim=-1)
            token_samples = torch.multinomial(
                token_probabilities.reshape(-1, token_probabilities.size(-1)),
                1,
                generator=generator,
            ).reshape(token_probabilities.shape[:-1])
            chosen = generated_ids[token_samples].cpu()
        else:
            chosen = generated_ids[allowed.argmax(dim=-1)].cpu()
        if stochastic:
            stop = (
                torch.rand(
                    root_stop.shape,
                    device=device,
                    generator=generator,
                )
                < root_stop.sigmoid()
            ).cpu()
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
            degree = degree_logits.argmax(dim=-1).cpu()
            direction = direction_logits.argmax(dim=-1).cpu()

        for row, index in enumerate(active):
            expanded: List[Tuple[int, int]] = []
            initial = rounds[index] == 0
            for position, (token, region) in enumerate(canvases[index]):
                if token != vocab.GAP:
                    expanded.append((token, region))
                    continue
                if initial and bool(stop[row, position]):
                    continue
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
                expanded.append((pivot, region))
                if right_child:
                    expanded.append((vocab.GAP, region))
            rounds[index] += 1
            for gap_index in range(len(examples[index].spans)):
                generated = sum(
                    token != vocab.GAP and region == gap_index
                    for token, region in expanded
                )
                if generated > max_decode_span:
                    unfinished[index] = True
            if unfinished[index]:
                expanded = [item for item in expanded if item[0] != vocab.GAP]
            canvases[index] = expanded

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
) -> Tuple[List[List[List[int]]], List[List[int]], List[List[bool]]]:
    """Draw ancestral samples without supplying a target length or canvas."""
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if any(len(example.spans) != 1 for example in examples):
        raise ValueError("sampling metrics currently require one gap per prompt")

    samples: List[List[List[int]]] = [[] for _ in examples]
    rounds: List[List[int]] = [[] for _ in examples]
    unfinished: List[List[bool]] = [[] for _ in examples]
    replicas = [
        (index, example)
        for index, example in enumerate(examples)
        for _ in range(samples_per_prompt)
    ]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for start in range(0, len(replicas), chunk_size):
        batch = replicas[start : start + chunk_size]
        predictions, batch_rounds, batch_unfinished = decode_frontier_model(
            model,
            [example for _, example in batch],
            vocab,
            device,
            max_rounds=max_rounds,
            max_decode_span=max_decode_span,
            stochastic=True,
            generator=generator,
            sample_tokens=sample_tokens,
        )
        for (owner, _), prediction, steps, failed in zip(
            batch, predictions, batch_rounds, batch_unfinished
        ):
            samples[owner].append(prediction[0])
            rounds[owner].append(steps)
            unfinished[owner].append(failed)
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
            if hasattr(model, "sample_structure"):
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
) -> Tuple[List[List[List[int]]], List[List[int]], List[List[bool]]]:
    """Grow shape and token beliefs together, then decode in parallel.

    No length or fixed output canvas is supplied.  Each expansion pass emits a
    topology decision and a soft lexical state for the same node.  Once the
    branching process terminates, the same MLM head fills every completed slot
    simultaneously.
    """
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

            sampled = model.sample_unified_structure(
                tokens,
                padding,
                steps,
                open_mask,
                slot_semantics=slot_semantics,
                generator=generator,
            )
            stops, degrees, directions, _, posterior_states = sampled[:5]
            stops = stops.cpu()
            degrees = degrees.cpu()
            directions = directions.cpu()
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
                        posterior_states[row, position].clone(),
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
