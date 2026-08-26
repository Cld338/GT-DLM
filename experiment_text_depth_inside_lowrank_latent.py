"""Finite shared regimes with component-specific low-rank output adapters."""

import argparse
import copy
import json
import math
import os
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from experiment import choose_device, parameter_count, seed_everything
from experiment_text_depth_inside_multigap import (
    collate_multi_prompt_contexts,
    multi_depth_gap_log_likelihoods,
)
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


class LowRankHeadSharedLatentModel(nn.Module):
    """Exact shared mixture with low-rank token, STOP, and topology adapters."""

    def __init__(self, base: IntervalInsideBoundaryModel, regimes: int, rank: int) -> None:
        super().__init__()
        if regimes < 1 or rank < 1:
            raise ValueError("regimes and rank must be positive")
        self.base = base
        hidden = base.encoder.token_embedding.embedding_dim
        vocab_size = base.token_head.out_features
        self.regime_head = nn.Linear(hidden, regimes)
        self.token_down = nn.Parameter(torch.empty(regimes, hidden, rank))
        self.token_up = nn.Parameter(torch.zeros(regimes, rank, vocab_size))
        self.stop_down = nn.Parameter(torch.empty(regimes, hidden, rank))
        self.stop_up = nn.Parameter(torch.zeros(regimes, rank))
        self.topology_down = nn.Parameter(torch.empty(regimes, 2 * hidden, rank))
        self.topology_up = nn.Parameter(torch.zeros(regimes, rank, 4))
        nn.init.zeros_(self.regime_head.weight)
        nn.init.zeros_(self.regime_head.bias)
        nn.init.normal_(self.token_down, std=0.02)
        nn.init.normal_(self.stop_down, std=0.02)
        nn.init.normal_(self.topology_down, std=0.02)

    @property
    def regimes(self) -> int:
        return self.token_down.size(0)

    @property
    def rank(self) -> int:
        return self.token_down.size(-1)

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def prompt_gate(self, encoded: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        visible = (~padding).unsqueeze(-1)
        pooled = (encoded * visible).sum(1) / visible.sum(1).clamp_min(1)
        return self.regime_head(pooled)

    def interval_logits(self, regime, context, left, right, depths):
        token, stop, hidden = self.base.interval_logits(context, left, right, depths)
        token_features = hidden.matmul(self.token_down[regime])
        stop_features = hidden.matmul(self.stop_down[regime])
        token = token + token_features.matmul(self.token_up[regime])
        stop = stop + stop_features.matmul(self.stop_up[regime])
        return token, stop, hidden

    def topology_logits(self, regime, hidden, chosen_tokens):
        logits = self.base.topology_logits(hidden, chosen_tokens)
        token = self.base.encoder.token_embedding(chosen_tokens)
        features = torch.cat((hidden, token), dim=-1)
        low_rank = features.matmul(self.topology_down[regime])
        return logits + low_rank.matmul(self.topology_up[regime])


def lowrank_shared_latent_log_likelihoods(
    model: LowRankHeadSharedLatentModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, padding, roots = collate_multi_prompt_contexts(examples, vocab, device)
    encoded = model.base.encode(tokens, padding)
    gate_logp = model.prompt_gate(encoded, padding).log_softmax(-1)
    component_exact, component_midpoint, component_gaps = [], [], []
    for regime in range(model.regimes):
        exact, midpoint, gaps = multi_depth_gap_log_likelihoods(
            model.base, examples, vocab, device, encoded=encoded,
            interval_logits_fn=lambda context, left, right, depths, z=regime:
                model.interval_logits(z, context, left, right, depths),
            topology_logits_fn=lambda hidden, chosen, z=regime:
                model.topology_logits(z, hidden, chosen),
        )
        component_exact.append(exact)
        component_midpoint.append(midpoint)
        component_gaps.append(gaps)
    component_exact = torch.stack(component_exact, dim=-1)
    component_midpoint = torch.stack(component_midpoint, dim=-1)
    component_gaps = torch.stack(component_gaps, dim=-1)
    joint_terms = gate_logp + component_exact
    joint = torch.logsumexp(joint_terms, dim=-1)
    midpoint = torch.logsumexp(gate_logp + component_midpoint, dim=-1)
    posterior = joint_terms.softmax(-1)
    owners = torch.tensor([root[0] for root in roots], dtype=torch.long, device=device)
    marginal_gaps = torch.logsumexp(gate_logp[owners] + component_gaps, dim=-1)
    return joint, midpoint, marginal_gaps, gate_logp, posterior


@torch.inference_mode()
def evaluate(model, examples, vocab, device, batch_size) -> Dict[str, object]:
    model.eval()
    joint_values, midpoint_values, gap_values = [], [], []
    gate_total = torch.zeros(model.regimes, device=device)
    posterior_total = torch.zeros(model.regimes, device=device)
    gate_entropy_total = 0.0
    posterior_entropy_total = 0.0
    count = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        joint, midpoint, gaps, gate_logp, posterior = lowrank_shared_latent_log_likelihoods(
            model, batch, vocab, device
        )
        joint_values.extend(joint.cpu().tolist())
        midpoint_values.extend(midpoint.cpu().tolist())
        gap_values.extend(gaps.cpu().tolist())
        gate = gate_logp.exp()
        gate_total += gate.sum(0)
        posterior_total += posterior.sum(0)
        gate_entropy_total += float((-(gate * gate_logp).sum(-1)).sum())
        posterior_entropy_total += float(
            (-(posterior * posterior.clamp_min(1e-30).log()).sum(-1)).sum()
        )
        count += len(batch)
    return {
        "joint_sequence_nll": -sum(joint_values) / count,
        "nll_per_gap": -sum(gap_values) / len(gap_values),
        "midpoint_joint_nll": -sum(midpoint_values) / count,
        "mean_marginal_gain_nats": sum(
            exact - midpoint for exact, midpoint in zip(joint_values, midpoint_values)
        ) / count,
        "mean_gate_probabilities": (gate_total / count).cpu().tolist(),
        "mean_posterior_probabilities": (posterior_total / count).cpu().tolist(),
        "mean_gate_entropy_nats": gate_entropy_total / count,
        "mean_posterior_entropy_nats": posterior_entropy_total / count,
        "effective_posterior_regimes": math.exp(posterior_entropy_total / count),
    }


def train_epoch(model, source, vocab, device, batch_size, optimizer, epoch) -> float:
    model.train()
    source.set_epoch(epoch)
    loader = DataLoader(source, batch_size=batch_size, shuffle=True, collate_fn=lambda rows: rows)
    total, count = 0.0, 0
    for examples in loader:
        optimizer.zero_grad(set_to_none=True)
        joint, _, _, _, _ = lowrank_shared_latent_log_likelihoods(
            model, examples, vocab, device
        )
        loss = -joint.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        total += float(-joint.detach().sum())
        count += len(examples)
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument("--artifact-dir", default="artifacts/text_depth_inside_lowrank_latent")
    parser.add_argument("--checkpoint", default="artifacts/text_depth_inside_multigap_screen/inside.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--regimes", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    with open(os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8") as handle:
        base_result = json.load(handle)
    config = base_result["config"]
    data_seed = int(config["seed"])
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(os.path.join(str(config["data_dir"]), "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(os.path.join(str(config["data_dir"]), "corpus.pt"), map_location="cpu", weights_only=True)
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    source = DynamicTextExampleDataset(
        corpus["train"], seed=args.seed, gap_counts=(2,), min_span=1, max_span=8,
        random_window_min=window_min, random_window_max=window_max,
    )
    validation_docs = random_length_windows(corpus["validation"], data_seed + 401, window_min, window_max)
    validation = sample_text_infilling_examples(
        validation_docs, data_seed + 201, gap_counts=(2,), min_span=1, max_span=8
    )
    base = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    ).to(device)
    base.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = LowRankHeadSharedLatentModel(base, args.regimes, args.rank).to(device)
    model.freeze_base()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    initial = evaluate(model, validation, vocab, device, args.batch_size)
    best_nll = initial["joint_sequence_nll"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    os.makedirs(args.artifact_dir, exist_ok=True)
    print("epoch 0 validation_joint_nll={:.4f}".format(best_nll), flush=True)
    for epoch in range(args.epochs):
        training_nll = train_epoch(model, source, vocab, device, args.batch_size, optimizer, epoch)
        metrics = evaluate(model, validation, vocab, device, args.batch_size)
        history.append({"epoch": epoch + 1, "training_nll": training_nll, "validation": metrics})
        print("epoch {} train_joint_nll={:.4f} validation_joint_nll={:.4f}".format(
            epoch + 1, training_nll, metrics["joint_sequence_nll"]
        ), flush=True)
        torch.save(model.state_dict(), os.path.join(args.artifact_dir, "latest.pt"))
        if metrics["joint_sequence_nll"] < best_nll:
            best_nll = metrics["joint_sequence_nll"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected_validation = evaluate(model, validation, vocab, device, args.batch_size)
    result = {
        "config": {**config, **vars(args), "data_seed": data_seed,
                   "objective": "lowrank_head_shared_latent_exact_inside"},
        "parameters": parameter_count(model),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_validation": initial, "history": history,
        "selected_epoch": best_epoch, "validation": selected_validation,
    }
    if args.evaluate_test:
        test_docs = random_length_windows(corpus["test"], data_seed + 403, window_min, window_max)
        test = sample_text_infilling_examples(
            test_docs, data_seed + 101, gap_counts=(2,), min_span=1, max_span=8
        )[:128]
        result["test"] = evaluate(model, test, vocab, device, args.batch_size)
    torch.save(model.state_dict(), os.path.join(args.artifact_dir, "lowrank_latent.pt"))
    with open(os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Low-rank head shared-latent validation screen", "",
        "| Regimes | Rank | Trainable parameters | Selected epoch | Validation NLL | Effective regimes |",
        "|---:|---:|---:|---:|---:|---:|",
        "| {} | {} | {:,} | {} | {:.3f} | {:.3f} |".format(
            args.regimes, args.rank, result["trainable_parameters"], best_epoch,
            selected_validation["joint_sequence_nll"],
            selected_validation["effective_posterior_regimes"],
        ),
    ]
    with open(os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
