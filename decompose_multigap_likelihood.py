"""Split the two-gap exact log likelihood into lexical, structural, and
tree-entropy components, and compare each against the baselines' own splits.

`research/ROADMAP.md` open item 3. The factorized exact model wins joint NLL by
`-5.9` to `-7.5` nats against both proper baselines, yet generates worse text
than they do (free-sample token accuracy `2.1%` against the masked baseline's
`3.7%`). That combination is only explicable if the likelihood advantage lives
somewhere other than token prediction, so this script measures where.

For a gap whose chart entries are `w_e = token_e + topology_e`, write `q` for
the posterior over ordered pivot trees, `q(T) proportional to exp(sum_{e in T}
w_e)`. Then exactly

    log p(x) = root + E_q[sum token_e] + E_q[sum topology_e] + H(q)
               |___structure___|   |__lexical__|  |__structure__|  |_entropy_|

because `H(q) = log Z - sum_e mu_e w_e` with edge marginals
`mu_e = d log Z / d w_e`. Those marginals are exactly what one backward pass
through the inside recurrence returns, so the split costs one extra gradient
evaluation and no training.

The entropy term is the interesting one. Both baselines assign each string a
single derivation, so their `H` is identically zero; only the latent-tree model
collects credit for the number of trees that explain the same string. A length-8
span admits Catalan(8) = 1430 ordered pivot trees, so this term alone can reach
`log 1430 = 7.3` nats. If it accounts for most of the measured advantage, the
advantage is real probability mass but is not evidence of better token
modeling -- and it predicts exactly the observed generation failure, since
sampling commits to one tree and cannot draw on the entropy.

Scope limit: the exact model's topology head is conditioned on the emitted
token, so "structure" and "lexical" are a decomposition of the objective's
factors, not a causal attribution of the model's capabilities. The identity
above is exact, and each model is split along its own factorization, which is
what makes the term-by-term comparison meaningful.
"""

import argparse
import json
import os
from typing import Dict, List, Sequence

import torch
from tokenizers import Tokenizer

from evaluate_text_sequence_likelihoods import (
    masked_log_likelihoods,
    paired_bootstrap,
    sequential_log_likelihoods,
)
from experiment import choose_device
from experiment_text_depth_inside_multigap import multi_depth_gap_log_likelihoods
from gtdlm.inside import (
    batched_depth_inside_log_partition,
    batched_depth_midpoint_tree_log_weight,
)
from gtdlm.model import (
    GapTreeFactorizedBoundaryModel,
    IntervalInsideBoundaryModel,
    LengthMaskedModel,
)
from gtdlm.text_data import (
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer

COMPONENTS = ("lexical", "structure", "tree_entropy")
REPORT_ORDER = (
    "factorized_depth_exact",
    "factorized_depth_exact_midpoint_tree",
    "sequential_filler",
    "length_masked",
)


def decompose_exact_batch(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    tree: str = "posterior",
) -> Dict[str, torch.Tensor]:
    """Return per-example lexical/structure/tree-entropy parts for one batch.

    ``tree`` selects which distribution over trees the token and topology
    expectations are taken under.

    ``"posterior"`` uses ``q(T | x)``, the gold-conditioned tree posterior. Its
    edge marginals are the gradient of the inside log partition. This is the
    split of the reported exact likelihood.

    ``"midpoint"`` uses the deterministic midpoint tree, whose pivots are
    ``(lo + hi) // 2`` and therefore depend on the span length alone, never on
    which tokens the span contains. Its "marginals" are the gradient of a plain
    sum, so they are 0/1 indicators of the single tree's edges, and the entropy
    term is identically zero. Comparing the two isolates how much of the
    lexical advantage requires choosing the tree with knowledge of the answer.
    """
    if tree not in ("posterior", "midpoint"):
        raise ValueError("tree must be 'posterior' or 'midpoint'")
    partition_fn = (
        batched_depth_inside_log_partition if tree == "posterior"
        else batched_depth_midpoint_tree_log_weight
    )
    with torch.no_grad():
        exact, midpoint, _, charts = multi_depth_gap_log_likelihoods(
            model, examples, vocab, device, return_charts=True
        )
    reference = exact if tree == "posterior" else midpoint
    owner = charts["owner"]
    gap_count = int(owner.numel())
    lexical = torch.zeros(gap_count, device=device)
    topology = torch.zeros(gap_count, device=device)
    entropy = torch.zeros(gap_count, device=device)
    root = torch.zeros(gap_count, device=device)
    for gap_index, value in charts["root"].items():
        root[gap_index] = value

    # Root-relative depth is the first chart axis. Stratifying the lexical term
    # by depth separates tokens predicted from a tight gold interval (deep
    # nodes) from the root token, which free generation must emit first with
    # only the prompt boundaries for context.
    max_depth = max(
        (chart.shape[0] for chart in charts["combined"].values()), default=1
    )
    depth_lexical = torch.zeros(len(examples), max_depth, device=device)
    depth_count = torch.zeros(len(examples), max_depth, device=device)

    # Group by span length so that charts of equal shape stack, mirroring the
    # likelihood path. Posterior edge marginals come from one backward pass.
    by_shape: Dict[tuple, List[int]] = {}
    for gap_index, chart in charts["combined"].items():
        by_shape.setdefault(tuple(chart.shape), []).append(gap_index)
    for shape, group in by_shape.items():
        stacked = torch.stack(
            [charts["combined"][index] for index in group]
        ).detach().requires_grad_(True)
        with torch.enable_grad():
            log_partition = partition_fn(stacked)
            marginals, = torch.autograd.grad(log_partition.sum(), stacked)
        token_part = torch.stack([charts["token"][index] for index in group])
        topology_part = torch.stack([charts["topology"][index] for index in group])
        dims = tuple(range(1, stacked.ndim))
        group_lexical = (marginals * token_part).sum(dim=dims)
        group_topology = (marginals * topology_part).sum(dim=dims)
        # H(q) = log Z - E_q[sum of used chart weights].
        group_entropy = log_partition.detach() - (group_lexical + group_topology)
        # Collapse every axis except batch and depth.
        interval_dims = tuple(range(2, stacked.ndim))
        by_depth_lexical = (marginals * token_part).sum(dim=interval_dims)
        by_depth_count = marginals.sum(dim=interval_dims)
        for offset, gap_index in enumerate(group):
            lexical[gap_index] = group_lexical[offset]
            topology[gap_index] = group_topology[offset]
            entropy[gap_index] = group_entropy[offset]
            example_index = int(owner[gap_index])
            depth = by_depth_lexical.size(1)
            depth_lexical[example_index, :depth] += by_depth_lexical[offset]
            depth_count[example_index, :depth] += by_depth_count[offset]

    per_example = {}
    for name, values in (
        ("lexical", lexical),
        ("structure", topology + root),
        ("tree_entropy", entropy),
    ):
        totals = torch.zeros(len(examples), device=device)
        totals.index_add_(0, owner, values)
        per_example[name] = totals.cpu()
    per_example["total"] = reference.cpu()
    per_example["depth_lexical"] = depth_lexical.cpu()
    per_example["depth_count"] = depth_count.cpu()
    reconstructed = sum(per_example[name] for name in COMPONENTS)
    residual = float((reconstructed - per_example["total"]).abs().max())
    if residual > 1e-3:
        raise AssertionError(
            "decomposition does not reconstruct the exact likelihood "
            "(max residual {:.6f})".format(residual)
        )
    return per_example


def decompose_exact(model, examples, vocab, device, batch_size, tree="posterior"):
    parts = [
        decompose_exact_batch(
            model, examples[start:start + batch_size], vocab, device, tree
        )
        for start in range(0, len(examples), batch_size)
    ]
    result = {
        key: torch.cat([part[key] for part in parts])
        for key in list(COMPONENTS) + ["total"]
    }
    # Batches can reach different maximum depths; pad before concatenating.
    depth = max(part["depth_lexical"].size(1) for part in parts)
    for key in ("depth_lexical", "depth_count"):
        result[key] = torch.cat([
            torch.nn.functional.pad(
                part[key], (0, depth - part[key].size(1))
            ) for part in parts
        ])
    return result


def decompose_sequential(model, examples, vocab, device, batch_size):
    total, stop, token = sequential_log_likelihoods(
        model, examples, vocab, device, batch_size, return_components=True
    )
    return {
        "lexical": token.cpu(),
        "structure": stop.cpu(),
        # Left-to-right filling gives each string exactly one derivation.
        "tree_entropy": torch.zeros(len(examples)),
        "total": total.cpu(),
    }


def decompose_masked(model, examples, vocab, device, batch_size):
    total, length, token = masked_log_likelihoods(
        model, examples, vocab, device, batch_size
    )
    return {
        "lexical": token.cpu(),
        "structure": length.cpu(),
        # One length draw plus a deterministic mask canvas: one derivation.
        "tree_entropy": torch.zeros(len(examples)),
        "total": total.cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", default="artifacts/text_trajectory")
    parser.add_argument(
        "--checkpoint-dir", default="artifacts/text_multigap_matched_training",
        help="directory holding the three from-scratch matched-training checkpoints",
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_multigap_decomposition"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    device = choose_device(args.device)
    with open(
        os.path.join(args.trajectory_dir, "results.json"), encoding="utf-8"
    ) as handle:
        trajectory = json.load(handle)
    config = trajectory["config"]
    data_seed = int(config["seed"])
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])

    # Exactly the split used by experiment_multigap_matched_training.py.
    test_docs = random_length_windows(
        corpus["test"], data_seed + 403, window_min, window_max
    )
    test = sample_text_infilling_examples(
        test_docs, data_seed + 101, gap_counts=(2,), min_span=1, max_span=8,
    )[:args.test_examples]

    shared = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    )

    def load(name, model):
        state = torch.load(
            os.path.join(args.checkpoint_dir, name + ".pt"),
            map_location=device, weights_only=True,
        )
        model.load_state_dict(state)
        return model.to(device).eval()

    exact_model = load(
        "factorized_depth_exact", IntervalInsideBoundaryModel(**shared)
    )
    sequential_model = load(
        "sequential_filler", GapTreeFactorizedBoundaryModel(**shared)
    )
    masked_model = load("length_masked", LengthMaskedModel(
        vocab.vocab_size, 16, d_model=int(config["d_model"]),
        nhead=int(config["heads"]), layers=int(config["layers"]),
        max_positions=256,
    ))

    print("decomposing {} two-gap test examples".format(len(test)))
    decompositions = {
        "factorized_depth_exact": decompose_exact(
            exact_model, test, vocab, device, args.batch_size
        ),
        # Same model, same tokens, but the tree is fixed by span length alone.
        "factorized_depth_exact_midpoint_tree": decompose_exact(
            exact_model, test, vocab, device, args.batch_size, tree="midpoint"
        ),
        "sequential_filler": decompose_sequential(
            sequential_model, test, vocab, device, args.batch_size
        ),
        "length_masked": decompose_masked(
            masked_model, test, vocab, device, args.batch_size
        ),
    }

    # Report as NLL contributions so that lower is better everywhere, matching
    # the sign convention of every other document in this project.
    nll = {
        name: {key: -values for key, values in parts.items()}
        for name, parts in decompositions.items()
    }
    means = {
        name: {key: float(values.mean()) for key, values in parts.items()}
        for name, parts in nll.items()
    }
    comparisons = {}
    contrasts = (
        "sequential_filler", "length_masked",
        # Same model and tokens, tree chosen without seeing them: the
        # difference is exactly the posterior-selection benefit.
        "factorized_depth_exact_midpoint_tree",
    )
    for other in contrasts:
        for key in list(COMPONENTS) + ["total"]:
            comparisons["exact_vs_{}_{}".format(other, key)] = paired_bootstrap(
                nll["factorized_depth_exact"][key], nll[other][key]
            )
    # The generation-relevant contrast: an x-independent tree against the
    # baselines, which never get to choose a tree using the answer either.
    for other in ("sequential_filler", "length_masked"):
        for key in list(COMPONENTS) + ["total"]:
            comparisons["midpoint_vs_{}_{}".format(other, key)] = paired_bootstrap(
                nll["factorized_depth_exact_midpoint_tree"][key], nll[other][key]
            )

    # Per-token nats by tree depth. The exact model's token head sees the gold
    # tokens flanking each interval, so deep nodes are predicted from a tight
    # two-sided context that free generation never has. Depth 0 is the root
    # token, which is emitted first from the prompt boundaries alone -- the
    # only depth whose conditioning generation actually reproduces.
    depth_lexical = decompositions["factorized_depth_exact"]["depth_lexical"]
    depth_count = decompositions["factorized_depth_exact"]["depth_count"]
    total_tokens = float(depth_count.sum())
    depth_profile = []
    for depth in range(depth_lexical.size(1)):
        count = float(depth_count[:, depth].sum())
        if count < 1e-6:
            continue
        depth_profile.append({
            "depth": depth,
            "expected_tokens": count,
            "share_of_tokens": count / total_tokens,
            "nats_per_token": float(-depth_lexical[:, depth].sum()) / count,
        })
    # The masked baseline predicts every span token in parallel from the prompt
    # alone, so its per-token cost is the fair reference for the root.
    masked_tokens = float(sum(len(span) for e in test for span in e.spans))
    baseline_per_token = {
        name: float(nll[name]["lexical"].sum()) / masked_tokens
        for name in nll
    }

    result = {
        "config": {
            **{key: config[key] for key in
               ("data_dir", "d_model", "heads", "layers", "seed",
                "random_window_min", "random_window_max") if key in config},
            **vars(args),
            "identity": "total_nll = lexical + structure + tree_entropy",
        },
        "mean_nll_components": means,
        "paired_comparisons": comparisons,
        "exact_lexical_by_depth": depth_profile,
        "lexical_nats_per_token": baseline_per_token,
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "decomposition.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# Where the two-gap likelihood advantage comes from", "",
        "Per-example NLL contributions on the {} held-out two-gap test".format(len(test)),
        "examples, for the from-scratch matched-training checkpoints. Lower is",
        "better. `total = lexical + structure + tree_entropy` holds exactly.",
        "",
        "| Model | Lexical | Structure | Tree entropy | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in REPORT_ORDER:
        row = means[name]
        lines.append("| `{}` | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            name, row["lexical"], row["structure"], row["tree_entropy"],
            row["total"],
        ))
    lines.extend([
        "",
        "| Comparison | Component | Mean NLL difference | Paired 95% CI |",
        "|---|---|---:|---:|",
    ])
    for prefix, others in (
        ("exact", contrasts),
        ("midpoint", ("sequential_filler", "length_masked")),
    ):
        for other in others:
            for key in list(COMPONENTS) + ["total"]:
                entry = comparisons["{}_vs_{}_{}".format(prefix, other, key)]
                lines.append(
                    "| {} vs `{}` | {} | {:+.3f} | [{:+.3f},{:+.3f}] |".format(
                        prefix, other, key, entry["mean_nll_difference"],
                        entry["bootstrap_95_low"], entry["bootstrap_95_high"],
                    )
                )
    lines.extend([
        "",
        "## Exact lexical term by tree depth", "",
        "Nats per predicted token. The chart conditions each token on the gold",
        "tokens flanking its interval, so deep nodes enjoy a tight two-sided",
        "context. Depth 0 is the root token, emitted first from the prompt",
        "boundaries alone -- the only depth whose conditioning free generation",
        "reproduces.",
        "",
        "| Depth | Expected tokens | Share | Nats / token |",
        "|---:|---:|---:|---:|",
    ])
    for entry in depth_profile:
        lines.append("| {} | {:.1f} | {:.1%} | {:.3f} |".format(
            entry["depth"], entry["expected_tokens"],
            entry["share_of_tokens"], entry["nats_per_token"],
        ))
    lines.extend([
        "",
        "| Model | Lexical nats / token |", "|---|---:|",
    ])
    for name in REPORT_ORDER:
        lines.append("| `{}` | {:.3f} |".format(name, baseline_per_token[name]))
    with open(
        os.path.join(args.artifact_dir, "DECOMPOSITION.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
