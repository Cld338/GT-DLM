"""Oracle-structure against free-running generation for the three two-gap models.

`research/ROADMAP.md` item 10, testing the hypothesis left standing by
`research/LIKELIHOOD_DECOMPOSITION.md`. On held-out likelihood the exact model
is better at tokens and tied on structure, yet it generates worse text. Four
explanations have been rejected. The surviving one is about decoding rather
than scoring: the masked baseline draws one length and then emits every token
in a single parallel pass conditioned on the prompt alone, so nothing it
generates feeds back into anything else; the tree model expands recursively,
so every token it emits becomes the interval boundary conditioning its
children and an early error changes the context for everything below it.

That predicts an asymmetry which this script measures directly. Each model
decodes the same held-out gaps twice:

- **oracle structure**: the gold length is supplied, and for the tree model the
  balanced midpoint tree as well, so only token choices are the model's own;
- **free**: the model supplies its own structure.

Token accuracy is measured per gap at matched length, since comparing token
positions requires the prediction and the target to line up.

Two design points, both learned from running it:

Greedy free decoding is uninformative here. All three models collapse to the
empty-length mode -- the masked baseline's free length-match rate comes out at
exactly its empty-span rate -- reproducing the collapse already documented in
`research/WINDOWED_SCREENING.md`. The free arm therefore has to be sampled.

Comparing sampled arms by matched length is also biased, because the free arm
only contributes gaps whose sampled length happened to be right, and those are
shorter and easier than the full set the oracle arm covers. The final
comparison restricts both arms to exactly the gaps where the free arm produced
at least one length match, which makes the subsets identical. Every residual
bias in that comparison favours the free arm, so it can fail to detect a
compounding drop but cannot manufacture one.
"""

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_multigap_sampling import encode_gap_roots
from evaluate_text_sequence_likelihoods import paired_bootstrap
from experiment import choose_device
from experiment_text_dynamic import decode_sequential_in_chunks
from experiment_text_inside import late_depth_topology_logits
from experiment_text_pilot import decode_text_masked_model
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import TextInfillingExample, TextVocabulary

MODELS = ("factorized_depth_exact", "sequential_filler", "length_masked")
ARMS = ("oracle_structure", "free")


def _single_gap_example(span: Sequence[int]) -> TextInfillingExample:
    """Wrap one gap so the shared per-gap metrics can score it."""
    return TextInfillingExample(segments=((), ()), spans=(tuple(span),))


def _neighbours(canvas, position, root_left, root_right):
    """Nearest emitted tokens flanking an open slot, else the prompt boundaries."""
    left = next(
        (canvas[index] for index in range(position - 1, -1, -1)
         if not isinstance(canvas[index], tuple)),
        root_left,
    )
    right = next(
        (canvas[index] for index in range(position + 1, len(canvas))
         if not isinstance(canvas[index], tuple)),
        root_right,
    )
    return int(left), int(right)


@torch.inference_mode()
def decode_tree_gaps(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    batch_size: int,
    oracle_structure: bool,
    max_depth: int = 8,
    max_tokens: int = 32,
    sample: bool = False,
) -> List[List[int]]:
    """Greedily expand every root gap, returning one token list per gap.

    With ``oracle_structure`` the gold span length seeds the canvas and pivots
    are taken at the midpoint, so the tree is fixed and only tokens are chosen.
    Otherwise the model supplies the root STOP decision and every topology
    decision itself, which is what free generation does.

    Gaps are decoded independently: the factorization makes them conditionally
    independent given the shared prompt encoding, which is computed once.

    ``sample`` draws tokens, STOP and topology from their distributions instead
    of taking the argmax. Greedy decoding of a genuinely uncertain length
    collapses to the mode, which on this corruption is the empty span, so the
    free arm has to be sampled for the comparison to mean anything.
    """
    contexts, roots_left, roots_right = encode_gap_roots(
        model, examples, vocab, device, batch_size
    )
    spans = [span for example in examples for span in example.spans]
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)

    if oracle_structure:
        canvases = [
            [("gap", len(span))] if span else [] for span in spans
        ]
    else:
        # Size is unknown, so an open slot carries only its boundaries.
        canvases = [[("gap",)] for _ in spans]

    for depth in range(max_depth):
        locations = []
        for gap_index, canvas in enumerate(canvases):
            for position, item in enumerate(canvas):
                if not isinstance(item, tuple):
                    continue
                left, right = _neighbours(
                    canvas, position, int(roots_left[gap_index]),
                    int(roots_right[gap_index]),
                )
                size = item[1] if oracle_structure else None
                locations.append((gap_index, position, size, left, right))
        if not locations:
            break
        gap_ids = torch.tensor(
            [item[0] for item in locations], dtype=torch.long, device=device
        )
        left = torch.tensor(
            [item[3] for item in locations], dtype=torch.long, device=device
        )
        right = torch.tensor(
            [item[4] for item in locations], dtype=torch.long, device=device
        )
        depths = torch.full_like(left, min(depth, 31))
        token_logits, stop_logits, hidden = model.interval_logits(
            contexts[gap_ids], left, right, depths
        )
        restricted = token_logits.index_select(-1, generated_ids)
        chosen = generated_ids[
            torch.multinomial(restricted.softmax(dim=-1), 1).flatten() if sample
            else restricted.argmax(dim=-1)
        ]
        if oracle_structure:
            decisions = {
                (item[0], item[1]): (item[2], int(chosen[index]), 0, False)
                for index, item in enumerate(locations)
            }
        else:
            topology_logits = late_depth_topology_logits(
                model.topology_logits(hidden, chosen), depths, 4, 0.0
            )
            topology = (
                torch.multinomial(topology_logits.softmax(-1), 1).flatten()
                if sample else topology_logits.argmax(dim=-1)
            )
            # Only the root may stop; a child gap created by a topology bit is
            # non-empty by construction.
            root_stop = (
                torch.rand_like(stop_logits) < stop_logits.sigmoid() if sample
                else stop_logits.sigmoid() >= 0.5
            )
            stops = (
                root_stop if depth == 0
                else torch.zeros_like(stop_logits, dtype=torch.bool)
            )
            decisions = {
                (item[0], item[1]): (
                    None, int(chosen[index]), int(topology[index]),
                    bool(stops[index]),
                )
                for index, item in enumerate(locations)
            }

        for gap_index, canvas in enumerate(canvases):
            expanded = []
            emitted = sum(
                1 for item in canvas if not isinstance(item, tuple)
            )
            for position, item in enumerate(canvas):
                if not isinstance(item, tuple):
                    expanded.append(item)
                    continue
                size, token, topology_value, stop = decisions[
                    (gap_index, position)
                ]
                if stop or emitted >= max_tokens:
                    continue
                if oracle_structure:
                    pivot = size // 2
                    if pivot:
                        expanded.append(("gap", pivot))
                    expanded.append(token)
                    if pivot + 1 < size:
                        expanded.append(("gap", size - pivot - 1))
                else:
                    if topology_value & 1:
                        expanded.append(("gap",))
                    expanded.append(token)
                    if topology_value & 2:
                        expanded.append(("gap",))
                emitted += 1
            canvases[gap_index] = expanded

    return [
        [int(item) for item in canvas if not isinstance(item, tuple)]
        for canvas in canvases
    ]


def per_gap_metrics(spans, predictions):
    """Score one prediction per gap with the shared lexical metrics."""
    return lexical_sampling_metrics(
        [_single_gap_example(span) for span in spans],
        [[prediction] for prediction in predictions],
        [[False] for _ in predictions],
    )


def per_gap_correct(spans, predictions):
    """Per-gap (correct tokens, target tokens) at matched length, for pairing."""
    rows = []
    for span, prediction in zip(spans, predictions):
        if not span or len(prediction) != len(span):
            rows.append((0.0, float(len(span))))
            continue
        rows.append((
            float(sum(a == b for a, b in zip(prediction, span))),
            float(len(span)),
        ))
    return rows


def flatten_gap_predictions(examples, predictions):
    """Turn per-example, per-gap decoder output into one row per gap."""
    return [
        prediction
        for example, rows in zip(examples, predictions)
        for prediction in rows[:len(example.spans)]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", default="artifacts/text_trajectory")
    parser.add_argument(
        "--checkpoint-dir", default="artifacts/text_multigap_matched_training"
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_multigap_generation"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--token-steps", type=int, default=1)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    device = choose_device(args.device)
    from decompose_multigap_likelihood import load_matched_checkpoints

    vocab, test, models, _ = load_matched_checkpoints(args, device)
    spans = [span for example in test for span in example.spans]
    print("decoding {} two-gap examples ({} gaps)".format(len(test), len(spans)))

    predictions: Dict[Tuple[str, str], List[List[int]]] = {}

    for oracle in (True, False):
        arm = "oracle_structure" if oracle else "free"
        predictions[("factorized_depth_exact", arm)] = decode_tree_gaps(
            models["factorized_depth_exact"], test, vocab, device,
            args.batch_size, oracle_structure=oracle,
        )
        masked = decode_text_masked_model(
            models["length_masked"], test, vocab, device, args.token_steps,
            oracle_length=oracle,
        )
        predictions[("length_masked", arm)] = flatten_gap_predictions(
            test, masked[0]
        )
        sequential = decode_sequential_in_chunks(
            models["sequential_filler"], test, vocab, device, 16,
            args.stop_threshold,
            oracle_lengths=(
                [[len(span) for span in example.spans] for example in test]
                if oracle else None
            ),
        )
        predictions[("sequential_filler", arm)] = flatten_gap_predictions(
            test, sequential[0]
        )

    metrics = {
        "{}::{}".format(name, arm): per_gap_metrics(spans, rows)
        for (name, arm), rows in predictions.items()
    }

    # Sampled arms for the tree model. Greedy free decoding collapses to the
    # empty-length mode for every model here, so it cannot test compounding;
    # sampling is what the calibration study uses and what the hypothesis needs.
    torch.manual_seed(args.seed)
    sampled, sampled_draws = {}, {}
    for oracle in (True, False):
        arm = "oracle_structure" if oracle else "free"
        draws = [
            decode_tree_gaps(
                models["factorized_depth_exact"], test, vocab, device,
                args.batch_size, oracle_structure=oracle, sample=True,
            )
            for _ in range(args.samples)
        ]
        sampled_draws[arm] = [
            [draw[gap] for draw in draws] for gap in range(len(spans))
        ]
        sampled[arm] = lexical_sampling_metrics(
            [_single_gap_example(span) for span in spans],
            sampled_draws[arm],
            [[False] * len(draws) for _ in spans],
        )
    metrics.update({
        "factorized_depth_exact_sampled::{}".format(arm): row
        for arm, row in sampled.items()
    })

    # The free arm only contributes gaps whose sampled length happened to be
    # right, and those are systematically shorter and easier than the full set
    # the oracle arm covers. Restricting both arms to exactly that subset makes
    # the two comparable, which is what a compounding test requires.
    shared_gaps = [
        gap for gap, span in enumerate(spans)
        if span and any(
            len(row) == len(span) for row in sampled_draws["free"][gap]
        )
    ]
    matched_subset = {}
    for arm in ARMS:
        correct = target = 0
        for gap in shared_gaps:
            span = spans[gap]
            rows = [
                row for row in sampled_draws[arm][gap] if len(row) == len(span)
            ]
            for row in rows:
                correct += sum(a == b for a, b in zip(row, span))
                target += len(span)
        matched_subset[arm] = {
            "gaps": len(shared_gaps),
            "token_accuracy": correct / max(1, target),
        }

    # Two denominators, answering two different questions.
    #
    # "all targets" charges a gap for every target token whenever the generated
    # length is wrong, so it is the end-to-end number and folds length errors in.
    # "matched" scores only gaps whose length came out right, which isolates
    # token quality from length prediction and is the compounding signature
    # proper. The matched subset of the free arm is selected by the model
    # getting the length right, so it is easier than the oracle arm's full set;
    # that bias favours the free arm and therefore cannot manufacture a drop.
    compounding = {}
    for name in MODELS:
        oracle_rows = per_gap_correct(spans, predictions[(name, "oracle_structure")])
        free_rows = per_gap_correct(spans, predictions[(name, "free")])
        targets = sum(row[1] for row in oracle_rows)
        compounding[name] = {
            "oracle_token_accuracy": sum(r[0] for r in oracle_rows) / max(1.0, targets),
            "free_token_accuracy": sum(r[0] for r in free_rows) / max(1.0, targets),
            "oracle_matched_token_accuracy": metrics[
                "{}::oracle_structure".format(name)
            ]["matched_length_token_accuracy"],
            "free_matched_token_accuracy": metrics[
                "{}::free".format(name)
            ]["matched_length_token_accuracy"],
            "free_length_match": metrics[
                "{}::free".format(name)
            ]["length_match_probability"],
            "paired_drop": paired_bootstrap(
                torch.tensor([r[0] for r in oracle_rows]),
                torch.tensor([r[0] for r in free_rows]),
            ),
        }

    result = {
        "config": vars(args),
        "gaps": len(spans),
        "metrics": metrics,
        "compounding": compounding,
        "sampled_matched_subset": matched_subset,
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "generation.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# Oracle structure against free generation", "",
        "Greedy decoding of {} held-out gaps. Token accuracy is measured at".format(
            len(spans)),
        "matched length. The `drop` column is the compounding signature: how",
        "much accuracy a model loses once it must supply its own structure.",
        "",
        "Two denominators. **Matched** scores only gaps whose generated length",
        "came out right, isolating token quality from length prediction, and is",
        "the compounding signature proper. **All targets** charges a gap for",
        "every target token when the length is wrong, so it is the end-to-end",
        "number. The free arm's matched subset is selected by the model getting",
        "the length right, so it is an easier set than the oracle arm's; that",
        "bias favours the free arm and cannot manufacture a drop.",
        "",
        "| Model | Matched oracle | Matched free | Drop | All-targets oracle | All-targets free |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        row = compounding[name]
        lines.append(
            "| `{}` | {:.1%} | {:.1%} | {:+.1f} pts | {:.1%} | {:.1%} |".format(
                name, row["oracle_matched_token_accuracy"],
                row["free_matched_token_accuracy"],
                100 * (row["free_matched_token_accuracy"]
                       - row["oracle_matched_token_accuracy"]),
                row["oracle_token_accuracy"], row["free_token_accuracy"],
            )
        )
    lines.extend([
        "",
        "## Sampled tree arms", "",
        "{} samples per gap, temperature 1. Greedy free decoding collapses to".format(args.samples),
        "the empty-length mode for every model above, so the compounding",
        "comparison has to be made under sampling.",
        "",
        "| Arm | Length match | Matched token accuracy | Matched edit similarity |",
        "|---|---:|---:|---:|",
    ])
    for arm in ARMS:
        row = metrics["factorized_depth_exact_sampled::{}".format(arm)]
        lines.append("| {} | {:.1%} | {:.1%} | {:.3f} |".format(
            arm, row["length_match_probability"],
            row["matched_length_token_accuracy"],
            row["matched_length_edit_similarity"],
        ))
    lines.extend([
        "",
        "Both arms restricted to the same {} gaps -- those where the free arm".format(
            matched_subset["free"]["gaps"]),
        "produced at least one length match -- so the subsets are comparable:",
        "",
        "| Arm | Token accuracy on shared gaps |",
        "|---|---:|",
        "| oracle_structure | {:.1%} |".format(
            matched_subset["oracle_structure"]["token_accuracy"]),
        "| free | {:.1%} |".format(matched_subset["free"]["token_accuracy"]),
        "",
        "| Model | Arm | Length match | Matched edit similarity | Exact |",
        "|---|---|---:|---:|---:|",
    ])
    for name in MODELS:
        for arm in ARMS:
            row = metrics["{}::{}".format(name, arm)]
            lines.append("| `{}` | {} | {:.1%} | {:.3f} | {:.1%} |".format(
                name, arm, row["length_match_probability"],
                row["matched_length_edit_similarity"],
                row["matched_length_exact_probability"],
            ))
    with open(
        os.path.join(args.artifact_dir, "GENERATION.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
