"""Size the headroom in the scaffold's single-pass parallel fill.

`fill_sampled_scaffolds` completes a scaffold with one masked-LM pass, so every
span position is predicted while every other span position is still a mask.
`diagnose_emission_context.py` measured what that costs on the semantic
branching checkpoint; this measures it on the checkpoint the scaffold actually
fills with, and then simulates the fix.

Three families are reported, all at gold length so the fill is isolated from
the length model:

    staircase     gold tokens revealed at `k` of the other span positions. k=0
                  is the current one-pass condition and k=n-1 is the upper bound
                  for any refill. This is optimistic: it reveals gold, not
                  predictions.
    commit-only   a confidence-ordered fill in `p` passes with no retraining.
                  Each pass commits the most confident still-masked positions
                  and re-encodes, and a committed position is never revisited.
                  p=1 is exactly the current decoder.
    mask-predict  the schedule from Ghazvininejad et al. (2019). Each round
                  predicts every masked position, then re-masks the least
                  confident ones, so a round-one mistake can be revised later.
                  This is the difference the literature's method turns on, and
                  the commit-only family does not speak to it.

RoBERTa was pretrained at 15% masking, so a fully masked span is far from its
pretraining distribution and a partly filled one is much closer. That is the
reason to expect the iterative families to gain without any retraining.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from experiment import choose_device, edit_distance, seed_everything
from gtdlm.model import PretrainedLengthMaskedModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def native_canvas(example, vocab, span_tokens):
    """Prompt with the span rendered as explicit native tokens or masks."""
    return (
        [vocab.LEFT]
        + list(example.segments[0])
        + list(span_tokens)
        + list(example.segments[-1])
        + [vocab.RIGHT]
    )


def decoded_pair_metrics(tokenizer, prediction, target):
    """Return tokenizer-independent character similarity and exactness."""
    predicted_text = tokenizer.decode(
        prediction,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    target_text = tokenizer.decode(
        target,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    similarity = 1.0 - edit_distance(predicted_text, target_text) / max(
        1, len(predicted_text), len(target_text)
    )
    return similarity, int(predicted_text == target_text)


def score_canvases(backbone, token_head, rows, vocab, device, generated_ids,
                   batch_size=16):
    """Return per-row log-probabilities over the generated vocabulary."""
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        width = max(len(canvas) for canvas, _ in batch)
        tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        attention = torch.zeros(
            (len(batch), width), dtype=torch.long, device=device
        )
        for row, (canvas, _) in enumerate(batch):
            tokens[row, : len(canvas)] = torch.tensor(canvas, device=device)
            attention[row, : len(canvas)] = 1
        with torch.no_grad():
            hidden = backbone(
                input_ids=tokens, attention_mask=attention
            ).last_hidden_state
            logits = token_head(hidden)
        allowed = logits.index_select(-1, generated_ids).log_softmax(dim=-1)
        for row, (_, positions) in enumerate(batch):
            outputs.append(allowed[row, positions].cpu())
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lexical-artifact-dir",
        default="artifacts/text_pretrained_masked_roberta_base",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_scaffold_iterative_fill"
    )
    parser.add_argument(
        "--data-config-dir",
        default="artifacts/text_semantic_branching_roberta_base_zero_interaction",
        help=(
            "artifact whose config supplies the corpus split parameters; the "
            "lexical checkpoint's own config does not record them, and sharing "
            "this source keeps the test prompts identical to "
            "diagnose_emission_context.py"
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        default="",
        help=(
            "evaluate on this corpus instead of the lexical checkpoint's own "
            "training corpus; required to compare checkpoints trained on "
            "different corpora on identical prompts"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument(
        "--passes", default="1,2,3,8",
        help="comma-separated pass budgets for the iterative family",
    )
    args = parser.parse_args()

    with open(
        os.path.join(args.lexical_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    with open(
        os.path.join(args.data_config_dir, "results.json"), encoding="utf-8"
    ) as handle:
        data_config = json.load(handle)["config"]

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    data_dir = args.corpus_dir or str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(data_config["data_seed"])
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"],
            data_seed + 403,
            int(data_config["random_window_min"]),
            int(data_config["random_window_max"]),
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=int(data_config["max_span"]),
    )[: args.examples]

    model = PretrainedLengthMaskedModel(
        vocab.vocab_size,
        int(config["max_span"]),
        vocab.GAP,
        vocab.PAD,
        tokenizer,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        max_length=int(config["max_length"]),
        local_files_only=True,
        native_vocabulary=True,
        attn_implementation=str(config.get("attention_implementation", "eager")),
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.lexical_artifact_dir, "masked.pt"),
        map_location=device,
        weights_only=True,
    ))
    model.eval()
    backbone = model.encoder.backbone
    token_head = model.token_head
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    index_of = {int(value): index for index, value in enumerate(
        vocab.generated_token_ids
    )}

    spans = [
        (example, list(example.spans[0]))
        for example in test
        if example.spans[0]
    ]

    # --- staircase: reveal k gold neighbours ------------------------------
    staircase = defaultdict(list)
    for example, span in spans:
        n = len(span)
        prefix = 1 + len(example.segments[0])
        for k in range(n):
            # Reveal the first k positions other than the target, in order.
            for target in range(n):
                others = [i for i in range(n) if i != target]
                revealed = set(others[:k])
                canvas_span = [
                    span[i] if i in revealed else vocab.GAP for i in range(n)
                ]
                canvas_span[target] = vocab.GAP
                staircase[k].append((
                    native_canvas(example, vocab, canvas_span),
                    prefix + target,
                    int(span[target]),
                ))

    staircase_results = {}
    for k in sorted(staircase):
        rows = [(canvas, [position]) for canvas, position, _ in staircase[k]]
        golds = [gold for _, _, gold in staircase[k]]
        scored = score_canvases(
            backbone, token_head, rows, vocab, device, generated_ids,
            args.batch_size,
        )
        nll, correct = 0.0, 0
        for logp, gold in zip(scored, golds):
            row = logp[0]
            nll -= float(row[index_of[gold]])
            correct += int(int(row.argmax()) == index_of[gold])
        staircase_results[str(k)] = {
            "revealed_neighbours": k,
            "positions": len(golds),
            "token_nll": nll / max(1, len(golds)),
            "top1_accuracy": correct / max(1, len(golds)),
        }

    # --- mask-predict: re-maskable fill, no retraining --------------------
    # The commit-only family below freezes a position once it is filled, so a
    # round-one mistake conditions every later round. Mask-Predict instead
    # recomputes the masked set from confidence each round, so any token can be
    # revised. That is the difference the literature's schedule turns on, and
    # the commit-only result does not speak to it.
    maskpredict_results = {}
    for budget in [int(value) for value in args.passes.split(",") if value]:
        correct, total, exact = 0, 0, 0
        decoded_similarity, decoded_exact = 0.0, 0
        for example, span in spans:
            n = len(span)
            prefix = 1 + len(example.segments[0])
            canvas_span = [vocab.GAP] * n
            confidence = [float("-inf")] * n
            for step in range(budget):
                masked = [i for i in range(n) if canvas_span[i] == vocab.GAP]
                if not masked:
                    break
                rows = [(
                    native_canvas(example, vocab, canvas_span),
                    [prefix + i for i in masked],
                )]
                logp = score_canvases(
                    backbone, token_head, rows, vocab, device, generated_ids,
                    args.batch_size,
                )[0]
                for j, position in enumerate(masked):
                    choice = int(logp[j].argmax())
                    canvas_span[position] = int(
                        vocab.generated_token_ids[choice]
                    )
                    confidence[position] = float(logp[j].max())
                # Re-mask the least confident positions for the next round.
                remaining_rounds = budget - step - 1
                if remaining_rounds <= 0:
                    break
                keep = int(round(n * (step + 1) / budget))
                order = sorted(range(n), key=lambda i: confidence[i])
                for position in order[: max(0, n - keep)]:
                    canvas_span[position] = vocab.GAP
                    confidence[position] = float("-inf")
            hits = sum(
                1 for i in range(n) if int(canvas_span[i]) == int(span[i])
            )
            correct += hits
            total += n
            exact += int(hits == n)
            similarity, is_exact = decoded_pair_metrics(
                tokenizer, canvas_span, span
            )
            decoded_similarity += similarity
            decoded_exact += is_exact
        maskpredict_results[str(budget)] = {
            "passes": budget,
            "positions": total,
            "top1_accuracy": correct / max(1, total),
            "exact_span_probability": exact / max(1, len(spans)),
            "character_edit_similarity": (
                decoded_similarity / max(1, len(spans))
            ),
            "decoded_exact_span_probability": (
                decoded_exact / max(1, len(spans))
            ),
        }

    # --- iterative: commit-only confidence-ordered fill --------------------
    iterative_results = {}
    for budget in [int(value) for value in args.passes.split(",") if value]:
        correct, total = 0, 0
        exact = 0
        decoded_similarity, decoded_exact = 0.0, 0
        for example, span in spans:
            n = len(span)
            prefix = 1 + len(example.segments[0])
            canvas_span = [vocab.GAP] * n
            remaining = set(range(n))
            passes = min(budget, n)
            for step in range(passes):
                rows = [(
                    native_canvas(example, vocab, canvas_span),
                    [prefix + i for i in sorted(remaining)],
                )]
                logp = score_canvases(
                    backbone, token_head, rows, vocab, device, generated_ids,
                    args.batch_size,
                )[0]
                order = sorted(remaining)
                confidence = [(float(logp[j].max()), order[j])
                              for j in range(len(order))]
                confidence.sort(reverse=True)
                left = passes - step
                commit = max(1, math.ceil(len(remaining) / left))
                for rank in range(min(commit, len(confidence))):
                    _, position = confidence[rank]
                    j = order.index(position)
                    choice = int(logp[j].argmax())
                    canvas_span[position] = int(
                        vocab.generated_token_ids[choice]
                    )
                    remaining.discard(position)
                if not remaining:
                    break
            hits = sum(
                1 for i in range(n) if int(canvas_span[i]) == int(span[i])
            )
            correct += hits
            total += n
            exact += int(hits == n)
            similarity, is_exact = decoded_pair_metrics(
                tokenizer, canvas_span, span
            )
            decoded_similarity += similarity
            decoded_exact += is_exact
        iterative_results[str(budget)] = {
            "passes": budget,
            "positions": total,
            "top1_accuracy": correct / max(1, total),
            "exact_span_probability": exact / max(1, len(spans)),
            "character_edit_similarity": (
                decoded_similarity / max(1, len(spans))
            ),
            "decoded_exact_span_probability": (
                decoded_exact / max(1, len(spans))
            ),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "config": vars(args),
        "spans": len(spans),
        "staircase": staircase_results,
        "iterative_commit_only": iterative_results,
        "maskpredict": maskpredict_results,
    }
    with open(
        os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)

    print("staircase (gold neighbours revealed, upper bound)")
    print("  k   positions   token NLL   top-1")
    for key in sorted(staircase_results, key=int):
        value = staircase_results[key]
        print("  %-3s %9d %11.4f %7.2f%%" % (
            key, value["positions"], value["token_nll"],
            100.0 * value["top1_accuracy"],
        ))
    print()
    for label, table in (
        ("commit-only confidence-ordered fill", iterative_results),
        ("mask-predict, re-maskable", maskpredict_results),
    ):
        print("%s (actual, no retraining)" % label)
        print("  passes   positions   top-1     exact span  decoded exact  char edit")
        for key in sorted(table, key=int):
            value = table[key]
            print("  %-8s %9d %8.2f%% %11.2f%% %13.2f%% %10.4f" % (
                key, value["positions"],
                100.0 * value["top1_accuracy"],
                100.0 * value["exact_span_probability"],
                100.0 * value["decoded_exact_span_probability"],
                value["character_edit_similarity"],
            ))
        print()


if __name__ == "__main__":
    main()
