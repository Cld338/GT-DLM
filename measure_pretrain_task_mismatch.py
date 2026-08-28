"""How much of the pretrained model's ability survives our task formulation.

`research/LIKELIHOOD_DECOMPOSITION.md` attributes most of the generation
deficit to how the pretrained encoder is attached. That attribution lumps two
things together, because pooling the prompt to one vector is at once a loss of
*information* and a move out of the space the backbone was trained to produce.

This script measures a third thing that both arms share and that neither
attribution covers: the project keeps only `AutoModel`, so `distilroberta`'s
masked-language-model head is discarded, and predictions are made over a 4,000
token custom BPE vocabulary instead of RoBERTa's 50,265. The single most
valuable part of the pretrained model for filling a blank -- the output side --
is thrown away and relearned from averaged input embeddings.

The reference point is what the pretrained model does on this task with **no
finetuning at all**: its own tokenizer, its own MLM head, the same held-out
spans. Comparing that against our trained models says how much the formulation
costs relative to using the pretrained model as it comes.

Everything is scored on decoded text rather than token ids, since the arms do
not share a vocabulary. Exact match and character-level similarity are
vocabulary-neutral; token accuracy is not, and is reported per arm only for
continuity with the rest of the project.
"""

import argparse
import json
import os
from typing import List, Sequence

import torch
from tokenizers import Tokenizer

from experiment import choose_device, edit_distance
from gtdlm.text_data import (
    TextInfillingExample,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def span_texts(examples: Sequence[TextInfillingExample], tokenizer):
    """Left context, gold span and right context as strings, per example."""
    rows = []
    for example in examples:
        rows.append((
            tokenizer.decode(list(example.segments[0]), skip_special_tokens=False),
            tokenizer.decode(list(example.spans[0]), skip_special_tokens=False),
            tokenizer.decode(list(example.segments[1]), skip_special_tokens=False),
        ))
    return rows


def character_scores(predictions: Sequence[str], targets: Sequence[str]):
    """Vocabulary-neutral scoring: exact match and character edit similarity."""
    exact, similarity, scored = 0, 0.0, 0
    for prediction, target in zip(predictions, targets):
        if not target:
            continue
        scored += 1
        exact += int(prediction == target)
        similarity += 1.0 - edit_distance(list(prediction), list(target)) / max(
            1, len(prediction), len(target)
        )
    return {
        "spans": scored,
        "exact_match": exact / max(1, scored),
        "character_similarity": similarity / max(1, scored),
    }


@torch.inference_mode()
def zero_shot_mlm(rows, model_name, cache_dir, device, local_files_only, batch_size):
    """Fill each span with the untouched pretrained MLM, in its own vocabulary.

    The number of masks is the gold span's length *in RoBERTa tokens*, which is
    the same oracle-length information our own arms receive.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, use_fast=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=local_files_only,
    ).to(device).eval()

    predictions: List[str] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts, widths = [], []
        for left, gold, right in batch:
            count = max(1, len(tokenizer(gold, add_special_tokens=False)["input_ids"]))
            widths.append(count)
            texts.append(left + tokenizer.mask_token * count + right)
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**inputs).logits
        chosen = logits.argmax(dim=-1)
        matches = inputs["input_ids"].eq(int(tokenizer.mask_token_id))
        for row, width in enumerate(widths):
            positions = matches[row].nonzero().flatten()[:width]
            predictions.append(
                tokenizer.decode(chosen[row, positions], skip_special_tokens=True)
                if positions.numel() else ""
            )
    return predictions


@torch.inference_mode()
def our_masked_baseline(examples, checkpoint, config, vocab, source_tokenizer,
                        device, args):
    """Decode our pretrained masked baseline at oracle length."""
    from experiment_pretrained_masked_baseline import decode_oracle_length
    from gtdlm.model import PretrainedLengthMaskedModel

    model = PretrainedLengthMaskedModel(
        vocab.vocab_size, args.max_span, vocab.GAP, vocab.PAD, source_tokenizer,
        model_name=args.model_name, cache_dir=args.cache_dir,
        max_length=256, local_files_only=args.local_files_only,
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    ids = decode_oracle_length(
        model, examples, vocab, device, args.batch_size
    )
    return [source_tokenizer.decode(row, skip_special_tokens=False) for row in ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument(
        "--masked-checkpoint",
        default="artifacts/text_pretrained_masked_baseline/masked.pt",
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_pretrain_task_mismatch"
    )
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--cache-dir", default=".hf_cache/hub")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    data_seed = int(config["seed"])
    source_tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(source_tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403,
            int(config["random_window_min"]), int(config["random_window_max"]),
        ),
        data_seed + 101, gap_counts=(1,), min_span=1, max_span=args.max_span,
    )[:args.examples]
    rows = span_texts(test, source_tokenizer)
    targets = [gold for _, gold, _ in rows]

    results = {}
    print("scoring {} held-out spans".format(len(test)))
    results["pretrained_zero_shot_mlm"] = character_scores(
        zero_shot_mlm(
            rows, args.model_name, args.cache_dir, device,
            args.local_files_only, args.batch_size,
        ),
        targets,
    )
    if os.path.exists(args.masked_checkpoint):
        results["our_masked_baseline"] = character_scores(
            our_masked_baseline(
                test, args.masked_checkpoint, config, vocab, source_tokenizer,
                device, args,
            ),
            targets,
        )

    lines = [
        "# What our formulation costs against the untouched pretrained model", "",
        "Same {} held-out spans, oracle length supplied to every arm, greedy".format(
            len(test)),
        "decoding. Scored on decoded text, since the arms do not share a",
        "vocabulary: the zero-shot arm predicts RoBERTa's 50,265 tokens with its",
        "own MLM head, ours predicts 4,000 custom BPE tokens with a head learned",
        "from averaged input embeddings.",
        "",
        "| Arm | Exact match | Character similarity | Spans |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "pretrained_zero_shot_mlm": "`distilroberta` MLM, **no finetuning**",
        "our_masked_baseline": "Our pretrained masked baseline (finetuned)",
    }
    for name, row in results.items():
        lines.append("| {} | {:.1%} | {:.3f} | {} |".format(
            labels.get(name, name), row["exact_match"],
            row["character_similarity"], row["spans"],
        ))
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "mismatch.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({"config": vars(args), "results": results}, handle, indent=2)
    with open(
        os.path.join(args.artifact_dir, "MISMATCH.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
