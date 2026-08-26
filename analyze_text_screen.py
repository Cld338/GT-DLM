"""Length-distribution and non-empty-span audit for the natural-text screen."""

import argparse
import collections
import json
import os
from typing import Dict, List, Sequence

import torch
from tokenizers import Tokenizer

from experiment import choose_device, edit_distance, seed_everything
from experiment_text_pilot import (
    decode_text_gap_model,
    decode_text_masked_model,
)
from experiment_text_factorized import decode_factorized_in_chunks
from gtdlm.model import (
    GapTreeConditionalBoundaryModel,
    GapTreeFactorizedBoundaryModel,
    LengthMaskedModel,
)
from gtdlm.text_data import TextInfillingExample, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def audit_lengths(
    examples: Sequence[TextInfillingExample], predictions: Sequence[List[List[int]]]
) -> Dict[str, object]:
    target_lengths: List[int] = []
    predicted_lengths: List[int] = []
    nonempty_length_correct: List[bool] = []
    nonempty_similarities: List[float] = []
    empty_correct: List[bool] = []
    for example, prediction in zip(examples, predictions):
        for target, predicted in zip(example.spans, prediction):
            target_lengths.append(len(target))
            predicted_lengths.append(len(predicted))
            if target:
                nonempty_length_correct.append(len(target) == len(predicted))
                nonempty_similarities.append(
                    1.0
                    - edit_distance(predicted, target)
                    / max(1, len(predicted), len(target))
                )
            else:
                empty_correct.append(len(predicted) == 0)
    under = sum(predicted < target for predicted, target in zip(predicted_lengths, target_lengths))
    over = sum(predicted > target for predicted, target in zip(predicted_lengths, target_lengths))
    count = max(1, len(target_lengths))
    return {
        "gaps": len(target_lengths),
        "target_mean_length": sum(target_lengths) / count,
        "predicted_mean_length": sum(predicted_lengths) / count,
        "target_empty_rate": sum(length == 0 for length in target_lengths) / count,
        "predicted_empty_rate": sum(length == 0 for length in predicted_lengths) / count,
        "empty_target_accuracy": (
            sum(empty_correct) / len(empty_correct) if empty_correct else 0.0
        ),
        "nonempty_length_accuracy": (
            sum(nonempty_length_correct) / len(nonempty_length_correct)
            if nonempty_length_correct
            else 0.0
        ),
        "nonempty_edit_similarity": (
            sum(nonempty_similarities) / len(nonempty_similarities)
            if nonempty_similarities
            else 0.0
        ),
        "under_generation_rate": under / count,
        "over_generation_rate": over / count,
        "target_length_histogram": dict(sorted(collections.Counter(target_lengths).items())),
        "predicted_length_histogram": dict(sorted(collections.Counter(predicted_lengths).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/wikitext_pilot")
    parser.add_argument("--artifact-dir", default="artifacts/text_screen")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        results = json.load(handle)
    config = results["config"]
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(args.data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    evaluation = {
        "iid_one_gap": sample_text_infilling_examples(
            corpus["test"], seed + 101, gap_counts=(1,), min_span=1, max_span=8
        ),
        "composition_two_gap": sample_text_infilling_examples(
            corpus["test"], seed + 103, gap_counts=(2,), min_span=1, max_span=8
        ),
        "length_ood_one_gap": sample_text_infilling_examples(
            corpus["test"],
            seed + 107,
            gap_counts=(1,),
            min_span=9,
            max_span=16,
            zero_length_probability=0.0,
        ),
    }
    gap_model = GapTreeConditionalBoundaryModel(
        vocab.vocab_size,
        vocab.action_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
    ).to(device)
    baseline = LengthMaskedModel(
        vocab.vocab_size,
        16,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
    ).to(device)
    gap_model.load_state_dict(
        torch.load(os.path.join(args.artifact_dir, "gap_tree.pt"), map_location=device, weights_only=True)
    )
    baseline.load_state_dict(
        torch.load(os.path.join(args.artifact_dir, "length_masked.pt"), map_location=device, weights_only=True)
    )
    factorized_path = os.path.join(
        os.path.dirname(args.artifact_dir), "text_factorized", "gap_tree_factorized.pt"
    )
    factorized = None
    if os.path.exists(factorized_path):
        factorized = GapTreeFactorizedBoundaryModel(
            vocab.vocab_size,
            gap_id=vocab.GAP,
            pad_id=vocab.PAD,
            d_model=int(config["d_model"]),
            nhead=int(config["heads"]),
            layers=int(config["layers"]),
            max_positions=256,
        ).to(device)
        factorized.load_state_dict(
            torch.load(factorized_path, map_location=device, weights_only=True)
        )
    audit: Dict[str, Dict[str, object]] = {}
    qualitative: List[Dict[str, str]] = []
    for slice_name, examples in evaluation.items():
        gap_output = decode_text_gap_model(gap_model, examples, vocab, device, 16)
        baseline_output = decode_text_masked_model(
            baseline, examples, vocab, device, token_steps=2, oracle_length=False
        )
        audit[slice_name] = {
            "gap_tree": audit_lengths(examples, gap_output[0]),
            "learned_length_masked": audit_lengths(examples, baseline_output[0]),
        }
        if factorized is not None:
            factorized_output = decode_factorized_in_chunks(
                factorized, examples, vocab, device, 16, 0.5
            )
            audit[slice_name]["factorized_gap_tree"] = audit_lengths(
                examples, factorized_output[0]
            )
        if slice_name == "iid_one_gap":
            for example, gap_prediction, baseline_prediction in zip(
                examples, gap_output[0], baseline_output[0]
            ):
                if not example.spans[0]:
                    continue
                qualitative.append(
                    {
                        "prompt": tokenizer.decode(example.prompt(vocab), skip_special_tokens=False),
                        "target": tokenizer.decode(list(example.spans[0])),
                        "gap_tree": tokenizer.decode(gap_prediction[0]),
                        "length_masked": tokenizer.decode(baseline_prediction[0]),
                    }
                )
                if len(qualitative) == 8:
                    break
    output = {"audit": audit, "qualitative_iid": qualitative}
    with open(
        os.path.join(args.artifact_dir, "analysis.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    lines = [
        "# Natural-text screening audit",
        "",
        "| Slice | Model | Target mean length | Predicted mean length | Empty prediction | Non-empty length | Non-empty edit | Under | Over |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slice_name, models in audit.items():
        for model_name, value in models.items():
            lines.append(
                "| {} | {} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                    slice_name,
                    model_name,
                    value["target_mean_length"],
                    value["predicted_mean_length"],
                    value["predicted_empty_rate"],
                    value["nonempty_length_accuracy"],
                    value["nonempty_edit_similarity"],
                    value["under_generation_rate"],
                    value["over_generation_rate"],
                )
            )
    with open(
        os.path.join(args.artifact_dir, "ANALYSIS.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
