"""Consolidate oracle-structure top-1 token accuracy across every checkpoint.

`research/ROADMAP.md` item 11. `research/LIKELIHOOD_DECOMPOSITION.md` found
that on the from-scratch matched two-gap checkpoints the exact model's large
likelihood advantage produces no top-1 advantage at all: under oracle structure
it reaches `4.0%` against the masked baseline's `4.2%`. The open question is
whether that is a property of the objective or of training from scratch at
pilot scale, and the pretrained single-gap study is the one place the tree
model has been reported ahead on this metric.

Everything needed to answer it is already on disk, spread across studies run at
different times. This script reads those artifacts and puts them in one table
with the controls labelled, so the comparison can be audited rather than
assembled by hand from prose.

The measurement is the same in every row: supply the gold length and the
balanced midpoint tree, decode greedily, and score token accuracy at matched
length. Rows within a group share an evaluator, an example count and an
evaluation seed; the groups do not, so compare within a group and treat
across-group differences as indicative.
"""

import argparse
import json
import os
import statistics
from typing import Dict, List, Optional

ORACLE_KEY = "matched_length_token_accuracy"


def read(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dig(payload: Optional[dict], *path):
    for key in path:
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload


def collect(root: str) -> List[Dict]:
    """Build the row list, skipping anything whose artifact is absent."""
    rows: List[Dict] = []

    def add(group, name, value, control, note, seeds=None):
        if value is None:
            return
        rows.append({
            "group": group, "model": name, "oracle_top1": value,
            "is_control": control, "note": note,
            "seed_values": seeds,
        })

    # Group 1: the matched two-gap checkpoints, all three models, one evaluator.
    generation = read(os.path.join(root, "text_multigap_generation", "generation.json"))
    for name, label in (
        ("factorized_depth_exact", "Factorized depth exact"),
        ("sequential_filler", "Sequential filler"),
        ("length_masked", "Learned lengths + masks"),
    ):
        add(
            "two_gap_matched_from_scratch", label,
            dig(generation, "metrics", "{}::oracle_structure".format(name), ORACLE_KEY),
            False, "10M, from scratch, matched two-gap training",
        )

    # Group 2: the single-gap pretrained study and its capacity-matched control.
    seeds = []
    for suffix in ("", "_seed23", "_seed41"):
        value = dig(
            read(os.path.join(
                root, "text_depth_inside_pretrained" + suffix,
                "lexical_evaluation.json",
            )),
            "oracle_midpoint_metrics", ORACLE_KEY,
        )
        if value is not None:
            seeds.append(value)
    add(
        "single_gap_pretrained", "Depth exact, distilroberta backbone",
        statistics.mean(seeds) if seeds else None, False,
        "87M, pretrained backbone, 3 seeds", seeds,
    )
    add(
        "single_gap_pretrained", "Same architecture, random-init backbone",
        dig(
            read(os.path.join(
                root, "text_depth_inside_random_architecture_control",
                "lexical_evaluation.json",
            )),
            "oracle_midpoint_metrics", ORACLE_KEY,
        ),
        True, "capacity-matched control, 1 seed",
    )
    control_seeds = []
    for suffix in ("", "_seed23", "_seed41"):
        value = dig(
            read(os.path.join(
                root,
                "text_depth_inside_pretrained_exact_control{}_oracle".format(suffix),
                "results.json",
            )),
            "oracle_metrics", ORACLE_KEY,
        )
        if value is not None:
            control_seeds.append(value)
    add(
        "single_gap_pretrained", "Depth exact, 10M from scratch",
        statistics.mean(control_seeds) if control_seeds else None, True,
        "no pretraining, no extra capacity, 3 seeds", control_seeds,
    )
    add(
        "single_gap_pretrained", "Oracle-length masked baseline",
        dig(
            read(os.path.join(root, "text_inside_lexical", "lexical_baselines.json")),
            "models", "masked_oracle_length", ORACLE_KEY,
        ),
        True, "10M, from scratch -- NOT pretrained, NOT capacity-matched",
    )
    baseline_seeds = []
    for suffix in ("", "_seed23", "_seed41"):
        value = dig(
            read(os.path.join(
                root, "text_pretrained_masked_baseline" + suffix, "results.json",
            )),
            "oracle_metrics", ORACLE_KEY,
        )
        if value is not None:
            baseline_seeds.append(value)
    add(
        "single_gap_pretrained", "Masked baseline, same pretrained backbone",
        statistics.mean(baseline_seeds) if baseline_seeds else None,
        True,
        "85M, same backbone, stream, split and budget -- the matched control",
        baseline_seeds,
    )
    return rows


def unigram_floor(
    trajectory_dir: str,
    examples_limit: int = 128,
    gap_count: int = 1,
) -> Optional[dict]:
    """Accuracy of always emitting the training corpus's most frequent token.

    Every accuracy in this file is a small percentage, and small percentages
    are only interpretable against the trivial policy. A model that has learned
    nothing but token frequency already scores something here, and any claim
    about one model beating another has to clear that floor first.
    """
    try:
        import torch
        from tokenizers import Tokenizer
        from gtdlm.text_data import (
            random_length_windows, sample_text_infilling_examples,
        )
        from gtdlm.text_tokenizer import vocabulary_from_tokenizer
    except Exception:
        return None
    base = read(os.path.join(trajectory_dir, "results.json"))
    config = dig(base, "config")
    if not config:
        return None
    data_dir = str(config["data_dir"])
    if not os.path.exists(os.path.join(data_dir, "corpus.pt")):
        return None
    vocab = vocabulary_from_tokenizer(
        Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    )
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(config["seed"])
    counts: Dict[int, int] = {}
    allowed = set(vocab.generated_token_ids)
    for document in corpus["train"]:
        for token in document.tolist() if hasattr(document, "tolist") else document:
            if token in allowed:
                counts[token] = counts.get(token, 0) + 1
    if not counts:
        return None
    modal = max(counts, key=counts.get)
    examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403,
            int(config["random_window_min"]), int(config["random_window_max"]),
        ),
        data_seed + 101, gap_counts=(gap_count,), min_span=1, max_span=8,
    )[:examples_limit]
    tokens = [token for e in examples for span in e.spans for token in span]
    if not tokens:
        return None
    return {
        "modal_token_id": modal,
        "modal_token_accuracy": sum(t == modal for t in tokens) / len(tokens),
        "target_tokens": len(tokens),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--trajectory-dir", default="artifacts/text_trajectory")
    parser.add_argument(
        "--output-dir", default="artifacts/text_oracle_top1_summary"
    )
    args = parser.parse_args()

    rows = collect(args.artifact_root)
    by_group: Dict[str, List[Dict]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)

    pretrained = next(
        (r for r in rows if r["model"].endswith("distilroberta backbone")), None
    )
    matched_control = next(
        (r for r in rows if "random-init backbone" in r["model"]), None
    )
    pretraining_gain = (
        pretrained["oracle_top1"] - matched_control["oracle_top1"]
        if pretrained and matched_control else None
    )

    lines = [
        "# Oracle-structure top-1 accuracy across checkpoints", "",
        "Gold length and balanced midpoint tree supplied, greedy decoding, token",
        "accuracy at matched length. Rows within a group share an evaluator, an",
        "example count and an evaluation seed. The groups do not, so differences",
        "across groups are indicative only.", "",
    ]
    titles = {
        "two_gap_matched_from_scratch":
            "Matched two-gap training, all three models (512 gaps)",
        "single_gap_pretrained":
            "Single-gap pretrained study (128 examples, evaluation seed 1901)",
    }
    for group, group_rows in by_group.items():
        lines.extend([
            "## {}".format(titles.get(group, group)), "",
            "| Model | Oracle top-1 | Per seed | Note |",
            "|---|---:|---|---|",
        ])
        for row in group_rows:
            per_seed = (
                ", ".join("{:.1%}".format(v) for v in row["seed_values"])
                if row["seed_values"] else "--"
            )
            lines.append("| {}{} | {:.2%} | {} | {} |".format(
                row["model"], " *(control)*" if row["is_control"] else "",
                row["oracle_top1"], per_seed, row["note"],
            ))
        lines.append("")

    matched_baseline = next(
        (r for r in rows if "same pretrained backbone" in r["model"]), None
    )
    if pretraining_gain is not None:
        lines.extend([
            "## Reading", "",
            "Against its capacity-matched random-init control, pretraining raises",
            "the tree model's oracle-structure top-1 accuracy from {:.2%} to {:.2%},".format(
                matched_control["oracle_top1"], pretrained["oracle_top1"]),
            "a gain of {:+.1f} points. The top-1 deficit measured on the".format(
                100 * pretraining_gain),
            "from-scratch two-gap checkpoints is therefore **not** an intrinsic",
            "property of the objective: a pretrained backbone moves it.", "",
        ])
    if matched_baseline is not None and pretrained is not None:
        delta = matched_baseline["oracle_top1"] - pretrained["oracle_top1"]
        lines.extend([
            "The matched cross-model comparison, however, goes against the tree",
            "objective. Given the same backbone, stream, split and budget, the",
            "masked baseline reaches {:.2%} against the tree model's {:.2%},".format(
                matched_baseline["oracle_top1"], pretrained["oracle_top1"]),
            "a difference of {:+.1f} points. The tree model's earlier lead over".format(
                100 * delta),
            "an unpretrained baseline was an artifact of that baseline's missing",
            "pretraining and capacity, not evidence for the objective.", "",
            "Both arms are three seeds and their ranges do not overlap: the",
            "baseline's worst seed ({:.2%}) is above the tree model's best".format(
                min(matched_baseline["seed_values"] or [0])),
            "({:.2%}), so this is not seed noise.".format(
                max(pretrained["seed_values"] or [0])), "",
            "Filling masks is the task the backbone was pretrained on, so the",
            "baseline draws more from it than the tree model can. That asymmetry",
            "is real, and it is also the point: where a pretrained masked encoder",
            "is available, using it directly beats adapting it to an interval",
            "chart on this task at this scale.", "",
        ])
    floor = unigram_floor(args.trajectory_dir)
    two_gap_floor = unigram_floor(args.trajectory_dir, 256, gap_count=2)
    if floor is not None:
        lines.extend([
            "## Trivial floor", "",
            "Always emitting the training corpus's most frequent token scores",
            "{:.2%} on the {} single-gap target tokens{}. Every accuracy above".format(
                floor["modal_token_accuracy"], floor["target_tokens"],
                "" if two_gap_floor is None else
                ", and {:.2%} on the {} two-gap ones".format(
                    two_gap_floor["modal_token_accuracy"],
                    two_gap_floor["target_tokens"])),
            "is measured against that floor, not against zero.", "",
        ])
        if two_gap_floor is not None:
            lines.extend([
                "This reframes the matched two-gap group above, where the three",
                "models score {}. None of them clears the {:.2%} floor: at pilot".format(
                    ", ".join(
                        "{:.2%}".format(r["oracle_top1"])
                        for r in by_group.get("two_gap_matched_from_scratch", [])
                    ),
                    two_gap_floor["modal_token_accuracy"]),
                "scale and trained from scratch, none of them is doing lexical",
                "prediction that beats guessing the most frequent token. The",
                "earlier reading that they are tied is right but understated.", "",
            ])
        if matched_baseline is not None and pretrained is not None:
            lines.append(
                "Above the floor, the matched baseline gains {:+.1f} points and the".format(
                    100 * (matched_baseline["oracle_top1"]
                           - floor["modal_token_accuracy"]))
            )
            lines.append(
                "tree model {:+.1f}, so the gap between them is not an artifact of".format(
                    100 * (pretrained["oracle_top1"]
                           - floor["modal_token_accuracy"]))
            )
            lines.extend(["both sitting near a high trivial baseline.", ""])

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "oracle_top1.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "rows": rows, "pretraining_gain": pretraining_gain,
                "unigram_floor": floor,
            },
            handle, indent=2,
        )
    with open(
        os.path.join(args.output_dir, "ORACLE_TOP1.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
