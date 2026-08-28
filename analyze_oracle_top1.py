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
    add(
        "single_gap_pretrained", "Masked baseline, same pretrained backbone",
        dig(
            read(os.path.join(
                root, "text_pretrained_masked_baseline", "results.json",
            )),
            "oracle_metrics", ORACLE_KEY,
        ),
        True, "85M, same backbone, stream, split and budget -- the matched control",
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts")
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
            "Filling masks is the task the backbone was pretrained on, so the",
            "baseline draws more from it than the tree model can. That asymmetry",
            "is real, and it is also the point: where a pretrained masked encoder",
            "is available, using it directly beats adapting it to an interval",
            "chart on this task at this scale.", "",
        ])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "oracle_top1.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"rows": rows, "pretraining_gain": pretraining_gain}, handle, indent=2
        )
    with open(
        os.path.join(args.output_dir, "ORACLE_TOP1.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
