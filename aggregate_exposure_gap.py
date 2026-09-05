"""Consolidate the exposure-gap arms against the fixed-mask-bank baseline.

Three checkpoints share a corpus, seed, budget and every optimization setting,
and differ only in the auxiliary added to the exact marginal:

- baseline    no auxiliary
- treatment   self-generated boundary tokens, scored under the exact posterior
- control     the identical term and record draw with the gold boundaries kept

The control is what separates "the substitution taught the model something"
from "a second token-likelihood term helps", which
`research/JOINT_LEXICAL_OBJECTIVE.md` already established on its own.
"""

import argparse
import json
import os
from typing import Dict

ROWS = (
    ("validation exact NLL", ("validation_likelihood", "sequence_nll"), "lower"),
    ("test exact NLL", ("test_likelihood", "sequence_nll"), "lower"),
    ("oracle-midpoint token NLL", ("test_oracle_midpoint_token_nll",), "lower"),
    ("length TV to prior", ("length_metrics", "marginal_tv_to_prior"), "lower"),
    ("length TV to empirical", ("length_metrics", "marginal_tv_to_empirical"), "lower"),
    ("conditional Brier", ("length_metrics", "conditional_brier"), "lower"),
    ("length match probability", ("length_metrics", "observed_target_match_probability"), "higher"),
    ("P(empty)", ("length_metrics", "predicted_empty_probability"), "-"),
    ("P(overflow)", ("length_metrics", "predicted_overflow_probability"), "-"),
    ("mean length", ("length_metrics", "predicted_capped_mean_length"), "-"),
)


def dig(blob: Dict, path):
    value = blob
    for key in path:
        value = value[key]
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir", default="artifacts/text_depth_inside_fixed_mask_bank"
    )
    parser.add_argument(
        "--treatment-dir", default="artifacts/text_exposure_self_boundary"
    )
    parser.add_argument(
        "--control-dir", default="artifacts/text_exposure_boundary_control"
    )
    parser.add_argument(
        "--diagnostic",
        default="artifacts/text_exposure_gap_diagnostic/exposure_gap.json",
    )
    parser.add_argument("--output-dir", default="artifacts/text_exposure_summary")
    args = parser.parse_args()

    arms = {}
    for name, directory in (
        ("baseline", args.baseline_dir),
        ("treatment", args.treatment_dir),
        ("control", args.control_dir),
    ):
        path = os.path.join(directory, "results.json")
        if not os.path.exists(path):
            print("missing", path)
            continue
        with open(path, encoding="utf-8") as handle:
            arms[name] = json.load(handle)

    if "baseline" not in arms:
        raise SystemExit("baseline results are required")

    table = {}
    for label, path, direction in ROWS:
        table[label] = {
            "direction": direction,
            "values": {name: dig(blob, path) for name, blob in arms.items()},
        }

    summary = {
        "arms": {
            name: {
                "artifact_dir": blob["config"]["artifact_dir"],
                "selected_epoch": blob["selected_epoch"],
                "self_boundary_weight": blob["config"].get("self_boundary_weight", 0.0),
                "self_boundary_control": blob["config"].get("self_boundary_control", False),
                "self_topology_weight": blob["config"].get("self_topology_weight", 0.0),
                "validation_history": [
                    row["validation_sequence_nll"] for row in blob["history"]
                ],
            }
            for name, blob in arms.items()
        },
        "metrics": table,
    }
    if os.path.exists(args.diagnostic):
        with open(args.diagnostic, encoding="utf-8") as handle:
            summary["diagnostic"] = json.load(handle)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "exposure_summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    names = [name for name in ("baseline", "treatment", "control") if name in arms]
    header = "| metric | " + " | ".join(names) + " | treatment-baseline | control-baseline |"
    lines = [header, "|---|" + "---:|" * (len(names) + 2)]
    for label, block in table.items():
        values = block["values"]
        cells = ["%.4f" % values[name] for name in names]
        deltas = []
        for other in ("treatment", "control"):
            if other in values:
                deltas.append("%+.4f" % (values[other] - values["baseline"]))
            else:
                deltas.append("-")
        lines.append("| " + label + " | " + " | ".join(cells + deltas) + " |")
    body = "\n".join(lines)
    with open(
        os.path.join(args.output_dir, "EXPOSURE_SUMMARY.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("# Exposure-gap arms\n\n" + body + "\n")
    print(body)


if __name__ == "__main__":
    main()
