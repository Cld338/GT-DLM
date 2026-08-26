"""Aggregate matched joint-versus-exact lexical experiments across seeds."""

import json
import os
import statistics


PAIRS = (
    {
        "seed": 17,
        "joint": "artifacts/text_depth_inside_joint",
        "control": "artifacts/text_depth_inside_pretrained_exact_control",
        "joint_lexical": "artifacts/text_depth_inside_joint_oracle",
        "control_lexical": "artifacts/text_depth_inside_pretrained_exact_control_oracle",
        "comparison": "artifacts/text_depth_inside_joint_vs_control/sequence_likelihoods.json",
        "comparison_key": "depth_inside_seed17_vs_pretrained_exact_control",
    },
    {
        "seed": 23,
        "joint": "artifacts/text_depth_inside_joint_seed23",
        "control": "artifacts/text_depth_inside_pretrained_exact_control_seed23",
        "joint_lexical": "artifacts/text_depth_inside_joint_seed23_oracle",
        "control_lexical": "artifacts/text_depth_inside_pretrained_exact_control_seed23_oracle",
        "comparison": "artifacts/text_depth_inside_joint_seed23_vs_control/sequence_likelihoods.json",
        "comparison_key": "joint_seed23_vs_exact_control_seed23",
    },
    {
        "seed": 41,
        "joint": "artifacts/text_depth_inside_joint_seed41",
        "control": "artifacts/text_depth_inside_pretrained_exact_control_seed41",
        "joint_lexical": "artifacts/text_depth_inside_joint_seed41_oracle",
        "control_lexical": "artifacts/text_depth_inside_pretrained_exact_control_seed41_oracle",
        "comparison": "artifacts/text_depth_inside_joint_seed41_vs_control/sequence_likelihoods.json",
        "comparison_key": "joint_seed41_vs_exact_control_seed41",
    },
)


def read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def metrics(directory, lexical_directory):
    training = read(os.path.join(directory, "results.json"))
    calibration = read(os.path.join(directory, "root_stop_calibration.json"))
    lexical = read(os.path.join(lexical_directory, "results.json"))
    return {
        "exact_nll": training["test_likelihood"]["sequence_nll"],
        "raw_tv": training["length_metrics"]["marginal_tv_to_prior"],
        "calibrated_tv": calibration["test"]["calibrated"]["marginal_tv_to_prior"],
        "lexical_token_nll": lexical["test_token_nll"],
        "oracle_token_accuracy": lexical["oracle_metrics"][
            "matched_length_token_accuracy"
        ],
    }


def main():
    rows = []
    for pair in PAIRS:
        joint = metrics(pair["joint"], pair["joint_lexical"])
        control = metrics(pair["control"], pair["control_lexical"])
        comparison = read(pair["comparison"])["paired_comparisons"][
            pair["comparison_key"]
        ]
        rows.append({
            "seed": pair["seed"],
            "control": control,
            "joint": joint,
            "joint_minus_control": {
                key: joint[key] - control[key] for key in joint
            },
            "paired_exact_nll": comparison,
        })
    summary = {}
    for metric in rows[0]["joint"]:
        differences = [row["joint_minus_control"][metric] for row in rows]
        summary[metric] = {
            "mean_joint_minus_control": statistics.mean(differences),
            "sample_sd": statistics.stdev(differences),
        }
    result = {"completed_seeds": [17, 23, 41], "rows": rows, "summary": summary}
    output_dir = "artifacts/text_depth_inside_joint_replication"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "replication.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Joint lexical objective replication", "",
        "Joint-minus-control deltas use matched lexical-pretrained initialization and training seed.",
        "", "| Seed | Control exact | Joint exact | Delta | Paired 95% CI | Control lexical | Joint lexical | Control cal. TV | Joint cal. TV |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = row["paired_exact_nll"]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:+.3f} | [{:+.3f},{:+.3f}] | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                row["seed"], row["control"]["exact_nll"],
                row["joint"]["exact_nll"],
                row["joint_minus_control"]["exact_nll"],
                ci["bootstrap_95_low"], ci["bootstrap_95_high"],
                row["control"]["lexical_token_nll"],
                row["joint"]["lexical_token_nll"],
                row["control"]["calibrated_tv"],
                row["joint"]["calibrated_tv"],
            )
        )
    lines.extend([
        "", "Three-seed mean joint-minus-control deltas:", "",
        "- exact NLL: `{:+.3f}`".format(summary["exact_nll"]["mean_joint_minus_control"]),
        "- aligned lexical token NLL: `{:+.3f}`".format(summary["lexical_token_nll"]["mean_joint_minus_control"]),
        "- calibrated TV: `{:+.3f}`".format(summary["calibrated_tv"]["mean_joint_minus_control"]),
        "", "The fixed validation-selected protocol is complete for all three seeds.",
    ])
    with open(os.path.join(output_dir, "REPLICATION.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
