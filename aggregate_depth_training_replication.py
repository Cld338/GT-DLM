"""Aggregate independent depth-inside training runs without reevaluation."""

import argparse
import json
import os
import statistics


def mean_sd(values):
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dirs",
        default=(
            "artifacts/text_depth_inside_screen,"
            "artifacts/text_depth_inside_seed23,"
            "artifacts/text_depth_inside_seed41"
        ),
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_depth_inside_training_replication"
    )
    args = parser.parse_args()
    rows = []
    for artifact_dir in args.artifact_dirs.split(","):
        with open(os.path.join(artifact_dir, "results.json"), encoding="utf-8") as handle:
            training = json.load(handle)
        with open(os.path.join(artifact_dir, "root_stop_calibration.json"), encoding="utf-8") as handle:
            calibration = json.load(handle)
        config = training["config"]
        rows.append({
            "artifact_dir": artifact_dir,
            "training_seed": int(config.get("training_seed", config["seed"])),
            "test_sequence_nll": training["test_likelihood"]["sequence_nll"],
            "midpoint_joint_nll": training["test_likelihood"]["midpoint_joint_nll"],
            "raw_tv": training["length_metrics"]["marginal_tv_to_prior"],
            "raw_js": training["length_metrics"]["marginal_js_to_prior_nats"],
            "raw_empty": training["length_metrics"]["predicted_empty_probability"],
            "raw_overflow": training["length_metrics"]["predicted_overflow_probability"],
            "root_bias": calibration["validation"]["root_stop_logit_bias"],
            "calibrated_tv": calibration["test"]["calibrated"]["marginal_tv_to_prior"],
            "calibrated_js": calibration["test"]["calibrated"]["marginal_js_to_prior_nats"],
            "calibrated_empty": calibration["test"]["calibrated"]["predicted_empty_probability"],
            "calibrated_overflow": calibration["test"]["calibrated"]["predicted_overflow_probability"],
        })
    fields = [
        "test_sequence_nll", "midpoint_joint_nll", "raw_tv", "raw_js",
        "raw_empty", "raw_overflow", "root_bias", "calibrated_tv",
        "calibrated_js", "calibrated_empty", "calibrated_overflow",
    ]
    summary = {field: mean_sd([row[field] for row in rows]) for field in fields}
    summary["raw_tv_pass_rate"] = sum(row["raw_tv"] < 0.20 for row in rows) / len(rows)
    result = {"config": vars(args), "runs": rows, "summary": summary}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_replication.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Depth-inside training-seed replication", "",
        "Validation/test data and prompts are fixed; initialization, shuffle, and dynamic training corruptions vary.",
        "", "| Seed | Test NLL | Raw TV | Raw P(empty) | Raw P(overflow) | Root bias | Cal. TV | Cal. P(overflow) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {training_seed} | {test_sequence_nll:.3f} | {raw_tv:.3f} | {raw_empty:.3f} | {raw_overflow:.3f} | {root_bias:.3f} | {calibrated_tv:.3f} | {calibrated_overflow:.3f} |".format(**row)
        )
    lines.extend([
        "",
        "| Summary | Test NLL | Raw TV | Raw P(empty) | Raw P(overflow) | Root bias | Cal. TV | Cal. P(overflow) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| Mean +/- SD | {:.3f}+/-{:.3f} | {:.3f}+/-{:.3f} | {:.3f}+/-{:.3f} | {:.3f}+/-{:.3f} | {:.3f}+/-{:.3f} | {:.3f}+/-{:.3f} | {:.3f}+/-{:.3f} |".format(
            summary["test_sequence_nll"]["mean"], summary["test_sequence_nll"]["sample_sd"],
            summary["raw_tv"]["mean"], summary["raw_tv"]["sample_sd"],
            summary["raw_empty"]["mean"], summary["raw_empty"]["sample_sd"],
            summary["raw_overflow"]["mean"], summary["raw_overflow"]["sample_sd"],
            summary["root_bias"]["mean"], summary["root_bias"]["sample_sd"],
            summary["calibrated_tv"]["mean"], summary["calibrated_tv"]["sample_sd"],
            summary["calibrated_overflow"]["mean"], summary["calibrated_overflow"]["sample_sd"],
        ),
        "", "Raw TV passes the preregistered `<0.20` gate in {}/{} training seeds.".format(
            sum(row["raw_tv"] < 0.20 for row in rows), len(rows)
        ),
    ])
    with open(os.path.join(args.output_dir, "TRAINING_REPLICATION.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
