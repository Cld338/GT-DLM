"""Select the joint lexical weight from validation metrics only."""

import argparse
import json
import os


DEFAULT_CANDIDATES = (
    (
        0.0,
        "artifacts/text_depth_inside_pretrained_exact_control",
        "artifacts/text_depth_inside_pretrained_exact_control_validation_lexical",
    ),
    (
        0.25,
        "artifacts/text_depth_inside_lambda025_validation",
        "artifacts/text_depth_inside_lambda025_validation_lexical",
    ),
    (
        0.5,
        "artifacts/text_depth_inside_lambda05_validation",
        "artifacts/text_depth_inside_lambda05_validation_lexical",
    ),
    (
        1.0,
        "artifacts/text_depth_inside_joint",
        "artifacts/text_depth_inside_joint_validation_lexical",
    ),
)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def candidate_row(weight, training_dir, lexical_dir):
    training = read_json(os.path.join(training_dir, "results.json"))
    lexical = read_json(os.path.join(lexical_dir, "results.json"))
    if lexical.get("evaluation_split") != "validation":
        raise ValueError("lexical candidate was not evaluated on validation")
    metrics = lexical["oracle_metrics"]
    return {
        "lexical_weight": weight,
        "training_dir": training_dir,
        "lexical_dir": lexical_dir,
        "validation_exact_nll": float(
            training["validation_likelihood"]["sequence_nll"]
        ),
        "validation_lexical_token_nll": float(
            lexical["validation_token_nll"]
        ),
        "validation_oracle_edit": float(
            metrics["matched_length_edit_similarity"]
        ),
        "validation_oracle_token_accuracy": float(
            metrics["matched_length_token_accuracy"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exact-slack", type=float, default=0.1,
        help="maximum validation exact-NLL degradation from lambda=0",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_depth_inside_lambda_selection"
    )
    args = parser.parse_args()
    rows = [candidate_row(*candidate) for candidate in DEFAULT_CANDIDATES]
    baseline = next(row for row in rows if row["lexical_weight"] == 0.0)
    threshold = baseline["validation_exact_nll"] + args.exact_slack
    for row in rows:
        row["exact_nll_delta_from_lambda0"] = (
            row["validation_exact_nll"] - baseline["validation_exact_nll"]
        )
        row["eligible"] = row["validation_exact_nll"] <= threshold
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("no lexical-weight candidate satisfies exact constraint")
    selected = min(
        eligible,
        key=lambda row: (
            row["validation_lexical_token_nll"], row["lexical_weight"]
        ),
    )
    result = {
        "selection_rule": (
            "minimize validation aligned lexical token NLL subject to validation "
            "exact NLL <= lambda0 + exact_slack"
        ),
        "exact_slack": args.exact_slack,
        "exact_threshold": threshold,
        "selected_lexical_weight": selected["lexical_weight"],
        "selected_training_dir": selected["training_dir"],
        "candidates": rows,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "selection.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Validation-only lexical-weight selection", "",
        "Rule: minimize validation aligned lexical token NLL subject to exact NLL no more than `{:.3f}` nats above lambda=0.".format(
            args.exact_slack
        ), "",
        "| Lambda | Exact NLL | Delta | Lexical token NLL | Oracle edit | Oracle token acc. | Eligible |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {:.2f} | {:.3f} | {:+.3f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                row["lexical_weight"], row["validation_exact_nll"],
                row["exact_nll_delta_from_lambda0"],
                row["validation_lexical_token_nll"],
                row["validation_oracle_edit"],
                row["validation_oracle_token_accuracy"],
                "yes" if row["eligible"] else "no",
            )
        )
    lines.extend([
        "", "Selected `lambda={:.2f}`.".format(selected["lexical_weight"]),
    ])
    with open(os.path.join(args.output_dir, "VALIDATION_SELECTION.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
