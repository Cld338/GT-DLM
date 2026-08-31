"""Build fixed Track A/B and DEFER strata from a scored corruption manifest."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


MIN_DIFFICULTY_CELL = 30


def quantile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("cannot compute a quantile from no values")
    return values[int(round((len(values) - 1) * probability))]


def length_regime(lengths):
    nonempty = [int(length) for length in lengths if int(length) > 0]
    if not nonempty:
        return "empty"
    maximum = max(nonempty)
    if maximum <= 2:
        return "short"
    if maximum <= 5:
        return "medium"
    return "long"


def record_features(record):
    lengths = [int(gap["length"]) for gap in record["gaps"]]
    nlls = []
    benefits = []
    for gap in record["gaps"]:
        scores = gap.get("scores", {})
        masked = scores.get("all_masked", {})
        if "compatible_joint_nll" in masked:
            nlls.append(float(masked["compatible_joint_nll"]))
        if "cross_gap_joint_information_gain_nats" in scores:
            benefits.append(float(
                scores["cross_gap_joint_information_gain_nats"]
            ))
    return {
        "example_id": record["example_id"],
        "source_document_id": record["source_document_id"],
        "source_document_sha256": record["source_document_sha256"],
        "split": record["split"],
        "policy": record["policy"],
        "gap_count": int(record["gap_count"]),
        "gap_lengths": lengths,
        "total_target_length": sum(lengths),
        "length_regime": length_regime(lengths),
        "has_empty_gap": any(length == 0 for length in lengths),
        "difficulty_joint_nll": (
            sum(nlls) / len(nlls) if nlls else None
        ),
        "cross_gap_information_gain_nats": (
            sum(benefits) / len(benefits) if benefits else None
        ),
        "scored_nonempty_gaps": len(nlls),
    }


def threshold_key(feature):
    return "{}|{}|{}".format(
        feature["gap_count"],
        feature["length_regime"],
        int(feature["has_empty_gap"]),
    )


def fit_difficulty_thresholds(features):
    grouped = defaultdict(list)
    fallback = defaultdict(list)
    global_values = []
    for feature in features:
        value = feature["difficulty_joint_nll"]
        if feature["split"] != "train" or value is None:
            continue
        grouped[threshold_key(feature)].append(value)
        fallback[str(feature["gap_count"])].append(value)
        global_values.append(value)
    if not global_values:
        raise ValueError("no scored train records are available")
    thresholds = {}
    keys = set(
        threshold_key(feature) for feature in features
        if feature["difficulty_joint_nll"] is not None
    )
    for key in sorted(keys):
        gap_count = key.split("|")[0]
        exact = grouped.get(key, [])
        values = (
            exact if len(exact) >= MIN_DIFFICULTY_CELL
            else fallback.get(gap_count) or global_values
        )
        thresholds[key] = {
            "count": len(values),
            "easy_max": quantile(values, 1.0 / 3.0),
            "medium_max": quantile(values, 2.0 / 3.0),
            "fallback": (
                "exact" if len(exact) >= MIN_DIFFICULTY_CELL
                else "gap_count" if fallback.get(gap_count)
                else "global"
            ),
            "exact_cell_count": len(exact),
        }
    return thresholds


def assign_difficulty(feature, thresholds):
    value = feature["difficulty_joint_nll"]
    if value is None:
        return "empty"
    threshold = thresholds[threshold_key(feature)]
    if value <= threshold["easy_max"]:
        return "easy"
    if value <= threshold["medium_max"]:
        return "medium"
    return "hard"


def fit_benefit_thresholds(records):
    values = []
    for record in records:
        if record["split"] != "train" or int(record["gap_count"]) < 2:
            continue
        for gap in record["gaps"]:
            value = gap.get("scores", {}).get(
                "cross_gap_joint_information_gain_nats"
            )
            if value is not None:
                values.append(float(value))
    if not values:
        raise ValueError("no scored train cross-GAP benefits are available")
    return {
        "count": len(values),
        "low_max": quantile(values, 1.0 / 3.0),
        "medium_max": quantile(values, 2.0 / 3.0),
    }


def assign_benefit(value, thresholds):
    if value <= thresholds["low_max"]:
        return "negative_or_low"
    if value <= thresholds["medium_max"]:
        return "neutral_or_mixed"
    return "positive_or_high"


def add_balancing_weights(features):
    counts = Counter(
        (
            feature["split"],
            feature["gap_count"],
            feature["length_regime"],
            feature["difficulty_bin"],
        )
        for feature in features
    )
    for feature in features:
        cell = (
            feature["split"],
            feature["gap_count"],
            feature["length_regime"],
            feature["difficulty_bin"],
        )
        feature["balanced_cell_count"] = counts[cell]
        feature["balanced_cell_weight"] = 1.0 / counts[cell]
    return {
        "|".join(map(str, key)): count for key, count in sorted(counts.items())
    }


def build_tracks(records):
    uniform = [
        record for record in records
        if record["policy"] == "uniform"
        and any("scores" in gap for gap in record["gaps"])
    ]
    features = [record_features(record) for record in uniform]
    difficulty_thresholds = fit_difficulty_thresholds(features)
    for feature in features:
        feature["difficulty_bin"] = assign_difficulty(
            feature, difficulty_thresholds
        )

    natural = [
        dict(feature) for feature in features
        if feature["split"] in ("validation", "test")
    ]
    lexical = [
        dict(feature) for feature in natural
        if feature["difficulty_joint_nll"] is not None
    ]
    lexical_cells = add_balancing_weights(lexical)
    empty = [
        dict(feature) for feature in natural
        if feature["difficulty_joint_nll"] is None
    ]

    benefit_thresholds = fit_benefit_thresholds(uniform)
    defer = []
    for record in uniform:
        if int(record["gap_count"]) < 2:
            continue
        for gap in record["gaps"]:
            value = gap.get("scores", {}).get(
                "cross_gap_joint_information_gain_nats"
            )
            if value is None:
                continue
            defer.append({
                "example_id": record["example_id"],
                "source_document_id": record["source_document_id"],
                "split": record["split"],
                "gap_index": int(gap["gap_index"]),
                "gap_length": int(gap["length"]),
                "information_gain_nats": float(value),
                "benefit_tercile": assign_benefit(
                    float(value), benefit_thresholds
                ),
            })
    return {
        "track_b_natural": natural,
        "track_a_length_difficulty_balanced": lexical,
        "empty_calibration": empty,
        "defer_strata": defer,
        "metadata": {
            "difficulty_definition": (
                "mean all-masked compatible-joint NLL over non-empty GAPs"
            ),
            "track_a_weight_definition": (
                "inverse count within split x GAP-count x length-regime x "
                "difficulty-bin cells"
            ),
            "benefit_definition": (
                "compatible-joint NLL before minus after other gold GAPs"
            ),
            "difficulty_thresholds_from_train": difficulty_thresholds,
            "minimum_exact_difficulty_cell": MIN_DIFFICULTY_CELL,
            "benefit_thresholds_from_train": benefit_thresholds,
            "track_a_cells": lexical_cells,
        },
    }


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        default="artifacts/selective_semantic_branching_data_audit_uniform_tracks",
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output_dir) if args.output_dir else audit_dir / "tracks"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = build_tracks(read_jsonl(audit_dir / "corruption_manifest.jsonl"))
    filenames = {
        "track_b_natural": "track_b_natural.jsonl",
        "track_a_length_difficulty_balanced": (
            "track_a_length_difficulty_balanced.jsonl"
        ),
        "empty_calibration": "empty_calibration.jsonl",
        "defer_strata": "defer_strata.jsonl",
    }
    counts = {}
    for name, filename in filenames.items():
        write_jsonl(output_dir / filename, tracks[name])
        counts[name] = len(tracks[name])
    with (output_dir / "track_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump({
            "source_audit_dir": str(audit_dir),
            "counts": counts,
            **tracks["metadata"],
        }, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "counts": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
