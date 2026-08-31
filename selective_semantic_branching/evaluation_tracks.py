"""Load fixed Track A/B evaluation sets for rollout scoring.

`build_evaluation_tracks.py` writes feature records, not token content, so a
track file alone cannot be decoded. This module joins a track back to the
corruption manifest that produced it and rebuilds the exact infilling examples.
The join key is `example_id`, which the audit guarantees is unique after
duplicate source documents are resolved.

Rollout metrics require one GAP per prompt, so multi-GAP track cells are
reported as skipped rather than silently dropped.
"""

import json
from pathlib import Path

from gtdlm.text_data import TextInfillingExample


TRACK_FILENAMES = (
    "track_a_length_difficulty_balanced.jsonl",
    "track_b_natural.jsonl",
    "empty_calibration.jsonl",
)


def resolve_track_path(track):
    """Require one track file: a tracks directory holds several of them."""
    path = Path(track)
    if path.is_dir():
        raise ValueError(
            "{} is a directory; name one track file, such as {}".format(
                path, ", ".join(TRACK_FILENAMES)
            )
        )
    return path


def default_manifest_path(track_path):
    """`tracks/<file>.jsonl` sits one level below its source audit directory."""
    return Path(track_path).parent.parent / "corruption_manifest.jsonl"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def example_from_record(record):
    return TextInfillingExample(
        segments=tuple(tuple(int(token) for token in segment)
                       for segment in record["segments"]),
        spans=tuple(tuple(int(token) for token in span)
                    for span in record["spans"]),
    )


def load_track_examples(track_path, manifest_path=None, split="test", limit=0):
    """Return the examples, their track records, and a provenance summary."""
    track_path = Path(track_path)
    manifest_path = Path(
        manifest_path if manifest_path else default_manifest_path(track_path)
    )
    sources = {}
    for record in read_jsonl(manifest_path):
        example_id = record["example_id"]
        if example_id in sources:
            raise ValueError("duplicate example_id in manifest: " + example_id)
        sources[example_id] = record

    examples = []
    records = []
    skipped_multi_gap = 0
    missing_source = []
    for record in read_jsonl(track_path):
        if split and record.get("split") != split:
            continue
        if int(record.get("gap_count", 1)) != 1:
            skipped_multi_gap += 1
            continue
        source = sources.get(record["example_id"])
        if source is None:
            missing_source.append(record["example_id"])
            continue
        examples.append(example_from_record(source))
        records.append(record)
        if limit and len(examples) >= limit:
            break

    if not examples:
        raise ValueError(
            "track {} split {} produced no single-GAP examples".format(
                track_path.name, split
            )
        )
    summary = {
        "track": str(track_path),
        "manifest": str(manifest_path),
        "split": split,
        "selected": len(examples),
        "skipped_multi_gap": skipped_multi_gap,
        "missing_source_ids": missing_source,
        "difficulty_bins": difficulty_counts(records),
        "balanced_weight_total": sum(
            float(record.get("balanced_cell_weight", 0.0)) for record in records
        ),
    }
    return examples, records, summary


def difficulty_counts(records):
    counts = {}
    for record in records:
        name = str(record.get("difficulty_bin", "unbinned"))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def difficulty_groups(records, minimum=1):
    """Map each difficulty bin to the example indices it owns."""
    groups = {}
    for index, record in enumerate(records):
        name = str(record.get("difficulty_bin", "unbinned"))
        groups.setdefault(name, []).append(index)
    return {
        name: indices
        for name, indices in sorted(groups.items())
        if len(indices) >= minimum
    }


def select(rows, indices):
    return [rows[index] for index in indices]
