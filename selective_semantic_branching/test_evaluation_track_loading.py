import json

import pytest

from selective_semantic_branching.evaluation_tracks import (
    default_manifest_path,
    difficulty_groups,
    load_track_examples,
    resolve_track_path,
    select,
)


MANIFEST = [
    {
        "example_id": "test:0:uniform:1:0",
        "segments": [[11, 12], [13]],
        "spans": [[21, 22]],
    },
    {
        "example_id": "test:1:uniform:1:0",
        "segments": [[14], [15, 16]],
        "spans": [[23]],
    },
    {
        "example_id": "test:2:uniform:2:0",
        "segments": [[17], [18], [19]],
        "spans": [[24], [25]],
    },
    {
        "example_id": "validation:0:uniform:1:0",
        "segments": [[31], [32]],
        "spans": [[41]],
    },
]

TRACK = [
    {
        "example_id": "test:0:uniform:1:0",
        "split": "test",
        "gap_count": 1,
        "difficulty_bin": "easy",
        "balanced_cell_weight": 0.5,
    },
    {
        "example_id": "test:2:uniform:2:0",
        "split": "test",
        "gap_count": 2,
        "difficulty_bin": "hard",
        "balanced_cell_weight": 0.25,
    },
    {
        "example_id": "test:1:uniform:1:0",
        "split": "test",
        "gap_count": 1,
        "difficulty_bin": "hard",
        "balanced_cell_weight": 0.25,
    },
    {
        "example_id": "validation:0:uniform:1:0",
        "split": "validation",
        "gap_count": 1,
        "difficulty_bin": "easy",
        "balanced_cell_weight": 1.0,
    },
]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


@pytest.fixture
def audit_dir(tmp_path):
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    write_jsonl(tmp_path / "corruption_manifest.jsonl", MANIFEST)
    write_jsonl(tracks / "track_a_length_difficulty_balanced.jsonl", TRACK)
    return tmp_path


def test_default_manifest_sits_above_the_tracks_directory(audit_dir):
    track = audit_dir / "tracks" / "track_a_length_difficulty_balanced.jsonl"
    assert default_manifest_path(track) == audit_dir / "corruption_manifest.jsonl"


def test_a_tracks_directory_is_rejected_in_favour_of_one_file(audit_dir):
    assert resolve_track_path("a/b/custom.jsonl").name == "custom.jsonl"
    with pytest.raises(ValueError):
        resolve_track_path(audit_dir / "tracks")


def test_track_selects_one_split_and_rebuilds_token_content(audit_dir):
    track = audit_dir / "tracks" / "track_a_length_difficulty_balanced.jsonl"
    examples, records, summary = load_track_examples(track, split="test")

    assert [record["example_id"] for record in records] == [
        "test:0:uniform:1:0", "test:1:uniform:1:0"
    ]
    assert examples[0].segments == ((11, 12), (13,))
    assert examples[0].spans == ((21, 22),)
    assert examples[0].reconstruct() == [11, 12, 21, 22, 13]
    assert summary["selected"] == 2
    assert summary["split"] == "test"
    assert summary["missing_source_ids"] == []


def test_multi_gap_cells_are_reported_not_silently_dropped(audit_dir):
    track = audit_dir / "tracks" / "track_a_length_difficulty_balanced.jsonl"
    _, _, summary = load_track_examples(track, split="test")
    assert summary["skipped_multi_gap"] == 1
    assert summary["difficulty_bins"] == {"easy": 1, "hard": 1}


def test_limit_truncates_in_track_order(audit_dir):
    track = audit_dir / "tracks" / "track_a_length_difficulty_balanced.jsonl"
    examples, records, summary = load_track_examples(track, split="test", limit=1)
    assert len(examples) == 1
    assert records[0]["example_id"] == "test:0:uniform:1:0"
    assert summary["balanced_weight_total"] == pytest.approx(0.5)


def test_a_split_with_no_single_gap_examples_is_an_error(audit_dir):
    track = audit_dir / "tracks" / "track_a_length_difficulty_balanced.jsonl"
    with pytest.raises(ValueError):
        load_track_examples(track, split="train")


def test_a_duplicate_manifest_key_is_rejected(audit_dir):
    write_jsonl(
        audit_dir / "corruption_manifest.jsonl", MANIFEST + [MANIFEST[0]]
    )
    track = audit_dir / "tracks" / "track_a_length_difficulty_balanced.jsonl"
    with pytest.raises(ValueError):
        load_track_examples(track, split="test")


def test_missing_source_records_are_listed(audit_dir):
    write_jsonl(audit_dir / "corruption_manifest.jsonl", MANIFEST[:1])
    track = audit_dir / "tracks" / "track_a_length_difficulty_balanced.jsonl"
    examples, _, summary = load_track_examples(track, split="test")
    assert len(examples) == 1
    assert summary["missing_source_ids"] == ["test:1:uniform:1:0"]


def test_difficulty_groups_index_into_the_rollout_rows(audit_dir):
    track = audit_dir / "tracks" / "track_a_length_difficulty_balanced.jsonl"
    _, records, _ = load_track_examples(track, split="test")
    groups = difficulty_groups(records)
    assert groups == {"easy": [0], "hard": [1]}
    assert difficulty_groups(records, minimum=2) == {}
    assert select(["a", "b"], groups["hard"]) == ["b"]
