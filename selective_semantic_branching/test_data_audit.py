import math

from gtdlm.text_data import TextInfillingExample, TextVocabulary
from selective_semantic_branching.audit_training_data import (
    empirical_mutual_information,
    example_intervals,
    bootstrap_mean_interval,
    cluster_bootstrap_mean_interval,
    deduplicate_documents,
    make_record,
    render_gap_query,
    stratified_sample,
    summarize_records,
)


class TinyTokenizer:
    @staticmethod
    def convert_ids_to_tokens(ids):
        return ["##x" if token == 12 else "x" for token in ids]


def test_manifest_record_round_trips_intervals_and_source_group():
    example = TextInfillingExample(
        ((1, 2), (5,), (8, 9)), ((3, 4), (6, 7))
    )
    assert example_intervals(example) == [(2, 4), (5, 7)]
    record = make_record(
        "train", 4, 11, example.reconstruct(), "uniform", 2, 0, 99,
        example, TinyTokenizer(),
    )
    assert record["source_document_id"] == "train:4"
    assert [gap["length"] for gap in record["gaps"]] == [2, 2]
    assert math.isclose(record["corruption_ratio"], 4.0 / 9.0)


def test_render_gap_query_reveals_only_other_gold_spans():
    vocab = TextVocabulary(
        32, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2, EXTRA_STRUCTURAL=(3,)
    )
    example = TextInfillingExample(
        ((4,), (5,), (6,)), ((10, 11), (12,))
    )
    masked, masked_position = render_gap_query(example, vocab, 1, False)
    revealed, revealed_position = render_gap_query(example, vocab, 1, True)
    assert masked == [1, 4, 9, 5, 9, 6, 2]
    assert masked_position == 4
    assert revealed == [1, 4, 10, 11, 5, 9, 6, 2]
    assert revealed_position == 5


def test_mutual_information_and_summary_are_deterministic():
    independent = [(0, 0), (0, 1), (1, 0), (1, 1)]
    dependent = [(0, 0), (0, 0), (1, 1), (1, 1)]
    assert math.isclose(empirical_mutual_information(independent), 0.0)
    assert math.isclose(empirical_mutual_information(dependent), math.log(2.0))
    gap1 = {
        "length": 1, "visible_copy": False,
        "starts_inside_wordpiece": False, "ends_inside_wordpiece": False,
        "compatible_action_count": 1,
    }
    gap2 = {**gap1, "length": 2, "compatible_action_count": 2}
    base = {
        "policy": "uniform", "gap_count": 2, "corruption_ratio": 0.2,
        "gaps": [gap1, gap2],
    }
    records = [
        {
            **base, "split": "train", "source_document_id": "train:0",
            "source_document_sha256": "a", "window_sha256": "wa",
        },
        {
            **base, "split": "test", "source_document_id": "test:0",
            "source_document_sha256": "b", "window_sha256": "wb",
        },
    ]
    summary = summarize_records(records, {"attempted": {}, "accepted": {}})
    assert summary["split_leakage_count"] == 0
    assert summary["groups"]["train|uniform|2"]["mean_gap_length"] == 1.5
    interval = bootstrap_mean_interval([1.0, 1.0, 1.0], samples=20)
    assert interval == [1.0, 1.0]
    cluster_interval = cluster_bootstrap_mean_interval(
        [[2.0, 2.0], [2.0]], samples=20
    )
    assert cluster_interval == [2.0, 2.0]


def test_summary_detects_exact_source_overlap_across_splits():
    gap = {
        "length": 1, "visible_copy": False,
        "starts_inside_wordpiece": False, "ends_inside_wordpiece": False,
        "compatible_action_count": 1,
    }
    records = [
        {
            "split": split, "policy": "uniform", "gap_count": 1,
            "corruption_ratio": 0.1, "gaps": [gap],
            "source_document_id": split + ":0",
            "source_document_sha256": "same", "window_sha256": split,
        }
        for split in ("train", "test")
    ]
    summary = summarize_records(records, {"attempted": {}, "accepted": {}})
    assert summary["split_leakage_count"] == 1
    assert summary["exact_window_overlap_count"] == 0


def test_document_deduplication_keeps_evaluation_copy():
    documents = {
        "train": [(0, (1, 2, 3)), (1, (4, 5, 6))],
        "validation": [(0, (7, 8, 9))],
        "test": [(0, (1, 2, 3))],
    }
    kept, excluded = deduplicate_documents(documents)
    assert [index for index, _ in kept["train"]] == [1]
    assert [index for index, _ in kept["test"]] == [0]
    assert excluded[0]["excluded_source_document_id"] == "train:0"
    assert excluded[0]["kept_source_document_id"] == "test:0"


def test_scoring_sample_round_robins_over_groups():
    records = [
        {"split": "train", "policy": policy, "gap_count": gap}
        for _ in range(3)
        for policy in ("uniform", "copy")
        for gap in (1, 2)
    ]
    sample = stratified_sample(records, 4)
    assert len({
        (record["policy"], record["gap_count"]) for record in sample
    }) == 4
