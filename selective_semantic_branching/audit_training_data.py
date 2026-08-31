"""Create a reproducible corruption manifest and audit SSB training data.

The structural path is CPU-only. With --score, a frozen SSB checkpoint also
measures each GAP before and after the other gold GAPs are revealed. That
difference diagnoses cross-GAP information; it is not used as a WAIT label.
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import choose_device, seed_everything
from gtdlm.model import PretrainedGapFrontierModel
from gtdlm.text_data import TextInfillingExample, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer
from selective_semantic_branching.data import compatible_root_actions


def stable_seed(base, *parts):
    payload = "|".join([str(base)] + [str(part) for part in parts])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % (2 ** 31)


def token_sequence_sha256(tokens):
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


def select_window(document, seed, minimum, maximum):
    if len(document) < minimum:
        raise ValueError("document is shorter than the minimum window")
    rng = random.Random(seed)
    length = rng.randint(minimum, min(maximum, len(document)))
    start = rng.randint(0, len(document) - length)
    return start, tuple(int(token) for token in document[start : start + length])


def example_intervals(example):
    intervals = []
    cursor = 0
    for segment, span in zip(example.segments[:-1], example.spans):
        cursor += len(segment)
        intervals.append((cursor, cursor + len(span)))
        cursor += len(span)
    cursor += len(example.segments[-1])
    if cursor != len(example.reconstruct()):
        raise AssertionError("intervals do not reconstruct the source")
    return intervals


def span_is_visible_elsewhere(span, observed):
    if not span:
        return True
    return any(
        tuple(observed[start : start + len(span)]) == tuple(span)
        for start in range(len(observed) - len(span) + 1)
    )


def make_record(
    split,
    document_index,
    window_start,
    window,
    policy,
    gap_count,
    candidate_index,
    seed,
    example,
    tokenizer,
    source_document_sha256=None,
):
    reconstructed = example.reconstruct()
    if reconstructed != list(window):
        raise AssertionError("corruption does not reconstruct its source")
    observed = [
        int(token) for segment in example.segments for token in segment
    ]
    pieces = tokenizer.convert_ids_to_tokens(reconstructed)
    gaps = []
    for gap_index, (span, interval) in enumerate(
        zip(example.spans, example_intervals(example))
    ):
        lo, hi = interval
        actions = compatible_root_actions(span)
        gaps.append({
            "gap_index": gap_index,
            "start": lo,
            "end": hi,
            "length": len(span),
            "left_context_tokens": lo,
            "right_context_tokens": len(reconstructed) - hi,
            "relative_start": lo / max(1, len(reconstructed)),
            "visible_copy": span_is_visible_elsewhere(span, observed),
            "compatible_token_count": len(set(token for token, _ in actions)),
            "compatible_action_count": len(actions),
            "starts_inside_wordpiece": (
                lo < len(pieces) and str(pieces[lo]).startswith("##")
            ),
            "ends_inside_wordpiece": (
                hi < len(pieces) and str(pieces[hi]).startswith("##")
            ),
        })
    removed = sum(len(span) for span in example.spans)
    return {
        "example_id": "{}:{}:{}:{}:{}".format(
            split, document_index, policy, gap_count, candidate_index
        ),
        "source_document_id": "{}:{}".format(split, document_index),
        "source_document_sha256": (
            source_document_sha256 or token_sequence_sha256(window)
        ),
        "window_sha256": token_sequence_sha256(window),
        "split": split,
        "document_index": document_index,
        "window_start": window_start,
        "window_length": len(window),
        "policy": policy,
        "gap_count": gap_count,
        "candidate_index": candidate_index,
        "corruption_seed": seed,
        "corruption_ratio": removed / max(1, len(window)),
        "segments": [list(segment) for segment in example.segments],
        "spans": [list(span) for span in example.spans],
        "gaps": gaps,
    }


def deduplicate_documents(eligible_by_split):
    owners = defaultdict(list)
    for split, documents in eligible_by_split.items():
        for document_index, document in documents:
            owners[token_sequence_sha256(document)].append(
                (split, document_index, document)
            )
    priority = {"test": 0, "validation": 1, "train": 2}
    kept = defaultdict(list)
    excluded = []
    for digest, copies in owners.items():
        copies.sort(key=lambda item: (priority.get(item[0], 3), item[1]))
        keeper = copies[0]
        kept[keeper[0]].append((keeper[1], keeper[2]))
        for split, document_index, _document in copies[1:]:
            excluded.append({
                "source_document_sha256": digest,
                "excluded_source_document_id": "{}:{}".format(
                    split, document_index
                ),
                "kept_source_document_id": "{}:{}".format(
                    keeper[0], keeper[1]
                ),
            })
    for documents in kept.values():
        documents.sort(key=lambda item: item[0])
    return kept, sorted(
        excluded, key=lambda item: item["excluded_source_document_id"]
    )


def build_candidate_records(corpus, tokenizer, args, policies, gap_counts):
    records = []
    attempted, accepted = Counter(), Counter()
    eligible_by_split = {}
    for split in ("train", "validation", "test"):
        if split not in corpus:
            continue
        eligible = [
            (index, document)
            for index, document in enumerate(corpus[split])
            if len(document) >= args.window_min
        ]
        if args.max_documents_per_split:
            eligible = eligible[: args.max_documents_per_split]
        eligible_by_split[split] = eligible
    eligible_before = {
        split: len(documents)
        for split, documents in eligible_by_split.items()
    }
    eligible_by_split, excluded_duplicates = deduplicate_documents(
        eligible_by_split
    )
    eligible_after = {
        split: len(eligible_by_split.get(split, []))
        for split in eligible_before
    }
    for split in ("train", "validation", "test"):
        eligible = eligible_by_split.get(split, [])
        for document_index, document in eligible:
            document_sha256 = token_sequence_sha256(document)
            window_start, window = select_window(
                document,
                stable_seed(args.seed, "window", split, document_index),
                args.window_min,
                args.window_max,
            )
            for policy in policies:
                for gap_count in gap_counts:
                    key = "{}|{}|{}".format(split, policy, gap_count)
                    for candidate_index in range(args.examples_per_document):
                        attempted[key] += 1
                        seed = stable_seed(
                            args.seed,
                            "corruption",
                            split,
                            document_index,
                            policy,
                            gap_count,
                            candidate_index,
                        )
                        examples = sample_text_infilling_examples(
                            [window],
                            seed=seed,
                            gap_counts=(gap_count,),
                            min_span=args.min_span,
                            max_span=args.max_span,
                            zero_length_probability=args.zero_length_probability,
                            span_policy=policy,
                        )
                        if not examples:
                            continue
                        accepted[key] += 1
                        records.append(make_record(
                            split,
                            document_index,
                            window_start,
                            window,
                            policy,
                            gap_count,
                            candidate_index,
                            seed,
                            examples[0],
                            tokenizer,
                            document_sha256,
                        ))
    generation = {
        "eligible_documents_before_deduplication": eligible_before,
        "eligible_documents": eligible_after,
        "excluded_exact_duplicates": excluded_duplicates,
        "attempted": dict(sorted(attempted.items())),
        "accepted": dict(sorted(accepted.items())),
        "acceptance_rate": {
            key: accepted[key] / count
            for key, count in sorted(attempted.items())
        },
    }
    return records, generation


def empirical_mutual_information(pairs):
    if not pairs:
        return 0.0
    joint = Counter(pairs)
    left = Counter(pair[0] for pair in pairs)
    right = Counter(pair[1] for pair in pairs)
    total = float(len(pairs))
    return sum(
        count / total * math.log(
            count * total / (left[first] * right[second])
        )
        for (first, second), count in joint.items()
    )


def average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def quantile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    position = int(round((len(values) - 1) * probability))
    return values[position]


def bootstrap_mean_interval(values, seed=7301, samples=2000):
    values = [float(value) for value in values]
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(average(rng.choice(values) for _ in values))
    return [quantile(means, 0.025), quantile(means, 0.975)]


def cluster_bootstrap_mean_interval(clusters, seed=7301, samples=2000):
    clusters = [
        [float(value) for value in cluster] for cluster in clusters if cluster
    ]
    if not clusters:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sampled = [rng.choice(clusters) for _ in clusters]
        means.append(average(
            value for cluster in sampled for value in cluster
        ))
    return [quantile(means, 0.025), quantile(means, 0.975)]


def summarize_records(records, generation):
    grouped = defaultdict(list)
    for record in records:
        grouped["{}|{}|{}".format(
            record["split"], record["policy"], record["gap_count"]
        )].append(record)
    groups = {}
    for key, examples in sorted(grouped.items()):
        gaps = [gap for example in examples for gap in example["gaps"]]
        lengths = Counter(int(gap["length"]) for gap in gaps)
        pairs = [
            (int(example["gaps"][0]["length"]), int(example["gaps"][1]["length"]))
            for example in examples if len(example["gaps"]) == 2
        ]
        total = sum(lengths.values())
        nonempty = [gap for gap in gaps if int(gap["length"]) > 0]
        groups[key] = {
            "examples": len(examples),
            "source_documents": len(set(
                example["source_document_id"] for example in examples
            )),
            "gap_length_count": {
                str(length): count for length, count in sorted(lengths.items())
            },
            "gap_length_probability": {
                str(length): count / total
                for length, count in sorted(lengths.items())
            } if total else {},
            "empty_gap_rate": average(
                int(int(gap["length"]) == 0) for gap in gaps
            ),
            "mean_gap_length": average(int(gap["length"]) for gap in gaps),
            "mean_corruption_ratio": average(
                float(example["corruption_ratio"]) for example in examples
            ),
            "visible_copy_rate_nonempty": average(
                bool(gap["visible_copy"]) for gap in nonempty
            ),
            "wordpiece_start_cut_rate_nonempty": average(
                bool(gap["starts_inside_wordpiece"]) for gap in nonempty
            ),
            "wordpiece_end_cut_rate_nonempty": average(
                bool(gap["ends_inside_wordpiece"]) for gap in nonempty
            ),
            "mean_compatible_actions_nonempty": average(
                int(gap["compatible_action_count"]) for gap in nonempty
            ),
            "two_gap_length_mutual_information_nats": (
                empirical_mutual_information(pairs)
            ),
        }
    source_splits = defaultdict(set)
    window_splits = defaultdict(set)
    for record in records:
        source_splits[
            record.get("source_document_sha256", record["source_document_id"])
        ].add(record["split"])
        window_splits[
            record.get("window_sha256", record["source_document_id"])
        ].add(record["split"])
    source_leakage = sorted(
        source for source, splits in source_splits.items() if len(splits) > 1
    )
    window_leakage = sorted(
        window for window, splits in window_splits.items() if len(splits) > 1
    )
    return {
        "records": len(records),
        "split_leakage_count": len(source_leakage),
        "split_leakage_examples": source_leakage[:20],
        "exact_window_overlap_count": len(window_leakage),
        "exact_window_overlap_examples": window_leakage[:20],
        "generation": generation,
        "groups": groups,
    }


def stratified_sample(records, limit):
    if not limit or limit >= len(records):
        return list(records)
    groups = defaultdict(list)
    for record in records:
        key = "{}|{}|{}".format(
            record["split"], record["policy"], record["gap_count"]
        )
        groups[key].append(record)
    selected = []
    positions = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            position = positions[key]
            if position >= len(groups[key]):
                continue
            selected.append(groups[key][position])
            positions[key] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def record_to_example(record):
    return TextInfillingExample(
        tuple(tuple(segment) for segment in record["segments"]),
        tuple(tuple(span) for span in record["spans"]),
    )


def render_gap_query(example, vocab, target_gap, reveal_other_gaps):
    tokens = [vocab.LEFT]
    target_position = -1
    for gap_index, span in enumerate(example.spans):
        tokens.extend(example.segments[gap_index])
        if gap_index == target_gap or not reveal_other_gaps:
            if gap_index == target_gap:
                target_position = len(tokens)
            tokens.append(vocab.GAP)
        else:
            tokens.extend(span)
    tokens.extend(example.segments[-1])
    tokens.append(vocab.RIGHT)
    if target_position < 0:
        raise AssertionError("target GAP was not rendered")
    return tokens, target_position


def load_model(artifact_dir, tokenizer, vocab, device):
    with (artifact_dir / "results.json").open(encoding="utf-8") as handle:
        config = json.load(handle)["config"]
    model = PretrainedGapFrontierModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        local_files_only=True,
        pretrained_tokenizer=tokenizer,
        detach_structure_encoder=False,
        direct_joint_actions=True,
        zero_joint_interaction=bool(config["zero_joint_interaction"]),
        per_node_frontier_features=bool(
            config.get("per_node_frontier_features", False)
        ),
        attn_implementation=str(config["attention_implementation"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(
        artifact_dir / "frontier.pt", map_location=device, weights_only=True
    ))
    model.eval()
    return model, config


def score_batch(model, queries, vocab, generated_ids, token_map, device):
    width = max(len(query["tokens"]) for query in queries)
    tokens = torch.full(
        (len(queries), width), vocab.PAD, dtype=torch.long, device=device
    )
    padding = torch.ones_like(tokens, dtype=torch.bool)
    for row, query in enumerate(queries):
        sequence = query["tokens"]
        tokens[row, : len(sequence)] = torch.tensor(sequence, device=device)
        padding[row, : len(sequence)] = False
    steps = torch.zeros(len(queries), dtype=torch.long, device=device)
    token_logits, root_stop, degree, direction, hidden = model(
        tokens, padding, steps
    )
    rows = torch.arange(len(queries), device=device)
    positions = torch.tensor(
        [int(query["position"]) for query in queries], device=device
    )
    token_logits = token_logits[rows, positions]
    degree = degree[rows, positions]
    direction = direction[rows, positions]
    hidden = hidden[rows, positions]
    token_logp = token_logits.index_select(
        -1, generated_ids
    ).log_softmax(dim=-1)
    marker_logp = model.marker_log_probs(degree, direction)
    joint_logp = model.joint_action_log_probs(
        token_logits, degree, direction, hidden, steps, generated_ids
    )
    output = []
    for row, query in enumerate(queries):
        span = [int(token) for token in query["span"]]
        stop_logit = root_stop[row, positions[row]]
        result = {
            "stop_probability": float(stop_logit.sigmoid()),
            "stop_nll": float(torch.nn.functional.softplus(
                -stop_logit if not span else stop_logit
            )),
            "token_entropy_nats": float(
                -(token_logp[row].exp() * token_logp[row]).sum()
            ),
            "marker_entropy_nats": float(
                -(marker_logp[row].exp() * marker_logp[row]).sum()
            ),
        }
        if span:
            mapped = token_map[torch.tensor(span, device=device)].tolist()
            if any(index < 0 for index in mapped):
                raise ValueError("target span contains a structural token")
            token_indices = torch.tensor(
                sorted(set(int(index) for index in mapped)), device=device
            )
            action_indices = torch.tensor(sorted(set(
                int(token_map[token]) * 4 + int(marker)
                for token, marker in compatible_root_actions(span)
            )), device=device)
            result["compatible_token_nll"] = float(-torch.logsumexp(
                token_logp[row, token_indices], dim=0
            ))
            result["compatible_joint_nll"] = float(-torch.logsumexp(
                joint_logp[row].flatten()[action_indices], dim=0
            ))
        output.append(result)
    return output


@torch.inference_mode()
def score_records(records, artifact_dir, tokenizer, vocab, device, batch_size, limit):
    model, config = load_model(artifact_dir, tokenizer, vocab, device)
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    token_map = torch.full(
        (vocab.vocab_size,), -1, dtype=torch.long, device=device
    )
    token_map[generated_ids] = torch.arange(generated_ids.numel(), device=device)
    selected = stratified_sample(records, limit)
    queries = []
    for record_index, record in enumerate(selected):
        example = record_to_example(record)
        for gap_index, span in enumerate(example.spans):
            for condition, reveal in (("all_masked", False), ("other_gold", True)):
                canvas, position = render_gap_query(
                    example, vocab, gap_index, reveal
                )
                queries.append({
                    "record_index": record_index,
                    "gap_index": gap_index,
                    "condition": condition,
                    "tokens": canvas,
                    "position": position,
                    "span": span,
                })
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        for query, scores in zip(
            batch, score_batch(
                model, batch, vocab, generated_ids, token_map, device
            )
        ):
            gap = selected[query["record_index"]]["gaps"][query["gap_index"]]
            gap.setdefault("scores", {})[query["condition"]] = scores
        if device.type == "cuda":
            torch.cuda.empty_cache()
    gains = []
    grouped_scores = defaultdict(lambda: defaultdict(list))
    grouped_gain_clusters = defaultdict(list)
    for record in selected:
        record_gains = []
        group = "{}|{}|{}".format(
            record["split"], record["policy"], record["gap_count"]
        )
        for gap in record["gaps"]:
            scores = gap.get("scores", {})
            before = scores.get("all_masked", {})
            after = scores.get("other_gold", {})
            for metric in (
                "compatible_token_nll",
                "compatible_joint_nll",
                "token_entropy_nats",
                "marker_entropy_nats",
                "stop_nll",
            ):
                if metric in before:
                    grouped_scores[group]["all_masked_" + metric].append(
                        float(before[metric])
                    )
            if "compatible_joint_nll" in before and "compatible_joint_nll" in after:
                gain = (
                    before["compatible_joint_nll"]
                    - after["compatible_joint_nll"]
                )
                scores["cross_gap_joint_information_gain_nats"] = gain
                if record["gap_count"] > 1:
                    gains.append(gain)
                    record_gains.append(gain)
                    grouped_scores[group][
                        "cross_gap_joint_information_gain_nats"
                    ].append(gain)
        if record_gains:
            grouped_gain_clusters[group].append(record_gains)
    score_groups = {}
    for group, metrics in sorted(grouped_scores.items()):
        score_groups[group] = {
            metric: {
                "count": len(values),
                "mean": average(values),
                "positive_rate": (
                    average(value > 0.0 for value in values)
                    if metric == "cross_gap_joint_information_gain_nats"
                    else None
                ),
                "median": quantile(values, 0.5),
                "p10": quantile(values, 0.1),
                "p90": quantile(values, 0.9),
                "bootstrap_mean_95pct": (
                    cluster_bootstrap_mean_interval(
                        grouped_gain_clusters[group],
                        stable_seed(7301, group, metric),
                    )
                    if metric == "cross_gap_joint_information_gain_nats"
                    else None
                ),
                "bootstrap_cluster_examples": (
                    len(grouped_gain_clusters[group])
                    if metric == "cross_gap_joint_information_gain_nats"
                    else None
                ),
            }
            for metric, values in sorted(metrics.items())
        }
    memory = {}
    if device.type == "cuda":
        memory = {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2 ** 30,
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "checkpoint": str(artifact_dir),
        "checkpoint_model_name": config["model_name"],
        "examples": len(selected),
        "queries": len(queries),
        "mean_two_gap_cross_gap_joint_information_gain_nats": average(gains),
        "two_gap_information_gain_count": len(gains),
        "groups": score_groups,
        "metric_definition": (
            "all_masked compatible-joint NLL minus NLL after other gold GAPs "
            "are revealed; confidence intervals resample source examples as "
            "clusters; diagnostic only, not a WAIT label"
        ),
        "cuda_memory": memory,
    }


def gpu_snapshot():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        used, total = result.stdout.strip().splitlines()[0].split(",")
        return {"used_mib": int(used), "total_mib": int(total)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="artifacts/wikitext_native_modernbert_base"
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/selective_semantic_branching_ssb2_gold_control",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/selective_semantic_branching_data_audit"
    )
    parser.add_argument("--policies", default="uniform,copy,anchored_copy")
    parser.add_argument("--gap-counts", default="1,2")
    parser.add_argument("--examples-per-document", type=int, default=1)
    parser.add_argument("--max-documents-per-split", type=int, default=4096)
    parser.add_argument("--min-span", type=int, default=1)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--zero-length-probability", type=float, default=0.2)
    parser.add_argument("--window-min", type=int, default=24)
    parser.add_argument("--window-max", type=int, default=96)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--score-examples", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.examples_per_document < 1:
        parser.error("--examples-per-document must be positive")
    if args.window_min < 1 or args.window_max < args.window_min:
        parser.error("invalid window bounds")
    policies = [value for value in args.policies.split(",") if value]
    gap_counts = [int(value) for value in args.gap_counts.split(",") if value]
    if not policies or not gap_counts or min(gap_counts) < 1:
        parser.error("policies and positive gap counts are required")

    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(data_dir), use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus_path = data_dir / "corpus.pt"
    corpus = torch.load(corpus_path, map_location="cpu", weights_only=True)
    records, generation = build_candidate_records(
        corpus, tokenizer, args, policies, gap_counts
    )
    audit = summarize_records(records, generation)
    before = gpu_snapshot() if args.score else None
    scoring = None
    if args.score:
        scoring = score_records(
            records,
            Path(args.artifact_dir),
            tokenizer,
            vocab,
            choose_device(args.device),
            args.score_batch_size,
            args.score_examples,
        )
        audit["scoring"] = scoring

    with (output_dir / "corruption_manifest.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    write_json(output_dir / "corruption_audit.json", audit)
    write_json(output_dir / "run_manifest.json", {
        "schema_version": 1,
        "config": vars(args),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_vocab_size": len(tokenizer),
        "record_count": len(records),
        "gpu_before": before,
        "gpu_after": gpu_snapshot() if args.score else None,
    })
    print(json.dumps({
        "output_dir": str(output_dir),
        "records": len(records),
        "split_leakage_count": audit["split_leakage_count"],
        "excluded_exact_duplicates": len(
            generation["excluded_exact_duplicates"]
        ),
        "scoring": scoring,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
