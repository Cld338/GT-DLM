"""Runtime support for the screened one-step root candidate ranker."""

import json
import math

import torch


FEATURE_NAMES = (
    "root_token_logp",
    "root_marker_logp",
    "child_count",
    "mean_child_joint_confidence",
    "minimum_child_joint_confidence",
    "mean_child_token_confidence",
    "mean_child_marker_confidence",
    "mean_child_token_entropy",
    "mean_child_marker_entropy",
    "reciprocal_token_rank",
    "prompt_length",
    "marker_leaf",
    "marker_left",
    "marker_right",
    "marker_both",
)


def load_root_lookahead_ranker(path, device):
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    if tuple(result["feature_names"]) != FEATURE_NAMES:
        raise ValueError("root lookahead feature schema does not match")
    ranker = result["ranker"]
    return {
        "mean": torch.tensor(ranker["feature_mean"], device=device),
        "scale": torch.tensor(ranker["feature_scale"], device=device),
        "weights": torch.tensor(ranker["weights"], device=device),
    }


def _pad_rows(rows, pad_id, device):
    width = max(len(row) for row in rows)
    tokens = torch.full(
        (len(rows), width), pad_id, dtype=torch.long, device=device
    )
    padding = torch.ones_like(tokens, dtype=torch.bool)
    for index, row in enumerate(rows):
        tokens[index, : len(row)] = torch.tensor(row, device=device)
        padding[index, : len(row)] = False
    return tokens, padding


def _expand_canvas(canvas, gap_id, token, marker):
    result = []
    found = 0
    for value, region in canvas:
        if value != gap_id:
            result.append((value, region))
            continue
        found += 1
        if marker in (1, 3):
            result.append((gap_id, region))
        result.append((int(token), region))
        if marker in (2, 3):
            result.append((gap_id, region))
    if found != 1:
        raise ValueError("root lookahead currently requires exactly one GAP")
    return result


@torch.inference_mode()
def rerank_root_actions(
    model,
    canvases,
    token_logits,
    degree_logits,
    direction_logits,
    tokens,
    padding,
    generated_ids,
    gap_id,
    pad_id,
    ranker,
    token_k=4,
    candidate_batch_size=4,
    stochastic=False,
    generator=None,
    temperature=1.0,
    cache=None,
):
    """Choose one root token/marker per row after one batched re-encoding."""
    if token_k < 1 or candidate_batch_size < 1:
        raise ValueError("root lookahead sizes must be positive")
    if temperature <= 0.0:
        raise ValueError("root lookahead temperature must be positive")
    gap_mask = tokens.eq(gap_id) & ~padding
    gap_counts = gap_mask.sum(dim=-1)
    if not bool(gap_counts.eq(1).all()):
        raise ValueError("root lookahead currently requires one GAP per row")
    positions = gap_mask.to(torch.long).argmax(dim=-1)
    rows = torch.arange(tokens.size(0), device=tokens.device)
    root_token_logp = token_logits[
        rows, positions
    ].index_select(-1, generated_ids).log_softmax(dim=-1)
    root_marker_logp = model.marker_log_probs(
        degree_logits[rows, positions], direction_logits[rows, positions]
    )
    top_values, top_indices = root_token_logp.topk(token_k, dim=-1)

    keys = [tuple((int(token), int(region)) for token, region in canvas)
            for canvas in canvases]
    unique_rows = []
    unique_by_key = {}
    for row, key in enumerate(keys):
        if cache is not None and key in cache:
            continue
        if key not in unique_by_key:
            unique_by_key[key] = len(unique_rows)
            unique_rows.append(row)

    candidate_canvases = []
    metadata = []
    for unique_owner, owner in enumerate(unique_rows):
        canvas = canvases[owner]
        for rank in range(token_k):
            generated_index = int(top_indices[owner, rank])
            token = int(generated_ids[generated_index])
            for marker in range(4):
                candidate_canvases.append(
                    _expand_canvas(canvas, gap_id, token, marker)
                )
                metadata.append((
                    unique_owner,
                    generated_index,
                    marker,
                    rank + 1,
                    float(top_values[owner, rank]),
                    float(root_marker_logp[owner, marker]),
                    len(canvas),
                ))

    future_features = []
    raw_candidates = [
        [token for token, _ in canvas] for canvas in candidate_canvases
    ]
    for start in range(0, len(raw_candidates), candidate_batch_size):
        raw = raw_candidates[start : start + candidate_batch_size]
        candidate_tokens, candidate_padding = _pad_rows(
            raw, pad_id, tokens.device
        )
        steps = torch.ones(len(raw), dtype=torch.long, device=tokens.device)
        child_token_logits, _, child_degree, child_direction, _ = model(
            candidate_tokens, candidate_padding, steps
        )
        child_token_logp = child_token_logits.index_select(
            -1, generated_ids
        ).log_softmax(dim=-1)
        child_marker_logp = model.marker_log_probs(
            child_degree, child_direction
        )
        child_gap_mask = candidate_tokens.eq(gap_id) & ~candidate_padding
        for row in range(len(raw)):
            child_positions = child_gap_mask[row].nonzero().flatten()
            if not len(child_positions):
                future_features.append((0.0,) * 6)
                continue
            token_logp = child_token_logp[row].index_select(
                0, child_positions
            )
            marker_logp = child_marker_logp[row].index_select(
                0, child_positions
            )
            token_probability = token_logp.exp()
            marker_probability = marker_logp.exp()
            token_confidence = token_probability.amax(dim=-1)
            marker_confidence = marker_probability.amax(dim=-1)
            joint_confidence = token_confidence * marker_confidence
            token_entropy = -(
                token_probability * token_logp
            ).sum(dim=-1) / math.log(token_probability.size(-1))
            marker_entropy = -(
                marker_probability * marker_logp
            ).sum(dim=-1) / math.log(4.0)
            future_features.append((
                float(joint_confidence.mean()),
                float(joint_confidence.min()),
                float(token_confidence.mean()),
                float(marker_confidence.mean()),
                float(token_entropy.mean()),
                float(marker_entropy.mean()),
            ))

    feature_rows = []
    for item, future in zip(metadata, future_features):
        _, _, marker, rank, token_logp, marker_logp, prompt_length = item
        feature_rows.append([
            token_logp,
            marker_logp,
            float(marker in (1, 2)) + 2.0 * float(marker == 3),
            *future,
            1.0 / rank,
            prompt_length / 100.0,
            *[float(marker == value) for value in range(4)],
        ])
    if feature_rows:
        features = torch.tensor(feature_rows, device=tokens.device).view(
            len(unique_rows), token_k * 4, -1
        )
        unique_scores = (
            (features - ranker["mean"]) / ranker["scale"]
        ).matmul(ranker["weights"])
        for unique_owner, owner in enumerate(unique_rows):
            value = (
                top_indices[owner].detach(), unique_scores[unique_owner].detach()
            )
            if cache is not None:
                cache[keys[owner]] = value
            unique_by_key[keys[owner]] = value
    selected_top_indices = []
    selected_scores = []
    for row, key in enumerate(keys):
        if cache is not None and key in cache:
            cached_top, cached_scores = cache[key]
        else:
            cached_top, cached_scores = unique_by_key[key]
        selected_top_indices.append(cached_top)
        selected_scores.append(cached_scores)
    top_indices = torch.stack(selected_top_indices)
    scores = torch.stack(selected_scores)
    if stochastic:
        selected = torch.multinomial(
            (scores / temperature).softmax(dim=-1),
            1,
            generator=generator,
        ).squeeze(-1)
    else:
        selected = scores.argmax(dim=-1)
    token_ranks = torch.div(selected, 4, rounding_mode="floor")
    markers = selected.remainder(4)
    chosen_generated = top_indices[rows, token_ranks]
    return generated_ids[chosen_generated], markers, scores
