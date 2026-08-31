"""Quality-first descendant DEFER scores from predicted counterfactuals."""

import torch


def expand_other_gaps(canvas, retained_position, gap_id, actions):
    """Keep one GAP and expand every other GAP with a predicted joint action."""
    result = []
    retained = None
    for position, (token, region) in enumerate(canvas):
        if position == retained_position:
            retained = len(result)
            result.append((token, region))
            continue
        action = actions.get(position)
        if action is None:
            result.append((token, region))
            continue
        predicted_token, marker = action
        if marker in (1, 3):
            result.append((gap_id, region))
        result.append((predicted_token, region))
        if marker in (2, 3):
            result.append((gap_id, region))
    if retained is None:
        raise ValueError("retained position must identify a GAP")
    return result, retained


@torch.inference_mode()
def predicted_defer_expand_scores(
    model,
    canvases,
    gap_mask,
    steps,
    chosen_tokens,
    chosen_markers,
    current_confidence,
    generated_ids,
    gap_id,
    pad_id,
    candidate_batch_size=4,
):
    """Return high scores for GAPs that benefit least from waiting.

    Other GAPs are expanded with their already sampled token/marker actions.
    The retained GAP is rescored after that predicted context is re-encoded.
    `current - future` confidence is the EXPAND score; therefore top-k keeps
    semantically promising WAIT candidates deferred.
    """
    if candidate_batch_size < 1:
        raise ValueError("DEFER candidate batch size must be positive")
    if getattr(model, "per_node_frontier_features", False):
        raise ValueError(
            "quality DEFER lookahead intentionally excludes depth/age features"
        )
    candidates = []
    metadata = []
    for row, canvas in enumerate(canvases):
        positions = gap_mask[row, : len(canvas)].nonzero().flatten().tolist()
        if int(steps[row]) == 0 or len(positions) < 2:
            continue
        actions = {
            position: (
                int(chosen_tokens[row, position]),
                int(chosen_markers[row, position]),
            )
            for position in positions
        }
        for position in positions:
            candidate, retained = expand_other_gaps(
                canvas, position, gap_id, actions
            )
            candidates.append(([token for token, _ in candidate], retained))
            metadata.append((row, position, int(steps[row])))
    scores = current_confidence.clone()
    device = current_confidence.device
    for start in range(0, len(candidates), candidate_batch_size):
        rows = candidates[start : start + candidate_batch_size]
        width = max(len(values) for values, _ in rows)
        tokens = torch.full(
            (len(rows), width), pad_id, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        positions = torch.zeros(len(rows), dtype=torch.long, device=device)
        future_steps = torch.zeros(len(rows), dtype=torch.long, device=device)
        for local, ((values, retained), (_, _, step)) in enumerate(
            zip(rows, metadata[start : start + len(rows)])
        ):
            tokens[local, : len(values)] = torch.tensor(values, device=device)
            padding[local, : len(values)] = False
            positions[local] = retained
            future_steps[local] = step + 1
        token_logits, _, degree, direction, hidden = model(
            tokens, padding, future_steps
        )
        local_rows = torch.arange(len(rows), device=device)
        joint = model.joint_action_log_probs(
            token_logits[local_rows, positions],
            degree[local_rows, positions],
            direction[local_rows, positions],
            hidden[local_rows, positions],
            future_steps,
            generated_ids,
        )
        future_confidence = joint.amax(dim=(1, 2))
        for local, (owner, position, _) in enumerate(
            metadata[start : start + len(rows)]
        ):
            scores[owner, position] = (
                current_confidence[owner, position] - future_confidence[local]
            )
    return scores
