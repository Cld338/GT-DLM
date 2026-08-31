from selective_semantic_branching.defer_lookahead import expand_other_gaps


def test_expand_other_gaps_retains_wait_node_and_uses_predicted_markers():
    canvas = [(1, -1), (9, 0), (5, -1), (9, 0), (2, -1)]
    actions = {1: (20, 3), 3: (21, 1)}
    expanded, retained = expand_other_gaps(
        canvas, retained_position=1, gap_id=9, actions=actions
    )
    assert expanded == [
        (1, -1), (9, 0), (5, -1), (9, 0), (21, 0), (2, -1)
    ]
    assert retained == 1
