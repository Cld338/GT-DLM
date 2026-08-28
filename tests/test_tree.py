import math
import random
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from analyze_topology_exposure import tuple_dependence
from adapt_multigap_proper_mle import masked_batch_logp, sequential_batch_logp
from analyze_shared_regime import conditional_metrics
from evaluate_text_sampling import (
    calibrated_topology_logits,
    distribution_metrics,
    sequential_uniform_state_optimum,
)
from experiment_text_joint_topology import (
    alternating_frontier_mask,
    frontier_stage_mask,
)
from experiment_inside_objective import brute_tree_scores
from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sequence_likelihoods import (
    masked_log_likelihoods,
    sequential_log_likelihoods,
)
from pretrain_depth_lexical import (
    lexical_batch_log_probabilities,
    midpoint_node_records,
)
from experiment_text_depth_inside import (
    depth_batch_log_likelihoods,
    reachable_depth_intervals,
)
from experiment_text_depth_inside_multigap import (
    collate_multi_prompt_contexts,
    multi_depth_gap_log_likelihoods,
)
from experiment_text_depth_inside_shared_latent import (
    SharedLatentDepthInsideModel,
    shared_latent_log_likelihoods,
)
from experiment_text_depth_inside_lowrank_latent import (
    LowRankHeadSharedLatentModel,
    lowrank_shared_latent_log_likelihoods,
)
from experiment_text_inside import batch_log_likelihoods
from experiment_text_inside import late_depth_topology_logits
from measure_span_identifiability import marginal_length_entropy
from measure_pretrained_span_identifiability import (
    identifiable_statistics,
    render_masked_text,
    unique_token_positions,
)
from measure_twin_intervention import paired_statistics
from evaluate_multigap_sampling import multigap_distribution_metrics
from evaluate_multigap_sampling import (
    bootstrap_target_length_covariance,
)
from gtdlm.text_data import (
    anchored_repeat_pairs,
    spans_remain_recoverable,
)
from gtdlm.data import (
    MultiGapProposalDataset,
    RangeVocabulary,
    build_strict_multi_gap_partition,
    build_strict_multi_gap_split,
    collate_compact_frontiers,
    collate_multi_triples,
    typed_multi_gap_signatures,
)
from gtdlm.model import (
    GapTreeBlockConditionalTopologyBoundaryModel,
    GapTreeFactorizedBoundaryModel,
    GapTreeCoupledFrontierBoundaryModel,
    GapTreeJointTopologyBoundaryModel,
    GapTreeRefinedTopologyBoundaryModel,
    GapTreeSharedRegimeBoundaryModel,
    GapTreeSymmetricBlockConditionalTopologyBoundaryModel,
    GapTreeThreeStageTopologyBoundaryModel,
    IntervalInsideBoundaryModel,
    LengthMaskedModel,
    PretrainedIntervalInsideModel,
    immediate_gap_boundaries,
)
from gtdlm.inside import (
    batched_inside_log_partition,
    batched_depth_inside_log_partition,
    batched_depth_midpoint_tree_log_weight,
    compatible_pivots,
    depth_inside_log_partition,
    depth_midpoint_tree_log_weight,
    inside_log_partition,
    midpoint_tree_log_weight,
    pivot_topology,
)
from gtdlm.text_data import (
    DynamicSequentialTextDataset,
    DynamicRegimeTreeTextDataset,
    DynamicTextExampleDataset,
    DynamicTreeTextDataset,
    TextGapProposalDataset,
    TextInfillingExample,
    TextVocabulary,
    collate_text_infilling,
    corrupt_token_sequence,
    sample_text_infilling_examples,
    random_length_windows,
    span_length_regime,
)
from gtdlm.text_tokenizer import (
    SPECIAL_TOKENS,
    split_documents,
    train_bpe_tokenizer,
    vocabulary_from_tokenizer,
)
from gtdlm.tree import (
    all_frontiers,
    all_tree_frontiers,
    build_pivot_tree,
    make_compact_frontier,
    make_frontier,
    make_tree_frontier,
    oracle_compact_rounds,
    oracle_parallel_rounds,
    pivot_tree_depth,
)


class FrontierTest(unittest.TestCase):
    def test_proper_text_baselines_score_every_gap(self):
        vocab = TextVocabulary(
            vocab_size=12, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4
        )
        example = TextInfillingExample(
            segments=((5,), (6,), (7,)), spans=((8, 9), (10,))
        )
        sequential = GapTreeFactorizedBoundaryModel(
            vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
            d_model=24, nhead=4, layers=1, max_positions=32, max_steps=8,
            dropout=0.0,
        )
        sequential_values = sequential_log_likelihoods(
            sequential, [example], vocab, torch.device("cpu"), 4
        )
        self.assertEqual(tuple(sequential_values.shape), (1,))
        self.assertTrue(torch.isfinite(sequential_values).all())
        sequential_train_values = sequential_batch_logp(
            sequential, [example], vocab, torch.device("cpu")
        )
        self.assertTrue(torch.allclose(
            sequential_values, sequential_train_values, atol=1e-6
        ))

        masked = LengthMaskedModel(
            vocab.vocab_size, 16, d_model=24, nhead=4, layers=1,
            max_positions=32, dropout=0.0,
        )
        totals, lengths, tokens = masked_log_likelihoods(
            masked, [example], vocab, torch.device("cpu"), 4
        )
        self.assertTrue(torch.isfinite(totals).all())
        self.assertTrue(torch.allclose(totals, lengths + tokens, atol=1e-6))
        batch = collate_text_infilling([example], vocab)
        masked_train_values = masked_batch_logp(
            masked, batch, vocab, torch.device("cpu")
        )
        self.assertTrue(torch.allclose(totals, masked_train_values, atol=1e-6))

    def test_multigap_inside_matches_one_gap_and_aggregates_gap_likelihoods(self):
        vocab = TextVocabulary(
            vocab_size=12, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4
        )
        model = IntervalInsideBoundaryModel(
            vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
            d_model=24, nhead=4, layers=1, max_positions=32, max_steps=8,
            dropout=0.0,
        )
        model.eval()
        one_gap = [
            TextInfillingExample(segments=((5, 6), (7,)), spans=((8, 9, 10),)),
            TextInfillingExample(segments=((6,), (8, 7)), spans=((),)),
        ]
        expected_exact, expected_midpoint = depth_batch_log_likelihoods(
            model, one_gap, vocab, torch.device("cpu"), 4, 0.0
        )
        exact, midpoint, gap_exact = multi_depth_gap_log_likelihoods(
            model, one_gap, vocab, torch.device("cpu")
        )
        self.assertTrue(torch.allclose(exact, expected_exact, atol=1e-6))
        self.assertTrue(torch.allclose(midpoint, expected_midpoint, atol=1e-6))
        self.assertTrue(torch.allclose(exact, gap_exact, atol=1e-6))

        two_gap = [
            TextInfillingExample(
                segments=((5,), (6,), (7,)), spans=((8, 9), (10,))
            )
        ]
        joint, _, individual = multi_depth_gap_log_likelihoods(
            model, two_gap, vocab, torch.device("cpu")
        )
        self.assertEqual(tuple(individual.shape), (2,))
        self.assertTrue(torch.allclose(joint[0], individual.sum(), atol=1e-6))
        (-joint.mean()).backward()
        self.assertGreater(float(model.token_head.weight.grad.abs().sum()), 0.0)

    def test_shared_latent_exactly_marginalizes_and_nests_factorized_model(self):
        vocab = TextVocabulary(
            vocab_size=12, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4
        )
        base = IntervalInsideBoundaryModel(
            vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
            d_model=24, nhead=4, layers=1, max_positions=32, max_steps=8,
            dropout=0.0,
        )
        examples = [TextInfillingExample(
            segments=((5,), (6,), (7,)), spans=((8, 9), (10,))
        )]
        factorized, _, _ = multi_depth_gap_log_likelihoods(
            base, examples, vocab, torch.device("cpu")
        )
        mixture = SharedLatentDepthInsideModel(base, regimes=3, offset_std=0.0)
        nested, _, _, _, _ = shared_latent_log_likelihoods(
            mixture, examples, vocab, torch.device("cpu")
        )
        self.assertTrue(torch.allclose(nested, factorized, atol=1e-6))

        with torch.no_grad():
            mixture.regime_offsets.normal_(std=0.1)
            mixture.regime_head.weight.normal_(std=0.1)
        joint, _, _, gate_logp, posterior = shared_latent_log_likelihoods(
            mixture, examples, vocab, torch.device("cpu")
        )
        tokens, padding, _ = collate_multi_prompt_contexts(
            examples, vocab, torch.device("cpu")
        )
        encoded = mixture.base.encode(tokens, padding)
        components = []
        for regime in range(mixture.regimes):
            offsets = mixture.regime_offsets[regime].unsqueeze(0)
            value, _, _ = multi_depth_gap_log_likelihoods(
                mixture.base, examples, vocab, torch.device("cpu"),
                encoded=encoded, context_offsets=offsets,
            )
            components.append(value)
        expected = torch.logsumexp(
            gate_logp + torch.stack(components, dim=-1), dim=-1
        )
        self.assertTrue(torch.allclose(joint, expected, atol=1e-6))
        self.assertTrue(torch.allclose(
            posterior.sum(-1), torch.ones(1), atol=1e-6
        ))
        (-joint.mean()).backward()
        self.assertGreater(float(mixture.regime_offsets.grad.abs().sum()), 0.0)
        self.assertGreater(float(mixture.regime_head.weight.grad.abs().sum()), 0.0)

    def test_lowrank_shared_latent_starts_nested_and_trains_component_heads(self):
        vocab = TextVocabulary(
            vocab_size=12, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4
        )
        base = IntervalInsideBoundaryModel(
            vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
            d_model=24, nhead=4, layers=1, max_positions=32, max_steps=8,
            dropout=0.0,
        )
        examples = [TextInfillingExample(
            segments=((5,), (6,), (7,)), spans=((8, 9), (10,))
        )]
        factorized, _, _ = multi_depth_gap_log_likelihoods(
            base, examples, vocab, torch.device("cpu")
        )
        mixture = LowRankHeadSharedLatentModel(base, regimes=2, rank=3)
        nested, _, _, _, _ = lowrank_shared_latent_log_likelihoods(
            mixture, examples, vocab, torch.device("cpu")
        )
        self.assertTrue(torch.allclose(nested, factorized, atol=1e-6))
        with torch.no_grad():
            mixture.token_up.normal_(std=0.05)
            mixture.stop_up.normal_(std=0.05)
            mixture.topology_up.normal_(std=0.05)
            mixture.regime_head.weight.normal_(std=0.05)
        joint, _, _, _, posterior = lowrank_shared_latent_log_likelihoods(
            mixture, examples, vocab, torch.device("cpu")
        )
        self.assertTrue(torch.allclose(posterior.sum(-1), torch.ones(1), atol=1e-6))
        (-joint.mean()).backward()
        self.assertGreater(float(mixture.token_up.grad.abs().sum()), 0.0)
        self.assertGreater(float(mixture.stop_up.grad.abs().sum()), 0.0)
        self.assertGreater(float(mixture.topology_up.grad.abs().sum()), 0.0)
        self.assertGreater(float(mixture.regime_head.weight.grad.abs().sum()), 0.0)

    def test_midpoint_lexical_pretraining_covers_tokens_and_backpropagates(self):
        for length in range(1, 9):
            records = midpoint_node_records(length)
            self.assertEqual(len(records), length)
            self.assertEqual(sorted(record[3] for record in records), list(range(length)))

        vocab = TextVocabulary(
            vocab_size=12, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4
        )
        examples = [
            TextInfillingExample(segments=((5, 6), (7,)), spans=((8, 9, 10, 11),)),
            TextInfillingExample(segments=((6,), (8, 7)), spans=((9, 10),)),
        ]
        model = IntervalInsideBoundaryModel(
            vocab_size=vocab.vocab_size,
            gap_id=vocab.GAP,
            pad_id=vocab.PAD,
            d_model=24,
            nhead=4,
            layers=1,
            max_positions=32,
            max_steps=8,
            dropout=0.0,
        )
        values = lexical_batch_log_probabilities(
            model, examples, vocab, torch.device("cpu")
        )
        self.assertEqual(tuple(values.shape), (6,))
        self.assertTrue(torch.isfinite(values).all())
        (-values.mean()).backward()
        self.assertGreater(float(model.token_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(model.encoder.step_embedding.weight.grad.abs().sum()), 0.0
        )

    def test_inside_dp_matches_all_ordered_trees_and_has_posterior_gradients(self):
        weights = torch.randn(5, 5, 4, requires_grad=True)
        exact = inside_log_partition(weights)
        brute = torch.logsumexp(torch.stack(brute_tree_scores(weights)), dim=0)
        midpoint = midpoint_tree_log_weight(weights)
        self.assertTrue(torch.allclose(exact, brute, atol=1e-6))
        self.assertGreaterEqual(float(exact), float(midpoint))
        exact.backward()
        # Every four-token tree contains exactly four pivot actions, so the
        # posterior expected count summed over all local actions is four.
        self.assertAlmostEqual(float(weights.grad.sum()), 4.0, places=5)

    def test_batched_inside_matches_individual_charts(self):
        weights = torch.randn(3, 5, 5, 4)
        batched = batched_inside_log_partition(weights)
        individual = torch.stack([
            inside_log_partition(chart) for chart in weights
        ])
        self.assertTrue(torch.allclose(batched, individual, atol=1e-6))

    def test_depth_inside_matches_depth_annotated_enumeration(self):
        weights = torch.randn(4, 5, 5, 4, requires_grad=True)

        def enumerate_scores(lo, hi, depth):
            if lo >= hi:
                return [weights.new_zeros(())]
            result = []
            for pivot in range(lo, hi):
                left = enumerate_scores(lo, pivot, depth + 1)
                right = enumerate_scores(pivot + 1, hi, depth + 1)
                for left_score in left:
                    for right_score in right:
                        result.append(
                            weights[depth, lo, hi, pivot]
                            + left_score + right_score
                        )
            return result

        exact = depth_inside_log_partition(weights)
        brute = torch.logsumexp(
            torch.stack(enumerate_scores(0, 4, 0)), dim=0
        )
        self.assertTrue(torch.allclose(exact, brute, atol=1e-6))
        exact.backward()
        self.assertAlmostEqual(float(weights.grad.sum()), 4.0, places=5)
        midpoint = depth_midpoint_tree_log_weight(weights.detach())
        self.assertGreaterEqual(float(exact.detach()), float(midpoint))

    def test_batched_depth_inside_matches_individual_charts_and_gradients(self):
        weights = torch.randn(3, 4, 5, 5, 4, requires_grad=True)
        batched = batched_depth_inside_log_partition(weights)
        individual = torch.stack([
            depth_inside_log_partition(chart) for chart in weights
        ])
        self.assertTrue(torch.allclose(batched, individual, atol=1e-6))
        batched_midpoint = batched_depth_midpoint_tree_log_weight(weights)
        individual_midpoint = torch.stack([
            depth_midpoint_tree_log_weight(chart) for chart in weights
        ])
        self.assertTrue(torch.allclose(
            batched_midpoint, individual_midpoint, atol=1e-6
        ))
        batched.sum().backward()
        self.assertAlmostEqual(float(weights.grad.sum()), 12.0, places=4)

    def test_late_depth_penalty_grows_with_child_count(self):
        logits = torch.zeros(3, 4)
        adjusted = late_depth_topology_logits(
            logits, torch.tensor([3, 4, 6]), start_depth=4, child_penalty=0.5
        )
        self.assertTrue(torch.equal(adjusted[0], torch.zeros(4)))
        self.assertTrue(torch.allclose(
            adjusted[1], torch.tensor([0.0, -0.5, -0.5, -1.0])
        ))
        self.assertTrue(torch.allclose(
            adjusted[2], torch.tensor([0.0, -1.5, -1.5, -3.0])
        ))

    def test_reachable_depth_intervals_start_at_one_root(self):
        self.assertEqual(reachable_depth_intervals(1), [(0, 0, 1)])
        self.assertEqual(
            set(reachable_depth_intervals(2)),
            {(0, 0, 2), (1, 0, 1), (1, 1, 2)},
        )

    def test_depth_text_likelihood_is_differentiable(self):
        vocab = TextVocabulary(40, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        example = corrupt_token_sequence(
            [10, 11, 12, 13, 14, 15], [(2, 5)]
        )
        model = IntervalInsideBoundaryModel(
            vocab_size=40, gap_id=vocab.GAP, pad_id=vocab.PAD,
            d_model=16, nhead=4, layers=1, max_positions=16,
        )
        exact, midpoint = depth_batch_log_likelihoods(
            model, [example], vocab, torch.device("cpu"), 2, 0.5
        )
        self.assertGreaterEqual(float(exact[0]), float(midpoint[0]))
        (-exact.mean()).backward()
        self.assertIsNotNone(model.encoder.step_embedding.weight.grad)

    def test_pretrained_context_depth_likelihood_is_differentiable(self):
        class StubSourceTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "".join(chr(65 + token_id) for token_id in token_ids)

        class StubPretrainedTokenizer:
            mask_token = "<mask>"
            mask_token_id = 9

            def __call__(self, texts, **kwargs):
                del kwargs
                rows = [[1, self.mask_token_id, 2] for _ in texts]
                return {
                    "input_ids": torch.tensor(rows),
                    "attention_mask": torch.ones(len(rows), 3, dtype=torch.long),
                }

        class StubBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=8)
                self.embedding = torch.nn.Embedding(16, 8)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, attention_mask):
                del attention_mask
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        vocab = TextVocabulary(12, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        example = corrupt_token_sequence([5, 6, 7, 8, 9], [(1, 3)])
        backbone = StubBackbone()
        model = PretrainedIntervalInsideModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            StubSourceTokenizer(),
            backbone=backbone,
            pretrained_tokenizer=StubPretrainedTokenizer(),
            initialize_custom_embeddings=False,
        )
        exact, midpoint = depth_batch_log_likelihoods(
            model, [example], vocab, torch.device("cpu"), 4, 0.0
        )
        self.assertTrue(torch.isfinite(exact).all())
        self.assertGreaterEqual(float(exact[0]), float(midpoint[0]))
        (-exact.mean()).backward()
        self.assertIsNotNone(backbone.embedding.weight.grad)
        self.assertIsNotNone(model.encoder.step_embedding.weight.grad)

    def test_lexical_metrics_condition_on_nonempty_length_matches(self):
        examples = [
            corrupt_token_sequence([10, 11, 12, 13], [(1, 3)]),
            corrupt_token_sequence([20, 21, 22], [(1, 1)]),
        ]
        samples = [
            [[11, 12], [11, 13], [11]],
            [[], [21], []],
        ]
        unfinished = [[False, False, False], [False, False, True]]
        metrics = lexical_sampling_metrics(examples, samples, unfinished)
        self.assertAlmostEqual(metrics["length_match_probability"], 0.5)
        self.assertEqual(metrics["matched_nonempty_pairs"], 2.0)
        self.assertAlmostEqual(metrics["matched_length_token_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["matched_length_exact_probability"], 0.5)
        self.assertAlmostEqual(metrics["unfinished_rate"], 1 / 6)

    def test_pivot_topology_compatibility(self):
        self.assertEqual(pivot_topology(0, 1, 0), 0)
        self.assertEqual(pivot_topology(0, 3, 0), 2)
        self.assertEqual(pivot_topology(0, 3, 1), 3)
        self.assertEqual(pivot_topology(0, 3, 2), 1)
        self.assertEqual(compatible_pivots(1, 0), [0])
        self.assertEqual(compatible_pivots(3, 3), [1])
        self.assertEqual(compatible_pivots(4, 3), [1, 2])

    def test_natural_text_inside_likelihood_is_differentiable(self):
        vocab = TextVocabulary(40, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        example = corrupt_token_sequence([10, 11, 12, 13, 14, 15], [(2, 5)])
        model = IntervalInsideBoundaryModel(
            vocab_size=40,
            gap_id=vocab.GAP,
            pad_id=vocab.PAD,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
        )
        exact, midpoint = batch_log_likelihoods(
            model, [example], vocab, torch.device("cpu")
        )
        self.assertGreaterEqual(float(exact[0]), float(midpoint[0]))
        (-exact.mean()).backward()
        self.assertIsNotNone(model.interval_projection.weight.grad)

    def test_inside_stop_probability_is_a_single_root_gate(self):
        vocab = TextVocabulary(40, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        nonempty = corrupt_token_sequence(
            [10, 11, 12, 13, 14, 15], [(2, 5)]
        )
        empty = corrupt_token_sequence([10, 11, 12, 13], [(2, 2)])
        model = IntervalInsideBoundaryModel(
            vocab_size=40,
            gap_id=vocab.GAP,
            pad_id=vocab.PAD,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
        )
        model.eval()
        with torch.no_grad():
            model.stop_head.weight.zero_()
            model.stop_head.bias.fill_(-0.7)
        first, _ = batch_log_likelihoods(
            model, [nonempty, empty], vocab, torch.device("cpu")
        )
        with torch.no_grad():
            model.stop_head.bias.fill_(0.4)
        second, _ = batch_log_likelihoods(
            model, [nonempty, empty], vocab, torch.device("cpu")
        )
        expected_nonempty = F.logsigmoid(torch.tensor(-0.4)) - F.logsigmoid(
            torch.tensor(0.7)
        )
        expected_empty = F.logsigmoid(torch.tensor(0.4)) - F.logsigmoid(
            torch.tensor(-0.7)
        )
        self.assertTrue(torch.allclose(second[0] - first[0], expected_nonempty))
        self.assertTrue(torch.allclose(second[1] - first[1], expected_empty))

    def test_inside_likelihood_excludes_structural_token_logits(self):
        vocab = TextVocabulary(40, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        example = corrupt_token_sequence(
            [10, 11, 12, 13, 14, 15], [(2, 5)]
        )
        model = IntervalInsideBoundaryModel(
            vocab_size=40,
            gap_id=vocab.GAP,
            pad_id=vocab.PAD,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
        )
        model.eval()
        first, _ = batch_log_likelihoods(
            model, [example], vocab, torch.device("cpu")
        )
        with torch.no_grad():
            model.token_head.bias[list(vocab.structural_ids)] += 100.0
        second, _ = batch_log_likelihoods(
            model, [example], vocab, torch.device("cpu")
        )
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_length_distribution_metrics_recognize_the_known_prior(self):
        lengths = [0, 0] + list(range(1, 9))
        examples = [
            corrupt_token_sequence(list(range(length + 4)), [(2, 2 + length)])
            for length in lengths
        ]
        prior = [0.2] + [0.1] * 8 + [0.0]
        metrics = distribution_metrics(examples, [prior] * len(examples))
        self.assertAlmostEqual(metrics["marginal_tv_to_prior"], 0.0)
        self.assertAlmostEqual(metrics["observed_target_match_probability"], 0.12)
        self.assertAlmostEqual(metrics["conditional_brier"], 0.88)

    def test_topology_temperature_and_class_bias_are_applied(self):
        logits = torch.tensor([[0.0, 2.0, 4.0, 6.0]])
        calibrated = calibrated_topology_logits(
            logits, 2.0, [0.3, 0.1, -0.1, -0.3]
        )
        self.assertTrue(torch.allclose(
            calibrated, torch.tensor([[0.3, 1.1, 1.9, 2.7]])
        ))

    def test_uniform_state_objective_reweights_toward_short_lengths(self):
        distribution = sequential_uniform_state_optimum([0.2] + [0.1] * 8)
        self.assertAlmostEqual(sum(distribution), 1.0)
        self.assertAlmostEqual(distribution[0], 0.5223337, places=6)
        self.assertGreater(distribution[0], 0.2)

    def test_midpoint_frontier_targets_have_cross_gap_dependence(self):
        vocab = TextVocabulary(80, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        examples = [
            corrupt_token_sequence(
                list(range(10, 10 + length + 4)), [(2, 2 + length)]
            )
            for length in range(3, 9)
        ]
        dataset = TextGapProposalDataset(
            examples, vocab, strategy="midpoint", seed=17
        )
        rows = tuple_dependence(dataset)
        depth_one = next(
            row for row in rows
            if row["depth"] == 1 and row["frontier_width"] == 2
        )
        self.assertAlmostEqual(depth_one["total_correlation_nats"], 0.549306, places=5)

    def test_regime_calibration_renormalizes_after_empty(self):
        probabilities = [[0.2, 0.4, 0.4] + [0.0] * 7]
        metrics = conditional_metrics(probabilities, regime=0)
        self.assertAlmostEqual(metrics["conditional_tv"], 0.0)
        self.assertAlmostEqual(metrics["bucket_adherence"], 1.0)

    def test_midpoint_frontiers(self):
        values = [10, 11, 12]
        gap = 1
        stop = 99

        canvas0, targets0 = make_frontier(values, 0, gap, stop)
        self.assertEqual(canvas0, [gap])
        self.assertEqual(targets0, [11])

        canvas1, targets1 = make_frontier(values, 1, gap, stop)
        self.assertEqual(canvas1, [gap, 11, gap])
        self.assertEqual(targets1, [10, -100, 12])

        canvas2, targets2 = make_frontier(values, 2, gap, stop)
        self.assertEqual(canvas2, [gap, 10, gap, 11, gap, 12, gap])
        self.assertEqual(targets2, [stop, -100, stop, -100, stop, -100, stop])

    def test_empty_span_closes(self):
        frontiers = all_frontiers([], gap_id=1, stop_action=9)
        self.assertEqual(frontiers, [([1], [9], 0)])

    def test_round_growth_is_logarithmic(self):
        for length in range(1, 65):
            rounds = oracle_parallel_rounds(length)
            upper_bound = math.ceil(math.log2(length + 1)) + 2
            self.assertLessEqual(rounds, upper_bound)

    def test_compact_frontier_predicts_children(self):
        values = [10, 11, 12]
        canvas0, actions0, left0, right0 = make_compact_frontier(
            values, 0, gap_id=1, stop_action=99
        )
        self.assertEqual(canvas0, [1])
        self.assertEqual(actions0, [11])
        self.assertEqual(left0, [1])
        self.assertEqual(right0, [1])

        canvas1, actions1, left1, right1 = make_compact_frontier(
            values, 1, gap_id=1, stop_action=99
        )
        self.assertEqual(canvas1, [1, 11, 1])
        self.assertEqual(actions1, [10, -100, 12])
        self.assertEqual(left1, [0, -100, 0])
        self.assertEqual(right1, [0, -100, 0])

    def test_compact_saves_closure_round(self):
        for length in range(1, 65):
            self.assertLess(
                oracle_compact_rounds(length), oracle_parallel_rounds(length)
            )

    def test_immediate_gap_boundaries(self):
        tokens = torch.tensor([[3, 8, 1, 12, 4, 0]])
        left, right, gaps = immediate_gap_boundaries(tokens, gap_id=1, pad_id=0)
        self.assertEqual(left.tolist(), [[0, 0, 8, 0, 0, 0]])
        self.assertEqual(right.tolist(), [[0, 0, 12, 0, 0, 0]])
        self.assertEqual(gaps.tolist(), [[False, False, True, False, False, False]])

    def test_uniform_tree_preserves_inorder_values(self):
        import random

        values = [10, 11, 12, 13, 14]
        tree = build_pivot_tree(
            0, len(values), strategy="uniform", rng=random.Random(7)
        )
        self.assertIsNotNone(tree)
        frontiers = all_tree_frontiers(values, tree, gap_id=1, stop_action=99)
        final_depth = pivot_tree_depth(tree) - 1
        canvas, _, _, _ = make_tree_frontier(
            values, tree, final_depth, gap_id=1, stop_action=99
        )
        visible = [token for token in canvas if token != 1]
        frontier_actions = [
            action
            for action in frontiers[-1][1]
            if action not in {-100, 99}
        ]
        reconstructed = sorted(visible + frontier_actions, key=values.index)
        self.assertEqual(reconstructed, values)

    def test_multi_gap_frontier_has_two_initial_actions(self):
        vocab = RangeVocabulary(12)
        dataset = MultiGapProposalDataset(
            [(2, 5, 9)],
            vocab,
            strategy="midpoint",
            seed=7,
            trees_per_example=1,
        )
        first = dataset[0]
        gap_positions = [
            index for index, token in enumerate(first["tokens"]) if token == vocab.GAP
        ]
        self.assertEqual(len(gap_positions), 2)
        self.assertEqual(
            [first["targets"][index] for index in gap_positions],
            [vocab.value(3), vocab.value(7)],
        )

    def test_multi_gap_mask_collation(self):
        vocab = RangeVocabulary(12)
        batch = collate_multi_triples([(2, 5, 9)], vocab)
        self.assertEqual(batch["length_targets"][0, 2].item(), 3)
        self.assertEqual(batch["length_targets"][0, 4].item(), 3)
        predicted = vocab.decode_values(
            [token for token in batch["token_targets"][0].tolist() if token >= 0]
        )
        self.assertEqual(predicted, [2, 3, 4, 6, 7, 8])

    def test_strict_multi_gap_split_has_no_typed_interval_leakage(self):
        train, test, heldout = build_strict_multi_gap_split(24, 12, seed=17)
        train_signatures = {
            signature
            for triple in train
            for signature in typed_multi_gap_signatures(triple)
        }
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertTrue(heldout)
        self.assertTrue(heldout.isdisjoint(train_signatures))
        self.assertTrue(
            all(
                any(signature in heldout for signature in typed_multi_gap_signatures(triple))
                for triple in test
            )
        )
        self.assertTrue(set(train).isdisjoint(test))

    def test_strict_three_way_partition_has_no_typed_interval_leakage(self):
        train, validation, test, validation_heldout, test_heldout = (
            build_strict_multi_gap_partition(24, 12, seed=17)
        )
        train_signatures = {
            signature
            for triple in train
            for signature in typed_multi_gap_signatures(triple)
        }
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(test)
        self.assertTrue(validation_heldout.isdisjoint(train_signatures))
        self.assertTrue(test_heldout.isdisjoint(train_signatures))
        self.assertTrue(
            all(
                any(signature in validation_heldout for signature in typed_multi_gap_signatures(triple))
                for triple in validation
            )
        )
        self.assertTrue(
            all(
                any(signature in test_heldout for signature in typed_multi_gap_signatures(triple))
                for triple in test
            )
        )
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(set(train).isdisjoint(test))
        self.assertTrue(set(validation).isdisjoint(test))

    def test_text_infilling_round_trip_and_collation(self):
        vocab = TextVocabulary(40, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        tokens = list(range(10, 20))
        example = corrupt_token_sequence(tokens, [(2, 4), (6, 7)])
        self.assertEqual(example.reconstruct(), tokens)
        self.assertEqual(example.spans, ((12, 13), (16,)))
        self.assertEqual(example.segments, ((10, 11), (14, 15), (17, 18, 19)))
        batch = collate_text_infilling([example], vocab)
        lengths = [
            value for value in batch["length_targets"][0].tolist() if value >= 0
        ]
        targets = [
            value for value in batch["token_targets"][0].tolist() if value >= 0
        ]
        self.assertEqual(lengths, [2, 1])
        self.assertEqual(targets, [12, 13, 16])

    def test_text_tree_frontier_preserves_multiple_context_segments(self):
        vocab = TextVocabulary(40, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        example = corrupt_token_sequence(list(range(10, 20)), [(2, 4), (6, 7)])
        dataset = TextGapProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        first = dataset[0]
        gap_positions = [
            index for index, token in enumerate(first["tokens"]) if token == vocab.GAP
        ]
        self.assertEqual(len(gap_positions), 2)
        self.assertEqual(
            [first["targets"][index] for index in gap_positions], [13, 16]
        )
        immutable = [10, 11, 14, 15, 17, 18, 19]
        self.assertTrue(all(token in first["tokens"] for token in immutable))

    def test_text_corruption_sampler_is_reproducible(self):
        documents = [list(range(10, 30)), list(range(30, 50))]
        first = sample_text_infilling_examples(documents, seed=7, examples_per_document=3)
        second = sample_text_infilling_examples(documents, seed=7, examples_per_document=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertTrue(all(example.reconstruct() in documents for example in first))

    def test_text_tokenizer_has_stable_structural_ids(self):
        documents = [
            "A small story about a red fox.",
            "The blue bird flew over the quiet river.",
            "A child found a bright green stone.",
        ]
        split = split_documents(documents, seed=17, train_fraction=2 / 3, validation_fraction=0)
        tokenizer = train_bpe_tokenizer(split["train"], vocab_size=64)
        vocab = vocabulary_from_tokenizer(tokenizer)
        self.assertEqual(
            [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS],
            list(range(len(SPECIAL_TOKENS))),
        )
        self.assertEqual(vocab.structural_ids, (0, 1, 2, 3, 4))
        self.assertEqual(sum(len(value) for value in split.values()), 3)

    def test_factorized_gap_model_separates_stop_and_token_logits(self):
        model = GapTreeFactorizedBoundaryModel(
            vocab_size=40,
            gap_id=1,
            pad_id=0,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
        )
        tokens = torch.tensor([[3, 10, 1, 11, 4]])
        token_logits, stop_logits, hidden = model(tokens)
        children = model.predict_children(hidden, torch.zeros_like(tokens))
        self.assertEqual(token_logits.shape, (1, 5, 40))
        self.assertEqual(stop_logits.shape, (1, 5))
        self.assertEqual(children.shape, (1, 5, 2))

    def test_joint_topology_model_predicts_four_correlated_states(self):
        model = GapTreeJointTopologyBoundaryModel(
            vocab_size=40,
            gap_id=1,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
        )
        tokens = torch.tensor([[3, 10, 1, 11, 4]])
        token_logits, stop_logits, hidden = model(tokens)
        topology = model.predict_topology(hidden, torch.zeros_like(tokens))
        self.assertEqual(token_logits.shape, (1, 5, 40))
        self.assertEqual(stop_logits.shape, (1, 5))
        self.assertEqual(topology.shape, (1, 5, 4))

    def test_coupled_frontier_model_predicts_sixteen_pairs(self):
        model = GapTreeCoupledFrontierBoundaryModel(
            vocab_size=40,
            gap_id=1,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
        )
        hidden_pairs = torch.randn(3, 2, 16)
        token_pairs = torch.zeros(3, 2, dtype=torch.long)
        logits = model.predict_topology_pair(hidden_pairs, token_pairs)
        self.assertEqual(logits.shape, (3, 16))

    def test_refined_topology_model_runs_one_set_pass(self):
        model = GapTreeRefinedTopologyBoundaryModel(
            vocab_size=40,
            gap_id=1,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
            refinement_dim=16,
        )
        tokens = torch.tensor([[3, 1, 10, 1, 4]])
        _, _, hidden = model(tokens)
        chosen = torch.zeros_like(tokens)
        provisional = torch.tensor([[4, 0, 4, 3, 4]])
        refined = model.refine_topology(
            hidden, chosen, provisional, tokens == 1
        )
        self.assertEqual(refined.shape, (1, 5, 4))

    def test_block_conditional_model_uses_alternating_frontier(self):
        valid = torch.tensor([
            [False, True, False, True, True],
            [True, False, False, False, True],
        ])
        anchors = alternating_frontier_mask(valid)
        self.assertTrue(torch.equal(anchors, torch.tensor([
            [False, True, False, False, True],
            [True, False, False, False, False],
        ])))
        model = GapTreeBlockConditionalTopologyBoundaryModel(
            vocab_size=40,
            gap_id=1,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
            refinement_dim=16,
        )
        self.assertTrue(model.conditional_block_topology)

        flipped = alternating_frontier_mask(
            valid, torch.tensor([True, True])
        )
        self.assertTrue(torch.equal(flipped, torch.tensor([
            [False, False, False, True, False],
            [False, False, False, False, True],
        ])))
        symmetric = GapTreeSymmetricBlockConditionalTopologyBoundaryModel(
            vocab_size=40,
            gap_id=1,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
            refinement_dim=16,
        )
        self.assertTrue(symmetric.symmetric_block_topology)

    def test_three_stage_frontier_partition_is_complete(self):
        valid = torch.tensor([[False, True, True, False, True, True]])
        stages = [frontier_stage_mask(valid, index, 3) for index in range(3)]
        self.assertTrue(torch.equal(stages[0], torch.tensor(
            [[False, True, False, False, False, True]]
        )))
        self.assertTrue(torch.equal(stages[1], torch.tensor(
            [[False, False, True, False, False, False]]
        )))
        self.assertTrue(torch.equal(stages[2], torch.tensor(
            [[False, False, False, False, True, False]]
        )))
        self.assertTrue(torch.equal(stages[0] | stages[1] | stages[2], valid))
        model = GapTreeThreeStageTopologyBoundaryModel(
            vocab_size=40,
            gap_id=1,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
            refinement_dim=16,
        )
        self.assertEqual(model.topology_stages, 3)

    def test_shared_regime_model_conditions_all_positions(self):
        model = GapTreeSharedRegimeBoundaryModel(
            vocab_size=40,
            gap_id=1,
            d_model=16,
            nhead=4,
            layers=1,
            max_positions=16,
        )
        tokens = torch.tensor([[3, 10, 1, 11, 4]])
        _, _, hidden = model(tokens)
        topology = model.predict_topology(
            hidden, torch.zeros_like(tokens), torch.tensor([2])
        )
        self.assertEqual(topology.shape, (1, 5, 4))
        self.assertTrue(torch.allclose(model.regime_prior.sum(), torch.tensor(1.0)))
        self.assertEqual([span_length_regime(x) for x in range(1, 9)], [0, 0, 1, 1, 1, 2, 2, 2])

    def test_dynamic_corruption_changes_by_epoch_and_is_reproducible(self):
        vocab = TextVocabulary(80, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        documents = [list(range(10 + 20 * index, 30 + 20 * index)) for index in range(8)]
        source = DynamicTextExampleDataset(documents, seed=17)
        epoch_zero = [source[index] for index in range(len(source))]
        source.set_epoch(1)
        epoch_one = [source[index] for index in range(len(source))]
        source.set_epoch(0)
        self.assertEqual(epoch_zero, [source[index] for index in range(len(source))])
        self.assertTrue(any(left != right for left, right in zip(epoch_zero, epoch_one)))
        tree = DynamicTreeTextDataset(source, vocab)
        sequential = DynamicSequentialTextDataset(source, vocab)
        regime_tree = DynamicRegimeTreeTextDataset(source, vocab)
        self.assertIn(vocab.GAP, tree[0]["tokens"])
        self.assertGreaterEqual(tree[0]["sample_weight"], 1.0)
        gap_index = sequential[0]["tokens"].index(vocab.GAP)
        self.assertNotEqual(sequential[0]["targets"][gap_index], -100)
        example = source[0]
        self.assertEqual(
            sequential[0]["sample_weight"],
            float(max(len(span) for span in example.spans) + 1),
        )
        batch = collate_compact_frontiers([tree[0], sequential[0]], vocab.PAD)
        self.assertEqual(batch["sample_weights"].shape, (2,))
        self.assertIn(regime_tree[0]["regime"], (0, 1, 2))

    def test_dynamic_random_windows_remove_fixed_document_length(self):
        documents = [list(range(10 + 120 * index, 130 + 120 * index)) for index in range(8)]
        source = DynamicTextExampleDataset(
            documents, seed=19, random_window_min=24, random_window_max=96
        )
        lengths_zero = [len(source[index].reconstruct()) for index in range(len(source))]
        source.set_epoch(1)
        lengths_one = [len(source[index].reconstruct()) for index in range(len(source))]
        self.assertTrue(all(24 <= length <= 96 for length in lengths_zero + lengths_one))
        self.assertGreater(len(set(lengths_zero + lengths_one)), 4)
        fixed = random_length_windows(documents, seed=23, min_length=24, max_length=96)
        self.assertTrue(all(24 <= len(window) <= 96 for window in fixed))


class SpanPolicyTests(unittest.TestCase):
    """Cover the context-constrained span policies and their diagnostics."""

    def documents(self):
        rng = random.Random(11)
        return [[rng.randrange(30) for _ in range(150)] for _ in range(40)]

    def observed_tokens(self, example):
        tokens = []
        for segment in example.segments:
            tokens.extend(segment)
        return tokens

    def test_uniform_policy_is_unchanged_by_the_new_parameter(self):
        documents = self.documents()
        default = sample_text_infilling_examples(documents, seed=5)
        explicit = sample_text_infilling_examples(
            documents, seed=5, span_policy="uniform"
        )
        self.assertEqual(
            [(e.segments, e.spans) for e in default],
            [(e.segments, e.spans) for e in explicit],
        )

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_text_infilling_examples(self.documents(), seed=5, span_policy="x")

    def test_copy_spans_survive_among_observed_tokens(self):
        examples = sample_text_infilling_examples(
            self.documents(), seed=5, gap_counts=(1, 2), span_policy="copy"
        )
        self.assertTrue(examples)
        for example in examples:
            observed = self.observed_tokens(example)
            for span in example.spans:
                width = len(span)
                self.assertTrue(any(
                    tuple(observed[start : start + width]) == span
                    for start in range(len(observed) - width + 1)
                ))

    def test_anchored_pairs_are_disjoint_and_flanked(self):
        rng = random.Random(3)
        tokens = [rng.randrange(4) for _ in range(120)]
        pairs = anchored_repeat_pairs(tokens, 1, 4, anchor=2)
        self.assertTrue(pairs)
        for first, second, span in pairs:
            block = span + 4
            self.assertGreaterEqual(second - first, block)
            self.assertEqual(
                tokens[first : first + block], tokens[second : second + block]
            )

    def test_anchored_copy_keeps_a_visible_twin(self):
        examples = sample_text_infilling_examples(
            self.documents(), seed=5, gap_counts=(1,), span_policy="anchored_copy"
        )
        self.assertTrue(examples)
        for example in examples:
            observed = self.observed_tokens(example)
            for span in example.spans:
                width = len(span)
                self.assertTrue(any(
                    tuple(observed[start : start + width]) == span
                    for start in range(len(observed) - width + 1)
                ))

    def test_recoverability_check_rejects_a_destroyed_twin(self):
        tokens = [1, 2, 3, 9, 1, 2, 3]
        self.assertTrue(spans_remain_recoverable(tokens, [(0, 3)], 1, 4))
        self.assertFalse(spans_remain_recoverable(tokens, [(0, 3), (4, 7)], 1, 4))

    def test_position_marker_length_follows_the_gap_offset(self):
        examples = sample_text_infilling_examples(
            self.documents(), seed=5, gap_counts=(1,),
            max_span=8, span_policy="position_marker",
        )
        self.assertTrue(examples)
        for example in examples:
            offset = len(example.segments[0])
            self.assertEqual(len(example.spans[0]), 1 + offset % 8)

    def test_local_marker_length_follows_the_preceding_token(self):
        examples = sample_text_infilling_examples(
            self.documents(), seed=5, gap_counts=(1,),
            max_span=8, span_policy="local_marker",
        )
        self.assertTrue(examples)
        for example in examples:
            left = example.segments[0]
            self.assertEqual(len(example.spans[0]), 1 + left[-1] % 8)

    def test_marginal_length_entropy_matches_a_known_distribution(self):
        example = TextInfillingExample(((1,), (2,), (3,)), ((4,), (5, 6)))
        self.assertAlmostEqual(
            marginal_length_entropy([example]), math.log(2), places=6
        )

    def test_pretrained_prompt_has_one_mask_and_omits_target(self):
        class StubTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "".join(chr(token_id) for token_id in token_ids)

        example = TextInfillingExample(((65, 66), (68,)), ((67,),))
        self.assertEqual(
            render_masked_text(example, StubTokenizer(), "<mask>"),
            "AB<mask>D",
        )

    def test_unique_mask_positions_rejects_missing_or_duplicate_masks(self):
        input_ids = torch.tensor([[1, 9, 2], [9, 3, 4]])
        self.assertTrue(torch.equal(unique_token_positions(input_ids, 9), torch.tensor([1, 0])))
        with self.assertRaises(ValueError):
            unique_token_positions(torch.tensor([[1, 2], [9, 9]]), 9)

    def test_identifiable_bootstrap_is_zero_for_the_empirical_prior(self):
        evaluation = {
            "targets": [1, 2, 1, 2],
            "example_nlls": [math.log(2)] * 4,
        }
        result = identifiable_statistics(
            evaluation, [0, 1, 0, 1], seed=3, bootstrap_samples=100
        )
        self.assertAlmostEqual(result["identifiable_nats"], 0.0, places=6)
        self.assertAlmostEqual(
            result["identifiable_nats_example_weighted"], 0.0, places=6
        )
        self.assertEqual(
            result["identifiable_nats_document_bootstrap_95_ci"], [0.0, 0.0]
        )

    def test_identifiable_statistics_rejects_misaligned_document_ids(self):
        evaluation = {
            "targets": [1, 2, 1, 2],
            "example_nlls": [math.log(2)] * 4,
        }
        with self.assertRaises(ValueError):
            identifiable_statistics(
                evaluation, [0, 1], seed=3, bootstrap_samples=10
            )

    def test_identifiable_statistics_weights_documents_not_examples(self):
        # One document contributes three examples and the other contributes
        # one, so the two weightings must disagree.
        evaluation = {
            "targets": [1, 1, 1, 2],
            "example_nlls": [0.0, 0.0, 0.0, 0.0],
        }
        result = identifiable_statistics(
            evaluation, [0, 0, 0, 1], seed=5, bootstrap_samples=50
        )
        self.assertNotAlmostEqual(
            result["identifiable_nats"],
            result["identifiable_nats_example_weighted"],
            places=6,
        )

    def test_twin_intervention_bootstrap_matches_document_weighted_estimand(self):
        result = paired_statistics(
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
            [0, 0, 0, 1],
            seed=7,
            bootstrap_samples=2000,
        )
        self.assertAlmostEqual(result["mean_nll_change"], 2.0)
        self.assertAlmostEqual(result["mean_nll_change_example_weighted"], 1.0)
        low, high = result["bootstrap_95_ci"]
        self.assertLessEqual(low, result["mean_nll_change"])
        self.assertGreaterEqual(high, result["mean_nll_change"])
        with self.assertRaises(ValueError):
            paired_statistics(
                [0.0], [0.0, 1.0], [0], seed=7, bootstrap_samples=10
            )

    def test_multigap_metrics_recover_deterministic_joint_targets(self):
        examples = [
            TextInfillingExample(((), (), ()), ((), (7,))),
            TextInfillingExample(((), (), ()), ((7, 8), (7, 8, 9))),
            TextInfillingExample(
                ((), (), ()), ((1, 2, 3, 4, 5, 6, 7, 8, 9), ())
            ),
        ]
        probabilities = []
        for example in examples:
            matrix = []
            for span in example.spans:
                row = [0.0] * 10
                row[len(span)] = 1.0
                matrix.append(row)
            probabilities.append(matrix)
        result = multigap_distribution_metrics(examples, probabilities)
        self.assertAlmostEqual(result["joint"]["marginal_tv_to_empirical"], 0.0)
        self.assertAlmostEqual(
            result["joint"]["observed_target_match_probability"], 1.0
        )
        self.assertAlmostEqual(result["joint"]["conditional_brier"], 0.0)
        self.assertAlmostEqual(
            result["total_length"]["marginal_tv_to_empirical"], 0.0
        )
        self.assertAlmostEqual(
            result["total_length"]["observed_target_match_probability"], 1.0
        )
        self.assertAlmostEqual(result["total_length"]["conditional_brier"], 0.0)
        self.assertAlmostEqual(
            result["joint"]["predicted_length_covariance"],
            result["joint"]["target_length_covariance"],
        )

        self.assertEqual(
            bootstrap_target_length_covariance(
                examples[:1], seed=3, bootstrap_samples=20
            ), [0.0, 0.0])

if __name__ == "__main__":
    unittest.main()
