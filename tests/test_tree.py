import math
import random
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from analyze_topology_exposure import tuple_dependence
from adapt_multigap_proper_mle import masked_batch_logp, sequential_batch_logp
from analyze_shared_regime import conditional_metrics
from calibrate_frontier_length import (
    balanced_length_target,
    cramer_cdf_distance,
    histogram_objective,
    parse_calibration_values,
    parse_search_indices,
    parse_seed_list,
    robust_seed_score,
)
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
from shape_prior import posterior_mean_token_depth
from exposure_gap import (
    pivot_posterior_marginals,
    record_posteriors,
    self_boundary_sources,
    self_boundary_token_loss,
    self_token_topology_loss,
)
from frontier_reencode import (
    apply_frontier_calibration_biases,
    collate_frontiers_with_history,
    decode_frontier_model,
    FixedScaffoldDerivationDataset,
    frontier_losses,
    markov_scaffold_losses,
    sample_frontier_scaffolds,
    sample_unified_scaffolds,
    conditional_scaffold_length_distribution,
    scaffold_length_distribution,
    persistent_scaffold_losses,
    scaffold_topology_losses,
    ScaffoldProposalDataset,
    sampled_length_probabilities,
    replace_with_generated_history,
    projected_total_progeny_distribution,
    sampled_trajectory_length_policy_loss,
    trajectory_energy_coefficients,
    topology_targets,
    unified_scaffold_losses,
)
from decompose_multigap_likelihood import decompose_exact_batch
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
from probe_conditional_length_context import gap_local_features
from probe_token_marker_information import topology_marker_targets
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
    PretrainedGapFrontierModel,
    PretrainedScaffoldTopologyModel,
    PretrainedUnifiedScaffoldModel,
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
    vocabulary_from_pretrained_tokenizer,
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

    def test_likelihood_decomposition_reconstructs_the_exact_value(self):
        """lexical + structure + tree entropy must equal the exact likelihood.

        The whole point of the decomposition diagnostic is that the split is
        exact rather than approximate, so the identity is pinned here. The
        entropy term must also be non-negative and must vanish for a
        single-token span, whose interval admits exactly one pivot tree.
        """
        vocab = TextVocabulary(
            vocab_size=12, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4
        )
        model = IntervalInsideBoundaryModel(
            vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
            d_model=24, nhead=4, layers=1, max_positions=32, max_steps=8,
            dropout=0.0,
        )
        model.eval()
        examples = [
            TextInfillingExample(
                segments=((5,), (6,), (7,)), spans=((8, 9, 10, 11), (10,))
            ),
            TextInfillingExample(segments=((6,), (8,), (7,)), spans=((), (9, 5))),
        ]
        parts = decompose_exact_batch(
            model, examples, vocab, torch.device("cpu")
        )
        # `total` is built as the sum of the three components, so checking it
        # against them would be a tautology. The claim worth pinning is that
        # the components reconstruct the likelihood the model actually reports.
        likelihood, _, _ = multi_depth_gap_log_likelihoods(
            model, examples, vocab, torch.device("cpu")
        )
        reconstructed = (
            parts["lexical"] + parts["structure"] + parts["tree_entropy"]
        )
        self.assertTrue(
            torch.allclose(reconstructed, likelihood.detach(), atol=1e-5)
        )
        self.assertTrue(bool((parts["tree_entropy"] > 0).any()))
        self.assertTrue(bool((parts["tree_entropy"] >= -1e-6).all()))

        single = [TextInfillingExample(segments=((5,), (6,)), spans=((8,),))]
        single_parts = decompose_exact_batch(
            model, single, vocab, torch.device("cpu")
        )
        self.assertAlmostEqual(
            float(single_parts["tree_entropy"][0]), 0.0, places=5
        )

        # The midpoint arm scores one fixed tree, so it must reconstruct the
        # midpoint joint weight, carry exactly zero entropy, and -- since a
        # single tree cannot beat the sum over all of them -- never score
        # better than the posterior arm.
        midpoint = decompose_exact_batch(
            model, examples, vocab, torch.device("cpu"), tree="midpoint"
        )
        self.assertTrue(torch.allclose(
            midpoint["lexical"] + midpoint["structure"] + midpoint["tree_entropy"],
            midpoint["total"], atol=1e-5,
        ))
        self.assertTrue(
            torch.allclose(midpoint["tree_entropy"], torch.zeros(2), atol=1e-6)
        )
        self.assertTrue(bool((midpoint["total"] <= parts["total"] + 1e-5).all()))

        # The topology-prior arm is an ELBO under the model's own topology
        # head: it must reconstruct, carry non-negative entropy, and sit at or
        # below the posterior arm, which is the tight bound.
        prior = decompose_exact_batch(
            model, examples, vocab, torch.device("cpu"), tree="topology_prior"
        )
        self.assertTrue(torch.allclose(
            prior["lexical"] + prior["structure"] + prior["tree_entropy"],
            prior["total"], atol=1e-5,
        ))
        self.assertTrue(bool((prior["tree_entropy"] >= -1e-6).all()))
        self.assertTrue(bool((prior["total"] <= parts["total"] + 1e-5).all()))

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

    def test_native_pretrained_vocabulary_reuses_embeddings_and_mlm_head(self):
        from gtdlm.model import PretrainedLengthMaskedModel

        class StubNativeTokenizer:
            pad_token_id = 1
            mask_token_id = 9
            bos_token_id = 0
            eos_token_id = 2
            cls_token_id = None
            sep_token_id = None
            all_special_ids = [0, 1, 2, 3, 9]
            mask_token = "<mask>"

            def __len__(self):
                return 16

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

        tokenizer = StubNativeTokenizer()
        vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
        self.assertNotIn(3, vocab.generated_token_ids)
        backbone = StubBackbone()
        mlm_head = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.GELU(),
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, len(tokenizer)),
        )
        model = PretrainedIntervalInsideModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            tokenizer,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=mlm_head,
            native_vocabulary=True,
            initialize_custom_embeddings=False,
        )
        self.assertIs(model.encoder.token_embedding, backbone.embedding)
        self.assertIs(model.token_head, mlm_head)
        example = corrupt_token_sequence([5, 6, 7, 8, 10], [(1, 3)])
        exact, midpoint = depth_batch_log_likelihoods(
            model, [example], vocab, torch.device("cpu"), 4, 0.0
        )
        self.assertTrue(torch.isfinite(exact).all())
        self.assertGreaterEqual(float(exact[0]), float(midpoint[0]))
        (-exact.mean()).backward()
        self.assertIsNotNone(backbone.embedding.weight.grad)
        self.assertIsNotNone(mlm_head[-1].weight.grad)

        baseline_backbone = StubBackbone()
        baseline_head = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.GELU(),
            torch.nn.LayerNorm(8), torch.nn.Linear(8, len(tokenizer)),
        )
        baseline = PretrainedLengthMaskedModel(
            vocab.vocab_size, 8, vocab.GAP, vocab.PAD, tokenizer,
            backbone=baseline_backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=baseline_head,
            native_vocabulary=True,
            initialize_custom_embeddings=False,
        )
        prompts = torch.tensor([
            [vocab.LEFT, 5, vocab.GAP, 8, vocab.RIGHT],
            [vocab.LEFT, 6, vocab.GAP, vocab.RIGHT, vocab.PAD],
        ])
        padding = prompts.eq(vocab.PAD)
        logits, valid = baseline.predict_tokens(prompts, padding, [2, 3])
        self.assertEqual(tuple(logits.shape), (2, 3, len(tokenizer)))
        self.assertEqual(valid.sum(dim=1).tolist(), [2, 3])
        logits[valid].mean().backward()
        self.assertIsNotNone(baseline_head[-1].weight.grad)

    def test_fixed_mask_bank_is_length_blind_and_differentiable(self):
        class StubTokenizer:
            pad_token_id = 1
            mask_token_id = 9
            bos_token_id = 0
            eos_token_id = 2
            cls_token_id = None
            sep_token_id = None
            all_special_ids = [0, 1, 2, 3, 9]
            mask_token = "<mask>"

            def __len__(self):
                return 16

        class ContextualBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=8)
                self.embedding = torch.nn.Embedding(16, 8)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, attention_mask):
                hidden = self.embedding(input_ids)
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                context = (hidden * mask).sum(1, keepdim=True) / mask.sum(
                    1, keepdim=True
                ).clamp_min(1)
                positions = torch.arange(
                    hidden.size(1), device=hidden.device, dtype=hidden.dtype
                ).view(1, -1, 1)
                return SimpleNamespace(
                    last_hidden_state=hidden + context + positions / 10
                )

        tokenizer = StubTokenizer()
        vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
        backbone = ContextualBackbone()
        mlm_head = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.GELU(),
            torch.nn.LayerNorm(8), torch.nn.Linear(8, len(tokenizer)),
        )
        model = PretrainedIntervalInsideModel(
            vocab.vocab_size, vocab.GAP, vocab.PAD, tokenizer,
            backbone=backbone, pretrained_tokenizer=tokenizer,
            pretrained_lm_head=mlm_head, native_vocabulary=True,
            fixed_mask_count=4, initialize_custom_embeddings=False,
            dropout=0.0,
        )
        # Same observed prompt, different hidden target lengths. Target length
        # must not change encoder input or the fixed bank.
        examples = [
            TextInfillingExample(((5,), (8,)), ((6,),)),
            TextInfillingExample(((5,), (8,)), ((6, 7, 10),)),
        ]
        rows = [example.prompt(vocab) for example in examples]
        tokens = torch.tensor(rows)
        padding = tokens.eq(vocab.PAD)
        encoded = model.encode(tokens, padding)
        self.assertEqual(tuple(model.encoder.mask_bank_states.shape), (2, 4, 8))
        self.assertTrue(torch.allclose(
            model.encoder.mask_bank_states[0],
            model.encoder.mask_bank_states[1],
        ))
        self.assertFalse(torch.allclose(
            model.encoder.mask_bank_states[0, 0],
            model.encoder.mask_bank_states[0, 1],
        ))
        gap_positions = torch.tensor([row.index(vocab.GAP) for row in rows])
        contexts = encoded[torch.arange(2), gap_positions]
        hidden = model.interval_hidden(
            contexts, torch.tensor([5, 5]), torch.tensor([8, 8]),
            torch.tensor([0, 0]), torch.tensor([0, 1]),
        )
        self.assertTrue(torch.allclose(hidden[0], hidden[1], atol=1e-6))

        exact, _, charts = depth_batch_log_likelihoods(
            model, examples, vocab, torch.device("cpu"), 4, 0.0,
            return_charts=True,
        )
        self.assertTrue(torch.isfinite(exact).all())
        for index, combined in charts["combined"].items():
            reachable = combined > float("-inf")
            self.assertTrue(torch.allclose(
                combined[reachable],
                charts["token"][index][reachable]
                + charts["topology"][index][reachable],
            ))
        (-exact.mean()).backward()
        self.assertIsNotNone(model.mask_bank_query.weight.grad)
        self.assertIsNotNone(model.mask_bank_residual_scale.grad)

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

    def test_pretrained_masked_baseline_reads_one_state_per_span_token(self):
        """The baseline must expose exactly as many mask states as span tokens.

        Its whole purpose is to be capacity- and pretraining-matched to the
        tree model, so the token pass has to read one backbone state per
        target token; a zero-length span still needs one readable position for
        the length head. Stubs stand in for the backbone so the contract is
        checked without downloading weights.
        """
        from gtdlm.model import PretrainedLengthMaskedModel

        hidden_size = 6

        class StubConfig:
            hidden_size = 6

        class StubBackbone(torch.nn.Module):
            config = StubConfig()

            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(32, hidden_size)

            def get_input_embeddings(self):
                return self.embeddings

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                del attention_mask, kwargs
                return type(
                    "Output", (), {"last_hidden_state": self.embeddings(input_ids)}
                )()

        class StubPretrainedTokenizer:
            mask_token = "<m>"
            mask_token_id = 9

            def __call__(self, texts, **kwargs):
                del kwargs
                rows = []
                for text in texts:
                    row = []
                    index = 0
                    while index < len(text):
                        if text.startswith(self.mask_token, index):
                            row.append(self.mask_token_id)
                            index += len(self.mask_token)
                        else:
                            row.append(1)
                            index += 1
                    rows.append(row)
                width = max(len(row) for row in rows)
                ids = [row + [0] * (width - len(row)) for row in rows]
                attention = [
                    [1] * len(row) + [0] * (width - len(row)) for row in rows
                ]
                return {
                    "input_ids": torch.tensor(ids),
                    "attention_mask": torch.tensor(attention),
                }

        class StubSourceTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "".join(chr(token_id) for token_id in token_ids)

        vocab = TextVocabulary(
            vocab_size=12, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4
        )
        model = PretrainedLengthMaskedModel(
            vocab.vocab_size, 8, vocab.GAP, vocab.PAD, StubSourceTokenizer(),
            backbone=StubBackbone(),
            pretrained_tokenizer=StubPretrainedTokenizer(),
            initialize_custom_embeddings=False,
            tie_token_embeddings=False,
        )
        examples = [
            TextInfillingExample(((65, 66), (68,)), ((67, 67, 67),)),
            TextInfillingExample(((65,), (68,)), ((),)),
        ]
        rows = [example.prompt(vocab) for example in examples]
        width = max(len(row) for row in rows)
        tokens = torch.full((2, width), vocab.PAD, dtype=torch.long)
        padding = torch.ones_like(tokens, dtype=torch.bool)
        for index, row in enumerate(rows):
            tokens[index, :len(row)] = torch.tensor(row)
            padding[index, :len(row)] = False

        logits, valid = model.predict_tokens(tokens, padding, [3, 0])
        self.assertEqual(int(valid[0].sum()), 3)
        # An empty span keeps one readable position rather than none.
        self.assertEqual(int(valid[1].sum()), 1)
        self.assertEqual(logits.shape[-1], vocab.vocab_size)
        self.assertEqual(
            tuple(model.predict_length(tokens, padding).shape), (2, 9)
        )

    def test_prompt_attention_gives_each_interval_its_own_context(self):
        """Interval records must read the backbone sequence, not one pooled vector.

        The chart's only link to the prompt is otherwise a single summary
        vector plus two static boundary embeddings, which
        research/LIKELIHOOD_DECOMPOSITION.md attributes 84% of the generation
        deficit to. This pins the three properties the fix must have: records
        of the same example with different boundaries get different attended
        context, attention never crosses examples, and padded key positions are
        excluded.
        """
        from gtdlm.model import PretrainedIntervalInsideModel

        hidden_size = 8

        class StubConfig:
            hidden_size = 8

        class StubBackbone(torch.nn.Module):
            config = StubConfig()

            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(32, hidden_size)

            def get_input_embeddings(self):
                return self.embeddings

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                del attention_mask, kwargs
                return type(
                    "Output", (), {"last_hidden_state": self.embeddings(input_ids)}
                )()

        class StubPretrainedTokenizer:
            mask_token = "<m>"
            mask_token_id = 9

            def __call__(self, texts, **kwargs):
                del kwargs
                rows = []
                for text in texts:
                    row, index = [], 0
                    while index < len(text):
                        if text.startswith(self.mask_token, index):
                            row.append(self.mask_token_id)
                            index += len(self.mask_token)
                        else:
                            row.append(1 + (ord(text[index]) % 7))
                            index += 1
                    rows.append(row)
                width = max(len(row) for row in rows)
                return {
                    "input_ids": torch.tensor(
                        [r + [0] * (width - len(r)) for r in rows]
                    ),
                    "attention_mask": torch.tensor(
                        [[1] * len(r) + [0] * (width - len(r)) for r in rows]
                    ),
                }

        class StubSourceTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                del skip_special_tokens
                return "".join(chr(t) for t in token_ids)

        vocab = TextVocabulary(
            vocab_size=12, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4
        )
        model = PretrainedIntervalInsideModel(
            vocab.vocab_size, vocab.GAP, vocab.PAD, StubSourceTokenizer(),
            backbone=StubBackbone(), pretrained_tokenizer=StubPretrainedTokenizer(),
            initialize_custom_embeddings=False, tie_token_embeddings=False,
            prompt_attention=True, dropout=0.0,
        ).eval()
        self.assertTrue(model.prompt_attention)

        # Two examples with different observed text, so cross-example leakage
        # would be visible.
        examples = [
            TextInfillingExample(((5, 6, 7), (8,)), ((10, 11),)),
            TextInfillingExample(((9,), (6,)), ((11,),)),
        ]
        rows = [e.prompt(vocab) for e in examples]
        width = max(len(r) for r in rows)
        tokens = torch.full((2, width), vocab.PAD, dtype=torch.long)
        padding = torch.ones_like(tokens, dtype=torch.bool)
        for i, r in enumerate(rows):
            tokens[i, :len(r)] = torch.tensor(r)
            padding[i, :len(r)] = False

        model.encoder.keep_prompt_states(True)
        encoded = model.encode(tokens, padding)
        self.assertIsNotNone(model.encoder.prompt_states)
        positions = [r.index(vocab.GAP) for r in rows]
        contexts = torch.stack([encoded[i, p] for i, p in enumerate(positions)])

        # Three records: two from example 0 with different boundaries, one
        # from example 1.
        owners = torch.tensor([0, 0, 1])
        left = torch.tensor([7, 10, 9])
        right = torch.tensor([8, 8, 6])
        depths = torch.tensor([0, 1, 0])
        hidden = model.interval_hidden(
            contexts[owners], left, right, depths, owners
        )
        self.assertEqual(tuple(hidden.shape), (3, hidden_size))
        # Same example, different boundaries -> different representation.
        self.assertFalse(torch.allclose(hidden[0], hidden[1], atol=1e-6))

        # Attention must not read another example's prompt: re-running with
        # example 1's states replaced must leave example 0's records intact.
        baseline = hidden.clone()
        model.encoder.prompt_states = model.encoder.prompt_states.clone()
        model.encoder.prompt_states[1] += 5.0
        shifted = model.interval_hidden(
            contexts[owners], left, right, depths, owners
        )
        self.assertTrue(torch.allclose(baseline[:2], shifted[:2], atol=1e-6))
        self.assertFalse(torch.allclose(baseline[2], shifted[2], atol=1e-6))

        # Positions that are already padding must carry no weight at all.
        model.encoder.keep_prompt_states(True)
        model.encode(tokens, padding)
        padded = (~model.encoder.prompt_mask[1]).nonzero().flatten()
        self.assertTrue(padded.numel(), "example 1 should be shorter than example 0")
        model.encoder.prompt_states = model.encoder.prompt_states.clone()
        model.encoder.prompt_states[1, padded[0]] += 100.0
        padded_shift = model.interval_hidden(
            contexts[owners], left, right, depths, owners
        )
        self.assertTrue(torch.allclose(baseline, padded_shift, atol=1e-5))

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


class ReencodedFrontierTests(unittest.TestCase):
    def build_native_model(self):
        class StubTokenizer:
            pad_token_id = 1
            mask_token_id = 9
            bos_token_id = 0
            eos_token_id = 2
            cls_token_id = None
            sep_token_id = None
            all_special_ids = [0, 1, 2, 3, 9]

            def __len__(self):
                return 20

        class ContextualBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=8)
                self.embedding = torch.nn.Embedding(20, 8)
                self.calls = 0

            def get_input_embeddings(self):
                return self.embedding

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                **kwargs,
            ):
                del kwargs
                self.calls += 1
                hidden = (
                    inputs_embeds
                    if inputs_embeds is not None
                    else self.embedding(input_ids)
                )
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                context = (hidden * mask).sum(1, keepdim=True) / mask.sum(
                    1, keepdim=True
                ).clamp_min(1.0)
                return SimpleNamespace(last_hidden_state=hidden + context)

        tokenizer = StubTokenizer()
        backbone = ContextualBackbone()
        head = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.GELU(),
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, len(tokenizer)),
        )
        model = PretrainedGapFrontierModel(
            len(tokenizer),
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            dropout=0.0,
        )
        return tokenizer, backbone, head, model

    def test_gap_local_probe_reads_mask_and_immediate_boundaries(self):
        tokens = torch.tensor([[0, 5, 9, 6, 2], [0, 7, 8, 9, 2]])
        hidden = torch.arange(2 * 5 * 3, dtype=torch.float).view(2, 5, 3)
        gap, boundary, difference = gap_local_features(hidden, tokens, 9)
        self.assertTrue(torch.equal(gap[0], hidden[0, 2]))
        self.assertTrue(torch.equal(gap[1], hidden[1, 3]))
        self.assertTrue(torch.equal(boundary[0, :3], hidden[0, 1]))
        self.assertTrue(torch.equal(boundary[0, -3:], hidden[0, 3]))
        self.assertTrue(torch.equal(
            difference[1, -3:], hidden[1, 2] - hidden[1, 4]
        ))

    def test_oracle_probe_marker_mapping_matches_joint_action_order(self):
        degree = torch.tensor([0, 1, 1, 2])
        direction = torch.tensor([-100, 0, 1, -100])
        self.assertEqual(
            topology_marker_targets(degree, direction).tolist(),
            [0, 1, 2, 3],
        )

    def test_frontier_reencodes_generated_tokens_and_separates_gradients(self):
        tokenizer, backbone, head, model = self.build_native_model()
        first = torch.tensor([[0, 5, 9, 6, 2]])
        second = torch.tensor([[0, 5, 10, 9, 6, 2]])
        first_outputs = model(first, first.eq(1), torch.tensor([0]))
        second_outputs = model(second, second.eq(1), torch.tensor([1]))
        self.assertEqual(backbone.calls, 2)
        self.assertFalse(torch.allclose(
            first_outputs[-1][0, 2], second_outputs[-1][0, 3]
        ))

        model.zero_grad(set_to_none=True)
        structure_loss = sum(
            output.sum() for output in second_outputs[1:4]
        )
        structure_loss.backward()
        self.assertIsNone(backbone.embedding.weight.grad)
        self.assertIsNotNone(model.degree_head.weight.grad)

        model.zero_grad(set_to_none=True)
        token_outputs = model(second, second.eq(1), torch.tensor([1]))
        token_outputs[0][0, 3].mean().backward()
        self.assertIsNotNone(backbone.embedding.weight.grad)
        self.assertIsNotNone(head[-1].weight.grad)

    def test_topology_targets_factor_degree_and_unary_direction(self):
        left = torch.tensor([[0, 1, 0, 1, -100]])
        right = torch.tensor([[0, 0, 1, 1, -100]])
        degree, direction = topology_targets(left, right)
        self.assertEqual(degree.tolist(), [[0, 1, 1, 2, -100]])
        self.assertEqual(direction.tolist(), [[-100, 0, 1, -100, -100]])

    def test_root_frontier_input_never_contains_hidden_length(self):
        vocab = TextVocabulary(20, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2)
        short = TextInfillingExample(((5,), (6,)), ((10,),))
        long = TextInfillingExample(((5,), (6,)), ((10, 11, 12),))
        short_root = TextGapProposalDataset(
            [short], vocab, strategy="midpoint", seed=17
        )[0]
        long_root = TextGapProposalDataset(
            [long], vocab, strategy="midpoint", seed=17
        )[0]
        self.assertEqual(short_root["tokens"], long_root["tokens"])
        self.assertEqual(short_root["tokens"], [1, 5, 9, 6, 2])

    def test_rollout_expands_two_child_gaps_in_one_round(self):
        vocab = TextVocabulary(
            20, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2,
            EXTRA_STRUCTURAL=(3, 4),
        )

        class ScriptedModel:
            def __init__(self):
                self.gap_counts = []

            def eval(self):
                return self

            def __call__(self, tokens, padding, steps):
                del padding
                batch, width = tokens.shape
                token = torch.full((batch, width, 20), -100.0)
                root = torch.full((batch, width), -100.0)
                degree = torch.full((batch, width, 3), -100.0)
                direction = torch.zeros((batch, width, 2))
                hidden = torch.zeros((batch, width, 4))
                for row in range(batch):
                    gaps = tokens[row].eq(vocab.GAP).nonzero().flatten().tolist()
                    self.gap_counts.append(len(gaps))
                    for order, position in enumerate(gaps):
                        if int(steps[row]) == 0:
                            token[row, position, 10] = 100.0
                            degree[row, position, 2] = 100.0
                        else:
                            token[row, position, 11 + order] = 100.0
                            degree[row, position, 0] = 100.0
                return token, root, degree, direction, hidden

        example = TextInfillingExample(((5,), (6,)), ((11, 10, 12),))
        model = ScriptedModel()
        predictions, rounds, unfinished = decode_frontier_model(
            model, [example], vocab, torch.device("cpu"),
            max_rounds=4, max_decode_span=8, stochastic=True,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertEqual(predictions, [[[11, 10, 12]]])
        self.assertEqual(rounds, [2])
        self.assertEqual(unfinished, [False])
        self.assertEqual(model.gap_counts, [1, 2])

        lengths, shape_rounds, shape_unfinished = sample_frontier_scaffolds(
            ScriptedModel(),
            [example],
            vocab,
            torch.device("cpu"),
            samples_per_prompt=2,
            chunk_size=2,
            max_rounds=4,
            max_decode_span=8,
            seed=3,
        )
        self.assertEqual(lengths, [[3, 3]])
        self.assertEqual(shape_rounds, [[2, 2]])
        self.assertEqual(shape_unfinished, [[False, False]])

    def test_selective_rollout_defers_lower_ranked_gap(self):
        vocab = TextVocabulary(
            20, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2,
            EXTRA_STRUCTURAL=(3, 4),
        )

        class ScriptedModel:
            def __init__(self):
                self.gap_counts = []

            def eval(self):
                return self

            def __call__(self, tokens, padding, steps):
                del padding
                batch, width = tokens.shape
                lexical = torch.full((batch, width, 20), -100.0)
                root = torch.full((batch, width), -100.0)
                degree = torch.full((batch, width, 3), -100.0)
                direction = torch.zeros((batch, width, 2))
                hidden = torch.zeros((batch, width, 4))
                for row in range(batch):
                    gaps = tokens[row].eq(vocab.GAP).nonzero().flatten().tolist()
                    self.gap_counts.append(len(gaps))
                    for order, position in enumerate(gaps):
                        if int(steps[row]) == 0:
                            lexical[row, position, 10] = 20.0
                            degree[row, position, 2] = 20.0
                        else:
                            # The first gap is more confident and is committed
                            # while the second remains open for the next pass.
                            lexical[row, position, 11 + order] = 20.0 - order
                            if order == 1:
                                lexical[row, position, 13] = 18.0
                            degree[row, position, 0] = 20.0
                return lexical, root, degree, direction, hidden

        example = TextInfillingExample(((5,), (6,)), ((11, 10, 11),))
        model = ScriptedModel()
        predictions, rounds, unfinished = decode_frontier_model(
            model,
            [example],
            vocab,
            torch.device("cpu"),
            max_rounds=4,
            max_decode_span=8,
            stochastic=False,
            selective_gap_fraction=0.5,
        )
        self.assertEqual(predictions, [[[11, 10, 11]]])
        self.assertEqual(rounds, [3])
        self.assertEqual(unfinished, [False])
        self.assertEqual(model.gap_counts, [1, 2, 1])

    def test_sampled_lengths_keep_empty_and_overflow_mass(self):
        probabilities = sampled_length_probabilities(
            [
                [[], [10], [10, 11, 12]],
                [[10] * 9, [10, 11], [10, 11]],
            ],
            [[False, False, False], [False, False, True]],
            support_max=2,
        )
        self.assertEqual(probabilities[0], [1 / 3, 1 / 3, 0.0, 1 / 3])
        self.assertEqual(probabilities[1], [0.0, 0.0, 1 / 3, 2 / 3])

    def test_frontier_losses_apply_state_importance_weights(self):
        vocab = TextVocabulary(20, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2)

        class FixedModel:
            def __call__(self, tokens, padding, steps):
                del padding, steps
                batch, width = tokens.shape
                token = torch.zeros(batch, width, vocab.vocab_size)
                root = torch.zeros(batch, width)
                root[:, 1] = 2.0
                degree = torch.zeros(batch, width, 3)
                direction = torch.zeros(batch, width, 2)
                hidden = torch.zeros(batch, width, 4)
                return token, root, degree, direction, hidden

        batch = {
            "tokens": torch.tensor([[1, 9, 2], [1, 9, 2]]),
            "padding": torch.zeros(2, 3, dtype=torch.bool),
            "steps": torch.zeros(2, dtype=torch.long),
            "targets": torch.tensor([
                [-100, vocab.stop_action, -100],
                [-100, 10, -100],
            ]),
            "left_targets": torch.tensor([
                [-100, -100, -100],
                [-100, 0, -100],
            ]),
            "right_targets": torch.tensor([
                [-100, -100, -100],
                [-100, 0, -100],
            ]),
            "sample_weights": torch.tensor([1.0, 3.0]),
        }
        losses = frontier_losses(
            FixedModel(), batch, vocab, torch.device("cpu")
        )
        expected = (
            F.binary_cross_entropy_with_logits(
                torch.tensor(2.0), torch.tensor(1.0)
            )
            + 3
            * F.binary_cross_entropy_with_logits(
                torch.tensor(2.0), torch.tensor(0.0)
            )
        ) / 4
        self.assertAlmostEqual(float(losses["root"]), float(expected), places=6)

    def test_scaffold_frontier_keeps_completed_nodes_masked(self):
        vocab = TextVocabulary(20, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2)
        example = TextInfillingExample(((5,), (6,)), ((10, 11, 12),))
        states = ScaffoldProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        second = states[1]
        gap_positions = [
            index for index, token in enumerate(second["tokens"])
            if token == vocab.GAP
        ]
        self.assertEqual(len(gap_positions), 3)
        self.assertEqual(
            [second["targets"][index] for index in gap_positions],
            [10, -100, 12],
        )
        self.assertEqual(
            [second["semantic_tokens"][index] for index in gap_positions],
            [-100, 11, -100],
        )

    def test_shape_prior_starts_context_free_and_saves_head_only(self):
        tokenizer, backbone, _, _ = self.build_native_model()
        model = PretrainedScaffoldTopologyModel(
            len(tokenizer),
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=2,
            residual_dim=4,
            dropout=0.0,
        )
        tokens = torch.tensor([
            [0, 5, 9, 6, 2],
            [0, 7, 9, 8, 2],
        ])
        padding = tokens.eq(tokenizer.pad_token_id)
        steps = torch.zeros(2, dtype=torch.long)
        open_mask = tokens.eq(tokenizer.mask_token_id)
        root, regime, degree, direction, _ = model.structure_logits(
            tokens, padding, steps, open_mask
        )
        self.assertTrue(torch.allclose(root[0], root[1]))
        self.assertTrue(torch.allclose(regime[0], regime[1]))
        self.assertTrue(torch.allclose(degree[0], degree[1]))
        self.assertTrue(torch.allclose(direction[0], direction[1]))
        self.assertFalse(any(
            name.startswith("backbone.")
            for name in model.topology_state_dict()
        ))
        self.assertFalse(any(parameter.requires_grad for parameter in backbone.parameters()))

    def test_scaffold_state_feedback_reads_only_realized_process_state(self):
        tokenizer, backbone, _, _ = self.build_native_model()
        model = PretrainedScaffoldTopologyModel(
            len(tokenizer),
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=2,
            residual_dim=4,
            state_feedback=True,
            dropout=0.0,
        )
        with torch.no_grad():
            model.completed_degree_prior.weight[1, 0] = 2.0
        tokens = torch.tensor([
            [0, 5, 9, 6, 2, 1],
            [0, 5, 9, 9, 6, 2],
        ])
        padding = tokens.eq(tokenizer.pad_token_id)
        open_mask = torch.tensor([
            [False, False, True, False, False, False],
            [False, False, False, True, False, False],
        ])
        steps = torch.zeros(2, dtype=torch.long)
        _, _, degree, _, _ = model.structure_logits(
            tokens, padding, steps, open_mask
        )
        self.assertAlmostEqual(
            float(degree[1, 3, 0, 0] - degree[0, 2, 0, 0]),
            2.0,
            places=5,
        )

    def test_node_local_semantic_codes_are_reencoded_and_supervised(self):
        tokenizer, backbone, _, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedScaffoldTopologyModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=2,
            residual_dim=4,
            semantic_codes=4,
            dropout=0.0,
        )
        example = TextInfillingExample(
            ((5,), (6,)), ((10, 11, 12),)
        )
        dataset = ScaffoldProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        batch = collate_compact_frontiers(
            [dataset[1]], vocab.PAD
        )
        losses = scaffold_topology_losses(
            model, batch, vocab, torch.device("cpu")
        )
        self.assertTrue(torch.isfinite(losses["semantic"]))
        self.assertEqual(int(losses["semantic_count"]), 2)
        losses["semantic"].backward()
        self.assertIsNotNone(model.semantic_head.weight.grad)
        self.assertIsNone(backbone.embedding.weight.grad)

        tokens = batch["tokens"]
        padding = batch["padding"]
        steps = batch["steps"]
        open_mask = batch["targets"].ne(-100)
        no_codes = torch.full_like(tokens, -1)
        with_codes = no_codes.clone()
        completed = batch["semantic_tokens"].ge(0)
        with_codes[completed] = model.semantic_token_codes[
            batch["semantic_tokens"][completed]
        ]
        first = model.structure_logits(
            tokens,
            padding,
            steps,
            open_mask,
            slot_codes=no_codes,
            return_semantic=True,
        )[-1]
        second = model.structure_logits(
            tokens,
            padding,
            steps,
            open_mask,
            slot_codes=with_codes,
            return_semantic=True,
        )[-1]
        self.assertFalse(torch.allclose(first, second))

    def test_sampled_semantic_code_reaches_the_next_frontier(self):
        vocab = TextVocabulary(
            20, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2,
            EXTRA_STRUCTURAL=(3, 4),
        )

        class ScriptedSemanticModel:
            semantic_codes = 4

            def __init__(self):
                self.seen_codes = []

            def eval(self):
                return self

            def sample_structure(
                self,
                tokens,
                padding,
                steps,
                open_mask,
                generator=None,
                forced_regimes=None,
                slot_codes=None,
                slot_semantics=None,
                return_semantic_codes=False,
                return_continuous_semantic=False,
            ):
                del (
                    padding,
                    generator,
                    forced_regimes,
                    slot_semantics,
                    return_continuous_semantic,
                )
                self.seen_codes.append(slot_codes.clone())
                stops = torch.zeros_like(tokens, dtype=torch.bool)
                degrees = torch.zeros_like(tokens)
                directions = torch.zeros_like(tokens)
                regimes = torch.zeros(tokens.size(0), dtype=torch.long)
                codes = torch.full_like(tokens, 2)
                for row in range(tokens.size(0)):
                    if int(steps[row]) == 0:
                        degrees[row, open_mask[row]] = 1
                self.assert_semantic_request = return_semantic_codes
                return stops, degrees, directions, regimes, codes

        model = ScriptedSemanticModel()
        example = TextInfillingExample(((5,), (6,)), ((10, 11),))
        lengths, rounds, unfinished = sample_frontier_scaffolds(
            model,
            [example],
            vocab,
            torch.device("cpu"),
            samples_per_prompt=1,
            chunk_size=1,
            max_rounds=4,
            max_decode_span=8,
            seed=3,
        )
        self.assertEqual(lengths, [[2]])
        self.assertEqual(rounds, [[2]])
        self.assertEqual(unfinished, [[False]])
        self.assertEqual(len(model.seen_codes), 2)
        self.assertTrue(bool(model.seen_codes[1].eq(2).any()))
        self.assertTrue(model.assert_semantic_request)

    def test_continuous_semantic_state_is_reconstructed_and_reencoded(self):
        tokenizer, backbone, _, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedScaffoldTopologyModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=2,
            residual_dim=4,
            continuous_semantic=True,
            dropout=0.0,
        )
        example = TextInfillingExample(
            ((5,), (6,)), ((10, 11, 12),)
        )
        dataset = ScaffoldProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        batch = collate_compact_frontiers([dataset[1]], vocab.PAD)
        losses = scaffold_topology_losses(
            model, batch, vocab, torch.device("cpu")
        )
        self.assertTrue(torch.isfinite(losses["semantic"]))
        self.assertEqual(int(losses["semantic_count"]), 2)
        losses["semantic"].backward()
        self.assertIsNotNone(model.semantic_vector_head.weight.grad)
        self.assertIsNone(backbone.embedding.weight.grad)

        tokens = batch["tokens"]
        empty_states = torch.zeros(
            1, tokens.size(1), model.d_model
        )
        gold_states = empty_states.clone()
        completed = batch["semantic_tokens"].ge(0)
        gold_states[completed] = model.token_semantic_states(
            batch["semantic_tokens"][completed]
        )
        first = model.structure_logits(
            tokens,
            batch["padding"],
            batch["steps"],
            batch["targets"].ne(-100),
            slot_semantics=empty_states,
            return_continuous_semantic=True,
        )[-1]
        second = model.structure_logits(
            tokens,
            batch["padding"],
            batch["steps"],
            batch["targets"].ne(-100),
            slot_semantics=gold_states,
            return_continuous_semantic=True,
        )[-1]
        self.assertFalse(torch.allclose(first, second))

    def test_sampled_continuous_state_reaches_the_next_frontier(self):
        vocab = TextVocabulary(
            20, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2,
            EXTRA_STRUCTURAL=(3, 4),
        )

        class ScriptedContinuousModel:
            semantic_codes = 0
            continuous_semantic = True
            d_model = 3
            semantic_embedding_mean = torch.zeros(3)

            def __init__(self):
                self.seen_states = []

            def eval(self):
                return self

            def sample_structure(
                self,
                tokens,
                padding,
                steps,
                open_mask,
                generator=None,
                forced_regimes=None,
                slot_codes=None,
                slot_semantics=None,
                return_semantic_codes=False,
                return_continuous_semantic=False,
            ):
                del (
                    padding,
                    generator,
                    forced_regimes,
                    slot_codes,
                    return_semantic_codes,
                )
                self.seen_states.append(slot_semantics.clone())
                stops = torch.zeros_like(tokens, dtype=torch.bool)
                degrees = torch.zeros_like(tokens)
                directions = torch.zeros_like(tokens)
                regimes = torch.zeros(tokens.size(0), dtype=torch.long)
                states = torch.ones(
                    *tokens.shape, self.d_model, device=tokens.device
                )
                for row in range(tokens.size(0)):
                    if int(steps[row]) == 0:
                        degrees[row, open_mask[row]] = 1
                self.assert_semantic_request = return_continuous_semantic
                return stops, degrees, directions, regimes, states

        model = ScriptedContinuousModel()
        example = TextInfillingExample(((5,), (6,)), ((10, 11),))
        rollout = sample_frontier_scaffolds(
            model,
            [example],
            vocab,
            torch.device("cpu"),
            samples_per_prompt=1,
            chunk_size=1,
            max_rounds=4,
            max_decode_span=8,
            seed=3,
            return_states=True,
        )
        lengths, rounds, unfinished, states = rollout
        self.assertEqual(lengths, [[2]])
        self.assertEqual(rounds, [[2]])
        self.assertEqual(unfinished, [[False]])
        self.assertEqual(len(model.seen_states), 2)
        self.assertTrue(bool(model.seen_states[1].ne(0).any()))
        self.assertEqual(tuple(states[0][0].shape), (2, 3))
        self.assertTrue(model.assert_semantic_request)

    def test_unified_scaffold_zero_gate_nests_native_mlm_and_trains_gate(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedUnifiedScaffoldModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            pretrained_lm_head=head,
            generated_token_ids=vocab.generated_token_ids,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=2,
            residual_dim=4,
            posterior_topk=4,
            dropout=0.0,
        )
        tokens = torch.tensor([[0, 5, 9, 9, 6, 2]])
        padding = tokens.eq(vocab.PAD)
        steps = torch.tensor([1])
        open_mask = torch.tensor([[False, False, False, True, False, False]])
        states = torch.randn(1, tokens.size(1), model.d_model)
        model.eval()
        baseline = model.unified_logits(
            tokens, padding, steps, open_mask
        )[0]
        gated_off = model.unified_logits(
            tokens,
            padding,
            steps,
            open_mask,
            slot_semantics=states,
        )[0]
        self.assertTrue(torch.equal(baseline, gated_off))

        example = TextInfillingExample(((5,), (6,)), ((10, 11, 12),))
        dataset = ScaffoldProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        batch = collate_compact_frontiers([dataset[1]], vocab.PAD)
        model.train()
        losses = unified_scaffold_losses(
            model, batch, vocab, torch.device("cpu")
        )
        total = losses["root"] + losses["topology"] + losses["lexical"]
        total.backward()
        self.assertIsNotNone(model.posterior_gate.grad)
        self.assertIsNone(backbone.embedding.weight.grad)
        self.assertTrue(all(
            parameter.grad is None for parameter in head.parameters()
        ))
        self.assertEqual(int(losses["lexical_count"]), 3)

        empty = ScaffoldProposalDataset(
            [TextInfillingExample(((5,), (6,)), ((),))],
            vocab,
            strategy="midpoint",
            seed=17,
        )
        empty_batch = collate_compact_frontiers([empty[0]], vocab.PAD)
        empty_losses = unified_scaffold_losses(
            model, empty_batch, vocab, torch.device("cpu")
        )
        self.assertTrue(torch.isfinite(empty_losses["lexical"]))
        self.assertEqual(int(empty_losses["lexical_count"]), 0)

    def test_unified_sampler_grows_without_length_then_fills_in_parallel(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedUnifiedScaffoldModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            pretrained_lm_head=head,
            generated_token_ids=vocab.generated_token_ids,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=1,
            residual_dim=4,
            posterior_topk=4,
            dropout=0.0,
        )
        with torch.no_grad():
            model.root_stop_prior.fill_(-20.0)
            model.degree_prior.fill_(-20.0)
            model.degree_prior[0, 0, 1] = 20.0
            model.degree_prior[1:, 0, 0] = 20.0
        before = backbone.calls
        samples, rounds, unfinished = sample_unified_scaffolds(
            model,
            [TextInfillingExample(((5,), (6,)), ((10, 11),))],
            vocab,
            torch.device("cpu"),
            samples_per_prompt=1,
            chunk_size=1,
            max_rounds=4,
            max_decode_span=8,
            seed=3,
        )
        self.assertEqual(len(samples[0][0]), 2)
        self.assertEqual(rounds, [[2]])
        self.assertEqual(unfinished, [[False]])
        # Two grow rounds plus one final parallel lexical pass.
        self.assertEqual(backbone.calls - before, 3)
        self.assertTrue(all(
            token in vocab.generated_token_ids for token in samples[0][0]
        ))

    def test_unified_gap_context_grows_and_fills_with_one_model(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedUnifiedScaffoldModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            pretrained_lm_head=head,
            generated_token_ids=vocab.generated_token_ids,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=1,
            residual_dim=4,
            posterior_topk=4,
            prompt_conditioned=True,
            dropout=0.0,
        )
        with torch.no_grad():
            model.root_stop_prior.fill_(-20.0)
            model.degree_prior.fill_(-20.0)
            model.degree_prior[0, 0, 1] = 20.0
            model.degree_prior[1:, 0, 0] = 20.0
        before = backbone.calls
        samples, rounds, unfinished = sample_unified_scaffolds(
            model,
            [TextInfillingExample(((5,), (6,)), ((10, 11),))],
            vocab,
            torch.device("cpu"),
            samples_per_prompt=1,
            chunk_size=1,
            max_rounds=4,
            max_decode_span=8,
            seed=3,
            conditional_context_source="gap",
        )
        self.assertEqual(len(samples[0][0]), 2)
        self.assertEqual(rounds, [[2]])
        self.assertEqual(unfinished, [[False]])
        # One fixed GAP encode, two grow passes, and one final parallel fill.
        self.assertEqual(backbone.calls - before, 4)
        self.assertTrue(all(
            token in vocab.generated_token_ids for token in samples[0][0]
        ))

    def test_conditional_growth_needs_no_per_round_backbone_pass(self):
        """Growth passes are dead computation once shape cannot read them.

        In the conditional mode the shape policy reads a context fixed at round
        zero, and `unified_logits` does not forward `slot_semantics` into
        `structure_logits`, so the node-local token posterior reaches only the
        topology coupling path -- never the backbone encode and never the token
        head. Dropping the per-round pass must therefore be exactly
        output-preserving, leaving two backbone passes for a whole generation
        no matter how many tokens it emits.
        """
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedUnifiedScaffoldModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            pretrained_lm_head=head,
            generated_token_ids=vocab.generated_token_ids,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=1,
            residual_dim=4,
            posterior_topk=4,
            prompt_conditioned=True,
            dropout=0.0,
        )
        with torch.no_grad():
            model.root_stop_prior.fill_(-20.0)
            model.degree_prior.fill_(-20.0)
            model.degree_prior[0, 0, 1] = 20.0
            model.degree_prior[1:, 0, 0] = 20.0
        examples = [TextInfillingExample(((5,), (6,)), ((10, 11),))]

        def rollout(skip):
            before = backbone.calls
            outputs = sample_unified_scaffolds(
                model,
                examples,
                vocab,
                torch.device("cpu"),
                samples_per_prompt=1,
                chunk_size=1,
                max_rounds=4,
                max_decode_span=8,
                seed=3,
                conditional_context_source="gap",
                skip_round_encoding=skip,
            )
            return outputs, backbone.calls - before

        control, control_calls = rollout(False)
        ablated, ablated_calls = rollout(True)
        self.assertEqual(control, ablated)
        self.assertEqual(control_calls, 4)
        # Round-zero context plus the final parallel fill, and nothing between.
        self.assertEqual(ablated_calls, 2)

    def test_skipping_round_encoding_requires_a_conditional_context(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedUnifiedScaffoldModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            pretrained_lm_head=head,
            generated_token_ids=vocab.generated_token_ids,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=1,
            residual_dim=4,
            posterior_topk=4,
            dropout=0.0,
        )
        with self.assertRaises(ValueError):
            sample_unified_scaffolds(
                model,
                [TextInfillingExample(((5,), (6,)), ((10, 11),))],
                vocab,
                torch.device("cpu"),
                samples_per_prompt=1,
                chunk_size=1,
                max_rounds=4,
                max_decode_span=8,
                seed=3,
                skip_round_encoding=True,
            )

    def test_exact_scaffold_length_dp_tracks_total_progeny(self):
        tokenizer, backbone, _, _ = self.build_native_model()
        model = PretrainedScaffoldTopologyModel(
            len(tokenizer),
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=1,
            residual_dim=4,
            dropout=0.0,
        )
        with torch.no_grad():
            model.root_stop_prior.fill_(-20.0)
            model.degree_prior.fill_(-20.0)
            model.degree_prior[0, 0, 1] = 20.0
            model.degree_prior[1:, 0, 0] = 20.0
        probabilities = scaffold_length_distribution(
            model, max_length=4, max_rounds=4
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)
        self.assertGreater(float(probabilities[2]), 0.999)

    def test_token_conditioned_topology_nests_and_uses_the_emitted_token(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        model = PretrainedGapFrontierModel(
            len(tokenizer),
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            token_conditioned_topology=True,
            dropout=0.0,
        )
        model.eval()
        tokens = torch.tensor([[0, 5, 9, 6, 2]])
        steps = torch.tensor([1])
        ids = torch.full_like(tokens, -1)
        ids[0, 2] = 10
        with torch.no_grad():
            plain = model(tokens, tokens.eq(1), steps)
            conditioned = model(tokens, tokens.eq(1), steps, ids)
        # Zero-initialized coupling must reproduce the token-independent policy.
        for left, right in zip(plain[1:4], conditioned[1:4]):
            self.assertTrue(torch.allclose(left, right, atol=1e-6))
        with torch.no_grad():
            model.token_condition.weight.normal_(std=0.5)
            shifted = model(tokens, tokens.eq(1), steps, ids)
            other = ids.clone()
            other[0, 2] = 11
            shifted_other = model(tokens, tokens.eq(1), steps, other)
        self.assertFalse(
            torch.allclose(shifted[2][0, 2], conditioned[2][0, 2], atol=1e-5)
        )
        # A different emitted token must give a different branching decision.
        self.assertFalse(
            torch.allclose(shifted[2][0, 2], shifted_other[2][0, 2], atol=1e-5)
        )
        # Positions with no emitted token stay untouched by the coupling.
        self.assertTrue(
            torch.allclose(shifted[2][0, 0], conditioned[2][0, 0], atol=1e-6)
        )

    def test_root_head_never_reads_the_emitted_token(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        model = PretrainedGapFrontierModel(
            len(tokenizer),
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            token_conditioned_topology=True,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            model.token_condition.weight.normal_(std=1.0)
        tokens = torch.tensor([[0, 5, 9, 6, 2]])
        steps = torch.tensor([0])
        absent = torch.full_like(tokens, -1)
        present = absent.clone()
        present[0, 2] = 10
        with torch.no_grad():
            without = model(tokens, tokens.eq(1), steps, absent)
            with_token = model(tokens, tokens.eq(1), steps, present)
        # An empty root span is exactly the case with no gold pivot to feed, so
        # a root head that saw the conditioning would read its own label.
        self.assertTrue(torch.allclose(without[1], with_token[1], atol=1e-6))
        self.assertFalse(
            torch.allclose(without[2][0, 2], with_token[2][0, 2], atol=1e-5)
        )

    def test_token_conditioned_topology_gradient_reaches_the_coupling(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        model = PretrainedGapFrontierModel(
            len(tokenizer),
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            token_conditioned_topology=True,
            dropout=0.0,
        )
        tokens = torch.tensor([[0, 5, 9, 6, 2]])
        ids = torch.full_like(tokens, -1)
        ids[0, 2] = 10
        outputs = model(tokens, tokens.eq(1), torch.tensor([1]), ids)
        outputs[2][0, 2].sum().backward()
        self.assertIsNotNone(model.token_condition.weight.grad)
        self.assertGreater(
            float(model.token_condition.weight.grad.abs().sum()), 0.0
        )
        # The coupling reads a detached embedding, so it must not pull the
        # backbone's input embeddings along with it.
        self.assertTrue(
            model.token_embedding.weight.grad is None
            or float(model.token_embedding.weight.grad.abs().sum()) == 0.0
        )

    def test_marginal_joint_preserves_token_and_marker_distributions(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            marginal_preserving_joint=True,
            joint_rank=4,
            joint_sinkhorn_iterations=24,
            dropout=0.0,
        )
        model.eval()
        tokens = torch.tensor([[0, 5, 9, 6, 2]])
        padding = tokens.eq(vocab.PAD)
        steps = torch.tensor([1])
        generated = torch.tensor(vocab.generated_token_ids)
        with torch.no_grad():
            outputs = model(tokens, padding, steps)
            node = tokens.eq(vocab.GAP)
            independent = model.joint_action_log_probs(
                outputs[0][node], outputs[2][node], outputs[3][node],
                outputs[4][node], steps.unsqueeze(1).expand_as(tokens)[node],
                generated,
            )
            token_logp = outputs[0][node].index_select(
                -1, generated
            ).log_softmax(dim=-1)
            marker_logp = model.marker_log_probs(
                outputs[2][node], outputs[3][node]
            )
        expected = token_logp.unsqueeze(-1) + marker_logp.unsqueeze(-2)
        self.assertTrue(torch.allclose(independent, expected, atol=1e-6))

        with torch.no_grad():
            model.joint_marker_projection.weight.normal_(std=0.5)
            model.joint_marker_projection.bias.normal_(std=0.5)
            coupled = model.joint_action_log_probs(
                outputs[0][node], outputs[2][node], outputs[3][node],
                outputs[4][node], steps.unsqueeze(1).expand_as(tokens)[node],
                generated,
            )
        self.assertFalse(torch.allclose(coupled, independent, atol=1e-5))
        self.assertTrue(torch.allclose(
            torch.logsumexp(coupled, dim=-1), token_logp, atol=2e-5
        ))
        self.assertTrue(torch.allclose(
            torch.logsumexp(coupled, dim=-2), marker_logp, atol=2e-5
        ))

    def test_marginal_joint_loss_trains_copula_without_teacher_forcing(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            marginal_preserving_joint=True,
            joint_rank=4,
            joint_sinkhorn_iterations=12,
            dropout=0.0,
        )
        example = TextInfillingExample(((5,), (6,)), ((10, 11, 12),))
        states = TextGapProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        batch = collate_compact_frontiers(
            [states[index] for index in range(len(states))], vocab.PAD
        )
        losses = frontier_losses(
            model, batch, vocab, torch.device("cpu")
        )
        self.assertTrue(torch.isfinite(losses["joint"]))
        self.assertEqual(
            int(losses["joint_count"]), int(losses["degree_count"])
        )
        (losses["root"] + losses["token"] + losses["joint"]).backward()
        self.assertIsNotNone(model.joint_marker_projection.weight.grad)
        self.assertGreater(
            float(model.joint_marker_projection.weight.grad.abs().sum()), 0.0
        )
        self.assertIsNotNone(model.degree_head.weight.grad)

    def test_marginal_joint_rollout_draws_one_token_marker_action(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            marginal_preserving_joint=True,
            joint_rank=4,
            dropout=0.0,
        )
        with torch.no_grad():
            model.root_stop_head.weight.zero_()
            model.root_stop_head.bias.fill_(-20.0)
            model.degree_head.weight.zero_()
            model.degree_head.bias.copy_(torch.tensor([20.0, -20.0, -20.0]))
            model.joint_marker_projection.weight.normal_(std=0.5)
        before = backbone.calls
        predictions, rounds, unfinished = decode_frontier_model(
            model,
            [TextInfillingExample(((5,), (6,)), ((10,),))],
            vocab,
            torch.device("cpu"),
            max_rounds=4,
            max_decode_span=8,
            stochastic=True,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertEqual(len(predictions[0][0]), 1)
        self.assertEqual(rounds, [1])
        self.assertEqual(unfinished, [False])
        self.assertEqual(backbone.calls - before, 1)
        self.assertIn(predictions[0][0][0], vocab.generated_token_ids)

    def test_direct_joint_actions_nest_the_independent_product(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            direct_joint_actions=True,
            joint_rank=4,
            dropout=0.0,
        )
        model.eval()
        tokens = torch.tensor([[0, 5, 9, 6, 2]])
        padding = tokens.eq(vocab.PAD)
        steps = torch.tensor([1])
        generated = torch.tensor(vocab.generated_token_ids)
        with torch.no_grad():
            outputs = model(tokens, padding, steps)
            node = tokens.eq(vocab.GAP)
            joint = model.joint_action_log_probs(
                outputs[0][node], outputs[2][node], outputs[3][node],
                outputs[4][node], steps.unsqueeze(1).expand_as(tokens)[node],
                generated,
            )
            token_logp = outputs[0][node].index_select(
                -1, generated
            ).log_softmax(dim=-1)
            marker_logp = model.marker_log_probs(
                outputs[2][node], outputs[3][node]
            )
        expected = token_logp.unsqueeze(-1) + marker_logp.unsqueeze(-2)
        self.assertTrue(torch.allclose(joint, expected, atol=1e-6))
        self.assertTrue(torch.allclose(
            torch.logsumexp(joint.flatten(start_dim=-2), dim=-1),
            torch.zeros(1),
            atol=1e-6,
        ))

    def test_zero_joint_interaction_stays_the_independent_product_when_trained(self):
        """The prescribed dependence ablation must not drift off zero.

        The nesting test above holds at initialization.  This one takes a real
        gradient step through the joint loss and checks that the table is still
        exactly the independent token/marker product afterwards, which is what
        makes the ablation attributable to the interaction alone.
        """
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            direct_joint_actions=True,
            zero_joint_interaction=True,
            joint_rank=4,
            dropout=0.0,
        )
        # The frozen projections must be invisible to the optimizer, so weight
        # decay cannot move them.
        self.assertFalse(model.joint_marker_projection.weight.requires_grad)
        self.assertFalse(model.joint_token_projection.weight.requires_grad)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertNotIn(
            id(model.joint_marker_projection.weight),
            {id(p) for p in trainable},
        )

        example = TextInfillingExample(((5,), (6,)), ((10, 11, 12),))
        states = TextGapProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        batch = collate_compact_frontiers(
            [states[index] for index in range(len(states))], vocab.PAD
        )
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)
        losses = frontier_losses(model, batch, vocab, torch.device("cpu"))
        (losses["root"] + losses["joint"]).backward()
        optimizer.step()

        # The interaction path receives no gradient at all.
        self.assertIsNone(model.joint_marker_projection.weight.grad)
        self.assertIsNone(model.joint_token_projection.weight.grad)

        # And the table still factorizes exactly after the step, even though
        # the token head and branching heads have moved.
        model.eval()
        tokens = torch.tensor([[0, 5, 9, 6, 2]])
        padding = tokens.eq(vocab.PAD)
        steps = torch.tensor([1])
        generated = torch.tensor(vocab.generated_token_ids)
        with torch.no_grad():
            outputs = model(tokens, padding, steps)
            node = tokens.eq(vocab.GAP)
            joint = model.joint_action_log_probs(
                outputs[0][node], outputs[2][node], outputs[3][node],
                outputs[4][node], steps.unsqueeze(1).expand_as(tokens)[node],
                generated,
            )
            token_logp = outputs[0][node].index_select(
                -1, generated
            ).log_softmax(dim=-1)
            marker_logp = model.marker_log_probs(
                outputs[2][node], outputs[3][node]
            )
        expected = token_logp.unsqueeze(-1) + marker_logp.unsqueeze(-2)
        self.assertTrue(torch.allclose(joint, expected, atol=1e-6))
        self.assertTrue(torch.allclose(
            torch.logsumexp(joint.flatten(start_dim=-2), dim=-1),
            torch.zeros(1),
            atol=1e-6,
        ))

    def test_zero_joint_interaction_survives_a_trained_checkpoint_load(self):
        """A checkpoint trained *with* interaction must not reintroduce it.

        `joint_marker_projection` is only zero by initialization, so loading a
        trained direct checkpoint into the ablation arm would silently restore
        the coupling if the table read those weights.  It must not.
        """
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        shared = dict(
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            direct_joint_actions=True,
            joint_rank=4,
            dropout=0.0,
        )
        trained = PretrainedGapFrontierModel(
            vocab.vocab_size, vocab.GAP, vocab.PAD, **shared
        )
        with torch.no_grad():
            trained.joint_marker_projection.weight.normal_(std=0.5)
            trained.joint_marker_projection.bias.normal_(std=0.5)
        ablated = PretrainedGapFrontierModel(
            vocab.vocab_size, vocab.GAP, vocab.PAD,
            zero_joint_interaction=True, **shared
        )
        ablated.load_state_dict(trained.state_dict())
        self.assertGreater(
            float(ablated.joint_marker_projection.weight.abs().sum()), 0.0
        )

        trained.eval()
        ablated.eval()
        tokens = torch.tensor([[0, 5, 9, 6, 2]])
        padding = tokens.eq(vocab.PAD)
        steps = torch.tensor([1])
        generated = torch.tensor(vocab.generated_token_ids)
        node = tokens.eq(vocab.GAP)
        tables = []
        for model in (trained, ablated):
            with torch.no_grad():
                outputs = model(tokens, padding, steps)
                tables.append(model.joint_action_log_probs(
                    outputs[0][node], outputs[2][node], outputs[3][node],
                    outputs[4][node],
                    steps.unsqueeze(1).expand_as(tokens)[node], generated,
                ))
        # Same weights, same inputs: the arms must differ only by the coupling,
        # and the ablated arm must be the exact independent product.
        self.assertFalse(torch.allclose(tables[0], tables[1], atol=1e-5))
        with torch.no_grad():
            outputs = ablated(tokens, padding, steps)
            token_logp = outputs[0][node].index_select(
                -1, generated
            ).log_softmax(dim=-1)
            marker_logp = ablated.marker_log_probs(
                outputs[2][node], outputs[3][node]
            )
        self.assertTrue(torch.allclose(
            tables[1],
            token_logp.unsqueeze(-1) + marker_logp.unsqueeze(-2),
            atol=1e-6,
        ))

    def test_direct_joint_loss_trains_token_marker_and_interaction_together(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            direct_joint_actions=True,
            joint_rank=4,
            dropout=0.0,
        )
        example = TextInfillingExample(((5,), (6,)), ((10, 11, 12),))
        states = TextGapProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        batch = collate_compact_frontiers(
            [states[index] for index in range(len(states))], vocab.PAD
        )
        losses = frontier_losses(model, batch, vocab, torch.device("cpu"))
        self.assertTrue(torch.isfinite(losses["joint"]))
        self.assertEqual(
            int(losses["joint_count"]), int(losses["degree_count"])
        )
        (losses["root"] + losses["joint"]).backward()
        self.assertIsNotNone(model.joint_marker_projection.weight.grad)
        self.assertGreater(
            float(model.joint_marker_projection.weight.grad.abs().sum()), 0.0
        )
        self.assertIsNotNone(model.degree_head.weight.grad)
        self.assertIsNotNone(head[-1].weight.grad)

    def test_direct_joint_semantic_branching_emits_seven_tokens_in_three_passes(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            direct_joint_actions=True,
            joint_rank=4,
            dropout=0.0,
        )
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head[-1].bias[10] = 20.0
            model.root_stop_head.weight.zero_()
            model.root_stop_head.bias.fill_(-20.0)
            model.degree_head.weight.zero_()
            model.degree_head.bias.zero_()
            model.direction_head.weight.zero_()
            model.direction_head.bias.zero_()
            model.calibration_degree_bias.fill_(-20.0)
            model.calibration_degree_bias[0, 2] = 20.0
            model.calibration_degree_bias[1, 2] = 20.0
            model.calibration_degree_bias[2:, 0] = 20.0
        before = backbone.calls
        predictions, rounds, unfinished = decode_frontier_model(
            model,
            [TextInfillingExample(((5,), (6,)), ((10,),))],
            vocab,
            torch.device("cpu"),
            max_rounds=4,
            max_decode_span=8,
            stochastic=False,
        )
        self.assertEqual(predictions[0][0], [10] * 7)
        self.assertEqual(rounds, [3])
        self.assertEqual(unfinished, [False])
        self.assertEqual(backbone.calls - before, 3)

    def test_generated_history_replaces_exact_lexical_ancestors(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            direct_joint_actions=True,
            joint_rank=4,
            dropout=0.0,
        )
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head[-1].bias[10] = 30.0
        example = TextInfillingExample(
            ((5,), (6,)), ((11, 12, 13, 14, 15, 16, 17),)
        )
        states = TextGapProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        current = dict(states[2])
        current["history_states"] = [dict(states[0]), dict(states[1])]
        batch = collate_frontiers_with_history([current], vocab.PAD)
        selected, replacements = replace_with_generated_history(
            model, batch, vocab, torch.device("cpu"), probability=1.0
        )
        self.assertEqual(selected, 1)
        self.assertEqual(replacements, 3)
        completed = batch["node_ids"].ge(0) & batch["targets"].lt(0)
        self.assertEqual(batch["tokens"][completed].tolist(), [10, 10, 10])
        active_ids = batch["node_ids"][batch["targets"].ge(0)].tolist()
        self.assertEqual(active_ids, [0, 2, 4, 6])

    def test_frontier_calibration_applies_round_slopes(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            direct_joint_actions=True,
            dropout=0.0,
        )
        apply_frontier_calibration_biases(
            model, [-0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.25]
        )
        self.assertAlmostEqual(float(model.calibration_root_bias), -0.5)
        self.assertTrue(torch.allclose(
            model.calibration_degree_bias[0],
            torch.tensor([-0.5, 0.0, -0.5]),
        ))
        self.assertTrue(torch.allclose(
            model.calibration_degree_bias[2],
            torch.tensor([-0.5, 0.0, 0.0]),
        ))

    def test_robust_frontier_calibration_scores_ordered_histograms(self):
        target = torch.tensor([0.5, 0.0, 0.5])
        nearby = torch.tensor([0.5, 0.5, 0.0])
        exact = cramer_cdf_distance(target, target)
        displaced = cramer_cdf_distance(nearby, target)
        self.assertAlmostEqual(float(exact), 0.0)
        self.assertGreater(float(displaced), 0.0)
        self.assertAlmostEqual(
            float(histogram_objective(target, target, "tv")), 0.0
        )
        self.assertAlmostEqual(robust_seed_score([1.0, 3.0], 0.0), 2.0)
        self.assertAlmostEqual(robust_seed_score([1.0, 3.0], 1.0), 3.0)
        self.assertEqual(parse_seed_list("1901, 1913,1901", 7), [1901, 1913])
        self.assertEqual(parse_seed_list("", 7), [7])
        self.assertEqual(
            parse_calibration_values("-1,-1,1,0,0,0,-1"),
            [-1.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        )
        self.assertEqual(parse_search_indices("0,2,2,6"), [0, 2, 6])
        self.assertEqual(parse_search_indices(""), list(range(7)))
        self.assertTrue(torch.allclose(
            balanced_length_target(3, torch.device("cpu")),
            torch.tensor([0.4, 0.2, 0.2, 0.2, 0.0]),
        ))

    def test_projected_rollout_length_tracks_leaf_chain_and_binary_tree(self):
        leaf = projected_total_progeny_distribution(
            torch.tensor([[1.0, 0.0, 0.0]]), cap=16, horizon=3
        )
        chain = projected_total_progeny_distribution(
            torch.tensor([[0.0, 1.0, 0.0]]), cap=16, horizon=3
        )
        binary = projected_total_progeny_distribution(
            torch.tensor([[0.0, 0.0, 1.0]]), cap=16, horizon=3
        )
        stopped = projected_total_progeny_distribution(
            torch.tensor([[0.0, 0.0, 1.0]]),
            cap=16,
            horizon=3,
            root_stop_probabilities=torch.ones(1),
        )
        self.assertAlmostEqual(float(leaf[1]), 1.0, places=6)
        self.assertAlmostEqual(float(chain[3]), 1.0, places=6)
        self.assertAlmostEqual(float(binary[7]), 1.0, places=6)
        self.assertAlmostEqual(float(stopped[0]), 1.0, places=6)

    def test_projected_rollout_length_loss_reaches_structure_heads(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            direct_joint_actions=True,
            joint_rank=4,
            dropout=0.0,
        )
        example = TextInfillingExample(((5,), (6,)), ((10, 11, 12),))
        states = TextGapProposalDataset(
            [example], vocab, strategy="midpoint", seed=17
        )
        batch = collate_compact_frontiers([states[0]], vocab.PAD)
        losses = frontier_losses(
            model,
            batch,
            vocab,
            torch.device("cpu"),
            rollout_length_cap=8,
            rollout_length_horizon=4,
            rollout_length_detach_backbone=True,
        )
        self.assertTrue(torch.isfinite(losses["rollout_length"]))
        self.assertEqual(int(losses["rollout_length_count"]), 1)
        losses["rollout_length"].backward()
        self.assertIsNotNone(model.degree_head.weight.grad)
        self.assertGreater(float(model.degree_head.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(head[-1].weight.grad)

        later = collate_compact_frontiers([states[1]], vocab.PAD)
        later_losses = frontier_losses(
            model,
            later,
            vocab,
            torch.device("cpu"),
            rollout_length_cap=8,
            rollout_length_horizon=4,
            rollout_length_root_only=True,
        )
        self.assertEqual(int(later_losses["rollout_length_count"]), 0)

    def test_trajectory_energy_coefficients_reward_distributional_distance(self):
        coefficients, energy = trajectory_energy_coefficients(
            torch.tensor([1, 1]), torch.tensor([3, 3]), scale=8
        )
        self.assertTrue(bool((coefficients > 0).all()))
        self.assertGreater(float(energy), 0.0)

    def test_sampled_trajectory_length_policy_is_structure_only(self):
        tokenizer, backbone, head, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedGapFrontierModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            pretrained_lm_head=head,
            direct_joint_actions=True,
            joint_rank=4,
            dropout=0.0,
        )
        examples = [
            TextInfillingExample(((5,), (6,)), ((10,),)),
            TextInfillingExample(((7,), (8,)), ((11, 12, 13),)),
        ]
        loss, energy, lengths = sampled_trajectory_length_policy_loss(
            model,
            examples,
            vocab,
            torch.device("cpu"),
            samples_per_prompt=1,
            max_rounds=3,
            max_decode_span=8,
            seed=17,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(energy))
        self.assertEqual(lengths.numel(), 2)
        loss.backward()
        self.assertIsNotNone(model.degree_head.weight.grad)
        self.assertIsNone(head[-1].weight.grad)
        self.assertIsNone(backbone.embedding.weight.grad)

    def build_conditional_scaffold(self, regimes=2, state_feedback=True):
        tokenizer, backbone, _, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedScaffoldTopologyModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=regimes,
            residual_dim=4,
            state_feedback=state_feedback,
            prompt_conditioned=True,
            dropout=0.0,
        )
        tokens = torch.tensor([
            [0, 5, 9, 6, 2],
            [0, 7, 9, 8, 2],
        ])
        return model, tokens

    def test_conditional_length_chart_is_normalized_per_prompt(self):
        model, tokens = self.build_conditional_scaffold()
        with torch.no_grad():
            model.root_gate.fill_(0.7)
            model.degree_gate.fill_(0.5)
            torch.nn.init.normal_(model.root_residual.weight, std=1.0)
            torch.nn.init.normal_(model.degree_residual.weight, std=1.0)
            context = model.prompt_shape_context(tokens)
            probabilities = conditional_scaffold_length_distribution(
                model, context, max_length=6, max_rounds=6
            )
        self.assertEqual(tuple(probabilities.shape), (2, 8))
        for row in probabilities:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=5)

    def test_gap_prompt_context_reads_native_mask_state(self):
        model, tokens = self.build_conditional_scaffold()
        with torch.no_grad():
            hidden = model.backbone(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
            ).last_hidden_state
            rows = torch.arange(tokens.size(0))
            gaps = tokens.eq(model.gap_id).to(torch.long).argmax(dim=1)
            expected = model.global_adapter(hidden[rows, gaps])
            actual = model.prompt_shape_context(tokens, source="gap")
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_gap_context_zero_gate_still_matches_shared_chart(self):
        model, tokens = self.build_conditional_scaffold()
        with torch.no_grad():
            context = model.prompt_shape_context(tokens, source="gap")
            conditional = conditional_scaffold_length_distribution(
                model, context, max_length=6, max_rounds=6
            )
            shared = scaffold_length_distribution(
                model, max_length=6, max_rounds=6
            )
        for row in conditional:
            self.assertTrue(torch.allclose(row, shared, atol=1e-6))

    def test_gap_conditional_scaffold_rollout_uses_one_fixed_encode(self):
        model, _ = self.build_conditional_scaffold(
            regimes=1, state_feedback=False
        )
        vocab = TextVocabulary(
            model.backbone.get_input_embeddings().num_embeddings,
            PAD=1,
            GAP=9,
            MASK=9,
            LEFT=0,
            RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        with torch.no_grad():
            model.root_stop_prior.fill_(-100.0)
            model.degree_prior.fill_(-100.0)
            model.degree_prior[:, 0, 0] = 100.0
        example = TextInfillingExample(((5,), (6,)), ((7,),))
        before = model.backbone.calls
        lengths, rounds, unfinished = sample_frontier_scaffolds(
            model,
            [example],
            vocab,
            torch.device("cpu"),
            samples_per_prompt=2,
            chunk_size=2,
            max_rounds=4,
            max_decode_span=8,
            seed=3,
            conditional_context_source="gap",
        )
        self.assertEqual(lengths, [[1, 1]])
        self.assertEqual(rounds, [[1, 1]])
        self.assertEqual(unfinished, [[False, False]])
        self.assertEqual(model.backbone.calls - before, 1)

    def test_zero_gate_conditional_chart_matches_context_free_chart(self):
        model, tokens = self.build_conditional_scaffold()
        with torch.no_grad():
            torch.nn.init.normal_(model.degree_prior, std=0.5)
            torch.nn.init.normal_(model.root_stop_prior, std=0.5)
            torch.nn.init.normal_(model.open_degree_prior.weight, std=0.3)
            # Gates stay at zero, so the prompt cannot enter and every prompt
            # must reproduce the single shared process exactly.
            context = model.prompt_shape_context(tokens)
            conditional = conditional_scaffold_length_distribution(
                model, context, max_length=6, max_rounds=6
            )
            shared = scaffold_length_distribution(
                model, max_length=6, max_rounds=6
            )
        for row in conditional:
            self.assertTrue(torch.allclose(row, shared, atol=1e-6))

    def test_conditional_chart_separates_prompts_and_backpropagates(self):
        model, tokens = self.build_conditional_scaffold()
        with torch.no_grad():
            model.degree_gate.fill_(1.0)
            model.root_gate.fill_(1.0)
            torch.nn.init.normal_(model.degree_residual.weight, std=2.0)
            torch.nn.init.normal_(model.root_residual.weight, std=2.0)
        context = model.prompt_shape_context(tokens)
        probabilities = conditional_scaffold_length_distribution(
            model, context, max_length=6, max_rounds=6
        )
        self.assertFalse(
            torch.allclose(probabilities[0], probabilities[1], atol=1e-4)
        )
        loss = -probabilities[torch.arange(2), torch.tensor([2, 3])].log().sum()
        loss.backward()
        self.assertIsNotNone(model.degree_residual.weight.grad)
        self.assertGreater(
            float(model.degree_residual.weight.grad.abs().sum()), 0.0
        )
        # The frozen backbone must stay out of the shape gradient path.
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.backbone.parameters()
        ))

    def test_conditional_chart_matches_monte_carlo_rollout(self):
        model, tokens = self.build_conditional_scaffold(
            regimes=1, state_feedback=False
        )
        with torch.no_grad():
            model.root_stop_prior.fill_(-1.0)
            model.degree_prior.fill_(0.0)
            model.degree_prior[:, 0, 0] = 1.0
            model.degree_prior[:, 0, 1] = 0.5
            model.degree_prior[:, 0, 2] = -0.5
        max_length, max_rounds = 8, 8
        with torch.no_grad():
            context = model.prompt_shape_context(tokens[:1])
            chart = conditional_scaffold_length_distribution(
                model, context, max_length=max_length, max_rounds=max_rounds
            )[0]
        generator = torch.Generator().manual_seed(11)
        counts = torch.zeros(max_length + 2)
        trials = 20000
        for _ in range(trials):
            root, _, degree, _ = model.conditional_shape_logits(context, 0, 1, 0)
            if torch.rand((), generator=generator) < root.sigmoid()[0]:
                counts[0] += 1
                continue
            emitted, frontier = 0, 1
            overflow = False
            for step in range(max_rounds):
                if frontier == 0:
                    break
                emitted += frontier
                if emitted > max_length:
                    overflow = True
                    break
                _, _, degree, _ = model.conditional_shape_logits(
                    context, step, frontier, emitted - frontier
                )
                probabilities = degree[0, 0].softmax(dim=-1)
                children = int(torch.multinomial(
                    probabilities, frontier, replacement=True,
                    generator=generator,
                ).sum())
                frontier = children
            if overflow or frontier > 0:
                counts[max_length + 1] += 1
            else:
                counts[emitted] += 1
        empirical = counts / trials
        self.assertLess(float(0.5 * (empirical - chart).abs().sum()), 0.02)

    def test_shared_regime_scaffold_loss_trains_shape_prior(self):
        tokenizer, backbone, _, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedScaffoldTopologyModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=2,
            residual_dim=4,
            dropout=0.0,
        )
        examples = [
            TextInfillingExample(((5,), (6,)), ((10, 11, 12),)),
            TextInfillingExample(((7,), (8,)), ((),)),
        ]
        dataset = ScaffoldProposalDataset(
            examples, vocab, strategy="midpoint", seed=17
        )
        batch = collate_compact_frontiers(
            [dataset[index] for index in range(len(dataset))], vocab.PAD
        )
        losses = scaffold_topology_losses(
            model, batch, vocab, torch.device("cpu")
        )
        loss = losses["root"] + losses["topology"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.degree_prior.grad)
        self.assertIsNone(backbone.embedding.weight.grad)

    def test_persistent_regime_loss_marginalizes_complete_derivations(self):
        tokenizer, backbone, _, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedScaffoldTopologyModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=3,
            residual_dim=4,
            dropout=0.0,
        )
        examples = [
            TextInfillingExample(((5,), (6,)), ((10, 11, 12),)),
            TextInfillingExample(((7,), (8,)), ((13, 14),)),
        ]
        dataset = FixedScaffoldDerivationDataset(
            examples, vocab, strategy="midpoint", seed=17
        )
        derivations = [dataset[index] for index in range(len(dataset))]
        self.assertEqual([len(rows) for rows in derivations], [2, 2])
        losses = persistent_scaffold_losses(
            model, derivations, vocab, torch.device("cpu")
        )
        loss = losses["root"] + losses["topology"]
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(losses["derivation_count"]), 2)
        loss.backward()
        self.assertIsNotNone(model.regime_prior.weight.grad)
        self.assertIsNotNone(model.degree_prior.grad)
        self.assertIsNone(backbone.embedding.weight.grad)

    def test_markov_regime_loss_trains_depth_transitions(self):
        tokenizer, backbone, _, _ = self.build_native_model()
        vocab = TextVocabulary(
            len(tokenizer), PAD=1, GAP=9, MASK=9, LEFT=0, RIGHT=2,
            EXTRA_STRUCTURAL=(3,),
        )
        model = PretrainedScaffoldTopologyModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            backbone=backbone,
            pretrained_tokenizer=tokenizer,
            regimes=3,
            residual_dim=4,
            dropout=0.0,
        )
        examples = [
            TextInfillingExample(((5,), (6,)), ((10, 11, 12),)),
            TextInfillingExample(((7,), (8,)), ((13, 14),)),
        ]
        dataset = FixedScaffoldDerivationDataset(
            examples, vocab, strategy="midpoint", seed=17
        )
        derivations = [dataset[index] for index in range(len(dataset))]
        losses = markov_scaffold_losses(
            model, derivations, vocab, torch.device("cpu")
        )
        loss = losses["root"] + losses["topology"]
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(losses["derivation_count"]), 2)
        loss.backward()
        self.assertIsNotNone(model.regime_transition.grad)
        self.assertTrue(torch.isfinite(model.regime_transition.grad).all())
        self.assertIsNotNone(model.degree_prior.grad)
        self.assertIsNone(backbone.embedding.weight.grad)


class ExposureGapTests(unittest.TestCase):
    def build_model(self):
        vocab = TextVocabulary(40, PAD=0, GAP=1, MASK=2, LEFT=3, RIGHT=4)
        model = IntervalInsideBoundaryModel(
            vocab_size=40, gap_id=vocab.GAP, pad_id=vocab.PAD,
            d_model=16, nhead=4, layers=1, max_positions=32, dropout=0.0,
        )
        examples = [
            corrupt_token_sequence([10, 11, 12, 13, 14, 15], [(2, 4)]),
            corrupt_token_sequence([20, 21, 22, 23, 24], [(1, 3)]),
        ]
        return vocab, model, examples

    def test_pivot_marginals_sum_to_the_span_length(self):
        """d logZ / d score is the posterior count of each (node, pivot) cell.

        Every tree emits each target token exactly once, so the marginals of
        one example must sum to its span length whatever the tree posterior is.
        """
        vocab, model, examples = self.build_model()
        exact, _, internals = depth_batch_log_likelihoods(
            model, examples, vocab, torch.device("cpu"), 2, 0.5,
            return_internals=True,
        )
        marginals = pivot_posterior_marginals(exact, internals["flat_scores"])
        self.assertTrue(torch.isfinite(marginals).all())
        self.assertTrue(marginals.ge(-1e-6).all())
        records = internals["records"]
        owners = torch.tensor(
            [records[index][0] for index in internals["pivot_record_indices"]]
        )
        for example_index, example in enumerate(examples):
            total = float(marginals[owners.eq(example_index)].sum())
            self.assertAlmostEqual(total, len(example.spans[0]), places=4)

    def test_record_posteriors_normalize_over_four_topologies(self):
        vocab, model, examples = self.build_model()
        exact, _, internals = depth_batch_log_likelihoods(
            model, examples, vocab, torch.device("cpu"), 2, 0.0,
            return_internals=True,
        )
        marginals = pivot_posterior_marginals(exact, internals["flat_scores"])
        usage, topology = record_posteriors(
            marginals,
            internals["pivot_record_indices"],
            internals["targets"],
            len(internals["records"]),
        )
        used = usage.gt(1e-8)
        self.assertTrue(bool(used.any()))
        self.assertTrue(torch.allclose(
            topology[used].sum(dim=-1), torch.ones(int(used.sum())), atol=1e-4
        ))
        # The root node of a nonempty span is always used exactly once.
        roots = [
            index for index, (_, depth, lo, hi) in enumerate(internals["records"])
            if depth == 0 and lo == 0
        ]
        for index in roots:
            self.assertAlmostEqual(float(usage[index]), 1.0, places=4)

    def test_self_boundary_sources_point_at_the_emitting_parent(self):
        records = [
            (0, 0, 0, 3), (0, 1, 0, 1), (0, 1, 0, 2), (0, 1, 1, 3),
            (0, 1, 2, 3), (0, 2, 0, 1), (0, 2, 1, 2), (0, 2, 2, 3),
        ]
        left, right = self_boundary_sources(records, {0: 3})
        lookup = {record: index for index, record in enumerate(records)}
        # The root sees only intact prompt context on both sides.
        self.assertEqual(left[0], -1)
        self.assertEqual(right[0], -1)
        # [1, 3) at depth 1 is the right child of [0, 3) at depth 0.
        self.assertEqual(left[lookup[(0, 1, 1, 3)]], lookup[(0, 0, 0, 3)])
        self.assertEqual(right[lookup[(0, 1, 1, 3)]], -1)
        # [0, 2) at depth 1 is the left child of [0, 3) at depth 0.
        self.assertEqual(right[lookup[(0, 1, 0, 2)]], lookup[(0, 0, 0, 3)])
        self.assertEqual(left[lookup[(0, 1, 0, 2)]], -1)
        # An interior singleton is bounded by two generated tokens.
        self.assertEqual(left[lookup[(0, 2, 1, 2)]], lookup[(0, 1, 0, 2)])
        self.assertEqual(right[lookup[(0, 2, 1, 2)]], lookup[(0, 1, 1, 3)])

    def test_self_token_topology_loss_trains_the_topology_head(self):
        vocab, model, examples = self.build_model()
        exact, _, internals = depth_batch_log_likelihoods(
            model, examples, vocab, torch.device("cpu"), 2, 0.0,
            return_internals=True,
        )
        marginals = pivot_posterior_marginals(exact, internals["flat_scores"])
        loss = self_token_topology_loss(model, internals, marginals, 2, 0.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(model.topology_head.weight.grad)
        self.assertGreater(float(model.topology_head.weight.grad.abs().sum()), 0.0)

    def test_self_boundary_loss_only_perturbs_generated_boundaries(self):
        vocab, model, examples = self.build_model()
        exact, _, internals = depth_batch_log_likelihoods(
            model, examples, vocab, torch.device("cpu"), 2, 0.0,
            return_internals=True,
        )
        marginals = pivot_posterior_marginals(exact, internals["flat_scores"])
        loss = self_boundary_token_loss(model, internals, marginals, 1.0)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.interval_projection.weight.grad)
        # With probability zero nothing is replaced, so the term vanishes.
        self.assertIsNone(
            self_boundary_token_loss(model, internals, marginals, 0.0)
        )

    def test_shape_prior_is_normalised_and_differentiable(self):
        """The prior measures shape only, and cannot be gamed by span length.

        Total posterior mass equals the span length for any model, since every
        tree emits every token exactly once, so the normaliser is a constant.
        A chain over n tokens has mean token depth (n-1)/2 and a balanced tree
        about log2(n), so the quantity separates the two shapes.
        """
        vocab, model, examples = self.build_model()
        exact, _, internals = depth_batch_log_likelihoods(
            model, examples, vocab, torch.device("cpu"), 2, 0.0,
            return_internals=True,
        )
        marginals = pivot_posterior_marginals(exact, internals["flat_scores"])
        span_tokens = sum(len(example.spans[0]) for example in examples)
        self.assertAlmostEqual(float(marginals.sum()), span_tokens, places=4)

        depth = posterior_mean_token_depth(exact, internals)
        self.assertTrue(torch.isfinite(depth))
        self.assertGreaterEqual(float(depth), 0.0)
        longest = max(len(example.spans[0]) for example in examples)
        self.assertLess(float(depth), longest)
        depth.backward()
        self.assertIsNotNone(model.topology_head.weight.grad)
        self.assertGreater(
            float(model.topology_head.weight.grad.abs().sum()), 0.0
        )

    def test_self_boundary_control_ignores_the_sampled_tokens(self):
        """The matched control must depend only on the record draw.

        At probability 1.0 every perturbable side is drawn, so the record
        selection is deterministic and the seed changes only which tokens are
        sampled. The treatment must move with those tokens and the control must
        not, which is exactly the substitution the control removes.
        """
        vocab, model, examples = self.build_model()
        exact, _, internals = depth_batch_log_likelihoods(
            model, examples, vocab, torch.device("cpu"), 2, 0.0,
            return_internals=True,
        )
        marginals = pivot_posterior_marginals(exact, internals["flat_scores"])
        control = []
        treatment = []
        for seed in (11, 29):
            torch.manual_seed(seed)
            control.append(float(self_boundary_token_loss(
                model, internals, marginals, 1.0, substitute=False
            )))
            torch.manual_seed(seed)
            treatment.append(float(self_boundary_token_loss(
                model, internals, marginals, 1.0, substitute=True
            )))
        self.assertAlmostEqual(control[0], control[1], places=6)
        self.assertNotAlmostEqual(treatment[0], treatment[1], places=6)
        self.assertNotAlmostEqual(treatment[0], control[0], places=6)

    def test_exposure_auxiliaries_never_read_the_target_length(self):
        """Identical prompts with different hidden lengths share every input.

        The auxiliaries may only consume the chart records, the model's own
        token distribution and the gold span the primary objective already
        scores. This checks the substituted boundaries come from sampled
        tokens rather than from anything the target length determines.
        """
        vocab, model, _ = self.build_model()
        short = TextInfillingExample(((10, 11), (13,)), ((12,),))
        long = TextInfillingExample(((10, 11), (13,)), ((12, 20, 21),))
        self.assertEqual(short.prompt(vocab), long.prompt(vocab))
        for example in (short, long):
            _, _, internals = depth_batch_log_likelihoods(
                model, [example], vocab, torch.device("cpu"), 2, 0.0,
                return_internals=True,
            )
            length = len(example.spans[0])
            left, right = self_boundary_sources(internals["records"], {0: length})
            for index, (_, depth, lo, hi) in enumerate(internals["records"]):
                if depth == 0:
                    self.assertEqual((left[index], right[index]), (-1, -1))
                if lo == 0:
                    self.assertEqual(left[index], -1)
                if hi == length:
                    self.assertEqual(right[index], -1)


if __name__ == "__main__":
    unittest.main()
