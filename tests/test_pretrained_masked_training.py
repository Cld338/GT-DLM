import pytest
from torch import nn

import experiment_pretrained_masked_baseline as masked_experiment
from experiment_pretrained_masked_baseline import (
    configure_trainable_backbone_layers,
    decoded_span_metrics,
    evaluate_token_nll,
    transformer_layers,
)
from gtdlm.model import pretrained_masked_lm_head
from gtdlm.text_data import TextInfillingExample


class DummyBackbone(nn.Module):
    def __init__(self, layers=4):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            nn.Linear(4, 4) for _ in range(layers)
        )


class DummyModernBackbone(nn.Module):
    def __init__(self, layers=4):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(layers))


class DummyModernMaskedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 8)


def test_configure_trainable_backbone_layers_keeps_only_top_layers():
    backbone = DummyBackbone()

    active = configure_trainable_backbone_layers(backbone, 2)

    assert active == 2
    assert not any(parameter.requires_grad for parameter in backbone.embedding.parameters())
    assert not any(
        parameter.requires_grad
        for layer in backbone.encoder.layer[:2]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in backbone.encoder.layer[2:]
        for parameter in layer.parameters()
    )


def test_configure_trainable_backbone_layers_minus_one_keeps_full_model():
    backbone = DummyBackbone()

    active = configure_trainable_backbone_layers(backbone, -1)

    assert active == 4
    assert all(parameter.requires_grad for parameter in backbone.parameters())


def test_configure_trainable_backbone_layers_rejects_excess_layers():
    with pytest.raises(ValueError, match="requested 5 trainable layers"):
        configure_trainable_backbone_layers(DummyBackbone(), 5)


def test_configure_trainable_modernbert_layers():
    backbone = DummyModernBackbone()

    active = configure_trainable_backbone_layers(backbone, 2)

    assert active == 2
    assert transformer_layers(backbone) is backbone.layers
    assert not any(
        parameter.requires_grad
        for layer in backbone.layers[:2]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in backbone.layers[2:]
        for parameter in layer.parameters()
    )


def test_modernbert_split_mlm_head_is_composed():
    masked_lm = DummyModernMaskedLM()
    hidden = masked_lm.head.weight.new_zeros((2, 3, 4))

    logits = pretrained_masked_lm_head(masked_lm)(hidden)

    assert logits.shape == (2, 3, 8)


def test_evaluate_token_nll_weights_batch_means_by_token_count(monkeypatch):
    examples = [object(), object(), object()]

    def fake_token_batch_loss(model, batch, vocab, device, max_span, mixed):
        if len(batch) == 2:
            return masked_experiment.torch.tensor(2.0), 2
        return masked_experiment.torch.tensor(4.0), 6

    monkeypatch.setattr(
        masked_experiment, "token_batch_loss", fake_token_batch_loss
    )

    value = evaluate_token_nll(nn.Module(), examples, None, None, 2, 8)

    assert value == pytest.approx((2.0 * 2 + 4.0 * 6) / 8)


class DummyTokenizer:
    def decode(self, token_ids, **kwargs):
        return "".join(chr(96 + token_id) for token_id in token_ids)


def test_decoded_span_metrics_excludes_empty_targets():
    examples = [
        TextInfillingExample(((), ()), ((),)),
        TextInfillingExample(((), ()), ((1, 2),)),
        TextInfillingExample(((), ()), ((3,),)),
    ]

    metrics = decoded_span_metrics(
        examples, [[], [1, 2], [4]], DummyTokenizer()
    )

    assert metrics["nonempty_spans"] == 2
    assert metrics["decoded_exact_span_probability"] == pytest.approx(0.5)
    assert metrics["character_edit_similarity"] == pytest.approx(0.5)
