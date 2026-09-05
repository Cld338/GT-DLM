"""Train the learned-length-plus-masks baseline on the pretrained backbone.

`research/ROADMAP.md` item 12, the control `research/LIKELIHOOD_DECOMPOSITION.md`
identifies as missing and gating.

Every comparison so far between the exact tree objective and learned lengths
plus masks has pitted an 87M pretrained tree model against a 10M from-scratch
baseline. Those differ in pretraining, capacity and objective at once, so the
tree model's lead does not isolate the objective, and this project has
accordingly withdrawn that comparison. This script removes the first two
differences by giving the baseline the same `distilroberta-base` backbone,
the same corruption stream, the same split and the same budget as
`experiment_text_depth_inside_pretrained.py`, then scores it with the same
oracle-length token metric.

Note which way the remaining asymmetry runs: filling masks is the task the
backbone was pretrained on, whereas the tree model has to adapt that backbone
to an interval chart. If the tree objective still leads here, it leads against
a baseline holding the advantage.
"""

import argparse
import json
import math
import os
import statistics
import time
from typing import List, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import Adafactor, AutoTokenizer, get_linear_schedule_with_warmup

from evaluate_inside_lexical import lexical_sampling_metrics
from experiment import choose_device, edit_distance, parameter_count, seed_everything
from gtdlm.model import PretrainedLengthMaskedModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import (
    vocabulary_from_pretrained_tokenizer,
    vocabulary_from_tokenizer,
)


def collate_prompts(
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
):
    """Render single-gap prompts in the layout the pretrained encoder expects."""
    rows = [example.prompt(vocab) for example in examples]
    width = max(len(row) for row in rows)
    tokens = torch.full(
        (len(rows), width), vocab.PAD, dtype=torch.long, device=device
    )
    padding = torch.ones_like(tokens, dtype=torch.bool)
    for index, row in enumerate(rows):
        tokens[index, :len(row)] = torch.tensor(row, device=device)
        padding[index, :len(row)] = False
    return tokens, padding


def transformer_layers(backbone):
    """Locate the ordered transformer block list across supported backbones."""
    candidates = (
        getattr(backbone, "layers", None),
        getattr(getattr(backbone, "encoder", None), "layer", None),
        getattr(getattr(backbone, "encoder", None), "layers", None),
        getattr(getattr(backbone, "transformer", None), "layer", None),
        getattr(getattr(backbone, "transformer", None), "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise ValueError("cannot locate transformer layers on the backbone")


def configure_trainable_backbone_layers(backbone, trainable_layers):
    """Freeze the backbone except for its top N transformer layers."""
    layers = transformer_layers(backbone)
    layer_count = len(layers)
    if trainable_layers < 0:
        return layer_count
    if trainable_layers > layer_count:
        raise ValueError(
            "requested {} trainable layers from a {}-layer backbone".format(
                trainable_layers, layer_count
            )
        )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if trainable_layers:
        for layer in layers[-trainable_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    return trainable_layers


def autocast_context(device, mixed_precision):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=mixed_precision,
    )


def length_batch_loss(model, examples, vocab, device, max_span, mixed_precision=False):
    """Length cross-entropy for one batch."""
    tokens, padding = collate_prompts(examples, vocab, device)
    spans = [example.spans[0] for example in examples]
    lengths = torch.tensor(
        [min(len(span), max_span) for span in spans],
        dtype=torch.long, device=device,
    )
    with autocast_context(device, mixed_precision):
        return F.cross_entropy(model.predict_length(tokens, padding), lengths)


def token_batch_loss(model, examples, vocab, device, max_span, mixed_precision=False):
    """Masked-token cross-entropy and target count for one batch."""
    spans = [example.spans[0] for example in examples]

    nonempty = [index for index, span in enumerate(spans) if span]
    if not nonempty:
        return None, 0
    subset = [examples[index] for index in nonempty]
    sub_tokens, sub_padding = collate_prompts(subset, vocab, device)
    counts = [len(spans[index]) for index in nonempty]
    with autocast_context(device, mixed_precision):
        logits, valid = model.predict_tokens(sub_tokens, sub_padding, counts)
        generated = torch.tensor(vocab.generated_token_ids, device=device)
        lookup = torch.full(
            (vocab.vocab_size,), -1, dtype=torch.long, device=device
        )
        lookup[generated] = torch.arange(len(generated), device=device)
        logits = logits.index_select(-1, generated)

        flat_logits, flat_targets = [], []
        for row, gap_index in enumerate(nonempty):
            span = spans[gap_index]
            usable = int(valid[row].sum())
            for position in range(min(len(span), usable, logits.size(1))):
                flat_logits.append(logits[row, position])
                flat_targets.append(lookup[span[position]])
    if not flat_logits:
        return None, 0
    with autocast_context(device, mixed_precision):
        token_loss = F.cross_entropy(
            torch.stack(flat_logits), torch.stack(flat_targets)
        )
    return token_loss, len(flat_targets)


def batch_losses(
    model, examples, vocab, device, max_span, mixed_precision=False
):
    """Length cross-entropy plus masked-token cross-entropy for one batch."""
    length_loss = length_batch_loss(
        model, examples, vocab, device, max_span, mixed_precision
    )
    token_loss, _ = token_batch_loss(
        model, examples, vocab, device, max_span, mixed_precision
    )
    if token_loss is None:
        return length_loss, length_loss.detach(), None
    return length_loss + token_loss, length_loss.detach(), token_loss.detach()


@torch.inference_mode()
def decode_oracle_length(
    model, examples, vocab, device, batch_size, mixed_precision=False
):
    """Greedily fill the gold number of masks, returning one list per example."""
    predictions: List[List[int]] = []
    generated = torch.tensor(vocab.generated_token_ids, device=device)
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        tokens, padding = collate_prompts(batch, vocab, device)
        counts = [len(example.spans[0]) for example in batch]
        with autocast_context(device, mixed_precision):
            logits, valid = model.predict_tokens(tokens, padding, counts)
        chosen = generated[
            logits.index_select(-1, generated).argmax(dim=-1)
        ].cpu()
        for row, count in enumerate(counts):
            usable = min(count, int(valid[row].sum()), chosen.size(1))
            predictions.append([int(chosen[row, i]) for i in range(usable)])
    return predictions


@torch.inference_mode()
def evaluate_token_nll(
    model, examples, vocab, device, batch_size, max_span, mixed_precision=False
):
    totals, counts = 0.0, 0
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        token_loss, token_positions = token_batch_loss(
            model, batch, vocab, device, max_span, mixed_precision
        )
        if token_loss is not None and token_positions:
            totals += float(token_loss) * token_positions
            counts += token_positions
    return totals / max(1, counts)


def decoded_span_metrics(examples, predictions, tokenizer):
    """Character-level scores that remain comparable across tokenizers."""
    similarities, exact = [], 0
    for example, prediction in zip(examples, predictions):
        target_ids = list(example.spans[0])
        if not target_ids:
            continue
        try:
            target = tokenizer.decode(
                target_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            predicted = tokenizer.decode(
                prediction,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            target = tokenizer.decode(target_ids, skip_special_tokens=True)
            predicted = tokenizer.decode(prediction, skip_special_tokens=True)
        target = target.strip()
        predicted = predicted.strip()
        similarities.append(
            1.0 - edit_distance(predicted, target) / max(
                1, len(predicted), len(target)
            )
        )
        exact += int(predicted == target)
    return {
        "nonempty_spans": len(similarities),
        "character_edit_similarity": (
            sum(similarities) / max(1, len(similarities))
        ),
        "decoded_exact_span_probability": exact / max(1, len(similarities)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument("--data-dir", default="")
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_pretrained_masked_baseline"
    )
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--cache-dir", default=".hf_cache/hub")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # Defaults mirror experiment_text_depth_inside_pretrained.py so that the
    # two arms receive the same budget.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help=(
            "physical batch size used for each forward/backward; zero uses "
            "--batch-size. Gradients are accumulated across micro-batches to "
            "preserve the effective batch"
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "recompute backbone activations during the backward pass instead of "
            "storing them; same gradients, ~30%% slower, and the difference "
            "between fitting roberta-large on an 8GB card and not"
        ),
    )
    parser.add_argument(
        "--low-memory-optimizer",
        action="store_true",
        help=(
            "step AdamW one parameter at a time instead of fusing all of them; "
            "same update and lower temporary memory, though full roberta-large "
            "may still need partial unfreezing or Adafactor on an 8GB card"
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adafactor"),
        default="adamw",
        help=(
            "optimizer family; Adafactor keeps Adam's first moment but "
            "factorizes the second moment to fit large models in limited VRAM"
        ),
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="use CUDA FP16 autocast with gradient scaling",
    )
    parser.add_argument(
        "--initial-loss-scale",
        type=float,
        default=1024.0,
        help=(
            "initial FP16 gradient scale; 1024 avoided startup overflows in "
            "the roberta-large partial-unfreeze smoke tests"
        ),
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="eager",
        help="backbone attention implementation; SDPA uses less activation memory",
    )
    parser.add_argument(
        "--memory-smoke-steps",
        type=int,
        default=0,
        help=(
            "run this many optimizer steps, report CUDA peak memory, and exit "
            "without validation or checkpoint writes"
        ),
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "skip optimization and evaluate the pretrained MLM at oracle span "
            "length; writes results.json but no checkpoint"
        ),
    )
    parser.add_argument(
        "--evaluation-checkpoint",
        default="",
        help=(
            "optional state_dict to load for --evaluate-only; omitted evaluates "
            "the untouched pretrained model"
        ),
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--random-init-backbone", action="store_true")
    parser.add_argument("--native-vocabulary", action="store_true")
    parser.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=-1,
        help=(
            "number of top transformer layers to train; 0 freezes the full "
            "backbone and -1 trains every layer"
        ),
    )
    parser.add_argument(
        "--bottleneck-context", action="store_true",
        help="restrict the token pass to the single mask-token summary vector "
             "the interval chart is limited to, isolating encoder access from "
             "the objective (research/LIKELIHOOD_DECOMPOSITION.md)",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    if args.data_dir:
        config["data_dir"] = args.data_dir
    data_seed = int(config["seed"])
    training_seed = data_seed if args.seed < 0 else args.seed
    seed_everything(training_seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    if args.mixed_precision and device.type != "cuda":
        parser.error("--mixed-precision requires a CUDA device")
    micro_batch_size = args.micro_batch_size or args.batch_size
    if micro_batch_size < 1 or micro_batch_size > args.batch_size:
        parser.error("--micro-batch-size must be between 1 and --batch-size")
    if args.low_memory_optimizer and args.optimizer != "adamw":
        parser.error("--low-memory-optimizer applies only to AdamW")
    if args.evaluation_checkpoint and not args.evaluate_only:
        parser.error("--evaluation-checkpoint requires --evaluate-only")

    data_dir = str(config["data_dir"])
    manifest_path = os.path.join(data_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if bool(manifest.get("native_vocabulary", False)) != args.native_vocabulary:
            parser.error(
                "--native-vocabulary must match the prepared corpus manifest"
            )
    if args.native_vocabulary:
        source_tokenizer = AutoTokenizer.from_pretrained(
            data_dir, use_fast=True, local_files_only=True
        )
        vocab = vocabulary_from_pretrained_tokenizer(source_tokenizer)
    else:
        source_tokenizer = Tokenizer.from_file(
            os.path.join(data_dir, "tokenizer.json")
        )
        vocab = vocabulary_from_tokenizer(source_tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])

    # Identical stream and splits to the pretrained tree run.
    source = DynamicTextExampleDataset(
        corpus["train"], seed=training_seed, gap_counts=(1,), min_span=1,
        max_span=args.max_span, random_window_min=window_min,
        random_window_max=window_max,
    )
    if args.max_train_examples:
        source.documents = source.documents[:args.max_train_examples]
    validation = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"], data_seed + 401, window_min, window_max
        ),
        data_seed + 201, gap_counts=(1,), min_span=1, max_span=args.max_span,
    )
    if args.max_validation_examples:
        validation = validation[:args.max_validation_examples]
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101, gap_counts=(1,), min_span=1, max_span=args.max_span,
    )[:args.examples]

    model = PretrainedLengthMaskedModel(
        vocab.vocab_size, args.max_span, vocab.GAP, vocab.PAD, source_tokenizer,
        model_name=args.model_name, cache_dir=args.cache_dir,
        max_length=args.max_length, local_files_only=args.local_files_only,
        random_init_backbone=args.random_init_backbone,
        bottleneck_context=args.bottleneck_context,
        native_vocabulary=args.native_vocabulary,
        attn_implementation=args.attention_implementation,
    )
    if args.evaluation_checkpoint:
        state = torch.load(
            args.evaluation_checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        del state
    model = model.to(device)
    try:
        active_backbone_layers = configure_trainable_backbone_layers(
            model.encoder.backbone, args.trainable_backbone_layers
        )
    except ValueError as error:
        parser.error(str(error))
    if args.gradient_checkpointing and active_backbone_layers:
        model.encoder.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.encoder.backbone.config.use_cache = False
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    backbone_layer_count = len(transformer_layers(model.encoder.backbone))
    print("pretrained masked baseline{}: {:,} parameters ({:,} trainable), "
          "{}/{} backbone layers trainable, {} train documents".format(
        " [bottleneck context]" if args.bottleneck_context else "",
        parameter_count(model), trainable_parameters, active_backbone_layers,
        backbone_layer_count, len(source)))

    if args.evaluate_only:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        evaluation_started = time.perf_counter()
        predictions = decode_oracle_length(
            model, test, vocab, device, args.eval_batch_size,
            args.mixed_precision,
        )
        oracle_metrics = lexical_sampling_metrics(
            test, [[row] for row in predictions], [[False] for _ in predictions]
        )
        decoded_metrics = decoded_span_metrics(test, predictions, source_tokenizer)
        test_token_nll = evaluate_token_nll(
            model, test, vocab, device, args.eval_batch_size, args.max_span,
            args.mixed_precision,
        )
        result = {
            "config": {
                **vars(args),
                "data_dir": config["data_dir"],
                "training_seed": training_seed,
            },
            "parameters": parameter_count(model),
            "trainable_parameters": trainable_parameters,
            "selected_epoch": 0,
            "history": [],
            "validation_token_nll": None,
            "test_token_nll": test_token_nll,
            "oracle_metrics": oracle_metrics,
            "decoded_oracle_metrics": decoded_metrics,
            "evaluation_seconds": time.perf_counter() - evaluation_started,
            "cuda_peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda"
                else None
            ),
            "cuda_peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                if device.type == "cuda"
                else None
            ),
        }
        os.makedirs(args.artifact_dir, exist_ok=True)
        with open(
            os.path.join(args.artifact_dir, "results.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(result, handle, indent=2)
        evaluation_label = (
            "checkpoint" if args.evaluation_checkpoint else "zero-shot"
        )
        print("{} test token NLL {:.4f}".format(
            evaluation_label, test_token_nll
        ))
        print(
            "oracle-length token accuracy {:.2%} | decoded exact {:.2%} | "
            "character edit similarity {:.4f}".format(
                oracle_metrics["matched_length_token_accuracy"],
                decoded_metrics["decoded_exact_span_probability"],
                decoded_metrics["character_edit_similarity"],
            )
        )
        if device.type == "cuda":
            print(
                "CUDA peak allocated {:.2f} GiB | peak reserved {:.2f} GiB"
                .format(
                    result["cuda_peak_allocated_gib"],
                    result["cuda_peak_reserved_gib"],
                )
            )
        return

    backbone_ids = {id(p) for p in model.encoder.backbone.parameters()}
    parameter_groups = [
        {"params": [p for p in model.parameters()
                    if id(p) in backbone_ids and p.requires_grad],
         "lr": args.backbone_lr},
        {"params": [p for p in model.parameters()
                    if id(p) not in backbone_ids and p.requires_grad],
         "lr": args.head_lr},
    ]
    if args.optimizer == "adafactor":
        optimizer = Adafactor(
            parameter_groups,
            lr=args.head_lr,
            beta1=0.9,
            weight_decay=args.weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
            foreach=False if args.low_memory_optimizer else None,
        )
    steps_per_epoch = math.ceil(len(source) / args.batch_size)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=args.initial_loss_scale,
        enabled=args.mixed_precision,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    smoke_started = time.perf_counter()

    history, best = [], None
    global_step = 0
    attempted_step = 0
    for epoch in range(args.epochs):
        source.set_epoch(epoch)
        loader = DataLoader(
            source, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda rows: rows,
        )
        model.train()
        if active_backbone_layers == 0:
            # A frozen representation should not drift through dropout.
            model.encoder.backbone.eval()
        running, seen = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            total_token_positions = sum(
                min(len(example.spans[0]), args.max_span)
                for example in batch
            )
            batch_length_value = 0.0
            batch_token_value = 0.0
            observed_token_positions = 0
            for start in range(0, len(batch), micro_batch_size):
                micro_batch = batch[start:start + micro_batch_size]
                length_loss = length_batch_loss(
                    model, micro_batch, vocab, device, args.max_span,
                    args.mixed_precision,
                )
                if not bool(torch.isfinite(length_loss)):
                    raise FloatingPointError(
                        "non-finite length loss at attempted step {}"
                        .format(attempted_step + 1)
                    )
                length_weight = len(micro_batch) / len(batch)
                scaler.scale(length_loss * length_weight).backward()
                batch_length_value += float(length_loss.detach()) * length_weight

                # Backward the length graph before constructing the token graph.
                # Both gradients still accumulate into the same optimizer step.
                token_loss, token_positions = token_batch_loss(
                    model, micro_batch, vocab, device, args.max_span,
                    args.mixed_precision,
                )
                if token_loss is not None and total_token_positions:
                    if not bool(torch.isfinite(token_loss)):
                        raise FloatingPointError(
                            "non-finite token loss at attempted step {}"
                            .format(attempted_step + 1)
                        )
                    token_weight = token_positions / total_token_positions
                    scaler.scale(token_loss * token_weight).backward()
                    batch_token_value += float(token_loss.detach()) * token_weight
                    observed_token_positions += token_positions
            if observed_token_positions != total_token_positions:
                raise ValueError(
                    "token target count changed during micro-batching: "
                    "expected {}, observed {}".format(
                        total_token_positions, observed_token_positions
                    )
                )
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0,
                foreach=False if args.low_memory_optimizer else None,
            )
            if not args.mixed_precision and not bool(torch.isfinite(gradient_norm)):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "non-finite FP32 gradient norm at attempted step {}"
                    .format(attempted_step + 1)
                )
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_skipped = scaler.get_scale() < scale_before_step
            if not step_skipped:
                scheduler.step()
                global_step += 1
            batch_value = batch_length_value + batch_token_value
            running += batch_value * len(batch)
            seen += len(batch)
            attempted_step += 1
            if args.memory_smoke_steps and device.type == "cuda":
                torch.cuda.synchronize(device)
                print(
                    "attempt {} update={}{} loss={:.4f} grad_norm={:.4f} "
                    "CUDA allocated={:.2f} GiB "
                    "reserved={:.2f} GiB peak_allocated={:.2f} GiB "
                    "peak_reserved={:.2f} GiB".format(
                        attempted_step,
                        global_step,
                        " [overflow skipped]" if step_skipped else "",
                        batch_value,
                        float(gradient_norm),
                        torch.cuda.memory_allocated(device) / (1024 ** 3),
                        torch.cuda.memory_reserved(device) / (1024 ** 3),
                        torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                        torch.cuda.max_memory_reserved(device) / (1024 ** 3),
                    ),
                    flush=True,
                )
                print(
                    "  elapsed={:.2f}s mean_attempt={:.3f}s".format(
                        time.perf_counter() - smoke_started,
                        (time.perf_counter() - smoke_started) / attempted_step,
                    ),
                    flush=True,
                )
            if args.memory_smoke_steps and global_step >= args.memory_smoke_steps:
                print("memory smoke complete; no checkpoint written", flush=True)
                return
        # The final batch's gradients otherwise occupy one full model copy
        # throughout validation.
        optimizer.zero_grad(set_to_none=True)
        validation_nll = evaluate_token_nll(
            model, validation, vocab, device, args.eval_batch_size, args.max_span,
            args.mixed_precision,
        )
        history.append({
            "epoch": epoch + 1,
            "training_loss": running / max(1, seen),
            "validation_token_nll": validation_nll,
        })
        marker = ""
        if best is None or validation_nll < best[1]:
            best = (epoch + 1, validation_nll)
            os.makedirs(args.artifact_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.artifact_dir, "masked.pt"),
            )
            marker = "  <- best"
        print("epoch {}/{} training_loss={:.4f} validation_token_nll={:.4f}{}".format(
            epoch + 1, args.epochs, running / max(1, seen), validation_nll, marker))

    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    state = torch.load(
        os.path.join(args.artifact_dir, "masked.pt"),
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state)
    del state
    predictions = decode_oracle_length(
        model, test, vocab, device, args.eval_batch_size, args.mixed_precision
    )
    oracle_metrics = lexical_sampling_metrics(
        test, [[row] for row in predictions], [[False] for _ in predictions]
    )
    decoded_metrics = decoded_span_metrics(test, predictions, source_tokenizer)
    test_token_nll = evaluate_token_nll(
        model, test, vocab, device, args.eval_batch_size, args.max_span,
        args.mixed_precision,
    )

    result = {
        "config": {**vars(args), "data_dir": config["data_dir"],
                   "training_seed": training_seed},
        "parameters": parameter_count(model),
        "selected_epoch": best[0] if best else 0,
        "history": history,
        "validation_token_nll": best[1] if best else None,
        "test_token_nll": test_token_nll,
        "oracle_metrics": oracle_metrics,
        "decoded_oracle_metrics": decoded_metrics,
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print("\nselected epoch {} | test token NLL {:.4f}".format(
        result["selected_epoch"], test_token_nll))
    print("oracle-length token accuracy {:.2%} | edit similarity {:.4f}".format(
        oracle_metrics["matched_length_token_accuracy"],
        oracle_metrics["matched_length_edit_similarity"],
    ))


if __name__ == "__main__":
    main()
