"""Download and prepare the official WikiText-2 raw splits for the pilot."""

import argparse
import json
import os
import random
from typing import Dict, List, Sequence

import datasets
import torch
from transformers import AutoTokenizer

from gtdlm.text_tokenizer import (
    SPECIAL_TOKENS,
    tokenize_documents,
    tokenize_pretrained_documents,
    train_bpe_tokenizer,
    vocabulary_from_pretrained_tokenizer,
    vocabulary_from_tokenizer,
)


DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"


def normalize_documents(rows: Sequence[str]) -> List[str]:
    """Treat each non-empty WikiText row as a document-like unit."""
    return [" ".join(row.split()) for row in rows if row.strip()]


def deterministic_limit(
    documents: Sequence[str], limit: int, seed: int
) -> List[str]:
    selected = list(documents)
    random.Random(seed).shuffle(selected)
    return selected if limit <= 0 else selected[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/wikitext_pilot")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-document-tokens", type=int, default=256)
    parser.add_argument("--max-train-documents", type=int, default=12000)
    parser.add_argument("--max-validation-documents", type=int, default=1000)
    parser.add_argument("--max-test-documents", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--native-vocabulary", action="store_true")
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--cache-dir", default=".hf_cache/hub")
    parser.add_argument(
        "--dataset-config",
        default=DATASET_CONFIG,
        help=(
            "HuggingFace wikitext config. The pilot corpus is "
            "wikitext-2-raw-v1, which yields about 287k training "
            "tokens; wikitext-103-raw-v1 is the large one"
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    corpus = datasets.load_dataset(DATASET_NAME, args.dataset_config)
    limits = {
        "train": args.max_train_documents,
        "validation": args.max_validation_documents,
        "test": args.max_test_documents,
    }
    documents: Dict[str, List[str]] = {}
    for split_name in ("train", "validation", "test"):
        normalized = normalize_documents(corpus[split_name]["text"])
        documents[split_name] = deterministic_limit(
            normalized,
            limits[split_name],
            seed=args.seed + {"train": 0, "validation": 1, "test": 2}[split_name],
        )

    if args.native_vocabulary:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            use_fast=True,
            local_files_only=args.local_files_only,
        )
        vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
        tokenize = tokenize_pretrained_documents
    else:
        tokenizer = train_bpe_tokenizer(documents["train"], args.vocab_size)
        vocab = vocabulary_from_tokenizer(tokenizer)
        tokenize = tokenize_documents
    tokenized = {
        name: tokenize(tokenizer, split_documents, args.max_document_tokens)
        for name, split_documents in documents.items()
    }
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    corpus_path = os.path.join(args.output_dir, "corpus.pt")
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    if args.native_vocabulary:
        tokenizer.save_pretrained(args.output_dir)
    else:
        tokenizer.save(tokenizer_path)
    torch.save(tokenized, corpus_path)
    manifest = {
        "dataset": DATASET_NAME,
        "config": args.dataset_config,
        "datasets_version": datasets.__version__,
        "seed": args.seed,
        "requested_vocab_size": (
            None if args.native_vocabulary else args.vocab_size
        ),
        "actual_vocab_size": vocab.vocab_size,
        "native_vocabulary": args.native_vocabulary,
        "model_name": args.model_name if args.native_vocabulary else None,
        "max_document_tokens": args.max_document_tokens,
        "limits": limits,
        "special_tokens": (
            list(tokenizer.all_special_tokens)
            if args.native_vocabulary
            else list(SPECIAL_TOKENS)
        ),
        "special_token_ids": {
            "PAD": vocab.PAD,
            "GAP": vocab.GAP,
            "MASK": vocab.MASK,
            "LEFT": vocab.LEFT,
            "RIGHT": vocab.RIGHT,
        },
        "documents": {name: len(value) for name, value in tokenized.items()},
        "tokens": {
            name: sum(len(document) for document in value)
            for name, value in tokenized.items()
        },
        "dataset_fingerprints": {
            name: corpus[name]._fingerprint for name in corpus.keys()
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
