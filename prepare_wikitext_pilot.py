"""Download and prepare the official WikiText-2 raw splits for the pilot."""

import argparse
import json
import os
import random
from typing import Dict, List, Sequence

import datasets
import torch

from gtdlm.text_tokenizer import (
    SPECIAL_TOKENS,
    tokenize_documents,
    train_bpe_tokenizer,
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
    args = parser.parse_args()

    corpus = datasets.load_dataset(DATASET_NAME, DATASET_CONFIG)
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

    tokenizer = train_bpe_tokenizer(documents["train"], args.vocab_size)
    vocab = vocabulary_from_tokenizer(tokenizer)
    tokenized = {
        name: tokenize_documents(
            tokenizer, split_documents, args.max_document_tokens
        )
        for name, split_documents in documents.items()
    }
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    corpus_path = os.path.join(args.output_dir, "corpus.pt")
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    tokenizer.save(tokenizer_path)
    torch.save(tokenized, corpus_path)
    manifest = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "datasets_version": datasets.__version__,
        "seed": args.seed,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": vocab.vocab_size,
        "max_document_tokens": args.max_document_tokens,
        "limits": limits,
        "special_tokens": list(SPECIAL_TOKENS),
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

