"""Prepare a document-split byte-level BPE corpus for the natural-text pilot."""

import argparse
import json
import os

import torch

from gtdlm.text_tokenizer import (
    SPECIAL_TOKENS,
    split_documents,
    tokenize_documents,
    train_bpe_tokenizer,
    vocabulary_from_tokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="UTF-8 text, one document per line")
    parser.add_argument("--output-dir", default="artifacts/text_pilot")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-document-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        documents = [line.strip() for line in handle if line.strip()]
    split = split_documents(
        documents,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    tokenizer = train_bpe_tokenizer(split["train"], args.vocab_size)
    vocab = vocabulary_from_tokenizer(tokenizer)
    tokenized = {
        name: tokenize_documents(
            tokenizer, split_documents_for_name, args.max_document_tokens
        )
        for name, split_documents_for_name in split.items()
    }
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    corpus_path = os.path.join(args.output_dir, "corpus.pt")
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    tokenizer.save(tokenizer_path)
    torch.save(tokenized, corpus_path)
    manifest = {
        "seed": args.seed,
        "source": os.path.abspath(args.input),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": vocab.vocab_size,
        "max_document_tokens": args.max_document_tokens,
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
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

