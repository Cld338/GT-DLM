"""Prepare matched BBC News slices for the backbone corpus-overlap control.

Two slices are built with an identical pipeline and one shared tokenizer:

* `legacy` uses months inside the CC-News window that RoBERTa was trained on,
  so its documents may have been seen by the pretrained backbone;
* `modern` uses months well after every corpus in that pretraining lineage,
  so its documents cannot have been.

Domain, register, publisher, document counts, document length, and vocabulary
are held fixed, so comparing the pretrained-minus-random-init gain across the
two slices separates pretraining-corpus overlap from domain shift.
"""

import argparse
import hashlib
import json
import os
import random
from typing import Dict, List, Sequence

import datasets
import torch

from gtdlm.text_tokenizer import (
    SPECIAL_TOKENS,
    train_bpe_tokenizer,
    vocabulary_from_tokenizer,
)


DATASET_NAME = "RealTimeData/bbc_news_alltime"


def load_articles(configs: Sequence[str], cache_dir: str) -> List[dict]:
    """Collect deduplicated articles across the requested month configs."""
    seen_links = set()
    seen_content = set()
    articles = []
    for config in configs:
        split = datasets.load_dataset(
            DATASET_NAME, config, split="train", cache_dir=cache_dir
        )
        for row in split:
            content = row.get("content") or ""
            link = row.get("link") or ""
            if not content.strip():
                continue
            digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
            if link in seen_links or digest in seen_content:
                continue
            seen_links.add(link)
            seen_content.add(digest)
            articles.append(
                {
                    "link": link,
                    "content": content,
                    "published_date": str(row.get("published_date")),
                    "config": config,
                }
            )
    return articles


def article_text(content: str) -> str:
    """Normalize one article to a single whitespace-joined string."""
    return " ".join(content.split())


def assign_articles(
    articles: Sequence[dict], seed: int, held_out_articles: int
) -> Dict[str, List[dict]]:
    """Partition articles between splits so that none straddles two of them."""
    ordered = list(articles)
    random.Random(seed).shuffle(ordered)
    # Held-out pools are reserved first, so any shortfall lands on training
    # rather than silently shrinking the evaluation sets.
    return {
        "test": ordered[:held_out_articles],
        "validation": ordered[held_out_articles : 2 * held_out_articles],
        "train": ordered[2 * held_out_articles :],
    }


def article_chunks(
    tokenizer,
    articles: Sequence[dict],
    limit: int,
    document_tokens: int,
    chunks_per_article: int,
) -> List[List[int]]:
    """Cut articles into equal-length token chunks until the split is filled.

    2017 and 2024 BBC articles are paragraphed very differently, so paragraph
    units cannot be matched across the two eras. Fixed-length chunks make the
    document length distribution identical by construction, which is what the
    downstream 24--96 token window sampler needs for the two arms of the
    control to be comparable. Partial trailing chunks are discarded.
    """
    documents: List[List[int]] = []
    for article in articles:
        if len(documents) >= limit:
            break
        tokens = tokenizer.encode(article_text(article["content"])).ids
        taken = 0
        for start in range(0, len(tokens), document_tokens):
            if taken >= chunks_per_article or len(documents) >= limit:
                break
            chunk = tokens[start : start + document_tokens]
            if len(chunk) < document_tokens:
                break
            documents.append(chunk)
            taken += 1
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-configs",
        default="2017-01,2017-02,2017-03,2017-04,2017-05,2017-06",
        help="months inside the RoBERTa CC-News window (through 2019-02)",
    )
    parser.add_argument(
        "--modern-configs",
        default="2024-07,2024-08,2024-09,2024-10,2024-11,2024-12",
        help="months after every corpus in the pretraining lineage",
    )
    parser.add_argument("--legacy-output-dir", default="artifacts/bbc_legacy_pilot")
    parser.add_argument("--modern-output-dir", default="artifacts/bbc_modern_pilot")
    parser.add_argument("--cache-dir", default=".hf_cache/datasets")
    parser.add_argument("--vocab-size", type=int, default=4000)
    parser.add_argument("--document-tokens", type=int, default=128)
    parser.add_argument("--max-train-documents", type=int, default=4000)
    parser.add_argument("--max-validation-documents", type=int, default=500)
    parser.add_argument("--max-test-documents", type=int, default=500)
    parser.add_argument("--held-out-articles", type=int, default=600)
    parser.add_argument("--chunks-per-article", type=int, default=2)
    parser.add_argument("--tokenizer-training-articles", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    limits = {
        "train": args.max_train_documents,
        "validation": args.max_validation_documents,
        "test": args.max_test_documents,
    }
    slices = {
        "legacy": {
            "configs": [c for c in args.legacy_configs.split(",") if c],
            "output_dir": args.legacy_output_dir,
        },
        "modern": {
            "configs": [c for c in args.modern_configs.split(",") if c],
            "output_dir": args.modern_output_dir,
        },
    }
    for name, entry in slices.items():
        articles = load_articles(entry["configs"], args.cache_dir)
        entry["articles"] = len(articles)
        entry["pools"] = assign_articles(articles, args.seed, args.held_out_articles)
        print(
            "{}: {} articles -> pools {}".format(
                name,
                len(articles),
                {k: len(v) for k, v in entry["pools"].items()},
            ),
            flush=True,
        )

    # One tokenizer for both slices: vocabulary and the induced length law must
    # not differ between the two arms of the control. It sees training-pool
    # articles only.
    shared_training_text = []
    for entry in slices.values():
        shared_training_text.extend(
            article_text(article["content"])
            for article in entry["pools"]["train"][: args.tokenizer_training_articles]
        )
    tokenizer = train_bpe_tokenizer(shared_training_text, args.vocab_size)
    vocab = vocabulary_from_tokenizer(tokenizer)

    for name, entry in slices.items():
        tokenized = {
            split_name: article_chunks(
                tokenizer,
                pool,
                limits[split_name],
                args.document_tokens,
                args.chunks_per_article,
            )
            for split_name, pool in entry["pools"].items()
        }
        shortfall = {
            split_name: limits[split_name] - len(value)
            for split_name, value in tokenized.items()
            if len(value) < limits[split_name]
        }
        if shortfall:
            raise RuntimeError(
                "{} slice is short of documents: {}".format(name, shortfall)
            )
        output_dir = entry["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
        torch.save(tokenized, os.path.join(output_dir, "corpus.pt"))
        manifest = {
            "dataset": DATASET_NAME,
            "slice": name,
            "configs": entry["configs"],
            "datasets_version": datasets.__version__,
            "seed": args.seed,
            "requested_vocab_size": args.vocab_size,
            "actual_vocab_size": vocab.vocab_size,
            "shared_tokenizer_training_articles": len(shared_training_text),
            "document_tokens": args.document_tokens,
            "chunks_per_article": args.chunks_per_article,
            "held_out_articles": args.held_out_articles,
            "limits": limits,
            "deduplicated_articles": entry["articles"],
            "special_tokens": list(SPECIAL_TOKENS),
            "special_token_ids": {
                "PAD": vocab.PAD,
                "GAP": vocab.GAP,
                "MASK": vocab.MASK,
                "LEFT": vocab.LEFT,
                "RIGHT": vocab.RIGHT,
            },
            "documents": {k: len(v) for k, v in tokenized.items()},
            "tokens": {
                k: sum(len(document) for document in v)
                for k, v in tokenized.items()
            },
        }
        with open(
            os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2)
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
