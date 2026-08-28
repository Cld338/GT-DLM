"""Reproducible document splitting and BPE preparation for the text pilot."""

import random
from typing import Dict, List, Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .text_data import TextVocabulary


SPECIAL_TOKENS = ("[PAD]", "[GAP]", "[MASK]", "[LEFT]", "[RIGHT]", "[UNK]")


def split_documents(
    documents: Sequence[str],
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> Dict[str, List[str]]:
    """Deterministically split whole documents, never individual token windows."""
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction > 1.0:
        raise ValueError("train and validation fractions exceed one")
    shuffled = [document for document in documents if document.strip()]
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_count = min(count, max(1, int(count * train_fraction))) if count else 0
    validation_count = int(count * validation_fraction)
    if count >= 3 and validation_fraction > 0 and validation_count == 0:
        validation_count = 1
    validation_count = min(validation_count, count - train_count)
    train = shuffled[:train_count]
    validation = shuffled[train_count : train_count + validation_count]
    test = shuffled[train_count + validation_count :]
    return {"train": train, "validation": validation, "test": test}


def train_bpe_tokenizer(
    train_documents: Sequence[str], vocab_size: int
) -> Tokenizer:
    """Train byte-level BPE with stable structural-token ids 0 through 4."""
    if not train_documents:
        raise ValueError("at least one training document is required")
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("vocab_size is smaller than the special-token set")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=False,
    )
    tokenizer.train_from_iterator(train_documents, trainer=trainer)
    return tokenizer


def vocabulary_from_tokenizer(tokenizer: Tokenizer) -> TextVocabulary:
    ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(value is None for value in ids.values()):
        raise ValueError("tokenizer is missing required special tokens")
    return TextVocabulary(
        tokenizer.get_vocab_size(),
        PAD=int(ids["[PAD]"]),
        GAP=int(ids["[GAP]"]),
        MASK=int(ids["[MASK]"]),
        LEFT=int(ids["[LEFT]"]),
        RIGHT=int(ids["[RIGHT]"]),
    )


def vocabulary_from_pretrained_tokenizer(tokenizer) -> TextVocabulary:
    """Use a Hugging Face masked-LM tokenizer as the model action space.

    The mask token doubles as the observed gap marker. Documents themselves
    are encoded without special tokens; ``TextInfillingExample.prompt`` adds
    the tokenizer's BOS/EOS pair around the corrupted sequence.
    """
    required = {
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "pretrained tokenizer is missing required ids: {}".format(
                ", ".join(missing)
            )
        )
    left = tokenizer.bos_token_id
    if left is None:
        left = tokenizer.cls_token_id
    right = tokenizer.eos_token_id
    if right is None:
        right = tokenizer.sep_token_id
    if left is None or right is None:
        raise ValueError("pretrained tokenizer must define BOS/EOS or CLS/SEP ids")
    specials = tuple(int(token) for token in tokenizer.all_special_ids)
    return TextVocabulary(
        len(tokenizer),
        PAD=int(tokenizer.pad_token_id),
        GAP=int(tokenizer.mask_token_id),
        MASK=int(tokenizer.mask_token_id),
        LEFT=int(left),
        RIGHT=int(right),
        EXTRA_STRUCTURAL=specials,
    )


def tokenize_documents(
    tokenizer: Tokenizer,
    documents: Sequence[str],
    max_document_tokens: int,
) -> List[List[int]]:
    if max_document_tokens < 1:
        raise ValueError("max_document_tokens must be positive")
    return [
        tokenizer.encode(document).ids[:max_document_tokens]
        for document in documents
        if document.strip()
    ]


def tokenize_pretrained_documents(
    tokenizer, documents: Sequence[str], max_document_tokens: int
) -> List[List[int]]:
    """Tokenize corpus text in a pretrained model's native vocabulary."""
    if max_document_tokens < 1:
        raise ValueError("max_document_tokens must be positive")
    return [
        tokenizer(
            document,
            add_special_tokens=False,
            truncation=True,
            max_length=max_document_tokens,
        )["input_ids"]
        for document in documents
        if document.strip()
    ]
