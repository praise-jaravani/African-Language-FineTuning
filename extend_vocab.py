"""
Extension B: extend the mmBERT-small tokeniser with Yoruba-specific BPE tokens
and initialise their embeddings as the mean of constituent subword embeddings.
Output: checkpoints/extended_model/
"""

import os
from collections import Counter

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import AutoModel, AutoTokenizer

from config import (
    BASE_MODEL,
    EXT_MODEL_DIR,
    LANGUAGE,
    NER_DATASET,
    NEWS_DATASET,
    VOCAB_MIN_FREQ,
    VOCAB_NEW_TOKENS,
)

DIACRITIC_WORDS = ["ọmọ", "ẹ̀kọ́", "Àdùnní", "ìwé", "oúnjẹ", "ìlú", "ẹ̀gbọ́n", "àárọ̀", "ọjọ́", "ìgbà"]


def collect_corpus() -> list:
    print("\n[Step 1] Collecting Yoruba corpus ...")
    news = load_dataset(NEWS_DATASET, LANGUAGE, split="train")
    ner  = load_dataset(NER_DATASET,  LANGUAGE, split="train", trust_remote_code=True)

    corpus = []
    corpus.extend(news["headline"])
    corpus.extend([" ".join(toks) for toks in ner["tokens"]])

    n_words = sum(len(s.split()) for s in corpus)
    print(f"  Sentences: {len(corpus):,}  |  Words: {n_words:,}")
    return corpus


def train_bpe(corpus: list) -> Tokenizer:
    print(f"\n[Step 2] Training BPE tokeniser "
          f"(vocab_size={VOCAB_NEW_TOKENS}, min_freq={VOCAB_MIN_FREQ}) ...")

    bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    bpe_tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_NEW_TOKENS,
        min_frequency=VOCAB_MIN_FREQ,
        special_tokens=["[UNK]"],
        show_progress=True,
    )
    bpe_tokenizer.train_from_iterator(corpus, trainer=trainer)

    print(f"  BPE vocab size after training: {bpe_tokenizer.get_vocab_size():,}")
    return bpe_tokenizer


def find_new_tokens(bpe_tokenizer: Tokenizer, original_tokenizer) -> list:
    """Return tokens in the Yoruba BPE vocab that are absent from mmBERT-small (length >= 2)."""
    print("\n[Step 3] Identifying new tokens ...")

    bpe_vocab      = set(bpe_tokenizer.get_vocab().keys())
    original_vocab = set(original_tokenizer.get_vocab().keys())

    new_tokens = sorted([
        t for t in bpe_vocab - original_vocab
        if len(t) >= 2 and t not in {"[UNK]"}
    ])

    print(f"  BPE vocab size:      {len(bpe_vocab):,}")
    print(f"  mmBERT vocab size:   {len(original_vocab):,}")
    print(f"  New tokens (len>=2): {len(new_tokens):,}")
    if new_tokens:
        print(f"  First 20 examples:  {new_tokens[:20]}")

    return new_tokens


def extend_tokenizer(original_tokenizer, new_tokens: list):
    print(f"\n[Step 4] Extending tokeniser with {len(new_tokens):,} new tokens ...")

    vocab_before = original_tokenizer.vocab_size
    n_added      = original_tokenizer.add_tokens(new_tokens)
    vocab_after  = len(original_tokenizer)

    print(f"  Vocab before: {vocab_before:,}")
    print(f"  Tokens added: {n_added:,}")
    print(f"  Vocab after:  {vocab_after:,}")

    original_tokenizer.save_pretrained(EXT_MODEL_DIR)
    print(f"  Extended tokeniser saved -> {EXT_MODEL_DIR}")

    return original_tokenizer, vocab_before


def extend_embeddings(extended_tokenizer, vocab_before: int, new_tokens: list):
    """
    Resize the embedding matrix and initialise each new token as the mean of
    the subword embeddings the original tokeniser produces for that token string.
    This preserves semantic signal from pre-training rather than using random init.
    """
    print(f"\n[Step 5] Extending model embeddings ...")

    model = AutoModel.from_pretrained(BASE_MODEL)
    original_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model.resize_token_embeddings(len(extended_tokenizer))
    embedding_weight = model.get_input_embeddings().weight.data

    n_mean_init = 0
    n_rand_init = 0

    with torch.no_grad():
        for new_token in new_tokens:
            new_id       = extended_tokenizer.convert_tokens_to_ids(new_token)
            original_ids = original_tokenizer(new_token, add_special_tokens=False)["input_ids"]

            if original_ids:
                embedding_weight[new_id] = embedding_weight[original_ids].mean(dim=0)
                n_mean_init += 1
            else:
                # Fallback: random init already set by resize_token_embeddings
                n_rand_init += 1

    print(f"  Mean-init embeddings:   {n_mean_init:,}")
    print(f"  Random-init embeddings: {n_rand_init:,} (no original subwords found)")

    model.save_pretrained(EXT_MODEL_DIR)
    print(f"  Extended model saved -> {EXT_MODEL_DIR}")

    return model


def verify(extended_tokenizer):
    print("\n[Step 6] Verification — reloading from EXT_MODEL_DIR ...")
    reloaded = AutoTokenizer.from_pretrained(EXT_MODEL_DIR)
    orig     = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("\nDiacritic word tokenisation comparison:")
    print(f"  {'Word':<14}  {'Before (mmBERT)':<40}  {'After (extended)'}")
    print("  " + "-" * 80)
    for word in DIACRITIC_WORDS:
        before = orig.tokenize(word)
        after  = reloaded.tokenize(word)
        try:
            print(f"  {word:<14}  {str(before):<40}  {after}")
        except UnicodeEncodeError:
            safe = word.encode("ascii", "replace").decode()
            print(f"  {safe:<14}  {str(before):<40}  {after}")

    print(f"\nVocab size: {orig.vocab_size:,} -> {len(reloaded):,}")


def main():
    print("=" * 60)
    print("EXTENSION B — Vocabulary Adaptation")
    print(f"  Base model:       {BASE_MODEL}")
    print(f"  Target vocab add: {VOCAB_NEW_TOKENS}")
    print(f"  Min frequency:    {VOCAB_MIN_FREQ}")
    print(f"  Output dir:       {EXT_MODEL_DIR}")
    print("=" * 60)

    corpus = collect_corpus()
    bpe_tokenizer = train_bpe(corpus)

    original_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    new_tokens         = find_new_tokens(bpe_tokenizer, original_tokenizer)

    if not new_tokens:
        print("\nWARNING: No new tokens found. "
              "Try increasing VOCAB_NEW_TOKENS in config.py.")

    extended_tokenizer, vocab_before = extend_tokenizer(original_tokenizer, new_tokens)
    extend_embeddings(extended_tokenizer, vocab_before, new_tokens)
    verify(extended_tokenizer)

    print("\nExtension B complete.")
    print(f"Extended model + tokeniser saved to: {EXT_MODEL_DIR}")
    print("Next steps: run train_news_extb.py and train_ner_extb.py")


if __name__ == "__main__":
    main()
