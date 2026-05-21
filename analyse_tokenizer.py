"""
Tokenizer fertility analysis — measures token-per-word ratio across the Yoruba corpus.

Usage:
    python analyse_tokenizer.py                                         # baseline mmBERT-small
    python analyse_tokenizer.py --tokenizer checkpoints/extended_model  # after Extension B
"""

import argparse
import json
from collections import Counter

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from config import BASE_MODEL, NEWS_DATASET, NER_DATASET, LANGUAGE, RESULTS_DIR

DIACRITIC_WORDS = ["ọmọ", "ẹ̀kọ́", "Àdùnní", "ìwé", "oúnjẹ", "ìlú", "ẹ̀gbọ́n", "àárọ̀", "ọjọ́", "ìgbà"]


def collect_corpus():
    print("Loading datasets...")
    news = load_dataset(NEWS_DATASET, LANGUAGE, split="train")
    ner  = load_dataset(NER_DATASET,  LANGUAGE, split="train", trust_remote_code=True)

    texts = list(news["headline"])
    texts += [" ".join(token_list) for token_list in ner["tokens"]]

    print(f"Corpus: {len(texts):,} sentences from news + NER train splits")
    return texts


def analyse(tokenizer_path: str, output_name: str) -> dict:
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    texts = collect_corpus()

    words = []
    for text in texts:
        words.extend(text.split())

    print(f"Analysing {len(words):,} words ...")
    tpw = np.array([len(tokenizer.tokenize(w)) for w in words], dtype=np.int32)
    tpw = np.where(tpw == 0, 1, tpw)   # guard against empty tokenisations

    mean_tpw   = float(np.mean(tpw))
    median_tpw = float(np.median(tpw))
    pct95_tpw  = float(np.percentile(tpw, 95))

    total  = len(tpw)
    counts = Counter(tpw.tolist())
    dist = {
        "1_token":   round(counts.get(1, 0) / total, 4),
        "2_tokens":  round(counts.get(2, 0) / total, 4),
        "3_tokens":  round(counts.get(3, 0) / total, 4),
        "4+_tokens": round(sum(v for k, v in counts.items() if k >= 4) / total, 4),
    }

    paired = sorted(zip(words, tpw.tolist()), key=lambda x: x[1], reverse=True)
    most_fragmented = [(w, t) for w, t in paired[:20]]

    diacritic_results = {}
    for word in DIACRITIC_WORDS:
        tokens = tokenizer.tokenize(word)
        diacritic_results[word] = {"tokens": tokens, "n_tokens": len(tokens)}

    print(f"\n{'='*60}")
    print(f"TOKENIZER FERTILITY — {tokenizer_path}")
    print(f"{'='*60}")
    print(f"Vocab size:          {tokenizer.vocab_size:,}")
    print(f"Words analysed:      {total:,}")
    print(f"Mean toks/word:      {mean_tpw:.3f}")
    print(f"Median toks/word:    {median_tpw:.3f}")
    print(f"95th percentile:     {pct95_tpw:.3f}")
    print(f"\nDistribution:")
    for k, v in dist.items():
        print(f"  {k:<12}: {v:.1%}")

    print(f"\n20 Most Fragmented Words:")
    for w, t in most_fragmented:
        try:
            print(f"  {t:2d} tokens  {w!r}")
        except UnicodeEncodeError:
            print(f"  {t:2d} tokens  {w.encode('ascii', 'replace').decode()!r}")

    print(f"\nDiacritic Word Tokenisation:")
    for word, info in diacritic_results.items():
        try:
            print(f"  {word:<12} -> {info['tokens']}")
        except UnicodeEncodeError:
            print(f"  {word.encode('ascii', 'replace').decode():<12} -> {info['tokens']}")

    results = {
        "tokenizer_path":         tokenizer_path,
        "vocab_size":             tokenizer.vocab_size,
        "n_sentences":            len(texts),
        "n_words":                total,
        "mean_tokens_per_word":   mean_tpw,
        "median_tokens_per_word": median_tpw,
        "pct95_tokens_per_word":  pct95_tpw,
        "distribution":           dist,
        "most_fragmented_20":     most_fragmented,
        "diacritic_words":        diacritic_results,
    }

    out_path = f"{RESULTS_DIR}{output_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved -> {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Tokenizer fertility analysis")
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Path to tokenizer directory (default: BASE_MODEL from config.py)"
    )
    args = parser.parse_args()

    if args.tokenizer:
        tokenizer_path = args.tokenizer
        output_name    = "tokenizer_analysis_extb"
    else:
        tokenizer_path = BASE_MODEL
        output_name    = "tokenizer_analysis_baseline"

    analyse(tokenizer_path, output_name)


if __name__ == "__main__":
    main()
