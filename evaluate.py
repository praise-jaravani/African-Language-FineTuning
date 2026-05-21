"""
Standalone evaluation — loads a saved checkpoint and reports test-split metrics.

Usage:
    python evaluate.py --checkpoint checkpoints/news_baseline --task news
    python evaluate.py --checkpoint checkpoints/ner_baseline  --task ner
    python evaluate.py --checkpoint checkpoints/news_extb     --task news
    python evaluate.py --checkpoint checkpoints/ner_extb      --task ner
"""

import argparse
import os

import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

import utils
from config import (
    BATCH_SIZE,
    LANGUAGE,
    MAX_SEQ_LEN,
    NER_DATASET,
    NEWS_DATASET,
    RESULTS_DIR,
    SEED,
)


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels         = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.input_ids[idx],      dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx],         dtype=torch.long),
        }


class NERDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.labels         = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.input_ids[idx],      dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx],         dtype=torch.long),
        }


def eval_news(checkpoint: str, device: torch.device) -> dict:
    checkpoint  = os.path.abspath(checkpoint)
    print(f"\nEvaluating news checkpoint: {checkpoint}")
    tokenizer   = AutoTokenizer.from_pretrained(checkpoint)
    model       = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    dataset  = load_dataset(NEWS_DATASET, LANGUAGE)
    features = dataset["train"].features
    if "label" in features and hasattr(features["label"], "names"):
        label_names = features["label"].names
        test_labels = dataset["test"]["label"]
    else:
        label_names = sorted(set(dataset["train"]["category"]))
        _lbl2id     = {l: i for i, l in enumerate(label_names)}
        test_labels = [_lbl2id[c] for c in dataset["test"]["category"]]
    test_split  = dataset["test"]

    enc = tokenizer(
        test_split["headline"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    loader = DataLoader(
        NewsDataset(enc, test_labels),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="News eval"):
            out = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
            )
            preds_all.extend(torch.argmax(out.logits, -1).cpu().tolist())
            labels_all.extend(batch["labels"].tolist())

    test_f1      = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels_all, preds_all, average=None,    zero_division=0)

    print(f"Test Macro-F1: {test_f1:.4f}")
    print(classification_report(labels_all, preds_all, target_names=label_names, zero_division=0))

    return {
        "task":          "news",
        "checkpoint":    checkpoint,
        "test_macro_f1": float(test_f1),
        "per_class_f1":  {label_names[i]: float(per_class_f1[i]) for i in range(len(label_names))},
        "label_names":   label_names,
    }


def eval_ner(checkpoint: str, device: torch.device) -> dict:
    checkpoint = os.path.abspath(checkpoint)
    print(f"\nEvaluating NER checkpoint: {checkpoint}")
    tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
    model      = AutoModelForTokenClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    dataset    = load_dataset(NER_DATASET, LANGUAGE, trust_remote_code=True)
    label_list = dataset["train"].features["ner_tags"].feature.names
    test_split = dataset["test"]

    enc     = tokenizer(
        test_split["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    aligned = utils.align_ner_labels(enc, test_split["ner_tags"])
    loader  = DataLoader(
        NERDataset(enc["input_ids"], enc["attention_mask"], aligned),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    pred_seqs, true_seqs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="NER eval"):
            out      = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
            )
            pred_ids = torch.argmax(out.logits, -1).cpu()
            for pred_row, true_row in zip(pred_ids.tolist(), batch["labels"].tolist()):
                pred_tags, true_tags = [], []
                for p, t in zip(pred_row, true_row):
                    if t != -100:
                        pred_tags.append(label_list[p])
                        true_tags.append(label_list[t])
                pred_seqs.append(pred_tags)
                true_seqs.append(true_tags)

    test_f1 = seqeval_f1(true_seqs, pred_seqs, zero_division=0)
    print(f"Test Span-F1: {test_f1:.4f}")
    print(seqeval_report(true_seqs, pred_seqs, zero_division=0))

    return {
        "task":         "ner",
        "checkpoint":   checkpoint,
        "test_span_f1": float(test_f1),
        "label_list":   label_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Standalone checkpoint evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint directory")
    parser.add_argument("--task",       required=True, choices=["news", "ner"])
    args = parser.parse_args()

    utils.set_seed(SEED)
    device = utils.get_device()

    if args.task == "news":
        results = eval_news(args.checkpoint, device)
    else:
        results = eval_ner(args.checkpoint, device)

    # Save with eval_ prefix so these don't overwrite the training result files
    ckpt_name   = os.path.basename(args.checkpoint.rstrip("/\\"))
    result_name = f"eval_{args.task}_{ckpt_name}"
    utils.save_results(result_name, results)


if __name__ == "__main__":
    main()
