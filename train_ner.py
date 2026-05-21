"""
Baseline: fine-tune mmBERT-small on MasakhaNER 2.0 (Yoruba) named entity recognition.
Metric: seqeval span-level F1 on test split.
Output: checkpoints/ner_baseline/  results/ner_baseline.json
"""

import os

import torch
from datasets import load_dataset
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

import utils
from config import (
    BASE_MODEL,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOP_PATIENCE,
    LANGUAGE,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    NER_DATASET,
    NUM_EPOCHS,
    SEED,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)


class NERDataset(Dataset):
    def __init__(self, input_ids: list, attention_mask: list, labels: list):
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


def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()

    return total_loss / len(loader)


def evaluate_ner(model, loader, label_list, device):
    model.eval()
    pred_seqs, true_seqs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            out      = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = torch.argmax(out.logits, dim=-1).cpu()

            for pred_row, true_row in zip(pred_ids.tolist(), labels.tolist()):
                pred_tags, true_tags = [], []
                for p, t in zip(pred_row, true_row):
                    if t != -100:
                        pred_tags.append(label_list[p])
                        true_tags.append(label_list[t])
                pred_seqs.append(pred_tags)
                true_seqs.append(true_tags)

    return pred_seqs, true_seqs


def main(
    model_dir: str = BASE_MODEL,
    ckpt_name: str = "ner_baseline",
    result_name: str = "ner_baseline",
):
    utils.set_seed(SEED)
    device = utils.get_device()

    config_dict = {
        "model":        model_dir,
        "dataset":      f"{NER_DATASET}/{LANGUAGE}",
        "max_seq_len":  MAX_SEQ_LEN,
        "lr":           LEARNING_RATE,
        "batch_size":   BATCH_SIZE,
        "epochs":       NUM_EPOCHS,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "patience":     EARLY_STOP_PATIENCE,
        "seed":         SEED,
    }

    print("=" * 60)
    print(f"TRAINING — MasakhaNER 2.0  [{result_name}]")
    print("=" * 60)
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    print("\nLoading dataset ...")
    dataset    = load_dataset(NER_DATASET, LANGUAGE, trust_remote_code=True)
    label_list = dataset["train"].features["ner_tags"].feature.names
    num_labels = len(label_list)
    print(f"Labels ({num_labels}): {label_list}")
    print(f"Split sizes — train: {len(dataset['train'])}, "
          f"val: {len(dataset['validation'])}, test: {len(dataset['test'])}")

    print(f"\nLoading tokenizer + model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForTokenClassification.from_pretrained(
        model_dir, num_labels=num_labels, ignore_mismatched_sizes=True
    ).to(device)

    print("Tokenising + aligning labels ...")

    def make_loader(split_name: str, shuffle: bool) -> DataLoader:
        split = dataset[split_name]
        enc   = tokenizer(
            split["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        aligned = utils.align_ner_labels(enc, split["ner_tags"])
        ds      = NERDataset(enc["input_ids"], enc["attention_mask"], aligned)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader("train",      shuffle=True)
    val_loader   = make_loader("validation", shuffle=False)
    test_loader  = make_loader("test",       shuffle=False)

    optimizer   = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler   = utils.get_scheduler(optimizer, WARMUP_STEPS, total_steps)

    ckpt_path    = os.path.abspath(os.path.join(CHECKPOINT_DIR, ckpt_name))
    best_val_f1  = 0.0
    patience_cnt = 0

    print("\nStarting training ...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss                   = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_pred_seqs, val_true_seqs = evaluate_ner(model, val_loader, label_list, device)
        val_f1                       = seqeval_f1(val_true_seqs, val_pred_seqs, zero_division=0)

        print(f"Epoch {epoch:2d} | loss={train_loss:.4f} | val_span_f1={val_f1:.4f}", end="")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  [saved] -> {ckpt_path}")
        else:
            patience_cnt += 1
            print(f"  (no improvement {patience_cnt}/{EARLY_STOP_PATIENCE})")
            if patience_cnt >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nLoading best checkpoint from {ckpt_path} ...")
    model = AutoModelForTokenClassification.from_pretrained(ckpt_path).to(device)

    test_pred_seqs, test_true_seqs = evaluate_ner(model, test_loader, label_list, device)
    test_f1                        = seqeval_f1(test_true_seqs, test_pred_seqs, zero_division=0)

    print("\n" + "=" * 60)
    print(f"TEST RESULTS — {result_name}")
    print("=" * 60)
    print(f"Test Span-F1: {test_f1:.4f}\n")
    print(seqeval_report(test_true_seqs, test_pred_seqs, zero_division=0))

    report_str = seqeval_report(test_true_seqs, test_pred_seqs, zero_division=0)
    per_entity = _parse_seqeval_report(report_str)

    metrics = {
        "best_val_span_f1":  float(best_val_f1),
        "test_span_f1":      float(test_f1),
        "per_entity_f1":     per_entity,
        "label_list":        label_list,
        "test_pred_tags":    test_pred_seqs,   # saved for per-entity analysis notebook
        "test_true_tags":    test_true_seqs,
        "checkpoint":        ckpt_path,
    }
    utils.save_results(result_name, metrics, config_dict)

    print(f"\nSummary | best_val_f1={best_val_f1:.4f} | test_f1={test_f1:.4f} | ckpt={ckpt_path}")


def _parse_seqeval_report(report_str: str) -> dict:
    """Extract per-entity F1 from a seqeval report string."""
    per_entity = {}
    for line in report_str.strip().splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] not in ("", "micro", "macro", "weighted"):
            try:
                per_entity[parts[0]] = float(parts[3])
            except ValueError:
                pass
    return per_entity


if __name__ == "__main__":
    main()
