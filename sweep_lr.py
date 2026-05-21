"""
LR sweep over {1e-5, 2e-5, 3e-5, 5e-5} for both tasks.
Trains on train split, evaluates on val split only — test set is never touched.
Output: results/lr_sweep.json
"""

import json
import os

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as seqeval_f1
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

import utils
from config import (
    BASE_MODEL, BATCH_SIZE, LANGUAGE, MAX_SEQ_LEN,
    NER_DATASET, NEWS_DATASET, RESULTS_DIR, SEED,
    WARMUP_STEPS, WEIGHT_DECAY,
)

LR_VALUES  = [1e-5, 2e-5, 3e-5, 5e-5]
NUM_EPOCHS = 5
PATIENCE   = 2


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels         = labels

    def __len__(self): return len(self.labels)

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

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.input_ids[idx],      dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx],         dtype=torch.long),
        }


def run_news(lr: float, device: torch.device) -> dict:
    utils.set_seed(SEED)

    dataset  = load_dataset(NEWS_DATASET, LANGUAGE)
    features = dataset["train"].features
    if "label" in features and hasattr(features["label"], "names"):
        label_names = features["label"].names
        _label_fn   = lambda split: split["label"]
    else:
        label_names = sorted(set(dataset["train"]["category"]))
        _lbl2id     = {l: i for i, l in enumerate(label_names)}
        _label_fn   = lambda split: [_lbl2id[c] for c in split["category"]]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label_names), ignore_mismatched_sizes=True
    ).to(device)

    def make_loader(split_name, shuffle):
        split = dataset[split_name]
        enc   = tokenizer(split["headline"], truncation=True,
                          max_length=MAX_SEQ_LEN, padding="max_length")
        return DataLoader(NewsDataset(enc, _label_fn(split)),
                          batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader("train",      shuffle=True)
    val_loader   = make_loader("validation", shuffle=False)

    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler   = utils.get_scheduler(optimizer, WARMUP_STEPS, len(train_loader) * NUM_EPOCHS)

    best_val_f1, patience_cnt = 0.0, 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"  LR={lr:.0e} E{epoch}", leave=False):
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
                preds_all.extend(torch.argmax(out.logits, -1).cpu().tolist())
                labels_all.extend(batch["labels"].tolist())

        val_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
        history.append({"epoch": epoch, "val_f1": float(val_f1)})
        print(f"  [news lr={lr:.0e}] epoch {epoch} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    return {"lr": lr, "task": "news", "best_val_f1": best_val_f1, "history": history}


def run_ner(lr: float, device: torch.device) -> dict:
    utils.set_seed(SEED)

    dataset    = load_dataset(NER_DATASET, LANGUAGE, trust_remote_code=True)
    label_list = dataset["train"].features["ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label_list), ignore_mismatched_sizes=True
    ).to(device)

    def make_loader(split_name, shuffle):
        split   = dataset[split_name]
        enc     = tokenizer(split["tokens"], is_split_into_words=True,
                            truncation=True, max_length=MAX_SEQ_LEN,
                            padding="max_length")
        aligned = utils.align_ner_labels(enc, split["ner_tags"])
        return DataLoader(NERDataset(enc["input_ids"], enc["attention_mask"], aligned),
                          batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader("train",      shuffle=True)
    val_loader   = make_loader("validation", shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = utils.get_scheduler(optimizer, WARMUP_STEPS, len(train_loader) * NUM_EPOCHS)

    best_val_f1, patience_cnt = 0.0, 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"  LR={lr:.0e} E{epoch}", leave=False):
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        model.eval()
        pred_seqs, true_seqs = [], []
        with torch.no_grad():
            for batch in val_loader:
                out      = model(input_ids=batch["input_ids"].to(device),
                                 attention_mask=batch["attention_mask"].to(device))
                pred_ids = torch.argmax(out.logits, -1).cpu()
                for pred_row, true_row in zip(pred_ids.tolist(), batch["labels"].tolist()):
                    p, t = [], []
                    for pi, ti in zip(pred_row, true_row):
                        if ti != -100:
                            p.append(label_list[pi])
                            t.append(label_list[ti])
                    pred_seqs.append(p); true_seqs.append(t)

        val_f1 = seqeval_f1(true_seqs, pred_seqs, zero_division=0)
        history.append({"epoch": epoch, "val_f1": float(val_f1)})
        print(f"  [ner  lr={lr:.0e}] epoch {epoch} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1; patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    return {"lr": lr, "task": "ner", "best_val_f1": best_val_f1, "history": history}


def main():
    utils.set_seed(SEED)
    device = utils.get_device()

    print("=" * 60)
    print("LR SWEEP")
    print(f"  LR values: {LR_VALUES}")
    print(f"  Tasks: news, ner  |  Max epochs: {NUM_EPOCHS}, patience: {PATIENCE}")
    print("=" * 60)

    news_results = [run_news(lr, device) for lr in LR_VALUES]
    ner_results  = [run_ner(lr, device)  for lr in LR_VALUES]

    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)
    print(f"\n{'LR':<10}  {'News val-F1':>12}  {'NER val-F1':>12}")
    print("-" * 38)
    for n, r in zip(news_results, ner_results):
        marker_n = " <-- best" if n["best_val_f1"] == max(x["best_val_f1"] for x in news_results) else ""
        marker_r = " <-- best" if r["best_val_f1"] == max(x["best_val_f1"] for x in ner_results)  else ""
        print(f"{n['lr']:<10.0e}  {n['best_val_f1']:>12.4f}{marker_n}  {r['best_val_f1']:>12.4f}{marker_r}")

    sweep_results = {
        "lr_values":    LR_VALUES,
        "news":         news_results,
        "ner":          ner_results,
        "best_news_lr": max(news_results, key=lambda x: x["best_val_f1"])["lr"],
        "best_ner_lr":  max(ner_results,  key=lambda x: x["best_val_f1"])["lr"],
    }

    out_path = os.path.join(RESULTS_DIR, "lr_sweep.json")
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
