"""
Baseline: fine-tune mmBERT-small on MasakhaNews (Yoruba) topic classification.
Metric: macro-F1 on test split.
Output: checkpoints/news_baseline/  results/news_baseline.json
"""

import os

import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import utils
from config import (
    BASE_MODEL,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOP_PATIENCE,
    LANGUAGE,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    NEWS_DATASET,
    NUM_EPOCHS,
    SEED,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)


class NewsDataset(Dataset):
    def __init__(self, encodings: dict, labels: list):
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


def evaluate_news(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds_all.extend(torch.argmax(out.logits, dim=-1).cpu().tolist())
            labels_all.extend(batch["labels"].tolist())
    return preds_all, labels_all


def main(
    model_dir: str = BASE_MODEL,
    ckpt_name: str = "news_baseline",
    result_name: str = "news_baseline",
):
    utils.set_seed(SEED)
    device = utils.get_device()

    config_dict = {
        "model":         model_dir,
        "dataset":       f"{NEWS_DATASET}/{LANGUAGE}",
        "max_seq_len":   MAX_SEQ_LEN,
        "lr":            LEARNING_RATE,
        "batch_size":    BATCH_SIZE,
        "epochs":        NUM_EPOCHS,
        "warmup_steps":  WARMUP_STEPS,
        "weight_decay":  WEIGHT_DECAY,
        "patience":      EARLY_STOP_PATIENCE,
        "seed":          SEED,
    }

    print("=" * 60)
    print(f"TRAINING — MasakhaNews  [{result_name}]")
    print("=" * 60)
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    print("\nLoading dataset ...")
    dataset  = load_dataset(NEWS_DATASET, LANGUAGE)
    features = dataset["train"].features

    # datasets <4 with loading script → integer ClassLabel 'label'
    # datasets 4+ parquet-only        → string 'category' field
    if "label" in features and hasattr(features["label"], "names"):
        label_names = features["label"].names
        _label_fn   = lambda split: split["label"]
    else:
        label_names = sorted(set(dataset["train"]["category"]))
        _lbl2id     = {l: i for i, l in enumerate(label_names)}
        _label_fn   = lambda split: [_lbl2id[c] for c in split["category"]]

    num_labels = len(label_names)
    print(f"Labels ({num_labels}): {label_names}")
    print(f"Split sizes — train: {len(dataset['train'])}, "
          f"val: {len(dataset['validation'])}, test: {len(dataset['test'])}")

    print(f"\nLoading tokenizer + model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels, ignore_mismatched_sizes=True
    ).to(device)

    print("Tokenising ...")

    def make_loader(split_name: str, shuffle: bool) -> DataLoader:
        split = dataset[split_name]
        enc   = tokenizer(
            split["headline"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        ds = NewsDataset(enc, _label_fn(split))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader("train",      shuffle=True)
    val_loader   = make_loader("validation", shuffle=False)
    test_loader  = make_loader("test",       shuffle=False)

    optimizer    = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * NUM_EPOCHS
    scheduler    = utils.get_scheduler(optimizer, WARMUP_STEPS, total_steps)

    ckpt_path    = os.path.abspath(os.path.join(CHECKPOINT_DIR, ckpt_name))
    best_val_f1  = 0.0
    patience_cnt = 0

    print("\nStarting training ...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss          = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_preds, val_lbls = evaluate_news(model, val_loader, device)
        val_f1              = f1_score(val_lbls, val_preds, average="macro", zero_division=0)

        print(f"Epoch {epoch:2d} | loss={train_loss:.4f} | val_macro_f1={val_f1:.4f}", end="")

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
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)

    test_preds, test_lbls = evaluate_news(model, test_loader, device)
    test_f1               = f1_score(test_lbls, test_preds, average="macro", zero_division=0)
    per_class_f1          = f1_score(test_lbls, test_preds, average=None,    zero_division=0)

    print("\n" + "=" * 60)
    print(f"TEST RESULTS — {result_name}")
    print("=" * 60)
    print(f"Test Macro-F1: {test_f1:.4f}\n")
    print(classification_report(test_lbls, test_preds, target_names=label_names, zero_division=0))

    metrics = {
        "best_val_macro_f1": float(best_val_f1),
        "test_macro_f1":     float(test_f1),
        "per_class_f1":      {label_names[i]: float(per_class_f1[i]) for i in range(num_labels)},
        "label_names":       label_names,
        "test_predictions":  test_preds,   # saved for confusion matrix in analysis notebook
        "test_true_labels":  test_lbls,
        "checkpoint":        ckpt_path,
    }
    utils.save_results(result_name, metrics, config_dict)

    print(f"\nSummary | best_val_f1={best_val_f1:.4f} | test_f1={test_f1:.4f} | ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
