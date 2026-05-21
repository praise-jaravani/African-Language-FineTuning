"""Shared helpers used across all training and evaluation scripts."""

import os
import json
import random
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as seqeval_f1
from transformers import get_linear_schedule_with_warmup


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    return device


def align_ner_labels(tokenized_inputs, labels, label_all_tokens: bool = False):
    """Align word-level NER labels to subword tokens; special tokens and continuations get -100."""
    aligned = []
    for i, label_seq in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label_seq[word_idx])
            else:
                label_ids.append(label_seq[word_idx] if label_all_tokens else -100)
            prev_word_idx = word_idx
        aligned.append(label_ids)
    return aligned


def compute_macro_f1(predictions, labels) -> float:
    """Macro-F1, filtering out -100 positions. Accepts flat or nested lists."""
    flat_preds, flat_labels = [], []
    if predictions and isinstance(predictions[0], (list, np.ndarray)):
        for pred_seq, label_seq in zip(predictions, labels):
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    flat_preds.append(p)
                    flat_labels.append(l)
    else:
        for p, l in zip(predictions, labels):
            if l != -100:
                flat_preds.append(p)
                flat_labels.append(l)
    return float(f1_score(flat_labels, flat_preds, average="macro", zero_division=0))


def compute_ner_f1(predictions, labels, label_list) -> float:
    """Seqeval span-level F1 from integer label sequences."""
    true_seqs, pred_seqs = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        true_tags, pred_tags = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_tags.append(label_list[l])
                pred_tags.append(label_list[p])
        true_seqs.append(true_tags)
        pred_seqs.append(pred_tags)
    return float(seqeval_f1(true_seqs, pred_seqs, zero_division=0))


def save_results(run_name: str, metrics_dict: dict, config_dict: dict = None) -> None:
    """Save metrics to results/{run_name}.json with an ISO timestamp."""
    from config import RESULTS_DIR
    data = dict(metrics_dict)
    data["timestamp"] = datetime.utcnow().isoformat()
    if config_dict is not None:
        data["config"] = config_dict
    path = os.path.join(RESULTS_DIR, f"{run_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved -> {path}")


def get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear schedule with warmup (standard for BERT fine-tuning)."""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
