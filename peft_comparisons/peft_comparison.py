"""
Extension D: Fine-tune mmBERT-small on MasakhaNews topic classification using PEFT methods (LoRA, IA3, Prompt Tuning).
Saves adapter checkpoints and metrics comparing parameter efficiency.

Usage:
    python peft_comparison.py --method lora --r 8
    python peft_comparison.py --method ia3
    python peft_comparison.py --method prompt --num_virtual_tokens 10
"""

import argparse
import os
import time

import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peft_comparisons import parser_configurations
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
    SEED,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from train_news import NewsDataset, evaluate_news, train_epoch


def main():
    args = parser_configurations.get_arguments()

    # Determine unique run and checkpoint name
    match args.method:
        case "lora":
            param_str = f"r{args.r}"
        case "prompt":
            param_str = f"v{args.num_virtual_tokens}"
        case _:
            param_str = "default"



    run_name = f"peft_{args.method}_{param_str}"
    ckpt_path = os.path.abspath(os.path.join(CHECKPOINT_DIR, run_name))

    utils.set_seed(SEED)
    device = utils.get_device()

    print("=" * 60)
    print(f"PEFT COMPARISON — MasakhaNews  [{run_name}]")
    print("=" * 60)
    print(f"  Method:               {args.method}")
    print(f"  Params:               {param_str}")
    print(f"  Learning Rate:        {args.lr}")
    print(f"  Batch Size:           {BATCH_SIZE}")
    print(f"  Max Epochs:           {args.epochs}")
    print(f"  Device:               {device}")
    print("=" * 60)

    # 1. Load dataset
    print("\nLoading dataset ...")
    dataset = load_dataset(NEWS_DATASET, LANGUAGE)
    features = dataset["train"].features

    if "label" in features and hasattr(features["label"], "names"):
        label_names = features["label"].names
        _label_fn = lambda split: split["label"]
    else:
        label_names = sorted(set(dataset["train"]["category"]))
        _lbl2id = {l: i for i, l in enumerate(label_names)}
        _label_fn = lambda split: [_lbl2id[c] for c in split["category"]]

    num_labels = len(label_names)
    print(f"Labels ({num_labels}): {label_names}")
    print(f"Split sizes — train: {len(dataset['train'])}, "
          f"val: {len(dataset['validation'])}, test: {len(dataset['test'])}")

    # 2. Load tokenizer and model
    print(f"\nLoading base tokenizer + model from {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    # 3. Apply PEFT Configuration
    match args.method:
        case "lora":

            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=args.r,
            lora_alpha=args.r * 2,
            lora_dropout=0.1,
            target_modules=["Wqkv", "Wo", "Wi"],
        )
            model = get_peft_model(model, peft_config)
        case "ia3":

            from peft import IA3Config, get_peft_model  

            peft_config = IA3Config(
            task_type="SEQ_CLS",
            target_modules=["Wqkv", "Wo"],
            feedforward_modules=["Wo"],
        )
            model = get_peft_model(model, peft_config)
        case "prompt":

            from peft import PromptTuningConfig, get_peft_model

            from peft import PromptTuningConfig, get_peft_model
            peft_config = PromptTuningConfig(
                task_type="SEQ_CLS",
                num_virtual_tokens=args.num_virtual_tokens,
            )
            model = get_peft_model(model, peft_config)




    trainable_params, all_params = model.get_nb_trainable_parameters()
    efficiency_pct = 100.0 * trainable_params / all_params
    print(f"  Trainable params: {trainable_params:,} / {all_params:,} ({efficiency_pct:.4f}%)")

    model = model.to(device)

    # 4. Prepare data loaders
    print("Tokenising ...")
    def make_loader(split_name: str, shuffle: bool) -> DataLoader:
        split = dataset[split_name]
        enc = tokenizer(
            split["headline"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        ds = NewsDataset(enc, _label_fn(split))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("validation", shuffle=False)
    test_loader = make_loader("test", shuffle=False)

    # 5. Optimizer, steps, and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    scheduler = utils.get_scheduler(optimizer, WARMUP_STEPS, total_steps)

    # 6. Training Loop with early stopping
    best_val_f1 = 0.0
    patience_cnt = 0
    training_start_time = time.time()

    print("\nStarting PEFT training ...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_preds, val_lbls = evaluate_news(model, val_loader, device)
        val_f1 = f1_score(val_lbls, val_preds, average="macro", zero_division=0)
        epoch_duration = time.time() - epoch_start

        print(f"Epoch {epoch:2d} | loss={train_loss:.4f} | val_macro_f1={val_f1:.4f} | time={epoch_duration:.1f}s", end="")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  [saved adapter] -> {ckpt_path}")
        else:
            patience_cnt += 1
            print(f"  (no improvement {patience_cnt}/{EARLY_STOP_PATIENCE})")
            if patience_cnt >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time:.1f}s.")

    # 7. Load best adapter checkpoint for evaluation (gotcha-proof loading)
    print(f"\nLoading best adapter from {ckpt_path} ...")
    from peft import PeftModel
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, ckpt_path).to(device)
    model.eval()

    # 8. Test split evaluation
    print("Evaluating on test split ...")
    test_preds, test_lbls = evaluate_news(model, test_loader, device)
    test_f1 = f1_score(test_lbls, test_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(test_lbls, test_preds, average=None, zero_division=0)

    print("\n" + "=" * 60)
    print(f"TEST RESULTS — {run_name}")
    print("=" * 60)
    print(f"Test Macro-F1: {test_f1:.4f}\n")
    print(classification_report(test_lbls, test_preds, target_names=label_names, zero_division=0))

    # 9. Save metrics
    metrics = {
        "peft_method": args.method,
        "parameters": param_str,
        "trainable_params": trainable_params,
        "all_params": all_params,
        "efficiency_percentage": efficiency_pct,
        "best_val_macro_f1": float(best_val_f1),
        "test_macro_f1": float(test_f1),
        "per_class_f1": {label_names[i]: float(per_class_f1[i]) for i in range(num_labels)},
        "training_time_seconds": total_training_time,
        "checkpoint": ckpt_path,
    }

    config_dict = {
        "model": BASE_MODEL,
        "peft_method": args.method,
        "peft_param": param_str,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
    }
    utils.save_results(run_name, metrics, config_dict)
    print(f"PEFT run summary | test_f1={test_f1:.4f} | trainable_pct={efficiency_pct:.4f}%")


if __name__ == "__main__":
    main()


