# CSC5035Z Assignment 2 — Fine-Tuning Language Models on African Language NLP Tasks

**Course:** CSC5035Z Natural Language Processing, UCT 2026  
**Student number:** JRVPRA001  
**Language:** Yoruba (`yor`)  
**Extension:** B — Vocabulary Adaptation

---

## Project Description

This project fine-tunes `jhu-clsp/mmBERT-small` (a compact multilingual BERT encoder) on two Yoruba NLP tasks from the AfroBench benchmark:

1. **MasakhaNews** — news topic classification (9 categories, metric: macro-F1)
2. **MasakhaNER 2.0** — named entity recognition (PER, ORG, LOC, DATE, metric: seqeval span-F1)

**Extension B (Vocabulary Adaptation):** The mmBERT-small tokeniser is extended with ~1,000 Yoruba-specific BPE tokens, each initialised as the mean of its constituent subword embeddings from the original model. Both tasks are then re-fine-tuned on this extended model to measure the impact on tokeniser fertility and downstream performance.

---

## Quickest Path — Inspect Pre-computed Results (No Training Required)

All result files are included in the submission. To view tables and figures without running any training:

**Option A — Jupyter (local):**

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install jupyter

# 3. Launch Jupyter and open the results notebook
jupyter notebook notebooks/results_analysis.ipynb
```

Then run all cells (Kernel → Restart & Run All).

**Option B — VS Code:**  
Open the project folder in VS Code, select the Python interpreter from your virtual environment, then open `notebooks/results_analysis.ipynb` and click **Run All**.

**Option C — Google Colab:**  
Upload `notebooks/results_analysis.ipynb` to Colab and upload the `results/` folder to your Drive at `MyDrive/csc5035z-a2/results/`. Run all cells.

---

## Full Pipeline — Reproduce Training from Scratch

Requires a GPU (tested on NVIDIA T4). Total runtime ~60–90 min.

### Option A — Jupyter (local)

```bash
# 1. Create and activate a virtual environment (Python ≥ 3.9 required)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install jupyter

# 3. Launch the notebook from the project root
jupyter notebook run.ipynb
```

Then run all cells top to bottom (Kernel → Restart & Run All). Cell 1 auto-detects that it is not running on Colab and sets the working directory to the project root automatically — no path editing needed.

### Option B — Google Colab (recommended for free GPU)

1. Upload the project folder to Google Drive at **exactly** `MyDrive/csc5035z-a2/`.
2. Open `run.ipynb` in Colab (File → Open notebook → Google Drive).
3. Run all cells top to bottom. Cell 1 mounts Drive and sets the working directory automatically.
4. If Colab disconnects, re-run Cell 1 and resume from the last completed step.

---

## Expected Outputs

| Directory / File | Contents |
|---|---|
| `checkpoints/news_baseline/` | Best MasakhaNews baseline model + tokeniser |
| `checkpoints/ner_baseline/`  | Best MasakhaNER baseline model + tokeniser |
| `checkpoints/extended_model/` | Extended vocab model + tokeniser (Extension B) |
| `checkpoints/news_extb/`     | Best Extension B MasakhaNews model |
| `checkpoints/ner_extb/`      | Best Extension B NER model |
| `results/news_baseline.json` | Baseline news metrics + predictions |
| `results/ner_baseline.json`  | Baseline NER metrics + predictions |
| `results/news_extb.json`     | Extension B news metrics |
| `results/ner_extb.json`      | Extension B NER metrics |
| `results/tokenizer_analysis_baseline.json` | Fertility stats before extension |
| `results/tokenizer_analysis_extb.json`     | Fertility stats after extension |
| `results/lr_sweep.json`      | LR sweep results across {1e-5, 2e-5, 3e-5, 5e-5} |
| `results/eval_news_news_baseline.json` | Authoritative test F1 for baseline news |
| `results/eval_ner_ner_baseline.json`   | Authoritative test F1 for baseline NER |
| `results/eval_news_news_extb.json`     | Authoritative test F1 for Extension B news |
| `results/eval_ner_ner_extb.json`       | Authoritative test F1 for Extension B NER |
| `results/fig_*.png`          | Report figures (from analysis notebook) |

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Base model | `jhu-clsp/mmBERT-small` | |
| Learning rate | `5e-5` | Optimal from sweep over {1e-5, 2e-5, 3e-5, 5e-5} on val split |
| Batch size | 16 | |
| Max sequence length | 128 | |
| Epochs | 5 | With early stopping |
| Early stopping patience | 2 | Epochs without val improvement |
| Warmup steps | 100 | |
| Weight decay | 0.01 | AdamW |
| Seed | 42 | |
| New vocab tokens | 1000 | Extension B BPE vocab size |
| Min token frequency | 5 | Extension B minimum corpus frequency |

---

## Hardware & Runtimes

| Script | Approx. runtime (T4 GPU) |
|---|---|
| `analyse_tokenizer.py` | ~2 min |
| `train_news.py` | ~5–10 min |
| `train_ner.py` | ~5–10 min |
| `extend_vocab.py` | ~3 min |
| `sweep_lr.py` | ~30–45 min (8 runs × 5 epochs) |
| `train_news_extb.py` | ~5–10 min |
| `train_ner_extb.py` | ~5–10 min |
| `evaluate.py` (×4) | ~5 min total |
| **Total (full pipeline)** | **~60–90 min** |

---

## File Structure

```
csc5035z-a2/
├── run.ipynb                  # Colab entry point
├── config.py                  # All hyperparameters
├── utils.py                   # Shared helpers
├── evaluate.py                # Standalone evaluation
├── analyse_tokenizer.py       # Tokenizer fertility analysis
├── train_news.py              # Baseline MasakhaNews
├── train_ner.py               # Baseline MasakhaNER 2.0
├── extend_vocab.py            # Extension B: vocab adaptation
├── sweep_lr.py                # LR sweep over {1e-5, 2e-5, 3e-5, 5e-5}
├── train_news_extb.py         # Extension B MasakhaNews
├── train_ner_extb.py          # Extension B MasakhaNER
├── notebooks/
│   └── results_analysis.ipynb
├── results/
├── checkpoints/
├── logs/
└── requirements.txt
```
