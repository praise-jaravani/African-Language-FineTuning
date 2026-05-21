"""
Extension B: fine-tune the vocabulary-extended model on MasakhaNews (Yoruba).

Identical to train_news.py except:
  - Loads from EXT_MODEL_DIR (extended tokeniser + model).
  - Checkpoint: checkpoints/news_extb/
  - Results:    results/news_extb.json
  - Hyperparameters are unchanged (do not re-tune).
"""

from train_news import main
from config import EXT_MODEL_DIR

if __name__ == "__main__":
    main(
        model_dir   = EXT_MODEL_DIR,
        ckpt_name   = "news_extb",
        result_name = "news_extb",
    )
