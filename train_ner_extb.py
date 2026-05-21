"""
Extension B: fine-tune the vocabulary-extended model on MasakhaNER 2.0 (Yoruba).

Identical to train_ner.py except:
  - Loads from EXT_MODEL_DIR (extended tokeniser + model).
  - Checkpoint: checkpoints/ner_extb/
  - Results:    results/ner_extb.json
  - Hyperparameters are unchanged (do not re-tune).
"""

from train_ner import main
from config import EXT_MODEL_DIR

if __name__ == "__main__":
    main(
        model_dir   = EXT_MODEL_DIR,
        ckpt_name   = "ner_extb",
        result_name = "ner_extb",
    )
