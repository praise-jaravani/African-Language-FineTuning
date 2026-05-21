import os

BASE_MODEL = "jhu-clsp/mmBERT-small"
LANGUAGE   = "yor"

NEWS_DATASET = "masakhane/masakhanews"
NER_DATASET  = "masakhane/masakhaner2"   # NOTE: not "masakhaner-2.0"

RESULTS_DIR    = "results/"
CHECKPOINT_DIR = "checkpoints/"
LOGS_DIR       = "logs/"
EXT_MODEL_DIR  = "checkpoints/extended_model/"

# Create output dirs on import — safe to call repeatedly
for _d in [RESULTS_DIR, CHECKPOINT_DIR, LOGS_DIR, EXT_MODEL_DIR]:
    os.makedirs(_d, exist_ok=True)

MAX_SEQ_LEN = 128

# Final chosen LR after sweep over {1e-5, 2e-5, 3e-5, 5e-5}
LEARNING_RATE       = 5e-5
BATCH_SIZE          = 16
NUM_EPOCHS          = 5
WARMUP_STEPS        = 100
WEIGHT_DECAY        = 0.01
EARLY_STOP_PATIENCE = 2

SEED = 42

VOCAB_NEW_TOKENS = 1000   # target vocab size for Yoruba BPE training
VOCAB_MIN_FREQ   = 5      # minimum corpus frequency to keep a token
