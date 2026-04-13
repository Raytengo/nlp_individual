import os

# ======================== Paths ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
BEST_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "best")

# ======================== Model ========================
MODEL_NAME = "/home/wuyifan/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf"

# ======================== LoRA =========================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ===================== Training ========================
EPOCHS = 3
BATCH_SIZE_PER_GPU = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
MAX_SEQ_LENGTH = 128
NUM_WORKERS = 3
SEED = 42

# =================== Inference =========================
MAX_NEW_TOKENS = 32
INFERENCE_BATCH_SIZE = 24

# ====================== Data ==========================
VAL_RATIO = 0.2

# ===================== Prompt ==========================
RESPONSE_TEMPLATE = "\n### Answer:"
