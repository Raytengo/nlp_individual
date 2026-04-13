import os
import re
from dataclasses import asdict, dataclass

# ======================== Paths ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EXPERIMENTS_DIR = os.path.join(OUTPUT_DIR, "experiments")
GRID_SEARCH_DIR = os.path.join(OUTPUT_DIR, "grid_search")

# ======================== Model ========================
MODEL_NAME = "/home/wuyifan/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf"

# ======================== LoRA =========================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

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
PROMPT_TEMPLATE = "Question: {question} Answer:"
RESPONSE_TEMPLATE = " Answer:"

DEFAULT_EXPERIMENT_NAME = "default"


def sanitize_experiment_name(name):
    """Return a filesystem-safe experiment name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or DEFAULT_EXPERIMENT_NAME


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    data_path: str = DATA_PATH
    output_root: str = OUTPUT_DIR
    model_name: str = MODEL_NAME
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    lora_target_modules: tuple = LORA_TARGET_MODULES
    epochs: int = EPOCHS
    batch_size_per_gpu: int = BATCH_SIZE_PER_GPU
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = LEARNING_RATE
    lr_scheduler: str = LR_SCHEDULER
    warmup_ratio: float = WARMUP_RATIO
    max_grad_norm: float = MAX_GRAD_NORM
    max_seq_length: int = MAX_SEQ_LENGTH
    num_workers: int = NUM_WORKERS
    seed: int = SEED
    max_new_tokens: int = MAX_NEW_TOKENS
    inference_batch_size: int = INFERENCE_BATCH_SIZE
    val_ratio: float = VAL_RATIO
    prompt_template: str = PROMPT_TEMPLATE
    response_template: str = RESPONSE_TEMPLATE

    @property
    def experiment_dir(self):
        return os.path.join(self.output_root, "experiments", self.experiment_name)

    @property
    def checkpoint_dir(self):
        return os.path.join(self.experiment_dir, "checkpoints")

    @property
    def log_dir(self):
        return os.path.join(self.experiment_dir, "logs")

    @property
    def plot_dir(self):
        return os.path.join(self.experiment_dir, "plots")

    @property
    def report_dir(self):
        return os.path.join(self.experiment_dir, "reports")

    @property
    def best_model_dir(self):
        return os.path.join(self.checkpoint_dir, "best")

    @property
    def config_path(self):
        return os.path.join(self.experiment_dir, "run_config.json")

    @property
    def checkpoint_metrics_path(self):
        return os.path.join(self.report_dir, "checkpoint_metrics.csv")

    @property
    def evaluation_summary_path(self):
        return os.path.join(self.report_dir, "evaluation_summary.json")

    @property
    def best_val_errors_path(self):
        return os.path.join(self.report_dir, "best_val_incorrect_predictions.csv")

    @property
    def best_train_errors_path(self):
        return os.path.join(self.report_dir, "best_train_incorrect_predictions.csv")

    @property
    def loss_plot_path(self):
        return os.path.join(self.plot_dir, "loss_curve.png")

    def to_dict(self):
        data = asdict(self)
        data["lora_target_modules"] = list(self.lora_target_modules)
        data["experiment_dir"] = self.experiment_dir
        data["checkpoint_dir"] = self.checkpoint_dir
        data["log_dir"] = self.log_dir
        data["plot_dir"] = self.plot_dir
        data["report_dir"] = self.report_dir
        data["best_model_dir"] = self.best_model_dir
        return data


def build_run_config(**overrides):
    values = RunConfig().to_dict()
    values.update({k: v for k, v in overrides.items() if v is not None})
    values["experiment_name"] = sanitize_experiment_name(values["experiment_name"])
    if isinstance(values.get("lora_target_modules"), list):
        values["lora_target_modules"] = tuple(values["lora_target_modules"])
    return RunConfig(**{
        key: values[key]
        for key in RunConfig.__dataclass_fields__
    })


DEFAULT_RUN_CONFIG = build_run_config()
CHECKPOINT_DIR = DEFAULT_RUN_CONFIG.checkpoint_dir
LOG_DIR = DEFAULT_RUN_CONFIG.log_dir
PLOT_DIR = DEFAULT_RUN_CONFIG.plot_dir
BEST_MODEL_DIR = DEFAULT_RUN_CONFIG.best_model_dir
