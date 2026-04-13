import os
import csv
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

from config import (
    DATA_PATH, CHECKPOINT_DIR, LOG_DIR, MODEL_NAME, SEED,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    EPOCHS, BATCH_SIZE_PER_GPU, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, LR_SCHEDULER, WARMUP_RATIO, MAX_GRAD_NORM,
    MAX_SEQ_LENGTH, NUM_WORKERS, VAL_RATIO, RESPONSE_TEMPLATE,
)
from data_utils import (
    load_and_clean_data, split_data, format_prompt,
    get_tokenizer, get_model_path,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path):
    """Load Llama-2-7B with bf16; attempt Flash Attention 2 with fallback."""
    kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="flash_attention_2", **kwargs
        )
        print("[INFO] Flash Attention 2 enabled")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        print("[INFO] Flash Attention 2 unavailable, using default attention")

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def formatting_func(examples):
    """Format a batch of examples for SFTTrainer."""
    return [
        format_prompt(q, a)
        for q, a in zip(examples["question"], examples["correct_answer"])
    ]


def save_loss_logs(trainer):
    """Extract train / val loss from trainer log history and save as CSV."""
    os.makedirs(LOG_DIR, exist_ok=True)

    train_losses, val_losses = [], []
    for entry in trainer.state.log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_losses.append({
                "step": entry["step"],
                "epoch": entry.get("epoch", ""),
                "loss": entry["loss"],
            })
        if "eval_loss" in entry:
            val_losses.append({
                "step": entry["step"],
                "epoch": entry.get("epoch", ""),
                "eval_loss": entry["eval_loss"],
            })

    train_csv = os.path.join(LOG_DIR, "train_loss.csv")
    with open(train_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "loss"])
        w.writeheader()
        w.writerows(train_losses)

    val_csv = os.path.join(LOG_DIR, "val_loss.csv")
    with open(val_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "eval_loss"])
        w.writeheader()
        w.writerows(val_losses)

    print(f"[INFO] Loss logs saved to {LOG_DIR}")


def main():
    set_seed(SEED)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ---- Data ----
    data = load_and_clean_data(DATA_PATH)
    train_data, val_data = split_data(data, VAL_RATIO, SEED)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # ---- Model & Tokenizer ----
    model_path = get_model_path(MODEL_NAME)
    tokenizer = get_tokenizer(model_path)
    model = load_model(model_path)

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    # ---- Data Collator (loss masking: only compute loss on answer tokens) ----
    # LLaMA tokenizer splits "### Answer:" differently in isolation vs in context.
    # In context (after \n): tokens are [2277, 29937, 673, 29901] = ['##','#','Answer',':']
    # We pass the token IDs directly so the collator can always find the boundary.
    response_template_ids = [2277, 29937, 673, 29901]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # ---- Training Arguments ----
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,
        per_device_eval_batch_size=BATCH_SIZE_PER_GPU,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        seed=SEED,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # ---- Train ----
    trainer.train()

    # ---- Post-training (rank 0 only) ----
    if local_rank == 0:
        save_loss_logs(trainer)

        # Evaluate all checkpoints and select best
        from evaluate import evaluate_all_checkpoints
        evaluate_all_checkpoints(train_data, val_data)

        # Plot loss curves
        from plot import plot_loss_curves
        plot_loss_curves()


if __name__ == "__main__":
    main()
