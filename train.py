import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from config import build_run_config
from data_utils import (
    format_prompt,
    get_model_path,
    get_tokenizer,
    load_and_clean_data,
    split_data,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA QA model.")
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--data-path")
    parser.add_argument("--output-root")
    parser.add_argument("--model-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-scheduler")
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def load_model(model_path):
    """Load base model with Flash Attention 2 fallback."""
    from transformers import AutoModelForCausalLM

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
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


def formatting_func(examples, prompt_template):
    """Format a batch of examples for SFTTrainer."""
    return [
        format_prompt(q, a, prompt_template=prompt_template)
        for q, a in zip(examples["question"], examples["correct_answer"])
    ]


def save_loss_logs(trainer, log_dir):
    """Extract train / val loss from trainer log history and save as CSV."""
    os.makedirs(log_dir, exist_ok=True)

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

    train_csv = os.path.join(log_dir, "train_loss.csv")
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss"])
        writer.writeheader()
        writer.writerows(train_losses)

    val_csv = os.path.join(log_dir, "val_loss.csv")
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "eval_loss"])
        writer.writeheader()
        writer.writerows(val_losses)

    print(f"[INFO] Loss logs saved to {log_dir}")


def save_run_config(run_config):
    os.makedirs(run_config.experiment_dir, exist_ok=True)
    with open(run_config.config_path, "w", encoding="utf-8") as f:
        json.dump(run_config.to_dict(), f, indent=2)
    print(f"[INFO] Run config saved to {run_config.config_path}")


def main(run_config, skip_eval=False, skip_plot=False):
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    set_seed(run_config.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ---- Data ----
    data = load_and_clean_data(run_config.data_path)
    train_data, val_data = split_data(data, run_config.val_ratio, run_config.seed)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # ---- Model & Tokenizer ----
    model_path = get_model_path(run_config.model_name)
    tokenizer = get_tokenizer(model_path)
    model = load_model(model_path)

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=run_config.lora_r,
        lora_alpha=run_config.lora_alpha,
        lora_dropout=run_config.lora_dropout,
        target_modules=list(run_config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()

    # ---- Data Collator ----
    response_template_ids = tokenizer.encode(
        run_config.response_template,
        add_special_tokens=False,
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # ---- Training Arguments ----
    os.makedirs(run_config.checkpoint_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=run_config.checkpoint_dir,
        num_train_epochs=run_config.epochs,
        per_device_train_batch_size=run_config.batch_size_per_gpu,
        per_device_eval_batch_size=run_config.batch_size_per_gpu,
        gradient_accumulation_steps=run_config.gradient_accumulation_steps,
        learning_rate=run_config.learning_rate,
        lr_scheduler_type=run_config.lr_scheduler,
        warmup_ratio=run_config.warmup_ratio,
        max_grad_norm=run_config.max_grad_norm,
        optim="adamw_torch",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        seed=run_config.seed,
        dataloader_num_workers=run_config.num_workers,
        dataloader_pin_memory=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=lambda examples: formatting_func(
            examples, run_config.prompt_template
        ),
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=run_config.max_seq_length,
    )

    save_run_config(run_config)
    trainer.train()

    if local_rank == 0:
        save_loss_logs(trainer, run_config.log_dir)

        if not skip_eval:
            from evaluate import evaluate_all_checkpoints

            evaluate_all_checkpoints(
                run_config,
                train_data=train_data,
                val_data=val_data,
            )

        if not skip_plot:
            from plot import plot_loss_curves

            plot_loss_curves(run_config)


if __name__ == "__main__":
    args = parse_args()
    run_config = build_run_config(
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        output_root=args.output_root,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_seq_length=args.max_seq_length,
        num_workers=args.num_workers,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_new_tokens=args.max_new_tokens,
        inference_batch_size=args.inference_batch_size,
        val_ratio=args.val_ratio,
    )
    main(run_config, skip_eval=args.skip_eval, skip_plot=args.skip_plot)
