import argparse
import csv
import json
import os
import shutil

from tqdm import tqdm

from config import build_run_config
from data_utils import format_prompt, get_model_path, load_and_clean_data, split_data


def normalize(text):
    """Normalize text for supplemental exact-match comparison."""
    text = text.lower().strip()
    text = text.strip(".,!?:;")
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA checkpoints.")
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--data-path")
    parser.add_argument("--output-root")
    parser.add_argument("--model-name")
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def run_inference(model, tokenizer, questions, run_config):
    """Batch inference using the teacher prompt format."""
    import torch

    model.eval()
    predictions = []

    for i in tqdm(
        range(0, len(questions), run_config.inference_batch_size),
        desc="Inference",
    ):
        batch_q = questions[i: i + run_config.inference_batch_size]
        prompts = [
            format_prompt(q, prompt_template=run_config.prompt_template)
            for q in batch_q
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=run_config.max_seq_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=run_config.max_new_tokens,
                do_sample=False,
            )

        for j, output_ids in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            new_tokens = output_ids[input_len:]
            predictions.append(
                tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )

    return predictions


def compute_accuracy(predictions, examples):
    """Compute teacher accuracy and return detailed incorrect cases."""
    correct_teacher = 0
    correct_exact = 0
    correct_norm_exact = 0
    incorrect_cases = []

    for idx, (pred, example) in enumerate(zip(predictions, examples)):
        gt = example["correct_answer"]
        teacher_match = gt in pred
        norm_pred = normalize(pred)
        norm_gt = normalize(gt)

        if teacher_match:
            correct_teacher += 1
        if pred.strip() == gt.strip():
            correct_exact += 1
        if norm_pred == norm_gt:
            correct_norm_exact += 1
        if not teacher_match:
            incorrect_cases.append({
                "index": idx,
                "question": example["question"],
                "true_answer": gt,
                "pred_answer": pred,
            })

    total = len(predictions)
    return {
        "accuracy": correct_teacher / total,
        "exact_match": correct_exact / total,
        "normalized_exact_match": correct_norm_exact / total,
        "total": total,
        "correct": correct_teacher,
        "incorrect_cases": incorrect_cases,
    }


def load_model_for_eval(checkpoint_path, run_config):
    """Load base model + LoRA adapter for evaluation."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = get_model_path(run_config.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def evaluate_split(model, tokenizer, examples, run_config):
    predictions = run_inference(
        model,
        tokenizer,
        [d["question"] for d in examples],
        run_config,
    )
    return compute_accuracy(predictions, examples)


def evaluate_checkpoint(checkpoint_path, train_data, val_data, run_config):
    """Evaluate a single checkpoint on both train and val sets."""
    import torch

    print(f"\n--- Evaluating: {checkpoint_path} ---")
    model, tokenizer = load_model_for_eval(checkpoint_path, run_config)

    val_metrics = evaluate_split(model, tokenizer, val_data, run_config)
    print(
        f"  Val  Accuracy: {val_metrics['accuracy']:.4f} "
        f"({val_metrics['correct']}/{val_metrics['total']})"
    )

    train_metrics = evaluate_split(model, tokenizer, train_data, run_config)
    print(
        f"  Train Accuracy: {train_metrics['accuracy']:.4f} "
        f"({train_metrics['correct']}/{train_metrics['total']})"
    )

    del model, tokenizer
    torch.cuda.empty_cache()

    return {"train": train_metrics, "val": val_metrics}


def save_incorrect_cases(path, incorrect_cases):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "question", "pred_answer", "true_answer"],
        )
        writer.writeheader()
        writer.writerows(incorrect_cases)
    print(f"[INFO] Incorrect predictions saved to {path}")


def save_checkpoint_metrics(run_config, all_results):
    os.makedirs(run_config.report_dir, exist_ok=True)
    with open(run_config.checkpoint_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "train_accuracy",
                "val_accuracy",
                "train_exact_match",
                "val_exact_match",
                "train_normalized_exact_match",
                "val_normalized_exact_match",
            ],
        )
        writer.writeheader()
        for checkpoint_path, metrics in all_results.items():
            writer.writerow({
                "checkpoint": os.path.basename(checkpoint_path),
                "train_accuracy": metrics["train"]["accuracy"],
                "val_accuracy": metrics["val"]["accuracy"],
                "train_exact_match": metrics["train"]["exact_match"],
                "val_exact_match": metrics["val"]["exact_match"],
                "train_normalized_exact_match": metrics["train"]["normalized_exact_match"],
                "val_normalized_exact_match": metrics["val"]["normalized_exact_match"],
            })


def save_evaluation_summary(run_config, best_ckpt, best_metrics, all_results):
    os.makedirs(run_config.report_dir, exist_ok=True)
    summary = {
        "experiment_name": run_config.experiment_name,
        "best_checkpoint": os.path.basename(best_ckpt) if best_ckpt else None,
        "best_checkpoint_path": best_ckpt,
        "best_train_accuracy": best_metrics["train"]["accuracy"] if best_metrics else None,
        "best_val_accuracy": best_metrics["val"]["accuracy"] if best_metrics else None,
        "best_train_exact_match": best_metrics["train"]["exact_match"] if best_metrics else None,
        "best_val_exact_match": best_metrics["val"]["exact_match"] if best_metrics else None,
        "best_train_normalized_exact_match": (
            best_metrics["train"]["normalized_exact_match"] if best_metrics else None
        ),
        "best_val_normalized_exact_match": (
            best_metrics["val"]["normalized_exact_match"] if best_metrics else None
        ),
        "best_val_error_file": run_config.best_val_errors_path,
        "best_train_error_file": run_config.best_train_errors_path,
        "all_checkpoints": {
            os.path.basename(path): {
                "train_accuracy": metrics["train"]["accuracy"],
                "val_accuracy": metrics["val"]["accuracy"],
            }
            for path, metrics in all_results.items()
        },
    }
    with open(run_config.evaluation_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def evaluate_all_checkpoints(run_config, train_data=None, val_data=None):
    """Evaluate every checkpoint in one experiment and save reports."""
    if train_data is None or val_data is None:
        data = load_and_clean_data(run_config.data_path)
        train_data, val_data = split_data(data, run_config.val_ratio, run_config.seed)

    if not os.path.isdir(run_config.checkpoint_dir):
        print(f"[WARN] Checkpoint directory not found: {run_config.checkpoint_dir}")
        return None

    checkpoints = sorted([
        os.path.join(run_config.checkpoint_dir, d)
        for d in os.listdir(run_config.checkpoint_dir)
        if d.startswith("checkpoint-")
    ])

    if not checkpoints:
        print(f"[WARN] No checkpoints found in {run_config.checkpoint_dir}")
        return None

    best_acc, best_ckpt, best_metrics = -1, None, None
    all_results = {}

    for ckpt in checkpoints:
        metrics = evaluate_checkpoint(ckpt, train_data, val_data, run_config)
        all_results[ckpt] = metrics
        if metrics["val"]["accuracy"] > best_acc:
            best_acc = metrics["val"]["accuracy"]
            best_ckpt = ckpt
            best_metrics = metrics

    if best_ckpt:
        os.makedirs(run_config.best_model_dir, exist_ok=True)
        for fname in os.listdir(best_ckpt):
            src = os.path.join(best_ckpt, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(run_config.best_model_dir, fname))
        save_incorrect_cases(
            run_config.best_val_errors_path,
            best_metrics["val"]["incorrect_cases"],
        )
        save_incorrect_cases(
            run_config.best_train_errors_path,
            best_metrics["train"]["incorrect_cases"],
        )
        print(
            f"\nBest checkpoint: {best_ckpt}  "
            f"(val accuracy: {best_acc:.4f})"
        )
        print(f"Copied to: {run_config.best_model_dir}")

    save_checkpoint_metrics(run_config, all_results)
    save_evaluation_summary(run_config, best_ckpt, best_metrics, all_results)
    return all_results


if __name__ == "__main__":
    args = parse_args()
    run_config = build_run_config(
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        output_root=args.output_root,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        inference_batch_size=args.inference_batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    evaluate_all_checkpoints(run_config)
