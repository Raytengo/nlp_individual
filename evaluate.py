import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from config import (
    DATA_PATH, CHECKPOINT_DIR, BEST_MODEL_DIR, MODEL_NAME, SEED,
    MAX_SEQ_LENGTH, MAX_NEW_TOKENS, INFERENCE_BATCH_SIZE, VAL_RATIO,
)
from data_utils import format_prompt, get_model_path


def normalize(text):
    """Normalize text for comparison: lowercase, strip whitespace & punctuation."""
    text = text.lower().strip()
    text = text.strip(".,!?:;")
    return text


def parse_prediction(generated_text):
    """Take content before first newline as prediction."""
    return generated_text.strip().split("\n")[0].strip()


def run_inference(model, tokenizer, questions, batch_size=16):
    """Batch inference — decode only newly generated tokens."""
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Inference"):
        batch_q = questions[i : i + batch_size]
        prompts = [format_prompt(q) for q in batch_q]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        for j, output_ids in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            new_tokens = output_ids[input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            predictions.append(parse_prediction(decoded))

    return predictions


def compute_accuracy(predictions, ground_truths):
    """Compute substring-match accuracy (main metric) + exact match variants."""
    correct_sub = 0
    correct_exact = 0
    correct_norm_exact = 0

    for pred, gt in zip(predictions, ground_truths):
        norm_pred = normalize(pred)
        norm_gt = normalize(gt)

        if norm_gt in norm_pred:
            correct_sub += 1
        if pred.strip() == gt.strip():
            correct_exact += 1
        if norm_pred == norm_gt:
            correct_norm_exact += 1

    total = len(predictions)
    return {
        "accuracy": correct_sub / total,
        "exact_match": correct_exact / total,
        "normalized_exact_match": correct_norm_exact / total,
        "total": total,
        "correct": correct_sub,
    }


def load_model_for_eval(checkpoint_path):
    """Load base model + LoRA adapter for evaluation (left-padding)."""
    model_path = get_model_path(MODEL_NAME)

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
    tokenizer.padding_side = "left"  # left-padding for batch inference

    return model, tokenizer


def evaluate_checkpoint(checkpoint_path, train_data, val_data):
    """Evaluate a single checkpoint on both train and val sets."""
    print(f"\n--- Evaluating: {checkpoint_path} ---")
    model, tokenizer = load_model_for_eval(checkpoint_path)

    # Validation
    val_preds = run_inference(
        model, tokenizer,
        [d["question"] for d in val_data],
        INFERENCE_BATCH_SIZE,
    )
    val_metrics = compute_accuracy(val_preds, [d["correct_answer"] for d in val_data])
    print(f"  Val  Accuracy: {val_metrics['accuracy']:.4f} "
          f"({val_metrics['correct']}/{val_metrics['total']})")

    # Training
    train_preds = run_inference(
        model, tokenizer,
        [d["question"] for d in train_data],
        INFERENCE_BATCH_SIZE,
    )
    train_metrics = compute_accuracy(
        train_preds, [d["correct_answer"] for d in train_data]
    )
    print(f"  Train Accuracy: {train_metrics['accuracy']:.4f} "
          f"({train_metrics['correct']}/{train_metrics['total']})")

    del model, tokenizer
    torch.cuda.empty_cache()

    return {"train": train_metrics, "val": val_metrics}


def evaluate_all_checkpoints(train_data, val_data):
    """Evaluate every epoch checkpoint; copy the best to best/ directory."""
    checkpoints = sorted([
        os.path.join(CHECKPOINT_DIR, d)
        for d in os.listdir(CHECKPOINT_DIR)
        if d.startswith("checkpoint-")
    ])

    if not checkpoints:
        print("[WARN] No checkpoints found.")
        return

    best_acc, best_ckpt = -1, None
    all_results = {}

    for ckpt in checkpoints:
        metrics = evaluate_checkpoint(ckpt, train_data, val_data)
        all_results[ckpt] = metrics
        if metrics["val"]["accuracy"] > best_acc:
            best_acc = metrics["val"]["accuracy"]
            best_ckpt = ckpt

    # Copy best checkpoint
    if best_ckpt:
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        for fname in os.listdir(best_ckpt):
            src = os.path.join(best_ckpt, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(BEST_MODEL_DIR, fname))
        print(f"\nBest checkpoint: {best_ckpt}  (val accuracy: {best_acc:.4f})")
        print(f"Copied to: {BEST_MODEL_DIR}")

    return all_results


if __name__ == "__main__":
    from data_utils import load_and_clean_data, split_data

    data = load_and_clean_data(DATA_PATH)
    train_data, val_data = split_data(data, VAL_RATIO, SEED)
    evaluate_all_checkpoints(train_data, val_data)
