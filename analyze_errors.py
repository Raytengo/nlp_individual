import argparse
import csv
import json
import os
import re
from collections import Counter

from config import build_run_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze incorrect predictions for one experiment or one CSV file."
    )
    parser.add_argument("--experiment-name")
    parser.add_argument("--errors-csv")
    parser.add_argument("--output-root")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def normalize(text):
    text = text.lower().strip()
    text = text.strip(".,!?:;\"'")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    return [tok for tok in re.findall(r"[a-z0-9]+", normalize(text)) if tok]


def detect_error_type(pred_answer, true_answer):
    pred_norm = normalize(pred_answer)
    true_norm = normalize(true_answer)

    if not pred_norm:
        return "empty_prediction"
    if pred_norm == true_norm:
        return "normalization_only_mismatch"
    if true_norm and true_norm in pred_norm:
        return "normalization_only_mismatch"

    pred_tokens = set(tokenize(pred_answer))
    true_tokens = set(tokenize(true_answer))
    if true_tokens and true_tokens.issubset(pred_tokens):
        return "contains_answer_words_but_wrong_format"
    if pred_tokens & true_tokens:
        return "partial_word_overlap"
    if len(pred_tokens) >= 8:
        return "verbose_wrong_answer"
    return "semantic_or_unknown_mismatch"


def resolve_input_path(args):
    if args.errors_csv:
        return args.errors_csv
    if not args.experiment_name:
        raise SystemExit("Provide either --errors-csv or --experiment-name")
    run_config = build_run_config(
        experiment_name=args.experiment_name,
        output_root=args.output_root,
    )
    return run_config.best_val_errors_path


def load_rows(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def top_counter_rows(counter, top_k, label):
    rows = []
    for value, count in counter.most_common(top_k):
        rows.append({label: value, "count": count})
    return rows


def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    errors_csv = resolve_input_path(args)
    rows = load_rows(errors_csv)
    if not rows:
        raise SystemExit(f"No rows found in {errors_csv}")

    error_type_counter = Counter()
    true_answer_counter = Counter()
    pred_answer_counter = Counter()
    question_length_counter = Counter()

    enriched_rows = []
    for row in rows:
        pred_answer = row.get("pred_answer", "")
        true_answer = row.get("true_answer", "")
        question = row.get("question", "")
        error_type = detect_error_type(pred_answer, true_answer)
        error_type_counter[error_type] += 1
        true_answer_counter[normalize(true_answer)] += 1
        pred_answer_counter[normalize(pred_answer)] += 1

        q_len = len(tokenize(question))
        if q_len <= 8:
            question_length_counter["short_question_0_8"] += 1
        elif q_len <= 15:
            question_length_counter["medium_question_9_15"] += 1
        else:
            question_length_counter["long_question_16_plus"] += 1

        enriched = dict(row)
        enriched["error_type"] = error_type
        enriched_rows.append(enriched)

    output_dir = os.path.dirname(errors_csv)
    enriched_csv = os.path.join(output_dir, "error_analysis_detailed.csv")
    save_csv(
        enriched_csv,
        enriched_rows,
        ["index", "question", "pred_answer", "true_answer", "error_type"],
    )

    summary = {
        "errors_csv": errors_csv,
        "total_errors": len(rows),
        "error_type_counts": dict(error_type_counter.most_common()),
        "most_common_missed_answers": top_counter_rows(
            true_answer_counter, args.top_k, "true_answer"
        ),
        "most_common_wrong_predictions": top_counter_rows(
            pred_answer_counter, args.top_k, "pred_answer"
        ),
        "question_length_buckets": dict(question_length_counter),
    }
    summary_json = os.path.join(output_dir, "error_analysis_summary.json")
    save_json(summary_json, summary)

    common_answers_csv = os.path.join(output_dir, "most_common_missed_answers.csv")
    save_csv(
        common_answers_csv,
        summary["most_common_missed_answers"],
        ["true_answer", "count"],
    )

    common_preds_csv = os.path.join(output_dir, "most_common_wrong_predictions.csv")
    save_csv(
        common_preds_csv,
        summary["most_common_wrong_predictions"],
        ["pred_answer", "count"],
    )

    print(f"[INFO] Detailed error analysis saved to {enriched_csv}")
    print(f"[INFO] Error summary saved to {summary_json}")
    print(f"[INFO] Missed-answer counts saved to {common_answers_csv}")
    print(f"[INFO] Wrong-prediction counts saved to {common_preds_csv}")
    print("[INFO] Top error types:")
    for error_type, count in error_type_counter.most_common():
        print(f"  {error_type}: {count}")


if __name__ == "__main__":
    main()
