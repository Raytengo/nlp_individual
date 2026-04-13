import json
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# 6 groups of conflicting annotations — unified answers
CONFLICT_RESOLUTIONS = {
    "Where are protons and neutrons located?": "nucleus",
    "What is the main function of the cardiovascular system?": "transporting substances around the body",
    "What is the simplest life cycle?": "haploid life cycle",
    "What is the basic unit of matter?": "atom",
    "What occurs when a parent cell splits into two identical daughter cells of the same size?": "binary fission",
    "What is the first part of the large intestine, where wastes enter from the small intestine?": "cecum",
}


def load_and_clean_data(data_path):
    """Load dataset, apply cleaning rules and conflict resolution."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data)

    # Remove null / empty fields
    data = [
        d for d in data
        if d.get("question") and d.get("correct_answer")
        and d["question"].strip() and d["correct_answer"].strip()
    ]

    # Remove question shorter than 3 characters
    data = [d for d in data if len(d["question"].strip()) >= 3]

    # --- Report stats (before conflict resolution) ---
    q_counts = Counter(d["question"] for d in data)
    dup_q = sum(1 for c in q_counts.values() if c > 1)

    qa_map = defaultdict(set)
    for d in data:
        qa_map[d["question"]].add(d["correct_answer"])
    conflict_count = sum(1 for ans in qa_map.values() if len(ans) > 1)

    # --- Resolve conflicts ---
    for d in data:
        if d["question"] in CONFLICT_RESOLUTIONS:
            d["correct_answer"] = CONFLICT_RESOLUTIONS[d["question"]]

    print("=== Data Cleaning Report ===")
    print(f"Original samples:        {original_count}")
    print(f"After cleaning:          {len(data)}")
    print(f"Duplicate questions:     {dup_q}")
    print(f"Conflicting annotations: {conflict_count}")
    print("============================")

    return data


def split_data(data, val_ratio=0.2, seed=42):
    """Random split into train / val sets."""
    train_data, val_data = train_test_split(
        data, test_size=val_ratio, random_state=seed
    )
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    return train_data, val_data


def format_prompt(question, answer=None):
    """Format a single example into prompt string."""
    if answer is not None:
        return f"### Question: {question}\n### Answer: {answer}"
    return f"### Question: {question}\n### Answer:"


def get_tokenizer(model_path):
    """Load and configure tokenizer for training (right-padding)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def get_model_path(model_name):
    """Download model from ModelScope if needed, return local path."""
    import os
    if os.path.exists(model_name):
        return model_name
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_name)
        print(f"Model path: {model_dir}")
        return model_dir
    except ImportError:
        print("modelscope not installed, using model name directly")
        return model_name
